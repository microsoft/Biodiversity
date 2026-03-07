# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
GPU-accelerated mel spectrogram computation for bioacoustics.

This module provides a single, configurable function for computing mel
spectrograms from audio windows.  It supports high-frequency fill strategies,
configurable mono-channel selection, and custom spectrogram naming.
"""

import os
import math
import struct
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

import torch
import torchaudio
import librosa
import soundfile as sf


__all__ = [
    "compute_mel_spectrograms_gpu",
    "default_spectrogram_path",
]


def default_spectrogram_path(win: Dict, spectrograms_path: str) -> str:
    """Return the default .npy path for a spectrogram window.

    Uses ``{basename(sound_path)}_{start}_{end}.npy``.
    """
    sound_filename = os.path.splitext(os.path.basename(win["sound_path"]))[0]
    return os.path.join(
        spectrograms_path,
        f"{sound_filename}_{int(win['start'])}_{int(win['end'])}.npy",
    )


def _read_wav_fallback(filepath: str):
    """Read audio from a WAV file whose RIFF size field overflowed (>4 GB).

    Falls back to manual header parsing + raw PCM decoding so that
    ``soundfile`` / ``libsndfile`` errors are bypassed.

    Returns
    -------
    (data, sample_rate) matching ``sf.read(..., always_2d=True)`` contract:
    *data* is float32 with shape ``(n_frames, n_channels)``.
    """
    with open(filepath, "rb") as f:
        f.read(4)  # b'RIFF'
        f.read(4)  # file-size field (unreliable for >4 GB)
        wave_tag = f.read(4)
        if wave_tag != b"WAVE":
            raise ValueError(f"Not a WAV file (no WAVE tag): {filepath}")

        channels = sample_rate = bits_per_sample = None
        data_offset = data_size = None

        while True:
            chunk_hdr = f.read(8)
            if len(chunk_hdr) < 8:
                break
            chunk_id = chunk_hdr[:4]
            chunk_sz = struct.unpack("<I", chunk_hdr[4:8])[0]

            if chunk_id == b"fmt ":
                fmt = f.read(chunk_sz)
                channels = struct.unpack("<H", fmt[2:4])[0]
                sample_rate = struct.unpack("<I", fmt[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt[14:16])[0]
            elif chunk_id == b"data":
                data_offset = f.tell()
                file_sz = os.path.getsize(filepath)
                if chunk_sz == 0 or data_offset + chunk_sz > file_sz:
                    data_size = file_sz - data_offset
                else:
                    data_size = chunk_sz
                break
            else:
                f.seek(chunk_sz + (chunk_sz & 1), 1)

        if channels is None or data_offset is None:
            raise ValueError(f"Could not parse WAV header: {filepath}")

        bytes_per_sample = bits_per_sample // 8
        frame_bytes = bytes_per_sample * channels
        data_size = (data_size // frame_bytes) * frame_bytes

        f.seek(data_offset)
        raw = f.read(data_size)

    if bits_per_sample == 16:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 24:
        n = len(raw) // 3
        b = np.frombuffer(raw[: n * 3], dtype=np.uint8).reshape(-1, 3)
        i32 = (
            b[:, 2].astype(np.int32) << 24
            | b[:, 1].astype(np.int32) << 16
            | b[:, 0].astype(np.int32) << 8
        ) >> 8
        samples = i32.astype(np.float32) / 8388608.0
    elif bits_per_sample == 32:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported bits_per_sample={bits_per_sample}: {filepath}")

    samples = samples.reshape(-1, channels)
    return samples, sample_rate


def compute_mel_spectrograms_gpu(
    windows: List[Dict],
    sample_rate: int,
    n_fft: int,
    hop_length: Optional[int],
    n_mels: int,
    top_db: float,
    spectrograms_path: str,
    save_npy: bool = True,
    f_min: float = 0.0,
    mono_channel: str = "left",
    fill_highfreq: bool = True,
    fill_mean_below_sr: bool = False,
    noise_db_mean: Optional[float] = None,
    noise_db_std: float = 3.0,
    storage_dtype: str = "float32",
    spectrogram_path_fn: Optional[Callable[[Dict, str], str]] = None,
) -> None:
    """GPU-accelerated mel spectrogram computation.

    Parameters
    ----------
    windows : list of dict
        Each dict must have ``sound_path``, ``start``, ``end`` keys.
    sample_rate : int
        Target sample rate. Audio is resampled if it differs.
    n_fft, hop_length, n_mels, top_db
        Mel spectrogram parameters.
    spectrograms_path : str
        Directory where ``.npy`` files are saved.
    save_npy : bool
        Whether to persist spectrograms to disk.
    f_min : float
        Minimum frequency (Hz) for the mel filterbank.
    mono_channel : str
        How to reduce stereo to mono: ``"left"``, ``"right"``, or ``"mean"``.
    fill_highfreq : bool
        Fill high-frequency bins above the original Nyquist when resampling.
    fill_mean_below_sr : bool
        If True use the mean of valid bands for fill; otherwise use 10th-
        percentile noise floor.
    noise_db_mean, noise_db_std : float
        Parameters for noise-floor estimation when *fill_mean_below_sr* is
        False.
    storage_dtype : str
        NumPy dtype for saved arrays (``"float16"`` or ``"float32"``).
    spectrogram_path_fn : callable, optional
        ``(win, spectrograms_path) -> str`` returning the full ``.npy`` path
        for a window.  Defaults to :func:`default_spectrogram_path`.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    if spectrogram_path_fn is None:
        spectrogram_path_fn = default_spectrogram_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    Path(spectrograms_path).mkdir(parents=True, exist_ok=True)

    # Group windows by sound_path
    by_sid = defaultdict(list)
    for idx, win in enumerate(windows):
        by_sid[win["sound_path"]].append((idx, win))

    # Check for existing spectrograms
    files_to_process = {}
    total_windows = 0
    existing_windows = 0

    print("Checking for existing spectrograms...")
    for audio_file_path, items in tqdm(by_sid.items(), desc="Checking files"):
        missing_items = []
        for idx, win in items:
            npy_path = spectrogram_path_fn(win, spectrograms_path)
            total_windows += 1

            if not os.path.exists(npy_path):
                missing_items.append((idx, win))
            else:
                existing_windows += 1

        if missing_items:
            files_to_process[audio_file_path] = missing_items

    print(f"Found {existing_windows}/{total_windows} existing spectrograms")
    print(f"Need to create {total_windows - existing_windows} spectrograms from {len(files_to_process)} audio files")

    if len(files_to_process) == 0:
        print("All spectrograms already exist! Skipping computation.")
        return

    for audio_file_path, items in tqdm(files_to_process.items(), desc="Processing files"):
        # Decode on CPU
        try:
            y, orig_sr = sf.read(audio_file_path, dtype="float32", always_2d=True)
        except sf.LibsndfileError:
            y, orig_sr = _read_wav_fallback(audio_file_path)
            print(f"  [WAV fallback] {os.path.basename(audio_file_path)} "
                  f"({os.path.getsize(audio_file_path) / (1024**3):.1f} GB)")
        if y.ndim == 2:
            if y.shape[1] == 1:
                y = y[:, 0]
            elif mono_channel == "left":
                y = y[:, 0]
            elif mono_channel == "right":
                y = y[:, 1]
            else:  # "mean"
                y = y.mean(axis=1)
        wav_cpu = torch.from_numpy(y).unsqueeze(0)

        # Resample if needed
        if orig_sr != sample_rate:
            wav_cpu = torchaudio.functional.resample(wav_cpu, orig_freq=orig_sr, new_freq=sample_rate)

        # Mel transform on GPU
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=sample_rate / 2.0,
            power=2.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

        to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=top_db
        ).to(device)

        for global_idx, win in tqdm(items):
            start = int(win["start"])
            end = int(win["end"])
            npy_path = spectrogram_path_fn(win, spectrograms_path)

            if not os.path.exists(npy_path):
                wav_win = wav_cpu[:, start:end].to(device)
                S = mel_tf(wav_win).squeeze(0)
                S_db = to_db(S)

                # Optional high-frequency fill
                if fill_highfreq and orig_sr < sample_rate:
                    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=f_min, fmax=sample_rate / 2.0)
                    nyq_orig = (float(orig_sr) / 2.0) - 2500
                    noise_mask = torch.from_numpy((mel_freqs > nyq_orig).astype(np.bool_)).to(device)
                    if noise_mask.any():
                        valid_mask = ~noise_mask
                        if fill_mean_below_sr:
                            vals = S_db[valid_mask, :]
                            if vals.numel() > 0:
                                mu = vals.mean().item()
                            else:
                                mu = -60.0
                            S_db[noise_mask, :] = mu
                        else:
                            if noise_db_mean is None:
                                vals = S_db[valid_mask, :].reshape(-1)
                                if vals.numel() == 0:
                                    mu = -60.0
                                else:
                                    v = vals.float().cpu()
                                    k = max(1, int(math.ceil(0.10 * v.numel())))
                                    mu = torch.kthvalue(v, k).values.item()
                            else:
                                mu = float(noise_db_mean)

                            S_db[noise_mask, :] = mu
                        S_db = torch.clamp(S_db, min=-top_db, max=20.0)

                if save_npy:
                    arr = S_db.detach().to("cpu").numpy()
                    if storage_dtype == "float16":
                        arr = arr.astype("float16", copy=False)
                    elif storage_dtype == "float32":
                        arr = arr.astype("float32", copy=False)
                    np.save(npy_path, arr)

                del wav_win, S, S_db
                torch.cuda.empty_cache()
