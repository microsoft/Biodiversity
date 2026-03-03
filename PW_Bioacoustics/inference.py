"""
Unified inference script supporting both binary and multiclass classification.

Usage:
    # Binary inference (default)
    python inference.py --checkpoint model.ckpt --audios_source /path/to/audio --dataset birds

    # Multiclass inference
    python inference.py --checkpoint model.ckpt --audios_source /path/to/audio --dataset whales \
        --num_classes 4 --class_names "No Whale,Humpback,Orca,Beluga"

    # Using config file
    python inference.py --config config/whales.yaml --checkpoint model.ckpt --audios_source /path/to/audio
"""

import os
import argparse
import re
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa
import soundfile as sf

# Import from PytorchWildlife core library
from PytorchWildlife.models.bioacoustics import ResNetClassifier
from PytorchWildlife.data.bioacoustics_datasets import ResizeTo, PerSampleNormalize
from PytorchWildlife.utils.bioacoustics_configs import load_config


class BioacousticsInferenceDataset(Dataset):
    """
    Dataset that reads spectrograms from .npy files whose paths are listed in a dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing at least the column specified by `x_col`.
    x_col : str
        Column containing the path to the .npy file (default: "spec_name").
    target_size : Optional[List[int]]
        [H, W] to resize spectrograms to; if None, keep original size.
    normalize : bool
        Whether to apply per-sample normalization.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        x_col: str = "spec_name",
        target_size: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.df = dataframe
        self.x_col = x_col
        self.paths = self.df[self.x_col].astype(str).tolist()
        self._resize = ResizeTo(target_size) if target_size is not None else None
        self._normalize = PerSampleNormalize() if normalize else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_npy(self, idx: int):
        path = self.paths[idx]
        try:
            arr = np.load(path)
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR loading file at index {idx}:")
            print(f"Path: {path}")
            print(f"Exception: {e}")
            print(f"{'='*80}\n")
            raise
        return arr, path

    def __getitem__(self, idx: int):
        arr, path = self._load_npy(idx)
        arr = arr.astype(np.float32, copy=False)

        # shape to [C,H,W]
        if arr.ndim == 2:
            arr = arr[None, ...]  # [1,H,W]
        elif arr.ndim == 3:
            # if [H,W,C], move channels first
            if arr.shape[0] not in (1, 2, 3) and arr.shape[-1] in (1, 2, 3):
                arr = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Unexpected .npy shape {arr.shape} at index {idx}")

        x = torch.from_numpy(arr)  # [C,H,W]

        if self._normalize is not None:
            x = self._normalize(x)

        if self._resize is not None:
            x = self._resize(x)

        return x, path


def build_inference_windows(
    audios_source: Union[str, List[str]],
    window_size_sec: float,
    overlap_sec: float,
    sample_rate: int,
) -> List[Dict]:
    """
    Build inference windows with fixed overlap.
    Each window is a dict with: 'window_id', 'sound_path', 'start', 'end'
    """
    window_size = int(window_size_sec * sample_rate)
    hop_size = int((window_size_sec - overlap_sec) * sample_rate)

    windows = []
    window_idx = 0

    # Determine whether audios_source is a folder or a list
    if isinstance(audios_source, str):
        wav_files = [
            os.path.join(audios_source, f)
            for f in os.listdir(audios_source)
            if f.lower().endswith(('.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg'))
            and not f.startswith('.')
        ]
    elif isinstance(audios_source, list):
        wav_files = audios_source
    else:
        raise TypeError("audios_source must be either a folder path (str) or a list of file paths (list[str])")

    for filename in wav_files:
        sound_duration = librosa.get_duration(path=filename)
        duration_samples = int(sound_duration * sample_rate)
        num_windows = math.ceil((duration_samples - window_size) / hop_size) + 1

        for i in range(num_windows):
            start = i * hop_size
            end = start + window_size

            if end > duration_samples:
                continue

            windows.append({
                'window_id': window_idx,
                'sound_path': filename,
                'start': start,
                'end': end,
            })
            window_idx += 1

    return windows


def compute_mel_spectrograms_gpu(
    windows: List[Dict],
    sample_rate: int,
    n_fft: int,
    hop_length: Optional[int],
    n_mels: int,
    top_db: float,
    spectrograms_path: str,
    save_npy: bool = True,
    fill_highfreq: bool = True,
    noise_db_mean: Optional[float] = None,
    noise_db_std: float = 3.0,
    storage_dtype: str = "float32",
) -> None:
    """
    GPU-accelerated mel spectrogram computation.
    """
    if hop_length is None:
        hop_length = n_fft // 4

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
            sound_filename = os.path.splitext(os.path.basename(win["sound_path"]))[0]
            base = f"{sound_filename}_{win['start']}_{win['end']}"
            npy_path = os.path.join(spectrograms_path, base + ".npy")
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
        y, orig_sr = sf.read(audio_file_path, dtype="float32", always_2d=False)
        if y.ndim == 2:
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
            f_min=0.0,
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
            sound_filename = os.path.splitext(os.path.basename(win["sound_path"]))[0]
            base = f"{sound_filename}_{start}_{end}"
            npy_path = os.path.join(spectrograms_path, base + ".npy")

            if not os.path.exists(npy_path):
                wav_win = wav_cpu[:, start:end].to(device)
                S = mel_tf(wav_win).squeeze(0)
                S_db = to_db(S)

                # Optional high-frequency fill
                if fill_highfreq and orig_sr < sample_rate:
                    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sample_rate / 2.0)
                    nyq_orig = (float(orig_sr) / 2.0) - 2500
                    noise_mask = torch.from_numpy((mel_freqs > nyq_orig).astype(np.bool_)).to(device)
                    if noise_mask.any():
                        valid_mask = ~noise_mask
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


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> ResNetClassifier:
    """Load trained model from checkpoint."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = ResNetClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    return model.to(device)


def run_inference_batch(
    model: ResNetClassifier,
    dataloader: DataLoader,
    sample_rate: int,
    num_classes: int = 2,
    annotations_json: Optional[str] = None,
    device: str = "cuda",
    temperature: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a batch of data. Supports both binary and multiclass.
    """
    is_binary = (num_classes == 2)
    model.eval()
    all_paths = []
    all_logits = []

    print(f"Running inference on {len(dataloader)} batches...")
    print(f"Mode: {'binary' if is_binary else f'multiclass ({num_classes} classes)'}")

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, paths = batch
            x = x.to(device)

            logits = model(x)
            if is_binary:
                logits = logits.squeeze(1)
            all_logits.append(logits.cpu().numpy())
            all_paths.extend(paths)

    # Parse audio paths, starts, and ends
    annotations = None
    if annotations_json is not None:
        with open(annotations_json, "r") as f:
            annotations = json.load(f)

    audios = []
    starts = []
    ends = []
    for p in all_paths:
        if "start" in p and "end" in p:
            try:
                sound_id = int(re.search(r'sid(\d+)_', p).group(1))
                if annotations:
                    audios.append(next(s["file_name_path"] for s in annotations["sounds"] if s["id"] == sound_id))
                else:
                    audios.append(f"sound_{sound_id}")
                starts.append(float(re.search(r'start(\d+)_end', p).group(1)) / sample_rate)
                ends.append(float(re.search(r'end(\d+)\_lab', p).group(1)) / sample_rate)
            except (AttributeError, StopIteration):
                basename = os.path.basename(p).replace(".npy", "")
                parts = basename.split("_")
                audios.append("_".join(parts[:-2]))
                starts.append(int(parts[-2]) / sample_rate)
                ends.append(int(parts[-1]) / sample_rate)
        else:
            basename = os.path.basename(p).replace(".npy", "")
            parts = basename.split("_")
            audios.append("_".join(parts[:-2]))
            starts.append(int(parts[-2]) / sample_rate)
            ends.append(int(parts[-1]) / sample_rate)

    all_logits = np.concatenate(all_logits)

    if is_binary:
        scaled_logits = all_logits / temperature
        probabilities = 1 / (1 + np.exp(-scaled_logits))
        predictions = (probabilities > 0.5).astype(int)
    else:
        logits_tensor = torch.tensor(all_logits) / temperature
        probabilities = F.softmax(logits_tensor, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)

    return {
        'paths': all_paths,
        'audios': audios,
        'starts': starts,
        'ends': ends,
        'predictions': predictions,
        'probabilities': probabilities,
    }


def process_inference_results_per_second(csv_path: str) -> pd.DataFrame:
    """
    Process inference results CSV and obtain a prediction for each second,
    averaging the predictions that overlap according to the start(s) and end(s) columns.
    """
    df = pd.read_csv(csv_path)
    unique_audios = df['audio'].unique()

    all_results = []

    for audio in unique_audios:
        audio_df = df[df['audio'] == audio].copy()

        min_start = int(np.floor(audio_df['start(s)'].min()))
        max_end = int(np.ceil(audio_df['end(s)'].max()))

        for second in range(min_start, max_end):
            overlapping = audio_df[
                ((audio_df['start(s)'] <= second) & (audio_df['end(s)'] > second)) |
                ((audio_df['start(s)'] < second + 1) & (audio_df['end(s)'] >= second + 1))
            ]

            if len(overlapping) > 0:
                weights = []
                for _, row in overlapping.iterrows():
                    overlap_start = max(row['start(s)'], second)
                    overlap_end = min(row['end(s)'], second + 1)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    weights.append(overlap_duration)

                weights = np.array(weights)

                if weights.sum() > 0:
                    weights = weights / weights.sum()

                    avg_prediction = np.average(overlapping['prediction'], weights=weights)
                    avg_probability = np.average(overlapping['probability'], weights=weights)
                    avg_confidence = np.average(overlapping['confidence'], weights=weights)

                    all_results.append({
                        'audio': audio,
                        'second': second,
                        'count_overlaps': len(overlapping),
                        'prediction': 1 if avg_prediction >= 0.5 else 0,
                        'avg_prediction': avg_prediction,
                        'avg_probability': avg_probability,
                        'avg_confidence': avg_confidence,
                    })

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(['audio', 'second']).reset_index(drop=True)

    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, 'per_second_results.csv')

    results_df.to_csv(output_path, index=False)
    print(f"Per-second results saved to: {output_path}")

    return results_df


def save_inference_results(
    results: Dict,
    output_path: str,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Save inference results to CSV in appropriate format."""
    is_binary = (num_classes == 2)

    if is_binary:
        results_df = pd.DataFrame({
            'audio': results['audios'],
            'start(s)': results['starts'],
            'end(s)': results['ends'],
            'prediction': results['predictions'],
            'probability': results['probabilities'],
            'confidence': np.abs(results['probabilities'] - 0.5) * 2,
        })
        results_df = results_df.sort_values('confidence', ascending=False)
    else:
        data = {
            'file_path': results['paths'],
            'audio': results['audios'],
            'start(s)': results['starts'],
            'end(s)': results['ends'],
            'prediction': results['predictions'],
        }

        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]

        for i, name in enumerate(class_names):
            col_name = name.replace(" ", "_") + "_prob"
            data[col_name] = results['probabilities'][:, i]

        results_df = pd.DataFrame(data)

    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run inference on bioacoustic sounds")

    # Config file (optional)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Audio source
    parser.add_argument("--audios_source", type=str, required=False,
                        help="Path to folder, JSON, or CSV with windows")

    # Classification mode
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes (2=binary, >2=multiclass)")
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="Class names for multiclass")

    # Audio parameters
    parser.add_argument("--window_size_sec", type=float, default=5.0)
    parser.add_argument("--overlap_sec", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=48000)

    # Spectrogram parameters
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=224)
    parser.add_argument("--top_db", type=float, default=80.0)

    # Model and inference
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Output
    parser.add_argument("--dataset", type=str, help="Dataset name for output directory")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--spectrograms_path", type=str, default=None)
    parser.add_argument("--annotations_json", type=str, default=None,
                        help="Annotations JSON for mapping sound IDs to paths")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        cfg = load_config(args.config)

        if args.num_classes == 2 and cfg.training.num_classes != 2:
            args.num_classes = cfg.training.num_classes
        if args.class_names is None and cfg.class_names:
            args.class_names = list(cfg.class_names.values())
        if args.window_size_sec == 5.0:
            args.window_size_sec = cfg.audio.window_size_sec
        if args.overlap_sec == 4.0:
            args.overlap_sec = cfg.audio.overlap_sec
        if args.sample_rate == 48000:
            args.sample_rate = cfg.audio.sample_rate
        if args.hop_length == 512:
            args.hop_length = cfg.spectrogram.hop_length
        if args.n_mels == 224:
            args.n_mels = cfg.spectrogram.n_mels
        if args.n_fft == 2048:
            args.n_fft = cfg.spectrogram.n_fft
        if args.top_db == 80.0:
            args.top_db = cfg.spectrogram.top_db
        if not args.dataset:
            args.dataset = cfg.name

    is_binary = (args.num_classes == 2)
    print(f"Running {'binary' if is_binary else f'multiclass ({args.num_classes} classes)'} inference")

    # Build windows
    if args.audios_source.endswith('.json'):
        with open(args.audios_source, 'r') as in_file:
            windows = json.load(in_file)
        df = pd.DataFrame(windows)
    elif args.audios_source.endswith('.csv'):
        df = pd.read_csv(args.audios_source)
        windows = df.to_dict('records')
    else:
        windows = build_inference_windows(
            audios_source=args.audios_source,
            window_size_sec=args.window_size_sec,
            overlap_sec=args.overlap_sec,
            sample_rate=args.sample_rate,
        )
        df = pd.DataFrame(windows)
        output_dir = os.path.join(".", "inference", args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        windows_path = os.path.join(output_dir, f"{args.dataset}_windows.json")
        with open(windows_path, 'w') as out_file:
            json.dump(windows, out_file, indent=2)
        print(f"Windows saved to: {windows_path}")

    # Setup output and spectrograms directories
    output_dir = os.path.join(".", "inference", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    if args.spectrograms_path:
        spectrograms_path = args.spectrograms_path
    else:
        spectrograms_path = os.path.join(output_dir, "spectrograms")
        os.makedirs(spectrograms_path, exist_ok=True)
        compute_mel_spectrograms_gpu(
            windows=windows,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            top_db=args.top_db,
            spectrograms_path=spectrograms_path,
            save_npy=True,
            fill_highfreq=True,
            noise_db_mean=None,
            noise_db_std=3.0,
            storage_dtype="float32",
        )

    # Build spec_name column
    if 'spec_name' not in df.columns and 'file_path' not in df.columns:
        if 'sound_id' in df.columns and 'label' in df.columns:
            df['spec_name'] = df.apply(
                lambda row: os.path.join(spectrograms_path,
                    f"sid{row.sound_id}_idx{row.window_id}_start{row.start}_end{row.end}_lab{row.label}.npy"),
                axis=1
            )
        elif 'sound_path' in df.columns:
            df['spec_name'] = df.apply(
                lambda row: os.path.join(spectrograms_path,
                    f"{os.path.splitext(os.path.basename(row['sound_path']))[0]}_{row['start']}_{row['end']}.npy"),
                axis=1
            )

    x_col = 'file_path' if 'file_path' in df.columns else 'spec_name'

    # Calculate target size
    n_frames = int(np.ceil((args.window_size_sec * args.sample_rate - args.n_fft) / args.hop_length)) + 1
    target_size = (args.n_mels, n_frames)
    print(f"Spectrogram size: {target_size}")

    # Create dataset
    dataset = BioacousticsInferenceDataset(
        dataframe=df,
        x_col=x_col,
        target_size=target_size,
        normalize=args.normalize,
    )
    print(f"Created dataset with {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"
    print(f"Using device: {args.device}")

    # Load model
    try:
        model = load_model_from_checkpoint(args.checkpoint, args.device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run inference
    try:
        results = run_inference_batch(
            model=model,
            dataloader=dataloader,
            sample_rate=args.sample_rate,
            num_classes=args.num_classes,
            annotations_json=args.annotations_json,
            device=args.device,
            temperature=args.temperature,
        )
        print("Inference completed successfully")
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # Save results
    suffix = "binary" if is_binary else "multiclass"
    results_path = os.path.join(output_dir, f"{suffix}_inference_results.csv")
    save_inference_results(
        results=results,
        output_path=results_path,
        num_classes=args.num_classes,
        class_names=args.class_names,
    )

    print("Inference pipeline completed successfully!")


if __name__ == "__main__":
    main()
