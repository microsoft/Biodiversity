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
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from PytorchWildlife core library
from PytorchWildlife.models.bioacoustics import ResNetClassifier
from PytorchWildlife.models.bioacoustics.resnet_classifier import load_model_from_checkpoint
from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config
from PytorchWildlife.data.bioacoustics.bioacoustics_datasets import BioacousticsInferenceDataset
from PytorchWildlife.data.bioacoustics.bioacoustics_windows import build_inference_windows
from PytorchWildlife.data.bioacoustics.bioacoustics_spectrograms import compute_mel_spectrograms_gpu


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
