# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Window generation utilities for bioacoustics audio segmentation.

This module provides functions for generating audio windows from annotated
audio files for training and inference.
"""

import json
import math
from typing import Dict, List

import numpy as np


__all__ = [
    "build_windows",
]


def build_windows(
    annotation_file: str,
    window_size_sec: float,
    overlap_sec: float,
    sample_rate: int,
    datasets_names: List[str],
    strategy: str = "sliding",
    negative_proportion: float = 0.5,
) -> List[Dict]:
    """
    Build audio windows with their labels using the specified strategy.

    Args:
        annotation_file: Path to the annotations JSON file.
        window_size_sec: Window size in seconds.
        overlap_sec: Overlap between windows in seconds.
        sample_rate: Sample rate for calculations.
        datasets_names: List of dataset names to match against file paths.
        strategy: Window generation strategy:
            - "sliding": Fixed overlap sliding windows across entire audio.
                        Labels based on annotation overlap.
            - "balanced": Centers windows on annotations for positives,
                         then samples negatives to achieve desired proportion.
        negative_proportion: Proportion of negatives for "balanced" strategy.
            0.5 means 50% negatives, 50% positives.

    Returns:
        List of window dicts with keys: 'window_id', 'dataset', 'sample_rate',
        'sound_id', 'start', 'end', 'label'.
    """
    if strategy == "sliding":
        return _build_windows_sliding(
            annotation_file, window_size_sec, overlap_sec,
            sample_rate, datasets_names
        )
    elif strategy == "balanced":
        return _build_windows_balanced(
            annotation_file, window_size_sec, overlap_sec,
            sample_rate, datasets_names, negative_proportion
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'sliding' or 'balanced'.")


def _build_windows_sliding(
    annotation_file: str,
    window_size_sec: float,
    overlap_sec: float,
    sample_rate: int,
    datasets_names: List[str],
) -> List[Dict]:
    """
    Sliding window strategy: generates windows with fixed overlap across entire audio.
    Labels each window based on whether it overlaps with any annotation.
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    sounds = {sound['id']: sound for sound in data['sounds']}
    annotations = data['annotations']

    window_size = int(window_size_sec * sample_rate)
    hop_size = int((window_size_sec - overlap_sec) * sample_rate)

    windows = []
    window_idx = 0

    for sound_id, sound in sounds.items():
        duration_samples = int(sound['duration'] * sample_rate)
        num_windows = math.ceil((duration_samples - window_size) / hop_size) + 1

        # Filter annotations for this sound
        sound_events = []
        for ev in annotations:
            if ev['sound_id'] == sound_id:
                sound_events.append((
                    int(ev['t_min'] * sample_rate),
                    int(ev['t_max'] * sample_rate)
                ))

        dataset = None
        for dataset_name in datasets_names:
            if dataset_name in sound['file_name_path']:
                dataset = dataset_name

        for i in range(num_windows):
            start = i * hop_size
            end = start + window_size

            if end > duration_samples:
                continue

            # Check overlap with any event
            label = 0
            for event_start, event_end in sound_events:
                if event_end > start and event_start < end:
                    label = 1
                    break

            windows.append({
                'window_id': window_idx,
                'dataset': dataset,
                'sample_rate': sound["sample_rate"],
                'sound_id': sound_id,
                'start': start,
                'end': end,
                'label': label
            })
            window_idx += 1

    return windows


def _build_windows_balanced(
    annotation_file: str,
    window_size_sec: float,
    overlap_sec: float,
    sample_rate: int,
    datasets_names: List[str],
    negative_proportion: float = 0.5,
) -> List[Dict]:
    """
    Balanced strategy: centers windows on annotations for positives,
    then samples negatives to achieve the desired class proportion.

    Args:
        negative_proportion: Final proportion of negative examples in the dataset.
            0.5 means 50% negatives, 50% positives (equal amounts).
            0.7 means 70% negatives, 30% positives.
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    sounds = {sound['id']: sound for sound in data['sounds']}
    annotations = data['annotations']

    window_size = int(window_size_sec * sample_rate)
    hop_size = int((window_size_sec - overlap_sec) * sample_rate)

    window_idx = 0
    positive_windows = []
    all_positive_regions = {}

    # Step 1: Extract all positive examples (annotations)
    for sound_id, sound in sounds.items():
        duration_samples = int(sound['duration'] * sample_rate)

        dataset = None
        for dataset_name in datasets_names:
            if dataset_name in sound['file_name_path']:
                dataset = dataset_name
                break

        sound_events = []
        for ev in annotations:
            if ev['sound_id'] == sound_id:
                sound_events.append((
                    int(ev['t_min'] * sample_rate),
                    int(ev['t_max'] * sample_rate)
                ))

        positive_regions = []
        for event_start, event_end in sound_events:
            # Center the window on the annotation
            annotation_center = (event_start + event_end) // 2
            win_start = annotation_center - window_size // 2
            win_end = win_start + window_size

            # Adjust if window goes beyond audio boundaries
            if win_start < 0:
                win_start = 0
                win_end = window_size
            elif win_end > duration_samples:
                win_end = duration_samples
                win_start = win_end - window_size

            # Only add if we have enough samples
            if win_end - win_start == window_size and win_end <= duration_samples:
                positive_windows.append({
                    'window_id': window_idx,
                    'dataset': dataset,
                    'sample_rate': sound["sample_rate"],
                    'sound_id': sound_id,
                    'start': win_start,
                    'end': win_end,
                    'label': 1
                })
                positive_regions.append((win_start, win_end))
                window_idx += 1

        all_positive_regions[sound_id] = positive_regions

    # Step 2: Sample negative examples based on proportion
    num_positives = len(positive_windows)
    # Calculate negatives needed: negatives / (positives + negatives) = negative_proportion
    num_negatives_needed = int(num_positives * negative_proportion / (1 - negative_proportion))

    print(f"Positive examples found: {num_positives}")
    print(f"Negative examples needed: {num_negatives_needed}")
    print(f"Desired proportion - Negatives: {negative_proportion:.1%}, Positives: {(1-negative_proportion):.1%}")

    negative_windows = []

    # Generate candidate negative windows for each sound
    for sound_id, sound in sounds.items():
        if sound_id not in all_positive_regions:
            continue

        duration_samples = int(sound['duration'] * sample_rate)
        positive_regions = all_positive_regions[sound_id]

        dataset = None
        for dataset_name in datasets_names:
            if dataset_name in sound['file_name_path']:
                dataset = dataset_name
                break

        # Generate all possible negative windows with overlap
        candidates = []
        start = 0
        while start + window_size <= duration_samples:
            end = start + window_size

            # Check if this window overlaps with any positive region
            is_negative = True
            for pos_start, pos_end in positive_regions:
                if not (end <= pos_start or start >= pos_end):
                    is_negative = False
                    break

            if is_negative:
                candidates.append((start, end))

            start += hop_size

        for start, end in candidates:
            negative_windows.append({
                'window_id': None,
                'dataset': dataset,
                'sample_rate': sound["sample_rate"],
                'sound_id': sound_id,
                'start': start,
                'end': end,
                'label': 0
            })

    # Shuffle and select the required number of negative examples
    np.random.shuffle(negative_windows)
    print(f"Negative examples available: {len(negative_windows)}")
    selected_negatives = negative_windows[:num_negatives_needed]

    # Assign window IDs to selected negatives
    for neg_win in selected_negatives:
        neg_win['window_id'] = window_idx
        window_idx += 1

    print(f"Negative examples selected: {len(selected_negatives)}")
    final_total = len(selected_negatives) + num_positives
    print(f"Final proportion - Negatives: {len(selected_negatives)/final_total:.1%}, Positives: {num_positives/final_total:.1%}")

    return positive_windows + selected_negatives
