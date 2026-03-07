# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Window generation utilities for bioacoustics audio segmentation.

This module provides functions for generating audio windows from annotated
audio files for training and inference.
"""

import json
import math
import os
from typing import Dict, List, Union

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


__all__ = [
    "build_windows",
    "build_inference_windows",
    "count_window_labels",
]


def count_window_labels(windows: List[Dict]) -> Dict:
    """Count label distribution in windows.

    Args:
        windows: List of window dicts as returned by :func:`build_windows`.

    Returns:
        Dictionary mapping each label value to its count.
    """
    counts: Dict = {}
    for w in windows:
        label = w.get('label', 0)
        counts[label] = counts.get(label, 0) + 1
    return counts


def build_windows(
    annotation_file: str,
    window_size_sec: float,
    overlap_sec: float,
    sample_rate: int,
    datasets_names: List[str],
    strategy: str = "sliding",
    negative_proportion: float = 0.5,
    multiclass: bool = False,
    min_overlap_sec: float = 0,
    custom_builder=None,
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
            - "customized": Delegates to a user-supplied ``custom_builder``
                           callable.
        negative_proportion: Proportion of negatives for "balanced" strategy.
            0.5 means 50% negatives, 50% positives.
        multiclass: If True, use the annotation's category_id as the label
            instead of a binary 0/1. Defaults to False (binary).
        min_overlap_sec: Minimum overlap in seconds between a window and an
            annotation for the window to be labelled positive. Defaults to 0
            (any overlap counts).
        custom_builder: Callable used when ``strategy="customized"``.  It
            receives ``(annotation_file, sample_rate, datasets_names)`` and
            must return a list of window dicts.

    Returns:
        List of window dicts with keys: 'window_id', 'dataset', 'sample_rate',
        'sound_id', 'start', 'end', 'label'.  When *multiclass* is True an
        extra 'ann_overlap' key is included with the overlap amount in samples.
    """
    if strategy == "sliding":
        return _build_windows_sliding(
            annotation_file, window_size_sec, overlap_sec,
            sample_rate, datasets_names, multiclass, min_overlap_sec
        )
    elif strategy == "balanced":
        return _build_windows_balanced(
            annotation_file, window_size_sec, overlap_sec,
            sample_rate, datasets_names, negative_proportion,
            multiclass, min_overlap_sec
        )
    elif strategy == "customized":
        if custom_builder is None:
            raise ValueError(
                "The 'customized' strategy requires a 'custom_builder' callable."
            )
        return custom_builder(
            annotation_file, sample_rate, datasets_names
        )
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Use 'sliding', 'balanced', or 'customized'."
        )


def _build_windows_sliding(
    annotation_file: str,
    window_size_sec: float,
    overlap_sec: float,
    sample_rate: int,
    datasets_names: List[str],
    multiclass: bool = False,
    min_overlap_sec: float = 0,
) -> List[Dict]:
    """
    Sliding window strategy: generates windows with fixed overlap across entire audio.
    Labels each window based on whether it overlaps with any annotation.

    When *multiclass* is False (default) the label is binary (0 or 1).
    When *multiclass* is True the label is the annotation's ``category_id``
    and an extra ``ann_overlap`` field records the overlap in samples.
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    sounds = {sound['id']: sound for sound in data['sounds']}
    annotations = data['annotations']

    window_size = int(window_size_sec * sample_rate)
    hop_size = int((window_size_sec - overlap_sec) * sample_rate)
    min_overlap_samples = int(min_overlap_sec * sample_rate)

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
                    int(ev['t_max'] * sample_rate),
                    ev.get('category_id', 1)
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
            ann_overlap = 0
            for event_start, event_end, category_id in sound_events:
                if event_end > start and event_start < end:
                    overlap = min(end, event_end) - max(start, event_start)
                    if overlap > min_overlap_samples:
                        label = category_id if multiclass else 1
                        ann_overlap = overlap
                        break

            win = {
                'window_id': window_idx,
                'dataset': dataset,
                'sample_rate': sound["sample_rate"],
                'sound_id': sound_id,
                'start': start,
                'end': end,
                'label': label,
            }
            if multiclass:
                win['ann_overlap'] = ann_overlap
            windows.append(win)
            window_idx += 1

    return windows


def _build_windows_balanced(
    annotation_file: str,
    window_size_sec: float,
    overlap_sec: float,
    sample_rate: int,
    datasets_names: List[str],
    negative_proportion: float = 0.5,
    multiclass: bool = False,
    min_overlap_sec: float = 0,
) -> List[Dict]:
    """
    Balanced strategy: centers windows on annotations for positives,
    then samples negatives to achieve the desired class proportion.

    Args:
        negative_proportion: Final proportion of negative examples in the dataset.
            0.5 means 50% negatives, 50% positives (equal amounts).
            0.7 means 70% negatives, 30% positives.
        multiclass: If True, use the annotation's category_id as the label.
        min_overlap_sec: Minimum overlap in seconds for negative rejection.
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    sounds = {sound['id']: sound for sound in data['sounds']}
    annotations = data['annotations']

    window_size = int(window_size_sec * sample_rate)
    hop_size = int((window_size_sec - overlap_sec) * sample_rate)
    min_overlap_samples = int(min_overlap_sec * sample_rate)

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
                    int(ev['t_max'] * sample_rate),
                    ev.get('category_id', 1)
                ))

        positive_regions = []
        for event_start, event_end, category_id in sound_events:
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
                win = {
                    'window_id': window_idx,
                    'dataset': dataset,
                    'sample_rate': sound["sample_rate"],
                    'sound_id': sound_id,
                    'start': win_start,
                    'end': win_end,
                    'label': category_id if multiclass else 1,
                }
                if multiclass:
                    overlap = (
                        min(win_end, event_end) - max(win_start, event_start)
                    )
                    win['ann_overlap'] = overlap
                positive_windows.append(win)
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

        # Filter annotations for this sound (needed for min_overlap check)
        sound_events = []
        if min_overlap_samples > 0:
            for ev in annotations:
                if ev['sound_id'] == sound_id:
                    sound_events.append((
                        int(ev['t_min'] * sample_rate),
                        int(ev['t_max'] * sample_rate),
                    ))

        # Generate all possible negative windows with overlap
        candidates = []
        start = 0
        while start + window_size <= duration_samples:
            end = start + window_size

            # Check if this window overlaps with any positive region
            is_negative = True
            for pos_start, pos_end in positive_regions:
                if not (end <= pos_start or start >= pos_end):
                    # When min_overlap_sec > 0, only reject if actual
                    # annotation overlap exceeds the threshold
                    if min_overlap_samples > 0:
                        for ev_start, ev_end in sound_events:
                            ov = min(end, ev_end) - max(start, ev_start)
                            if ov > min_overlap_samples:
                                is_negative = False
                                break
                    else:
                        is_negative = False
                    break

            if is_negative:
                candidates.append((start, end))

            start += hop_size

        for start, end in candidates:
            win = {
                'window_id': None,
                'dataset': dataset,
                'sample_rate': sound["sample_rate"],
                'sound_id': sound_id,
                'start': start,
                'end': end,
                'label': 0,
            }
            if multiclass:
                win['ann_overlap'] = 0
            negative_windows.append(win)

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


def build_inference_windows(
    audios_source: Union[str, List[str]],
    window_size_sec: float,
    overlap_sec: float,
    sample_rate: int,
) -> List[Dict]:
    """
    Build inference windows with fixed overlap from audio files.

    Parameters
    ----------
    audios_source : str or list of str
        Path to a directory of audio files, or a list of audio file paths.
    window_size_sec : float
        Window size in seconds.
    overlap_sec : float
        Overlap between consecutive windows in seconds.
    sample_rate : int
        Target sample rate for computing window boundaries.

    Returns
    -------
    list of dict
        Each dict has keys: ``window_id``, ``sound_path``, ``start``, ``end``.
    """
    if librosa is None:
        raise ImportError("librosa is required for build_inference_windows. Install with: pip install librosa")

    window_size = int(window_size_sec * sample_rate)
    hop_size = int((window_size_sec - overlap_sec) * sample_rate)

    windows = []
    window_idx = 0

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
