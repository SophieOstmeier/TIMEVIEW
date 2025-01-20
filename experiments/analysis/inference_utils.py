from datetime import datetime
import matplotlib.pyplot as plt
import json
import os
import random
import numpy as np
from typing import Dict, List
from contextlib import contextmanager
import sys
import io
import ipdb
from contextlib import contextmanager, nullcontext
import math


def get_splits_from_summary(timestamp, results_dir, summary_filename):
    # Load summary file
    with open(os.path.join(results_dir, summary_filename), "r") as summary_file:
        summary = json.load(summary_file)

    # Find the entry matching the timestamp
    for result in summary:
        if result["timestamp"] == timestamp:
            train_size = result["train_size"]
            val_size = result["val_size"]
            test_size = 1 - train_size - val_size
            seed = result["seed"]

            return [train_size, val_size, test_size], seed

    raise ValueError(f"No entry found for timestamp {timestamp} in summary file")


def normalize_feature_to_np(input_dict: dict) -> np.ndarray:
    if not input_dict:
        return np.array([])

    normalized_values = []

    for feature_data in input_dict.values():
        # Determine whether to use 'range' or 'possible_values' for bounds
        try:
            bounds_key = (
                "possible_values" if "possible_values" in feature_data else "range"
            )

            if bounds_key == "possible_values":
                # Convert possible values list to min/max bounds
                values = feature_data[bounds_key]
                if isinstance(values[0], str):
                    bounds = {"min": 0, "max": 1}

                else:
                    bounds = {"min": min(values), "max": max(values)}
            else:
                bounds = feature_data[bounds_key]

            # Perform min-max normalization
            value = feature_data["value"]
            if isinstance(value, str):
                value = 0 if value == feature_data[bounds_key][0] else 1
            min_val = bounds["min"]
            max_val = bounds["max"]

            normalized_value = (value - min_val) / (max_val - min_val)
            normalized_values.append(normalized_value)
        except:
            ipdb.set_trace()

    return np.array(normalized_values)


def mute_prints(should_mute=True):
    """Returns context manager that optionally suppresses prints"""
    if should_mute:

        @contextmanager
        def mute():
            stdout = sys.stdout  # Store original stdout
            sys.stdout = io.StringIO()  # Redirect stdout to string buffer
            try:
                yield
            finally:
                sys.stdout = stdout  # Restore original stdout

        return mute()
    return nullcontext()


def random_value(feature_name: str, dataset: str, dataset_features):
    """Generate random value for a feature within its specified range"""
    feature_ranges = dataset_features[dataset]
    if feature_name in feature_ranges:
        if isinstance(feature_ranges[feature_name], list):
            return random.choice(feature_ranges[feature_name])
        else:
            min_val, max_val = feature_ranges[feature_name]
            return round(random.uniform(min_val, max_val), 2)
    return None


def get_base_features(dataset: str, dataset_features) -> Dict:
    """Generate base features dictionary based on dataset"""
    feature_ranges = dataset_features[dataset]
    return {
        feature: random_value(feature, dataset, dataset_features)
        for feature in feature_ranges.keys()
    }
