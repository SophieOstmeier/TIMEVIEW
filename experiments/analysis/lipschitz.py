import numpy as np
from typing import Dict, List
import itertools
from tqdm import tqdm
import pandas as pd
import ipdb

from scipy.special import gamma
import numpy as np
import pandas as pd
from typing import Dict
from tqdm import tqdm


def estimate_lipschitz_constant(
    feature_types: Dict[str, str],
    feature_ranges: Dict[str, tuple],
    constant_features: Dict[str, float] = {},
    dataset=None,
) -> float:
    print("Estimating Lipschitz constant...")

    if dataset is None or not hasattr(dataset, "X") or not hasattr(dataset, "ys"):
        raise ValueError("Dataset must be provided with X and ys attributes")

    trajectories = dataset.ys
    features = dataset.X

    # Filter out constant features from consideration
    variable_features = {
        f: t for f, t in feature_types.items() if f not in constant_features
    }

    # Sample trajectories
    n_trajectories = len(trajectories)
    sampled_indices = np.random.choice(
        n_trajectories,
        size=min(n_trajectories, 1000),  # Cap sample size for computational efficiency
        replace=False,
    )

    # Ensure features and trajectories align
    if isinstance(features, pd.DataFrame):
        features = features.iloc[sampled_indices]
    trajectories = [trajectories[i] for i in sampled_indices]

    max_ratio = 0
    n_combinations = 0
    total_ratio = 0

    # Find minimum value for each trajectory
    minimums = [min(traj) for traj in trajectories]

    # Compare all pairs of trajectories
    for i in tqdm(range(len(trajectories))):
        for j in range(i + 1, len(trajectories)):
            # Calculate output difference (difference in minimums)
            output_diff = abs(minimums[i] - minimums[j])

            # Calculate normalized input difference
            input_diff = 0
            for feature in variable_features:
                feature_range = feature_ranges[feature]

                # for binary features
                min_range = feature_range[0]
                max_range = feature_range[1]
                if isinstance(min_range, str):
                    max_range = 1
                    min_range = 0

                range_size = max_range - min_range

                # Skip features with zero range to avoid division by zero
                if range_size == 0:
                    continue

                i_features = features[feature].iloc[i]
                j_features = features[feature].iloc[j]
                # for binary features
                if isinstance(i_features, str):
                    i_features = (
                        1 if features[feature].iloc[i] == feature_range[1] else 0
                    )
                    j_features = (
                        1 if features[feature].iloc[j] == feature_range[1] else 0
                    )

                feature_diff = abs(i_features - j_features) / range_size
                input_diff += feature_diff**2

            input_diff = np.sqrt(input_diff)

            # Avoid division by zero
            if input_diff > 0:
                ratio = output_diff / input_diff
                max_ratio = max(max_ratio, ratio)
                total_ratio += ratio
                n_combinations += 1

    average_ratio = total_ratio / n_combinations if n_combinations > 0 else 0
    print(f"Analyzed {n_combinations} trajectory pairs")
    print(f"Average ratio: {average_ratio:.2f}")
    print(f"Maximum ratio (Lipschitz constant): {max_ratio:.2f}")

    return max_ratio, average_ratio
