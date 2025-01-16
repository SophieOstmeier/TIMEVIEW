# First, keep all your existing setup code
# %matplotlib inline
import matplotlib.pyplot as plt
import sys
import np

sys.path.append("../../")
from timeview.visualize import expert_tts_plot, grid_tts_plot
from experiments.datasets import load_dataset
from experiments.benchmark import (
    load_column_transformer,
    create_benchmark_datasets_if_not_exist,
)
from timeview.lit_module import load_model
from experiments.baselines import YNormalizer
from experiments.analysis.analysis_utils import find_results
from experiments.benchmark import generate_indices
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import product
import numpy as np
import pandas as pd
from timeview.basis import BSplineBasis
import os
import json


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


def evaluate_model_predictions(
    settings,
    dataset,
    feature_types,
    feature_ranges,
    trajectories,
    time_points,
    train_size=0.7,
    val_size=0.15,
    seed=0,
    gamma=0.05,
    n_perturbations=100,
):
    """
    Evaluate model predictions using multiple metrics.
    """
    # Generate train/val/test splits
    _, _, test_indices = generate_indices(len(dataset), train_size, val_size, seed)

    # Get test data features and trajectories from dataset
    test_features = dataset.X.iloc[test_indices]
    test_trajectories = [dataset.ys[i] for i in test_indices]

    def normalize_features(features):
        """Normalize features to [0,1] range for distance calculation"""
        normalized = {}
        for feature, value in features.items():
            if feature_types[feature] == "continuous":
                min_val, max_val = feature_ranges[feature]
                normalized[feature] = (value - min_val) / (max_val - min_val)
            else:
                # For categorical features, use one-hot encoding
                possible_values = feature_ranges[feature]
                normalized[feature] = 1 if value in possible_values else 0
        return normalized

    def euclidean_distance(features1, features2):
        """Calculate normalized Euclidean distance between feature sets"""
        norm1 = normalize_features(features1)
        norm2 = normalize_features(features2)

        squared_diff_sum = 0
        for feature in norm1:
            squared_diff_sum += (norm1[feature] - norm2[feature]) ** 2

        return np.sqrt(squared_diff_sum)

    # Calculate declines for each trajectory
    declines = trajectories[:, -1] - trajectories[:, 0]  # End point minus start point

    # Find trajectory with least decline (most negative change)
    best_idx = np.argmin(declines)
    best_setting = settings[best_idx]

    # Find closest matching example in test data
    min_distance = float("inf")
    closest_example = None
    closest_idx = None

    # Convert test_features rows to dictionaries and find closest example
    for i, feature_row in test_features.iterrows():
        feature_dict = feature_row.to_dict()
        distance = euclidean_distance(best_setting, feature_dict)

        if distance < min_distance:
            min_distance = distance
            closest_example = feature_dict
            closest_idx = i

    # Calculate metrics
    # 1. Prediction Error - using decline (end - start)
    predicted_decline = trajectories[best_idx][-1] - trajectories[best_idx][0]

    # Get actual trajectory for closest example
    actual_trajectory = dataset.ys[closest_idx]
    actual_decline = actual_trajectory[-1] - actual_trajectory[0]

    E_pred = np.abs(predicted_decline - actual_decline)

    # 2. Relative Improvement - using average decline across test set
    test_declines = [traj[-1] - traj[0] for traj in test_trajectories]
    avg_test_decline = np.mean(test_declines)

    R_imp = ((actual_decline - avg_test_decline) / avg_test_decline) * 100

    # 3. Stability Score - using decline values
    def generate_perturbation(features):
        """Generate small random perturbations of features"""
        perturbed = features.copy()
        for feature, value in features.items():
            if feature_types[feature] == "continuous":
                min_val, max_val = feature_ranges[feature]
                range_size = max_val - min_val
                perturbation = np.random.uniform(
                    -gamma * range_size, gamma * range_size
                )
                perturbed[feature] = np.clip(value + perturbation, min_val, max_val)
            # For categorical features, we don't apply perturbations
        return perturbed

    # Generate perturbed versions and their decline values
    # perturbed_declines = []
    # for _ in range(n_perturbations):
    #     perturbed_features = generate_perturbation(best_setting)
    #     # Get trajectory for perturbed features through original model inference
    #     perturbed_trajectory = dataset._tumor_volume_2(
    #         time_points,
    #         perturbed_features.values(),
    #     )
    #     perturbed_declines.append(perturbed_trajectory[-1] - perturbed_trajectory[0])

    # S_score = np.std(perturbed_declines) / np.mean(perturbed_declines)

    return {
        "prediction_error": E_pred,
        "relative_improvement": R_imp,
        # "stability_score": S_score,
        "closest_example": closest_example,
        "closest_idx": closest_idx,
        "normalized_distance": min_distance,
        "test_indices": test_indices,
    }


# Now add the inference function
def tts_inference(
    litmodel,
    dataset,
    feature_combinations,
    n_points=100,
    y_normalizer=None,
    column_transformer=None,
    return_transition_points=False,
):
    """
    [Previous docstring remains the same]
    """

    # Convert input to DataFrame if it's a list of dictionaries
    if isinstance(feature_combinations, list):
        feature_combinations = pd.DataFrame(feature_combinations)

    # Verify features
    feature_names = dataset.get_feature_names()
    for feature in feature_names:
        if feature not in feature_combinations.columns:
            raise ValueError(f"Missing feature: {feature}")

    # Get time points
    time_horizon = litmodel.config.T
    t = np.linspace(0, time_horizon, n_points)

    # Initialize BSpline basis
    config = litmodel.config
    bspline = BSplineBasis(
        config.n_basis, (0, config.T), internal_knots=config.internal_knots
    )

    # Transform features if needed
    if column_transformer is not None:
        transformed_features = column_transformer.transform(
            feature_combinations[feature_names]
        )
    else:
        transformed_features = feature_combinations[feature_names].values

    # Get trajectories
    trajectories = []
    transition_points_list = []

    for i in range(len(transformed_features)):
        # Get trajectory
        trajectory = litmodel.model.forecast_trajectory(transformed_features[i], t)

        # Apply inverse transform if needed
        if y_normalizer is not None:
            trajectory = y_normalizer.inverse_transform(
                trajectory.reshape(-1, 1)
            ).flatten()

        trajectories.append(trajectory)

        # Get transition points if requested
        if return_transition_points:
            coeffs = litmodel.model.predict_latent_variables(
                transformed_features[i].reshape(1, -1)
            )
            _, transition_points = bspline.get_template_from_coeffs(coeffs[0, :])

            # Get values at transition points
            transition_values = litmodel.model.forecast_trajectory(
                transformed_features[i], np.array(transition_points)
            )

            # Apply inverse transform if needed
            if y_normalizer is not None:
                transition_values = y_normalizer.inverse_transform(
                    transition_values.reshape(-1, 1)
                ).flatten()

            transition_points_list.append((transition_points, transition_values))

    result = {"t": t, "trajectories": np.array(trajectories)}

    if return_transition_points:
        result["transition_points"] = transition_points_list

    return result


def get_feature_combinations(
    feature_types,
    feature_ranges,
    constant_features,
    n_subset=100,
):
    # Separate varying and constant features
    varying_features = {
        k: v for k, v in feature_ranges.items() if k not in constant_features
    }

    # Create feature points dictionary
    feature_points = {}
    for feature, range_val in varying_features.items():
        if feature_types[feature] == "continuous":
            feature_points[feature] = np.linspace(range_val[0], range_val[1], n_subset)
        else:  # categorical or binary
            feature_points[feature] = range_val

    # Generate all combinations using itertools.product
    feature_names = list(feature_points.keys())
    combinations = product(*[feature_points[f] for f in feature_names])

    # Create settings list
    feature_combinations = []
    for combo in combinations:
        setting = constant_features.copy()  # Start with constant features
        setting.update(dict(zip(feature_names, combo)))  # Add varying features
        feature_combinations.append(setting)

    # Print summary
    print(f"\nAnalysis Summary:")
    print(f"Number of combinations: {len(feature_combinations)}")
    print("\nConstant features:")
    for k, v in constant_features.items():
        print(f"  {k}: {v}")
    print("\nVarying features:")
    for k in varying_features.keys():
        if feature_types[k] == "continuous":
            print(
                f"  {k}: {n_subset} points from {feature_ranges[k][0]} to {feature_ranges[k][1]}"
            )
        else:
            print(f"  {k}: {len(feature_ranges[k])} categories")

    # Example settings
    print("\nExample settings:")
    for i, setting in enumerate(feature_combinations[:3]):
        print(f"\nSetting {i+1}:")
        for k, v in setting.items():
            print(f"  {k}: {v}")

    summary_dict = {
        "n_combinations": len(feature_combinations),
        "constant_features": constant_features,
        "varying_features": varying_features,
        "feature_ranges": feature_ranges,
        "feature_types": feature_types,
    }

    return feature_combinations, summary_dict


def run_inference(
    dataset,
    litmodel,
    column_transformer,
    y_normalizer,
    constant_features,
    splits: list,
    n_subset=10,
    seed=0,
):
    # Get feature information
    feature_ranges = dataset.get_feature_ranges()
    feature_types = dataset.get_feature_types()

    # Get feature combinations
    feature_combinations, summary_dict = get_feature_combinations(
        feature_ranges=feature_ranges,
        feature_types=feature_types,
        constant_features=constant_features,
        n_subset=n_subset,
    )

    results = tts_inference(
        litmodel=litmodel,
        dataset=dataset,
        feature_combinations=feature_combinations,
        y_normalizer=y_normalizer,
        column_transformer=column_transformer,
        return_transition_points=False,
    )

    # Calculate decline for each trajectory
    time_points = results["t"]
    trajectories = results["trajectories"]

    declines = trajectories[:, -1] - trajectories[:, 0]  # End point minus start point

    # Find index of trajectory with least decline
    best_idx = np.argmin(declines)

    # Print the feature setting with least decline
    # Print the feature setting with least decline
    print("\nFeature setting with least decline:")
    print(f"Total change in y: {declines[best_idx]:.2f}")
    print("Feature values (value [range]):")
    best_setting = feature_combinations[best_idx]

    # First print constant features with their values
    print("\nConstant features:")
    for key, value in constant_features.items():
        print(f"  {key}: {value}")

    # Then print varying features with their ranges
    print("\nVarying features:")
    for key, value in best_setting.items():
        if key not in constant_features:  # Only print varying features
            if feature_types[key] == "continuous":
                range_val = feature_ranges[key]
                print(
                    f"  {key}: {value:.2f} [range: {range_val[0]:.2f} to {range_val[1]:.2f}]"
                )
            else:  # categorical or binary
                range_val = feature_ranges[key]
                print(
                    f"  {key}: {value} [possible values: {', '.join(map(str, range_val))}]"
                )

    # Optionally highlight this trajectory in the plot
    plt.figure(figsize=(12, 8))
    # Plot all trajectories in light gray
    for trajectory in results["trajectories"]:
        plt.plot(time_points, trajectory, alpha=0.1, color="gray")
    # Plot the best trajectory in blue
    plt.plot(
        time_points, trajectories[best_idx], "b-", linewidth=2, label="Least decline"
    )
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(
        f"Trajectories with {constant_features} \n(n={len(feature_combinations)})"
    )
    plt.legend()
    plt.grid(True)

    # Save the figure
    filename = os.path.join(
        "/dataNAS/people/sostm/TIMEVIEW/plots",
        f"trajectories{constant_features}.png",
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved as: {filename}")
    evaluation_results = evaluate_model_predictions(
        settings=feature_combinations,
        dataset=dataset,
        feature_types=feature_types,
        feature_ranges=feature_ranges,
        trajectories=trajectories,
        time_points=time_points,
        train_size=splits[0],
        val_size=splits[1],
        seed=seed,
    )

    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"Prediction Error: {evaluation_results['prediction_error']:.4f}")
    print(f"Relative Improvement: {evaluation_results['relative_improvement']:.2f}%")
    # print(f"Stability Score: {evaluation_results['stability_score']:.4f}")
    print(
        f"Distance to closest example: {evaluation_results['normalized_distance']:.4f}"
    )


if __name__ == "__main__":
    # Your existing setup code
    dataset_name = "flchain_1000"
    # dataset_name = "synthetic_tumor_wilkerson_1"
    # constant_features = {"age": 75.0, "initial_tumor_volume": 0.10}
    model_name = "TTS"
    constant_features = {"age": 75.0, "sex": "F"}
    root = "/dataNAS/people/sostm/TIMEVIEW/experiments"
    results_dir = f"{root}/benchmarks"
    summary_filename = f"{root}/benchmarks/summary.json"
    dataset_description_path = f"{root}/dataset_descriptions"

    create_benchmark_datasets_if_not_exist(
        dataset_description_path=f"{root}/dataset_descriptions"
    )
    results = find_results(
        dataset_name,
        model_name,
        results_dir=results_dir,
        summary_filename=summary_filename,
    )

    if len(results) == 0:
        print(f"No results found for {dataset_name} and {model_name}")
        print("Make sure you run your experiments from ../run_scripts")
    elif len(results) > 1:
        print("Multiple results found for the given dataset and model")
        print("We take the last one but it may produce unexpected results")

    timestamp = results[-1]

    spilt_sizes, seed = get_splits_from_summary(
        timestamp, results_dir, summary_filename
    )
    litmodel = load_model(
        timestamp, seed=661058651, benchmarks_folder=f"{root}/benchmarks"
    )
    dataset = load_dataset(
        dataset_name,
        dataset_description_path=f"{root}/dataset_descriptions",
        data_folder=f"{root}/data",
    )
    column_transformer = load_column_transformer(
        timestamp, benchmarks_dir=f"{root}/benchmarks"
    )
    y_normalizer = YNormalizer.load_from_benchmark(
        timestamp, model_name, benchmark_dir=f"{root}/benchmarks"
    )

    run_inference(
        litmodel=litmodel,
        dataset=dataset,
        y_normalizer=y_normalizer,
        column_transformer=column_transformer,
        constant_features=constant_features,
        splits=spilt_sizes,
        seed=seed,
    )
