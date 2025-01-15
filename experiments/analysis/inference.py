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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import product


def analyze_trajectories(
    litmodel,
    dataset,
    column_transformer,
    y_normalizer,
    constant_features,
    n_subset=5,
    plot_dir="/dataNAS/people/sostm/TIMEVIEW/plots",
):
    """
    Comprehensive analysis of trajectories with constant and varying features.

    Parameters:
    -----------
    litmodel : TTSModel
        The trained trajectory time series model
    dataset : Dataset
        The dataset object containing feature information
    column_transformer : ColumnTransformer
        Transformer used to preprocess features
    y_normalizer : YNormalizer
        Normalizer for the target variable
    constant_features : dict
        Dictionary of features to keep constant and their values
    n_subset : int, optional (default=5)
        Number of points to sample for each continuous feature
    plot_dir : str, optional
        Directory to save plots

    Returns:
    --------
    dict with keys:
        'settings': List of all feature combinations tested
        'results': Dict containing trajectories and time points
        'best_combination': Dict of features giving best trajectory
        'best_trajectory': Array of best trajectory values
        'best_decline': Float value of smallest decline
    """
    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # Get feature information
    feature_ranges = dataset.get_feature_ranges()
    feature_types = dataset.get_feature_types()

    # Separate constant and varying features
    varying_features = {
        k: v for k, v in feature_ranges.items() if k not in constant_features
    }

    # Create ranges for continuous varying features
    feature_points = {}
    for feature, range_val in varying_features.items():
        if isinstance(range_val[0], (int, float)):  # Continuous feature
            feature_points[feature] = np.linspace(range_val[0], range_val[1], n_subset)
        else:  # Discrete feature
            feature_points[feature] = range_val

    # Generate all combinations
    feature_names = list(feature_points.keys())
    combinations = product(*[feature_points[f] for f in feature_names])

    # Create settings list
    settings = []
    for combo in combinations:
        setting = constant_features.copy()
        setting.update(dict(zip(feature_names, combo)))
        settings.append(setting)

    print(f"\nTotal number of combinations: {len(settings)}")
    print("\nExample of first few settings:")
    for i, setting in enumerate(settings[:3]):
        print(f"\nSetting {i+1}:")
        for k, v in setting.items():
            print(f"{k}: {v}")

    # Get predictions
    results = tts_inference(
        litmodel=litmodel,
        dataset=dataset,
        feature_combinations=settings,
        y_normalizer=y_normalizer,
        column_transformer=column_transformer,
        return_transition_points=False,
    )

    # Calculate declines and find best trajectory
    declines = []
    for trajectory in results["trajectories"]:
        decline = trajectory[-1] - trajectory[0]
        declines.append(decline)

    best_idx = np.argmax(declines)

    print("\nFeature combination leading to smallest decline:")
    for feature, value in settings[best_idx].items():
        print(f"{feature}: {value}")
    print(f"\nDecline value: {declines[best_idx]:.4f}")

    # Create plots
    # 1. All trajectories
    plt.figure(figsize=(12, 8))
    time_points = results["t"]
    for trajectory in results["trajectories"]:
        plt.plot(time_points, trajectory, alpha=0.1)
    plt.xlabel("Time")
    plt.ylabel("Value")

    constant_features_str = ", ".join(
        [f"{k}={v}" for k, v in constant_features.items()]
    )
    plt.title(f"All Trajectories\n({constant_features_str})")
    plt.grid(True)

    filename = os.path.join(plot_dir, "all_trajectories.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Best trajectory highlighted
    plt.figure(figsize=(12, 8))

    # Plot all trajectories in light gray
    for trajectory in results["trajectories"]:
        plt.plot(time_points, trajectory, alpha=0.1, color="gray")

    # Plot the best trajectory in red
    plt.plot(
        time_points,
        results["trajectories"][best_idx],
        "r-",
        linewidth=2,
        label="Best trajectory",
    )

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(
        f"All Trajectories with Best Trajectory Highlighted\n({constant_features_str})"
    )
    plt.grid(True)
    plt.legend()

    filename = os.path.join(plot_dir, "best_trajectory.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "settings": settings,
        "results": results,
        "best_combination": settings[best_idx],
        "best_trajectory": results["trajectories"][best_idx],
        "best_decline": declines[best_idx],
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
    import numpy as np
    import pandas as pd
    from timeview.basis import BSplineBasis

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


if __name__ == "__main__":
    # Your existing setup code
    dataset_name = "flchain_1000"
    model_name = "TTS"
    root = "/dataNAS/people/sostm/TIMEVIEW/experiments"
    create_benchmark_datasets_if_not_exist(
        dataset_description_path=f"{root}/dataset_descriptions"
    )
    results = find_results(
        dataset_name,
        model_name,
        results_dir=f"{root}/benchmarks",
        summary_filename=f"{root}/benchmarks/summary.json",
    )

    if len(results) == 0:
        print(f"No results found for {dataset_name} and {model_name}")
        print("Make sure you run your experiments from ../run_scripts")
    elif len(results) > 1:
        print("Multiple results found for the given dataset and model")
        print("We take the last one but it may produce unexpected results")

    timestamp = results[-1]
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
    # First, let's get the ranges from the dataset
    feature_ranges = dataset.get_feature_ranges()
    feature_types = dataset.get_feature_types()

    # Set constant values
    constant_features = {"age": 75.0, "sex": "F"}  # fixed age  # fixed sex

    # Create variations for other features
    n_points = 100
    creatinine_range = np.linspace(
        feature_ranges["creatinine"][0], feature_ranges["creatinine"][1], n_points
    )
    kappa_range = np.linspace(
        feature_ranges["kappa"][0], feature_ranges["kappa"][1], n_points
    )
    lambda_range = np.linspace(
        feature_ranges["lambda"][0], feature_ranges["lambda"][1], n_points
    )

    # For discrete features, use all possible values
    flc_grp_values = feature_ranges["flc.grp"]
    mgus_values = feature_ranges["mgus"]

    # Create all combinations with constant age and sex
    n_subset = 5  # take 5 evenly spaced points from each continuous range
    settings = [
        {
            **constant_features,  # unpack constant features
            "creatinine": creat,
            "kappa": kap,
            "lambda": lam,
            "flc.grp": flc,
            "mgus": mgus,
        }
        for creat in np.linspace(creatinine_range[0], creatinine_range[-1], n_subset)
        for kap in np.linspace(kappa_range[0], kappa_range[-1], n_subset)
        for lam in np.linspace(lambda_range[0], lambda_range[-1], n_subset)
        for flc in flc_grp_values
        for mgus in mgus_values
    ]

    print(f"\nTotal number of combinations: {len(settings)}")
    print("\nExample of first few settings:")
    for i, setting in enumerate(settings[:3]):
        print(f"\nSetting {i+1}:")
        for k, v in setting.items():
            print(f"{k}: {v}")

    # Get predictions
    results = tts_inference(
        litmodel=litmodel,
        dataset=dataset,
        feature_combinations=settings,
        y_normalizer=y_normalizer,
        column_transformer=column_transformer,
        return_transition_points=False,
    )

    # Plot results
    import os

    plot_dir = "/dataNAS/people/sostm/TIMEVIEW/plots"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    time_points = results["t"]
    for trajectory in results["trajectories"]:
        plt.plot(time_points, trajectory, alpha=0.1)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(
        f'Trajectories with Age={constant_features["age"]}, Sex={constant_features["sex"]} \n(n={len(settings)})'
    )
    plt.grid(True)

    # Save the figure
    filename = os.path.join(
        plot_dir,
        f"trajectories_age{constant_features['age']}_sex{constant_features['sex']}.png",
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved as: {filename}")
