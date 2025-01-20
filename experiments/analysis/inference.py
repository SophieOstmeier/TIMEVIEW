# First, keep all your existing setup code
# %matplotlib inline
import matplotlib.pyplot as plt
import sys
import np
import ipdb

sys.path.append("../../")
from experiments.benchmark import generate_indices
from lipschitz import estimate_lipschitz_constant
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
from experiments.datasets import load_dataset
from experiments.benchmark import (
    load_column_transformer,
    create_benchmark_datasets_if_not_exist,
)
from timeview.lit_module import load_model
from experiments.baselines import YNormalizer
from experiments.analysis.analysis_utils import find_results
import os
import json
import datetime
from experiments.analysis.inference_construct_grid import (
    get_feature_combinations_l2,
    get_feature_combinations,
)
from experiments.analysis.inference_utils import get_splits_from_summary


def evaluate_model_predictions(
    dataset,
    feature_types,
    feature_ranges,
    trajectories,
    time_points,
    feature_combinations,
    constant_features,
    result_dir,
    train_size=0.7,
    val_size=0.15,
    seed=0,
):

    # Find minimum for each trajectory
    min_per_trajectory = [
        (i, np.argmin(traj), np.min(traj)) for i, traj in enumerate(trajectories)
    ]

    # Find the overall minimum
    best_traj_idx, best_time_idx, best_value = min(
        min_per_trajectory, key=lambda x: x[2]
    )
    # Get the best setting
    best_setting = feature_combinations[best_traj_idx]

    # Create a results dictionary
    summary_dict = {
        "best_decline": {
            "value": float(best_value),
            "index": float(best_traj_idx),
            "time_index": float(best_time_idx),
        },
        "constant_features": {key: value for key, value in constant_features.items()},
        "best_varying_features": {},
    }

    # Add varying features with their values and ranges
    for key, value in best_setting.items():
        if key not in constant_features:
            feature_info = {"value": value, "type": feature_types[key]}

            if feature_types[key] == "continuous":
                feature_info["range"] = {
                    "min": feature_ranges[key][0],
                    "max": feature_ranges[key][1],
                }
            else:  # categorical or binary
                feature_info["possible_values"] = feature_ranges[key]

            summary_dict["best_varying_features"][key] = feature_info
    plt.figure(figsize=(12, 8))
    # Plot all trajectories in light gray
    for trajectory in trajectories:
        plt.plot(time_points, trajectory, alpha=0.1, color="gray")
    # Plot the best trajectory in blue
    plt.plot(
        time_points,
        trajectories[best_traj_idx],
        "b-",
        linewidth=2,
        label="Least decline",
    )
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(
        f"Trajectories with {constant_features} \n(n={len(feature_combinations)})"
    )
    plt.legend()
    plt.grid(True)

    # Save the figure
    filename = f"{result_dir}/trajectories{constant_features}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved as: {filename}")
    # Generate train/val/test splits
    _, _, test_indices = generate_indices(len(dataset), train_size, val_size, seed)

    # Calculate metrics for best settings trajectory
    min_y = np.min(trajectories[best_traj_idx])
    avg_y = np.mean(trajectories[best_traj_idx])
    decline = trajectories[best_traj_idx][-1] - trajectories[best_traj_idx][0]

    print(f"Best Settings Trajectory Metrics:")
    print(f"Minimum y value: {min_y:.2f}")
    print(f"Average y value: {avg_y:.2f}")
    print(f"Decline in y: {decline:.2f}")

    return {
        "min_y": float(min_y),
        "avg_y": float(avg_y),
        "decline": float(decline),
        **summary_dict,
    }


# Now add the inference function
def tts_inference(
    litmodel,
    dataset,
    feature_combinations,
    y_normalizer=None,
    column_transformer=None,
    return_transition_points=False,
):
    n_points = len(dataset.ys[0])
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

    print("Forecasting trajectories...")
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


def run_inference(
    dataset,
    litmodel,
    column_transformer,
    y_normalizer,
    constant_features,
    result_dir,
    splits: list,
    n_subset=10,
    seed=0,
    packing_type="static",
    analyze_lipschitz=False,
    epsilon=0.01,
):
    # Get feature information
    feature_ranges = dataset.get_feature_ranges()
    feature_types = dataset.get_feature_types()

    # Get feature combinations
    if packing_type == "static":
        feature_combinations, summary_dict = get_feature_combinations(
            feature_ranges=feature_ranges,
            feature_types=feature_types,
            constant_features=constant_features,
            n_subset=n_subset,
            epsilon=epsilon,
        )
    elif packing_type == "covering":
        feature_combinations, summary_dict = get_feature_combinations_l2(
            feature_ranges=feature_ranges,
            feature_types=feature_types,
            constant_features=constant_features,
            n_subset=n_subset,
            epsilon=epsilon,
        )
    else:
        raise ValueError(f"Unknown packing type: {packing_type}")

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

    L_estimate = {}
    # not on every run
    if analyze_lipschitz:
        L_estimate = estimate_lipschitz_constant(
            dataset=dataset,
            feature_ranges=feature_ranges,
            feature_types=feature_types,
        )
        L_estimate = {"L_estimate_max": L_estimate[0], "L_estimate_mean": L_estimate[1]}

    evaluation_results = evaluate_model_predictions(
        dataset=dataset,
        feature_types=feature_types,
        feature_ranges=feature_ranges,
        trajectories=trajectories,
        time_points=time_points,
        feature_combinations=feature_combinations,
        train_size=splits[0],
        val_size=splits[1],
        seed=seed,
        result_dir=result_dir,
        constant_features=constant_features,
    )

    print("\nEvaluation Metrics:")
    print(f"Minimum y value: {evaluation_results['min_y']:.4f}")
    print(f"Average y value: {evaluation_results['avg_y']:.4f}")
    print(f"Decline in y: {evaluation_results['decline']:.4f}")

    return {**L_estimate, **summary_dict, **evaluation_results}


def get_best_feature_combination(
    dataset_name,
    model_name,
    root,
    result_dir,
    summary_filename,
    constant_features,
    n_subset,
    epsilon,
    packing_type,
    dataset_description_path,
    analyze_lipschitz=False,
    bayesian_optimization=False,
):
    #### inherited start ####
    create_benchmark_datasets_if_not_exist(
        dataset_description_path=dataset_description_path
    )
    results = find_results(
        dataset_name,
        model_name,
        results_dir=result_dir,
        summary_filename=summary_filename,
    )

    if len(results) == 0:
        print(f"No results found for {dataset_name} and {model_name}")
        print("Make sure you run your experiments from ../run_scripts")
    elif len(results) > 1:
        print("Multiple results found for the given dataset and model")
        print("We take the last one but it may produce unexpected results")

    timestamp = results[-1]

    spilt_sizes, seed = get_splits_from_summary(timestamp, result_dir, summary_filename)

    litmodel = load_model(
        timestamp, seed=661058651, benchmarks_folder=f"{root}/benchmarks"
    )
    try:
        dataset = load_dataset(
            dataset_name,
            dataset_description_path=f"{root}/dataset_descriptions",
            data_folder=f"{root}/data",
        )
    except:
        dataset = load_dataset(
            dataset_name,
            dataset_description_path=f"{root}/dataset_descriptions",
            # data_folder=f"{root}/data",
        )
    column_transformer = load_column_transformer(
        timestamp, benchmarks_dir=f"{root}/benchmarks"
    )
    y_normalizer = YNormalizer.load_from_benchmark(
        timestamp, model_name, benchmark_dir=f"{root}/benchmarks"
    )
    #### inherited end ####
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_dir = f"{root}/results/run_{timestamp}_time_{time}"
    os.makedirs(result_dir, exist_ok=True)

    summary_dict = run_inference(
        litmodel=litmodel,
        dataset=dataset,
        y_normalizer=y_normalizer,
        column_transformer=column_transformer,
        constant_features=constant_features,
        splits=spilt_sizes,
        seed=seed,
        n_subset=n_subset,
        result_dir=result_dir,
        packing_type=packing_type,
        analyze_lipschitz=analyze_lipschitz,
        epsilon=epsilon,
    )

    # save summary dict
    summary_dict["timestamp"] = timestamp
    summary_dict["model_name"] = model_name
    summary_dict["dataset_name"] = dataset_name
    summary_dict["seed"] = seed
    summary_dict["spilt_sizes"] = spilt_sizes

    with open(f"{result_dir}/summary.json", "w") as f:
        json.dump(
            summary_dict,
            f,
            indent=4,
        )
    return summary_dict


if __name__ == "__main__":
    pass
