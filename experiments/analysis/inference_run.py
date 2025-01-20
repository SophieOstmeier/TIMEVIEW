from experiments.analysis.inference import get_best_feature_combination
from datetime import datetime
import matplotlib.pyplot as plt
import json
import random
import numpy as np
from typing import Dict
from experiments.analysis.inference_utils import *

# Initial seeds for reproducibility
BASE_SEED = 42
N_ITERATIONS = 10
MUTE_PRINTS = True
EPSILON = 0.5
N_SUBSET = 5
packing_type = "covering"  # covering
# Setup code remains the same...
dataset_name = "flchain_1000"  # Change this line to switch datasets
dataset_name = "synthetic_tumor_wilkerson_1"  # Change this line to switch datasets
###############################################################################

model_name = "TTS"
root = "/dataNAS/people/sostm/TIMEVIEW/experiments"
result_dir = f"{root}/benchmarks"
summary_filename = f"{root}/benchmarks/summary.json"
dataset_description_path = f"{root}/dataset_descriptions"


if dataset_name == "flchain_1000":
    LIPSCHITZ_CONSTANT = 25.45
elif dataset_name == "synthetic_tumor_wilkerson_1":
    LIPSCHITZ_CONSTANT = 1.22
else:
    raise ValueError(
        "dataset_name must be either 'flchain_1000' or 'synthetic_tumor_wilkerson_1'"
    )
if packing_type == "covering":
    N_ITERATIONS = 1
    N_SUBSET = 2
# Define feature ranges and possible values
DATASET_FEATURES = {
    "flchain_1000": {
        "sex": ["M", "F"],
        "creatinine": (0.40, 2.00),
        "kappa": (0.01, 5.00),
        "lambda": (0.04, 5.00),
        "flc.grp": list(range(1, 11)),  # 1 to 10
        "mgus": ["no", "yes"],
    },
    "synthetic_tumor_wilkerson_1": {
        "age": (20, 80),
        "weight": (40, 100),
        "initial_tumor_volume": (0.1, 0.5),
        "dosage": (0.0, 1.0),
    },
}


def run_single_iteration(
    iteration_seed: int,
    dataset_name: str,
    dataset_features,
    epsilon: float,
    n_subset_iter,
    mute,
) -> Dict:
    # Set different seeds for each iteration
    random.seed(iteration_seed)
    np.random.seed(iteration_seed)

    # Generate random feature states based on dataset
    base_features = get_base_features(dataset_name, dataset_features)

    feature_order = list(base_features.keys())
    random.shuffle(feature_order)

    feature_states = []
    for i in range(1, len(feature_order) + 1):
        selected_featurnes = feature_order[:i]
        state = {k: base_features[k] for k in selected_featurnes}
        feature_states.append(state)

    feature_states = list(reversed(feature_states))

    convergence_data = {
        "dimensions": [],
        "combinations_needed": [],
        "min_y_achieved": [],
    }

    with mute_prints(should_mute=mute):  # Mute all prints within this function
        for features in feature_states:
            state_results = []
            for n_subset in range(1, n_subset_iter):
                result = get_best_feature_combination(
                    dataset_name=dataset_name,
                    constant_features=features,
                    model_name=model_name,
                    result_dir=result_dir,
                    summary_filename=summary_filename,
                    dataset_description_path=dataset_description_path,
                    n_subset=n_subset,
                    epsilon=epsilon,
                    packing_type=packing_type,
                    root=root,
                )
                result["n_subset"] = n_subset
                state_results.append(result)

            # Find convergence point
            converged_result = None
            for i, current in enumerate(state_results):
                if current["best_varying_features"].keys() == 0:
                    break
                if all(
                    np.sqrt(
                        np.sum(
                            (
                                normalize_feature_to_np(
                                    current["best_varying_features"]
                                )
                                - normalize_feature_to_np(
                                    future["best_varying_features"]
                                )
                            )
                            ** 2
                        )
                    )
                    <= epsilon
                    for future in state_results[i + 1 :]
                ):
                    converged_result = current
                    break

            if converged_result is None:
                # If no convergence found, use the last result
                Warning(
                    f"No convergence found for feature set {features}. Using the last result."
                )
                converged_result = state_results[-1]

            # Store convergence data for this feature set
            convergence_data["dimensions"].append(len(features))
            convergence_data["combinations_needed"].append(
                converged_result["n_combinations"]
            )
            convergence_data["min_y_achieved"].append(converged_result["min_y"])

        return convergence_data


if __name__ == "__main__":
    if dataset_name not in DATASET_FEATURES:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Supported datasets are: {list(DATASET_FEATURES.keys())}"
        )

    # Run iterations
    all_iterations_data = []

    try:
        from tqdm import tqdm

        iterator = tqdm(range(N_ITERATIONS), desc="Running iterations")
    except ImportError:
        print(f"Running {N_ITERATIONS} iterations...")
        iterator = range(N_ITERATIONS)

    for i in iterator:
        iteration_seed = BASE_SEED + i
        if not isinstance(iterator, tqdm):
            print(
                f"\rIteration {i+1}/{N_ITERATIONS} (seed: {iteration_seed})",
                end="",
                flush=True,
            )
        iteration_data = run_single_iteration(
            iteration_seed,
            dataset_name,
            DATASET_FEATURES,
            EPSILON,
            N_SUBSET,
            MUTE_PRINTS,
        )
        all_iterations_data.append(iteration_data)

    if not isinstance(iterator, tqdm):
        print()

    # Create both plots
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # First plot (original convergence analysis)
    dimensions = sorted(
        set(dim for data in all_iterations_data for dim in data["dimensions"]),
        reverse=True,
    )
    combinations_by_dimension = {dim: [] for dim in dimensions}

    for data in all_iterations_data:
        for dim, combs in zip(data["dimensions"], data["combinations_needed"]):
            combinations_by_dimension[dim].append(combs)

    means = []
    std_errs = []
    for dim in dimensions:
        values = combinations_by_dimension[dim]
        means.append(np.mean(values))
        std_errs.append(np.std(values) / np.sqrt(len(values)))

    x_values = list(range(1, len(dimensions) + 1))
    # compute theorectical results
    theoretical_results = []

    for dim in x_values:
        theoretical_results.append((LIPSCHITZ_CONSTANT / (2 * EPSILON)) ** dim)

    ax1.errorbar(
        x_values,
        means,
        yerr=std_errs,
        fmt="o-",
        capsize=5,
        capthick=1,
        elinewidth=1,
        markersize=8,
        label="Mean with 95% CI",
    )
    print("theoretical_results", theoretical_results)
    # Plot theoretical results with just a line
    ax1.plot(x_values, theoretical_results, "o-", markersize=8, label="Upper bound")

    ax1.set_xlabel("Number of varying Dimensions")
    ax1.set_ylabel("Combinations Needed for Convergence")
    ax1.set_title(f"Convergence Analysis - {dataset_name}")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.set_xticks(dimensions)
    max_mean_plus_err = max(m + e for m, e in zip(means, std_errs))
    ax1.set_ylim(0, max_mean_plus_err * 1.1)
    ax1.legend()

    # Second plot (N_c vs y_min for max dimensions)
    max_dim = max(dimensions)  # Get the maximum number of dimensions
    y_min_by_nc = {}  # Dictionary to store y_min values for each N_c

    for data in all_iterations_data:
        # Find the entry with maximum dimensions
        max_dim_index = data["dimensions"].index(max_dim)
        y_min = data["min_y_achieved"][max_dim_index]
        n_combinations = data["combinations_needed"][max_dim_index]

        if n_combinations not in y_min_by_nc:
            y_min_by_nc[n_combinations] = []
        y_min_by_nc[n_combinations].append(y_min)

    # Calculate means and standard errors for y_min values
    nc_values = sorted(y_min_by_nc.keys())
    y_min_means = []
    y_min_std_errs = []

    for nc in nc_values:
        values = y_min_by_nc[nc]
        y_min_means.append(np.mean(values))
        y_min_std_errs.append(np.std(values) / np.sqrt(len(values)))

    plt.tight_layout()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        f"convergence_analysis_{dataset_name}_{N_ITERATIONS}iter_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )

    results = {
        "dimensions": dimensions,
        "means": means,
        "standard_errors": std_errs,
        "raw_data": all_iterations_data,
        "nc_vs_ymin": {
            "nc_values": nc_values,
            "y_min_means": y_min_means,
            "y_min_std_errs": y_min_std_errs,
        },
    }

    with open(
        f"convergence_analysis_{dataset_name}_{N_ITERATIONS}iter_{timestamp}.json", "w"
    ) as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved as:")
    print(
        f"- convergence_analysis_{dataset_name}_{N_ITERATIONS}iter_{timestamp}_{packing_type}.png"
    )
    print(
        f"- convergence_analysis_{dataset_name}_{N_ITERATIONS}iter_{timestamp}_{packing_type}.json"
    )
