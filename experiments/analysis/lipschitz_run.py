import matplotlib.pyplot as plt
import numpy as np
from experiments.analysis.inference import get_best_feature_combination
import ipdb

# Your parameters
# dataset_name = "flchain_1000"
dataset_name = "synthetic_tumor_wilkerson_1"
model_name = "TTS"
root = "/dataNAS/people/sostm/TIMEVIEW/experiments"
result_dir = f"{root}/benchmarks"
summary_filename = f"{root}/benchmarks/summary.json"
dataset_description_path = f"{root}/dataset_descriptions"
packing_type = "static"
# Store results
combinations = []
y_mins = []

# Run analysis for each n_subset
for i in range(1, 5):
    result = get_best_feature_combination(
        dataset_name=dataset_name,
        model_name=model_name,
        root=root,
        result_dir=result_dir,
        summary_filename=summary_filename,
        n_subset=i,
        packing_type=packing_type,
        dataset_description_path=dataset_description_path,
        analyze_lipschitz=True,
        constant_features={},
    )

    combinations.append(result["n_combinations"])
    y_mins.append(result["min_y"])  # Assuming y_min is in the result

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(combinations, y_mins, "o-", linewidth=2, markersize=8)

# Customize the plot
plt.xlabel("Number of Feature Combinations")
plt.ylabel("Minimum Y Value")
plt.title(f"Feature Combination Analysis for {dataset_name}\nModel: {model_name}")

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Ensure x-axis shows whole numbers
plt.xticks(combinations)

# Add value labels on points
for x, y in zip(combinations, y_mins):
    plt.annotate(
        f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )

plt.tight_layout()

# Save the plot
plt.savefig(f"{dataset_name}_feature_combination_analysis.png")
plt.show()
