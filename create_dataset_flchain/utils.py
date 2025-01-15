import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(df, features):
    n_features = len(features)
    n_rows = math.ceil(n_features / 2)  # Adjust rows dynamically
    plt.figure(figsize=(12, 5 * n_rows))  # Dynamic height adjustment

    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, 2, i)  # Dynamic grid adjustment
        plt.scatter(df["age"], df[feature], alpha=0.7, label="Data points")

        # Fit a line to the data
        try:
            m, b = np.polyfit(df["age"], df[feature], 1)  # Linear fit (y = mx + b)
            plt.plot(
                df["age"],
                m * df["age"] + b,
                color="red",
                label=f"y = {m:.2f}x + {b:.2f}",
            )
        except:
            print(f"Could not fit a line to {feature} vs Age")

        # Add title, labels, and legend
        plt.title(f"{feature} vs Age")
        plt.xlabel("Age")
        plt.ylabel(feature)
        plt.legend()

    plt.tight_layout()

    # Ensure the plots directory exists
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/scatter_plots_with_lines.png")
    plt.show()


if __name__ == "__main__":
    features = ["creatinine", "kappa", "lambda", "mgus", "sex"]
    df = pd.read_csv("experiments/data/flchain/flchain.csv")
    plot_scatter(df, features)
