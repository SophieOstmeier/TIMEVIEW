import torch
import gpytorch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(train_x.shape[1]), num_tasks=train_y.shape[1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]),
            num_tasks=train_y.shape[1],
            rank=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GaussianProcessGenerator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.sex_encoder = LabelEncoder()
        self.mgus_encoder = LabelEncoder()
        self.model = None
        self.likelihood = None

    def prepare_data(self, df):
        # Encode categorical variables
        df["sex_encoded"] = self.sex_encoder.fit_transform(df["sex"])
        df["mgus_encoded"] = self.mgus_encoder.fit_transform(df["mgus"])

        # Prepare features
        X = df[["t", "y", "sex_encoded"]].copy()
        X["age_plus_t"] = (df["age"] * 365 + df["t"]) // 365

        # Prepare targets
        y = df[["creatinine", "flc.grp", "kappa", "lambda", "mgus_encoded"]].copy()

        # Scale the data
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)

        return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(
            y_scaled, dtype=torch.float32
        )

    def train(self, train_df, training_iterations=10):
        X, y = self.prepare_data(train_df)
        X, y = X.to(self.device), y.to(self.device)

        # Initialize model and likelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=y.shape[1]
        )
        self.model = MultitaskGPModel(X, y, self.likelihood)

        self.model.to(self.device)
        self.likelihood.to(self.device)

        # Train the model
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(
                    f"Iteration {i+1}/{training_iterations} - Loss: {loss.item():.3f}"
                )

    def generate(self, new_features_df):
        self.model.eval()
        self.likelihood.eval()

        # Prepare new features
        X_new = new_features_df[["t", "y", "sex"]].copy()
        X_new["sex_encoded"] = self.sex_encoder.transform(X_new["sex"])
        X_new["age_plus_t"] = new_features_df["age"] + new_features_df["t"]
        X_new = X_new[["t", "y", "sex_encoded", "age_plus_t"]]

        # Process in batches to avoid memory issues
        batch_size = 100
        generated_data = []

        for start_idx in range(0, len(X_new), batch_size):
            end_idx = min(start_idx + batch_size, len(X_new))
            batch = X_new.iloc[start_idx:end_idx]

            # Scale and convert to tensor
            batch_scaled = self.feature_scaler.transform(batch)
            batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(
                self.device
            )

            # Generate one sample per input
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(
                100
            ):
                prediction = self.model(batch_tensor)
                # Sample shape will be [batch_size, num_features]
                samples = prediction.sample().cpu().numpy()

            # Inverse transform the scaled values
            samples_unscaled = self.target_scaler.inverse_transform(samples)

            # Create DataFrame for the batch
            batch_df = pd.DataFrame(
                samples_unscaled,
                columns=["creatinine", "flc.grp", "kappa", "lambda", "mgus_encoded"],
            )

            # Convert mgus back to categorical
            batch_df["mgus"] = self.mgus_encoder.inverse_transform(
                np.round(batch_df["mgus_encoded"]).astype(int)
            )
            batch_df = batch_df.drop("mgus_encoded", axis=1)

            generated_data.append(batch_df)

        # Combine all batches
        return pd.concat(generated_data, ignore_index=True)


# Example usage
if __name__ == "__main__":
    # Load training data
    # data = pd.read_csv("experiments/data/flchain/flchain.csv")
    # training_data = data[data["t"] == 0.0].sample(frac=0.1)

    # # Initialize generator
    # print("Initializing generator...")
    # generator = MultiGPGenerator(training_data)

    # Load your training data
    data = pd.read_csv("experiments/data/flchain/flchain.csv")
    training_data = data[data["t"] == 0.0].sample(frac=0.1)

    # Initialize and train the generator
    generator = GaussianProcessGenerator()
    generator.train(training_data)

    model_path = "models/flchain_generator.pth"
    print(f"Saving model to {model_path}...")
    torch.save(generator.model.state_dict(), model_path)
    print("Model saved successfully!")
    # Create example new features
    new_features = data[data["t"] != 0.0]
    # Remove target columns from new features
    new_features = new_features.drop(
        columns=["creatinine", "flc.grp", "kappa", "lambda", "mgus"]
    )

    # Generate samples
    print("Generating samples...")
    generated_samples = generator.generate(new_features)
    print("Generated samples:")
    print(generated_samples)
    # save
    generated_samples.to_csv(
        "experiments/data/flchain/generated_samples.csv", index=False
    )
