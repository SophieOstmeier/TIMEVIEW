import numpy as np
from itertools import product


def get_feature_combinations_l2(
    feature_types, feature_ranges, constant_features, epsilon=1e-2, n_subset=None
):
    # Separate varying and constant features
    varying_features = {
        k: v for k, v in feature_ranges.items() if k not in constant_features
    }
    print(f"Varying features: {varying_features}")

    # Count total varying dimensions
    n_dims = len(varying_features)

    # Function to normalize any value to [0,1]
    def normalize_value(value, feature, range_val):
        if feature_types[feature] == "continuous":
            return (value - range_val[0]) / (range_val[1] - range_val[0])
        else:  # discrete/binary
            if isinstance(range_val[0], str):
                return 1 if value == range_val[1] else 0
            return value

    def l2_distance(point1, point2):
        squared_diffs = 0
        for feat in varying_features:
            val1 = normalize_value(point1[feat], feat, varying_features[feat])
            val2 = normalize_value(point2[feat], feat, varying_features[feat])
            squared_diffs += (val1 - val2) ** 2
        return np.sqrt(squared_diffs)

    # Calculate number of points needed based on total dimensions
    if n_dims > 0:
        n_points = max(int(1 / (epsilon / np.sqrt(n_dims))), 2)
    else:
        n_points = 1

    # Create feature points
    feature_points = {}
    for feature, range_val in varying_features.items():
        if feature_types[feature] == "continuous":
            feature_points[feature] = np.linspace(range_val[0], range_val[1], n_points)
        else:  # categorical/binary
            feature_points[feature] = range_val

    # Generate candidate combinations
    feature_names = list(feature_points.keys())
    candidates = product(*[feature_points[f] for f in feature_names])
    candidates = [dict(zip(feature_names, combo)) for combo in candidates]

    # Select points based on L2 coverage
    feature_combinations = []
    for candidate in candidates:
        if not feature_combinations or all(
            l2_distance(candidate, existing) > epsilon
            for existing in feature_combinations
        ):
            setting = constant_features.copy()
            setting.update(candidate)
            feature_combinations.append(setting)

    # Print summary
    print(f"\nAnalysis Summary:")
    print(f"Number of combinations: {len(feature_combinations)}")
    print(f"Epsilon (L2 coverage): {epsilon}")
    print(f"Total varying dimensions: {n_dims}")
    print(f"Points per continuous dimension: {n_points}")

    print("\nConstant features:")
    for k, v in constant_features.items():
        print(f"  {k}: {v}")

    print("\nVarying features:")
    for k in varying_features.keys():
        if feature_types[k] == "continuous":
            print(
                f"  {k}: {n_points} points from {varying_features[k][0]} to {varying_features[k][1]}"
            )
        else:
            print(f"  {k}: {len(varying_features[k])} categories")

    # Example settings
    print("\nExample settings:")
    for i, setting in enumerate(feature_combinations[:3]):
        print(f"\nSetting {i+1}:")
        for k, v in setting.items():
            print(f"  {k}: {v}")
    if len(feature_combinations) > 3:
        for i, setting in enumerate(feature_combinations[-3:]):
            print(f"\nSetting {len(feature_combinations)-2+i}:")
            for k, v in setting.items():
                print(f"  {k}: {v}")

    summary_dict = {
        "n_combinations": len(feature_combinations),
        "constant_features": constant_features,
        "varying_features": varying_features,
        "feature_ranges": feature_ranges,
        "feature_types": feature_types,
        "epsilon": epsilon,
        "n_dims": n_dims,
        "points_per_dim": n_points,
    }

    return feature_combinations, summary_dict


def get_feature_combinations(
    feature_types,
    feature_ranges,
    constant_features,
    n_subset=100,
    epsilon=1e-3,
):
    # Separate varying and constant features
    varying_features = {
        k: v for k, v in feature_ranges.items() if k not in constant_features
    }
    print(f"Varying features: {varying_features}")

    if len(constant_features.keys()) == 0:
        assert len(varying_features) == len(feature_ranges), "All features are varying"
    else:
        assert len(varying_features) == len(feature_ranges) - len(
            constant_features
        ), "Some features are missing"

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

    # # Example settings
    # print("\nExample settings:")

    # for i, setting in enumerate(feature_combinations[:3]):
    #     print(f"\nSetting {i+1}:")
    #     for k, v in setting.items():
    #         print(f"  {k}: {v}")
    # for i, setting in enumerate(feature_combinations[:-3]):
    #     print(f"\nSetting {i+1}:")
    #     for k, v in setting.items():
    #         print(f"  {k}: {v}")

    summary_dict = {
        "n_combinations": len(feature_combinations),
        "constant_features": constant_features,
        "varying_features": varying_features,
        "feature_ranges": feature_ranges,
        "feature_types": feature_types,
    }

    return feature_combinations, summary_dict
