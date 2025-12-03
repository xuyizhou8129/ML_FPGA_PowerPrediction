# src/main.py

"""
Pipeline:
1. Load CSV.
2. Train/test split.
3. Standardize features.
4. Run linear feature-selection boosting.
5. Run nonlinear (quadratic) boosting and compare.
"""

from typing import Tuple
from pathlib import Path

from preprocessing import (
    load_dataset,
    train_test_split,
    standardize,
    FEATURE_COLS,
)
from feature_selection import run_linear_feature_selection
from non_linear_corre import run_nonlinear_boosting
from plotting_utils import plot_mse_vs_features

# Script is in src/, so go up 1 level to project root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "design_space_power_subset.csv"


def main() -> Tuple[float, float]:
    # 1. Load raw data
    X, y, feature_names = load_dataset(str(DATA_PATH))
    print(f"Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")

    # 2. Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(
        f"Train samples: {X_train.shape[0]}, "
        f"Test samples: {X_test.shape[0]}"
    )

    # 3. Standardize features
    X_train_std, X_test_std, mean, std = standardize(X_train, X_test)
    print("Feature means (train):", mean)
    print("Feature stds  (train):", std)

    # 4. Linear boosting feature selection
    print("\n=== Linear Feature-Selection via Boosting ===")
    lin_model, lin_train_rmse, lin_test_rmse = (
        run_linear_feature_selection(
            X_train_std,
            y_train,
            X_test_std,
            y_test,
            feature_names=feature_names,
            # or a smaller number if you want sparsity
            max_rounds=len(FEATURE_COLS),
            verbose=True,
        )
    )

    # Plot MSE vs number of features for linear model
    print(
        "\n=== Plotting Linear Model: Average MSE vs Number of Features ==="
    )
    linear_plot_path = REPO_ROOT / "data" / "linear_avgmse_features.png"
    linear_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_mse_vs_features(lin_model, save_path=linear_plot_path)

    # 5. Nonlinear (quadratic) boosting
    print("\n=== Nonlinear (Quadratic) Boosting ===")
    nl_model, nl_train_rmse, nl_test_rmse = run_nonlinear_boosting(
        X_train_std,
        y_train,
        X_test_std,
        y_test,
        feature_names=feature_names,
        max_rounds=40,  # cap rounds because feature space explodes
        include_interactions=True,
        verbose=True,
    )

    # Plot MSE vs number of features for nonlinear model
    print(
        "\n=== Plotting Nonlinear Model: Average MSE vs Number of Features ==="
    )
    nonlinear_plot_path = REPO_ROOT / "data" / "nonlinear_avgmse_feature.png"
    nonlinear_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_mse_vs_features(nl_model, save_path=nonlinear_plot_path)

    print("\n=== Summary ===")
    print(
        f"Linear  RMSE: train={lin_train_rmse:.3f}, test={lin_test_rmse:.3f}"
    )
    print(
        f"Nonlin  RMSE: train={nl_train_rmse:.3f}, test={nl_test_rmse:.3f}"
    )

    return lin_test_rmse, nl_test_rmse


if __name__ == "__main__":
    main()
