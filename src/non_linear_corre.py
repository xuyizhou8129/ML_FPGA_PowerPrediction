# src/non-linear_corre.py

from typing import List, Tuple, Optional
import numpy as np

from feature_selection import FeatureBoostingRegressor


def make_quadratic_features(
    X: np.ndarray, feature_names: List[str], include_interactions: bool = False
) -> Tuple[np.ndarray, List[str]]:

    N, D = X.shape

    # Start with original features
    feats = [X]
    names = feature_names.copy()

    # Squared terms
    X_sq = X ** 2
    feats.append(X_sq)
    names += [f"{name}^2" for name in feature_names]

    # Interaction terms
    if include_interactions:
        inter_list = []
        inter_names = []
        for i in range(D):
            for j in range(i + 1, D):
                inter_list.append(X[:, i] * X[:, j])
                inter_names.append(f"{feature_names[i]}*{feature_names[j]}")
        if inter_list:
            feats.append(np.column_stack(inter_list))
            names += inter_names

    X_expanded = np.column_stack(feats)
    return X_expanded, names


def run_nonlinear_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    max_rounds: Optional[int] = None,
    include_interactions: bool = False,
    verbose: bool = True,
) -> Tuple[FeatureBoostingRegressor, float, float]:

    X_train_nl, names_nl = make_quadratic_features(
        X_train, feature_names, include_interactions=include_interactions
    )
    X_test_nl, _ = make_quadratic_features(
        X_test, feature_names, include_interactions=include_interactions
    )

    model = FeatureBoostingRegressor(
        max_rounds=max_rounds, tol=1e-6, verbose=verbose
    )
    model.fit(X_train_nl, y_train, X_test_nl, y_test)

    train_rmse = model.score_rmse(X_train_nl, y_train)
    test_rmse = model.score_rmse(X_test_nl, y_test)

    # Calculate tolerance percentages
    train_tol_5 = model.score_tolerance_percentage(X_train_nl, y_train, 0.05)
    train_tol_10 = model.score_tolerance_percentage(X_train_nl, y_train, 0.10)
    train_tol_15 = model.score_tolerance_percentage(X_train_nl, y_train, 0.15)
    test_tol_5 = model.score_tolerance_percentage(X_test_nl, y_test, 0.05)
    test_tol_10 = model.score_tolerance_percentage(X_test_nl, y_test, 0.10)
    test_tol_15 = model.score_tolerance_percentage(X_test_nl, y_test, 0.15)

    if verbose:
        print("\n=== Nonlinear Model Coefficients ===")
        print(f"Intercept (w₀): {model.intercept_:.6f}")
        print(f"\nAll feature coefficients (total: {len(names_nl)} features):")
        for idx, name in enumerate(names_nl):
            coef = model.coefs_[idx]
            selected = "✓" if idx in model.selected_features_ else " "
            print(f"  [{selected}] {idx:3d}: {name:50s} w = {coef:10.6f}")

        print("\nSelected features (in order):")
        for idx in model.selected_features_:
            print(f"  {idx}: {names_nl[idx]} (w={model.coefs_[idx]:.6f})")

        print(
            f"\nNonlinear boosting RMSE - train: {train_rmse:.3f}, "
            f"test: {test_rmse:.3f}"
        )
        print("\nTolerance Accuracy:")
        print(f"  Train - Within 5%:  {train_tol_5:.2f}%, "
              f"Within 10%: {train_tol_10:.2f}%, "
              f"Within 15%: {train_tol_15:.2f}%")
        print(f"  Test  - Within 5%:  {test_tol_5:.2f}%, "
              f"Within 10%: {test_tol_10:.2f}%, "
              f"Within 15%: {test_tol_15:.2f}%")

    return model, train_rmse, test_rmse
