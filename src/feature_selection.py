# src/feature_selection.py
from typing import List, Optional, Tuple
import numpy as np


class FeatureBoostingRegressor:

    def __init__(
        self,
        max_rounds: Optional[int] = None,
        tol: float = 1e-6,
        verbose: bool = False,
    ):

        self.max_rounds = max_rounds
        self.tol = tol
        self.verbose = verbose

        self.intercept_: float = 0.0
        self.coefs_: Optional[np.ndarray] = None
        self.selected_features_: List[int] = []
        self.train_mse_history_: List[float] = []
        self.test_mse_history_: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> "FeatureBoostingRegressor":
        N, D = X.shape
        max_rounds = self.max_rounds or D

        # Initialize with constant predictor w0 = mean(y)
        self.intercept_ = float(y.mean())
        residual = y - self.intercept_

        self.coefs_ = np.zeros(D, dtype=float)
        self.selected_features_ = []
        self.train_mse_history_ = []
        self.test_mse_history_ = []
        selected_mask = np.zeros(D, dtype=bool)

        prev_mse = np.mean(residual**2)

        # Record initial MSE (0 features, just intercept)
        self.train_mse_history_.append(prev_mse)
        if X_test is not None and y_test is not None:
            test_pred = np.full_like(y_test, self.intercept_)
            test_mse = np.mean((y_test - test_pred) ** 2)
            self.test_mse_history_.append(test_mse)

        for k in range(max_rounds):
            best_feat = None
            best_w = 0.0
            best_mse = prev_mse

            for j in range(D):
                if selected_mask[j]:
                    continue

                xj = X[:, j]
                denom = float(xj @ xj)
                if denom == 0.0:
                    continue

                wj = float(xj @ residual / denom)
                preds = wj * xj
                mse = np.mean((residual - preds) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_feat = j
                    best_w = wj

            if best_feat is None:
                if self.verbose:
                    print(f"[Round {k}] No feature improves MSE; stopping.")
                break

            improvement = prev_mse - best_mse
            if improvement < self.tol:
                if self.verbose:
                    msg = (f"[Round {k}] Improvement {improvement:.3e} < "
                           f"tol {self.tol:.3e}; stopping.")
                    print(msg)
                break

            # Commit the best feature and update residuals
            selected_mask[best_feat] = True
            self.selected_features_.append(best_feat)
            self.coefs_[best_feat] = best_w

            x_best = X[:, best_feat]
            residual = residual - best_w * x_best
            prev_mse = best_mse

            # Record MSE after adding this feature
            self.train_mse_history_.append(best_mse)
            if X_test is not None and y_test is not None:
                test_pred = self.predict(X_test)
                test_mse = np.mean((y_test - test_pred) ** 2)
                self.test_mse_history_.append(test_mse)

            if self.verbose:
                msg = (f"[Round {k}] Added feature {best_feat} with "
                       f"weight {best_w:.6f}, MSE={best_mse:.4f}, "
                       f"improvement={improvement:.4f}")
                print(msg)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefs_ is None:
            raise RuntimeError("Model not fitted yet.")

        return self.intercept_ + X @ self.coefs_

    def score_rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Root Mean Squared Error (RMSE) on given data.
        """
        y_pred = self.predict(X)
        return float(np.sqrt(np.mean((y_pred - y) ** 2)))

    def score_tolerance_percentage(
        self, X: np.ndarray, y: np.ndarray, tolerance: float
    ) -> float:
        y_pred = self.predict(X)
        # Avoid division by zero for actual values near zero
        relative_error = np.abs(y_pred - y) / np.maximum(np.abs(y), 1e-10)
        within_tolerance = np.mean(relative_error <= tolerance) * 100.0
        return float(within_tolerance)


def run_linear_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    max_rounds: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[FeatureBoostingRegressor, float, float]:
    model = FeatureBoostingRegressor(
        max_rounds=max_rounds, tol=1e-6, verbose=verbose
    )
    model.fit(X_train, y_train, X_test, y_test)

    train_rmse = model.score_rmse(X_train, y_train)
    test_rmse = model.score_rmse(X_test, y_test)

    # Calculate tolerance percentages
    train_tol_5 = model.score_tolerance_percentage(X_train, y_train, 0.05)
    train_tol_10 = model.score_tolerance_percentage(X_train, y_train, 0.10)
    train_tol_15 = model.score_tolerance_percentage(X_train, y_train, 0.15)
    test_tol_5 = model.score_tolerance_percentage(X_test, y_test, 0.05)
    test_tol_10 = model.score_tolerance_percentage(X_test, y_test, 0.10)
    test_tol_15 = model.score_tolerance_percentage(X_test, y_test, 0.15)

    if verbose:
        print("\n=== Linear Model Coefficients ===")
        print(f"Intercept (w₀): {model.intercept_:.6f}")
        print("\nAll feature coefficients:")
        for idx, name in enumerate(feature_names):
            coef = model.coefs_[idx]
            selected = "✓" if idx in model.selected_features_ else " "
            print(f"  [{selected}] {idx}: {name:40s} w = {coef:10.6f}")

        print("\nSelected features (in order):")
        for idx in model.selected_features_:
            name = feature_names[idx]
            coef_val = model.coefs_[idx]
            print(f"  {idx}: {name} (w={coef_val:.6f})")

        msg = (f"\nLinear boosting RMSE - train: {train_rmse:.3f}, "
               f"test: {test_rmse:.3f}")
        print(msg)
        print("\nTolerance Accuracy:")
        print(f"  Train - Within 5%:  {train_tol_5:.2f}%, "
              f"Within 10%: {train_tol_10:.2f}%, "
              f"Within 15%: {train_tol_15:.2f}%")
        print(f"  Test  - Within 5%:  {test_tol_5:.2f}%, "
              f"Within 10%: {test_tol_10:.2f}%, "
              f"Within 15%: {test_tol_15:.2f}%")

    return model, train_rmse, test_rmse
