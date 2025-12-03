# src/plotting_utils.py

from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt


def plot_mse_vs_features(
    model,
    save_path: Optional[Path] = None,
) -> None:
    train_mse = model.train_mse_history_
    test_mse = model.test_mse_history_

    num_features = list(range(len(train_mse)))

    plt.figure(figsize=(10, 6))
    plt.plot(
        num_features, train_mse, 'o-', label='Train MSE',
        linewidth=2, markersize=6
    )
    if len(test_mse) > 0:
        plt.plot(
            num_features, test_mse, 's-', label='Test MSE',
            linewidth=2, markersize=6
        )

    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title(
        'MSE vs Number of Features', fontsize=14, fontweight='bold'
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show(block=False)
