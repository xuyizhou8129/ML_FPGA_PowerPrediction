# Interpretable FPGA Power Estimation Using Greedy Boosting and Quadratic Feature Expansion

**📄 [Full Project Report](https://www.overleaf.com/project/691162af603b1a669403d144)**

Machine learning project for predicting FPGA power consumption using High-Level Synthesis (HLS) synthesis metrics. This project implements greedy forward feature selection with boosting to build linear and nonlinear regression models for power prediction.

## Overview

This project predicts total power consumption of FPGA implementations based on HLS synthesis characteristics. It uses a greedy boosting algorithm to sequentially select features that minimize mean squared error (MSE), comparing both linear and nonlinear (quadratic) models.

## Features

- **Greedy Forward Feature Selection**: Implements a boosting-based algorithm that sequentially selects features to minimize training MSE
- **Linear and Nonlinear Models**: Compares linear regression with quadratic feature expansion (including interaction terms)
- **Comprehensive Evaluation**: Tracks training and test MSE as features are added, providing insights into model performance and overfitting
- **Visualization**: Generates plots showing how average MSE changes with the number of selected features

## Dataset

**Data Source**: The dataset used in this project is derived from the [ML4Accel-Dataset](https://github.com/UT-LCA/ML4Accel-Dataset), a well-curated open-source dataset for ML-guided chip design developed at UT Austin.

The dataset contains HLS synthesis metrics and power measurements from FPGA implementations:

**Input Features:**
- `hls_synth__latency_best_cycles`: Best-case latency in cycles
- `hls_synth__latency_average_cycles`: Average latency in cycles
- `hls_synth__latency_worst_cycles`: Worst-case latency in cycles
- `hls_synth__resources_lut_used`: LUT (Look-Up Table) resource usage
- `hls_synth__resources_ff_used`: Flip-flop resource usage
- `hls_synth__resources_dsp_used`: DSP (Digital Signal Processing) resource usage
- `hls_synth__resources_bram_used`: BRAM (Block RAM) resource usage
- `hls_synth__resources_uram_used`: UltraRAM resource usage

**Target Variable:**
- `impl__power__total_power`: Total power consumption

## Project Structure

```
ML_FPGA_PowerPrediction/
├── data/
│   ├── design_space_power_subset.csv    # Main dataset
│   ├── linear_avgmse_features.png       # Linear model MSE plot
│   └── nonlinear_avgmse_feature.png     # Nonlinear model MSE plot
├── src/
│   ├── main.py                          # Main entry point
│   ├── preprocessing.py                 # Data loading and preprocessing
│   ├── feature_selection.py             # Greedy boosting algorithm
│   ├── non_linear_corre.py             # Nonlinear feature expansion
│   └── plotting_utils.py                # Visualization utilities
├── extract_power_features/
│   └── extract_power_data.py           # Data extraction script
└── README.md
```

## Output

The script generates:

1. **Console Output**: 
   - Dataset statistics
   - Feature selection progress
   - Model coefficients
   - RMSE metrics for train and test sets
   - Tolerance accuracy (percentage of predictions within 5%, 10%, 15% of actual values)

2. **Visualization Plots**:
   - `data/linear_avgmse_features.png`: Average MSE vs number of features for linear model
   - `data/nonlinear_avgmse_feature.png`: Average MSE vs number of features for nonlinear model

   Both plots show:
   - Training MSE (blue line with circles)
   - Test MSE (orange line with squares)

## Key Insights

- **Training MSE** always decreases (or stays constant) as features are added (greedy algorithm property)
- **Test MSE** may increase due to overfitting when features don't generalize well
- The plots help identify the optimal number of features that balance model complexity and generalization

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation and CSV reading
- `matplotlib`: Plotting and visualization

