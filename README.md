# ML FPGA Power Prediction

Machine learning project for predicting FPGA power consumption using High-Level Synthesis (HLS) synthesis metrics. This project implements greedy forward feature selection with boosting to build linear and nonlinear regression models for power prediction.

## Overview

This project predicts total power consumption of FPGA implementations based on HLS synthesis characteristics. It uses a greedy boosting algorithm to sequentially select features that minimize mean squared error (MSE), comparing both linear and nonlinear (quadratic) models.

## Features

- **Greedy Forward Feature Selection**: Implements a boosting-based algorithm that sequentially selects features to minimize training MSE
- **Linear and Nonlinear Models**: Compares linear regression with quadratic feature expansion (including interaction terms)
- **Comprehensive Evaluation**: Tracks training and test MSE as features are added, providing insights into model performance and overfitting
- **Visualization**: Generates plots showing how average MSE changes with the number of selected features

## Dataset

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

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install numpy pandas matplotlib
```

## Usage

### Running the Main Pipeline

Execute the main script to run the complete pipeline:

```bash
cd src
python main.py
```

This will:
1. Load the dataset from `data/design_space_power_subset.csv`
2. Split data into training (80%) and test (20%) sets
3. Standardize features (zero mean, unit variance)
4. Run linear feature selection with greedy boosting
5. Generate and save plot: `data/linear_avgmse_features.png`
6. Run nonlinear (quadratic) feature selection
7. Generate and save plot: `data/nonlinear_avgmse_feature.png`
8. Print summary statistics comparing both models

### Algorithm Details

#### Greedy Forward Feature Selection

The algorithm uses a boosting approach:

1. **Initialization**: Start with a constant predictor (mean of target values)
2. **Feature Selection**: At each round, evaluate all remaining features and select the one that minimizes residual MSE
3. **Residual Update**: Update residuals by subtracting the selected feature's contribution
4. **Stopping Criteria**: Stop when no feature improves MSE or improvement is below tolerance

The algorithm is greedy because it makes locally optimal choices at each step without considering future feature interactions.

#### Model Types

- **Linear Model**: Uses original features with linear coefficients
- **Nonlinear Model**: Expands feature space to include:
  - Squared terms (x² for each feature)
  - Interaction terms (xᵢ × xⱼ for all feature pairs)

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

## Report

For detailed analysis and results, see the project report:
[Overleaf Project](https://www.overleaf.com/project/691162af603b1a669403d144)

## License

This project is part of a course assignment (EE 375, Fall 2025).

## Author

Developed for the Final Project in EE 375 (Fall 2025).
