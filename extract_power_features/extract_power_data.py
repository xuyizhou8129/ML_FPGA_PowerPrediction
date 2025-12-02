import pandas as pd
from pathlib import Path

# ==== CONFIGURATION ====
# Script is in ML_FPGA_PowerPrediction/extract_power_features/,
# so go up 2 levels to project root
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = (REPO_ROOT / "ML4Accel_Dataset" / "ML4Accel-Dataset" /
       "hls_experiments" / "data" / "design_space_v2_transformed.csv")
DST = (REPO_ROOT / "ML_FPGA_PowerPrediction" / "data" /
       "design_space_power_subset.csv")

COLUMNS = [
    "hls_synth__latency_best_cycles",
    "hls_synth__latency_average_cycles",
    "hls_synth__latency_worst_cycles",

    "hls_synth__resources_lut_used",
    "hls_synth__resources_ff_used",
    "hls_synth__resources_dsp_used",
    "hls_synth__resources_bram_used",
    "hls_synth__resources_uram_used",

    "impl__power__total_power",
    "impl__power__dynamic_power",
    "impl__power__static_power",
]

print(f"Loading: {SRC}")
df = pd.read_csv(SRC)

# Sanity check
missing = [c for c in COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep only desired columns
df = df[COLUMNS]

# Drop rows with missing values
df = df.dropna()

# Save output
DST.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DST, index=False)

print(f"Saved new dataset to: {DST}")
print(f"Final shape: {df.shape}")
