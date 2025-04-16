import pandas as pd
from lifelines.utils import concordance_index

# Load results and clinical data
results = pd.read_csv("results.csv")
clinical = pd.read_csv("data/clinical.csv")

# Merge on sample_id
merged = pd.merge(results, clinical, on="sample_id")

# Drop missing values
merged = merged.dropna(subset=["OS_time", "OS_status", "predicted_risk"])

# Calculate C-index
c_index = concordance_index(
    event_times=merged["OS_time"],
    predicted_scores=-merged["predicted_risk"],  # negate because higher risk → shorter survival
    event_observed=merged["OS_status"]
)

print(f"C-index: {c_index:.4f}")
