import pandas as pd

# Load prediction results and clinical data
results = pd.read_csv("results.csv")
clinical = pd.read_csv("data/clinical.csv")

# Standardize sample_id format
results["sample_id"] = results["sample_id"].str[:15]
clinical["sample_id"] = clinical["sample_id"].str[:15]

# Merge data
df = pd.merge(results, clinical, on="sample_id", how="inner")
print("Merged shape:", df.shape)

# Drop missing values
df = df.dropna(subset=["predicted_risk", "OS_time", "OS_status"])
df = df.rename(columns={"OS_time": "duration", "OS_status": "event"})

# Check if there are valid samples
print("Valid sample count:", len(df))
if len(df) == 0:
    raise ValueError("No valid samples after merging and cleaning.")

# Save the data for Cox model
df[["sample_id", "predicted_risk", "duration", "event"]].to_csv("cox_input.csv", index=False)
print("✅ Cox input saved to cox_input.csv")





