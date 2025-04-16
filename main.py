import pandas as pd
import torch
from fusion_model import MultiOmicsModel

# Limit methylation features
N_METHYLATION_FEATURES = 10000

# Load data
rna_df = pd.read_csv("data/processed/rna_matched.csv", index_col=0)
mutation_df = pd.read_csv("data/processed/mutation_matched.csv", index_col=0)
methylation_df = pd.read_csv(
    "data/processed/methylation_matched.csv",
    index_col=0,
    usecols=[0] + list(range(1, N_METHYLATION_FEATURES + 1))
)

# Normalize sample IDs
rna_df.index = rna_df.index.str[:15].str.upper()
mutation_df.index = mutation_df.index.str[:15].str.upper()
methylation_df.index = methylation_df.index.str[:15].str.upper()

# Drop duplicate IDs
rna_df = rna_df[~rna_df.index.duplicated()]
mutation_df = mutation_df[~mutation_df.index.duplicated()]
methylation_df = methylation_df[~methylation_df.index.duplicated()]

# Fill NaNs
rna_df = rna_df.fillna(0)
mutation_df = mutation_df.fillna(0)
methylation_df = methylation_df.fillna(0)

# Align sample IDs
common_ids = sorted(set(rna_df.index) & set(mutation_df.index) & set(methylation_df.index))
rna_df = rna_df.loc[common_ids]
mutation_df = mutation_df.loc[common_ids]
methylation_df = methylation_df.loc[common_ids]

# Show shapes
print("✅ Aligned shapes:")
print("rna:", rna_df.shape)
print("mutation:", mutation_df.shape)
print("methylation:", methylation_df.shape)

# Convert to tensors
rna = torch.tensor(rna_df.values, dtype=torch.float32)
mutation = torch.tensor(mutation_df.values, dtype=torch.float32)
methylation = torch.tensor(methylation_df.values, dtype=torch.float32)

# Check for NaNs before model
print("🔍 Checking input tensors for NaN or Inf:")
print("rna has NaN:", torch.isnan(rna).any().item())
print("mutation has NaN:", torch.isnan(mutation).any().item())
print("methylation has NaN:", torch.isnan(methylation).any().item())

# Initialize model
model = MultiOmicsModel(input_dims={
    "methylation": methylation.shape[1],
    "rna": rna.shape[1],
    "mutation": mutation.shape[1],
})

# Inference
model.eval()
with torch.no_grad():
    output = model({
        "methylation": methylation,
        "rna": rna,
        "mutation": mutation,
    })

# Save outputs
risk_scores = output[:, 0].numpy()
embeddings = output[:, 1:].numpy()

results_df = pd.DataFrame({
    "sample_id": common_ids,
    "predicted_risk": risk_scores
})
print("✅ Saving results.csv...")
print("Total samples:", len(results_df))
print("Missing predicted_risk:", results_df["predicted_risk"].isnull().sum())

results_df.to_csv("results.csv", index=False)
pd.DataFrame(embeddings, index=common_ids).to_csv("patient_embeddings.csv")

print("✅ Prediction complete. Results saved to results.csv and patient_embeddings.csv")

