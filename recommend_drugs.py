import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Load patient embeddings and drug fingerprints
patient_df = pd.read_csv("patient_embeddings.csv", index_col=0)
drug_df = pd.read_csv("data/drug_fingerprints.csv", index_col=0)

# Convert to matrix
patient_matrix = patient_df.values
drug_matrix = drug_df.values

# Reduce drug fingerprint from 2048 -> 32 dimensions
pca = PCA(n_components=patient_matrix.shape[1])
drug_reduced = pca.fit_transform(drug_matrix)

# Compute cosine similarity
similarity = cosine_similarity(patient_matrix, drug_reduced)

# Save top-5 recommendations
top_k = 5
recommendations = {}
for i, pid in enumerate(patient_df.index):
    top_drugs = drug_df.index[np.argsort(-similarity[i])[:top_k]]
    recommendations[pid] = top_drugs.tolist()

recommend_df = pd.DataFrame.from_dict(recommendations, orient="index")
recommend_df.index.name = "sample_id"
recommend_df.columns = [f"rank_{i+1}" for i in range(top_k)]
recommend_df.to_csv("recommendations.csv")

print("recommendations.csv saved.")
