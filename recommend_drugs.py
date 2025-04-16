import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Load patient embeddings and drug fingerprints
patient_df = pd.read_csv("patient_embeddings.csv", index_col=0)
drug_df    = pd.read_csv("data/drug_fingerprints.csv", index_col=0)

patient_matrix = patient_df.values        # shape: (n_patients, embedding_dim)
drug_matrix    = drug_df.values           # shape: (n_drugs, fingerprint_dim)

# Reduce drug fingerprint dimensions to match patient embedding dimension
embedding_dim = patient_matrix.shape[1]   # e.g. 32

pca = PCA(n_components=embedding_dim)
drug_reduced    = pca.fit_transform(drug_matrix)  # shape: (n_drugs, embedding_dim)
patient_reduced = patient_matrix                  # already (n_patients, embedding_dim)

# Compute cosine similarity between each patient and each drug
similarity = cosine_similarity(patient_reduced, drug_reduced)

# Select top-5 drug recommendations for each patient
top_k = 5
recommendations = {
    pid: drug_df.index[np.argsort(-similarity[i])[:top_k]].tolist()
    for i, pid in enumerate(patient_df.index)
}

# Build and save recommendations DataFrame
recommend_df = pd.DataFrame.from_dict(
    recommendations,
    orient="index",
    columns=[f"rank_{i+1}" for i in range(top_k)]
)
recommend_df.index.name = "sample_id"

recommend_df.to_csv("recommendations.csv", index=True)
print("recommendations.csv saved with top-5 drug recommendations per patient.")

