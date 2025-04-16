import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os

df = pd.read_csv("data/raw/gdsc_all_drugs_official.csv")

def get_smiles_from_pubchem(cid):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip()
    except:
        return None

def smiles_to_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return list(fp)
    except:
        return None

tqdm.pandas()
df["smiles"] = df["pubchem_id"].progress_apply(get_smiles_from_pubchem)
df["fingerprint"] = df["smiles"].progress_apply(smiles_to_fingerprint)

fps_df = df.dropna(subset=["fingerprint"]).copy()
fps_matrix = pd.DataFrame(fps_df["fingerprint"].tolist(), index=fps_df["drug_name"])
fps_matrix.index.name = "drug_name"

os.makedirs("data", exist_ok=True)
fps_matrix.to_csv("data/drug_fingerprints.csv")
print("drug_fingerprints.csv saved.")
