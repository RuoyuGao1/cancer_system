import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(page_title="Cancer Prognosis & Drug Recommendation", layout="wide")

st.title("🧬 Breast Cancer Prognosis System")
st.markdown("This system provides survival risk scores and drug recommendations based on multi‑omics data.")

st.sidebar.header("🔍 Search Patient")
search_id = st.sidebar.text_input("Enter Patient ID:")

def load_data():
    results  = pd.read_csv("results.csv")
    clinical = pd.read_csv("cox_input.csv")
    df = pd.merge(results, clinical, on="sample_id", how="inner")
    if "predicted_risk_x" in df.columns:
        df["predicted_risk"] = df["predicted_risk_x"]
        drop_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
        df = df.drop(columns=drop_cols)
    df["sample_id"] = (
        df["sample_id"]
          .astype(str)
          .str.upper()
          .str.strip()
          .str[:15]
    )
    return df

df = load_data()

if search_id:
    filtered = df[df["sample_id"].str.contains(search_id.upper())]
    if not filtered.empty:
        st.subheader(f"🎯 Patient Info: {filtered.iloc[0]['sample_id']}")
        st.dataframe(filtered)
        st.metric("Predicted Risk Score", f"{filtered.iloc[0]['predicted_risk']:.4f}")
    else:
        st.warning("Patient not found.")
else:
    st.subheader("📋 Full Patient Table")
    st.dataframe(df.head(50))

st.subheader("📊 Risk Group Distribution")
if df["predicted_risk"].notna().sum() > 1:
    median_risk = df["predicted_risk"].median()
    df["risk_group"] = df["predicted_risk"].apply(lambda x: "High" if x > median_risk else "Low")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="predicted_risk", hue="risk_group", bins=30, kde=True, ax=ax)
    plt.axvline(median_risk, linestyle="--", label="Median")
    plt.legend()
    st.pyplot(fig)
else:
    st.warning("Insufficient data to plot risk distribution.")

st.subheader("📈 Kaplan–Meier Curve")
try:
    km_image = Image.open("km_curve.png")
    st.image(km_image, caption="Survival Curve by Predicted Risk")
except FileNotFoundError:
    st.warning("KM curve not available. Please run km_plot.py first.")

# Drug recommendations
st.subheader("💊 Drug Recommendation")
try:
    recs = pd.read_csv("recommendations.csv", index_col=0)
    if search_id:
        sid = search_id.upper().strip()[:15]
        if sid in recs.index:
            st.write(recs.loc[[sid]])
        else:
            st.warning("No drug recommendations for this patient.")
    else:
        st.write(recs.head(10))
except FileNotFoundError:
    st.info("Drug recommendation file not found. Please run recommend_drugs.py.")




