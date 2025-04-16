import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(page_title="Cancer Prognosis & Drug Recommendation", layout="wide")

st.title("🧬 Breast Cancer Prognosis System")
st.markdown("This system provides survival risk scores and drug recommendations based on multi-omics data.")

# Sidebar search
st.sidebar.header("🔍 Search Patient")
search_id = st.sidebar.text_input("Enter Patient ID:")

# Load data
@st.cache_data
def load_data():
    results = pd.read_csv("results.csv")
    clinical = pd.read_csv("data/clinical.csv")  # Corrected path
    df = pd.merge(results, clinical, on="sample_id", how="inner")
    df["sample_id"] = df["sample_id"].str.upper().str.strip().str[:15]
    return df

df = load_data()

# Display filtered data
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

# Risk group plot
st.subheader("📊 Risk Group Distribution")
if df["predicted_risk"].dropna().shape[0] > 1:
    thresh = df["predicted_risk"].median()
    df["risk_group"] = df["predicted_risk"].apply(lambda x: "High" if x > thresh else "Low")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="predicted_risk", hue="risk_group", bins=30, kde=True, palette="coolwarm", ax=ax)
    plt.axvline(thresh, color="black", linestyle="--", label="Median")
    plt.legend()
    st.pyplot(fig)

else:
    st.warning("Insufficient predicted risk data to plot distribution.")

# KM plot
st.subheader("📈 Kaplan-Meier Curve")
try:
    km_image = Image.open("km_curve.png")
    st.image(km_image, caption="Survival Curve by Predicted Risk")
except:
    st.warning("KM plot not available. Please run km_plot.py first.")

# Drug recommendation (if exists)
st.subheader("💊 Drug Recommendation")
try:
    recs = pd.read_csv("recommended_drugs.csv")
    st.write(recs.head(10))
except:
    st.info("Drug recommendation file not found. Please run recommend_drugs.py.")



