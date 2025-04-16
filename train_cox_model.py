import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# Load risk scores and clinical data
df = pd.read_csv("cox_input.csv")

# Median split into high and low risk
median_risk = df["predicted_risk"].median()
df["risk_group"] = df["predicted_risk"].apply(lambda x: "High" if x >= median_risk else "Low")

# Fit KM
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

for group in ["High", "Low"]:
    ix = df["risk_group"] == group
    kmf.fit(df.loc[ix, "duration"], df.loc[ix, "event"], label=group)
    kmf.plot_survival_function()

# Log-rank test
results = logrank_test(
    df[df["risk_group"] == "High"]["duration"],
    df[df["risk_group"] == "Low"]["duration"],
    event_observed_A=df[df["risk_group"] == "High"]["event"],
    event_observed_B=df[df["risk_group"] == "Low"]["event"]
)

plt.title("Kaplan-Meier Curve by Risk Group\n(p = {:.4f})".format(results.p_value))
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.savefig("km_curve.png")
plt.show()


