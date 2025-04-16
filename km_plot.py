import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Load the clinical data
df = pd.read_csv("cox_input.csv")

# Create KaplanMeierFitter instance
kmf = KaplanMeierFitter()

# Define high and low risk groups based on the median predicted risk
median_risk = df["predicted_risk"].median()
df["risk_group"] = ["High Risk" if x > median_risk else "Low Risk" for x in df["predicted_risk"]]

# Plot the Kaplan-Meier curve for both risk groups
plt.figure(figsize=(10, 6))

for group in df["risk_group"].unique():
    group_df = df[df["risk_group"] == group]
    kmf.fit(durations=group_df["duration"], event_observed=group_df["event"], label=group)
    kmf.plot_survival_function()

plt.title("Kaplan-Meier Survival Curves by Risk Group")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.legend()
plt.show()



