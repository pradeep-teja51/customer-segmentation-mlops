import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

cols = ["Recency", "Frequency", "Monetary"]

# Reference data: what the model was trained on
reference = pd.read_csv("data/rfm.csv")[cols]

# Current data: simulated "new" customers
current = reference.sample(n=1000, random_state=42).copy()
current["Recency"] = current["Recency"] + 60      # customers are buying less recently
current["Monetary"] = current["Monetary"] * 1.5   # customers are spending more
# Frequency is left unchanged on purpose, as a control

report = Report([DataDriftPreset()])
result = report.run(current_data=current, reference_data=reference)
result.save_html("monitoring/drift_report.html")
print("Drift report saved to monitoring/drift_report.html")