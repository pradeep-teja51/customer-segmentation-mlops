import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---- Step 1: Load RFM data ----
print("Loading RFM data...")
rfm = pd.read_csv("data/rfm.csv")

# ---- Step 2: Scale features ----
X = rfm[["Recency", "Frequency", "Monetary"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Step 3: Start an MLflow experiment ----
import dagshub

dagshub.init(
    repo_owner="marinedipradeep2006",
    repo_name="customer-segmentation-mlops",
    mlflow=True
)

mlflow.set_experiment("customer_segmentation")
k = 4
with mlflow.start_run():
    # Train the model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm["Segment"] = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, rfm["Segment"])

    # Log parameters (settings we chose)
    mlflow.log_param("n_clusters", k)
    mlflow.log_param("algorithm", "KMeans")

    # Log metrics (results)
    mlflow.log_metric("silhouette_score", score)

    # Log the trained model itself
    mlflow.sklearn.log_model(kmeans, "kmeans_model")

    print(f"Final model: k={k}, Silhouette Score={score:.3f}")

# ---- Step 4: Show and save results ----
print(rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean())
print(rfm["Segment"].value_counts())
rfm.to_csv("data/rfm_segmented.csv", index=False)
print("Segmented data saved to data/rfm_segmented.csv")

import joblib

# ---- Step 6: Save the model and scaler for later use in the API ----
joblib.dump(kmeans, "src/kmeans_model.pkl")
joblib.dump(scaler, "src/scaler.pkl")
print("Model and scaler saved for API use.")