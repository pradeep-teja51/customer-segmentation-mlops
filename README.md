## Live Dashboard

Explore the deployed SEGMENTIQ customer segmentation dashboard:

**[Open SEGMENTIQ Dashboard](https://customer-segmentation-mlops-wyevujnjtnx7avpqdb4i8p.streamlit.app/)**

# Customer Segmentation Pipeline (MLOps)

![CI](https://github.com/pradeep-teja51/customer-segmentation-mlops/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps project that segments retail customers with RFM analysis and K-Means clustering. It includes experiment tracking, data versioning, pipeline orchestration, a prediction API, a dashboard, drift monitoring, Docker and CI.

Built for the Machine Learning Operations course by M Pradeep Teja and D Sai Praneeth.

## Problem

Businesses often treat all customers the same. This project groups customers by purchasing behaviour so marketing can treat each group differently, and builds the pipeline around the model so the work is reproducible and deployable.

## Results

Dataset: UCI Online Retail. 541,909 raw transactions were cleaned to 392,692, giving 4,338 unique customers.

Model: K-Means with k = 4 on standardized RFM features, silhouette score 0.616. k = 2 scored higher, but it mostly split a few extreme outliers from everyone else, so k = 4 was chosen for segments a marketing team can act on.

| Segment | Customers | Avg Recency (days) | Avg Frequency (orders) | Avg Monetary | Meaning |
|---|---|---|---|---|---|
| 0 | 3,054 | 43.7 | 3.7 | 1,354 | Regular customers |
| 1 | 1,067 | 248.1 | 1.6 | 479 | At-risk / lapsed |
| 2 | 13 | 7.4 | 82.5 | 127,188 | VIP / wholesale |
| 3 | 204 | 15.5 | 22.3 | 12,691 | High-value loyal |

## Pipeline

| Stage | Tool | File |
|---|---|---|
| Data ingestion and cleaning | pandas | `src/ingest.py` |
| RFM feature engineering | pandas | `src/features.py` |
| Exploratory analysis | Jupyter | `notebooks/eda.ipynb` |
| Training and experiment tracking | scikit-learn, MLflow | `src/train.py` |
| Data versioning | DVC | `data.dvc` |
| Orchestration | Prefect | `pipeline/flow.py` |
| Prediction API | FastAPI | `api/main.py` |
| Dashboard | Streamlit | `dashboard/app.py` |
| Drift monitoring | Evidently | `monitoring/drift_report.py` |
| Containerization | Docker | `Dockerfile` |
| CI | GitHub Actions | `.github/workflows/ci.yml` |

## Project structure

```
api/          FastAPI service
dashboard/    Streamlit dashboard
src/          ingestion, features, training, saved model and scaler
pipeline/     Prefect flow
monitoring/   Evidently drift report
notebooks/    EDA notebook
tests/        API tests
```

## Run the prediction API with Docker

```
docker build -t customer-segmentation-app .
docker run -p 8000:8000 customer-segmentation-app
```

Open http://127.0.0.1:8000/docs and call `POST /predict-segment`:

```json
{"recency": 15, "frequency": 22, "monetary": 12000}
```

This returns `{"segment": 3}`, the High-Value Loyal segment.

## Run the full pipeline locally

The `data/` folder is tracked by DVC and is not in this repository, and no DVC remote is configured. Download the [UCI Online Retail dataset](https://archive.ics.uci.edu/static/public/352/online+retail.zip) and place `Online Retail.xlsx` in `data/`.

`requirements.txt` holds only the packages the serving container needs. For the full pipeline, install the extra tools as well:

```
pip install -r requirements.txt
pip install jupyter mlflow dvc prefect evidently openpyxl matplotlib seaborn plotly
python pipeline/flow.py
```

`pipeline/flow.py` calls `venv/Scripts/python.exe`, so it expects a Windows virtual environment named `venv`. Then start the dashboard:

```
streamlit run dashboard/app.py
```

## Tests

```
pip install pytest httpx
python -m pytest tests -v
```

## Monitoring

`monitoring/drift_report.py` compares the training RFM data with a simulated "new" sample in which Recency and Monetary are shifted on purpose. Evidently flagged both shifted columns as drifted and left Frequency alone.

## Status and limitations

- The dashboard is deployed on Streamlit Community Cloud. The prediction API is containerized with Docker but is not hosted publicly.
- CI tests and builds the API service only. The dashboard needs `data/rfm_segmented.csv`, which is not in the repository.
- The drift check uses simulated data, and drift does not trigger retraining automatically.
- The two API tests are a small suite, not full coverage.
