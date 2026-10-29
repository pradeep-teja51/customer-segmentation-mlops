
from pathlib import Path
import sqlite3
import subprocess
import html
import re

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SEGMENTIQ | MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent.parent


def path(name):
    return ROOT / name


def load_csv(name):
    try:
        file = path(name)
        return pd.read_csv(file) if file.is_file() else None
    except Exception:
        return None


def safe(value):
    return html.escape(str(value))


def heading(label, title, description=""):
    st.html(f"""
    <div class="section-head">
        <div class="eyebrow">{safe(label)}</div>
        <div class="section-title">{safe(title)}</div>
        <div class="section-description">{safe(description)}</div>
    </div>
    """)


def metric(label, value, detail="", color="#5de7f2"):
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">{safe(label)}</div>
        <div class="metric-value" style="color:{color}">
            {safe(value)}
        </div>
        <div class="metric-detail">{safe(detail)}</div>
    </div>
    """)


# ============================================================
# DESIGN
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #050a12;
    --panel: #0c1726;
    --border: #203b51;
    --cyan: #5de7f2;
    --blue: #8ab8ff;
    --green: #55e6a5;
    --white: #edf6ff;
    --muted: #8ca2bb;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 50% -20%, #173654 0%, transparent 48%),
        linear-gradient(145deg, #050a12, #07111e 55%, #050a12);
    color: var(--white);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    width: 100% !important;
    max-width: none !important;
    padding: 1.2rem clamp(14px, 2.2vw, 38px) 2rem !important;
    margin: 0 !important;
}

section.main > div {
    max-width: none !important;
}

.brand-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 8px 0 18px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--border);
}

.brand {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -2px;
}

.brand span {
    color: var(--cyan);
}

.brand-sub {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 2px;
    margin-top: 6px;
}

.header-status {
    text-align: right;
    color: var(--muted);
    font: 10px 'DM Mono', monospace;
    line-height: 1.8;
}

.hero {
    padding: clamp(20px, 2.4vw, 32px);
    border: 1px solid #28506a;
    border-radius: 17px;
    background: linear-gradient(115deg, #10283d, #091421);
    box-shadow: 0 0 30px rgba(45,195,220,.06);
    margin-bottom: 20px;
}

.hero-title {
    font-size: clamp(25px, 3vw, 42px);
    font-weight: 800;
    letter-spacing: -1.8px;
    line-height: 1.2;
}

.eyebrow {
    color: var(--cyan);
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1.7px;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.metric-card {
    height: 100%;
    min-height: 112px;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #203e55;
    background: linear-gradient(145deg, #102237, #091320);
}

.metric-label {
    font-size: 9px;
    font-weight: 800;
    color: var(--muted);
    letter-spacing: 1.2px;
}

.metric-value {
    font-size: clamp(21px, 2.2vw, 31px);
    font-weight: 800;
    letter-spacing: -1px;
    margin: 9px 0 4px;
    overflow-wrap: anywhere;
}

.metric-detail {
    color: #8198b2;
    font-size: 10px;
}

.section-head {
    border-bottom: 1px solid var(--border);
    padding: 4px 0 12px;
    margin: 15px 0 13px;
}

.section-title {
    color: var(--white);
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -.5px;
}

.section-description {
    color: var(--muted);
    font-size: 10px;
    margin-top: 5px;
    line-height: 1.5;
}

.stButton > button {
    background: #102a3d;
    color: white;
    border: 1px solid #2b6078;
    border-radius: 8px;
    font-weight: 700;
}

.stButton > button:hover {
    border-color: var(--cyan);
    color: var(--cyan);
}

div[data-testid="stMetric"] {
    background: #0b1624;
    border: 1px solid #203b51;
    border-radius: 10px;
    padding: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #203b51;
    border-radius: 9px;
}

hr {
    border-color: var(--border);
}

footer {
    visibility: hidden;
}

@media(max-width:700px) {
    .block-container {
        padding: 1rem 12px 2rem !important;
    }

    .brand-row {
        align-items: flex-start;
    }

    .header-status {
        font-size: 8px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA AND MODEL LOADING
# ============================================================

rfm = load_csv("dashboard/rfm_segmented.csv")

if rfm is None:
    rfm = load_csv("data/rfm_segmented.csv")

required_columns = {
    "Recency", "Frequency", "Monetary", "Segment"
}

if rfm is not None and not required_columns.issubset(rfm.columns):
    rfm = None

model = None
scaler = None
model_error = None

try:
    if path("src/kmeans_model.pkl").is_file():
        model = joblib.load(path("src/kmeans_model.pkl"))

    if path("src/scaler.pkl").is_file():
        scaler = joblib.load(path("src/scaler.pkl"))

except Exception as exc:
    model_error = str(exc)



# ============================================================
# MLFLOW EXPERIMENTS - DAGSHUB REMOTE TRACKING
# ============================================================

def load_mlflow():
    try:
        import os
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = st.secrets.get("MLFLOW_TRACKING_URI")
        username = st.secrets.get("MLFLOW_TRACKING_USERNAME")
        password = st.secrets.get("MLFLOW_TRACKING_PASSWORD")

        if not tracking_uri or not username or not password:
            return None, "DagsHub credentials are missing from Streamlit Secrets."

        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        experiments = client.search_experiments()
        rows = []

        for experiment in experiments:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=100,
                order_by=["attributes.start_time DESC"],
            )

            for run in runs:
                row = {
                    "experiment_name": experiment.name,
                    "run_uuid": run.info.run_id,
                    "status": run.info.status,
                    "start_time": pd.to_datetime(
                        run.info.start_time,
                        unit="ms",
                        errors="coerce",
                    ),
                    "end_time": pd.to_datetime(
                        run.info.end_time,
                        unit="ms",
                        errors="coerce",
                    ),
                }

                row.update(run.data.params)
                row.update(run.data.metrics)
                rows.append(row)

        if not rows:
            return pd.DataFrame(), None

        return pd.DataFrame(rows), None

    except Exception as exc:
        return None, f"DagsHub MLflow connection failed: {exc}"


mlflow_runs, mlflow_error = load_mlflow()
drift_report = path("monitoring/drift_report.html")



# ============================================================
# HEADER
# Technology tags, summary line, and footer removed
# ============================================================

st.html("""
<div class="brand-row">
    <div>
        <div class="brand">SEGMENT<span>IQ</span></div>
        <div class="brand-sub">CUSTOMER SEGMENTATION / MLOPS</div>
    </div>
    <div class="header-status">
        <span style="color:#55e6a5">● DASHBOARD ACTIVE</span><br>
    </div>
</div>

<div class="hero">
    <div class="eyebrow">TECHNICAL OVERVIEW</div>
    <div class="hero-title">
        Customer Segmentation MLOps
    </div>
</div>
""")


if rfm is None:
    st.error(
        "Segmented customer CSV is unavailable or missing required columns."
    )
    st.stop()

features = ["Recency", "Frequency", "Monetary"]

numeric_rfm = rfm[features].apply(
    pd.to_numeric,
    errors="coerce",
)


# ============================================================
# 01 / CLUSTERING AND RFM OVERVIEW
# ============================================================

kpis = st.columns(5)

with kpis[0]:
    metric("CUSTOMER RECORDS", f"{len(rfm):,}", "Segmented dataset")

with kpis[1]:
    metric(
        "CLUSTERS",
        rfm["Segment"].nunique(),
        "Distinct cluster labels",
        "#55e6a5",
    )

with kpis[2]:
    metric(
        "MEAN RECENCY",
        f"{numeric_rfm['Recency'].mean():,.2f}",
        "Days since last purchase",
        "#8ab8ff",
    )

with kpis[3]:
    metric(
        "MEAN FREQUENCY",
        f"{numeric_rfm['Frequency'].mean():,.2f}",
        "Order frequency",
        "#b6a0ff",
    )

with kpis[4]:
    metric(
        "MEAN MONETARY",
        f"{numeric_rfm['Monetary'].mean():,.2f}",
        "Customer spend",
    )

left, right = st.columns([1.05, 1], gap="large")

with left:
    heading(
        "01 / CLUSTERING",
        "K-Means segment distribution",
        "Customer count for each cluster label",
    )

    st.bar_chart(
        rfm["Segment"].value_counts().sort_index(),
        height=300,
        use_container_width=True,
    )

with right:
    heading(
        "02 / RFM PROFILES",
        "Cluster feature means",
        "Average Recency, Frequency and Monetary by segment",
    )

    st.dataframe(
        rfm.groupby("Segment")[features].mean().round(2),
        use_container_width=True,
        height=300,
    )


# ============================================================
# 03 / MODEL INFERENCE
# ============================================================

left, right = st.columns([1.05, 1], gap="large")

with left:
    heading(
        "03 / INFERENCE",
        "K-Means prediction",
        "Input RFM features → saved scaler → trained clustering model",
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        recency = col1.number_input(
            "Recency (days)", min_value=0, value=30
        )

        frequency = col2.number_input(
            "Frequency (orders)", min_value=0, value=5
        )

        monetary = col3.number_input(
            "Monetary spend", min_value=0.0, value=1000.0
        )

        predict = st.form_submit_button(
            "RUN PREDICTION",
            type="primary",
        )

    if predict:
        if model is None or scaler is None:
            st.error("The model or scaler could not be loaded.")

            if model_error:
                st.caption(model_error)

        else:
            try:
                sample = np.array(
                    [[recency, frequency, monetary]],
                    dtype=float,
                )

                transformed = scaler.transform(sample)
                predicted_segment = model.predict(transformed)[0]

                st.success(f"Predicted cluster: {predicted_segment}")

                matching = rfm.loc[
                    rfm["Segment"] == predicted_segment,
                    features,
                ]

                if not matching.empty:
                    st.caption("Historical cluster profile")
                    st.dataframe(
                        matching.mean().to_frame("Mean").round(2),
                        use_container_width=True,
                    )

            except Exception as exc:
                st.error(f"Prediction error: {exc}")

with right:
    heading(
        "04 / MODEL EVALUATION",
        "Model artifacts and metrics",
        "Values are shown when available in the saved estimator or MLflow",
    )

    cols = st.columns(2)

    with cols[0]:
        metric(
            "K-MEANS MODEL",
            "LOADED" if model is not None else "MISSING",
            "src/kmeans_model.pkl",
            "#55e6a5" if model is not None else "#ffc978",
        )

    with cols[1]:
        metric(
            "FEATURE SCALER",
            "LOADED" if scaler is not None else "MISSING",
            "src/scaler.pkl",
            "#55e6a5" if scaler is not None else "#ffc978",
        )

    if model is not None and hasattr(model, "n_clusters"):
        st.metric("Configured cluster count", model.n_clusters)

    if model is not None and hasattr(model, "inertia_"):
        st.metric("K-Means inertia", f"{model.inertia_:,.3f}")

    if mlflow_runs is not None:
        for column in mlflow_runs.columns:
            if "silhouette" in str(column).lower():
                scores = pd.to_numeric(
                    mlflow_runs[column],
                    errors="coerce",
                ).dropna()

                if not scores.empty:
                    st.metric(
                        f"Recorded {column}",
                        f"{scores.max():.4f}",
                    )
                    break


# ============================================================
# 05 / MLFLOW EXPERIMENT TRACKING
# ============================================================

st.divider()

heading(
    "05 / EXPERIMENT TRACKING",
    "MLflow experiments",
    "Experiment runs, parameters, statuses and logged metrics",
)

if mlflow_runs is None:
    st.warning(mlflow_error or "MLflow run records are unavailable.")

elif mlflow_runs.empty:
    st.info("No experiment runs were found in the database.")

else:
    st.dataframe(
        mlflow_runs,
        use_container_width=True,
        height=260,
    )

    st.download_button(
        "EXPORT EXPERIMENT RESULTS",
        mlflow_runs.to_csv(index=False).encode("utf-8"),
        file_name="mlflow_experiments.csv",
        mime="text/csv",
    )


# ============================================================
# 06 / DATA QUALITY
# Full-width section, above Data Drift
# ============================================================

st.divider()

heading(
    "06 / DATA QUALITY",
    "RFM validation checks",
    "Missing values, duplicate records and negative feature values",
)

quality = st.columns(3)

quality[0].metric(
    "Missing values",
    f"{int(numeric_rfm.isna().sum().sum()):,}",
)

quality[1].metric(
    "Duplicate rows",
    f"{int(rfm.duplicated().sum()):,}",
)

quality[2].metric(
    "Negative values",
    f"{int((numeric_rfm < 0).sum().sum()):,}",
)

selected_feature = st.selectbox(
    "Feature distribution",
    features,
)

distribution = (
    numeric_rfm[selected_feature]
    .dropna()
    .round(0)
    .value_counts()
    .sort_index()
    .head(60)
)

st.bar_chart(
    distribution,
    height=250,
    use_container_width=True,
)


# ============================================================
# 07 / DATA DRIFT MONITORING
# Full-width Evidently report
# ============================================================

st.divider()

heading(
    "07 / DATA DRIFT",
    "Evidently monitoring",
    "Generated drift report from the existing monitoring workflow",
)

if drift_report.is_file():
    st.success("Generated drift report is available.")

    report_html = drift_report.read_text(
        encoding="utf-8",
        errors="replace",
    )

    components.html(
        report_html,
        height=850,
        scrolling=True,
    )

    st.download_button(
        "DOWNLOAD DRIFT REPORT",
        drift_report.read_bytes(),
        file_name="drift_report.html",
        mime="text/html",
    )

else:
    st.warning("No generated monitoring/drift_report.html found.")

st.caption(
    "A drift report is evidence of a drift check on the data used "
    "to generate it. It does not establish continuous production "
    "monitoring unless that workflow is configured."
)


# ============================================================
# 08 / DVC DATA VERSIONING
# ============================================================

st.divider()

heading(
    "08 / DATA VERSIONING",
    "DVC data tracking and version metadata",
    "Tracked dataset outputs, metadata and local workspace changes",
)

dvc_metadata = path("data.dvc")
dvc_config = path(".dvc/config")

tracked_outputs = []
metadata_text = ""

if dvc_metadata.is_file():
    try:
        metadata_text = dvc_metadata.read_text(
            encoding="utf-8",
            errors="replace",
        )

        tracked_outputs = re.findall(
            r"(?m)^\s*path:\s*(.+?)\s*$",
            metadata_text,
        )

    except OSError:
        metadata_text = ""

if tracked_outputs:
    dvc_metrics = st.columns(3)

    with dvc_metrics[0]:
        metric(
            "TRACKED OUTPUTS",
            len(tracked_outputs),
            "Paths recorded in data.dvc",
        )

    with dvc_metrics[1]:
        metric(
            "METADATA SIZE",
            f"{len(metadata_text.encode('utf-8')):,} B",
            "data.dvc file size",
            "#8ab8ff",
        )

    with dvc_metrics[2]:
        tracking_entries = len(
            re.findall(
                r"(?m)^\s*-\s*(?:md5|etag|path|hash):",
                metadata_text,
            )
        )

        metric(
            "TRACKING ENTRIES",
            tracking_entries,
            "Metadata entries parsed",
            "#55e6a5",
        )

    st.markdown("#### Tracked data")

    tracked_df = pd.DataFrame({
        "Tracked output": tracked_outputs,
    })

    st.dataframe(
        tracked_df,
        use_container_width=True,
        hide_index=True,
    )

    
    st.markdown("#### Dataset and remote storage")

    hash_match = re.search(
        r"(?m)^\s*-\s*md5:\s*(.+?)\s*$",
        metadata_text,
    )
    size_match = re.search(
        r"(?m)^\s*size:\s*(\d+)\s*$",
        metadata_text,
    )
    files_match = re.search(
        r"(?m)^\s*nfiles:\s*(\d+)\s*$",
        metadata_text,
    )

    dataset_hash = hash_match.group(1) if hash_match else "Unavailable"
    dataset_size = int(size_match.group(1)) if size_match else None
    file_count = int(files_match.group(1)) if files_match else None

    remote_url = "Not configured"

    if dvc_config.is_file():
        try:
            config_text = dvc_config.read_text(
                encoding="utf-8",
                errors="replace",
            )
            endpoint_match = re.search(
                r"(?m)^\s*endpointurl\s*[=:]\s*(.+?)\s*$",
                config_text,
            )
            url_match = re.search(
                r'(?m)^\s*url\s*[=:]\s*(.+?)\s*$',
                config_text,
            )



            if endpoint_match and "dagshub.com" in endpoint_match.group(1):

                remote_url = "DagsHub DVC"

            elif url_match and "dagshub.com" in url_match.group(1):
                remote_url = "DagsHub DVC"

            

        except OSError:
            pass

    info_cols = st.columns(4)

    with info_cols[0]:
        metric(
            "DATASET SIZE",
            f"{dataset_size / (1024 * 1024):.2f} MB"
            if dataset_size is not None else "Unavailable",
            "Tracked dataset",
        )

    with info_cols[1]:
        metric(
            "FILES TRACKED",
            file_count if file_count is not None else "Unavailable",
            "Files in data/ folder",
            "#8ab8ff",
        )

    with info_cols[2]:
        metric(
            "DATASET HASH",
            dataset_hash[:12] + "..."
            if dataset_hash != "Unavailable" else dataset_hash,
            "Version identifier",
            "#55e6a5",
        )

    with info_cols[3]:
        metric(
            "REMOTE STORAGE",
            remote_url,
            "DVC remote configuration",
            "#b6a0ff",
        )


else:
    st.info(
        "No simple tracked output paths could be read from data.dvc. "
        "The raw metadata can be inspected below if the file is present."
    )

if metadata_text:
    with st.expander("View data.dvc metadata"):
        st.code(metadata_text[:8000], language="yaml")

if dvc_config.is_file():
    try:
        config_text = dvc_config.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        config_text = ""

    if config_text.strip():
        with st.expander("View DVC configuration"):
            st.code(config_text[:4000], language="ini")

try:
    dvc_result = subprocess.run(
        ["dvc", "status"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    st.markdown("#### Workspace change status")

    if dvc_result.returncode == 0:
        dvc_output = (
            dvc_result.stdout.strip()
            or "DVC reports no workspace changes."
        )

        st.success("DVC status command completed.")
        st.code(dvc_output)

    else:
        dvc_output = (
            dvc_result.stderr.strip()
            or dvc_result.stdout.strip()
        )

        if dvc_output:
            st.warning("DVC returned a workspace status message.")
            st.code(dvc_output[:3000])

except (FileNotFoundError, subprocess.TimeoutExpired):
    pass

st.caption(
    "The metadata describes tracked outputs. Workspace status is shown "
    "only when the DVC command can run; it does not independently prove "
    "that data was pushed to a remote storage location."
)
