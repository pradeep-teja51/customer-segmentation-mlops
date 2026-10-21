from prefect import flow, task
import subprocess

@task
def run_ingest():
    print("Running data ingestion...")
    subprocess.run(["venv/Scripts/python.exe", "src/ingest.py"], check=True)

@task
def run_features():
    print("Running feature engineering...")
    subprocess.run(["venv/Scripts/python.exe", "src/features.py"], check=True)

@task
def run_train():
    print("Running model training...")
    subprocess.run(["venv/Scripts/python.exe", "src/train.py"], check=True)

@flow(name="customer-segmentation-pipeline")
def full_pipeline():
    run_ingest()
    run_features()
    run_train()

if __name__ == "__main__":
    full_pipeline()