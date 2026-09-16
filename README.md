# PMLDL Assignment 1 — Exoplanet Classification Pipeline

This project implements an automated MLOps pipeline for classifying Kepler
exoplanet candidates as `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE`.

The raw source is NASA's
[Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)
dataset on Kaggle.

## Pipeline

```text
cumulative.csv
    → prepare_data
    → train.csv + test.csv
    → train_model
    → exoplanet_classifier.joblib + metrics.json
    → deploy
    → FastAPI container + Streamlit container
```

1. **Data engineering** loads the raw CSV, removes duplicates, imputes missing
   numerical values with training-set medians, detects outliers, and produces
   stratified training and testing datasets.
2. **Model engineering** applies feature transformations, trains a balanced
   multinomial logistic-regression model, evaluates it, logs the run in MLflow,
   and packages the model with joblib.
3. **Deployment** builds and starts separate FastAPI and Streamlit containers
   with Docker Compose.

DVC defines the pipeline dependencies. A cron job runs `dvc repro` every five
minutes.

## Repository layout

```text
code/
  datasets/          Data engineering scripts
  models/            Feature engineering and model training script
  deployment/
    api/             FastAPI service and Dockerfile
    app/             Streamlit application and Dockerfile
    docker-compose.yml
data/
  raw/               Downloaded Kaggle dataset, not committed to Git
  processed/         Generated train/test datasets, managed by DVC
models/              Generated packaged model, managed by DVC
notebooks/           Reserved for exploratory notebooks
scripts/              Automation script
requirements.txt     Python dependencies for the complete pipeline
```

`services/airflow` is intentionally not included because the project uses DVC,
not Airflow.

## Prerequisites

- Git
- Python 3.14
- Docker Desktop with WSL integration enabled
- A Kaggle account to download the raw dataset

## Installation

Clone the repository and enter it:

```bash
git clone https://github.com/ttrlen/pmldl-assignment-1.git
cd pmldl-assignment-1
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The repository already contains the DVC configuration. Verify it:

```bash
dvc status
```

## Raw data

Download `cumulative.csv` from the Kaggle dataset page and place it here:

```text
data/raw/cumulative.csv
```

The raw dataset is excluded from Git. This keeps the repository lightweight and
documents the source needed to reproduce the pipeline.

## Run the pipeline

Run every stage:

```bash
dvc repro
```

Display evaluation metrics:

```bash
dvc metrics show
```

Check whether the data pipeline is current:

```bash
dvc status
```

The deployment stage is configured as `always_changed`, so DVC intentionally
marks it as changed after a status check. This ensures the scheduler invokes
Docker Compose on every scheduled cycle.

## Access the application and API

After `dvc repro` completes, open:

- Streamlit application: `http://localhost:8501`
- FastAPI Swagger documentation: `http://localhost:8000/docs`
- API health check: `http://localhost:8000/health`

Check the two running containers:

```bash
docker compose -f code/deployment/docker-compose.yml ps
```

Stop the application and API:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

## MLflow experiment tracking

Start the MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://localhost:5000` and select the `exoplanet-classification`
experiment. The interface displays parameters, metrics, the classification
report, and the logged model for each training run.

## Automated schedule

First verify that cron is running:

```bash
sudo systemctl enable --now cron
sudo systemctl status cron --no-pager
```

Test the automation script manually:

```bash
bash scripts/run_pipeline.sh
tail -n 30 logs/pipeline.log
```

Open the personal cron table:

```bash
crontab -e
```

```cron
*/5 * * * * bash /home/ttrlen/PMLDLproj/pmldl-assignment-1/scripts/run_pipeline.sh
```

Verify the schedule:

```bash
crontab -l
```

Follow automation output:

```bash
tail -f logs/pipeline.log
```

The absolute path in the cron entry must be changed if the repository is cloned
to a different directory.
