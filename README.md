# PMLDL Assignment 1 — MLOps pipeline for exoplanet classification

This repository contains an automated machine-learning pipeline that classifies
Kepler exoplanet candidates as `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE`.

The raw source is the NASA dataset
[Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)
hosted on Kaggle.

## Planned pipeline

1. **Data engineering**: load raw data, clean missing values and outliers, then
   create training and testing datasets.
2. **Model engineering**: engineer features, train and evaluate a classifier,
   log metrics and package the trained model.
3. **Deployment**: serve the model through a FastAPI API and use it from a
   Streamlit web application. The API and application run in separate Docker
   containers.

The pipeline will be defined with DVC and scheduled to run automatically.

## Repository layout

```text
code/
  datasets/          # Data download, cleaning, and train/test split code
  models/            # Feature engineering, training, and evaluation code
  deployment/
    api/             # FastAPI service and its Dockerfile
    app/             # Streamlit application and its Dockerfile
data/
  raw/               # Downloaded source dataset (not committed to Git)
  processed/         # Generated train/test datasets (not committed to Git)
models/              # Generated packaged model (not committed to Git)
```

Detailed setup and launch instructions will be added as the project is built.
