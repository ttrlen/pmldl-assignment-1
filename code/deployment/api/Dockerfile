FROM python:3.14-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY code/deployment/api/main.py /app/main.py
COPY models/exoplanet_classifier.joblib /app/models/exoplanet_classifier.joblib

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
