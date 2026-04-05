# GoPay Fraud Engine

Python ML microservice for transaction fraud scoring (0-100) with risk bands and recommendations.

## Setup

```bash
cd fraud-engine

# 1) Create isolated virtual environment
python -m venv .venv

# 2) Install dependencies into venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# 3) Train model (creates fraud_model.pkl)
./.venv/Scripts/python.exe train.py

# 4) Run API service
./.venv/Scripts/python.exe app.py
```

Service runs on `http://localhost:5002`.

## Environment health check

```bash
cd fraud-engine
./.venv/Scripts/python.exe -m pip check
```

If this prints `No broken requirements found.`, the engine environment is clean and isolated.
