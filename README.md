# ✈️ US Airline Delay Prediction — MLOps Project

A complete **end-to-end MLOps pipeline** that predicts the number of flight delays in a month using structured airline data. This project integrates data ingestion, model training, hyperparameter tuning, evaluation and experiment tracking (MLflow) following modular and scalable MLOps practices.

---

## 🔍 Dataset Description

> Monthly summary of how each airline performed at a specific airport, capturing delay statistics and their causes.

- **Target Column**: **`arr_del15`** — Number of flights delayed (arrival delay ≥ 15 minutes) for each **carrier-airport pair** in a given **month**.


## 📁 Project Structure

```

US\_Airline/
├── backend/         # FastAPI application for model serving via REST API
├── ml/              # Core ML pipeline: training, evaluation, and inference logic
├── prisma-db/       # PostgreSQL schema managed via Prisma ORM
└── README.md

```

---

## 🧭 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/GITHUBsumesh/US_Airline.git
cd US_Airline
```

### 2️⃣ Setup the Python environment

```bash
conda create -n flight-full python=3.11 -y
conda activate flight-full
pip install -r requirements.txt
```

Or use the provided YAML:

```bash
conda env create -f environment.yaml
conda activate flight-full
```

### 3️⃣ Start FastAPI server

```bash
uvicorn backend.app:app --reload
```

---

### 🧾 Sample Prediction Data
Use the sample test CSV for predictions:

📂[test_data](./ml/test_data/test_data.csv)

---


## 📝 Folder Breakdown

### 📦 `backend/` – API Layer

```text
├── db/             # SQLAlchemy configuration
├── models/         # Pydantic and DB models
├── routes/         # Train and Predict endpoints
├── services/       # Business logic for pipeline handling
├── templates/      # Jinja2 HTML templates
└── utils/          # Adds ml path to sys, for backend access
```

### 🧠 `ml/` – Machine Learning Core

```text
├── src/                # Component-wise pipeline implementation
├── data_schema/        # Schema for input dataset
├── dataset/            # Raw dataset source
├── final_model/        # Best trained model + preprocessor
├── notebooks/          # EDA and development notebooks
├── test_data/          # Sample test file for predictions
└── prediction_output/  # CSV output of predictions
```

### 🗃️ `prisma-db/` – PostgreSQL Integration

```text
├── prisma/             # DB schema and setup
```

---

## 🚀 API Endpoints

| Endpoint        | Method | Description                                     |
| --------------- | ------ | ----------------------------------------------- |
| `/api/train/`   | `GET`  | Triggers full training pipeline                 |
| `/api/predict/` | `POST` | Upload test `.csv` and return delay predictions |

---

## 🔁 Pipeline Flow

### 🔹 Step 1: Data Ingestion

* Reads cleaned rows from PostgreSQL DB (excluding null `arr_del15`)
* Splits into `train.csv` and `test.csv` (80/20)
* Saves Data Ingestion Artifact

### 🔹 Step 2: Data Validation

* Drops irrelevant columns
* Validates schema and shape
* Generates **Data Drift Report**
* Saves validated CSVs and artifact

### 🔹 Step 3: Data Transformation

* Drops `arr_del15` during transformation
* For each model:

  * Builds a tailored `ColumnTransformer`

    * Imputes: Mean (numeric), Most Frequent (categorical)
    * Scales: `StandardScaler` for applicable models
    * Encodes:

      * One-Hot: `"carrier"` (21 unique values)
      * Ordinal: `"airport"` (353 unique values)
  * Saves transformed NumPy arrays and preprocessor

### 🔹 Step 4: Model Trainer

* Supports multiple regressors:

  * Linear, Ridge, Lasso, KNN, SVR, RandomForest, ExtraTrees, XGBoost, LightGBM, GradientBoosting, DNN, CatBoost
* Model-specific processing:

  * `CatBoost`: Works on raw strings, no encoding/scaling
  * `DNN`: Uses `RepeatedKFold`, early stopping
  * Others: Hyperparameter tuning via `RandomizedSearchCV`
* R² Score is used to evaluate each model
* Best model selected and retrained on full train set
* Model + preprocessor saved under `/final_model/`
* Metrics and model performance logged to Dagshub

---

## 🧠 Prediction Pipeline

1. Loads trained model and preprocessor (`final_model/`)
2. Accepts input CSV via API or CLI
3. Transforms the data using the saved pipeline
4. Returns predictions and logs them to DB + CSV

---

## 🛠️ Execution Flow

| Script/File           | Purpose                                |
|-----------------------|----------------------------------------|
| `ml/push_data.py`     | Loads complete raw dataset into DB     |
| `backend/train.py`    | Triggers end-to-end training pipeline  |
| `backend/predict.py`  | Triggers prediction pipeline           |

---
## 🧪 Technologies Used

- **ML/DL**: Scikit-Learn, CatBoost, XGBoost, Keras
- **MLOps**: MLflow, joblib, YAML-based config, Dagshub
- **Database**: PostgreSQL + Prisma ORM
- **Serving**: FastAPI (REST)
- **Logging**: Python logging + YAML reporting

---

## 🚧 Future Improvements

The following features are planned and under development:

* 🐳 **Dockerization**

  * Dockerize the entire application stack (FastAPI + PostgreSQL + Streamlit + MLflow)
  * Provide a `docker-compose.yml` for local setup

* 🌐 **Streamlit Web UI**

  * Interactive front-end for model training and CSV-based prediction
  * Include visualizations for model performance and data exploration

---

## ✅ How to Contribute

* [ ] Fork the repository
* [ ] Open issues for bugs or enhancements
* [ ] Submit PRs with clear commits and test cases

---

## 📌 Author

**Sumesh**
*Engineering Student | MLOps & Fullstack Enthusiast*

📧 [LinkedIn](https://linkedin.com/in/sumesh-ranjan-majee-yokoso)
📂 [GitHub](https://github.com/GITHUBsumesh)

---
