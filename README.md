# ✈️ US Airline Delay Prediction — MLOps Project

## A complete end-to-end MLOps project that predicts the number of flight that will be delayed in a month using structured airline data. This project integrates model training, experiment tracking, database logging, and deployment via modern tooling.


## 🔍 Dataset Represents:
"Monthly summary of how an airline performed at an airport (in terms of delays and causes)"

## 📁 Project Structure

```
US_Airline/
├── backend/         # FastAPI application for serving ML predictions via REST API
├── docker/          # Dockerfiles and Docker Compose for all services
├── frontend/        # Streamlit web UI for user interaction and predictions
├── ml/              # Training, evaluation, model selection, and inference logic
├── prisma-db/       # PostgreSQL database schema managed via Prisma ORM
└── README.md
```

## 🧭 Getting Started

clone the repo
'''
git clone https://github.com/GITHUBsumesh/US_Airline.git
'''
clone the environment
'''
conda env create -f environment.yaml
'''
or 
create a new env
'''
conda create -n flight-full python==3.11 -y
conda activate flight-mlops
pip install -r requirements.txt
'''
start the fastapi server
'''
uvicorn backend.app:app --reload
'''
for predicting use
![test data](./ml/test_data/test_data.csv)

## Folder Description

### backend

'''
backend/
├── db/             # db config with SQLAlchemy
├── models/         # models to use postgres db
├── routes/         # routes for training and prediction
├── services/       # service files for training and prediction
|── templates/      # html templates to display the table of prediction or error
|── utils/          # file to app ml folder path in backend and to use all ml packages inside backend
'''

### ml

'''
ml/
├── src/                # main ml pipeline
├── data_schema/        # a schema.yaml file of the dataset
├── dataset/            # contains the dataset
├── final_model/        # the best model after training
|── notebooks/          # jupiter notebooks to look how data is present
|── test_data/          # test data to be used for prediction
|── prediction_output   # prediction output 
'''

### prisma_db

'''
prisma_db
|── prisma          # schema for db
'''

## Flow of execution

1. ml/push_data.py    -> pushes the entire raw data(400k rows) into db
2. backend/train.py   -> triggers the training pipeline
3. backend/predict.py -> triggers the prediction pipeline

## Detailed Training Pipeline

✅ Step 1: Data Ingestion
 > Data is read from db (where Target Column is not null) and stored in a csv "cleaned flights"
 > DataFrame is split into training and testing in ratio 0.2
 > A Data Ingestion Artifact is created 

✅ Step 2: Data Validation
 > Columns that are not required are removed from test.csv and train.csv 
 > Number of columns are validated
 > [Data Drift](https://www.datacamp.com/tutorial/understanding-data-drift-model-drift) report is created
 > A validated test.csv and train.csv is created
 > A Data Validation Artifact is created 

✅ Step 3: Data Transformation
 > Target Column is dropped from the test.csv and train .csv
 > For every model
    1.  A preprocessor is created by
        * Imputing missing rows 
            - most frequent for categorical columns
            - mean for numeric columns
        * Scaling numeric columns only
            - StandScaler 
        * Encoding Categorical Columns only
            - One Hot for carrier
            - Ordinal for airport
    2.  Preprocessor is fit and transformed on training data
        And only tranformed on test data
    3.  Two Numpy arrays are created 
            train_arr =  transformed train + Targer Column
            test_arr =  transformed test + Targer Column
    4.  A Data Transformation Artifact is created 

✅ Step 4: Model Training & Evaluation
 >  Every model is trained on different hyperparameters
    1. Catboost
        - As it is requires raw data, it is evaluated on validation files
        - Then fitted directly
    2. DNN
        - RepeatedKFold
    3. Linear
        - Directly fitted
    4. Others
        - RandomizedSearchCV to get best model then fitted on that model
 >  r2 score is calculated for all models and a report is made
 >  Highest r2 score is choosen as the best model
 >  MLFlow tracks the test metric of the best model
 >  Prediction is made and the best model and its preprocessor is saved in '/final_model'
 >  A Model Trainer Artifact is created 




## Detailed Prediction Pipeline

1.  The input csv is converted to pandas dataframe
2.  The final preprocessor and model objects are loaded into an AirLineModel class
3.  The predict function used the preprocessor to transform the data frame and then predict
4.  The prediction is stored in db and also in a csv file

