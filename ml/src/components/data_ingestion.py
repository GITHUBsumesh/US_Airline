from ml.src.utils.path_config import ML_ROOT
from src.exception.exception import AirLineException
from src.logging.logger import logging
from src.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME

## configuration of the Data Ingestion Config
import psycopg2
from io import StringIO
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise AirLineException(e,sys)
        
        

    def read_postgres_fast_exclude_id(self,table_name):
        """
        Read data from postgres database excluding the 'id' column.
        """
        try:
            conn = psycopg2.connect(DATABASE_URL, sslmode='require')
            cur = conn.cursor()

            buffer = StringIO()

            # List all columns except 'id'
            cur.execute(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s AND column_name != 'id'
                ORDER BY ordinal_position;
            """, (table_name,))
            columns = [row[0] for row in cur.fetchall()]
            column_list = ', '.join(columns)

            # COPY selected columns only
            copy_sql = f'COPY (SELECT {column_list} FROM "{table_name}") TO STDOUT WITH CSV HEADER'
            cur.copy_expert(copy_sql, buffer)
            buffer.seek(0)

            df = pd.read_csv(buffer)
            print(f"✅ Read {len(df)} rows from '{table_name}' (excluding 'id')")
            return df

        except Exception as e:
            print(f"[DATA_INGESTION ERROR] {e}") 
            raise AirLineException(e,sys)
        finally:
            cur.close()
            conn.close()
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise AirLineException(e,sys)
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")

            
        except Exception as e:
            raise AirLineException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe=self.read_postgres_fast_exclude_id(DATA_INGESTION_COLLECTION_NAME)
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise AirLineException("Failed to load data", sys)
