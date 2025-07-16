import os
import pandas as pd
import psycopg2
import yaml
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

RAW_CSV_PATH = "dataset/raw/Airline_Delay_Cause_Smaller.csv"
SCHEMA_PATH = "data_schema/schema.yaml"
TABLE_NAME = "US_Airline"
DATABASE_URL = os.getenv("DATABASE_URL")

PYTHON_TYPE_MAP = {
    "int64": int,
    "float64": float,
    "object": str
}

def load_schema(schema_path: str):
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    expected_types = {}
    for col_def in schema["columns"]:
        for col_name, col_type in col_def.items():
            py_type = PYTHON_TYPE_MAP.get(col_type)
            if py_type is None:
                raise ValueError(f"Unsupported type: {col_type}")
            expected_types[col_name] = py_type

    return expected_types

def validate_raw_csv(csv_path: str, expected_types: dict) -> pd.DataFrame:
    # ✅ Properly read CSV with quoted strings
    df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
    print(f"📊 Raw shape: {df.shape}")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    for col, dtype in expected_types.items():
        if col not in df.columns:
            print(f"⚠️ Missing column: {col}")
            continue

        if dtype in (int, float):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == str:
            df[col] = df[col].astype(str).str.strip()
    return df

def bulk_insert_postgres(df: pd.DataFrame, table_name: str):
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        cur = conn.cursor()

        # Convert DataFrame to CSV in memory
        buffer = StringIO()
        expected_order = [
        "year", "month", "carrier", "carrier_name", "airport", "airport_name",
        "arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct", "security_ct",
        "late_aircraft_ct", "arr_cancelled", "arr_diverted", "arr_delay",
        "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"
        ]
        df = df[expected_order]
        df.to_csv(buffer, index=False)
        
        buffer.seek(0)

        # Print preview
        print("🔍 Preview of CSV being loaded:")
        preview = "\n".join(buffer.getvalue().splitlines()[:3])
        print(preview)
        buffer.seek(0)

        cur.copy_expert(
            sql=f'''
            COPY "{table_name}" (
                year, month, carrier, carrier_name, airport, airport_name,
                arr_flights, arr_del15, carrier_ct, weather_ct, nas_ct, security_ct,
                late_aircraft_ct, arr_cancelled, arr_diverted, arr_delay,
                carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay
            ) FROM STDIN WITH CSV HEADER
            ''',
            file=buffer
        )

        conn.commit()
        print(f"🚀 Successfully inserted data into '{table_name}'")
    except Exception as e:
        print(f"❌ Load failed: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    expected_types = load_schema(SCHEMA_PATH)
    df_clean = validate_raw_csv(RAW_CSV_PATH, expected_types)
    bulk_insert_postgres(df_clean, TABLE_NAME)
