import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import yaml

load_dotenv()

# Paths
RAW_CSV_PATH = "dataset/raw/Airline_Delay_Cause_Smaller.csv"
CLEAN_CSV_PATH = "dataset/processed/cleaned_flights.csv"
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

    return expected_types, schema.get("numerical_columns", [])

def validate_and_clean(csv_path: str, expected_types: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"📊 Original shape: {df.shape}")

    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    for col, dtype in expected_types.items():
        if col not in df.columns:
            print(f"⚠️ Missing column: {col}")
            continue

        if dtype == int:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
        elif dtype == float:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype == str:
            df[col] = df[col].astype(str).str.strip()

    numeric_cols = [col for col, typ in expected_types.items() if typ in (int, float)]
    before = df.shape[0]
    df.dropna(subset=numeric_cols, inplace=True)
    after = df.shape[0]
    print(f"🧹 Dropped {before - after} rows with invalid numeric values.")
    return df

def export_clean_csv(df: pd.DataFrame, out_path: str):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save the cleaned CSV
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📁 Cleaned CSV saved to: {out_path} — {len(df)} rows")
    
def preview_csv(csv_path: str):
    print(f"\n🔍 Previewing first 5 lines of: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        for i in range(5):
            print(f.readline().strip())
    print("✅ Preview complete.\n")

def bulk_insert_postgres(csv_path: str, table_name: str):
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        cur = conn.cursor()

        preview_csv(csv_path)

        with open(csv_path, 'r', encoding='utf-8') as f:
            cur.copy_expert(
                sql=f'''
                COPY "{table_name}" (
                    year, month, carrier, carrier_name, airport, airport_name,
                    arr_flights, arr_del15, carrier_ct, weather_ct, nas_ct, security_ct,
                    late_aircraft_ct, arr_cancelled, arr_diverted, arr_delay,
                    carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay
                ) FROM STDIN WITH CSV HEADER
                ''',
                file=f
            )

        conn.commit()
        print(f"🚀 Successfully loaded {csv_path} into '{table_name}'")
    except Exception as e:
        print(f"❌ Load failed: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    EXPECTED_TYPES, numerical_columns = load_schema(SCHEMA_PATH)
    df_clean = validate_and_clean(RAW_CSV_PATH, EXPECTED_TYPES)
    export_clean_csv(df_clean, CLEAN_CSV_PATH)
    bulk_insert_postgres(CLEAN_CSV_PATH, TABLE_NAME)
