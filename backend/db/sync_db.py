import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# psycopg2 expects 'postgres://' or 'postgresql://'
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

conn = psycopg2.connect(DATABASE_URL)
