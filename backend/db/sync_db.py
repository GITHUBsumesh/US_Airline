from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv
load_dotenv()

raw_url = os.getenv("DATABASE_URL")
if not raw_url:
    raise ValueError("DATABASE_URL is not set.")

DATABASE_URL = raw_url

# Create a synchronous SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)

# Session maker for sync sessions
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Declare base for models
Base = declarative_base()
