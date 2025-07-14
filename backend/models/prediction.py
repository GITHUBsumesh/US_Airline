from sqlalchemy import Column, Integer, DateTime, JSON
from sqlalchemy.sql import func
from backend.db.async_db import Base

class Prediction(Base):
    __tablename__ = "Prediction"  # Match Prisma schema if used

    id = Column(Integer, primary_key=True, index=True)
    predictedValue = Column(Integer, nullable=False)
    inputFeatures = Column(JSON, nullable=True)
    createdAt = Column(DateTime(timezone=True), server_default=func.now())
