# models.py
from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    embedding = Column(String)  # Store iECG embedding as a serialized string
    created_at = Column(DateTime, default=datetime.now)
