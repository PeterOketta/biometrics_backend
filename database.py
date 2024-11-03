from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base  
import os

DATABASE_URL = "postgresql://postgres:postgres@db/postgres"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
Base.metadata.create_all(bind=engine)
