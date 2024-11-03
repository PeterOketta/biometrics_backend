# schemas.py
from pydantic import BaseModel
from typing import List

class EnrollmentData(BaseModel):
    user_id: str
    name: str 
    ecg_data: List[float]

class ECGData(BaseModel):
    ecg_data: List[float]  # Only ECG data is required for authentication
