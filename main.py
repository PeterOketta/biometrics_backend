from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cryptography.fernet import Fernet
import json
import logging
import os
from database import get_db
from models import User
from model_inference import ECGModelInference
from preprocess import preprocess_ecg_series, segment_signal
from schemas import EnrollmentData, ECGData
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()
model = ECGModelInference()
encryption_key = os.getenv("ENCRYPTION_KEY")
cipher = Fernet(encryption_key)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/enroll")
async def enroll(data: EnrollmentData, db: Session = Depends(get_db)):
    try:
        # Check if we have enough data
        if len(data.ecg_data) < 200:
            raise HTTPException(
                status_code=400,
                detail=f"ECG data too short. Need at least 200 samples, got {len(data.ecg_data)}",
            )

        # Preprocess the ECG signal
        processed_ecg = preprocess_ecg_series(data.ecg_data)

        # Segment into windows of 200 samples
        try:
            segments = segment_signal(processed_ecg, window_size=200, max_segments=40)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Generate embeddings for each segment
        context_vectors = []
        for segment in segments:
            try:
                embedding = model.generate_embedding(segment)
                context_vectors.append(embedding)
            except Exception as e:
                print(f"Error processing segment: {str(e)}")
                continue

        if not context_vectors:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any valid embeddings from the segments",
            )

        # Calculate mean embedding (iECG)
        iECG = np.mean(context_vectors, axis=0).tolist()
        iECG_json = json.dumps(iECG).encode("utf-8")
        encrypted_iECG = cipher.encrypt(iECG_json).decode("utf-8")
        # Save to database
        new_user = User(
            user_id=data.user_id,
            name=data.name,
            embedding=encrypted_iECG,
            created_at=datetime.utcnow(),
        )
        db.add(new_user)
        db.commit()

        return {
            "message": "Enrollment successful",
            "user_id": data.user_id,
            "name": data.name,
            "segments_processed": len(segments),
            "embeddings_generated": len(context_vectors),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/authenticate")
async def authenticate(data: ECGData, db: Session = Depends(get_db)):
    try:
        # Log incoming request for debugging
        logging.info(f"Received authentication request for ECG data of length {len(data.ecg_data)}")

        # Preprocess the ECG signal
        processed_ecg = preprocess_ecg_series(data.ecg_data)
        
        # Segment into windows of 200 samples
        segments = segment_signal(processed_ecg, window_size=200, max_segments=30)
        
        # Generate embeddings for each segment
        context_vectors = []
        for segment in segments:
            try:
                embedding = model.generate_embedding(segment)
                context_vectors.append(embedding)
            except Exception as e:
                logging.error(f"Error processing segment: {str(e)}")
                continue
        
        if not context_vectors:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
        
        # Calculate mean embedding
        incoming_embedding = np.mean(context_vectors, axis=0)
        
        # Compare with stored templates
        threshold = 0.4
        best_similarity = 0.0
        best_user_name = None
        
        users = db.query(User).all()
        for user in users:
            # Decrypt the stored embedding
            try:
                encrypted_embedding = user.embedding.encode('utf-8')
                stored_embedding_json = cipher.decrypt(encrypted_embedding).decode('utf-8')
                stored_embedding = np.array(eval(stored_embedding_json))
            except Exception as e:
                logging.error(f"Error decrypting or processing stored embedding for user {user.name}: {str(e)}")
                continue
            
            # Calculate similarity
            similarity_score = cosine_similarity([incoming_embedding], [stored_embedding])[0, 0]
            logging.info(f"Similarity score for user {user.name}: {similarity_score}")

            if similarity_score > best_similarity:
                best_similarity = similarity_score
                best_user_name = user.name
        
        if best_similarity >= threshold:
            return {
                "message": "Authentication successful",
                "name": best_user_name,
                "similarity_score": float(best_similarity)
            }
        else:
            raise HTTPException(
                status_code=401,
                detail=f"Authentication failed (best similarity: {best_similarity:.3f})"
            )
            
    except Exception as e:
        logging.error(f"An error occurred in the authenticate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))