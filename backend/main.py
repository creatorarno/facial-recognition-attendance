from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from datetime import datetime
import json
from deepface import DeepFace
import cv2
import numpy as np
from pathlib import Path

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploaded_faces"
ATTENDANCE_FILE = "attendance.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Models
class StudentRegistration(BaseModel):
    name: str
    student_id: str

class AttendanceRecord(BaseModel):
    name: str
    student_id: str
    timestamp: str
    image_path: str

# Helper functions
def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_attendance(records):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(records, f, indent=2)

def get_student_image_path(student_id):
    """Get the path to a student's registered image"""
    student_dir = os.path.join(UPLOAD_DIR, student_id)
    if os.path.exists(student_dir):
        images = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            return os.path.join(student_dir, images[0])
    return None

def save_image(file_content: bytes, student_id: str, filename: str) -> str:
    """Save uploaded image to student directory"""
    student_dir = os.path.join(UPLOAD_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)
    
    filepath = os.path.join(student_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(file_content)
    
    return filepath

def verify_face_with_deepface(img1_path: str, img2_path: str, threshold: float = 0.6) -> dict:
    """
    Verify if two images contain the same person using DeepFace
    Returns: dict with 'verified' (bool) and 'distance' (float)
    """
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name='Facenet512',  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib
            detector_backend='opencv',  # Options: opencv, ssd, dlib, mtcnn, retinaface
            distance_metric='cosine',  # Options: cosine, euclidean, euclidean_l2
            enforce_detection=True
        )
        
        # Adjust verification based on custom threshold
        result['verified'] = result['distance'] < threshold
        return result
        
    except Exception as e:
        print(f"Face verification error: {str(e)}")
        return {'verified': False, 'distance': 1.0, 'error': str(e)}

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Facial Recognition Attendance System API (DeepFace)"}

@app.post("/register")
async def register_student(
    name: str = File(...),
    student_id: str = File(...),
    image: UploadFile = File(...)
):
    """Register a new student with their face image"""
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image content
        contents = await image.read()
        
        # Save image
        filename = f"registered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = save_image(contents, student_id, filename)
        
        # Verify face is detectable
        try:
            # Try to detect face in the image
            DeepFace.extract_faces(
                img_path=filepath,
                detector_backend='opencv',
                enforce_detection=True
            )
        except Exception as e:
            # Clean up if face detection fails
            os.remove(filepath)
            raise HTTPException(
                status_code=400,
                detail=f"No face detected in image. Please upload a clear photo. Error: {str(e)}"
            )
        
        return JSONResponse(content={
            "message": "Student registered successfully",
            "student_id": student_id,
            "name": name,
            "image_path": filepath
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/mark-attendance")
async def mark_attendance(image: UploadFile = File(...)):
    """Mark attendance by comparing uploaded image with registered students"""
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save temporary image
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        
        contents = await image.read()
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        # Try to detect face in uploaded image
        try:
            DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend='opencv',
                enforce_detection=True
            )
        except Exception as e:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"No face detected in uploaded image. Error: {str(e)}"
            )
        
        # Get all registered students
        if not os.path.exists(UPLOAD_DIR):
            os.remove(temp_path)
            raise HTTPException(status_code=404, detail="No students registered yet")
        
        student_dirs = [d for d in os.listdir(UPLOAD_DIR) 
                       if os.path.isdir(os.path.join(UPLOAD_DIR, d))]
        
        if not student_dirs:
            os.remove(temp_path)
            raise HTTPException(status_code=404, detail="No students registered yet")
        
        # Try to match with each registered student
        best_match = None
        best_distance = float('inf')
        
        for student_id in student_dirs:
            registered_image = get_student_image_path(student_id)
            if not registered_image:
                continue
            
            result = verify_face_with_deepface(temp_path, registered_image)
            
            if result['verified'] and result['distance'] < best_distance:
                best_distance = result['distance']
                best_match = student_id
        
        # Clean up temp file
        os.remove(temp_path)
        
        if best_match:
            # Load attendance records
            records = load_attendance()
            
            # Create attendance record
            timestamp = datetime.now().isoformat()
            attendance_record = {
                "student_id": best_match,
                "timestamp": timestamp,
                "confidence": float(1 - best_distance),  # Convert distance to confidence
                "verified": True
            }
            
            records.append(attendance_record)
            save_attendance(records)
            
            return JSONResponse(content={
                "success": True,
                "student_id": best_match,
                "timestamp": timestamp,
                "confidence": float(1 - best_distance),
                "message": f"Attendance marked for student {best_match}"
            })
        else:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "No matching student found. Please register first."
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attendance marking failed: {str(e)}")

@app.get("/attendance")
async def get_attendance():
    """Get all attendance records"""
    try:
        records = load_attendance()
        return JSONResponse(content={"records": records, "total": len(records)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch attendance: {str(e)}")

@app.get("/students")
async def get_students():
    """Get list of all registered students"""
    try:
        if not os.path.exists(UPLOAD_DIR):
            return JSONResponse(content={"students": [], "total": 0})
        
        student_dirs = [d for d in os.listdir(UPLOAD_DIR) 
                       if os.path.isdir(os.path.join(UPLOAD_DIR, d))]
        
        students = []
        for student_id in student_dirs:
            image_path = get_student_image_path(student_id)
            if image_path:
                students.append({
                    "student_id": student_id,
                    "image_path": image_path,
                    "registered_date": datetime.fromtimestamp(
                        os.path.getctime(image_path)
                    ).isoformat()
                })
        
        return JSONResponse(content={"students": students, "total": len(students)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch students: {str(e)}")

@app.delete("/student/{student_id}")
async def delete_student(student_id: str):
    """Delete a registered student"""
    try:
        student_dir = os.path.join(UPLOAD_DIR, student_id)
        if os.path.exists(student_dir):
            shutil.rmtree(student_dir)
            return JSONResponse(content={
                "success": True,
                "message": f"Student {student_id} deleted successfully"
            })
        else:
            raise HTTPException(status_code=404, detail="Student not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete student: {str(e)}")

@app.delete("/attendance")
async def clear_attendance():
    """Clear all attendance records"""
    try:
        save_attendance([])
        return JSONResponse(content={
            "success": True,
            "message": "All attendance records cleared"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear attendance: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)