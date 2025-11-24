from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from datetime import datetime
import json
import cv2
import numpy as np
import pickle

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
if os.path.exists("../frontend"):
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Directories
UPLOAD_DIR = "uploaded_faces"
ATTENDANCE_FILE = "attendance.json"
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.json"
STUDENTS_FILE = "students.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

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

def load_students():
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_students(students):
    with open(STUDENTS_FILE, 'w') as f:
        json.dump(students, f, indent=2)

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

def train_model():
    """Train the LBPH recognizer on all registered faces"""
    print("Training model...")
    faces = []
    ids = []
    label_map = {}
    current_id = 0
    
    # Check if we have any students
    if not os.path.exists(UPLOAD_DIR):
        print("No upload directory found.")
        return False

    student_dirs = [d for d in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, d))]
    
    if not student_dirs:
        print("No students found.")
        return False

    for student_id in student_dirs:
        label_map[str(current_id)] = student_id
        student_path = os.path.join(UPLOAD_DIR, student_id)
        
        for filename in os.listdir(student_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(student_path, filename)
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Detect faces
                detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                
                for (x, y, w, h) in detected_faces:
                    faces.append(img[y:y+h, x:x+w])
                    ids.append(current_id)
        
        current_id += 1

    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_FILE)
        with open(LABELS_FILE, 'w') as f:
            json.dump(label_map, f)
        print("Model trained successfully.")
        return True
    else:
        print("No faces found to train.")
        return False

def load_model():
    """Load the trained model and labels"""
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
        recognizer.read(MODEL_FILE)
        with open(LABELS_FILE, 'r') as f:
            label_map = json.load(f)
        # Convert keys back to int for internal usage if needed, but we look up by int ID
        return label_map
    return None

# Initialize model on startup if exists
load_model()

# API Endpoints
@app.get("/")
async def root():
    if os.path.exists("../frontend/index.html"):
        return FileResponse("../frontend/index.html")
    return {"message": "Facial Recognition Attendance System API (OpenCV LBPH)"}

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
        
        # Verify face is detectable before saving
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
             raise HTTPException(status_code=400, detail="No face detected. Please upload a clear photo.")
        
        # Save image
        filename = f"registered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = save_image(contents, student_id, filename)
        
        # Save student metadata
        students = load_students()
        students[student_id] = {
            "name": name,
            "registered_at": datetime.now().isoformat()
        }
        save_students(students)
        
        # Retrain model
        train_model()
        
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
        
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Detect face
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")
        
        # Load model and labels
        label_map = load_model()
        if not label_map:
             raise HTTPException(status_code=400, detail="System not trained yet. Please register students first.")
        
        # Predict
        # We take the largest face if multiple
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        roi_gray = img_gray[y:y+h, x:x+w]
        
        label_id, confidence = recognizer.predict(roi_gray)
        
        # Confidence in LBPH is distance: 0 is perfect match.
        # Usually < 50 is good, < 80 is acceptable.
        # We'll use a threshold of 70 for now.
        THRESHOLD = 70
        
        if confidence < THRESHOLD:
            student_id = label_map.get(str(label_id))
            if student_id:
                # Get student name
                students = load_students()
                student_name = students.get(student_id, {}).get("name", f"Student {student_id}")
                
                # Log attendance
                records = load_attendance()
                timestamp = datetime.now().isoformat()
                
                attendance_record = {
                    "student_id": student_id,
                    "timestamp": timestamp,
                    "confidence": float(confidence), # Lower is better
                    "verified": True,
                    "name": student_name
                }
                
                records.append(attendance_record)
                save_attendance(records)
                
                return JSONResponse(content={
                    "success": True,
                    "student_id": student_id,
                    "timestamp": timestamp,
                    "confidence": float(confidence),
                    "message": f"Welcome, {student_name}!"
                })
        
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Face not recognized. Please register first or try again."
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
        
        students_data = load_students()
        student_dirs = [d for d in os.listdir(UPLOAD_DIR) 
                       if os.path.isdir(os.path.join(UPLOAD_DIR, d))]
        
        students = []
        for student_id in student_dirs:
            image_path = get_student_image_path(student_id)
            if image_path:
                student_info = students_data.get(student_id, {})
                students.append({
                    "student_id": student_id,
                    "name": student_info.get("name", "Unknown"),
                    "image_path": image_path,
                    "registered_date": student_info.get("registered_at", datetime.fromtimestamp(
                        os.path.getctime(image_path)
                    ).isoformat())
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
            # Retrain model after deletion
            train_model()
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