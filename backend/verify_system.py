import requests
import os

BASE_URL = "http://localhost:8000"

def test_root():
    try:
        response = requests.get(BASE_URL + "/")
        print(f"Root endpoint: {response.status_code}")
        assert response.status_code == 200
    except Exception as e:
        print(f"Root endpoint failed: {e}")

def test_register_student():
    # Create a dummy image
    with open("test_face.jpg", "wb") as f:
        f.write(os.urandom(1024)) # Dummy content, won't work for real face detection but tests endpoint structure
    
    # We expect 400 because it's not a real face, but 500 means server error
    try:
        files = {'image': ('test_face.jpg', open('test_face.jpg', 'rb'), 'image/jpeg')}
        data = {'name': 'Test Student', 'student_id': '12345'}
        response = requests.post(BASE_URL + "/register", files=files, data=data)
        print(f"Register endpoint: {response.status_code} - {response.text}")
        # It should be 400 (no face) or 200 (if we used a real face)
        # If 500, something is wrong with code
    except Exception as e:
        print(f"Register endpoint failed: {e}")
    finally:
        if os.path.exists("test_face.jpg"):
            os.remove("test_face.jpg")

if __name__ == "__main__":
    print("Testing API...")
    test_root()
    test_register_student()
