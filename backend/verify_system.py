import os
import time
import requests
import subprocess
import sys

# Configuration
BASE_URL = "http://localhost:8001"
TEST_IMAGE_PATH = r"c:\Users\ahmad\Documents\GitHub\facial-recognition-attendance\backend\uploaded_faces\S25CSEU1058\registered_20251124_213157.jpg"
STUDENT_ID = "TEST_STUDENT_001"
STUDENT_NAME = "Test Student"

def start_server():
    print("Starting backend server...")
    # Start uvicorn on port 8001
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(5)  # Wait for server to start
    return process

def stop_server(process):
    print("Stopping backend server...")
    process.terminate()
    process.wait()

def test_registration():
    print("\nTesting Registration...")
    url = f"{BASE_URL}/register"
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        return False

    with open(TEST_IMAGE_PATH, "rb") as f:
        files = {"image": ("test.jpg", f, "image/jpeg")}
        data = {"name": STUDENT_NAME, "student_id": STUDENT_ID}
        try:
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                print("Registration Successful:", response.json())
                return True
            else:
                print("Registration Failed:", response.text)
                return False
        except Exception as e:
            print(f"Registration Error: {e}")
            return False

def test_attendance():
    print("\nTesting Attendance Marking...")
    url = f"{BASE_URL}/mark-attendance"
    
    with open(TEST_IMAGE_PATH, "rb") as f:
        files = {"image": ("test.jpg", f, "image/jpeg")}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print("Attendance Marked Successfully:", response.json())
                return True
            else:
                print("Attendance Marking Failed:", response.text)
                return False
        except Exception as e:
            print(f"Attendance Error: {e}")
            return False

def test_get_attendance():
    print("\nTesting Get Attendance Logs...")
    url = f"{BASE_URL}/attendance"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"Retrieved {data['total']} records.")
            # Verify our student is in there
            found = False
            for record in data['records']:
                if record['student_id'] == STUDENT_ID:
                    found = True
                    break
            
            if found:
                print("Test student record found in logs.")
                return True
            else:
                print("Test student record NOT found in logs.")
                return False
        else:
            print("Get Attendance Failed:", response.text)
            return False
    except Exception as e:
        print(f"Get Attendance Error: {e}")
        return False

def test_cleanup():
    print("\nCleaning up...")
    url = f"{BASE_URL}/student/{STUDENT_ID}"
    try:
        requests.delete(url)
        print("Test student deleted.")
    except:
        pass

def main():
    server_process = start_server()
    try:
        if test_registration():
            time.sleep(2) # Wait for training if needed
            if test_attendance():
                test_get_attendance()
        
        test_cleanup()
    finally:
        stop_server(server_process)

if __name__ == "__main__":
    main()
