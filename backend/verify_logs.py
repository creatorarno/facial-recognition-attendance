import requests
import json

try:
    response = requests.get("http://localhost:8000/attendance")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response Keys: {list(data.keys())}")
    print(f"Records Type: {type(data.get('records'))}")
except Exception as e:
    print(f"Error: {e}")
