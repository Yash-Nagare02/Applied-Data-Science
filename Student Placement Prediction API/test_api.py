import requests
import json

# Test data - sample student details
test_data = {
    "gender": "M",
    "ssc_p": 85.5,
    "ssc_b": "Central",
    "hsc_p": 78.2,
    "hsc_b": "Central",
    "hsc_s": "Commerce",
    "degree_p": 75.8,
    "degree_t": "Comm&Mgmt",
    "workex": "Yes",
    "etest_p": 82.3,
    "specialisation": "Mkt&Fin",
    "mba_p": 68.9
}

# API endpoint
url = "http://localhost:5000/predict"

# Make prediction request
try:
    response = requests.post(url, json=test_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ API Test Successful!")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
except requests.exceptions.ConnectionError:
    print("❌ Connection Error: Make sure the Flask API is running (python app.py)")
except Exception as e:
    print(f"❌ Test failed: {e}")