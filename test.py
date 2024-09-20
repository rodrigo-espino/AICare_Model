import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    'Cholesterol': 200,
    'Glucose': 90,
    'HDL Chol': 50,
    'Chol/HDL ratio': 4,
    'Age': 45,
    'Gender': 1,
    'Height': 175,
    'Weight': 80,
    'BMI': 26.1,
    'Systolic BP': 120,
    'Diastolic BP': 80,
    'waist': 85,
    'hip': 100,
    'Waist/hip ratio': 0.85
}

response = requests.post(url, json=data)
print(response.json())
