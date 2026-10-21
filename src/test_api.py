import requests

response = requests.post(
    "http://127.0.0.1:8000/predict-segment",
    json={"recency": 15, "frequency": 22, "monetary": 12000}
)

print(response.json())