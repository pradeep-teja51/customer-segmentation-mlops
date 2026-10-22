from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_segment():
    payload = {"recency": 15, "frequency": 22, "monetary": 12000}
    response = client.post("/predict-segment", json=payload)
    assert response.status_code == 200
    assert response.json() == {"segment": 3}