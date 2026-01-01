import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_predict_sentiment_success():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {"text": "I really love how this model works!"}
        response = await ac.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert "sentiment" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


@pytest.mark.asyncio
async def test_predict_sentiment_empty_text():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {"text": ""}
        response = await ac.post("/predict", json=payload)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_invalid_payload():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {"not_text_field": "hello"}
        response = await ac.post("/predict", json=payload)

    assert response.status_code == 422