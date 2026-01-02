import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app

@pytest.mark.asyncio
async def test_predict_sentiment_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {"text": "I really love how this model works!"}

        response = await ac.post("/predict", json=payload)
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_predict_sentiment_empty_text():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as ac:
        payload = {"text": ""}
        response = await ac.post("/predict", json=payload)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_invalid_payload():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as ac:
        payload = {"not_text_field": "hello"}
        response = await ac.post("/predict", json=payload)

    assert response.status_code == 422