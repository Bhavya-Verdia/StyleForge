import pytest
from httpx import AsyncClient, ASGITransport
import sys
import os

# Add the server directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set mock mode for tests
os.environ["USE_MOCK"] = "true"

from main import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mock_mode"] is True

@pytest.mark.asyncio
async def test_metrics_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "eval_samples" in data

@pytest.mark.asyncio
async def test_adapters_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/adapters")
    assert response.status_code == 200
    data = response.json()
    assert "adapters" in data
    assert "active" in data

@pytest.mark.asyncio
async def test_generate_base_mock():
    # Test the mock generation stream
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/generate/base", json={
            "prompt": "test prompt",
            "max_tokens": 100,
            "temperature": 0.7
        })
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # Just checking it returns stream data
    content = response.text
    assert "data: " in content
    assert "done" in content
