"""Live API endpoint test verifying POST /api/v1/analyze against schema."""

import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

def test_live_analyze_endpoint():
    client = TestClient(app)

    # Generate test image
    img = Image.new("RGB", (400, 300), color=(120, 180, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("sample_test.jpg", buf, "image/jpeg")}
    )

    print("\n" + "=" * 60)
    print("LIVE FASTAPI POST /api/v1/analyze RESPONSE STATUS:", response.status_code)
    print("RESPONSE JSON PAYLOAD:")
    data = response.json()
    print(data)
    print("=" * 60 + "\n")

    assert response.status_code == 200
    assert data["status"] == "success"
    assert "analysis_id" in data
    assert "verdict" in data
    assert "probabilities" in data
    assert "metadata" in data
    assert data["verdict"]["label"] in ["Real", "AI-generated"]
    assert 0.0 <= data["verdict"]["confidence"] <= 1.0
    assert abs(data["probabilities"]["real"] + data["probabilities"]["ai_generated"] - 1.0) < 0.01

if __name__ == "__main__":
    test_live_analyze_endpoint()
