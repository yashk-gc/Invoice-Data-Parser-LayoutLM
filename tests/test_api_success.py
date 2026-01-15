from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_extract_invoice_api():
    with open("data/raw/sample_page_0.jpg", "rb") as f:
        response = client.post(
            "/extract-invoice",
            files={"file": ("invoice.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    assert "extracted_fields" in response.json()
