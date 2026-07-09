from io import BytesIO

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_upload_score_accepts_semicolon_delimited_realistic_columns():
    csv_content = """name;phone;severity;wait_mins;vulnerable;vip;category
Alice;1234567890;4;10;yes;no;billing
Bob;0987654321;2;0;no;yes;technical
"""

    response = client.post(
        "/upload-score",
        files={"file": ("calls.csv", BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["rows"][0]["priority_bucket"] in {"High", "Medium", "Low"}
    assert body["rows"][0]["name"] == "Alice"
    assert body["rows"][1]["category"] == "technical"
