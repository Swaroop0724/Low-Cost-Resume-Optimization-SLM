"""API tests — run with: pytest tests/test_api.py -v"""
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

RESUME = "Jane Smith | jane@email.com | Python Engineer, 3 years experience. Django, PostgreSQL, Docker."
JD     = "Senior Backend Engineer — Python, FastAPI, PostgreSQL, AWS. 3+ years required."

def test_root():        assert client.get("/").json()["status"] == "online"
def test_health():      assert "model_loaded" in client.get("/health").json()
def test_schema():      assert "schema" in client.get("/schema").json()
def test_no_resume():   assert client.post("/optimize/text", json={"resume":"","job_description":JD}).status_code == 400
def test_no_jd():       assert client.post("/optimize/text", json={"resume":RESUME,"job_description":""}).status_code == 400
def test_optimize():
    r = client.post("/optimize/text", json={"resume":RESUME,"job_description":JD})
    assert r.status_code == 200
    d = r.json()
    assert "optimized_resume" in d and "schema_valid" in d and "latency_seconds" in d
