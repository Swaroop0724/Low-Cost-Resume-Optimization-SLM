"""FastAPI backend — Resume Optimization SLM"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from app.model_inference import ResumeOptimizer
from app.text_extractor import extract_text_from_upload
from app.schema_validator import validate_resume_json
from app.quality_scorer import compute_quality_metrics

app = FastAPI(title="Resume Optimization SLM", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

optimizer = ResumeOptimizer()

class TextRequest(BaseModel):
    resume: str
    job_description: str

class OptimizeResponse(BaseModel):
    optimized_resume: dict
    schema_valid: bool
    quality_metrics: dict
    latency_seconds: float
    model: str = "Qwen3-4B + QLoRA r=16"

@app.get("/")
def root():
    return {"status": "online", "service": "Resume Optimization SLM", "version": "1.0.0"}

@app.get("/health")
def health():
    import torch
    return {
        "status": "healthy",
        "model_loaded": optimizer.is_loaded(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_used_gb": round(torch.cuda.memory_allocated()/1e9, 2) if torch.cuda.is_available() else None,
    }

@app.get("/schema")
def get_schema():
    return {"schema": {
        "personal_information": {"name": "str", "email": "str", "phone": "str", "location": "str", "socials": []},
        "summary": "str",
        "experiences": [{"designation": "str", "company": "str", "start_date": "str", "end_date": "str", "points": []}],
        "education": [{"degree": "str", "institution": "str", "start_date": "str", "end_date": "str", "gpa": "str"}],
        "skills": [{"name": "str", "data": []}],
        "projects": [{"name": "str", "description": "str", "technologies": []}],
        "certifications": [{"name": "str", "issuing_organization": "str"}],
        "awards": [], "languages": [], "publications": []
    }}

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_file(resume_file: UploadFile = File(...), job_description: str = ""):
    t0 = time.time()
    resume_text = await extract_text_from_upload(resume_file)
    if not resume_text.strip():
        raise HTTPException(400, "Could not extract text from resume file.")
    if not job_description.strip():
        raise HTTPException(400, "job_description is required.")
    raw = optimizer.generate(resume_text, job_description)
    parsed, valid = validate_resume_json(raw)
    metrics = compute_quality_metrics(raw, resume_text, job_description)
    return OptimizeResponse(optimized_resume=parsed, schema_valid=valid,
                            quality_metrics=metrics, latency_seconds=round(time.time()-t0, 2))

@app.post("/optimize/text", response_model=OptimizeResponse)
def optimize_text(req: TextRequest):
    t0 = time.time()
    if not req.resume.strip():
        raise HTTPException(400, "resume is required.")
    if not req.job_description.strip():
        raise HTTPException(400, "job_description is required.")
    raw = optimizer.generate(req.resume, req.job_description)
    parsed, valid = validate_resume_json(raw)
    metrics = compute_quality_metrics(raw, req.resume, req.job_description)
    return OptimizeResponse(optimized_resume=parsed, schema_valid=valid,
                            quality_metrics=metrics, latency_seconds=round(time.time()-t0, 2))
