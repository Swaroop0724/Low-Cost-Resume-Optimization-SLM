# 🎯 Low-Cost Resume Optimization via Distillation of Large Language Model Behavior into a Fine-Tuned SLM

> DTSC 5082 · Seminar in Research & Research Methodology · Group 9  
> Professor: Clifford K. Whitworth

---

## 🌐 Live Demo

| Page | Link |
|---|---|
| 🚀 **ResumeAI — Live Demo** | [Open Demo](https://swaroop0724.github.io/Low-Cost-Resume-Optimization-SLM/ResumeAI.html) |
| 📋 **Model Deployment Writeup** | [Open Page](https://swaroop0724.github.io/Low-Cost-Resume-Optimization-SLM/Model_Deployment.html) |
| 🗂️ **JSON Schema Viewer** | [Open Page](https://swaroop0724.github.io/Low-Cost-Resume-Optimization-SLM/resume_json_schema.html) |

---

## 📌 Overview

This project builds a **production-ready resume optimization system** powered by a **Qwen3-4B Small Language Model (SLM)** fine-tuned with **QLoRA**. It takes a resume + job description as input and returns a tailored, structured JSON resume — at **94%+ lower cost** than GPT-4 API.

The key idea: instead of paying for GPT-4 on every request, we used **Gemini 2.5 Flash Batch API** as a cheap teacher model to generate 1,530 training labels, then distilled that knowledge into a small model that runs locally for near-zero cost.

---

## 🗂️ Project Structure

```
resume-optimizer/
├── notebooks/
│   ├── pre_processing.ipynb     ← Phase 1: resume extraction + job scraping + Gemini batch
│   ├── eda.ipynb                ← Phase 2: EDA — ROUGE-L, lexical diversity, outliers
│   └── tuning.ipynb             ← Phase 3: QLoRA fine-tuning + evaluation ← RUN THIS
│
├── files/
│   ├── job-scraper/
│   │   ├── cleaned_scraped_jobs.csv   ← 249 job descriptions (19 categories)
│   │   └── job_links_merged.csv       ← job metadata
│   └── dataset/
│       ├── extracted-resume-data/     ← resume_text.jsonl (2,128 resumes)
│       ├── batch_requests/            ← 4 Gemini batch input files
│       ├── batch_results/             ← 2 Gemini batch output files
│       ├── resume-data-with-job-description/
│       │   └── final_resume_dataset.parquet  ← 1,530 rows (resume + JD + Gemini label)
│       └── final-training-data/
│           └── final_training_ready.jsonl    ← 1,482 clean training examples ← USE THIS
│
├── app/
│   ├── main.py              ← FastAPI server (5 endpoints)
│   ├── model_inference.py   ← Qwen3-4B + LoRA adapter inference
│   ├── text_extractor.py    ← PDF/DOCX → plain text
│   ├── schema_validator.py  ← JSON output validation
│   └── quality_scorer.py    ← skill coverage + hallucination detection
│
├── docs/
│   ├── ResumeAI.html            ← Live demo frontend
│   ├── Model_Deployment.html    ← Deployment writeup
│   └── resume_json_schema.html  ← JSON schema viewer
│
├── tests/test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

```
1,800+ Resume PDFs/DOCX
        ↓ pre_processing.ipynb
resume_text.jsonl (2,128 resumes)
        ↓
Paired with 249 scraped job descriptions
        ↓
Gemini 2.5 Flash Batch API → tailored resume JSON (training labels)
        ↓
final_resume_dataset.parquet (1,530 rows)
        ↓
final_training_ready.jsonl (1,482 clean instruction-tuning examples)
        ↓
QLoRA fine-tune Qwen3-4B on T4 GPU (~1 hour)
        ↓
Fine-tuned model → ResumeAI.html live demo
```

---

## 🚀 How to Run

### Option 1 — Use the live demo (no setup needed)
👉 [Open ResumeAI Demo](https://swaroop0724.github.io/Low-Cost-Resume-Optimization-SLM/ResumeAI.html)
- Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)
- Upload your resume (PDF/DOCX/TXT)
- Paste a job description
- Click Run → get tailored resume in seconds

### Option 2 — Fine-tune the model yourself (Google Colab T4 GPU)
1. Upload project folder to Google Drive
2. Open `notebooks/tuning.ipynb` in Colab
3. Set runtime to **T4 GPU**
4. Add `HF_TOKEN` in Colab Secrets
5. Update `PROJECT_ROOT` path to your Drive folder
6. Run all cells (~1 hour)

### Option 3 — Run the API locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API endpoints:
- `GET /` — health check
- `GET /health` — GPU status
- `GET /schema` — JSON output schema
- `POST /optimize` — upload PDF resume
- `POST /optimize/text` — send resume as text

---

## 📊 Dataset Summary

| File | Rows | Description |
|---|---|---|
| resume_text.jsonl | 2,128 | Raw extracted resume texts (PDF + DOCX) |
| final_resume_dataset.parquet | 1,530 | Resume + JD + Gemini tailored resume |
| final_training_ready.jsonl | 1,482 | Clean instruction-tuning format |

📦 **Dataset Download:** [Google Drive](https://drive.google.com/drive/folders/1JGO8wLvreC-R33vcZxXMGV1vS62LvyfV?usp=sharing) *(too large for GitHub)*

---

## 🧠 Model Results

| Metric | Base Model | Fine-tuned |
|---|---|---|
| ROUGE-L | 0.41 | 0.68 |
| Improvement | — | **+65.9%** |
| VRAM required | 28 GB (fp32) | 6.2 GB (4-bit) |
| Memory saved | — | **77.9%** |
| Cost vs GPT-4 | — | **90% cheaper** |

---

## 💰 Cost Comparison

| Solution | Monthly Cost (1K users) | Latency |
|---|---|---|
| GPT-4o API | ~$8,000/mo | 3–8s |
| ResumeAI SLM (ours) | ~$800/mo | <120ms |

**Annual savings: $86,400**

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `QLoRA` · `PEFT` · `TRL` · `Qwen3-4B` · `Gemini 2.5 Flash` · `FastAPI` · `Selenium` · `Docker` · `Groq API`

---

## 👥 Team — Group 9

| Name | Contribution |
|---|---|
| Vaishnav Busha | Backend architecture, FastAPI, data pipeline |
| Swaroop | Job scraping, Gemini batch API, EDA |
| Nishith | QLoRA fine-tuning, training dataset |
| Talha Khan | Validation, quality reporting, documentation |

---

## 📜 License

MIT License
