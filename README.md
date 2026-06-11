# 🎯 Low-Cost Resume Optimization via Distillation of Large Language Model Behavior into a Fine-Tuned SLM

> DTSC 5082 · Seminar in Research & Research Methodology · Group 9  
> Professor: Clifford K. Whitworth

---

## 📌 Overview
Fine-tune **Qwen3-4B** with **QLoRA** to tailor resumes to job descriptions at 94%+ lower cost than GPT-4.

---

## 🗂️ Project Structure
```
resume-optimizer/
├── notebooks/
│   ├── pre_processing.ipynb              ← Phase 1: data pipeline (DONE)
│   ├── eda.ipynb                         ← Phase 2: EDA (DONE)
│   └── resume_optimizer_finetuning.ipynb ← Phase 3: QLoRA training ← RUN THIS
│
├── files/
│   ├── job-scraper/
│   │   ├── cleaned_scraped_jobs.csv      ← 249 job descriptions
│   │   └── job_links_merged.csv          ← job metadata
│   └── dataset/
│       ├── extracted-resume-data/        ← resume_text.jsonl (2128 resumes)
│       ├── batch_requests/               ← 4 Gemini batch input files
│       ├── batch_results/                ← 2 Gemini batch output files
│       ├── resume-data-with-job-description/
│       │   ├── final_resume_dataset.parquet  ← 1530 rows (resume+JD+label)
│       │   └── final_resume_dataset.jsonl
│       └── final-training-data/
│           ├── final_training_dataset.jsonl         ← raw training data
│           ├── final_training_dataset_cleaned.jsonl ← cleaned
│           └── final_training_ready.jsonl           ← USE THIS (1482 valid rows)
│
├── app/
│   ├── main.py              ← FastAPI server
│   ├── model_inference.py   ← Qwen3-4B + LoRA inference
│   ├── text_extractor.py    ← PDF/DOCX → text
│   ├── schema_validator.py  ← JSON validation
│   └── quality_scorer.py    ← ROUGE-L + hallucination
│
├── docs/
│   ├── ResumeAI.html           ← Live demo frontend
│   ├── Model_Deployment.html   ← Deployment writeup
│   └── resume_json_schema.html ← JSON schema viewer
│
├── tests/test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Fine-tune (Google Colab T4 GPU)
1. Upload this folder to Google Drive
2. Open `notebooks/resume_optimizer_finetuning.ipynb` in Colab
3. Set runtime to T4 GPU
4. Add `HF_TOKEN` in Colab Secrets
5. Update `PROJECT_ROOT` path to match your Drive folder
6. Run all cells (~1 hour)

### Step 2 — Run the API
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Step 3 — Use the frontend
Open `docs/ResumeAI.html` in browser → add Groq API key → upload resume → run.

---

## 📊 Dataset Summary
| File | Rows | Description |
|---|---|---|
| resume_text.jsonl | 2,128 | Raw extracted resume texts |
| final_resume_dataset.parquet | 1,530 | Resume + JD + Gemini tailored resume |
| final_training_ready.jsonl | 1,482 | Clean instruction-tuning format |

---

## 🧠 Model Results
| Metric | Base Model | Fine-tuned |
|---|---|---|
| ROUGE-L | 0.41 | 0.68 |
| Improvement | — | +65.9% |
| VRAM | 28 GB (fp32) | 6.2 GB (4-bit) |
| Memory saved | — | 77.9% |

---

## 📦 Dataset

Data files are stored on Google Drive (too large for GitHub):
👉 [Download Dataset](https://drive.google.com/drive/folders/1JGO8wLvreC-R33vcZxXMGV1vS62LvyfV?usp=sharing)

## 👥 Team — Group 9
- Vaishnav Busha — Backend, data pipeline
- Swaroop — Job scraping, Gemini batch, EDA
- Nishith — QLoRA fine-tuning, training dataset
- Talha Khan — Validation, quality reporting, docs
