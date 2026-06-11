"""Loads Qwen3-4B + QLoRA adapter and runs inference."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct"
ADAPTER_PATH  = "./model-output/resume-optimizer-lora"  # update after training

SYSTEM_PROMPT = """You create a tailored resume based on the job description.
Your task:
1. Read the RESUME_TEXT.
2. Read the JOB_DESCRIPTION.
3. Use only the information inside these two.
4. Follow the SCHEMA exactly.
5. Write a tailored resume in JSON using the SCHEMA.
6. Do not output anything outside the JSON."""

class ResumeOptimizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._load()

    def _load(self):
        try:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb, device_map="auto")
            if Path(ADAPTER_PATH).exists():
                self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)
                print(f"✅ Fine-tuned model loaded from {ADAPTER_PATH}")
            else:
                self.model = base
                print("⚠️  Adapter not found — using base model only")
            self._loaded = True
        except Exception as e:
            print(f"❌ Model load failed: {e}")

    def is_loaded(self): return self._loaded

    def generate(self, resume_text: str, job_description: str, max_new_tokens: int = 1024) -> str:
        if not self._loaded: raise RuntimeError("Model not loaded.")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f'RESUME_TEXT:\n"""\n{resume_text[:3000]}\n"""\n\nJOB_DESCRIPTION:\n"""\n{job_description[:2000]}\n"""'},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      temperature=0.1, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
