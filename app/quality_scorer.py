"""Quality metrics: skill coverage, hallucination rate, skill count."""
import json, re
from typing import Dict

def compute_quality_metrics(raw: str, resume: str, jd: str) -> Dict:
    jd_tok  = set(_tok(jd))
    out_tok = set(_tok(raw))
    res_tok = set(_tok(resume))
    coverage = round(len(out_tok & jd_tok) / max(len(jd_tok), 1), 3)
    suspicious = [w for w in (out_tok - res_tok - jd_tok) if len(w) > 6]
    halluc = max(0, len(suspicious) - 10)
    try:
        parsed = json.loads(raw)
        skill_count = sum(len(sg.get("data", [])) for sg in parsed.get("skills", []))
    except:
        skill_count = 0
    return {"skill_coverage": coverage, "hallucination_count": halluc,
            "hallucination_rate": round(halluc / max(len(out_tok), 1), 4), "skill_item_count": skill_count}

def _tok(text):
    return re.findall(r"[a-z][a-z0-9+#.]{2,}", text.lower())
