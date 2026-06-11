"""Validate model JSON output against resume schema."""
import json, re
from typing import Tuple

REQUIRED = {"personal_information", "summary", "experiences", "skills"}

def validate_resume_json(raw: str) -> Tuple[dict, bool]:
    parsed = _extract(raw)
    if parsed is None: return {}, False
    return parsed, len(REQUIRED - set(parsed.keys())) == 0

def _extract(text):
    if not text: return None
    text = text.strip()
    for attempt in [
        lambda t: json.loads(t),
        lambda t: json.loads(t[t.find("```json")+7:t.find("```",t.find("```json")+7)].strip()) if "```json" in t else None,
        lambda t: json.loads(t[t.find("{"):t.rfind("}")+1]) if t.find("{") >= 0 else None,
    ]:
        try:
            r = attempt(text)
            if r is not None: return r
        except: pass
    return None
