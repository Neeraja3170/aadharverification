"""
ocr_engine.py
─────────────
Aadhaar card OCR using EasyOCR exclusively.
"""

import re
import unicodedata
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import difflib

# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR reader — lazy-loaded singleton
# ─────────────────────────────────────────────────────────────────────────────

_READER = None

def get_reader():
    global _READER
    if _READER is None:
        _READER = easyocr.Reader(["en"], gpu=False)
    return _READER


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

OCR_TARGET_W = 1800
OCR_TARGET_H = 1200
MIN_W        = 600
MIN_H        = 400
MAX_W        = 4000
MAX_H        = 3000

def preprocess_image(image_path: str) -> tuple:
    img = Image.open(image_path)
    try:
        exif = img._getexif()
        if exif:
            orientation = exif.get(274)
            rotate_map  = {3: 180, 6: 270, 8: 90}
            if orientation in rotate_map:
                img = img.rotate(rotate_map[orientation], expand=True)
    except Exception:
        pass

    if img.mode != "RGB":
        img = img.convert("RGB")

    original_size = img.size
    w, h          = original_size
    was_resized   = False

    # Optimized for speed: Use BOX for downscaling, reduce target size slightly
    if w > MAX_W or h > MAX_H:
        ratio = min(MAX_W / w, MAX_H / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img   = img.resize((new_w, new_h), Image.BOX) # Faster than LANCZOS
        w, h  = new_w, new_h
        was_resized = True

    # Moderate target size (1500 is usually enough for EasyOCR)
    TARGET_W = 1500 
    if w > TARGET_W:
        ratio = TARGET_W / w
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img   = img.resize((new_w, new_h), Image.BILINEAR)
        was_resized = True

    # Fast enhancement
    img = img.filter(ImageFilter.SHARPEN)

    preprocessed_path = image_path + "_preprocessed.jpg"
    img.save(preprocessed_path, "JPEG", quality=95)

    new_size = img.size
    return preprocessed_path, original_size, new_size, was_resized


# ─────────────────────────────────────────────────────────────────────────────
# AADHAAR CARD AUTHENTICITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

AADHAAR_REQUIRED_KEYWORDS = ["uidai", "aadhaar", "aadhar", "unique identification", "government of india", "भारत सरकार", "आधार"]

def is_aadhaar_card(lines: list) -> dict:
    full_text_lower = " ".join(lines).lower()
    keywords_found = [kw for kw in AADHAAR_REQUIRED_KEYWORDS if kw in full_text_lower]
    has_keywords = len(keywords_found) >= 1
    has_12_digit = bool(re.search(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", " ".join(lines)))
    has_date = bool(re.search(r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b|\byear\s*of\s*birth\b|\bdob\b", full_text_lower))

    if not has_keywords and not has_12_digit:
        return {"is_aadhaar": False, "reason": "Not an Aadhaar card image.", "keywords_found": keywords_found}
    
    return {"is_aadhaar": True, "reason": "Valid Aadhaar card detected", "keywords_found": keywords_found}


# ─────────────────────────────────────────────────────────────────────────────
# VERHOEFF CHECKSUM
# ─────────────────────────────────────────────────────────────────────────────

VERHOEFF_TABLE_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
VERHOEFF_TABLE_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]

def verhoeff_check(number: str) -> bool:
    if len(number) != 12 or not number.isdigit(): return False
    c = 0
    for i, digit in enumerate(reversed(number)):
        p = VERHOEFF_TABLE_P[i % 8][int(digit)]
        c = VERHOEFF_TABLE_D[c][p]
    return c == 0

def validate_aadhaar_rules(number: str) -> dict:
    if not number or not number.isdigit() or len(number) != 12:
        return {"valid": False, "rule_verhoeff": False}
    r2 = number[0] not in ("0", "1")
    r3 = len(set(number)) > 1
    r5 = verhoeff_check(number)
    return {"valid": all([r2, r3, r5]), "rule_verhoeff": r5}


# ─────────────────────────────────────────────────────────────────────────────
# OCR & EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def run_easyocr(image_path: str):
    reader = get_reader()
    results = reader.readtext(image_path, detail=1, paragraph=False)
    if not results: return [], []
    results_sorted = sorted(results, key=lambda r: r[0][0][1])
    lines = [text for (_bbox, text, _conf) in results_sorted if text.strip()]
    return lines, results_sorted

def extract_aadhaar_number(lines: list) -> dict:
    full_text = " ".join(lines)
    candidates = []
    for m in re.finditer(r"\b(\d{4})[\s\-](\d{4})[\s\-](\d{4})\b", full_text):
        candidates.append(m.group(1) + m.group(2) + m.group(3))
    for m in re.finditer(r"\b(\d{12})\b", full_text):
        candidates.append(m.group(1))
    for line in lines:
        digits = re.sub(r"\D", "", line)
        if len(digits) == 12: candidates.append(digits)
    
    seen = list(set(candidates))
    for c in seen:
        val = validate_aadhaar_rules(c)
        if val["valid"]:
            return {"number": c, "formatted": f"{c[:4]} {c[4:8]} {c[8:]}", "validation": val, "found": True}
    return {"number": "", "formatted": "", "validation": {"valid": False}, "found": False}

def extract_dob(lines: list) -> dict:
    full_text = " ".join(lines)
    # Common OCR mistakes: O -> 0, I/l -> 1
    # We clean the text for numeric patterns but keep separators
    cleaned_text = full_text.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').replace('|', '1')
    
    patterns = [
        (r"DOB[:\s]*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})", "dmy"),
        (r"Date\s*of\s*Birth[:\s]*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})", "dmy"),
        (r"Birth[:\s]*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})", "dmy"),
        (r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b", "dmy"),
    ]
    for pat, fmt in patterns:
        m = re.search(pat, cleaned_text, re.IGNORECASE)
        if m:
            d, mo, y = m.groups()
            # Basic sanity check for day/month
            try:
                if 1 <= int(d) <= 31 and 1 <= int(mo) <= 12:
                    return {"dob": f"{d.zfill(2)}/{mo.zfill(2)}/{y}", "found": True}
            except ValueError:
                continue
    
    # Year of birth only
    m = re.search(r"Year\s*of\s*Birth[:\s]*(\d{4})", cleaned_text, re.IGNORECASE)
    if m: return {"dob": f"??/??/{m.group(1)}", "found": True}
    return {"dob": "", "found": False}


# ─────────────────────────────────────────────────────────────────────────────
# NAME MATCHING (FUZZY)
# ─────────────────────────────────────────────────────────────────────────────

SKIP_WORDS = {"government", "of", "india", "unique", "identification", "authority", "uidai", "aadhaar", "aadhar", "male", "female", "dob", "date", "birth"}

def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z\s]", "", text.lower()).strip()

def _is_name_line(line: str) -> bool:
    s = line.strip()
    if len(s) < 3 or not re.search(r"[A-Za-z]", s): return False
    if any(w in s.lower() for w in SKIP_WORDS): return False
    return True

def extract_name_candidates(lines: list) -> list:
    return [l.strip() for l in lines if _is_name_line(l.strip())]

def match_name(candidates: list, first: str, last: str) -> dict:
    f_clean = _clean_text(first).split()
    l_clean = _clean_text(last).split()
    input_words = set(f_clean + l_clean)
    
    if not input_words:
        return {"matched": False, "match_type": "none", "ocr_name": "no input", "score": 0.0}

    for c in candidates:
        c_raw = c.strip()
        c_clean = _clean_text(c_raw)
        c_words = set(c_clean.split())
        
        if not c_words: continue
        
        # Strict Word Set Match (Exact Spelling, Order Independent)
        # Using .issubset allows users to omit middle names or initials 
        # that might be on the card, but requires exact spelling for 
        # every word they DID enter.
        if input_words.issubset(c_words):
            return {
                "matched": True, 
                "match_type": "exact_spelling", 
                "ocr_name": c_raw, 
                "score": 1.0
            }
            
    return {"matched": False, "match_type": "none", "ocr_name": "not detected", "score": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# GENDER & FINAL
# ─────────────────────────────────────────────────────────────────────────────

def extract_gender(lines: list) -> str:
    t = " ".join(lines).lower()
    if re.search(r"f[e|a|h][m|h|n][a|e]le", t) or "महिला" in t: return "F"
    if "male" in t or "पुरुष" in t: return "M"
    return ""

def _compute_confidence(aadhaar, dob, name_match) -> float:
    s = 0.0
    if aadhaar.get("found"): s += 0.4
    if dob.get("found"): s += 0.3
    if name_match.get("matched"): s += 0.3
    return round(s, 2)

def extract_aadhaar_fields(image_path: str, first_name: str = "", last_name: str = "") -> dict:
    prep_path, orig_size, new_size, was_res = preprocess_image(image_path)
    lines, raw = run_easyocr(prep_path)
    card_check = is_aadhaar_card(lines)
    if not card_check["is_aadhaar"]: raise ValueError(card_check["reason"])
    
    aadhaar = extract_aadhaar_number(lines)
    dob = extract_dob(lines)
    name_candidates = extract_name_candidates(lines)
    name_match = match_name(name_candidates, first_name, last_name)
    gender = extract_gender(lines)
    conf = _compute_confidence(aadhaar, dob, name_match)
    
    return {
        "aadhaar": aadhaar, "dob": dob, "name_match": name_match, "gender": gender,
        "raw_lines": lines, "confidence": conf,
        "image_info": {"original_size": orig_size, "processed_size": new_size, "was_resized": was_res},
        "card_check": card_check, "aadhaar_number": aadhaar["number"],
        "name_detected": name_match["matched"],
        "verhoeff_passed": aadhaar["validation"].get("rule_verhoeff", False)
    }