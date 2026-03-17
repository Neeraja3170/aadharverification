import os, re, hashlib, datetime, tempfile, logging, time
from flask import Flask, request, jsonify, make_response
from werkzeug.utils import secure_filename
from PIL import Image
from ocr_engine import extract_aadhaar_fields

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# In production, set this via environment variable
app.secret_key = os.environ.get("FLASK_SECRET", "aadhaar-prod-2024-secure-fallback")

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "aadhaar_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}
users_db    = {}

# ── CORS (Production Refined) ────────────────────────────────────────────────
# Default to common local ports, but allow override via ENV
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500,http://localhost:5501,http://127.0.0.1:5501").split(",")

@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    
    response.headers["Access-Control-Allow-Methods"]     = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"]     = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        resp = make_response()
        origin = request.headers.get("Origin")
        if origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Methods"]     = "GET,POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"]     = "Content-Type,Authorization"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp, 200


# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalize_digits(s):
    return re.sub(r"\D", "", s)

def normalize_date(s):
    s = s.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m: return f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
    m = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$", s)
    if m: return f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"
    return s


# ── VERIFICATION LOGIC ────────────────────────────────────────────────────────

def verify_details(form: dict, ocr: dict) -> dict:
    checks  = {}
    details = {}

    # 1. Aadhaar Number
    form_uid = normalize_digits(form.get("aadhaarNumber", ""))
    ocr_uid  = ocr.get("aadhaar_number", "")
    ocr_val  = ocr.get("aadhaar", {}).get("validation", {})

    uid_match = bool(ocr_uid) and (form_uid == ocr_uid)
    checks["aadhaar_number"] = uid_match
    checks["verhoeff"]       = ocr_val.get("rule_verhoeff", False)
    checks["aadhaar_rules"]  = ocr_val.get("valid", False)

    details["form_aadhaar"] = f"****{form_uid[-4:]}" if len(form_uid) >= 4 else "?"
    details["ocr_aadhaar"]  = f"****{ocr_uid[-4:]}"  if len(ocr_uid)  >= 4 else "not found"

    # 2. Date of Birth
    form_dob = normalize_date(form.get("dob", ""))
    ocr_dob  = normalize_date(ocr.get("dob", {}).get("dob", ""))

    checks["dob"] = (form_dob == ocr_dob) if ocr_dob else False
    details["form_dob"] = form_dob
    details["ocr_dob"]  = ocr_dob or "not detected"

    # 3. Name Match
    name_match = ocr.get("name_match", {})
    checks["name"] = name_match.get("matched", False)
    details["ocr_name"] = name_match.get("ocr_name", "not detected")
    details["form_name"] = name_match.get("form_name", "")

    # Overall Result
    uid_ok  = checks["aadhaar_number"] and checks["aadhaar_rules"]
    dob_ok  = checks["dob"] is True
    name_ok = checks["name"]
    passed  = uid_ok and dob_ok and name_ok

    return {
        "passed"    : passed,
        "checks"    : checks,
        "details"   : details,
        "confidence": ocr.get("confidence", 0.0),
        "ocr_raw"   : ocr.get("raw_lines", [])[:10],
    }


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "easyocr", "gpu": False})


@app.route("/api/register/verify-aadhaar", methods=["GET", "POST"])
def verify_aadhaar():
    if request.method == "GET":
        return jsonify({
            "message": "This is a verification endpoint. It only accepts POST requests with Aadhaar image data.",
            "status": "ready"
        }), 200

    # Validate file
    if "aadhaarImage" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["aadhaarImage"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400
    
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "Invalid format. Use JPG/PNG/WEBP."}), 400

    # Save uploaded file
    filename  = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Read form data
    form_data = {k: request.form.get(k, "").strip() for k in
                 ["firstName", "lastName", "dob", "aadhaarNumber",
                  "address", "email", "phone", "gender"]}

    try:
        ocr_result = extract_aadhaar_fields(
            save_path,
            first_name=form_data["firstName"],
            last_name=form_data["lastName"]
        )
    except ValueError as ve:
        return jsonify({"error": str(ve), "not_aadhaar": True}), 422
    except Exception as exc:
        logger.error(f"OCR Exception: {exc}")
        return jsonify({"error": "Server error during processing"}), 500

    # Verify
    v = verify_details(form_data, ocr_result)

    if not v["passed"]:
        reasons = []
        c = v["checks"]
        if not c.get("aadhaar_number"): reasons.append("Aadhaar number mismatch")
        if not c.get("aadhaar_rules"):  reasons.append("Failed UIDAI validation rules")
        if not c.get("dob"):           reasons.append("Date of birth mismatch")
        if not c.get("name"):          reasons.append("Name mismatch")

        return jsonify({
            "success"   : False,
            "message"   : "; ".join(reasons) or "Verification failed",
            "checks"    : v["checks"],
            "details"   : v["details"],
            "confidence": v["confidence"],
        }), 400

    # Success — store user
    uid_hash = hashlib.md5((form_data["email"] + form_data["aadhaarNumber"]).encode()).hexdigest()[:8]

    users_db[uid_hash] = {
        "id"           : uid_hash,
        "firstName"    : form_data["firstName"],
        "lastName"     : form_data["lastName"],
        "email"        : form_data["email"],
        "phone"        : form_data["phone"],
        "dob"          : form_data["dob"],
        "gender"       : form_data["gender"],
        "address"      : form_data["address"],
        "aadhaar"      : form_data["aadhaarNumber"][-4:],
        "verified"     : True,
        "joinedAt"     : datetime.datetime.utcnow().isoformat(),
        "ocr_confidence": v["confidence"],
    }

    # Auto-cleanup files
    try:
        if os.path.exists(save_path): os.remove(save_path)
        prep_path = save_path + "_preprocessed.jpg"
        if os.path.exists(prep_path): os.remove(prep_path)
    except Exception as e:
        logger.warning(f"File cleanup failed: {e}")

    logger.info(f"Successfully registered user: {uid_hash}")
    return jsonify({
        "success"   : True,
        "message"   : "Registered successfully!",
        "user"      : users_db[uid_hash],
        "confidence": v["confidence"],
    })


@app.route("/api/user/<uid>", methods=["GET"])
def get_user(uid):
    u = users_db.get(uid)
    return jsonify(u) if u else (jsonify({"error": "User not found"}), 404)


# ── STARTUP ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    try:
        from waitress import serve
        logger.info(f"Serving via Waitress (Production) on port {port}")
        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        logger.warning("Waitress not found, falling back to Flask dev server.")
        app.run(host="0.0.0.0", port=port, debug=False)