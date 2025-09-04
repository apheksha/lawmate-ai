# app.py — corrected & improved single-file LawMate AI
import os
import io
import re
import json
import time
import pathlib
import tempfile
import hashlib
import logging
import platform
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import plotly.express as px
import bleach
from weasyprint import HTML

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Index, delete as sa_delete
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Security
import bcrypt

# OCR
import pytesseract

# PDF -> images
try:
    from pdf2image import convert_from_bytes
    _HAS_PDF2IMAGE = True
except Exception:
    _HAS_PDF2IMAGE = False

# Transformers
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# optional googletrans
try:
    from googletrans import Translator as GoogleTranslator
    _HAS_GOOGLETRANS = True
except Exception:
    _HAS_GOOGLETRANS = False

# ------------------ CONFIG ------------------ #
# env overrides
DB_URL = os.getenv("DATABASE_URL", "sqlite:///lawmate.db")
CACHE_PATH = pathlib.Path(os.getenv("CACHE_FILE", "lawmate_cache.json"))
HF_MODEL = os.getenv("HF_MODEL", "mrm8488/T5-base-finetuned-cuad")
TRANSLATION_BACKEND = os.getenv("TRANSLATION_BACKEND", "none")  # 'hf' | 'google' | 'none'
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-en-ROMANCE")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))  # 10 MB
SESSION_TIMEOUT_MIN = int(os.getenv("SESSION_TIMEOUT_MIN", "45"))  # auto-logout minutes
LOGIN_RATE_LIMIT = int(os.getenv("LOGIN_RATE_LIMIT", "5"))  # failed attempts before lock
LOGIN_LOCK_MIN = int(os.getenv("LOGIN_LOCK_MIN", "10"))  # lock minutes

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lawmate")

# ------------------ DB SETUP ------------------ #
Base = declarative_base()
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    failed_logins = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    contracts = relationship("Contract", back_populates="owner", cascade="all, delete-orphan")

class Contract(Base):
    __tablename__ = "contracts"
    id = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=False)
    summary = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    owner = relationship("User", back_populates="contracts")
    qas = relationship("QACache", back_populates="contract", cascade="all, delete-orphan")

class QACache(Base):
    __tablename__ = "qa_cache"
    id = Column(Integer, primary_key=True)
    contract_id = Column(Integer, ForeignKey("contracts.id"))
    question = Column(Text)
    answer = Column(Text)
    risk_info = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    contract = relationship("Contract", back_populates="qas")

Index("ix_qacache_question", QACache.question)
Index("ix_contract_file_name", Contract.file_name)

Base.metadata.create_all(bind=engine)

# ------------------ Compatibility shim for rerun ------------------ #
# Some older Streamlit versions use experimental_rerun; new versions use rerun
if not hasattr(st, "rerun"):
    if hasattr(st, "experimental_rerun"):
        st.rerun = st.experimental_rerun
    else:
        def _no_rerun():
            raise RuntimeError("Streamlit rerun not available in this version.")
        st.rerun = _no_rerun

# ------------------ SAFE JSON CACHE ------------------ #
def load_cache(path: pathlib.Path = CACHE_PATH) -> Dict[str, Any]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load cache: %s", e)
            return {}
    return {}

def save_cache(cache: Dict[str, Any], path: pathlib.Path = CACHE_PATH):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

if 'cache' not in st.session_state:
    st.session_state.cache = load_cache()

# ------------------ HELPERS ------------------ #
def sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^\w\-_\. ]", "_", name)
    return name[:200]

def generate_hash(text: str, question: str) -> str:
    return hashlib.sha256((text + "|" + question).encode("utf-8")).hexdigest()

def now_utc():
    return datetime.utcnow()

# ------------------ AUTH HELPERS ------------------ #
PWD_SALT_ROUNDS = 12

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(PWD_SALT_ROUNDS)).decode()

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode())
    except Exception:
        return False

def password_valid_policy(pw: str) -> (bool, str):
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", pw):
        return False, "Password should contain at least one uppercase letter."
    if not re.search(r"[a-z]", pw):
        return False, "Password should contain at least one lowercase letter."
    if not re.search(r"[0-9]", pw):
        return False, "Password should contain at least one digit."
    return True, ""

# Session expiry management stored in session_state
if 'session_time' not in st.session_state:
    st.session_state.session_time = None

def is_session_expired() -> bool:
    if st.session_state.session_time is None:
        return True
    return datetime.utcnow() - st.session_state.session_time > timedelta(minutes=SESSION_TIMEOUT_MIN)

def refresh_session_time():
    st.session_state.session_time = datetime.utcnow()

# ------------------ OCR HELPERS ------------------ #
LANG_CODE_MAP = {"English": "eng", "Hindi": "hin", "French": "fra"}

def tesseract_languages_available() -> List[str]:
    try:
        langs = pytesseract.get_languages(config='')
        return langs
    except Exception:
        return []

def extract_pdf_text(file_stream: io.BytesIO, lang: str = "eng") -> str:
    # try PyPDF2 text extraction first
    try:
        from PyPDF2 import PdfReader
        file_stream.seek(0)
        reader = PdfReader(file_stream)
        text = ""
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text
    except Exception:
        pass

    # OCR fallback
    if not _HAS_PDF2IMAGE:
        st.error("pdf2image is not installed; PDF OCR disabled. Install pdf2image + Poppler.")
        return ""

    available = tesseract_languages_available()
    if available and lang not in available:
        st.warning(f"Tesseract language '{lang}' not installed. Available: {available}. Falling back to 'eng'.")
        lang_use = 'eng'
    else:
        lang_use = lang

    try:
        file_stream.seek(0)
        images = convert_from_bytes(file_stream.read())
        text = ""
        for img in images:
            page_text = pytesseract.image_to_string(img, lang=lang_use)
            text += page_text + "\n"
        return text
    except Exception as e:
        st.error("OCR failed. Ensure Poppler and tesseract lang packs are installed.")
        logger.exception(e)
        return ""

def extract_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

# ------------------ RISK SCORING (improved) ------------------ #
RISK_PATTERNS = [
    (r"penalt(y|ies|amount|fee)", "High", 3),
    (r"terminate(ion|d)? without (notice|cause|consent)", "High", 3),
    (r"indemnif(y|ication)", "High", 2),
    (r"shall\s+not|must\s+not", "High", 2),
    (r"shall\b", "Medium", 1),
    (r"may\s+not|subject to", "Medium", 1),
    (r"confidential", "Medium", 1),
    (r"warrant(y|ies)?", "Low", 1),
]
NEGATION_WORDS = {"not", "no", "without", "never"}

def compute_risk(clause: str) -> Dict[str, Any]:
    score = 0
    reasons = []
    ctext = clause.lower()
    for pattern, label, weight in RISK_PATTERNS:
        m = re.search(pattern, ctext)
        if m:
            window = ctext[max(0, m.start()-30):m.start()]
            neg = any(n in window for n in NEGATION_WORDS)
            if neg:
                reasons.append(f"Found '{m.group()}' but negated in context.")
                score -= weight
            else:
                reasons.append(f"Matched '{m.group()}' -> {label}")
                score += weight
    
    if score >= 3:
        level = "High"
    elif score >= 1:
        level = "Medium"
    else:
        level = "Low"
    return {"score": score, "level": level, "reasons": reasons}

# ------------------ MODEL LOADING & QUERY ------------------ #
@st.cache_resource(show_spinner=False)
def load_qa_pipeline(model_name=HF_MODEL):
    if not _HAS_TRANSFORMERS:
        return None
    try:
        return pipeline("text2text-generation", model=model_name, device=-1)
    except Exception as e:
        logger.warning("Failed to load %s: %s", model_name, e)
        return None

qa_pipeline = load_qa_pipeline()

@st.cache_resource(show_spinner=False)
def load_qa_qa_pipeline():
    if not _HAS_TRANSFORMERS:
        return None
    try:
        return pipeline("question-answering")
    except Exception:
        return None

qa_qa_pipeline = load_qa_qa_pipeline()

def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    parts = []
    sentences = re.split(r'(?<=[.?!]\s)', text)
    cur = ""
    for s in sentences:
        if len(cur) + len(s) <= max_chars:
            cur += s
        else:
            if cur:
                parts.append(cur)
            cur = s
    if cur:
        parts.append(cur)
    return parts

def query_model_with_chunking(text: str, question: str) -> str:
    key = generate_hash(text, question)
    if key in st.session_state.cache:
        return st.session_state.cache[key]['answer']

    if qa_qa_pipeline:
        best = {"score": -1, "answer": ""}
        for chunk in chunk_text(text, max_chars=1500):
            try:
                out = qa_qa_pipeline(question=question, context=chunk)
                sc = out.get("score", 0)
                if sc > best["score"]:
                    best = {"score": sc, "answer": out.get("answer", "")}
            except Exception:
                continue
        ans = best["answer"] or "No confident answer found."
        st.session_state.cache[key] = {"answer": ans, "created_at": now_utc().isoformat()}
        save_cache(st.session_state.cache)
        return ans

    if qa_pipeline:
        chunks = chunk_text(text, max_chars=2500)[:3]
        prompt = "Context:\n" + "\n\n".join(chunks) + f"\n\nQuestion: {question}"
        try:
            out = qa_pipeline(prompt, max_length=512)
            ans = out[0].get('generated_text') if isinstance(out, list) else str(out)
        except Exception as e:
            logger.exception("Model inference error: %s", e)
            ans = "Model inference failed."
        st.session_state.cache[key] = {"answer": ans, "created_at": now_utc().isoformat()}
        save_cache(st.session_state.cache)
        return ans
    
    return "No analysis model is available. Please check the `transformers` library installation."

# ------------------ TRANSLATION ------------------ #
@st.cache_resource(show_spinner=False)
def load_translation_pipeline(model_name=TRANSLATION_MODEL):
    if not _HAS_TRANSFORMERS:
        return None
    try:
        return pipeline("translation", model=model_name)
    except Exception:
        return None

translation_pipeline = load_translation_pipeline() if TRANSLATION_BACKEND == "hf" else None

def translate_text(text: str, dest: str = "en") -> str:
    if dest == "en" or TRANSLATION_BACKEND == "none":
        return text
    
    if TRANSLATION_BACKEND == "google" and _HAS_GOOGLETRANS:
        try:
            gt = GoogleTranslator()
            return gt.translate(text, dest=dest).text
        except Exception:
            return text
            
    if TRANSLATION_BACKEND == "hf" and translation_pipeline:
        try:
            out = translation_pipeline(text)
            if isinstance(out, list) and out:
                return out[0].get("translation_text", text)
            return str(out)
        except Exception:
            return text
            
    # Fallback: if googletrans available, try it
    if _HAS_GOOGLETRANS:
        try:
            gt = GoogleTranslator()
            return gt.translate(text, dest=dest).text
        except Exception:
            return text

    return text

# ------------------ EXPORTS ------------------ #
ALLOWED_TAGS = ['div', 'span', 'br', 'strong', 'em', 'p', 'h1', 'h2', 'h3', 'ul', 'li', 'b']
ALLOWED_ATTRS = {'*': ['style']}

def sanitize_html(text: str) -> str:
    text = text.replace("\n", "<br/>")
    return bleach.clean(text, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)

def export_pdf_with_metadata(answer_text: str, risk_info: List[Dict[str,Any]], username: str = "anonymous"):
    html_content = f"<h2>LawMate AI — Analysis</h2><p><strong>User:</strong> {username} &nbsp;&nbsp; <strong>Date:</strong> {now_utc().isoformat()}</p>"
    html_content += "<h3>Answer</h3><div>" + sanitize_html(answer_text) + "</div>"
    html_content += "<h3>Risk Summary</h3><ul>"
    for r in risk_info:
        # include reasons joined
        reasons_str = ""
        if isinstance(r.get("reasons"), list):
            reasons_str = " — " + ", ".join(r.get("reasons"))
        elif r.get("reasons"):
            reasons_str = " — " + str(r.get("reasons"))
        html_content += f"<li><strong>{r['level']}</strong> — {sanitize_html(r['clause'])}{sanitize_html(reasons_str)}</li>"
    html_content += "</ul>"
    
    html_path = None
    pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as fh:
            fh.write(html_content)
            html_path = fh.name
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_tmp:
            pdf_path = pdf_tmp.name
            HTML(filename=html_path).write_pdf(pdf_path)
            
        return pdf_path
    finally:
        if html_path and os.path.exists(html_path):
            os.remove(html_path)

# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(page_title="LawMate AI", layout="wide")
st.title("⚖️ LawMate AI — Optimized Single-file")

# Sidebar: auth & settings
with st.sidebar:
    st.header("Account")
    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user and is_session_expired():
        st.warning("Session expired due to inactivity. Please log in again.")
        st.session_state.user = None
        st.rerun()

    if st.session_state.user is None:
        mode = st.radio("Login or Sign up", ["Login", "Sign up"], key="auth_mode")
        if mode == "Sign up":
            new_user = st.text_input("Username", key="su_user")
            new_pass = st.text_input("Password", type="password", key="su_pass")
            if st.button("Create account"):
                ok, msg = password_valid_policy(new_pass)
                if not ok:
                    st.error(msg)
                else:
                    with SessionLocal() as db:
                        if db.query(User).filter_by(username=new_user).first():
                            st.error("Username exists.")
                        else:
                            u = User(username=new_user, password_hash=hash_password(new_pass))
                            db.add(u)
                            db.commit()
                            st.success("Account created. You may now login.")
        else:
            user = st.text_input("Username", key="li_user")
            pwd = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login"):
                with SessionLocal() as db:
                    dbu = db.query(User).filter_by(username=user).first()
                    if not dbu:
                        st.error("Invalid credentials.")
                    else:
                        if dbu.locked_until and dbu.locked_until > datetime.utcnow():
                            wait = int((dbu.locked_until - datetime.utcnow()).total_seconds() / 60) + 1
                            st.error(f"Account locked due to failed attempts. Try again in {wait} minutes.")
                        elif verify_password(pwd, dbu.password_hash):
                            st.session_state.user = {"id": dbu.id, "username": dbu.username}
                            dbu.failed_logins = 0
                            dbu.locked_until = None
                            db.commit()
                            refresh_session_time()
                            st.success(f"Welcome, {dbu.username}!")
                            st.rerun()
                        else:
                            dbu.failed_logins = (dbu.failed_logins or 0) + 1
                            if dbu.failed_logins >= LOGIN_RATE_LIMIT:
                                dbu.locked_until = datetime.utcnow() + timedelta(minutes=LOGIN_LOCK_MIN)
                                st.error(f"Too many failed tries. Account locked for {LOGIN_LOCK_MIN} minutes.")
                            else:
                                st.error("Invalid credentials.")
                            db.commit()
    else:
        st.write(f"Logged in as **{st.session_state.user['username']}**")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

    st.markdown("---")
    st.header("App Settings")
    st.info(f"Upload limit: {MAX_UPLOAD_MB} MB")
    st.write("OCR languages available on your machine (tesseract):")
    try:
        langs = tesseract_languages_available()
        st.write(langs if langs else "Could not detect (tesseract not installed)")
    except Exception:
        st.write("Unknown")

# Main area
if st.session_state.user is None:
    st.info("Log in or sign up to begin analyzing contracts.")
    st.stop()

# Upload + options
st.subheader("Upload contract")
col1, col2 = st.columns([3,1])
with col1:
    uploaded = st.file_uploader("PDF or TXT", type=["pdf","txt"], accept_multiple_files=False)
    sample = st.selectbox("Or use sample", ["None", "Sample TXT", "Sample PDF"])
with col2:
    ocr_lang = st.selectbox("OCR Language", list(LANG_CODE_MAP.keys()))
    out_lang = st.selectbox("Output language", ["en","hi","fr","es"], index=0)

text = ""
file_name = None
if 'file_data' in st.session_state:
    text = st.session_state.file_data['text']
    file_name = st.session_state.file_data['file_name']

if uploaded:
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File too large (> {MAX_UPLOAD_MB} MB).")
        uploaded = None
        st.stop()
    
    file_name = sanitize_filename(uploaded.name)
    content = uploaded.read()
    if uploaded.name.lower().endswith(".txt"):
        text = extract_txt(content)
    else:
        text = extract_pdf_text(io.BytesIO(content), LANG_CODE_MAP[ocr_lang])
    
    st.session_state.file_data = {"text": text, "file_name": file_name}
    st.session_state.pop("last_result", None)
    st.rerun()

if sample != "None" and not uploaded and 'file_data' not in st.session_state:
    sample_path = "samples/sample.txt" if "TXT" in sample else "samples/sample.pdf"
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            b = f.read()
        text = extract_txt(b) if sample_path.endswith(".txt") else extract_pdf_text(io.BytesIO(b), LANG_CODE_MAP[ocr_lang])
        file_name = os.path.basename(sample_path)
        st.session_state.file_data = {"text": text, "file_name": file_name}
        st.session_state.pop("last_result", None)
        st.rerun()
    else:
        st.warning("No sample files included. Upload a file.")

if not text:
    st.info("Upload a file or choose a sample to begin.")
    st.stop()

st.markdown("---")
st.markdown("### Document preview (first 10k chars)")
st.text_area("Preview", value=text[:10000], height=250)

# Q/A
question = st.text_input("Ask a question about this contract (short, specific):")
col_a, col_b = st.columns(2)
with col_a:
    analyze_btn = st.button("Analyze")
with col_b:
    summary_btn = st.button("Quick Summary (auto)")

# Handle button press by setting processing flag and rerunning
if analyze_btn:
    if not question:
        st.error("Write a question first.")
    else:
        st.session_state.processing = "analyze"
        st.rerun()
        
if summary_btn:
    st.session_state.processing = "summary"
    st.rerun()

# Processing: analyze
if st.session_state.get("processing") == "analyze":
    with st.spinner("Analyzing contract with AI..."):
        try:
            refresh_session_time()
            with SessionLocal() as db:
                owner_id = st.session_state.user['id']
                existing = db.query(Contract).filter_by(file_name=file_name, owner_id=owner_id).first()
                if not existing:
                    c = Contract(file_name=file_name or f"uploaded_{int(time.time())}", text=text, owner_id=owner_id)
                    db.add(c)
                    db.commit()
                    contract_id = c.id
                else:
                    contract_id = existing.id
            
            answer = query_model_with_chunking(text, question)
            answer_translated = translate_text(answer, dest=out_lang)
            clauses = [c.strip() for c in re.split(r"\n{1,}|[.\n]", answer_translated) if c.strip()][:50]
            risk_info = []
            for cl in clauses:
                r = compute_risk(cl)
                risk_info.append({"clause": cl, "level": r["level"], "score": r["score"], "reasons": r["reasons"]})
            
            with SessionLocal() as db:
                qa = QACache(contract_id=contract_id, question=question, answer=answer_translated, risk_info=json.dumps(risk_info))
                db.add(qa)
                db.commit()
                qa_id = qa.id  # <-- store id for deletion / reference
            
            # include qa_id in last_result so delete works
            st.session_state.last_result = {"answer": answer_translated, "risk_info": risk_info, "contract_id": contract_id, "question": question, "qa_id": qa_id}
            st.success("Analysis complete.")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            logger.exception(e)
    st.session_state.processing = None
    st.rerun()

# Processing: summary
elif st.session_state.get("processing") == "summary":
    with st.spinner("Generating summary..."):
        try:
            refresh_session_time()
            summary_text = ""
            if qa_pipeline:
                try:
                    out = qa_pipeline("Summarize:\n" + text[:5000], max_length=200)
                    summary_text = out[0].get("generated_text") if isinstance(out, list) else str(out)
                except Exception as e:
                    logger.warning("Summary model failed: %s", e)
                    summary_text = text[:500]
            else:
                summary_text = text[:500]
            # store summary to display after rerun (avoid printing inside spinner)
            st.session_state.summary_temp = summary_text
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
            logger.exception(e)
    st.session_state.processing = None
    st.rerun()

# If a stored summary exists, show it and remove
if st.session_state.get("summary_temp"):
    st.markdown("---")
    st.info("Auto summary (generated):")
    st.write(st.session_state.pop("summary_temp"))

# Show last result if available
if st.session_state.get("last_result"):
    res = st.session_state.last_result
    st.markdown("---")
    st.header("Result")
    st.markdown("### Answer")
    st.write(res["answer"])
    st.markdown("### Clause-level Risk Analysis")
    for i, r in enumerate(res["risk_info"]):
        color = {"High": "#ff4b4b", "Medium": "#ffb04b", "Low": "#7bd389"}.get(r["level"], "#cccccc")
        with st.expander(f"Clause {i+1} — Risk: {r['level']} (score {r['score']})"):
            st.markdown(f"<div style='color:{color}'><strong>Clause:</strong> {r['clause']}</div>", unsafe_allow_html=True)
            if r.get('reasons'):
                st.write("Reasons:")
                for reason in r['reasons']:
                    st.write("- " + reason)
    
    st.markdown("### Dashboard")
    df = pd.DataFrame(res["risk_info"])
    if not df.empty:
        counts = df['level'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
        fig = px.bar(x=counts.index, y=counts.values, labels={'x':'Risk Level','y':'Count'}, title='Risk Level Counts')
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Download PDF Report"):
            try:
                pdfp = export_pdf_with_metadata(res["answer"], res["risk_info"], username=st.session_state.user['username'])
                with open(pdfp, "rb") as f:
                    st.download_button("Download PDF", f, file_name=f"analysis_{int(time.time())}.pdf", mime="application/pdf")
                os.remove(pdfp)
            except Exception as e:
                st.error(f"PDF export failed: {e}")
    with c2:
        if st.button("Download JSON"):
            try:
                jname = f"analysis_{int(time.time())}.json"
                payload = {"meta": {"user": st.session_state.user['username'], "ts": now_utc().isoformat()},
                           "question": res["question"], "answer": res["answer"], "risk_info": res["risk_info"]}
                with open(jname, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                with open(jname, "rb") as f:
                    st.download_button("Download JSON", f, file_name=jname, mime="application/json")
                os.remove(jname)
            except Exception as e:
                st.error(f"JSON export failed: {e}")
    with c3:
        if st.button("🔊 Read Aloud"):
            try:
                if platform.system() == 'Darwin':
                    subprocess.run(["say", res["answer"]])
                else:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.say(res["answer"])
                    engine.runAndWait()
            except Exception as e:
                st.error("TTS failed: " + str(e))
    with c4:
        if st.button("Delete this result"):
            st.session_state.confirm_delete_qa = True
            st.rerun()
    
    if st.session_state.get('confirm_delete_qa'):
        st.warning("Are you sure you want to delete this result?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, Delete"):
                try:
                    qa_id = res.get('qa_id')
                    if qa_id:
                        with SessionLocal() as db:
                            db.query(QACache).filter_by(id=qa_id).delete()
                            db.commit()
                    st.session_state.pop("last_result", None)
                    st.session_state.pop("confirm_delete_qa", None)
                    st.success("Result deleted successfully.")
                except Exception as e:
                    st.error(f"Failed to delete: {e}")
                    logger.exception(e)
                st.rerun()
        with col_no:
            if st.button("Cancel"):
                st.session_state.pop("confirm_delete_qa", None)
                st.rerun()

# Sidebar: Previous Contracts
st.sidebar.markdown("---")
st.sidebar.header("Your Contracts & Q&A")
with SessionLocal() as db:
    user_id = st.session_state.user['id']
    contracts = db.query(Contract).filter_by(owner_id=user_id).order_by(Contract.created_at.desc()).all()
    for c in contracts:
        with st.sidebar.expander(f"{c.file_name} — {c.created_at.date()}"):
            st.write(c.text[:300] + ("..." if len(c.text) > 300 else ""))
            qas = db.query(QACache).filter_by(contract_id=c.id).order_by(QACache.created_at.desc()).all()
            for q in qas:
                st.markdown(f"**Q:** {q.question}")
                st.markdown(f"**A:** {q.answer[:300]}...")
                if st.button(f"Load QA {q.id}", key=f"load_qa_{q.id}"):
                    try:
                        risk_info = json.loads(q.risk_info) if q.risk_info else []
                    except Exception:
                        risk_info = []
                    st.session_state.last_result = {"answer": q.answer, "risk_info": risk_info, "contract_id": c.id, "question": q.question, "qa_id": q.id}
                    st.rerun()
            
            if st.button(f"Delete Contract", key=f"del_contract_{c.id}"):
                st.session_state[f"confirm_delete_contract_{c.id}"] = True
                st.rerun()
            
            if st.session_state.get(f"confirm_delete_contract_{c.id}"):
                st.warning("All associated Q&As will also be deleted. Are you sure?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Confirm Delete", key=f"confirm_del_{c.id}"):
                        try:
                            with SessionLocal() as db_del:
                                db_del.execute(sa_delete(Contract).where(Contract.id == c.id))
                                db_del.commit()
                            st.session_state.pop(f"confirm_delete_contract_{c.id}", None)
                            st.success("Contract and all Q&As deleted.")
                        except Exception as e:
                            st.error(f"Failed to delete contract: {e}")
                            logger.exception(e)
                        st.rerun()
                with col_no:
                    if st.button("Cancel", key=f"cancel_del_{c.id}"):
                        st.session_state.pop(f"confirm_delete_contract_{c.id}", None)
                        st.rerun()

# Footer
st.markdown("---")
st.write("Built for demo/educational use. For production, add HTTPS, model serving, and managed DB.")
