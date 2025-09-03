import streamlit as st
from transformers import pipeline
import pyttsx3
import os
import hashlib
import pickle
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
import plotly.express as px
import pandas as pd
import time
from weasyprint import HTML
from googletrans import Translator
import platform, subprocess
# -------------------- Configuration -------------------- #
CACHE_FILE = "cache.pkl"
DB_FILE = "legal_assistant.db"

# -------------------- Load Hugging Face Model -------------------- #
generator = pipeline(
    "text2text-generation",
    model="mrm8488/T5-base-finetuned-cuad",
    device=-1
)

# -------------------- Offline Cache -------------------- #
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}

def generate_hash(text, question):
    return hashlib.md5((text + question).encode()).hexdigest()

# -------------------- SQLite Setup -------------------- #
Base = declarative_base()
engine = create_engine(f'sqlite:///{DB_FILE}')
Session = sessionmaker(bind=engine)
session = Session()

class Contract(Base):
    __tablename__ = 'contracts'
    id = Column(Integer, primary_key=True)
    file_name = Column(String, unique=True)
    text = Column(Text)
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class QACache(Base):
    __tablename__ = 'qa_cache'
    id = Column(Integer, primary_key=True)
    contract_id = Column(Integer)
    question = Column(Text)
    answer = Column(Text)
    risk_info = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# -------------------- TTS -------------------- #

def speak_text(text):
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["say", text])
    else:
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")
translator = Translator()

def safe_translate(text: str, target_lang: str = "en") -> str:
    try:
        lang_map = {
            "English": "en",
            "Hindi": "hi",
            "Spanish": "es",
            "French": "fr"
        }
        target_code = lang_map.get(target_lang, "en")
        if target_code == "en":
            return text
        translated = translator.translate(text, dest=target_code)
        return translated.text
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text
# -------------------- Helper Functions -------------------- #
def risk_scoring(clause):
    keywords = {
        "penalty": "High",
        "terminate": "High",
        "shall": "Medium",
        "may": "Low",
        "without consent": "High",
    }
    for word, score in keywords.items():
        if word in clause.lower():
            return score
    return "Low"

def color_code_clause(clause, score):
    color = {"High": "red", "Medium": "orange", "Low": "green"}
    return f"<span style='color:{color[score]}'>{clause}</span>"

def ask_model(text, question):
    key = generate_hash(text, question)
    if key in cache:
        return cache[key]
    with st.spinner("Analyzing contract..."):
        time.sleep(1)
        result = generator(f"Text: {text}\nQuestion: {question}", max_length=400)
    answer = result[0]['generated_text']
    cache[key] = answer
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    return answer

def extract_pdf_text(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        if text.strip():
            return text
    except:
        pass
    # fallback OCR
    file.seek(0)
    images = convert_from_bytes(file.read())
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text

def save_contract(file_name, text, summary=""):
    existing = session.query(Contract).filter_by(file_name=file_name).first()
    if not existing:
        contract = Contract(file_name=file_name, text=text, summary=summary)
        session.add(contract)
        session.commit()
        return contract.id
    else:
        return existing.id

def save_qa(contract_id, question, answer, risk_info):
    qa_entry = QACache(contract_id=contract_id, question=question, answer=answer, risk_info=risk_info)
    session.add(qa_entry)
    session.commit()

def export_pdf(text, filename="contract_summary.pdf"):
    HTML(string=text).write_pdf(filename)
    return filename

def export_txt(text, filename="contract_summary.txt"):
    with open(filename, "w") as f:
        f.write(text)
    return filename

# -------------------- Streamlit UI -------------------- #
st.set_page_config(page_title="LawMate AI", layout="wide", initial_sidebar_state="expanded")

st.title("⚖️ LawMate AI – Contract Analyzer")

# -------------------- File Upload Section -------------------- #
with st.expander("📂 Upload or Select a Contract", expanded=True):
    uploaded_file = st.file_uploader("Upload your contract (.txt or .pdf)", type=["txt","pdf"])
    demo_choice = st.selectbox("Select a sample contract", ["None", "Sample NDA", "Sample Employment"])

    all_contracts = session.query(Contract).all()
    prev_contract = st.selectbox("Or load a previously analyzed contract", ["None"] + [c.file_name for c in all_contracts])

# -------------------- Load Contract -------------------- #
contract_id, file_name, text = None, None, ""

if demo_choice != "None" and uploaded_file is None:
    file_path = "data/sample_contract.pdf" if demo_choice == "Sample NDA" else "data/contracts.txt"
    with open(file_path, "rb") as f:
        text = extract_pdf_text(f) if file_path.endswith(".pdf") else f.read().decode()
    file_name = os.path.basename(file_path)
elif uploaded_file:
    text = extract_pdf_text(uploaded_file) if uploaded_file.name.endswith(".pdf") else uploaded_file.read().decode()
    file_name = uploaded_file.name
elif prev_contract != "None":
    contract_obj = session.query(Contract).filter_by(file_name=prev_contract).first()
    if contract_obj:
        text = contract_obj.text
        file_name = contract_obj.file_name
        contract_id = contract_obj.id

# -------------------- Q&A Section -------------------- #
question = st.text_input("Ask a question about this contract:")
language = st.selectbox("Output Language", ["English","Hindi","Spanish","French"])

if text and question:
    contract_id = save_contract(file_name, text)
    raw_answer = ask_model(text, question)

    # TODO: Translation fallback (currently no deep_translator)
    if language != "English":
        raw_answer = safe_translate(raw_answer, language)


    # Display clauses + risk
    clauses = raw_answer.split("\n")
    risk_info = []
    st.markdown("### 📑 Contract Analysis")
    for i, clause in enumerate(clauses):
        score = risk_scoring(clause)
        risk_info.append({"Clause": clause, "Risk": score})
        with st.expander(f"Clause {i+1} - Risk: {score}"):
            st.markdown(color_code_clause(clause, score), unsafe_allow_html=True)

    save_qa(contract_id, question, raw_answer, str(risk_info))

    # Dashboard
    st.markdown("### 📊 Compliance Dashboard")
    df = pd.DataFrame(risk_info)
    if not df.empty:
        fig_bar = px.bar(df, x="Clause", y=[1]*len(df), color="Risk", title="Clause Risk Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)

        risk_count = df['Risk'].value_counts()
        fig_pie = px.pie(values=risk_count.values, names=risk_count.index, title="Risk Level Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Timeline
    st.markdown("### 📅 Contract Timeline (Sample Deadlines)")
    timeline_data = pd.DataFrame([
        dict(Task="Payment Deadline", Start='2025-09-05', Finish='2025-09-10'),
        dict(Task="Contract Renewal", Start='2025-10-01', Finish='2025-10-02'),
        dict(Task="Penalty Trigger", Start='2025-11-15', Finish='2025-11-15')
    ])
    fig_gantt = px.timeline(timeline_data, x_start="Start", x_end="Finish", y="Task", title="Contract Deadlines Timeline")
    fig_gantt.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_gantt, use_container_width=True)

    # Export + TTS
    col1, col2, col3 = st.columns(3)
    with col1:
        pdf_file = export_pdf(raw_answer)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name=pdf_file, mime="application/pdf")
    with col2:
        txt_file = export_txt(raw_answer)
        with open(txt_file, "rb") as f:
            st.download_button("⬇️ Download TXT", f, file_name=txt_file, mime="text/plain")
    with col3:
        if st.button("🔊 Read Aloud"):
            speak_text(raw_answer)

# -------------------- Previous Q&A -------------------- #

st.sidebar.header("📜 Previous Q&A")
if contract_id:
    prev_qa = session.query(QACache).filter_by(contract_id=contract_id).all()
    for qa in prev_qa:
        with st.sidebar.expander(f"Q: {qa.question}"):
            st.write(qa.answer)
            st.write(qa.risk_info)
