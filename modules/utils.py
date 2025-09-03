import os, hashlib, pickle, pyttsx3, pdfkit, PyPDF2
import pytesseract
from PIL import Image

CACHE_FILE = "cache.pkl"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}

def generate_hash(text, question):
    return hashlib.md5((text + question).encode()).hexdigest()

def speak_text(text):
    tts = pyttsx3.init()
    tts.say(text)
    tts.runAndWait()

def export_pdf(text, filename="outputs/pdf/output.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pdfkit.from_string(text, filename)

def export_txt(text, filename="outputs/txt/output.txt"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(text)

def extract_pdf_text(file):
    # If scanned, use OCR
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        if not text.strip():
            # fallback OCR
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(file.read())
            text = ""
            for page in pages:
                text += pytesseract.image_to_string(page)
        return text
    except:
        return ""
