# ⚖️ LawMate AI

LawMate AI is a Streamlit-based application for **contract analysis and risk assessment**. It allows users to upload contracts (PDF or TXT), ask questions, get clause-level risk analysis, and download reports in PDF, DOCX, or JSON formats.

---

## Features

### User Account & Authentication
- Sign up with username, password, and optional security question.
- Login with session management and inactivity timeout.
- Password reset using security question.
- Login lockout on repeated failed attempts.

### File Upload & Management
- Upload single PDF or TXT files for analysis.
- Batch OCR for multiple files simultaneously.
- View previously uploaded contracts and QA history.
- Lazy-loading of contract list and Q&As in the sidebar.

### Document Processing
- OCR for PDFs (via pytesseract + pdf2image).
- Text extraction from TXT files.
- Auto-detect language for OCR (optional).
- Preview document text with risky terms highlighted.

### Question & Answer (QA)
- Ask short, specific questions about uploaded contracts.
- Generate quick summaries of the contract.
- Clause-level risk scoring (Low / Medium / High).
- View risk distribution as heatmap and pie chart.
- Search within document and previous Q&As.

### Exports & Actions
- Download analysis results as:
  - **PDF** (weasyprint or reportlab)
  - **DOCX**
  - **JSON**
- Text-to-speech for reading the answer aloud.
- Diff uploaded contract with previous version.
- Delete QA or contract entries.

### Diagnostics & Admin
- Quick checks for dependencies: OCR, Hugging Face model, DOCX support.
- Database connectivity check.
- Model availability test.
- Session timeout info and upload size limit.

---
