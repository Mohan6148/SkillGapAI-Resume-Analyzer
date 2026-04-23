# AI Skill Gap Analyzer Pro

An end-to-end Streamlit application that compares a resume against a job description, extracts and classifies skills, computes gap analysis, generates ATS-style scoring, and suggests a personalized learning path.

## Highlights

- Modern UI with guided workflow (Upload -> Extract -> Analyze -> Gaps -> Visualize -> ATS)
- Resume and JD parsing for PDF, DOCX, and TXT
- Text normalization and personal data cleanup pipeline
- Rule-based + optional AI-hybrid skill extraction
- Semantic skill matching and missing skill identification
- ATS score with factor-level recommendations
- Learning path suggestions based on missing skills
- Downloadable report generation (PDF/HTML fallback)

## Tech Stack

- Python 3.12
- Streamlit
- spaCy (with lightweight fallback if model is unavailable)
- sentence-transformers
- pandas, numpy, plotly, matplotlib, seaborn
- reportlab

## Project Structure

```text
skillgapAI/
|- app.py
|- chatbot.py
|- askill_ext.py
|- gap_analysys.py
|- adata_ingestion&parsing.py
|- main.py
|- requirements.txt
|- skills_list.txt
|- src/
|  |- pipeline.py
|  |- skill_extractor.py
|  |- file_readers/
|  |  |- file_readers_pdf.py
|  |  |- file_readers_docx.py
|  |  |- file_readers_txt.py
|  |- text_cleaner/
|     |- remove_personal.py
|     |- section_normalizer.py
|     |- txt_cleaner.py
```

## Supported Input Formats

- Resume: PDF, DOCX, TXT, pasted text
- Job Description: PDF, DOCX, TXT, pasted text

## Local Setup (Windows PowerShell)

From project root:

```powershell
cd E:\skillgapAI
```

Create virtual environment (if not already created):

```powershell
python -m venv .venv
```

Activate virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the main app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Open in browser:

```text
http://localhost:8501
```

## Alternate Entrypoints

Run milestone ingestion app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run "adata_ingestion&parsing.py"
```

Run launcher:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Notes on Models and Warnings

- You do not need to run `python -m spacy download en_core_web_sm` for basic startup in this project.
- If you see `ModuleNotFoundError: No module named 'torchvision'` inside Streamlit watcher logs, it is usually a non-blocking warning caused by optional `transformers` image modules.

## License

This project has MIT License