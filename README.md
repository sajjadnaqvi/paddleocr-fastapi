# OCR Image to Text - Lab 3 Agentic Document Understanding

This project implements Lab 3 from the course: a hybrid document intelligence pipeline that combines OCR, layout detection, reading order, and VLM tools, then orchestrates them with a LangChain agent.

## What’s Included
- `L6.ipynb`: Full lab implementation
- `transcription.md`: The lab walkthrough transcript

## Prerequisites
- Python 3.10+ recommended (the notebook was authored with 3.10)
- An OpenAI API key

## Setup
1. Create and activate a virtual environment
2. Install dependencies
3. Add your API key to `.env`

### 1) Create and activate venv (Windows PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

### 3) Configure environment
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

## Required Assets
Place these files in the project root (you said you will add test files later):
- `report_original.png` (document image for OCR and layout detection)
- `architecture.png` (optional; only used for a markdown diagram)

## Run the Lab
Open `L6.ipynb` in Jupyter or VS Code and run cells top to bottom.

## Notes
- `layoutreader` is installed from GitHub and used for reading order detection.
- VLM tools require the OpenAI key to be present in `.env`.

