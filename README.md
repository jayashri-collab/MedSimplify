# MedSimplify 🩺

MedSimplify is an AI-powered Python application built to translate complex clinical medical reports into simple, patient-friendly language. It targets the divide between clinical jargon and patient understanding by presenting complex radiology, hematology, and discharge reports in plain, easily readable English.

---

## 🌟 Key Features

1. **AI-Driven Text Simplification**: Leverages the instruction-tuned Hugging Face `google/flan-t5-base` model to rephrase clinical narratives.
2. **Medical Terminology Dictionary & Fallback**: Scans reports for 100+ common medical terms and substitutes/highlights them. If the AI model fails to load or run (e.g. out of memory), the system automatically falls back to dictionary translation.
3. **Interactive Term Highlighting**: Highlights medical terms in both original and simplified text with hoverable, patient-friendly tooltip definitions.
4. **Interactive Glossary**: Generates a custom patient glossary listing all clinical terms detected in the text alongside definitions.
5. **PDF Upload & Processing**: Extracts raw text from medical report PDFs using `pypdf`.
6. **Downloadable Reports**: Compiles a printable formatted text document containing the patient-friendly report, glossary, and original reference report.
7. **Premium Streamlit UI**: Fully responsive, card-based interface with custom Outfit typography, modern layouts, and real-time execution status tracking.

---

## 📁 Project Structure

```
C:/MedSimplify/
├── app.py                      # Main Streamlit dashboard frontend
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── preprocessing/
│   └── clean_text.py           # PDF text extraction and sentence chunking utilities
├── model/
│   └── inference.py            # Hugging Face model loading and inference pipelines
├── utils/
│   └── helper_functions.py     # Medical dictionary, regex-based highlighting, glossary and download formatters
└── sample_reports/
    ├── sample_radiology.txt    # Mock chest X-ray report
    └── sample_blood_test.txt   # Mock CBC & metabolic panel report
```

---

## 🚀 Setup & Execution

### Prerequisites
* **Python 3.10+** (The project was tested and built on Python 3.13)
* Internet connection (upon the first run, the app will automatically download the `google/flan-t5-base` model, which is ~990MB).

### Installation

1. Navigate to the project folder:
   ```bash
   cd C:/MedSimplify
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   * **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Windows (Command Prompt)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit web application with:
```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## 🩺 Technical Design & Fallback

* **Context Window Chunking**: To ensure long medical reports do not overflow the model's 512-token input limit, MedSimplify parses text into paragraphs and sentences, dynamically grouping them into chunks of at most 1,200 characters before forwarding them to the AI pipeline.
* **Streamlit Resource Caching**: Uses `@st.cache_resource` to keep the Flan-T5 model loaded in memory, eliminating start-up delays on dashboard interaction.
* **Error Bounds**: If GPU/CUDA initialization fails, or transformers libraries encounter errors, a robust try-except wrapper catches the exception and immediately invokes dictionary-based replacement, displaying a warning banner.
* **Interactive Tooltips**: Built using pure HTML and CSS custom class styling within markdown, providing instant client-side rendering without JavaScript dependencies.

---

*Disclaimer: MedSimplify is an educational tool meant to increase medical report accessibility. It does not replace professional consultation, diagnostic analysis, or direct care recommendations from clinical practitioners.*
