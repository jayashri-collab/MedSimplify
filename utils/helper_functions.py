import re

# Comprehensive Medical Dictionary
MEDICAL_DICT = {
    "hypertension": "high blood pressure",
    "hypotension": "low blood pressure",
    "myocardial infarction": "heart attack",
    "dyspnea": "shortness of breath",
    "arrhythmia": "irregular heartbeat",
    "tachycardia": "abnormally fast heart rate",
    "bradycardia": "abnormally slow heart rate",
    "cardiomegaly": "enlarged heart",
    "atelectasis": "partial collapse of the lung",
    "opacity": "cloudiness or shadow on scan",
    "consolidation": "lung tissue filled with liquid instead of air",
    "pneumothorax": "collapsed lung (air outside the lung)",
    "pleural effusion": "fluid around the lungs",
    "edema": "swelling caused by fluid buildup",
    "erythema": "redness of the skin",
    "pruritus": "itching",
    "cephalalgia": "headache",
    "syncope": "fainting",
    "vertigo": "dizziness or spinning sensation",
    "hematoma": "a bruise or collection of blood outside blood vessels",
    "fibrosis": "scarring of tissue",
    "neoplasm": "tumor or abnormal growth",
    "benign": "non-cancerous / harmless",
    "malignant": "cancerous / dangerous",
    "metastasis": "spread of cancer to other parts of the body",
    "renal": "kidney-related",
    "hepatic": "liver-related",
    "pulmonary": "lung-related",
    "cardiac": "heart-related",
    "ophthalmic": "eye-related",
    "febrile": "feverish",
    "analgesic": "pain reliever",
    "antipyretic": "fever reducer",
    "hyperglycemia": "high blood sugar",
    "hypoglycemia": "low blood sugar",
    "hypernatremia": "high sodium levels in blood",
    "hyponatremia": "low sodium levels in blood",
    "thrombocytopenia": "low platelet count (risk of bleeding)",
    "leukocytosis": "high white blood cell count (often indicates infection)",
    "leukopenia": "low white blood cell count (risk of infection)",
    "anemia": "low red blood cell count (causes fatigue)",
    "erythrocytes": "red blood cells",
    "leukocytes": "white blood cells",
    "thrombocytes": "platelets",
    "hematuria": "blood in the urine",
    "proteinuria": "excess protein in the urine",
    "dyspepsia": "indigestion",
    "gastroesophageal reflux": "acid reflux",
    "gastritis": "inflammation of the stomach lining",
    "otitis media": "middle ear infection",
    "pharyngitis": "sore throat",
    "myalgia": "muscle pain",
    "arthralgia": "joint pain",
    "osteoarthritis": "wear-and-tear arthritis",
    "osteoporosis": "weak or brittle bones",
    "atherosclerosis": "hardening of the arteries",
    "hyperlipidemia": "high cholesterol/fats in the blood",
    "hypercholesterolemia": "high cholesterol",
    "cholecystectomy": "surgical removal of the gallbladder",
    "appendectomy": "surgical removal of the appendix",
    "laceration": "cut or tear in the skin",
    "contusion": "bruise",
    "fracture": "broken bone",
    "neuropathy": "nerve damage (causes numbness or tingling)",
    "ischemia": "insufficient blood supply to an organ",
    "thrombosis": "blood clot inside a blood vessel",
    "embolism": "blocked artery (often from a traveling blood clot)",
    "stenosis": "abnormal narrowing of a passage",
    "aneurysm": "bulging/weakened wall of an artery",
    "lesion": "area of damaged tissue or abnormal change",
    "nodule": "small lump or growth",
    "cyst": "fluid-filled sac",
    "polyp": "small growth projecting from a mucous membrane",
    "carcinoma": "cancer originating in skin or lining tissues",
    "idiopathic": "of unknown cause",
    "etiology": "cause of a disease",
    "prognosis": "expected outcome of a disease",
    "acute": "sudden and severe",
    "chronic": "long-term and ongoing",
    "bilateral": "affecting both sides",
    "unilateral": "affecting one side",
    "anterior": "front of the body",
    "posterior": "back of the body",
    "lateral": "outer side of the body",
    "medial": "middle or inner side",
    "proximal": "closer to the center of the body",
    "distal": "further from the center of the body",
    "subcutaneous": "under the skin",
    "intravenous": "into a vein",
    "intramuscular": "into a muscle",
    "oral": "by mouth",
    "asymptomatic": "showing no signs or symptoms of illness",
    "symptomatic": "showing signs of illness",
    "palliative": "relieving pain without curing the cause",
    "contraindication": "reason why a drug/treatment should not be used",
    "prophylaxis": "preventative treatment",
    "sputum": "mucus coughed up from the lungs",
    "prandial": "related to a meal",
    "postprandial": "after a meal",
    "renal insufficiency": "poor kidney function",
    "angina": "chest pain due to reduced blood flow to heart",
    "hemorrhage": "severe bleeding",
    "hyperthyroidism": "overactive thyroid gland",
    "hypothyroidism": "underactive thyroid gland",
    "edematous": "swollen with fluid",
    "lymphadenopathy": "swollen lymph nodes",
    "hepatomegaly": "enlarged liver",
    "splenomegaly": "enlarged spleen"
}

def get_medical_dictionary() -> dict[str, str]:
    """
    Returns the core medical terminology dictionary.
    """
    return MEDICAL_DICT

def dictionary_simplify(text: str) -> str:
    """
    Translates complex jargon terms using rule-based dictionary substitution.
    Uses regex word boundaries for case-insensitive replacements.
    """
    if not text:
        return ""
        
    simplified = text
    # Sort terms by length in descending order to avoid partial matching issues
    # (e.g., matching 'otitis media' before 'otitis')
    sorted_terms = sorted(MEDICAL_DICT.keys(), key=len, reverse=True)
    
    for term in sorted_terms:
        # Match word boundaries to prevent replacing sub-words
        pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
        # We replace with "plain_term (original_term)" for clarity or just plain_term
        # Let's replace with plain_term for simplification, or plain_term [original_term]
        replacement = MEDICAL_DICT[term]
        simplified = pattern.sub(replacement, simplified)
        
        # Also handle plural forms (basic check, e.g., 'nodules' or 'cysts')
        plural_term = term + 's'
        plural_pattern = re.compile(rf'\b{re.escape(plural_term)}\b', re.IGNORECASE)
        plural_replacement = replacement + 's'
        simplified = plural_pattern.sub(plural_replacement, simplified)
        
    return simplified

def highlight_medical_terms(text: str) -> str:
    """
    Surrounds medical terminology in the text with HTML tags for tooltips/styling.
    Example: <span class="med-highlight" title="swelling">edema</span>
    """
    if not text:
        return ""
        
    highlighted = text
    sorted_terms = sorted(MEDICAL_DICT.keys(), key=len, reverse=True)
    
    for term in sorted_terms:
        # Avoid double-highlighting by matching only text not already within an HTML span
        # Simple negative lookahead/lookbehind or regex replacement
        # Using a pattern that matches word boundaries, case-insensitive
        pattern = re.compile(rf'\b({re.escape(term)})(s)?\b', re.IGNORECASE)
        
        # We define a custom replacement function to preserve the exact case and plurals
        def repl(match):
            original = match.group(0)
            definition = MEDICAL_DICT[term]
            return f'<span class="med-highlight" title="Definition: {definition}">{original}</span>'
            
        highlighted = pattern.sub(repl, highlighted)
        
    return highlighted

def generate_glossary(text: str) -> dict[str, str]:
    """
    Scans the text for medical terms present in the dictionary and returns
    a subset dictionary of all terms found and their definitions.
    """
    if not text:
        return {}
        
    found_terms = {}
    # Use lowercase for matching
    text_lower = text.lower()
    
    for term, definition in MEDICAL_DICT.items():
        # Search using word boundaries
        if re.search(rf'\b{re.escape(term)}(s)?\b', text_lower):
            found_terms[term] = definition
            
    return found_terms

def format_report_download(
    original: str,
    simplified: str,
    glossary: dict[str, str],
    risk_level: str = "Unknown",
    risk_reason: str = "",
    detected_specialty: str = "General Medicine",
    lifestyle_recommendations: list[str] = None,
    metadata: dict[str, str] = None,
    translated_explanation: str = "",
    target_language: str = "English"
) -> str:
    """
    Formats the translation output into a readable text document ready for download.
    Cleans up any HTML formatting tags for the plaintext file output.
    """
    # Clean up HTML tags from simplified text for download format
    clean_simplified = simplified
    # Replace metadata list items: <li><b>Label:</b> Value</li> -> * Label: Value
    clean_simplified = re.sub(r'(?i)<li[^>]*>\s*<b>([^<]+)</b>\s*(.*?)\s*</li>', r'* \1 \2', clean_simplified)
    # Replace regular list items: <li>Text</li> -> * Text
    clean_simplified = re.sub(r'(?i)<li[^>]*>\s*(.*?)\s*</li>', r'* \1', clean_simplified)
    # Strip all remaining HTML tags (like <ul>, <div>, <span>)
    clean_simplified = re.sub(r'<[^>]+>', '', clean_simplified)
    # Replace multiple newlines with at most two newlines
    clean_simplified = re.sub(r'\n{3,}', '\n\n', clean_simplified)
    clean_simplified = clean_simplified.strip()

    doc = []
    doc.append("==================================================")
    doc.append("            MEDSIMPLIFY PATIENT REPORT            ")
    doc.append("==================================================")
    doc.append("\nDisclaimer: This tool is AI-assisted and for educational purposes only.")
    doc.append("It does not replace professional medical advice, diagnosis, or treatment.\n")
    
    doc.append("--------------------------------------------------")
    doc.append("PATIENT & REPORT DETAILS")
    doc.append("--------------------------------------------------")
    if metadata:
        doc.append(f"Patient Name: {metadata.get('name', 'Not specified')}")
        doc.append(f"Age:          {metadata.get('age', 'Not specified')}")
        doc.append(f"Gender:       {metadata.get('gender', 'Not specified')}")
        doc.append(f"Report Date:  {metadata.get('date', 'Not specified')}")
    else:
        doc.append("Patient Details: Not specified")
    doc.append(f"Detected Specialty: {detected_specialty}")
    
    doc.append("\n--------------------------------------------------")
    doc.append("1. CLINICAL RISK ASSESSMENT")
    doc.append("--------------------------------------------------")
    doc.append(f"Risk Category: {risk_level}")
    doc.append(f"Assessment Reason: {risk_reason}")
    
    doc.append("\n--------------------------------------------------")
    doc.append("2. PATIENT-FRIENDLY SIMPLIFIED REPORT (ENGLISH)")
    doc.append("--------------------------------------------------")
    doc.append(clean_simplified)
    
    if target_language.lower() != "english" and translated_explanation:
        doc.append("\n--------------------------------------------------")
        doc.append(f"2b. PATIENT-FRIENDLY TRANSLATED REPORT ({target_language.upper()})")
        doc.append("--------------------------------------------------")
        doc.append(translated_explanation.strip())
    
    doc.append("\n--------------------------------------------------")
    doc.append("3. LIFESTYLE RECOMMENDATIONS")
    doc.append("--------------------------------------------------")
    if lifestyle_recommendations:
        for rec in lifestyle_recommendations:
            doc.append(f"* {rec}")
    else:
        doc.append("No specific lifestyle recommendations generated.")
    
    doc.append("\n--------------------------------------------------")
    doc.append("4. MEDICAL GLOSSARY OF TERMS DETECTED")
    doc.append("--------------------------------------------------")
    if glossary:
        for term, definition in sorted(glossary.items()):
            doc.append(f"* {term.capitalize()}: {definition}")
    else:
        doc.append("No specific complex medical terms detected in dictionary.")
        
    doc.append("\n--------------------------------------------------")
    doc.append("5. ORIGINAL CLINICAL REPORT REFERENCE")
    doc.append("--------------------------------------------------")
    doc.append(original)
    doc.append("\n==================================================")
    doc.append("Developed By: Jayashri V. Hiremath (B.Tech CSE) & Tanishka Desai (B.Tech CSI) | Presidency University")
    doc.append("==================================================")
    
    return "\n".join(doc)

def highlight_report_headers(text: str) -> str:
    """
    Detects standard section headers (Clinical History, Findings, Impression, 
    Assessment, Diagnosis, Recommendations, Clinical Notes, Clinical Note) 
    case-insensitively at the beginning of any line and wraps them in HTML 
    strong tags with a teal color (#2dd4bf) and nice vertical spacing.
    """
    if not text:
        return ""
        
    highlighted = text
    headers = [
        "Clinical History", "Findings", "Impression", "Assessment", 
        "Diagnosis", "Recommendations", "Clinical Notes", "Clinical Note"
    ]
    
    for h in headers:
        # Match from the start of a line, optional space, then header name, optional colon
        pattern = re.compile(rf'^(?:[ \t]*)(?:{re.escape(h)})\b\s*:?', re.IGNORECASE | re.MULTILINE)
        
        def repl(match):
            original = match.group(0)
            display_title = original.strip().rstrip(':')
            return f'<strong style="color: #2dd4bf; font-size: 1.05rem; display: block; margin-top: 14px; margin-bottom: 6px;">{display_title}:</strong>'
            
        highlighted = pattern.sub(repl, highlighted)
        
    return highlighted

