import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import re

# Import our helper functions
from utils.helper_functions import dictionary_simplify
from preprocessing.clean_text import (
    chunk_text, 
    extract_patient_metadata,
    convert_to_bullet_points
)

@st.cache_resource(show_spinner="Loading google/flan-t5-base model... (This may take several minutes on the first run)")
def load_simplification_model():
    """
    Loads and caches the google/flan-t5-base model and tokenizer directly.
    Uses GPU if available.
    """
    model_name = "google/flan-t5-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with correct precision based on device
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    return tokenizer, model

@st.cache_resource(show_spinner="Loading translation model... (This may take several minutes on the first run)")
def load_translation_model():
    """
    Loads and caches the NLLB-200 model and tokenizer for English to Indic translation.
    Uses GPU if available.
    """
    model_name = "facebook/nllb-200-distilled-600M"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    return tokenizer, model

def translate_text(text: str, target_lang: str) -> str:
    """
    Translates English text to a target language (Hindi, Kannada, Tamil, Telugu)
    using the cached NLLB-200 model.
    """
    if not text.strip():
        return ""
    if target_lang.lower() == "english":
        return text
        
    lang_mapping = {
        "hindi": "hin_Deva",
        "kannada": "kan_Knda",
        "tamil": "tam_Taml",
        "telugu": "tel_Telu"
    }
    
    tgt_lang_code = lang_mapping.get(target_lang.lower())
    if not tgt_lang_code:
        raise ValueError(f"Unsupported target language: {target_lang}")
        
    try:
        tokenizer, model = load_translation_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tokenize and run translation
        inputs = tokenizer(text, return_tensors="pt").to(device)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512
            )
            
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.strip()
    except Exception as e:
        raise RuntimeError(f"Translation model failed: {str(e)}")

def simplify_text_chunk_ai(model_pair, chunk: str) -> str:
    """
    Simplifies a medical report using the Flan-T5 model directly.
    Uses the user-specified prompt template and generation settings.
    """
    tokenizer, model = model_pair
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Exact user-specified prompt template
    prompt = (
        "You are a medical report simplification assistant.\n\n"
        "Your task is to convert ONLY the medical findings, diagnosis, assessment, impression, observations, and recommendations into simple patient-friendly language.\n\n"
        "Ignore and do NOT include:\n\n"
        "* Patient Name\n"
        "* Age\n"
        "* Gender\n"
        "* Date\n"
        "* Hospital Name\n"
        "* Doctor Name\n"
        "* Physician Name\n"
        "* Patient ID\n"
        "* Contact Information\n"
        "* Headers and administrative details\n\n"
        "Focus only on:\n\n"
        "* Symptoms\n"
        "* Findings\n"
        "* Diagnosis\n"
        "* Assessment\n"
        "* Impression\n"
        "* Recommendations\n\n"
        "Return a concise explanation in 3-6 simple sentences.\n\n"
        f"Medical Report:\n\n{chunk}\n\n"
        "Patient-Friendly Explanation:"
    )
    
    try:
        # Tokenize and run generation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
            
        simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return simplified.strip()
    except Exception as e:
        raise RuntimeError(f"Model generation error: {str(e)}")

def run_risk_assessment_ai(model_pair, text: str) -> str:
    """
    Runs risk assessment using the user's specific prompt template and generation settings.
    """
    tokenizer, model = model_pair
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # User-specified risk template
    prompt = (
        "You are a medical risk assessment assistant.\n\n"
        "Analyze the medical findings, diagnosis, impression, and recommendations from the report.\n\n"
        "Classify the overall health risk into ONLY one category:\n\n"
        "🟢 Low Risk\n"
        "🟡 Moderate Risk\n"
        "🔴 High Risk\n\n"
        "Then provide:\n\n"
        "1. Risk Level\n"
        "2. Short Reason (1-2 sentences)\n\n"
        "Do not mention patient name, age, gender, hospital details, or administrative information.\n\n"
        f"Medical Report:\n\n{text}\n\n"
        "Output Format:\n\n"
        "Risk Level: [Low Risk / Moderate Risk / High Risk]\n\n"
        "Reason:\n"
        "[Simple explanation]"
    )
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
            
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()
    except Exception as e:
        raise RuntimeError(f"Model risk assessment generation error: {str(e)}")

def clean_admin_info_from_text(text: str) -> str:
    """
    Removes lines containing administrative details to focus strictly on clinical sections.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Keywords to check for removal of matching lines (case-insensitive)
    remove_keywords = [
        "patient name", "age", "gender", "date", "physician", "doctor", 
        "hospital", "report id", "contact information", "patient id", 
        "phone", "email", "address", "clinic"
    ]
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            cleaned_lines.append("")
            continue
            
        lower_line = line_strip.lower()
        
        # Check if line contains any of the search terms
        if any(keyword in lower_line for keyword in remove_keywords):
            continue
            
        cleaned_lines.append(line_strip)
        
    return "\n".join(cleaned_lines).strip()

def extract_relevant_sections(text: str) -> str:
    """
    Keeps only clinical findings and impressions sections.
    """
    lines = text.split('\n')
    relevant_lines = []
    
    keep_headers = [
        "findings", "impression", "assessment", "diagnosis", 
        "recommendations", "clinical notes", "clinical note", 
        "results", "test results"
    ]
    exclude_headers = [
        "clinical history", "history", "patient info", 
        "patient metadata", "examination", "demographics", 
        "contact", "admission"
    ]
    
    text_lower = text.lower()
    has_keep_headers = any(h in text_lower for h in keep_headers)
    
    keep_lines = not has_keep_headers  # If no headers match at all, keep everything
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            if keep_lines:
                relevant_lines.append("")
            continue
            
        lower_line = line_strip.lower()
        
        # Check if line is a section header
        if any(h in lower_line for h in keep_headers):
            keep_lines = True
        elif any(h in lower_line for h in exclude_headers):
            keep_lines = False
            
        if keep_lines:
            relevant_lines.append(line_strip)
            
    # Clean up double newlines
    result = re.sub(r'\n{3,}', '\n\n', "\n".join(relevant_lines))
    return result.strip()

def dictionary_risk_assessment(text: str) -> tuple[str, str]:
    """
    Fallback risk classifier that matches specific clinical keywords to assign risk categories
    and reasons based on strict fallback rules.
    """
    text_lower = text.lower()
    
    # 1. Detect High Risk conditions
    high_conditions = []
    if "chronic kidney disease" in text_lower or ("kidney" in text_lower and "disease" in text_lower) or "renal failure" in text_lower or "ckd" in text_lower:
        high_conditions.append("Chronic kidney disease")
    if "heart failure" in text_lower or "cardiac failure" in text_lower or "congestive heart" in text_lower or "myocardial infarction" in text_lower or "heart attack" in text_lower:
        high_conditions.append("Heart failure")
    if "stroke" in text_lower or "cerebrovascular" in text_lower or "cva" in text_lower:
        high_conditions.append("Stroke")
    if "cancer" in text_lower or "malignant" in text_lower or "carcinoma" in text_lower or "neoplasm" in text_lower or "metastasis" in text_lower or "tumor" in text_lower:
        high_conditions.append("Cancer")
    if "coronary artery" in text_lower or "coronary disease" in text_lower or "cad" in text_lower:
        high_conditions.append("Coronary artery disease")
    if "atrial fibrillation" in text_lower or "afib" in text_lower or "a-fib" in text_lower:
        high_conditions.append("Atrial fibrillation")
        
    # 2. Detect Moderate Risk conditions
    mod_conditions = []
    if "diabetes" in text_lower or "diabetic" in text_lower or "hyperglycemia" in text_lower or "blood sugar" in text_lower:
        mod_conditions.append("Diabetes")
    if "hypertension" in text_lower or "high blood pressure" in text_lower or "htn" in text_lower:
        mod_conditions.append("Hypertension")
    if "hyperlipidemia" in text_lower or "high cholesterol" in text_lower or "lipid" in text_lower or "hypercholesterolemia" in text_lower:
        mod_conditions.append("Hyperlipidemia")
    if "tachycardia" in text_lower or "fast heart rate" in text_lower or "rapid heart" in text_lower:
        mod_conditions.append("Tachycardia")
    if "copd" in text_lower or "chronic obstructive pulmonary" in text_lower or "emphysema" in text_lower or "bronchitis" in text_lower:
        mod_conditions.append("COPD")
    if "obesity" in text_lower or "obese" in text_lower or "overweight" in text_lower or "bmi" in text_lower:
        mod_conditions.append("Obesity")
        
    # 3. Detect Low Risk conditions
    low_conditions = []
    if "deficiency" in text_lower or "anemia" in text_lower or "low vitamin" in text_lower:
        low_conditions.append("Mild deficiencies")
    if "benign" in text_lower or "cyst" in text_lower or "polyp" in text_lower or "nodule" in text_lower or "lesion" in text_lower or "minor finding" in text_lower:
        low_conditions.append("Minor findings")
    if "routine" in text_lower or "observation" in text_lower or "asymptomatic" in text_lower or "stable" in text_lower or "normal" in text_lower:
        low_conditions.append("Routine observations")
        
    # Apply Fallback classification rules in order of severity:
    # 1. High Risk Rule
    if len(high_conditions) >= 2:
        return "High Risk", f"Multiple severe conditions identified: {', '.join(high_conditions)}. These represent high health risks."
    elif len(high_conditions) == 1:
        return "High Risk", f"Severe clinical condition identified: {high_conditions[0]}. This requires close medical supervision."
        
    # 2. Moderate Risk Rule
    if len(mod_conditions) >= 2:
        return "Moderate Risk", f"Multiple metabolic or respiratory disorders identified: {', '.join(mod_conditions)}. These require ongoing medical management."
    elif len(mod_conditions) == 1:
        return "Moderate Risk", f"Moderate health concern identified: {mod_conditions[0]}. This requires monitoring and lifestyle or medical intervention."
        
    # 3. Low Risk Rule
    if len(low_conditions) >= 2:
        return "Low Risk", f"Multiple minor findings or routine observations identified: {', '.join(low_conditions)}. No immediate clinical danger."
    elif len(low_conditions) == 1:
        return "Low Risk", f"Routine observation or minor finding identified: {low_conditions[0]}. These findings appear stable."
    elif len(low_conditions) > 0:
        return "Low Risk", f"Routine observations or minor findings identified: {', '.join(low_conditions)}. These findings appear stable."
        
    # 4. Default: DO NOT default to Low Risk when AI/keyword parsing fails.
    # Default to Moderate Risk with a clear caution reason.
    return "Moderate Risk", "No specific high, moderate, or low risk keywords were detected in the clinical report. Classified as Moderate Risk for clinical safety; please consult a healthcare professional."

def run_risk_assessment(text: str, method: str = "AI Model + Dictionary") -> tuple[str, str]:
    """
    Executes medical risk assessment and returns (Risk Level, Reason).
    Falls back to local keyword classification if AI parsing fails, ensuring accurate risk outputs.
    """
    clinical_text = extract_relevant_sections(clean_admin_info_from_text(text))
    if not clinical_text.strip():
        clinical_text = clean_admin_info_from_text(text)
        
    if method == "Dictionary Only (Fast Fallback)":
        return dictionary_risk_assessment(clinical_text)
        
    # Load model for AI assessment
    try:
        model_pair = load_simplification_model()
        raw_result = run_risk_assessment_ai(model_pair, clinical_text)
        
        # Check for risk category using a more flexible regex
        risk_match = re.search(r'(?i)(?:risk\s*level|risk\s*category|risk|level)\s*:\s*(low|moderate|high)\s*risk', raw_result)
        risk_val = None
        if risk_match:
            risk_val = risk_match.group(1).strip().lower()
        else:
            # Try a looser search for "low risk", "moderate risk", "high risk"
            loose_match = re.search(r'(?i)\b(low|moderate|high)\s*risk\b', raw_result)
            if loose_match:
                risk_val = loose_match.group(1).strip().lower()
                
        # Try to extract the reason
        reason_val = None
        reason_match = re.search(r'(?i)reason\s*:\s*([\s\S]+)', raw_result)
        if reason_match:
            reason_val = reason_match.group(1).strip()
        else:
            # Extract reason by stripping the matched risk level from the raw result
            if risk_val:
                clean_reason = re.sub(r'(?i)risk\s*level:\s*(low|moderate|high)\s*risk', '', raw_result)
                clean_reason = re.sub(r'(?i)\b(low|moderate|high)\s*risk\b', '', clean_reason)
                clean_reason = clean_reason.strip()
                if clean_reason:
                    reason_val = clean_reason
                    
        if risk_val and reason_val:
            if risk_val == "high":
                risk_level = "High Risk"
            elif risk_val == "moderate":
                risk_level = "Moderate Risk"
            else:
                risk_level = "Low Risk"
            return risk_level, reason_val
        else:
            # AI parsing failed to find correct format or risk level, run fallback
            return dictionary_risk_assessment(clinical_text)
    except Exception as e:
        # Model load or inference failed, run fallback
        return dictionary_risk_assessment(clinical_text)

def detect_medical_specialty(text: str) -> str:
    """
    Scans report text to automatically identify the medical specialty.
    Categorizes into Cardiology, Neurology, Pulmonology, Nephrology, Hematology, Radiology,
    and defaults to General Medicine.
    """
    text_lower = text.lower()
    
    specialties = {
        "Cardiology": [
            "cardiac", "heart", "myocardial", "tachycardia", "bradycardia", 
            "electrocardiogram", "ecg", "ekg", "coronary", "arrhythmia", 
            "atrial fibrillation", "afib", "murmur", "atherosclerosis", "angina",
            "valve", "atrium", "ventricle", "pacemaker", "infarction",
            "hypertension", "hypotension", "blood pressure"
        ],
        "Neurology": [
            "brain", "neurology", "neurological", "stroke", "cva", "tia", 
            "cerebral", "headache", "migraine", "seizure", "neuropathy", 
            "spinal cord", "syncope", "vertigo", "reflex", "reflexes", "eeg"
        ],
        "Pulmonology": [
            "lung", "pulmonary", "respiration", "respiratory", "dyspnea", 
            "copd", "asthma", "bronchus", "bronchial", "pleura", "pleural", 
            "atelectasis", "pneumonia", "sputum", "bronchiectasis", "emphysema"
        ],
        "Nephrology": [
            "kidney", "renal", "nephrology", "creatinine", "bun", "glomerular", 
            "gfr", "egfr", "proteinuria", "hematuria", "uremia", "dialysis", "nephron"
        ],
        "Hematology": [
            "blood", "anemia", "hemoglobin", "platelet", "thrombocytopenia", 
            "white blood cell", "wbc", "rbc", "leukocytosis", "leukopenia", 
            "leukemia", "lymphoma", "clotting", "coagulation", "hematocrit", "platelets"
        ],
        "Radiology": [
            "x-ray", "xray", "mri", "ct scan", "ultrasound", "radiography", 
            "chest film", "opacity", "consolidation", "imaging", "radiological", 
            "scan", "shadow", "contrast enhancement", "tomography", "sonogram"
        ]
    }
    
    scores = {spec: 0 for spec in specialties}
    for spec, keywords in specialties.items():
        for kw in keywords:
            scores[spec] += text_lower.count(kw)
            
    # Determine the specialty with highest matched keywords
    max_spec = "General Medicine"
    max_score = 0
    for spec, score in scores.items():
        if score > max_score:
            max_score = score
            max_spec = spec
            
    return max_spec

def generate_lifestyle_recommendations(text: str) -> list[str]:
    """
    Generates 3-5 action-oriented, patient-friendly lifestyle recommendations
    based on keywords detected in the clinical text. Contains no drug prescriptions.
    """
    text_lower = text.lower()
    recommendations = []
    
    # Heart Failure / CAD / Atrial Fibrillation / Cardiology
    if any(k in text_lower for k in ["heart failure", "cardiac", "myocardial", "heart attack", "coronary", "cad", "atrial fibrillation", "afib", "a-fib"]):
        recommendations.append("Maintain a low-sodium (salt) diet to help manage fluid levels and reduce work on the heart.")
        recommendations.append("Monitor your weight daily and report any sudden increases (e.g., 2-3 lbs in a day) to your healthcare provider.")
        recommendations.append("Engage in regular, light physical activity (such as gentle daily walks) only as approved by your physician.")
        recommendations.append("Incorporate stress reduction practices, limit alcohol intake, and strictly avoid tobacco products.")

    # Stroke / Neurology
    if any(k in text_lower for k in ["stroke", "cerebrovascular", "cva", "neurological", "brain", "tia"]):
        recommendations.append("Incorporate physical movement or gentle coordination stretching exercises daily to promote blood flow.")
        recommendations.append("Follow a heart-healthy nutrition plan (such as the Mediterranean diet) emphasizing whole grains and fresh produce.")
        recommendations.append("Monitor blood pressure and blood sugar parameters regularly to minimize secondary cardiovascular risks.")

    # Chronic Kidney Disease / Nephrology
    if any(k in text_lower for k in ["kidney", "renal", "creatinine", "bun", "gfr", "ckd"]):
        recommendations.append("Adopt a kidney-friendly diet (managing protein, sodium, and potassium levels as guided by a clinical dietitian).")
        recommendations.append("Stay adequately hydrated, but be sure to follow any specific fluid limit guidance from your kidney specialist.")
        recommendations.append("Avoid over-the-counter NSAID pain medications (like ibuprofen or naproxen), which can place extra strain on the kidneys.")

    # Cancer
    if any(k in text_lower for k in ["cancer", "malignant", "carcinoma", "neoplasm", "metastasis", "tumor"]):
        recommendations.append("Prioritize restful sleep and plan short periods of rest during the day to help manage clinical fatigue.")
        recommendations.append("Focus on nutrient-dense meals and small, frequent portions to support your immune system and strength.")
        recommendations.append("Incorporate light movement (such as brief 10-minute walks) to help preserve muscle function and enhance mood.")

    # Diabetes / Moderate metabolic
    if any(k in text_lower for k in ["diabetes", "diabetic", "hyperglycemia", "blood sugar"]):
        recommendations.append("Monitor your blood sugar levels regularly and keep a structured log for your doctor.")
        recommendations.append("Focus on a balanced diet rich in fiber, lean proteins, and complex carbohydrates while minimizing refined sugars.")
        recommendations.append("Engage in regular cardiovascular exercise (like brisk walking) to help improve insulin sensitivity.")

    # Hypertension
    if any(k in text_lower for k in ["hypertension", "high blood pressure", "htn"]):
        recommendations.append("Reduce your dietary sodium (salt) intake by avoiding processed foods and seasoning.")
        recommendations.append("Incorporate at least 30 minutes of moderate physical activity (like walking or cycling) most days of the week.")
        recommendations.append("Practice regular relaxation techniques (such as deep breathing exercises or meditation) to manage stress.")

    # Hyperlipidemia
    if any(k in text_lower for k in ["hyperlipidemia", "cholesterol", "lipid", "hypercholesterolemia"]):
        recommendations.append("Focus on foods high in soluble fiber (such as oats, beans, and fresh fruits) to help lower cholesterol.")
        recommendations.append("Limit foods high in saturated and trans fats, replacing them with healthy fats like olive oil in moderation.")
        recommendations.append("Incorporate consistent aerobic exercise into your weekly routine to boost heart health.")

    # Tachycardia
    if any(k in text_lower for k in ["tachycardia", "fast heart rate", "elevated heart"]):
        recommendations.append("Strictly avoid cardiac stimulants, including excessive caffeine, nicotine, and over-the-counter decongestants.")
        recommendations.append("Practice calming techniques (like progressive muscle relaxation or mindfulness) to manage heart rate fluctuations.")
        recommendations.append("Stay well-hydrated throughout the day, as dehydration can contribute to an elevated heart rate.")

    # COPD / Pulmonology
    if any(k in text_lower for k in ["copd", "emphysema", "pulmonary", "respiratory", "asthma", "bronchitis"]):
        recommendations.append("Strictly avoid exposure to cigarette smoke, secondhand smoke, and environmental air pollutants.")
        recommendations.append("Practice breathing exercises (like pursed-lip or diaphragmatic breathing) to help improve lung ventilation.")
        recommendations.append("Ensure you receive your recommended annual vaccinations (such as flu and pneumonia shots) to protect your lungs.")

    # Obesity
    if any(k in text_lower for k in ["obesity", "obese", "overweight", "bmi"]):
        recommendations.append("Work towards a gradual, sustainable weight reduction by maintaining a light caloric deficit with whole foods.")
        recommendations.append("Combine aerobic activities (like walking or swimming) with light strength exercises to support metabolism.")
        recommendations.append("Keep a daily food and activity journal to increase mindfulness of eating patterns.")

    # Deficiencies / Low Risk / General Fallback
    if not recommendations or len(recommendations) < 3:
        if "deficiency" in text_lower or "deficiencies" in text_lower:
            recommendations.append("Incorporate foods naturally rich in the deficient nutrient (or follow professional guidance on supplementation).")
        recommendations.append("Maintain a balanced diet rich in colorful vegetables, fresh fruits, whole grains, and lean proteins.")
        recommendations.append("Drink plenty of water throughout the day (aiming for 6 to 8 glasses) to support optimal metabolic function.")
        recommendations.append("Aim for 7 to 9 hours of quality, uninterrupted sleep every night to facilitate tissue repair and recovery.")
        recommendations.append("Incorporate at least 150 minutes of moderate physical activity (such as brisk walking) per week.")

    # Return a unique set, sliced to 3-5 recommendations
    seen = set()
    unique_recs = []
    for r in recommendations:
        if r not in seen:
            seen.add(r)
            unique_recs.append(r)
            
    return unique_recs[:5]

def run_simplification(text: str, method: str = "AI Model + Dictionary") -> tuple[str, str, bool]:
    """
    Main function to simplify medical reports.
    Extracts metadata, cleans administrative lines, filters keep sections, and translates content.
    Returns: (simplified_text, log_message, was_fallback_used)
    """
    if not text.strip():
        return "", "Empty input provided.", False

    # 1. Clean admin info
    cleaned_admin = clean_admin_info_from_text(text)
    
    # 2. Filter relevant clinical sections
    clinical_text = extract_relevant_sections(cleaned_admin)
    
    # Safeguard: If sections extraction left us with empty text, fall back to cleaned text
    if not clinical_text.strip():
        clinical_text = cleaned_admin
    
    # Check if model loads successfully (for AI mode)
    model_pair = None
    model_load_err = None
    if method == "AI Model + Dictionary (Recommended)":
        try:
            model_pair = load_simplification_model()
        except Exception as e:
            model_load_err = str(e)

    # Determine if fallback is active
    was_fallback = (method == "Dictionary Only (Fast Fallback)") or (model_pair is None)
    
    # Translate the clinical content
    if was_fallback:
        simplified_clinical = dictionary_simplify(clinical_text)
    else:
        try:
            # Chunk clinical text to fit token limits
            chunks = chunk_text(clinical_text, max_chars=1200)
            simplified_chunks = []
            
            for chunk in chunks:
                simplified_chunk = simplify_text_chunk_ai(model_pair, chunk)
                # Apply dictionary replacement on top of AI generation
                simplified_chunk_post = dictionary_simplify(simplified_chunk)
                simplified_chunks.append(simplified_chunk_post)
                
            simplified_clinical = "\n\n".join(simplified_chunks)
        except Exception as e:
            simplified_clinical = dictionary_simplify(clinical_text)
            was_fallback = True

    # Format simplified clinical findings into bullet points (3-5 sentences)
    bullet_clinical = convert_to_bullet_points(simplified_clinical)
        
    # Setup log messages
    if method == "Dictionary Only (Fast Fallback)":
        log_msg = "Simplified using rule-based terminology dictionary."
    elif model_pair is not None:
        log_msg = "Successfully simplified using google/flan-t5-base AI model with dictionary post-processing."
    else:
        log_msg = f"Model load failed ({model_load_err}). Automatically fell back to dictionary translation."
        
    return bullet_clinical, log_msg, was_fallback

@st.cache_resource(show_spinner="Loading zero-shot image classification model... (This may take a minute on the first run)")
def load_image_classifier():
    """
    Loads and caches the zero-shot image classification pipeline using CLIP.
    """
    from transformers import pipeline
    # Load CLIP model for zero-shot image classification
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

def analyze_medical_image(image) -> dict:
    """
    Performs zero-shot analysis on a medical image using CLIP.
    Returns a dictionary with type, prediction, confidence, risk_level, risk_reason, and patient_friendly_explanation.
    """
    # 1. Load the classifier
    classifier = load_image_classifier()
    
    # 2. Determine general image type (X-Ray vs Wound)
    type_labels = ["an X-ray medical scan", "a medical photograph of a wound or skin injury"]
    res_type = classifier(image, candidate_labels=type_labels)
    best_type = res_type[0]['label']
    
    detected_type = ""
    is_xray = True
    
    if best_type == "an X-ray medical scan":
        is_xray = True
    else:
        is_xray = False
        
    # 3. Subtype & Condition Classification
    if is_xray:
        # Determine X-Ray type
        xray_subtypes = {
            "Chest X-Ray": "chest x-ray",
            "Hand X-Ray": "hand x-ray",
            "Wrist X-Ray": "wrist x-ray",
            "Finger X-Ray": "finger x-ray",
            "Arm X-Ray": "arm x-ray",
            "Elbow X-Ray": "elbow x-ray",
            "Shoulder X-Ray": "shoulder x-ray",
            "Spine X-Ray": "spine x-ray",
            "Pelvis X-Ray": "pelvis x-ray",
            "Hip X-Ray": "hip x-ray",
            "Leg X-Ray": "leg x-ray",
            "Knee X-Ray": "knee x-ray",
            "Ankle X-Ray": "ankle x-ray",
            "Foot X-Ray": "foot x-ray",
            "Dental X-Ray": "dental x-ray",
            "General Bone X-Ray": "general bone x-ray"
        }
        res_subtype = classifier(image, candidate_labels=list(xray_subtypes.values()))
        best_subtype_val = res_subtype[0]['label']
        
        # Map back to display name
        detected_subtype = "General Bone X-Ray"
        for display_name, val in xray_subtypes.items():
            if val == best_subtype_val:
                detected_subtype = display_name
                break
                
        # Condition classification
        if detected_subtype == "Chest X-Ray":
            cond_map = {
                "normal chest": "Normal",
                "pneumonia lung infection": "Pneumonia",
                "tuberculosis lung infection": "Tuberculosis",
                "pleural effusion fluid around lungs": "Pleural Effusion",
                "lung opacity shadow": "Lung Opacity",
                "covid-19 lung findings": "COVID-like Findings",
                "cardiomegaly enlarged heart": "Cardiomegaly"
            }
            res_cond = classifier(image, candidate_labels=list(cond_map.keys()))
            best_cond_label = res_cond[0]['label']
            best_cond = cond_map[best_cond_label]
            cond_conf = res_cond[0]['score']
        elif detected_subtype == "Dental X-Ray":
            cond_map = {
                "normal tooth": "Normal",
                "dental caries cavity": "Dental Caries",
                "impacted tooth": "Impacted Tooth",
                "dental infection abscess": "Infection",
                "bone loss in jaw": "Bone Loss"
            }
            res_cond = classifier(image, candidate_labels=list(cond_map.keys()))
            best_cond_label = res_cond[0]['label']
            best_cond = cond_map[best_cond_label]
            cond_conf = res_cond[0]['score']
        else: # Bone X-Rays
            cond_map = {
                "normal bone": "Normal",
                "bone fracture": "Fracture",
                "joint dislocation": "Dislocation",
                "osteoarthritis joint wear": "Osteoarthritis",
                "bone abnormality": "Bone Abnormality"
            }
            res_cond = classifier(image, candidate_labels=list(cond_map.keys()))
            best_cond_label = res_cond[0]['label']
            best_cond = cond_map[best_cond_label]
            cond_conf = res_cond[0]['score']
            
        detected_type = f"{detected_subtype}"
        final_prediction = best_cond
        final_confidence = cond_conf
        
    else: # Wound
        wound_subtypes = {
            "Minor Cut": "minor cut wound",
            "Healing Wound": "healing wound",
            "Infected Wound": "infected wound",
            "Diabetic Ulcer": "diabetic ulcer wound",
            "Pressure Ulcer": "pressure ulcer wound",
            "Burn Injury": "burn injury wound",
            "Surgical Wound": "surgical wound",
            "Skin Infection": "skin infection wound"
        }
        res_subtype = classifier(image, candidate_labels=list(wound_subtypes.values()))
        best_subtype_val = res_subtype[0]['label']
        
        detected_subtype = "Healing Wound"
        for display_name, val in wound_subtypes.items():
            if val == best_subtype_val:
                detected_subtype = display_name
                break
                
        # Condition classification
        cond_map = {
            "wound healing normally": "Healing Normally",
            "infected wound": "Infected Wound",
            "burn injury": "Burn Injury",
            "diabetic ulcer": "Diabetic Ulcer",
            "pressure ulcer": "Pressure Ulcer",
            "skin infection": "Skin Infection"
        }
        res_cond = classifier(image, candidate_labels=list(cond_map.keys()))
        best_cond_label = res_cond[0]['label']
        best_cond = cond_map[best_cond_label]
        cond_conf = res_cond[0]['score']
        
        # Override detected type to just show the detected wound type
        detected_type = f"{detected_subtype}"
        final_prediction = best_cond
        final_confidence = cond_conf

    # 4. Risk mapping
    risk_level = "Low Risk"
    risk_reason = ""
    
    if final_prediction == "Normal" or final_prediction == "Healing Normally" or detected_type == "Minor Cut" or detected_type == "Healing Wound":
        risk_level = "Low Risk"
        risk_reason = "The analysis shows normal scan findings or a normally healing wound, representing a low overall health risk."
    elif final_prediction in ["Pneumonia", "Fracture", "Osteoarthritis", "Bone Abnormality", "Dental Caries", "Impacted Tooth", "Bone Loss", "Infected Wound", "Skin Infection", "Infection"]:
        risk_level = "Moderate Risk"
        risk_reason = f"A moderate health concern has been detected: {final_prediction}. This requires professional medical attention and follow-up care."
    elif final_prediction in ["Dislocation", "Burn Injury", "Tuberculosis", "Pleural Effusion", "Lung Opacity", "COVID-like Findings", "Cardiomegaly", "Diabetic Ulcer", "Pressure Ulcer"]:
        risk_level = "High Risk"
        risk_reason = f"A high-risk clinical finding has been detected: {final_prediction}. This indicates a potentially severe condition that requires urgent medical evaluation and intervention."
    else:
        risk_level = "Moderate Risk"
        risk_reason = f"An abnormal condition has been detected: {final_prediction}. Please consult a healthcare professional for diagnosis."

    # 5. Patient friendly explanation (3-5 simple sentences)
    explanations = {
        "Normal": "Your scan shows no signs of active disease, fractures, or structural abnormalities. All tissues appear within healthy limits. This is a routine finding and suggests normal function. Continue with standard health checkups.",
        "Healing Normally": "The photograph shows a wound that is healing normally with healthy new tissue formation. There are no active signs of infection, such as excessive redness, swelling, or drainage. Keep the area clean and protected to support recovery.",
        "Pneumonia": "The chest scan indicates signs of pneumonia, which is an inflammation of the lungs typically caused by an infection. This condition causes the air sacs to fill with fluid or pus, leading to symptoms like cough or fever. A medical evaluation is necessary to prescribe proper care (e.g. antibiotics or rest).",
        "Tuberculosis": "The chest scan suggests findings consistent with tuberculosis, which is a bacterial infection of the lungs. It is a serious condition that requires diagnostic lab verification and targeted treatment. Please consult a physician promptly for comprehensive follow-up.",
        "Pleural Effusion": "The X-ray indicates signs of pleural effusion, which is an abnormal collection of fluid in the space surrounding the lungs. This buildup can put pressure on the lungs and cause shortness of breath. Medical intervention is needed to determine the cause and determine if draining is required.",
        "Lung Opacity": "The scan displays an area of lung opacity, which refers to a shadow or cloudy spot in the lung fields. Shadows can represent localized fluid, inflammation, infection, or scarring. A healthcare professional should evaluate this alongside your clinical symptoms.",
        "COVID-like Findings": "The chest scan shows patterns commonly observed in viral lung infections, including COVID-19. These patterns correspond to patches of lung inflammation. A diagnostic lab test is recommended, and you should follow guidance on isolation if symptomatic.",
        "Cardiomegaly": "The scan shows cardiomegaly, which is an enlargement of the heart. This is not a disease itself but a sign of an underlying cardiovascular condition (such as high blood pressure or valve disease) that makes the heart work harder. Further testing is advised to check your heart function.",
        "Fracture": "The X-ray suggests a break or crack in the bone structure, known as a fracture. This requires immediate orthopedic support to decide if stabilization (like splinting, casting, or resetting) is needed. Rest the injured area and avoid applying weight.",
        "Dislocation": "The X-ray shows a joint dislocation, meaning the bones forming the joint have been forced out of their normal positions. This is an urgent medical condition requiring immediate specialist reduction. Do not attempt to move the limb or joint yourself.",
        "Osteoarthritis": "The X-ray displays signs of osteoarthritis, which is wear-and-tear of joint cartilage. This can lead to stiffness, localized pain, and reduced mobility. Management options include physical therapy, weight control, and joint care techniques.",
        "Bone Abnormality": "The scan indicates an unusual structure or changes in the density of the bone, classified as a bone abnormality. This requires a diagnostic check by a specialist to determine the exact cause and whether further scans are necessary.",
        "Dental Caries": "The scan shows dental caries, commonly known as cavities or tooth decay. This is caused by acid-producing bacteria eroding the tooth enamel. You should schedule a visit with your dentist to clean and fill the tooth before decay reaches the nerve.",
        "Impacted Tooth": "The dental scan shows an impacted tooth, meaning the tooth is blocked from properly growing through the gum line. This commonly affects wisdom teeth and can cause pain, crowding, or decay in neighboring teeth. A dentist should review if extraction is needed.",
        "Infection": "The scan indicates an active dental infection or abscess around the root of the tooth or gums. Dental infections require prompt intervention (like a root canal, drainage, or antibiotics) to prevent the spread of bacteria. Contact your dentist as soon as possible.",
        "Bone Loss": "The X-ray suggests localized bone loss in the jaw area, which often results from chronic gum disease or missing teeth. This reduces support for the teeth and can lead to tooth mobility. A dentist can suggest therapies to stabilize the bone.",
        "Infected Wound": "The photograph shows signs of a wound infection, such as increased redness, swelling, or drainage. Localized skin infections require professional care to determine if topical or oral antibiotics are necessary. Keep the wound clean and dry.",
        "Burn Injury": "The photograph shows a burn injury on the skin surface. Depending on the depth and surface area of the burn, medical dressings, skin protectants, and hydration are essential. Do not pop any blisters, and seek medical assessment.",
        "Diabetic Ulcer": "The image shows a diabetic ulcer, which is an open sore typically occurring on the feet of individuals with diabetes. These ulcers heal slowly and have a high risk of deep tissue infection. Careful management, offloading pressure, and regular podiatry visits are required.",
        "Pressure Ulcer": "The image displays a pressure ulcer, also called a bedsore, caused by sustained pressure restricting blood flow to the skin. These are critical wounds that require pressure relief, specialized wound dressings, and regular positioning support.",
        "Skin Infection": "The photograph shows a localized skin infection (such as cellulitis), indicated by spreading redness and warmth. Skin infections can progress rapidly and require diagnostic evaluation for appropriate prescription treatment. Monitor for systemic symptoms like fever."
    }
    
    explanation_text = explanations.get(final_prediction, "No specific automated explanation template found. Please consult your physician to interpret the findings.")
    
    return {
        "image_type": detected_type,
        "prediction": final_prediction,
        "confidence": float(final_confidence),
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "explanation": explanation_text
    }

