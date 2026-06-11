import re
from pypdf import PdfReader

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts plain text from an uploaded PDF file or path using pypdf.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n" + page_text
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

def clean_report_text(text: str) -> str:
    """
    Cleans raw medical report text by:
    - Normalizing whitespaces, carriage returns, and line endings.
    - Formatting patient details and merging metadata split over newlines.
    - Reconstructing sentences split by PDF column/line margins.
    - Removing excessive line breaks and duplicate empty lines.
    - Removing isolated words or stray characters caused by PDF extraction.
    """
    if not text:
        return ""
    
    # 1. Normalize line endings and standard spaces
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 2. Merge patient details / metadata split over multiple newlines (e.g. Patient Name:\nDavid Wilson)
    # Target common labels
    labels = [
        "Patient Name", "Name", "Age", "Gender", "Sex", "Date", 
        "Physician", "Doctor", "Report ID", "Patient ID", "Hospital", 
        "Clinic", "Contact Information", "Contact"
    ]
    for label in labels:
        # Match case-insensitive label optionally followed by a colon, then any number of newlines/whitespace, then value line
        pattern = re.compile(rf'\b({re.escape(label)})\b\s*:?\s*\n+\s*([^\n\r]+)', re.IGNORECASE)
        text = pattern.sub(r'\1: \2', text)
        
    # Merge units like "Years", "Years Old", "Yrs", "Yr", "Male", "Female" if split after the label line
    text = re.sub(r'(?i)\b(Age:\s*\d+)\s*\n+\s*(years\s*old|years|year|yrs|yr)\b', r'\1 \2', text)
    text = re.sub(r'(?i)\b(Gender|Sex):\s*\n+\s*(male|female)\b', r'\1: \2', text)
    
    # 3. Strip leading/trailing whitespaces from each line
    lines = [line.strip() for line in text.split('\n')]
    
    # 4. Remove isolated characters/words and reconstruct broken sentences
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip empty lines for this processing stage
        if not line:
            cleaned_lines.append("")
            i += 1
            continue
            
        # Skip lines that are stray page markers
        if re.match(r'(?i)^page\s+\d+\s+of\s+\d+$', line):
            i += 1
            continue
            
        # Try to merge with the next line(s) if we are in the middle of a sentence
        while i + 1 < len(lines):
            next_line = lines[i+1].strip()
            if not next_line:
                break
                
            # If the current line does not end with sentence punctuation or standard markers,
            # and the next line doesn't start with an uppercase letter representing a new sentence/heading
            current_ends_punctuation = line[-1] in ['.', '!', '?', ':', ';', '-', '*', '=', '+'] if line else True
            next_starts_lowercase = next_line[0].islower() if next_line else False
            
            # Common headers or labels to avoid merging into sentences
            next_is_header = any(next_line.lower().startswith(h.lower()) for h in [
                "clinical history", "findings", "impression", "assessment", 
                "diagnosis", "recommendations", "clinical notes", "clinical note",
                "patient name:", "age:", "gender:", "date:", "physician:", "doctor:",
                "report id:", "patient id:", "hospital:", "clinic:", "contact information:"
            ])
            
            # Merge if:
            # - Current line ends mid-sentence and next line starts lowercase and is not a header
            # - Next line is a single short stray word or punctuation mark
            if (not current_ends_punctuation and next_starts_lowercase and not next_is_header) or \
               (len(next_line.split()) == 1 and not current_ends_punctuation and not next_is_header and len(next_line) < 15):
                line = line + " " + next_line
                i += 1
            else:
                break
                
        # Skip isolated/stray single characters caused by PDF parsing error (unless it's a number/bullet)
        if len(line) == 1 and not line.isalnum() and line not in ['-', '*', '+']:
            i += 1
            continue
            
        cleaned_lines.append(line)
        i += 1
        
    # Combine back
    cleaned = '\n'.join(cleaned_lines)
    
    # 5. Remove multiple consecutive blank lines, collapse to at most a single blank line (\n\n)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Ensure there's a newline between patient metadata lines so they stack nicely, and remove blank lines between metadata
    pattern = re.compile(r'((?:Patient Name|Age|Gender|Sex|Date|Physician|Doctor|Report ID|Patient ID|Hospital|Clinic|Contact):\s*[^\n]+)\n\n+((?:Patient Name|Age|Gender|Sex|Date|Physician|Doctor|Report ID|Patient ID|Hospital|Clinic|Contact):\s*[^\n]+)', re.IGNORECASE)
    prev = ""
    while prev != cleaned:
        prev = cleaned
        cleaned = pattern.sub(r'\1\n\2', cleaned)
    
    return cleaned.strip()

def parse_medical_report_sections(text: str) -> dict[str, str]:
    """
    Parses a medical report into standard sections: Clinical History, Findings, and Impression.
    This helps in organizing and summarizing report components separately.
    """
    sections = {
        "history": "",
        "findings": "",
        "impression": ""
    }
    
    if not text:
        return sections
        
    lines = text.split('\n')
    current_section = None
    section_buffers = {
        "history": [],
        "findings": [],
        "impression": []
    }
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        lower_line = line_strip.lower()
        
        # Check for section boundaries
        if any(h in lower_line for h in ["clinical history", "history:", "history of", "presentation"]):
            current_section = "history"
            # Remove the header title from buffer to keep content clean
            continue
        elif any(h in lower_line for h in ["findings:", "examination:", "results:", "test results:", "findings"]):
            current_section = "findings"
            continue
        elif any(h in lower_line for h in ["impression:", "impression", "impressions:", "clinical notes:", "clinical notes", "summary:"]):
            current_section = "impression"
            continue
            
        # Append line to active section buffer
        if current_section:
            section_buffers[current_section].append(line_strip)
        else:
            # If no header was encountered yet, put everything in findings by default
            section_buffers["findings"].append(line_strip)
            
    # Combine buffers
    sections["history"] = "\n".join(section_buffers["history"]).strip()
    sections["findings"] = "\n".join(section_buffers["findings"]).strip()
    sections["impression"] = "\n".join(section_buffers["impression"]).strip()
    
    return sections

def extract_patient_metadata(text: str) -> dict[str, str]:
    """
    Scans clinical text to extract patient name, age, gender, and report date.
    Returns a dictionary of findings with default placeholders.
    """
    metadata = {
        "name": "Not specified",
        "age": "Not specified",
        "gender": "Not specified",
        "date": "Not specified"
    }
    
    if not text:
        return metadata
        
    # Try finding Patient Name
    name_match = re.search(r'(?i)(?:patient name|patient|name):\s*([^\n\r,]+)', text)
    if name_match:
        metadata["name"] = name_match.group(1).strip()
        
    # Try finding Age & Gender patterns like "68-year-old male"
    age_gender_match = re.search(r'(?i)\b(\d{1,3})\s*-?\s*year\s*-?\s*old\s*(male|female)?\b', text)
    if age_gender_match:
        metadata["age"] = age_gender_match.group(1).strip() + " years old"
        if age_gender_match.group(2):
            metadata["gender"] = age_gender_match.group(2).strip().capitalize()
            
    # Try finding separate Age label
    if metadata["age"] == "Not specified":
        age_match = re.search(r'(?i)\bage:\s*(\d+)\b', text)
        if age_match:
            metadata["age"] = age_match.group(1).strip() + " years old"
            
    # Try finding separate Gender/Sex label
    if metadata["gender"] == "Not specified":
        gender_match = re.search(r'(?i)\b(?:sex|gender):\s*(male|female)\b', text)
        if gender_match:
            metadata["gender"] = gender_match.group(1).strip().capitalize()
            
    # Try finding Date
    date_match = re.search(r'(?i)(?:date of collection|date):\s*([^\n\r,]+)', text)
    if date_match:
        metadata["date"] = date_match.group(1).strip()
        
    return metadata

def convert_to_bullet_points(text: str) -> str:
    """
    Splits continuous translated text into HTML list items (bullet points) 
    by sentence or line dividers, keeping them clean.
    """
    if not text or not text.strip():
        return ""
        
    # Split by paragraphs
    paragraphs = text.split('\n\n')
    bullet_items = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Split paragraph into sentences
        # Matches periods/exclamations/question marks followed by space
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if sentence_clean:
                # Add trailing period if missing
                if sentence_clean[-1] not in ['.', '!', '?']:
                    sentence_clean += '.'
                bullet_items.append(f"<li>{sentence_clean}</li>")
                
    if bullet_items:
        return '<ul style="margin-top: 5px; margin-bottom: 10px; padding-left: 20px; color: #cbd5e1;">\n' + "\n".join(bullet_items) + '\n</ul>'
    return ""

def chunk_text(text: str, max_chars: int = 1500) -> list[str]:
    """
    Splits text into chunks of maximum characters while trying to keep paragraphs
    and sentences intact. This helps fit text within Flan-T5 token limits.
    """
    if not text:
        return []
        
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If a single paragraph is larger than max_chars, split by sentence
        if len(paragraph) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                
            # Split paragraph into sentences (basic regex for sentence ending)
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # If a single sentence is extremely long, force-split it
                    if len(sentence) > max_chars:
                        for i in range(0, len(sentence), max_chars):
                            chunks.append(sentence[i:i+max_chars].strip())
                        current_chunk = ""
                    else:
                        current_chunk = sentence + " "
        else:
            if len(current_chunk) + len(paragraph) + 2 <= max_chars:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return [c for c in chunks if c]

def ensure_sample_images():
    """
    Generates mock medical scan phantoms (Chest X-Ray, Knee Fracture, Infected Wound)
    if they do not already exist.
    """
    import os
    from PIL import Image, ImageDraw
    os.makedirs("sample_reports", exist_ok=True)
    
    chest_path = "sample_reports/sample_chest_xray.png"
    if not os.path.exists(chest_path):
        # Create a mock grayscale chest X-Ray (lung outlines, ribs, spine, heart shadow)
        img = Image.new("L", (256, 256), color=30)
        draw = ImageDraw.Draw(img)
        # Spine
        draw.line([(128, 10), (128, 246)], fill=160, width=6)
        # Ribs
        for y in range(40, 210, 24):
            draw.arc([30, y-12, 120, y+12], start=180, end=360, fill=130, width=3)
            draw.arc([136, y-12, 226, y+12], start=180, end=360, fill=130, width=3)
        # Heart shadow
        draw.ellipse([90, 130, 150, 190], fill=80)
        # Add slight cloudiness representing lung opacity
        draw.ellipse([45, 80, 85, 120], fill=60)
        img.save(chest_path)
        
    fracture_path = "sample_reports/sample_knee_fracture.png"
    if not os.path.exists(fracture_path):
        # Create a mock bone X-Ray with a fracture line
        img = Image.new("L", (256, 256), color=40)
        draw = ImageDraw.Draw(img)
        # Femur bone (top)
        draw.rounded_rectangle([96, 10, 160, 110], radius=8, fill=190)
        # Tibia bone (bottom)
        draw.rounded_rectangle([96, 136, 160, 246], radius=8, fill=190)
        # Patella bone (knee cap)
        draw.ellipse([70, 115, 95, 140], fill=150)
        # Fracture crack line on the Femur
        draw.line([(100, 70), (140, 85)], fill=40, width=3)
        img.save(fracture_path)
        
    wound_path = "sample_reports/sample_wound.png"
    if not os.path.exists(wound_path):
        # Create a mock skin wound image (RGB skin patch with red abrasion/infection)
        img = Image.new("RGB", (256, 256), color=(224, 165, 135))
        draw = ImageDraw.Draw(img)
        # Red raw skin region
        draw.ellipse([70, 70, 186, 186], fill=(168, 54, 54))
        # Yellow infection pus spots
        draw.ellipse([95, 95, 115, 115], fill=(188, 188, 70))
        draw.ellipse([140, 130, 155, 145], fill=(178, 188, 60))
        # Healing pink edges
        draw.arc([66, 66, 190, 190], start=0, end=360, fill=(210, 100, 110), width=4)
        img.save(wound_path)

