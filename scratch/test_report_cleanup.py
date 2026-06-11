import sys
import os

# Add MedSimplify root to python path
sys.path.append("c:/MedSimplify")

from preprocessing.clean_text import clean_report_text
from utils.helper_functions import highlight_report_headers

def test_report_formatting():
    print("=== Testing Clinical Report Normalization ===")
    
    # Input with split metadata, duplicate line breaks, and broken sentences
    input_text = """
Patient Name:
David Wilson

Age:
36

Years

Gender:
Male

Date:
09



Clinical History:
Patient presented with acute
shortness of breath.


Findings:
ECG shows acute myocardial
infarction.
The lungs are clear of any consolidations.
No
pleural effusion.

Impression:
Cardiovascular abnormality.
"""
    
    print("\n--- Original Raw Input ---")
    print(input_text)
    
    # Clean text
    cleaned = clean_report_text(input_text)
    print("\n--- Cleaned & Reconstructed Output ---")
    print(cleaned)
    
    # Highlight headers
    highlighted = highlight_report_headers(cleaned)
    print("\n--- Highlighted Section Headers HTML ---")
    print(highlighted)
    
    # Simple assertions to verify
    assert "Patient Name: David Wilson" in cleaned, "Failed to merge patient name!"
    assert "Age: 36 Years" in cleaned, "Failed to merge age and years!"
    assert "Gender: Male" in cleaned, "Failed to merge gender!"
    assert "Date: 09" in cleaned, "Failed to merge date!"
    assert "Patient presented with acute shortness of breath." in cleaned, "Failed to merge broken sentence!"
    assert "ECG shows acute myocardial infarction." in cleaned, "Failed to merge broken sentence in findings!"
    assert 'color: #2dd4bf' in highlighted, "Failed to apply header style color!"
    assert 'Findings:' in highlighted, "Failed to format Findings header!"
    assert 'Clinical History:' in highlighted, "Failed to format Clinical History header!"
    
    print("\nSUCCESS: All normalization and styling tests passed successfully!")

if __name__ == "__main__":
    test_report_formatting()
