import sys
import os
from PIL import Image

# Add MedSimplify root to python path absolutely
sys.path.append("c:/MedSimplify")

import streamlit as st
from model.inference import analyze_medical_image

def test_image_analysis():
    print("=== Testing Medical Image Analysis Module ===")
    
    # Create a small dummy image in memory (224x224 RGB)
    dummy_img = Image.new("RGB", (224, 224), color="blue")
    
    print("\nRunning analysis on mock image...")
    try:
        result = analyze_medical_image(dummy_img)
        print("\nAnalysis successful! Result keys:")
        for k, v in result.items():
            print(f"- {k}: {v}")
            
        # Assertions
        assert "image_type" in result, "Result missing 'image_type'!"
        assert "prediction" in result, "Result missing 'prediction'!"
        assert "confidence" in result, "Result missing 'confidence'!"
        assert "risk_level" in result, "Result missing 'risk_level'!"
        assert "risk_reason" in result, "Result missing 'risk_reason'!"
        assert "explanation" in result, "Result missing 'explanation'!"
        
        # Verify specific fields
        assert isinstance(result["confidence"], float), "Confidence must be a float!"
        assert result["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"], "Invalid risk level!"
        assert len(result["explanation"]) > 20, "Explanation is too short!"
        
        print("\nSUCCESS: All automated image analysis validation checks passed successfully!")
    except Exception as e:
        print(f"\nFAILED: Exception during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_image_analysis()
