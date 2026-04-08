import pdfplumber
import easyocr
import numpy as np
from PIL import Image
import requests
import os
from dotenv import load_dotenv

load_dotenv()

#CONFIG
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

def llm_call(prompt):
    try:
        response = requests.post(API_URL, headers=headers, json={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3000, 
            "temperature": 0.1
        }, timeout=90)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM Call Error: {e}")
        return ""

# Initialize OCR Reader (Loaded once)
reader = easyocr.Reader(['en'])

def extract_text(uploaded_file):
    """Determines file type and extracts text accordingly."""
    file_type = uploaded_file.type
    text = ""

    if "pdf" in file_type:
        with pdfplumber.open(uploaded_file) as pdf:
            # Try digital extraction first
            text = " ".join([page.extract_text() or "" for page in pdf.pages])
            
            # Fallback to OCR if PDF is a scan (no text found)
            if not text.strip():
                # Convert the first page to a numpy array for EasyOCR
                first_page = pdf.pages[0].to_image().original
                text = " ".join(reader.readtext(np.array(first_page), detail=0))
    else:
        
        img = Image.open(uploaded_file)
        text = " ".join(reader.readtext(np.array(img), detail=0))
    
    return text

def generate_invoice_summary(text):
    if not text.strip():
        return "Error: No text extracted."
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise data extraction engine for a new specialized project. 
Extract the key details from the user's text and return them in a structured list and make sure to summarize the entire document or image in atleast 200 words in the same font.
If information is missing, do not hallucinate; write 'Not Found'.<|eot_id|><|start_header_id|>user<|end_header_id|>

DATA TO ANALYZE:
{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a professional data analyzer."},
            {"role": "user", "content": prompt}
        ],
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.1
        }
    }

    #API CALL
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Router Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"System Error: {str(e)}"