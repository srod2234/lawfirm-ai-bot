import fitz  # PyMuPDF
import os

def extract_text_from_pdfs(folder_path="data"):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            doc = fitz.open(os.path.join(folder_path, filename))
            for page in doc:
                all_text += page.get_text()
            doc.close()
    return all_text
