# ingest.py

import fitz                        # PyMuPDF
from PIL import Image             # Pillow
import pytesseract                # Tesseract OCR

from sqlmodel import Session
from models import engine, Document, Page

def ingest_pdf(path: str, owner_id: int) -> int:
    """
    Ingest a PDF at `path` for user `owner_id`:
     - Creates a Document record
     - Splits into pages
     - Extracts embedded text or OCRs scanned pages
     - Stores each page in the Page table
    Returns the created Document.id.
    """
    # 1. Create the Document entry
    doc = Document(owner_id=owner_id, label=path.split("/")[-1], file_path=path)

    # 2. Persist doc & capture ID
    with Session(engine) as sess:
        sess.add(doc)
        sess.commit()
        sess.refresh(doc)
        doc_id = doc.id

        # 3. Process each PDF page
        pdf = fitz.open(path)
        for i, pg in enumerate(pdf):
            raw_text = pg.get_text().strip()
            is_scanned = not bool(raw_text)
            if is_scanned:
                pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                raw_text = pytesseract.image_to_string(img)

            page = Page(
                document_id=doc_id,
                page_number=i + 1,
                text=raw_text or "[No text extracted]",
                is_scanned=is_scanned
            )
            sess.add(page)

        sess.commit()

    return doc_id
