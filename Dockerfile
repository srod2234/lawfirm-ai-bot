# Use a lightweight Python base
FROM python:3.10-slim

# Install Tesseract OCR and its data files
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
 && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements (or use pip install directly)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]