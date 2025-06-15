# Dockerfile

FROM python:3.10-slim

# Install Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose no fixed port; Streamlit will use $PORT
EXPOSE 8501

# Start Streamlit on the environment's $PORT
CMD ["bash", "-lc", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
