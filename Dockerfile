FROM python:3.10-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker cache optimization
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Download Stanza models
RUN python stanza_download.py

# Download HuggingFace emotion model
RUN python emotion_model_download.py

# Expose FastAPI port
EXPOSE 8025

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=5 CMD curl -f http://localhost:8025/docs || exit 1

# Start FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8025"]