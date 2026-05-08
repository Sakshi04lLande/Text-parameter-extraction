FROM python:3.10-slim

RUN apt-get update && apt-get install -y 
build-essential 
curl 
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip 
&& pip install --no-cache-dir -r requirements.txt

# Download models at build time

COPY stanza_download.py .
RUN python stanza_download.py

COPY . .



EXPOSE 8025

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8025"]
