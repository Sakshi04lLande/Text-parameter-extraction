
import json
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API key from .env
API_KEY = os.getenv("MENTAL_HEALTH_ANALYZER_KEY")

# Load payload.json
with open("payload.json", "r", encoding="utf-8") as f:
    payload = json.load(f)

# API endpoint
url = "http://localhost:8025/analyze"

# Request headers
headers = {
    "x-api-key": API_KEY
}

try:
    # Send POST request
    response = requests.post(
        url,
        json=payload,
        headers=headers
    )

    # Print formatted response
    print(json.dumps(
        response.json(),
        indent=2,
        ensure_ascii=False
    ))

except Exception as e:
    print(f"Error: {e}")

