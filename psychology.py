import os
import json
import re
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def build_prompt(sentence: str) -> str:
    return f"""
You are a clinical psychology expert.

Analyze the sentence and detect ALL psychological signals, even if subtle.

Sentence: "{sentence}"

Rules:
- Mark TRUE only if the signal is clearly expressed, not just implied weakly.
- self_reference_density = proportion of self-focus (0–1)

Return ONLY JSON:

{{
 "absolutist": true/false,
 "helplessness": true/false,
 "catastrophizing": true/false,
 "external_locus": true/false,
 "rumination": true/false,
 "uncertainty": true/false,
 "avoidance": true/false,
 "threat_anticipation": true/false,
 "self_reference_density": number,
 "time_focus": "past" or "present" or "future"
}}
"""

def analyze_sentence(sentence):
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": build_prompt(sentence)}]
        )

        text = response.content[0].text

        # ✅ Extract JSON safely
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")

        return json.loads(match.group())

    except Exception as e:
        print("LLM ERROR:", str(e))
        return {
            "absolutist": False,
            "helplessness": False,
            "catastrophizing": False,
            "external_locus": False,
            "rumination": False,
            "uncertainty": False,
            "avoidance": False,
            "threat_anticipation": False,
            "self_reference_density": 0.0,
            "time_focus": "present"
        }