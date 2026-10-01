import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


def generate_gemini_summary(transcript: str):
    prompt = f"""
You are EduSummarize AI.

Summarize this lecture transcript. It may contain:
- Hindi
- English
- Hindi-English mixed speech
- Technical terms

Rules:
- Do not invent facts.
- Preserve important technical terms.
- Keep the summary in the same language style as the transcript.
- Include important definitions and concepts.
- Return valid JSON only.

Required JSON format:
{{
  "title": "short lecture title",
  "summary": "concise summary",
  "key_points": [
    "important point 1",
    "important point 2"
  ],
  "keywords": [
    "keyword 1",
    "keyword 2"
  ]
}}

Transcript:
{transcript}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json"
        )
    )

    return json.loads(response.text)