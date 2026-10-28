import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TRANSCRIPT_CHARACTERS = 120_000
QUIZ_QUESTION_COUNT = 5
FLASHCARD_COUNT = 10


class GeminiSummaryError(Exception):
    """Base error for failures while generating a Gemini summary."""


class GeminiConfigurationError(GeminiSummaryError):
    """The Gemini client is not configured with an API key."""


class GeminiAuthenticationError(GeminiSummaryError):
    """Gemini rejected the configured API key."""


class GeminiResponseError(GeminiSummaryError):
    """Gemini returned an unusable summary."""


class GeminiQuizResponseError(GeminiResponseError):
    """Gemini returned an unusable quiz."""


class GeminiFlashcardResponseError(GeminiResponseError):
    """Gemini returned unusable flashcards."""


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is not configured."
        )
    try:
        return genai.Client(api_key=api_key)
    except ValueError as error:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is invalid."
        ) from error


def _parse_summary(response_text: str) -> dict[str, Any]:
    try:
        summary = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise GeminiResponseError(
            "Gemini returned invalid JSON."
        ) from error

    if not isinstance(summary, dict):
        raise GeminiResponseError("Gemini returned an invalid summary.")

    required_fields = ("title", "overview", "key_points", "keywords")
    if any(field not in summary for field in required_fields):
        raise GeminiResponseError("Gemini returned an incomplete summary.")

    if (
        not isinstance(summary["title"], str)
        or not isinstance(summary["overview"], str)
        or not isinstance(summary["key_points"], list)
        or not isinstance(summary["keywords"], list)
        or not all(isinstance(item, str) for item in summary["key_points"])
        or not all(isinstance(item, str) for item in summary["keywords"])
    ):
        raise GeminiResponseError("Gemini returned an invalid summary format.")

    return {
        "title": summary["title"].strip(),
        "overview": summary["overview"].strip(),
        "key_points": summary["key_points"],
        "keywords": summary["keywords"],
    }


def generate_gemini_summary(transcript: str) -> dict[str, Any]:
    if not transcript.strip():
        raise GeminiResponseError("Transcript is empty.")

    bounded_transcript = transcript[:MAX_TRANSCRIPT_CHARACTERS]
    prompt = f"""
You are EduSummarize AI. Summarize the lecture transcript below.

The transcript may be in Hindi, English, or Hinglish. Write the title,
overview, key points, and keywords in the transcript's dominant language style.
Preserve important technical terms and do not invent facts. If the transcript
was truncated, summarize only the content provided.

Return valid JSON only with exactly this structure:
{{
  "title": "short lecture title",
  "overview": "concise lecture overview",
  "key_points": ["important point 1", "important point 2"],
  "keywords": ["keyword 1", "keyword 2"]
}}

Transcript:
{bounded_transcript}
"""

    client = _get_client()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
    except errors.ClientError as error:
        error_code = getattr(error, "code", None)
        if error_code is None:
            error_code = getattr(error, "status_code", None)
        if error_code in {401, 403}:
            raise GeminiAuthenticationError(
                "Gemini rejected the configured API key."
            ) from error
        raise GeminiSummaryError("Gemini request failed.") from error
    except (errors.ServerError, errors.APIError) as error:
        raise GeminiSummaryError("Gemini request failed.") from error
    except RuntimeError as error:
        raise GeminiSummaryError("Gemini request failed.") from error
    finally:
        client.close()

    return _parse_summary(getattr(response, "text", None))


def generate_gemini_summary_from_file(transcript_file: Path) -> dict[str, Any]:
    try:
        with transcript_file.open("r", encoding="utf-8") as file:
            transcript_data = json.load(file)
        transcript = transcript_data["transcription"]["text"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise GeminiResponseError(
            "Saved transcript is invalid."
        ) from error

    if not isinstance(transcript, str):
        raise GeminiResponseError("Saved transcript text is invalid.")

    return generate_gemini_summary(transcript)


def _normalise_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _parse_quiz(response_text: str, transcript: str) -> list[dict[str, Any]]:
    try:
        quiz = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise GeminiQuizResponseError("Gemini returned invalid quiz JSON.") from error

    if not isinstance(quiz, list) or len(quiz) != QUIZ_QUESTION_COUNT:
        raise GeminiQuizResponseError(
            f"Gemini must return exactly {QUIZ_QUESTION_COUNT} questions."
        )

    normalised_transcript = _normalise_text(transcript)
    validated_questions = []
    seen_questions = set()
    ambiguous_markers = (
        "all of the above",
        "none of the above",
        "cannot be determined",
        "both a and b",
    )
    for question in quiz:
        if not isinstance(question, dict):
            raise GeminiQuizResponseError("Gemini returned an invalid question.")

        required_fields = (
            "question",
            "options",
            "correct_answer",
            "explanation",
            "difficulty",
            "topic",
            "supporting_text",
        )
        if any(field not in question for field in required_fields):
            raise GeminiQuizResponseError("A quiz question is incomplete.")

        question_text = question["question"]
        options = question["options"]
        correct_answer = question["correct_answer"]
        explanation = question["explanation"]
        difficulty = question["difficulty"]
        topic = question["topic"]
        supporting_text = question["supporting_text"]

        if (
            not isinstance(question_text, str)
            or not isinstance(options, list)
            or len(options) != 4
            or not all(isinstance(option, str) and option.strip() for option in options)
            or len({_normalise_text(option) for option in options}) != 4
            or not isinstance(correct_answer, str)
            or correct_answer not in options
            or not isinstance(explanation, str)
            or not explanation.strip()
            or difficulty not in {"easy", "medium", "hard"}
            or not isinstance(topic, str)
            or not topic.strip()
            or not isinstance(supporting_text, str)
            or not supporting_text.strip()
        ):
            raise GeminiQuizResponseError("Gemini returned an invalid question format.")

        question_lower = _normalise_text(question_text)
        if (
            not question_text.strip().endswith("?")
            or any(marker in question_lower for marker in ambiguous_markers)
            or _normalise_text(supporting_text) not in normalised_transcript
            or question_lower in seen_questions
        ):
            raise GeminiQuizResponseError(
                "Gemini returned an unsupported or ambiguous question."
            )

        seen_questions.add(question_lower)
        validated_questions.append(
            {
                "question": question_text.strip(),
                "options": [option.strip() for option in options],
                "correct_answer": correct_answer.strip(),
                "explanation": explanation.strip(),
                "difficulty": difficulty,
                "topic": topic.strip(),
                "supporting_text": supporting_text.strip(),
            }
        )

    return validated_questions


def generate_gemini_quiz(
    transcript: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not transcript.strip():
        raise GeminiQuizResponseError("Transcript is empty.")

    bounded_transcript = transcript[:MAX_TRANSCRIPT_CHARACTERS]
    summary_context = json.dumps(summary, ensure_ascii=False)
    prompt = f"""
You are EduSummarize AI. Create exactly five evidence-based multiple-choice
questions from the lecture transcript and its summary.

The transcript may be Hindi, English, or Hinglish. Keep each question and its
explanation in the transcript's dominant language style. Every question must
have one clearly correct answer supported directly by the transcript.
Do not test information absent from the transcript. Do not use "all of the
above", "none of the above", subjective wording, trick questions, or ambiguous
questions. The supporting_text must be an exact contiguous quote from the
transcript (spacing may be normalized).

Return valid JSON only as an array of exactly five objects:
[
  {{
    "question": "Question ending with ?",
    "options": ["option 1", "option 2", "option 3", "option 4"],
    "correct_answer": "The exact text of one option",
    "explanation": "Why the answer is correct based on the transcript",
    "difficulty": "easy",
    "topic": "Topic",
    "supporting_text": "Exact supporting transcript quote"
  }}
]

Allowed difficulty values: easy, medium, hard.

Gemini summary:
{summary_context}

Transcript:
{bounded_transcript}
"""

    client = _get_client()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
    except errors.ClientError as error:
        error_code = getattr(error, "code", None)
        if error_code is None:
            error_code = getattr(error, "status_code", None)
        if error_code in {401, 403}:
            raise GeminiAuthenticationError(
                "Gemini rejected the configured API key."
            ) from error
        raise GeminiSummaryError("Gemini request failed.") from error
    except (errors.ServerError, errors.APIError, RuntimeError) as error:
        raise GeminiSummaryError("Gemini request failed.") from error
    finally:
        client.close()

    return _parse_quiz(getattr(response, "text", None), bounded_transcript)


def generate_gemini_quiz_from_files(
    transcript_file: Path,
    summary_file: Path,
) -> list[dict[str, Any]]:
    try:
        with transcript_file.open("r", encoding="utf-8") as file:
            transcript_data = json.load(file)
        with summary_file.open("r", encoding="utf-8") as file:
            summary_data = json.load(file)
        transcript = transcript_data["transcription"]["text"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise GeminiQuizResponseError(
            "Saved transcript or Gemini summary is invalid."
        ) from error

    if not isinstance(transcript, str) or not isinstance(summary_data, dict):
        raise GeminiQuizResponseError(
            "Saved transcript or Gemini summary is invalid."
        )

    return generate_gemini_quiz(transcript, summary_data)


def _parse_flashcards(response_text: str) -> list[dict[str, Any]]:
    try:
        flashcards = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise GeminiFlashcardResponseError(
            "Gemini returned invalid flashcard JSON."
        ) from error

    if not isinstance(flashcards, list) or len(flashcards) != FLASHCARD_COUNT:
        raise GeminiFlashcardResponseError(
            f"Gemini must return exactly {FLASHCARD_COUNT} flashcards."
        )

    validated_flashcards = []
    for flashcard in flashcards:
        if not isinstance(flashcard, dict):
            raise GeminiFlashcardResponseError(
                "Gemini returned an invalid flashcard."
            )

        required_fields = ("question", "answer", "topic", "difficulty")
        if any(field not in flashcard for field in required_fields):
            raise GeminiFlashcardResponseError(
                "A flashcard is incomplete."
            )

        question = flashcard["question"]
        answer = flashcard["answer"]
        topic = flashcard["topic"]
        difficulty = flashcard["difficulty"]
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or not isinstance(topic, str)
            or not topic.strip()
            or difficulty not in {"easy", "medium", "hard"}
        ):
            raise GeminiFlashcardResponseError(
                "Gemini returned an invalid flashcard format."
            )

        validated_flashcards.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
                "topic": topic.strip(),
                "difficulty": difficulty,
            }
        )

    return validated_flashcards


def generate_gemini_flashcards(
    transcript: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not transcript.strip():
        raise GeminiFlashcardResponseError("Transcript is empty.")

    bounded_transcript = transcript[:MAX_TRANSCRIPT_CHARACTERS]
    summary_context = json.dumps(summary, ensure_ascii=False)
    prompt = f"""
You are EduSummarize AI. Create exactly ten topic-wise flashcards from the
lecture transcript and its summary.

The transcript may be Hindi, English, or Hinglish. Write each flashcard in the
transcript's dominant language style. Use only facts supported by the
transcript; do not invent information. Cover the important topics from the
lecture and assign each flashcard a concise topic and one difficulty value.

Return valid JSON only as an array of exactly ten objects:
[
  {{
    "question": "A concise question",
    "answer": "The answer supported by the lecture",
    "topic": "Topic",
    "difficulty": "easy"
  }}
]

Allowed difficulty values: easy, medium, hard.

Gemini summary:
{summary_context}

Transcript:
{bounded_transcript}
"""

    client = _get_client()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
    except errors.ClientError as error:
        error_code = getattr(error, "code", None)
        if error_code is None:
            error_code = getattr(error, "status_code", None)
        if error_code in {401, 403}:
            raise GeminiAuthenticationError(
                "Gemini rejected the configured API key."
            ) from error
        raise GeminiSummaryError("Gemini request failed.") from error
    except (errors.ServerError, errors.APIError, RuntimeError) as error:
        raise GeminiSummaryError("Gemini request failed.") from error
    finally:
        client.close()

    return _parse_flashcards(getattr(response, "text", None))


def generate_gemini_flashcards_from_files(
    transcript_file: Path,
    summary_file: Path,
) -> list[dict[str, Any]]:
    try:
        with transcript_file.open("r", encoding="utf-8") as file:
            transcript_data = json.load(file)
        with summary_file.open("r", encoding="utf-8") as file:
            summary_data = json.load(file)
        transcript = transcript_data["transcription"]["text"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise GeminiFlashcardResponseError(
            "Saved transcript or Gemini summary is invalid."
        ) from error

    if not isinstance(transcript, str) or not isinstance(summary_data, dict):
        raise GeminiFlashcardResponseError(
            "Saved transcript or Gemini summary is invalid."
        )

    return generate_gemini_flashcards(transcript, summary_data)
