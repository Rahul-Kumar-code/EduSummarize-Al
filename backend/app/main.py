from pathlib import Path
from uuid import uuid4
from io import BytesIO
import shutil
import json
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    LayoutError,
    Paragraph,
    PageBreak,
    SimpleDocTemplate,
    Spacer,
)
from .services.audio_service import convert_to_wav
from .services.transcription_service import transcribe_audio
from .services.summary_service import generate_summary
from .services.youtube_service import download_youtube_audio
from .services.gemini_summary_service import (
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiResponseError,
    GeminiFlashcardResponseError,
    GeminiQuizResponseError,
    GeminiSummaryError,
    generate_gemini_summary_from_file,
    generate_gemini_flashcards_from_files,
    generate_gemini_quiz_from_files,
)

app = FastAPI(title="EduSummarize AI")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

TRANSCRIPT_DIR = Path("transcripts")
TRANSCRIPT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mp4", ".m4a", ".webm"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


class YouTubeUploadRequest(BaseModel):
    url: str


class QuizSubmissionRequest(BaseModel):
    answers: list[str]


def _pdf_text(value: object) -> str:
    return str(value).encode("ascii", "replace").decode("ascii")


def _pdf_paragraph(
    value: object,
    style: ParagraphStyle,
) -> Paragraph:
    return Paragraph(_pdf_text(value).replace("&", "&amp;").replace("<", "&lt;"), style)


def _build_lecture_pdf(
    lecture_id: str,
    transcript_data: dict,
    summary_data: dict,
    quiz_data: dict,
    flashcards_data: dict,
    analytics_data: dict,
) -> bytes:
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ExportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f3a5f"),
        spaceAfter=18,
    )
    heading_style = ParagraphStyle(
        "ExportHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1f3a5f"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ExportBody",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "ExportSmall",
        parent=body_style,
        fontSize=9,
        leading=11,
    )

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=f"Lecture Export - {lecture_id}",
    )
    story = [
        _pdf_paragraph("EduSummarize AI - Lecture Export", title_style),
        _pdf_paragraph(f"Lecture ID: {lecture_id}", body_style),
        Spacer(1, 8),
        _pdf_paragraph("Summary", heading_style),
        _pdf_paragraph(summary_data.get("title", "Untitled lecture"), body_style),
        _pdf_paragraph(summary_data.get("overview", ""), body_style),
    ]

    key_points = summary_data.get("key_points", [])
    if isinstance(key_points, list) and key_points:
        story.append(_pdf_paragraph("Key points", heading_style))
        story.append(
            ListFlowable(
                [
                    ListItem(_pdf_paragraph(point, body_style))
                    for point in key_points
                ],
                bulletType="bullet",
            )
        )

    transcript = transcript_data.get("transcription", {})
    transcript_text = transcript.get("text", "") if isinstance(transcript, dict) else ""
    story.extend(
        [
            _pdf_paragraph("Transcript", heading_style),
            _pdf_paragraph(transcript_text, small_style),
            _pdf_paragraph("Quiz", heading_style),
        ]
    )
    questions = quiz_data.get("questions", [])
    if isinstance(questions, list):
        for number, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                continue
            story.extend(
                [
                    _pdf_paragraph(
                        f"{number}. {question.get('question', '')}",
                        body_style,
                    ),
                    _pdf_paragraph(
                        f"Answer: {question.get('correct_answer', '')}",
                        small_style,
                    ),
                    _pdf_paragraph(
                        f"Explanation: {question.get('explanation', '')}",
                        small_style,
                    ),
                    Spacer(1, 4),
                ]
            )

    story.append(_pdf_paragraph("Flashcards", heading_style))
    flashcards = flashcards_data.get("flashcards", [])
    if isinstance(flashcards, list):
        for number, flashcard in enumerate(flashcards, start=1):
            if not isinstance(flashcard, dict):
                continue
            story.extend(
                [
                    _pdf_paragraph(
                        f"{number}. {flashcard.get('question', '')}",
                        body_style,
                    ),
                    _pdf_paragraph(
                        (
                            f"Answer: {flashcard.get('answer', '')} "
                            f"({flashcard.get('topic', '')}, "
                            f"{flashcard.get('difficulty', '')})"
                        ),
                        small_style,
                    ),
                ]
            )

    story.extend(
        [
            PageBreak(),
            _pdf_paragraph("Learning Analytics", heading_style),
            _pdf_paragraph(
                (
                    f"Score: {analytics_data.get('correct_answers', 0)} / "
                    f"{analytics_data.get('total_questions', 0)} "
                    f"({analytics_data.get('percentage', 0)}%)"
                ),
                body_style,
            ),
            _pdf_paragraph(
                f"Strong topics: {', '.join(analytics_data.get('strong_topics', [])) or 'None'}",
                body_style,
            ),
            _pdf_paragraph(
                (
                    "Topics needing revision: "
                    f"{', '.join(analytics_data.get('topics_needing_revision', [])) or 'None'}"
                ),
                body_style,
            ),
        ]
    )
    document.build(story)
    return buffer.getvalue()


def is_youtube_url(url: str) -> bool:
    try:
        parsed_url = urlparse(url.strip())
        hostname = (parsed_url.hostname or "").lower().rstrip(".")
    except ValueError:
        return False

    if parsed_url.scheme not in {"http", "https"}:
        return False
    if parsed_url.username or parsed_url.password:
        return False
    if hostname not in YOUTUBE_HOSTS:
        return False

    if hostname == "youtu.be":
        return bool(parsed_url.path.strip("/"))

    return (
        parsed_url.path == "/watch"
        and bool(parse_qs(parsed_url.query).get("v", [""])[0])
    ) or any(
        parsed_url.path.startswith(prefix)
        and len(parsed_url.path.removeprefix(prefix).strip("/")) > 0
        for prefix in ("/shorts/", "/embed/", "/live/")
    )


@app.get("/")
def home():
    return {"message": "EduSummarize AI backend is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/lectures/upload")
async def upload_lecture(file: UploadFile = File(...)):
    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only MP3, WAV, MP4, M4A, and WEBM files are allowed."
        )

    lecture_id = str(uuid4())
    saved_path = UPLOAD_DIR / f"{lecture_id}{extension}"

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Lecture uploaded successfully",
        "lecture_id": lecture_id,
        "filename": original_name,
        "saved_file": str(saved_path)
    }


@app.post("/lectures/youtube")
def upload_youtube_lecture(request: YouTubeUploadRequest):
    youtube_url = request.url.strip()
    if not is_youtube_url(youtube_url):
        raise HTTPException(
            status_code=400,
            detail="A valid public YouTube URL is required."
        )

    lecture_id = str(uuid4())

    try:
        downloaded_file = download_youtube_audio(
            youtube_url,
            lecture_id,
            UPLOAD_DIR
        )
    except RuntimeError:
        raise HTTPException(
            status_code=502,
            detail="Unable to download audio from the YouTube URL."
        )

    return {
        "message": "YouTube lecture downloaded successfully",
        "lecture_id": lecture_id,
        "filename": downloaded_file.name,
        "processing_status": "downloaded",
        "status": "downloaded"
    }


@app.post("/lectures/{lecture_id}/transcribe")
def transcribe_lecture(lecture_id: str):
    matching_files = list(UPLOAD_DIR.glob(f"{lecture_id}.*"))

    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail="Lecture file not found"
        )

    input_file = matching_files[0]
    wav_file = UPLOAD_DIR / f"{lecture_id}.wav"

    try:
        convert_to_wav(input_file, wav_file)
        transcription = transcribe_audio(wav_file)

        transcript_file = TRANSCRIPT_DIR / f"{lecture_id}.json"

        transcript_data = {
            "lecture_id": lecture_id,
            "status": "transcribed",
            "transcription": transcription
        }

        with transcript_file.open("w", encoding="utf-8") as file:
            json.dump(
                transcript_data,
                file,
                ensure_ascii=False,
                indent=2
            )

        return transcript_data

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(error)}"
        )

@app.get("/lectures/{lecture_id}/transcript")
def get_transcript(lecture_id: str):
    transcript_file = TRANSCRIPT_DIR / f"{lecture_id}.json"

    if not transcript_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Transcribe the lecture first."
        )

    with transcript_file.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.post("/lectures/{lecture_id}/gemini-summary")
def gemini_summarize_lecture(lecture_id: str):
    transcript_file = TRANSCRIPT_DIR / f"{lecture_id}.json"

    if not transcript_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Transcribe the lecture first."
        )

    try:
        summary = generate_gemini_summary_from_file(transcript_file)
    except GeminiConfigurationError:
        raise HTTPException(
            status_code=503,
            detail="Gemini summarization is not configured."
        )
    except GeminiAuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="The configured Gemini API key was rejected."
        )
    except GeminiResponseError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except GeminiSummaryError:
        raise HTTPException(
            status_code=502,
            detail="Gemini summarization failed."
        )

    summary_data = {
        "lecture_id": lecture_id,
        **summary,
        "status": "summarized",
        "method": "Gemini",
    }
    summary_file = TRANSCRIPT_DIR / f"{lecture_id}_gemini_summary.json"

    try:
        with summary_file.open("w", encoding="utf-8") as file:
            json.dump(summary_data, file, ensure_ascii=False, indent=2)
    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Unable to save the Gemini summary."
        )

    return summary_data


@app.post("/lectures/{lecture_id}/quiz")
def generate_lecture_quiz(lecture_id: str):
    transcript_file = TRANSCRIPT_DIR / f"{lecture_id}.json"
    summary_file = TRANSCRIPT_DIR / f"{lecture_id}_gemini_summary.json"

    if not transcript_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Transcribe the lecture first."
        )
    if not summary_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Gemini summary not found. Generate the Gemini summary first."
        )

    try:
        questions = generate_gemini_quiz_from_files(
            transcript_file,
            summary_file,
        )
    except GeminiConfigurationError:
        raise HTTPException(
            status_code=503,
            detail="Gemini quiz generation is not configured."
        )
    except GeminiAuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="The configured Gemini API key was rejected."
        )
    except GeminiQuizResponseError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except GeminiSummaryError:
        raise HTTPException(
            status_code=502,
            detail="Gemini quiz generation failed."
        )

    quiz_data = {
        "lecture_id": lecture_id,
        "questions": questions,
        "status": "generated",
        "method": "Gemini",
    }
    quiz_file = TRANSCRIPT_DIR / f"{lecture_id}_quiz.json"

    try:
        with quiz_file.open("w", encoding="utf-8") as file:
            json.dump(quiz_data, file, ensure_ascii=False, indent=2)
    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Unable to save the generated quiz."
        )

    return quiz_data


@app.post("/lectures/{lecture_id}/flashcards")
def generate_lecture_flashcards(lecture_id: str):
    transcript_file = TRANSCRIPT_DIR / f"{lecture_id}.json"
    summary_file = TRANSCRIPT_DIR / f"{lecture_id}_gemini_summary.json"

    if not transcript_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Transcribe the lecture first."
        )
    if not summary_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Gemini summary not found. Generate the Gemini summary first."
        )

    try:
        flashcards = generate_gemini_flashcards_from_files(
            transcript_file,
            summary_file,
        )
    except GeminiConfigurationError:
        raise HTTPException(
            status_code=503,
            detail="Gemini flashcard generation is not configured."
        )
    except GeminiAuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="The configured Gemini API key was rejected."
        )
    except GeminiFlashcardResponseError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except GeminiSummaryError:
        raise HTTPException(
            status_code=502,
            detail="Gemini flashcard generation failed."
        )

    flashcards_data = {
        "lecture_id": lecture_id,
        "flashcards": flashcards,
        "status": "generated",
        "method": "Gemini",
    }
    flashcards_file = TRANSCRIPT_DIR / f"{lecture_id}_flashcards.json"

    try:
        with flashcards_file.open("w", encoding="utf-8") as file:
            json.dump(flashcards_data, file, ensure_ascii=False, indent=2)
    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Unable to save the generated flashcards."
        )

    return flashcards_data


@app.post("/lectures/{lecture_id}/quiz/submit")
def submit_lecture_quiz(
    lecture_id: str,
    submission: QuizSubmissionRequest,
):
    quiz_file = TRANSCRIPT_DIR / f"{lecture_id}_quiz.json"

    if not quiz_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Quiz not found. Generate the quiz first."
        )

    try:
        with quiz_file.open("r", encoding="utf-8") as file:
            quiz_data = json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=422,
            detail="Saved quiz is invalid."
        )

    questions = quiz_data.get("questions") if isinstance(quiz_data, dict) else None
    if (
        not isinstance(questions, list)
        or not questions
        or not all(isinstance(question, dict) for question in questions)
    ):
        raise HTTPException(
            status_code=422,
            detail="Saved quiz is invalid."
        )

    if len(submission.answers) != len(questions):
        raise HTTPException(
            status_code=422,
            detail=f"Exactly {len(questions)} answers are required."
        )

    correct_count = 0
    correct_questions = []
    incorrect_questions = []

    for index, (question, selected_answer) in enumerate(
        zip(questions, submission.answers)
    ):
        options = question.get("options")
        correct_answer = question.get("correct_answer")
        question_text = question.get("question")
        explanation = question.get("explanation")
        if (
            not isinstance(options, list)
            or not isinstance(correct_answer, str)
            or not all(isinstance(option, str) for option in options)
            or not isinstance(question_text, str)
            or not question_text.strip()
            or not isinstance(explanation, str)
            or not explanation.strip()
            or correct_answer not in options
            or selected_answer not in options
        ):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid answer for question {index + 1}."
            )

        result = {
            "question_number": index + 1,
            "question": question_text,
            "selected_answer": selected_answer,
            "correct_answer": correct_answer,
            "explanation": explanation,
        }
        if selected_answer == correct_answer:
            correct_count += 1
            correct_questions.append(result)
        else:
            incorrect_questions.append(result)

    total_questions = len(questions)
    percentage = round((correct_count / total_questions) * 100, 2)
    attempt_data = {
        "lecture_id": lecture_id,
        "score": correct_count,
        "total_questions": total_questions,
        "percentage": percentage,
        "correct_questions": correct_questions,
        "incorrect_questions": incorrect_questions,
        "status": "completed",
    }
    attempt_file = TRANSCRIPT_DIR / f"{lecture_id}_attempt.json"

    try:
        with attempt_file.open("w", encoding="utf-8") as file:
            json.dump(attempt_data, file, ensure_ascii=False, indent=2)
    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Unable to save the quiz attempt."
        )

    return attempt_data


@app.get("/lectures/{lecture_id}/analytics")
def get_lecture_analytics(lecture_id: str):
    quiz_file = TRANSCRIPT_DIR / f"{lecture_id}_quiz.json"
    attempt_file = TRANSCRIPT_DIR / f"{lecture_id}_attempt.json"

    if not quiz_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Quiz not found. Generate the quiz first."
        )
    if not attempt_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Quiz attempt not found. Submit the quiz first."
        )

    try:
        with quiz_file.open("r", encoding="utf-8") as file:
            quiz_data = json.load(file)
        with attempt_file.open("r", encoding="utf-8") as file:
            attempt_data = json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=422,
            detail="Saved quiz or quiz attempt is invalid."
        )

    questions = quiz_data.get("questions") if isinstance(quiz_data, dict) else None
    correct_questions = (
        attempt_data.get("correct_questions")
        if isinstance(attempt_data, dict)
        else None
    )
    incorrect_questions = (
        attempt_data.get("incorrect_questions")
        if isinstance(attempt_data, dict)
        else None
    )
    if (
        not isinstance(questions, list)
        or not questions
        or not all(isinstance(question, dict) for question in questions)
        or not isinstance(correct_questions, list)
        or not isinstance(incorrect_questions, list)
    ):
        raise HTTPException(
            status_code=422,
            detail="Saved quiz or quiz attempt is invalid."
        )

    question_results = {}
    for result, is_correct in (
        [(result, True) for result in correct_questions]
        + [(result, False) for result in incorrect_questions]
    ):
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("question_number"), int)
            or result["question_number"] < 1
            or result["question_number"] > len(questions)
            or result["question_number"] in question_results
        ):
            raise HTTPException(
                status_code=422,
                detail="Saved quiz or quiz attempt is invalid."
            )
        question_results[result["question_number"]] = is_correct

    topic_totals = {}
    for question_number, question in enumerate(questions, start=1):
        topic = question.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            raise HTTPException(
                status_code=422,
                detail="Saved quiz or quiz attempt is invalid."
            )
        if question_number not in question_results:
            raise HTTPException(
                status_code=422,
                detail="Saved quiz or quiz attempt is incomplete."
            )

        topic = topic.strip()
        topic_data = topic_totals.setdefault(
            topic,
            {"topic": topic, "total_questions": 0, "correct_answers": 0},
        )
        topic_data["total_questions"] += 1
        if question_results[question_number]:
            topic_data["correct_answers"] += 1

    topic_scores = []
    for topic_data in topic_totals.values():
        topic_percentage = round(
            (topic_data["correct_answers"] / topic_data["total_questions"]) * 100,
            2,
        )
        topic_scores.append(
            {
                **topic_data,
                "percentage": topic_percentage,
            }
        )

    strong_topics = [
        topic_data["topic"]
        for topic_data in topic_scores
        if topic_data["percentage"] >= 70
    ]
    topics_needing_revision = [
        topic_data["topic"]
        for topic_data in topic_scores
        if topic_data["percentage"] < 70
    ]
    total_questions = len(questions)
    correct_answers = sum(question_results.values())

    return {
        "lecture_id": lecture_id,
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "percentage": round((correct_answers / total_questions) * 100, 2),
        "topic_scores": topic_scores,
        "strong_topics": strong_topics,
        "topics_needing_revision": topics_needing_revision,
    }


@app.get("/lectures/{lecture_id}/export-pdf")
def export_lecture_pdf(lecture_id: str):
    artifact_files = {
        "transcript": TRANSCRIPT_DIR / f"{lecture_id}.json",
        "summary": TRANSCRIPT_DIR / f"{lecture_id}_gemini_summary.json",
        "quiz": TRANSCRIPT_DIR / f"{lecture_id}_quiz.json",
        "flashcards": TRANSCRIPT_DIR / f"{lecture_id}_flashcards.json",
        "attempt": TRANSCRIPT_DIR / f"{lecture_id}_attempt.json",
    }
    for artifact_name, artifact_file in artifact_files.items():
        if not artifact_file.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Saved {artifact_name} not found. "
                    "Complete the lecture processing steps first."
                ),
            )

    try:
        artifact_data = {}
        for artifact_name, artifact_file in artifact_files.items():
            with artifact_file.open("r", encoding="utf-8") as file:
                artifact_data[artifact_name] = json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=422,
            detail="One or more saved lecture artifacts are invalid."
        )

    if not all(isinstance(data, dict) for data in artifact_data.values()):
        raise HTTPException(
            status_code=422,
            detail="One or more saved lecture artifacts are invalid."
        )

    try:
        analytics_data = get_lecture_analytics(lecture_id)
        pdf_content = _build_lecture_pdf(
            lecture_id,
            artifact_data["transcript"],
            artifact_data["summary"],
            artifact_data["quiz"],
            artifact_data["flashcards"],
            analytics_data,
        )
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, LayoutError):
        raise HTTPException(
            status_code=500,
            detail="Unable to generate the lecture PDF."
        )

    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{lecture_id}_lecture_export.pdf"'
            )
        },
    )