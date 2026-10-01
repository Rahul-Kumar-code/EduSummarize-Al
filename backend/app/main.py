from pathlib import Path
from uuid import uuid4
import shutil
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import HTTPException
from .services.audio_service import convert_to_wav
from .services.transcription_service import transcribe_audio
from .services.summary_service import generate_summary

app = FastAPI(title="EduSummarize AI")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

TRANSCRIPT_DIR = Path("transcripts")
TRANSCRIPT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mp4", ".m4a", ".webm"}


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

@app.post("/lectures/{lecture_id}/summarize")
def summarize_lecture(lecture_id: str):
    transcript_file = TRANSCRIPT_DIR / f"{lecture_id}.json"

    if not transcript_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Transcribe the lecture first."
        )

    with transcript_file.open("r", encoding="utf-8") as file:
        transcript_data = json.load(file)

    transcript_text = transcript_data["transcription"]["text"]
    summary = generate_summary(transcript_text)

    summary_data = {
        "lecture_id": lecture_id,
        "summary": summary,
        "method": "TextRank"
    }

    summary_file = TRANSCRIPT_DIR / f"{lecture_id}_summary.json"

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(
            summary_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    return summary_data