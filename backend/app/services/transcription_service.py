from pathlib import Path
from faster_whisper import WhisperModel

model = None


def get_model():
    global model

    if model is None:
        model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

    return model


def transcribe_audio(audio_path: Path):
    whisper_model = get_model()

    segments, info = whisper_model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True
    )

    transcript_segments = []

    for segment in segments:
        transcript_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    return {
        "language": info.language,
        "duration": info.duration,
        "text": " ".join(
            item["text"] for item in transcript_segments
        ),
        "segments": transcript_segments
    }