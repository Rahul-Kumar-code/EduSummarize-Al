from pathlib import Path
import os
import subprocess
import sys


def download_youtube_audio(url: str, lecture_id: str, upload_dir: Path):
    output_template = upload_dir / f"{lecture_id}.%(ext)s"

    command = [
    sys.executable,
    "-m",
    "yt_dlp",
    "--no-playlist",
    "--js-runtimes",
    "deno",
    "-x",
    "--audio-format",
    "mp3",
    "--audio-quality",
    "5",
    "-o",
    str(output_template),
    url,
]

    cookies_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
    if cookies_browser:
        command[command.index("-x"):command.index("-x")] = [
            "--cookies-from-browser",
            cookies_browser,
        ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    downloaded_files = [
        file for file in upload_dir.glob(f"{lecture_id}.*")
        if file.suffix.lower() not in {".wav", ".json"}
    ]

    if not downloaded_files:
        raise RuntimeError("YouTube audio file was not created.")

    return downloaded_files[0]