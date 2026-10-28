from pathlib import Path
import os
import subprocess
import sys


def download_youtube_audio(url: str, lecture_id: str, upload_dir: Path):
    output_template = upload_dir / f"{lecture_id}.%(ext)s"
    expected_file = upload_dir / f"{lecture_id}.mp3"

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
    "--no-progress",
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

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )
    except subprocess.TimeoutExpired as error:
        _remove_downloaded_files(upload_dir, lecture_id)
        raise RuntimeError("YouTube audio download timed out.") from error
    except OSError as error:
        _remove_downloaded_files(upload_dir, lecture_id)
        raise RuntimeError("Unable to start the YouTube audio downloader.") from error

    if result.returncode != 0:
        _remove_downloaded_files(upload_dir, lecture_id)
        raise RuntimeError("yt-dlp could not download the YouTube audio.")

    if not expected_file.is_file():
        _remove_downloaded_files(upload_dir, lecture_id)
        raise RuntimeError("YouTube audio file was not created.")

    return expected_file


def _remove_downloaded_files(upload_dir: Path, lecture_id: str):
    for file in upload_dir.glob(f"{lecture_id}.*"):
        if file.is_file():
            file.unlink()