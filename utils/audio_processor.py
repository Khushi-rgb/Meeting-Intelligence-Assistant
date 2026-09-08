import yt_dlp
from pydub import AudioSegment
import os


# =========================
# FFmpeg Configuration
# =========================

if os.name == "nt":
    # Windows - use your local FFmpeg installation
    FFMPEG_DIR = (
        r"C:\Users\pinki\Downloads\ffmpeg-8.1.1-essentials_build"
        r"\ffmpeg-8.1.1-essentials_build\bin"
    )

    FFMPEG_EXE = os.path.join(FFMPEG_DIR, "ffmpeg.exe")

    AudioSegment.converter = FFMPEG_EXE

else:
    # Railway / Linux
    # FFmpeg is installed by Railway and available in system PATH
    FFMPEG_DIR = None


# =========================
# Download Directory
# =========================

DOWNLOAD_DIR = "downloades"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# =========================
# Download YouTube Audio
# =========================

def download_youtube_audio(url: str):

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",

        "outtmpl": output_path,

        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        },

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        "quiet": True,
    }

    # Use local Windows FFmpeg path only on Windows
    if FFMPEG_DIR:
        ydl_opts["ffmpeg_location"] = FFMPEG_DIR

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(
            url,
            download=True
        )

        print("INFO KEYS:", info.keys())

        print(
            "Duration:",
            info.get("duration")
        )

        base = os.path.splitext(
            ydl.prepare_filename(info)
        )[0]

        filename = base + ".wav"

        video_title = info.get("title")

        thumbnail = info.get("thumbnail")

        print(
            "prepare_filename:",
            ydl.prepare_filename(info)
        )

        print(
            "wav filename:",
            filename
        )

        print(
            "exists:",
            os.path.exists(filename)
        )

    return filename, video_title, thumbnail


# =========================
# Convert Local File to WAV
# =========================

def convert_to_wav(input_path: str) -> str:

    """Convert any audio/video file to WAV format using pydub."""

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    audio = AudioSegment.from_file(
        input_path
    )

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        output_path,
        format="wav"
    )

    return output_path


# =========================
# Chunk Audio
# =========================

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:

    audio = AudioSegment.from_wav(
        wav_path
    )

    chunk_ms = (
        chunk_minutes
        * 60
        * 1000
    )

    chunks = []

    for i, start in enumerate(
        range(
            0,
            len(audio),
            chunk_ms
        )
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = (
            f"{wav_path}_chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(
            chunk_path
        )

    return chunks


# =========================
# Process Input
# =========================

def process_input(source: str):

    # YouTube / Online URL
    if (
        source.startswith("http://")
        or source.startswith("https://")
    ):

        print(
            "Detected YouTube URL. "
            "Downloading audio..."
        )

        (
            wav_path,
            video_title,
            thumbnail
        ) = download_youtube_audio(
            source
        )

    # Local uploaded file
    else:

        print(
            "Detected local file. "
            "Converting to WAV..."
        )

        wav_path = convert_to_wav(
            source
        )

        video_title = "Local File"

        thumbnail = None


    # =========================
    # Get Duration
    # =========================

    audio = AudioSegment.from_wav(
        wav_path
    )

    duration_seconds = (
        len(audio) // 1000
    )


    # =========================
    # Chunk Audio
    # =========================

    print(
        "Chunking audio..."
    )

    chunks = chunk_audio(
        wav_path
    )


    return (
        chunks,
        duration_seconds,
        video_title,
        thumbnail
    )