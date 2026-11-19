"""Utilidades compartidas por los comandos. Solo stdlib."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def utf8_console():
    """En Windows la consola es cp1252 y rompe con tildes y eñes."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def hhmmss(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_time(sec, sep=","):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def ffmpeg_hint(system=None):
    system = system or platform.system()
    if system == "Windows":
        return "winget install Gyan.FFmpeg  (y abrí una terminal nueva)"
    if system == "Darwin":
        return "brew install ffmpeg"
    return "sudo dnf install ffmpeg  /  sudo apt install ffmpeg"


def token_hint(system=None):
    system = system or platform.system()
    if system == "Windows":
        how = "setx HF_TOKEN hf_...   (y abrí una terminal nueva)"
    elif system == "Darwin":
        how = "echo 'export HF_TOKEN=hf_...' >> ~/.zshrc"
    else:
        how = "echo 'export HF_TOKEN=hf_...' >> ~/.bashrc"
    return f"{how}   — o bien: hf auth login"


def require(tool):
    """Ruta de un ejecutable del PATH, o termina diciendo cómo instalarlo."""
    exe = shutil.which(tool)
    if exe is None:
        hint = ffmpeg_hint() if tool in ("ffmpeg", "ffprobe") else "revisá la instalación"
        raise SystemExit(f"Falta {tool} en el PATH. {hint}")
    return exe


def duration_of(path: Path) -> float:
    out = subprocess.run(
        [require("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return float(out.stdout.strip())


def to_wav16k(src: Path, dst: Path):
    """pyannote y whisper quieren 16 kHz mono; ffmpeg lee cualquier contenedor."""
    subprocess.run(
        [require("ffmpeg"), "-loglevel", "error", "-y", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )
