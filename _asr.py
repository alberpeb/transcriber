"""Paso ASR — corre en el entorno de whisper-ctranslate2 (CUDA 12).

Además de transcribir, vuelve a decodificar por separado cualquier tramo largo
que haya quedado sin texto. Whisper a veces se saltea audio bajo o con ruido, y
así no se pierde habla real (ya pasó: un tramo de 6 minutos salió casi vacío).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

wav, model_name, language, device, out, prompt_path, duration = sys.argv[1:8]
duration = float(duration)

# Sesga el reconocimiento hacia los nombres y el vocabulario del proyecto.
initial_prompt = None
if prompt_path != "-" and Path(prompt_path).exists():
    initial_prompt = Path(prompt_path).read_text(encoding="utf-8").strip() or None

compute = "int8_float16" if device == "cuda" else "int8"
model = WhisperModel(model_name, device=device, compute_type=compute)

COMMON = dict(
    vad_filter=False,          # el VAD descartaba habla real en grabaciones del celu
    word_timestamps=True,
    condition_on_previous_text=False,
    hallucination_silence_threshold=2,
    initial_prompt=initial_prompt,
)


def decode(path, language, offset=0.0):
    segments, info = model.transcribe(path, language=language or None, **COMMON)
    words = [
        {"start": w.start + offset, "end": w.end + offset, "word": w.word}
        for seg in segments
        for w in (seg.words or [])
    ]
    return words, info.language


words, detected = decode(wav, language)

# Tramos de más de 40 s sin una sola palabra: los reintenta aislados.
GAP = 40.0
gaps, cursor = [], 0.0
for w in words:
    if w["start"] - cursor > GAP:
        gaps.append((cursor, w["start"]))
    cursor = max(cursor, w["end"])
if duration - cursor > GAP:
    gaps.append((cursor, duration))

recovered = 0
if gaps:
    with tempfile.TemporaryDirectory() as tmp:
        for i, (start, end) in enumerate(gaps):
            piece = str(Path(tmp) / f"gap{i}.wav")
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start),
                 "-to", str(end), "-i", wav, "-c", "copy", piece],
                check=True,
            )
            extra, _ = decode(piece, detected, offset=start)
            recovered += len(extra)
            words.extend(extra)
    words.sort(key=lambda w: w["start"])

with open(out, "w", encoding="utf-8") as f:
    json.dump(
        {
            "language": detected,
            "words": words,
            "gaps_checked": len(gaps),
            "words_recovered": recovered,
        },
        f,
    )
