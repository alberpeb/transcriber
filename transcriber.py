"""Transcribe audio/vídeo, identifica quién habla y deja una versión revisada.

Tres pasos, cada uno en su propio proceso:

  1. ASR            Whisper (faster-whisper), con recuperación de tramos perdidos.
  2. Diarización    pyannote, que dice cuándo cambia la voz.
  3. Revisión       un modelo de lenguaje pone los nombres reales leyendo el
                    contenido, porque la diarización sola se equivoca cuando la
                    grabación es pobre.

ASR y diarización corren separados a propósito: ctranslate2 usa CUDA 12 y torch
usa CUDA 13, y en el mismo proceso se pisan las librerías de cuDNN.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
TOOLS = Path.home() / ".local/share/uv/tools"
WHISPER_ENV = TOOLS / "whisper-ctranslate2"          # ctranslate2, CUDA 12
WHISPER_NV = WHISPER_ENV / "lib/python3.12/site-packages/nvidia"
PYANNOTE_ENV = TOOLS / "pyannote-audio"              # torch, CUDA 13
CONTEXT = Path.home() / ".config/transcriber/context.md"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def have_gpu():
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True)
        return (WHISPER_NV / "cublas/lib/libcublas.so.12").exists()
    except Exception:
        return False


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def to_wav16k(src: Path, dst: Path):
    """pyannote y whisper quieren 16 kHz mono; ffmpeg lee cualquier contenedor."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def vocabulary(context: str) -> str:
    """Términos del contexto, para sesgar el reconocimiento hacia ellos."""
    words = []
    for line in context.splitlines():
        line = line.strip()
        if line.startswith("- **"):                      # nombres de personas
            words.append(line.split("**")[1])
        elif line and not line.startswith(("#", "-", ">")) and "," in line:
            words += [w.strip() for w in line.split(",") if 2 < len(w.strip()) < 40]
    seen = list(dict.fromkeys(words))[:60]
    return ", ".join(seen) + "." if seen else ""


def run_asr(wav, model, language, device, out, prompt_file, seconds):
    env = dict(os.environ)
    if device == "cuda":
        libs = [WHISPER_NV / d / "lib" for d in ("cublas", "cudnn", "cuda_nvrtc")]
        env["LD_LIBRARY_PATH"] = ":".join(
            [str(p) for p in libs] + [env.get("LD_LIBRARY_PATH", "")]
        )
    subprocess.run(
        [str(WHISPER_ENV / "bin/python3"), str(HERE / "_asr.py"), str(wav), model,
         language or "", device, str(out), str(prompt_file), str(seconds)],
        check=True, env=env,
    )


def run_diar(wav, device, min_spk, max_spk, out):
    subprocess.run(
        [str(PYANNOTE_ENV / "bin/python3"), str(HERE / "_diar.py"), str(wav), device,
         str(min_spk or "-"), str(max_spk or "-"), str(out)],
        check=True,
    )


def speaker_at(turns, start, end):
    """Speaker que más solapa con [start, end]; si no solapa nadie, el más cercano."""
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(end, t["end"]) - max(start, t["start"])
        if overlap > best_overlap:
            best, best_overlap = t["speaker"], overlap
    if best:
        return best
    mid = (start + end) / 2
    nearest = min(
        turns,
        key=lambda t: 0 if t["start"] <= mid <= t["end"]
        else min(abs(mid - t["start"]), abs(mid - t["end"])),
        default=None,
    )
    return nearest["speaker"] if nearest else None


def group(words, turns):
    """Agrupa palabras consecutivas del mismo speaker en bloques."""
    blocks = []
    for w in words:
        spk = speaker_at(turns, w["start"], w["end"])
        if blocks and blocks[-1]["speaker"] == spk:
            blocks[-1]["end"] = w["end"]
            blocks[-1]["text"] += w["word"]
        else:
            blocks.append(
                {"speaker": spk, "start": w["start"], "end": w["end"], "text": w["word"]}
            )
    for b in blocks:
        b["text"] = b["text"].strip()
    return [b for b in blocks if b["text"]]


def hhmmss(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_time(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(blocks, src: Path, outdir: Path, language: str):
    base = outdir / src.stem

    with open(f"{base}.speakers.txt", "w") as f:
        for b in blocks:
            f.write(f"[{hhmmss(b['start'])}] {b['speaker'] or '?'}: {b['text']}\n")

    with open(f"{base}.speakers.srt", "w") as f:
        for i, b in enumerate(blocks, 1):
            f.write(f"{i}\n{srt_time(b['start'])} --> {srt_time(b['end'])}\n")
            f.write(f"{b['speaker'] or '?'}: {b['text']}\n\n")

    payload = {
        "source": str(src),
        "language": language,
        "speakers": sorted({b["speaker"] or "?" for b in blocks}),
        "duration": round(blocks[-1]["end"], 2) if blocks else 0,
        "blocks": [
            {"speaker": b["speaker"] or "?", "start": round(b["start"], 2),
             "end": round(b["end"], 2), "text": b["text"]}
            for b in blocks
        ],
    }
    with open(f"{base}.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return base, payload


def main():
    p = argparse.ArgumentParser(
        description="Transcribe vídeo/audio, identifica quién habla y revisa el resultado."
    )
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output-dir", type=Path)
    p.add_argument("-l", "--language", help="es, en... (por defecto: autodetecta)")
    p.add_argument("-m", "--model", default="large-v3-turbo")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--context", type=Path, default=CONTEXT,
                   help=f"Quiénes hablan y vocabulario (por defecto: {CONTEXT})")
    p.add_argument("--no-refine", action="store_true",
                   help="No pasar la revisión final por el modelo de lenguaje")
    p.add_argument("--cpu", action="store_true", help="Forzar CPU (~20x más lento)")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No existe: {args.input}")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("Falta HF_TOKEN. Ponelo en ~/.bashrc: export HF_TOKEN=hf_...")

    device = "cpu" if args.cpu else ("cuda" if have_gpu() else "cpu")
    if device == "cpu":
        log("Aviso: sin GPU la diarización va ~20x más lenta.")

    outdir = args.output_dir or args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)

    context = args.context.read_text() if args.context.exists() else ""
    seconds = duration_of(args.input)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav, asr_out, diar_out = tmp / "a.wav", tmp / "asr.json", tmp / "diar.json"
        prompt_file = tmp / "prompt.txt"
        prompt_file.write_text(vocabulary(context))

        log(f"[1/4] Extrayendo audio de {args.input.name} ({hhmmss(seconds)})...")
        to_wav16k(args.input, wav)

        log(f"[2/4] Transcribiendo ({args.model}, {device})...")
        run_asr(wav, args.model, args.language, device, asr_out, prompt_file, seconds)

        log(f"[3/4] Separando voces ({device})...")
        run_diar(wav, device, args.min_speakers, args.max_speakers, diar_out)

        asr = json.loads(asr_out.read_text())
        turns = json.loads(diar_out.read_text())

    if asr["gaps_checked"]:
        log(f"      {asr['gaps_checked']} tramo(s) sin texto reintentados, "
            f"{asr['words_recovered']} palabras recuperadas")

    blocks = group(asr["words"], turns)
    base, payload = write_outputs(blocks, args.input, outdir, asr["language"])

    raw = Path(f"{base}.speakers.txt")
    revisado = Path(f"{base}.revisado.txt")
    done = False
    if args.no_refine:
        log("[4/4] Revisión salteada (--no-refine)")
    else:
        log("[4/4] Revisando y poniendo nombres...")
        meta = (f"Archivo: {args.input.name}. Duración: {hhmmss(payload['duration'])}. "
                f"Idioma detectado: {payload['language']}. "
                f"La separación automática encontró {len(payload['speakers'])} voces.")
        had_old = revisado.exists()
        done = subprocess.run(
            [sys.executable, str(HERE / "_refine.py"), str(raw), str(args.context),
             meta, str(revisado)]
        ).returncode == 0
        if not done:
            log("      Sin revisión: queda la versión cruda. Para habilitarla, abrí una")
            log("      terminal, ejecutá 'claude' una vez para renovar la sesión, y")
            log("      volvé a correr esto.")
            if had_old:
                log(f"      Ojo: {revisado.name} es de una corrida anterior, no de esta.")

    log(f"\n{payload['language']}  ·  {hhmmss(payload['duration'])}  ·  "
        f"{len(asr['words'])} palabras  ·  {len(payload['speakers'])} voces")
    log(f"\n{revisado if done else raw}")
    if done:
        log(f"{raw}  (crudo, con las etiquetas automáticas)")
    log(f"{base}.speakers.srt\n{base}.json")


if __name__ == "__main__":
    main()
