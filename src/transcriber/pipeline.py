"""Transcribe audio/vídeo, identifica quién habla y deja una versión revisada.

Cuatro pasos; los pesados corren cada uno en su propio proceso y entorno:

  1. Audio          ffmpeg lo pasa a WAV 16 kHz mono.
  2. ASR            Whisper (faster-whisper), con recuperación de tramos perdidos.
  3. Diarización    pyannote, que dice cuándo cambia la voz.
  4. Revisión       un modelo de lenguaje pone los nombres reales leyendo el
                    contenido, porque la diarización sola se equivoca cuando la
                    grabación es pobre.

ASR y diarización corren separados a propósito: ctranslate2 usa CUDA 12 y el
torch de pyannote trae otro CUDA, y en el mismo proceso se pisan las librerías
de cuDNN. Este orquestador usa solo la stdlib.
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from . import hardware, paths
from .util import duration_of, hhmmss, log, srt_time, to_wav16k, token_hint, utf8_console


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


def run_worker(python, script, args, step, env=None):
    """Corre un paso en su entorno. Siempre con lista de argumentos, nunca shell."""
    try:
        subprocess.run(
            [str(python), str(paths.worker(script)), *[str(a) for a in args]],
            check=True, env=env,
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"Falló {step} (código {e.returncode}). "
            "Corré `transcriber doctor` para revisar la instalación."
        ) from None


def run_asr(python, wav, model, language, device, out, prompt_file, seconds):
    run_worker(
        python, "_asr.py",
        [wav, model, language or "", device, out, prompt_file, seconds],
        "la transcripción", env=paths.asr_child_env(python),
    )


def run_diar(python, wav, device, min_spk, max_spk, out):
    run_worker(
        python, "_diar.py", [wav, device, min_spk or "-", max_spk or "-", out],
        "la separación de voces",
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


def write_outputs(blocks, src: Path, outdir: Path, language: str):
    base = outdir / src.stem

    with open(f"{base}.speakers.txt", "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(f"[{hhmmss(b['start'])}] {b['speaker'] or '?'}: {b['text']}\n")

    with open(f"{base}.speakers.srt", "w", encoding="utf-8") as f:
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
    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return base, payload


def main(argv=None):
    utf8_console()
    context_default = paths.context_path()
    p = argparse.ArgumentParser(
        prog="transcribe-speakers",
        description="Transcribe vídeo/audio, identifica quién habla y revisa el resultado.",
    )
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output-dir", type=Path)
    p.add_argument("-l", "--language", help="es, en... (por defecto: autodetecta)")
    p.add_argument("-m", "--model", default="large-v3-turbo")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--context", type=Path, default=context_default,
                   help=f"Quiénes hablan y vocabulario (por defecto: {context_default})")
    p.add_argument("--no-refine", action="store_true",
                   help="No pasar la revisión final por el modelo de lenguaje")
    p.add_argument("--cpu", action="store_true", help="Forzar CPU (~20x más lento)")
    args = p.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"No existe: {args.input}")

    asr_python, _ = paths.find_env("asr")
    diar_python, _ = paths.find_env("diar")
    if asr_python is None or diar_python is None:
        raise SystemExit("Faltan los entornos de Whisper y pyannote. Corré: transcriber setup")
    if paths.hf_token()[0] is None:
        raise SystemExit(
            "Falta el token de HuggingFace para el modelo de voces.\n"
            f"  {token_hint()}\n"
            "Y aceptá las condiciones en "
            "https://huggingface.co/pyannote/speaker-diarization-community-1"
        )

    system = platform.system()
    gpu = hardware.nvidia_gpu()
    asr_device = hardware.asr_device(
        args.cpu, system, gpu is not None,
        paths.has_cuda_libs(paths.env_dir_of(asr_python), system),
    )
    diar_device = hardware.diar_device(args.cpu, system, platform.machine(), gpu is not None)
    if diar_device == "cpu":
        log("Aviso: sin GPU la diarización va ~20x más lenta.")

    outdir = args.output_dir or args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)

    context = args.context.read_text(encoding="utf-8") if args.context.exists() else ""
    seconds = duration_of(args.input)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav, asr_out, diar_out = tmp / "a.wav", tmp / "asr.json", tmp / "diar.json"
        prompt_file = tmp / "prompt.txt"
        prompt_file.write_text(vocabulary(context), encoding="utf-8")

        log(f"[1/4] Extrayendo audio de {args.input.name} ({hhmmss(seconds)})...")
        to_wav16k(args.input, wav)

        log(f"[2/4] Transcribiendo ({args.model}, {asr_device})...")
        run_asr(asr_python, wav, args.model, args.language, asr_device, asr_out,
                prompt_file, seconds)

        log(f"[3/4] Separando voces ({diar_device})...")
        run_diar(diar_python, wav, diar_device, args.min_speakers, args.max_speakers, diar_out)

        asr = json.loads(asr_out.read_text(encoding="utf-8"))
        turns = json.loads(diar_out.read_text(encoding="utf-8"))

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
            [sys.executable, str(paths.worker("_refine.py")), str(raw), str(args.context),
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
