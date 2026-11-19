"""transcribe — pasa audio o vídeo a texto con Whisper, sin separar voces.

Usa el mismo entorno y el mismo paso ASR que `transcribe-speakers`, así que
hereda sus cuidados: VAD apagado, idioma votado sobre varias ventanas y
reintento de los tramos que quedaron sin texto.
"""

import argparse
import json
import platform
import tempfile
from pathlib import Path

from . import hardware, paths
from .pipeline import run_asr
from .util import duration_of, log, srt_time, to_wav16k, utf8_console

FORMATS = ("txt", "srt", "vtt", "tsv", "json")

EPILOG = """\
Ejemplos
  transcribe clase.mp4                       # texto junto al vídeo
  transcribe -l es -f srt entrevista.mkv     # subtítulos en español
  transcribe -m medium --cpu podcast.mp3
"""


def render(fmt, result):
    """Texto del archivo de salida en el formato pedido. Función pura."""
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result["segments"]
        if s["text"].strip()
    ]
    if fmt == "txt":
        return "".join(f"{s['text']}\n" for s in segments)
    if fmt == "srt":
        return "".join(
            f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n\n"
            for i, s in enumerate(segments, 1)
        )
    if fmt == "vtt":
        return "WEBVTT\n\n" + "".join(
            f"{srt_time(s['start'], '.')} --> {srt_time(s['end'], '.')}\n{s['text']}\n\n"
            for s in segments
        )
    if fmt == "tsv":
        return "start\tend\ttext\n" + "".join(
            f"{round(s['start'] * 1000)}\t{round(s['end'] * 1000)}\t"
            f"{s['text'].replace(chr(9), ' ')}\n"
            for s in segments
        )
    if fmt == "json":
        payload = {
            "text": " ".join(s["text"] for s in segments),
            "segments": [
                {"id": i, "start": round(s["start"], 3), "end": round(s["end"], 3),
                 "text": s["text"]}
                for i, s in enumerate(segments)
            ],
            "language": result["language"],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    raise ValueError(f"formato desconocido: {fmt}")


def write_outputs(result, src: Path, outdir: Path, fmt):
    written = []
    for f in (FORMATS if fmt == "all" else (fmt,)):
        path = outdir / f"{src.stem}.{f}"
        path.write_text(render(f, result), encoding="utf-8")
        written.append(path)
    return written


def transcribe_one(src: Path, python, args, device):
    """Devuelve el resultado del paso ASR, o None si falló."""
    seconds = duration_of(src)
    with tempfile.TemporaryDirectory() as tmp:
        wav, out = Path(tmp) / "a.wav", Path(tmp) / "asr.json"
        to_wav16k(src, wav)
        for dev in dict.fromkeys((device, "cpu")):       # GPU y, si falla, CPU
            code = run_asr(python, wav, out, args.model, dev, seconds,
                           language=args.language, threads=8, check=False)
            if code == 0:
                return json.loads(out.read_text(encoding="utf-8"))
            if dev == "cuda":
                log("transcribe: falló en GPU, reintentando en CPU...")
    return None


def main(argv=None):
    utf8_console()
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Pasa audio o vídeo a texto con Whisper (sin separar voces).",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="+", type=Path, metavar="archivo")
    p.add_argument("-m", "--model", default="large-v3-turbo",
                   help="tiny | base | small | medium | large-v3 | large-v3-turbo "
                        "(por defecto: large-v3-turbo)")
    p.add_argument("-l", "--lang", "--language", dest="language",
                   help="es, en, fr... (por defecto: se detecta solo)")
    p.add_argument("-f", "--format", default="txt", choices=(*FORMATS, "all"),
                   help="formato de salida (por defecto: txt)")
    p.add_argument("-o", "--output-dir", type=Path,
                   help="dónde dejar el resultado (por defecto: junto al archivo)")
    dev = p.add_mutually_exclusive_group()
    dev.add_argument("--cpu", action="store_true", help="forzar CPU")
    dev.add_argument("--gpu", "--cuda", dest="gpu", action="store_true", help="forzar GPU")
    args = p.parse_args(argv)

    python, _ = paths.find_env("asr")
    if python is None:
        raise SystemExit("Falta el entorno de Whisper. Corré: transcriber setup")

    system = platform.system()
    if args.gpu:
        device = "cuda"
    else:
        device = hardware.asr_device(
            args.cpu, system, hardware.nvidia_gpu() is not None,
            paths.has_cuda_libs(paths.env_dir_of(python), system),
        )

    status = 0
    for src in args.files:
        if not src.is_file():
            log(f"transcribe: no existe '{src}'")
            status = 1
            continue
        log(f"==> {src}  (modelo {args.model}, {device})")
        result = transcribe_one(src, python, args, device)
        if result is None:
            status = 1
            continue
        outdir = args.output_dir or src.parent
        outdir.mkdir(parents=True, exist_ok=True)
        for path in write_outputs(result, src, outdir, args.format):
            log(str(path))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
