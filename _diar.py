"""Paso diarización — corre en el entorno de pyannote (torch).

Va en un proceso y un entorno aparte del ASR a propósito: ctranslate2 usa CUDA 12
y el torch que trae pyannote viene con otro CUDA; juntos se pisan cuDNN.
"""

import json
import os
import sys
import wave

MODEL = "pyannote/speaker-diarization-community-1"
SMALL_GPU_GB = 6.0      # por debajo de esto, el batch por defecto (32) no entra
SMALL_BATCH = 8


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def batch_size_for(vram_gb, override=None):
    """Batch de segmentación y embeddings; None deja el que trae pyannote.

    En GPUs de 4 GB el batch por defecto llena la memoria: la corrida termina
    bien, pero llena el log de avisos de OOM del allocator de CUDA.
    """
    if override:
        return max(1, int(override))
    if vram_gb is not None and vram_gb < SMALL_GPU_GB:
        return SMALL_BATCH
    return None


def resolve_device(requested, cuda_ok, mps_ok):
    """Baja a CPU si el dispositivo pedido no está disponible en este torch."""
    if requested == "cuda" and not cuda_ok:
        return "cpu"
    if requested == "mps" and not mps_ok:
        return "cpu"
    return requested


def token_help():
    return (
        "No pude bajar el modelo de voces. Hace falta:\n"
        f"  1. Aceptar las condiciones en https://huggingface.co/{MODEL}\n"
        "  2. Un token de lectura (https://huggingface.co/settings/tokens) en la\n"
        "     variable HF_TOKEN, o haber corrido `hf auth login`.\n"
        "Corré `transcriber doctor` para ver qué falta."
    )


def main(argv):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    wav, device, min_spk, max_spk, out = argv[:5]

    # Si a MPS le falta alguna operación, que caiga a CPU en vez de abortar.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    from pyannote.audio import Pipeline

    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    actual = resolve_device(device, torch.cuda.is_available(), mps_ok)
    if actual != device:
        log(f"      Aviso: '{device}' no está disponible en este entorno; voy por CPU "
            "(~1x tiempo real). Revisá con `transcriber doctor`.")
        device = actual

    # token=None hace que huggingface_hub use el token guardado por `hf auth login`.
    try:
        pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN") or None)
    except Exception as e:
        log(f"{type(e).__name__}: {e}\n\n{token_help()}")
        return 3
    if pipeline is None:
        log(token_help())
        return 3

    vram_gb = None
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    batch = batch_size_for(vram_gb, os.environ.get("TRANSCRIBER_DIAR_BATCH"))
    if batch:
        for attr in ("segmentation_batch_size", "embedding_batch_size"):
            if hasattr(pipeline, attr):
                setattr(pipeline, attr, batch)

    kwargs = {}
    if min_spk != "-":
        kwargs["min_speakers"] = int(min_spk)
    if max_spk != "-":
        kwargs["max_speakers"] = int(max_spk)

    # El WAV ya es 16 kHz mono PCM de 16 bits: se carga con la stdlib y se le pasa
    # a pyannote en memoria. Así no depende de torchcodec ni de las librerías
    # compartidas de FFmpeg, que en Windows y macOS suelen no estar.
    with wave.open(wav, "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        if w.getsampwidth() != 2:
            raise SystemExit("El WAV tiene que ser PCM de 16 bits.")
        frames = bytearray(w.readframes(w.getnframes()))
    waveform = torch.frombuffer(frames, dtype=torch.int16).float() / 32768.0
    waveform = waveform.reshape(-1, channels).mean(dim=1).unsqueeze(0)
    audio = {"waveform": waveform, "sample_rate": rate}

    def run(dev):
        pipeline.to(torch.device(dev))
        return pipeline(audio, **kwargs)

    try:
        output = run(device)
    except Exception as e:
        if device != "mps":
            raise
        log(f"      Aviso: falló en MPS ({type(e).__name__}); reintento en CPU.")
        output = run("cpu")

    # pyannote 4 devuelve un DiarizeOutput; la anotación está en speaker_diarization.
    ann = output.speaker_diarization
    turns = [
        {"start": seg.start, "end": seg.end, "speaker": label}
        for seg, _, label in ann.itertracks(yield_label=True)
    ]

    with open(out, "w", encoding="utf-8") as f:
        json.dump(turns, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
