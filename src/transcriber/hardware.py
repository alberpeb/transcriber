"""Detección de hardware y elección de dispositivo. Solo stdlib."""

import subprocess


def parse_nvidia_smi(text):
    """Primera GPU de `nvidia-smi --query-gpu=name,memory.total,driver_version`."""
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            try:
                vram = int(float(parts[1]))
            except ValueError:
                vram = 0
            return {"name": parts[0], "vram_mb": vram, "driver": parts[2]}
    return None


def nvidia_gpu():
    """Datos de la GPU NVIDIA, o None si no hay (o no anda el driver)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(out.stdout)


def asr_device(force_cpu, system, has_nvidia, has_cuda_libs):
    """ctranslate2 solo usa GPU con CUDA: en macOS siempre va por CPU."""
    if force_cpu or system == "Darwin":
        return "cpu"
    return "cuda" if has_nvidia and has_cuda_libs else "cpu"


def diar_device(force_cpu, system, machine, has_nvidia):
    """pyannote usa CUDA con NVIDIA y MPS en Apple Silicon."""
    if force_cpu:
        return "cpu"
    if system == "Darwin":
        return "mps" if machine == "arm64" else "cpu"
    return "cuda" if has_nvidia else "cpu"


def torch_backend(system, has_nvidia):
    """Valor de `uv pip install --torch-backend` para el entorno de voces.

    None deja el torch por defecto de PyPI: en Linux ya trae CUDA y en macOS trae
    MPS. En Windows el de PyPI es solo CPU, así que con NVIDIA se reinstala desde
    el índice de PyTorch; "auto" hace que uv elija el build de CUDA según el
    driver que reporta nvidia-smi.
    """
    return "auto" if system == "Windows" and has_nvidia else None
