"""`transcriber doctor`: revisa la instalación y dice, en una línea por cosa,
qué falta y cómo arreglarlo."""

import platform
import shutil
import subprocess
import urllib.error
import urllib.request

from . import envs, hardware, paths
from ._diar import MODEL
from .util import ffmpeg_hint, token_hint

OK, WARN, MISSING = "ok", "aviso", "falta"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """No sigue redirecciones, para no mandarle el token a otro host."""

    def redirect_request(self, *args, **kwargs):
        return None


def gated_status(token, timeout=15):
    """Código HTTP al pedir un archivo del modelo gated; None si no hay red."""
    request = urllib.request.Request(
        f"https://huggingface.co/{MODEL}/resolve/main/config.yaml",
        method="HEAD",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "transcriber-doctor"},
    )
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


def gated_check(status):
    """Traduce el código HTTP a (estado, detalle). Función pura."""
    url = f"https://huggingface.co/{MODEL}"
    if status is None:
        return WARN, "no pude consultar huggingface.co (¿sin conexión?)"
    if 200 <= status < 400:
        return OK, "acceso concedido"
    if status == 401:
        return MISSING, "el token no es válido: creá otro en https://huggingface.co/settings/tokens"
    if status == 403:
        return MISSING, f"entrá con tu cuenta y aceptá las condiciones en {url}"
    return WARN, f"respuesta inesperada de huggingface.co (HTTP {status})"


def uv_version():
    try:
        out = subprocess.run(["uv", "--version"], capture_output=True, text=True,
                             encoding="utf-8", check=True, timeout=20)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def checks():
    """Lista de (estado, nombre, detalle)."""
    system, machine = platform.system(), platform.machine()
    rows = []

    for tool in ("ffmpeg", "ffprobe"):
        exe = shutil.which(tool)
        rows.append((OK, tool, exe) if exe else (MISSING, tool, ffmpeg_hint(system)))

    uv = uv_version()
    rows.append((OK, "uv", uv) if uv else
                (WARN, "uv", "no está en el PATH (hace falta para `transcriber setup`): "
                             "https://docs.astral.sh/uv/"))

    gpu = hardware.nvidia_gpu()
    if gpu:
        rows.append((OK, "GPU", f"{gpu['name']}, {gpu['vram_mb']} MB, driver {gpu['driver']}"))
    elif system == "Darwin":
        chip = "Apple Silicon (MPS para las voces)" if machine == "arm64" else "Intel (todo CPU)"
        rows.append((OK, "GPU", chip))
    else:
        rows.append((WARN, "GPU", "sin NVIDIA: todo va por CPU, unas 20 veces más lento"))

    asr_python, asr_origin = paths.find_env("asr")
    asr_info = {}
    if asr_python is None:
        rows.append((MISSING, "entorno Whisper", "corré: transcriber setup"))
    else:
        asr_info = envs.probe_asr(asr_python)
        if "error" in asr_info:
            rows.append((MISSING, "entorno Whisper",
                         f"{asr_python} no anda ({asr_info['error']}): "
                         "transcriber setup --force"))
        else:
            rows.append((OK, "entorno Whisper",
                         f"faster-whisper {asr_info['faster_whisper']}, ctranslate2 "
                         f"{asr_info['ctranslate2']} — {asr_python} ({asr_origin})"))

    diar_python, diar_origin = paths.find_env("diar")
    diar_info = {}
    if diar_python is None:
        rows.append((MISSING, "entorno voces", "corré: transcriber setup"))
    else:
        diar_info = envs.probe_diar(diar_python)
        if "error" in diar_info:
            rows.append((MISSING, "entorno voces",
                         f"{diar_python} no anda ({diar_info['error']}): "
                         "transcriber setup --force"))
        else:
            rows.append((OK, "entorno voces",
                         f"pyannote {diar_info['pyannote']}, torch {diar_info['torch']} "
                         f"— {diar_python} ({diar_origin})"))

    if asr_python is not None and diar_python is not None:
        asr_dev = hardware.asr_device(
            False, system, gpu is not None,
            paths.has_cuda_libs(paths.env_dir_of(asr_python), system),
        )
        diar_dev = hardware.diar_device(False, system, machine, gpu is not None)
        if diar_dev == "cuda" and not diar_info.get("cuda", True):
            diar_dev = "cpu"
        if diar_dev == "mps" and not diar_info.get("mps", True):
            diar_dev = "cpu"
        detail = f"Whisper: {asr_dev} · voces: {diar_dev}"
        wasted = gpu is not None and "cpu" in (asr_dev, diar_dev)
        if wasted:
            detail += " — hay GPU pero un entorno no la usa: transcriber setup --force"
        rows.append((WARN if wasted else OK, "dispositivo", detail))

    token, origin = paths.hf_token()
    if token is None:
        rows.append((MISSING, "token HuggingFace", token_hint(system)))
        rows.append((WARN, "modelo de voces", "sin token no puedo verificar el acceso"))
    else:
        rows.append((OK, "token HuggingFace", origin))
        state, detail = gated_check(gated_status(token))
        rows.append((state, "modelo de voces", detail))

    claude = shutil.which("claude")
    rows.append((OK, "claude", claude) if claude else
                (WARN, "claude", "no está en el PATH: el paso 4 (nombres) se saltea y queda "
                                 "la versión cruda"))

    context = paths.context_path()
    rows.append((OK, "contexto", str(context)) if context.exists() else
                (WARN, "contexto", f"no existe {context} (opcional; `transcriber setup` "
                                   "copia un ejemplo)"))
    return rows


def main(args):
    rows = checks()
    width = max(len(name) for _, name, _ in rows)
    for state, name, detail in rows:
        print(f"[{state:^5}] {name:<{width}}  {detail}")
    missing = sum(1 for state, _, _ in rows if state == MISSING)
    print()
    print("Todo en orden." if not missing
          else f"Falta resolver {missing} cosa(s) marcada(s) [falta].")
    return 1 if missing else 0
