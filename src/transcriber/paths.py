"""Dónde vive cada cosa en cada sistema operativo. Solo stdlib.

Las funciones reciben `system`, `env` y `home` para poder testearlas sin tocar
la máquina; sin argumentos usan los valores reales.
"""

import os
import platform
import subprocess
from importlib import resources
from pathlib import Path

APP = "transcriber"

# Entornos de la v0.1, instalados con `uv tool install`. Se siguen usando si
# existen y todavía no se crearon los nuevos con `transcriber setup`.
LEGACY = {"asr": "whisper-ctranslate2", "diar": "pyannote-audio"}
PYTHON_VAR = {"asr": "TRANSCRIBER_ASR_PYTHON", "diar": "TRANSCRIBER_DIAR_PYTHON"}

_ASK_UV = object()


def _defaults(system, env, home):
    return (
        system or platform.system(),
        os.environ if env is None else env,
        Path.home() if home is None else Path(home),
    )


def data_dir(system=None, env=None, home=None) -> Path:
    """Donde `transcriber setup` crea los entornos."""
    system, env, home = _defaults(system, env, home)
    if env.get("TRANSCRIBER_HOME"):
        return Path(env["TRANSCRIBER_HOME"])
    if system == "Windows":
        base = Path(env["LOCALAPPDATA"]) if env.get("LOCALAPPDATA") else home / "AppData" / "Local"
        return base / APP
    if system == "Darwin":
        return home / "Library" / "Application Support" / APP
    return home / ".local" / "share" / APP


def context_path(system=None, env=None, home=None) -> Path:
    """El archivo de contexto por defecto (participantes y vocabulario)."""
    system, env, home = _defaults(system, env, home)
    if env.get("TRANSCRIBER_CONTEXT"):
        return Path(env["TRANSCRIBER_CONTEXT"])
    if system == "Windows":
        base = Path(env["APPDATA"]) if env.get("APPDATA") else home / "AppData" / "Roaming"
        return base / APP / "context.md"
    if system == "Darwin":
        return home / "Library" / "Application Support" / APP / "context.md"
    return home / ".config" / APP / "context.md"


def venv_python(env_dir, system=None) -> Path:
    system = system or platform.system()
    if system == "Windows":
        return Path(env_dir) / "Scripts" / "python.exe"
    return Path(env_dir) / "bin" / "python"


def env_dir_of(python) -> Path:
    """Raíz del venv de un intérprete. Sin resolver symlinks: en Linux y macOS
    bin/python apunta al Python base y nos sacaría del entorno."""
    return Path(python).parent.parent


def site_packages(env_dir, system=None):
    system = system or platform.system()
    if system == "Windows":
        return Path(env_dir) / "Lib" / "site-packages"
    found = sorted((Path(env_dir) / "lib").glob("python3*/site-packages"))
    return found[0] if found else None


def nvidia_lib_dirs(env_dir, system=None) -> list:
    """Carpetas con las librerías de CUDA que instalan los paquetes nvidia-*-cu12.

    En Linux son .so en nvidia/<lib>/lib; en Windows, DLL en nvidia/<lib>/bin.
    """
    system = system or platform.system()
    sp = site_packages(env_dir, system)
    if sp is None:
        return []
    sub = "bin" if system == "Windows" else "lib"
    return sorted(d for d in (sp / "nvidia").glob(f"*/{sub}") if d.is_dir())


def has_cuda_libs(env_dir, system=None) -> bool:
    """¿El entorno ASR tiene cuBLAS de CUDA 12? Sin eso ctranslate2 no usa la GPU."""
    return any(
        d.parent.name == "cublas" and any(d.glob("*cublas*12*"))
        for d in nvidia_lib_dirs(env_dir, system)
    )


def uv_tool_dir(system=None, env=None, home=None):
    """Directorio de `uv tool`: se lo pregunta a uv y, si no está, usa el default."""
    system, env, home = _defaults(system, env, home)
    try:
        out = subprocess.run(
            ["uv", "tool", "dir"], capture_output=True, text=True, encoding="utf-8",
            check=True, timeout=20,
        )
        if out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    if system == "Windows":
        base = Path(env["APPDATA"]) if env.get("APPDATA") else home / "AppData" / "Roaming"
        return base / "uv" / "tools"
    return home / ".local" / "share" / "uv" / "tools"


def find_env(kind, system=None, env=None, home=None, tool_dir=_ASK_UV):
    """Intérprete del entorno `kind` ('asr' o 'diar') y de dónde salió.

    Orden: variable de escape, entorno creado por `transcriber setup`, entorno
    viejo de `uv tool`. Devuelve (None, "falta") si no hay ninguno.
    """
    system, env, home = _defaults(system, env, home)
    if env.get(PYTHON_VAR[kind]):
        return Path(env[PYTHON_VAR[kind]]), "variable"
    own = venv_python(data_dir(system, env, home) / "envs" / kind, system)
    if own.exists():
        return own, "setup"
    if tool_dir is _ASK_UV:
        tool_dir = uv_tool_dir(system, env, home)
    if tool_dir is not None:
        legacy = venv_python(Path(tool_dir) / LEGACY[kind], system)
        if legacy.exists():
            return legacy, "uv tool"
    return None, "falta"


def asr_child_env(python, system=None, env=None) -> dict:
    """Entorno para lanzar el paso ASR con las librerías de CUDA 12 a la vista."""
    system = system or platform.system()
    child = dict(os.environ if env is None else env)
    dirs = [str(d) for d in nvidia_lib_dirs(env_dir_of(python), system)]
    if not dirs:
        return child
    if system == "Windows":
        # ctranslate2 carga cuBLAS y cuDNN recién al crear el modelo y las busca
        # en el PATH; el worker además las registra con os.add_dll_directory.
        child["PATH"] = os.pathsep.join(dirs + [child.get("PATH", "")])
        child["TRANSCRIBER_DLL_DIRS"] = os.pathsep.join(dirs)
    else:
        previous = [child["LD_LIBRARY_PATH"]] if child.get("LD_LIBRARY_PATH") else []
        child["LD_LIBRARY_PATH"] = ":".join(dirs + previous)
    return child


def hf_token_file(env=None, home=None) -> Path:
    """Donde `hf auth login` guarda el token."""
    _, env, home = _defaults(None, env, home)
    if env.get("HF_TOKEN_PATH"):
        return Path(env["HF_TOKEN_PATH"])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "token"
    cache = Path(env["XDG_CACHE_HOME"]) if env.get("XDG_CACHE_HOME") else home / ".cache"
    return cache / "huggingface" / "token"


def hf_token(env=None, home=None):
    """(token, origen) o (None, "") si no hay ni variable ni token guardado."""
    _, env, home = _defaults(None, env, home)
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if env.get(var, "").strip():
            return env[var].strip(), f"variable {var}"
    path = hf_token_file(env, home)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    return (token, str(path)) if token else (None, "")


def worker(name) -> Path:
    """Ruta de un script del paquete que corre en otro intérprete (_asr.py...)."""
    return Path(str(resources.files(APP) / name))
