"""`transcriber setup`: crea los dos entornos aislados con uv.

Son dos a propósito, en las tres plataformas: ctranslate2 (Whisper) necesita
CUDA 12 y el torch que trae pyannote viene con otro CUDA. En el mismo entorno
explotan con CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
"""

import json
import platform
import shutil
import subprocess
from importlib import resources

from . import hardware, paths
from .util import ffmpeg_hint, log, token_hint

PYTHON = "3.12"
ASR_PACKAGES = ["faster-whisper>=1.1"]
ASR_CUDA = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]
# pyannote 4 es la primera con el modelo community-1 y la salida DiarizeOutput.
# Sin el piso, uv puede resolver una 3.x vieja que no importa con el torch actual.
DIAR_PACKAGES = ["pyannote.audio>=4", "pydub"]

PROBE_ASR = (
    "import json, faster_whisper, ctranslate2\n"
    "try:\n    n = ctranslate2.get_cuda_device_count()\n"
    "except Exception:\n    n = 0\n"
    "print(json.dumps({'faster_whisper': faster_whisper.__version__,"
    " 'ctranslate2': ctranslate2.__version__, 'cuda': n > 0}))"
)
PROBE_DIAR = (
    "import json, warnings\n"
    "warnings.simplefilter('ignore')\n"
    "import torch\n"
    "import pyannote.audio as pa\n"
    "mps = getattr(torch.backends, 'mps', None)\n"
    "print(json.dumps({'pyannote': pa.__version__, 'torch': torch.__version__,"
    " 'cuda': torch.cuda.is_available(),"
    " 'mps': bool(mps and mps.is_available())}))"
)


def plan(system, machine, gpu, force_cpu=False, torch_backend=None):
    """Qué se instala en cada entorno según el hardware. Función pura."""
    nvidia = gpu is not None and not force_cpu and system != "Darwin"
    return {
        "asr": ASR_PACKAGES + (ASR_CUDA if nvidia else []),
        "diar": list(DIAR_PACKAGES),
        "torch_backend": torch_backend or hardware.torch_backend(system, nvidia),
        "asr_device": "cuda" if nvidia else "cpu",
        "diar_device": hardware.diar_device(force_cpu, system, machine, gpu is not None),
    }


def uv(*args):
    subprocess.run(["uv", *[str(a) for a in args]], check=True)


def probe(python, code, env=None, timeout=300):
    """Corre `code` en un entorno y devuelve el JSON que imprime, o {'error': ...}."""
    try:
        out = subprocess.run(
            [str(python), "-c", code], capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines:
        detail = (out.stderr or out.stdout).strip().splitlines()
        return {"error": detail[-1] if detail else f"código {out.returncode}"}
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {"error": lines[-1]}


def probe_asr(python):
    return probe(python, PROBE_ASR, env=paths.asr_child_env(python))


def probe_diar(python):
    return probe(python, PROBE_DIAR)


def create_env(target, packages):
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    uv("venv", "--python", PYTHON, target)
    python = paths.venv_python(target)
    uv("pip", "install", "--python", python, *packages)
    return python


def installed_version(python, dist):
    out = subprocess.run(
        [str(python), "-c",
         f"import importlib.metadata as m; print(m.version('{dist}'))"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return out.stdout.strip().split("+")[0]


def swap_torch(python, backend):
    """Reemplaza torch y torchaudio por el build de `backend`, en la misma versión.

    Va en un segundo paso, y no con `--torch-backend` en la instalación entera,
    porque uv también llevaría torchcodec (dependencia de pyannote) al índice de
    PyTorch, y ahí no hay ruedas de torchcodec con CUDA para Windows.
    """
    pins = [f"{d}=={installed_version(python, d)}" for d in ("torch", "torchaudio")]
    try:
        uv("pip", "install", "--python", python, "--no-deps",
           "--reinstall-package", "torch", "--reinstall-package", "torchaudio",
           *pins, "--torch-backend", backend)
    except subprocess.CalledProcessError:
        log(f"Aviso: no pude instalar {' '.join(pins)} con --torch-backend {backend}. "
            "El entorno de voces queda en CPU. Probá otro valor: "
            "transcriber setup --force --torch-backend cu128")
        return False
    return True


def copy_context_example():
    target = paths.context_path()
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    example = resources.files(paths.APP) / "context.example.md"
    target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    log(f"==> {target} (copiado del ejemplo: editalo)")


def main(args):
    if shutil.which("uv") is None:
        raise SystemExit("Falta uv en el PATH. Instalalo desde https://docs.astral.sh/uv/")

    system, machine = platform.system(), platform.machine()
    gpu = hardware.nvidia_gpu()
    todo = plan(system, machine, gpu, args.cpu, args.torch_backend)

    if gpu:
        log(f"GPU: {gpu['name']}, {gpu['vram_mb']} MB, driver {gpu['driver']}")
    elif system == "Darwin":
        log("macOS: Whisper va por CPU (ctranslate2 no usa la GPU de Apple); "
            f"las voces por {todo['diar_device']}.")
    else:
        log("Sin GPU NVIDIA: todo va por CPU, unas 20 veces más lento.")

    labels = {"asr": "Whisper (ASR)", "diar": "pyannote (separación de voces)"}
    for kind in ("asr", "diar"):
        python, origin = paths.find_env(kind)
        if origin == "variable":
            log(f"==> {labels[kind]}: se usa {python} ({paths.PYTHON_VAR[kind]}); no lo toco.")
            continue
        if origin == "uv tool" and not args.force:
            log(f"==> {labels[kind]}: sigo usando el entorno de `uv tool` de la v0.1\n"
                f"    {python}\n    Para crear el nuevo: transcriber setup --force")
            continue
        if origin == "setup" and not args.force:
            log(f"==> {labels[kind]}: ya existe ({python}). Con --force se rehace.")
            continue

        target = paths.data_dir() / "envs" / kind
        log(f"==> {labels[kind]}: creando {target}")
        python = create_env(target, todo[kind])
        if kind == "diar" and todo["torch_backend"]:
            log(f"==> torch con GPU (--torch-backend {todo['torch_backend']})")
            swap_torch(python, todo["torch_backend"])

        info = probe_asr(python) if kind == "asr" else probe_diar(python)
        if "error" in info:
            raise SystemExit(f"El entorno quedó mal: {info['error']}")
        log("    " + ", ".join(f"{k} {v}" for k, v in info.items()))

    copy_context_example()

    pending = []
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pending.append(f"Instalar ffmpeg: {ffmpeg_hint(system)}")
    if paths.hf_token()[0] is None:
        pending.append(
            "Aceptar las condiciones del modelo en "
            "https://huggingface.co/pyannote/speaker-diarization-community-1,\n"
            "    crear un token de lectura en https://huggingface.co/settings/tokens y cargarlo:\n"
            f"    {token_hint(system)}"
        )
    if pending:
        log("\nFalta:")
        for i, item in enumerate(pending, 1):
            log(f"  {i}. {item}")
    log("\nListo. Revisá todo con: transcriber doctor")
    return 0
