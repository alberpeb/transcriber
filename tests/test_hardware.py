import pytest

from transcriber import envs, hardware
from transcriber._diar import batch_size_for, resolve_device
from transcriber.doctor import MISSING, OK, WARN, gated_check


def test_parse_nvidia_smi():
    gpu = hardware.parse_nvidia_smi("NVIDIA GeForce RTX 5070 Ti, 16303, 617.42\n")
    assert gpu == {"name": "NVIDIA GeForce RTX 5070 Ti", "vram_mb": 16303, "driver": "617.42"}
    two = "NVIDIA GeForce GTX 1650, 4096, 580.95.05\nNVIDIA RTX A2000, 12282, 580.95.05\n"
    assert hardware.parse_nvidia_smi(two)["vram_mb"] == 4096
    assert hardware.parse_nvidia_smi("") is None
    assert hardware.parse_nvidia_smi("No devices were found\n") is None


@pytest.mark.parametrize("force_cpu, system, nvidia, libs, expected", [
    (False, "Linux", True, True, "cuda"),
    (False, "Windows", True, True, "cuda"),
    (True, "Linux", True, True, "cpu"),          # --cpu
    (False, "Linux", True, False, "cpu"),        # hay GPU pero el entorno no trae cuBLAS
    (False, "Linux", False, True, "cpu"),        # librerías sin GPU (o driver caído)
    (False, "Darwin", False, False, "cpu"),
    (False, "Darwin", True, True, "cpu"),        # ctranslate2 no tiene GPU en macOS
])
def test_asr_device(force_cpu, system, nvidia, libs, expected):
    assert hardware.asr_device(force_cpu, system, nvidia, libs) == expected


@pytest.mark.parametrize("force_cpu, system, machine, nvidia, expected", [
    (False, "Linux", "x86_64", True, "cuda"),
    (False, "Windows", "AMD64", True, "cuda"),
    (False, "Linux", "x86_64", False, "cpu"),
    (False, "Darwin", "arm64", False, "mps"),    # Apple Silicon
    (False, "Darwin", "x86_64", False, "cpu"),   # Mac Intel
    (True, "Darwin", "arm64", False, "cpu"),
    (True, "Windows", "AMD64", True, "cpu"),
])
def test_diar_device(force_cpu, system, machine, nvidia, expected):
    assert hardware.diar_device(force_cpu, system, machine, nvidia) == expected


def test_torch_backend_solo_hace_falta_en_windows_con_nvidia():
    assert hardware.torch_backend("Windows", True) == "auto"
    assert hardware.torch_backend("Windows", False) is None
    assert hardware.torch_backend("Linux", True) is None       # el de PyPI ya trae CUDA
    assert hardware.torch_backend("Darwin", False) is None


GPU = {"name": "RTX", "vram_mb": 16000, "driver": "617.42"}


def test_plan_nvidia_en_linux_y_windows():
    linux = envs.plan("Linux", "x86_64", GPU)
    assert "nvidia-cublas-cu12" in linux["asr"] and "nvidia-cudnn-cu12" in linux["asr"]
    assert linux["torch_backend"] is None
    assert (linux["asr_device"], linux["diar_device"]) == ("cuda", "cuda")

    windows = envs.plan("Windows", "AMD64", GPU)
    assert "nvidia-cublas-cu12" in windows["asr"]
    assert windows["torch_backend"] == "auto"
    assert envs.plan("Windows", "AMD64", GPU, torch_backend="cu128")["torch_backend"] == "cu128"


def test_plan_sin_gpu_o_forzando_cpu_no_instala_cuda():
    for plan in (envs.plan("Linux", "x86_64", None), envs.plan("Windows", "AMD64", GPU, True)):
        assert not any(p.startswith("nvidia-") for p in plan["asr"])
        assert plan["torch_backend"] is None
        assert (plan["asr_device"], plan["diar_device"]) == ("cpu", "cpu")


def test_plan_macos():
    plan = envs.plan("Darwin", "arm64", None)
    assert not any(p.startswith("nvidia-") for p in plan["asr"])
    assert (plan["asr_device"], plan["diar_device"]) == ("cpu", "mps")


def test_plan_siempre_dos_entornos_y_pyannote_4():
    plan = envs.plan("Linux", "x86_64", GPU)
    assert any(p.startswith("faster-whisper") for p in plan["asr"])
    assert "pyannote.audio>=4" in plan["diar"]
    assert not any("pyannote" in p or "torch" in p for p in plan["asr"])
    assert not any("whisper" in p or "nvidia" in p for p in plan["diar"])


def test_batch_size_baja_en_gpus_chicas():
    assert batch_size_for(4.0) == 8             # la GTX de 4 GB
    assert batch_size_for(5.9) == 8
    assert batch_size_for(16.0) is None         # se deja el de pyannote
    assert batch_size_for(None) is None         # CPU o MPS
    assert batch_size_for(4.0, override="2") == 2
    assert batch_size_for(16.0, override="64") == 64
    assert batch_size_for(4.0, override="") == 8


def test_resolve_device_cae_a_cpu_si_torch_no_lo_tiene():
    assert resolve_device("cuda", cuda_ok=True, mps_ok=False) == "cuda"
    assert resolve_device("cuda", cuda_ok=False, mps_ok=False) == "cpu"
    assert resolve_device("mps", cuda_ok=False, mps_ok=True) == "mps"
    assert resolve_device("mps", cuda_ok=False, mps_ok=False) == "cpu"
    assert resolve_device("cpu", cuda_ok=True, mps_ok=True) == "cpu"


def test_gated_check():
    assert gated_check(200)[0] == OK
    assert gated_check(302)[0] == OK            # redirección al archivo: hay acceso
    assert gated_check(401)[0] == MISSING
    state, detail = gated_check(403)
    assert state == MISSING and "pyannote/speaker-diarization-community-1" in detail
    assert gated_check(None)[0] == WARN
    assert gated_check(500)[0] == WARN
