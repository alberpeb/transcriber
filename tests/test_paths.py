import os
from pathlib import Path

import pytest

from transcriber import paths

HOME = Path("/home/ana")


def test_data_dir_por_plataforma():
    assert paths.data_dir("Linux", {}, HOME) == HOME / ".local/share/transcriber"
    assert paths.data_dir("Darwin", {}, HOME) == HOME / "Library/Application Support/transcriber"
    env = {"LOCALAPPDATA": r"C:\Users\ana\AppData\Local"}
    assert paths.data_dir("Windows", env, HOME) == Path(env["LOCALAPPDATA"]) / "transcriber"
    assert paths.data_dir("Windows", {}, HOME) == HOME / "AppData/Local/transcriber"


def test_data_dir_se_puede_sobreescribir():
    env = {"TRANSCRIBER_HOME": "/datos/tr", "LOCALAPPDATA": "x"}
    for system in ("Linux", "Darwin", "Windows"):
        assert paths.data_dir(system, env, HOME) == Path("/datos/tr")


def test_context_path_por_plataforma():
    assert paths.context_path("Linux", {}, HOME) == HOME / ".config/transcriber/context.md"
    assert paths.context_path("Darwin", {}, HOME) == (
        HOME / "Library/Application Support/transcriber/context.md")
    env = {"APPDATA": r"C:\Users\ana\AppData\Roaming"}
    assert paths.context_path("Windows", env, HOME) == (
        Path(env["APPDATA"]) / "transcriber" / "context.md")


def test_context_path_linux_no_depende_de_xdg():
    """En la v0.1 era ~/.config fijo: tiene que seguir siendo el mismo archivo."""
    env = {"XDG_CONFIG_HOME": "/otro/lado"}
    assert paths.context_path("Linux", env, HOME) == HOME / ".config/transcriber/context.md"


def test_context_path_variable():
    env = {"TRANSCRIBER_CONTEXT": "/tmp/mi contexto.md", "APPDATA": "x"}
    for system in ("Linux", "Darwin", "Windows"):
        assert paths.context_path(system, env, HOME) == Path("/tmp/mi contexto.md")


def test_venv_python():
    assert paths.venv_python("/e/asr", "Linux") == Path("/e/asr/bin/python")
    assert paths.venv_python("/e/asr", "Darwin") == Path("/e/asr/bin/python")
    assert paths.venv_python("/e/asr", "Windows") == Path("/e/asr/Scripts/python.exe")


def test_env_dir_of_no_resuelve_symlinks():
    assert paths.env_dir_of("/e/asr/bin/python") == Path("/e/asr")
    assert paths.env_dir_of(Path("C:/e/asr/Scripts/python.exe")) == Path("C:/e/asr")


def make_env(root, system, libs=("cublas", "cudnn")):
    """Arma un venv falso con los paquetes nvidia-* como los deja pip."""
    if system == "Windows":
        python, sp, sub = root / "Scripts/python.exe", root / "Lib/site-packages", "bin"
        files = {"cublas": "cublas64_12.dll", "cudnn": "cudnn64_9.dll"}
    else:
        python, sp, sub = root / "bin/python", root / "lib/python3.12/site-packages", "lib"
        files = {"cublas": "libcublas.so.12", "cudnn": "libcudnn.so.9"}
    python.parent.mkdir(parents=True)
    python.write_text("")
    sp.mkdir(parents=True)
    for lib in libs:
        (sp / "nvidia" / lib / sub).mkdir(parents=True)
        (sp / "nvidia" / lib / sub / files[lib]).write_text("")
    return python


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_nvidia_lib_dirs_y_has_cuda_libs(tmp_path, system):
    python = make_env(tmp_path / "asr", system)
    env_dir = paths.env_dir_of(python)
    dirs = paths.nvidia_lib_dirs(env_dir, system)
    assert [d.parent.name for d in dirs] == ["cublas", "cudnn"]
    assert all(d.name == ("bin" if system == "Windows" else "lib") for d in dirs)
    assert paths.has_cuda_libs(env_dir, system)


@pytest.mark.parametrize("system", ["Linux", "Windows", "Darwin"])
def test_entorno_solo_cpu_no_tiene_cuda(tmp_path, system):
    python = make_env(tmp_path / "asr", system, libs=())
    assert paths.nvidia_lib_dirs(paths.env_dir_of(python), system) == []
    assert not paths.has_cuda_libs(paths.env_dir_of(python), system)
    assert not paths.has_cuda_libs(tmp_path / "no-existe", system)


def test_asr_child_env_linux_usa_ld_library_path(tmp_path):
    python = make_env(tmp_path / "asr", "Linux")
    base = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/opt/lib"}
    child = paths.asr_child_env(python, "Linux", base)
    dirs = [str(d) for d in paths.nvidia_lib_dirs(paths.env_dir_of(python), "Linux")]
    assert len(dirs) == 2
    assert child["LD_LIBRARY_PATH"] == ":".join(dirs + ["/opt/lib"])    # las nuestras, primero
    assert child["PATH"] == "/usr/bin"
    assert base["LD_LIBRARY_PATH"] == "/opt/lib"            # no toca el original

    sin_previo = paths.asr_child_env(python, "Linux", {"PATH": "/usr/bin"})
    assert not sin_previo["LD_LIBRARY_PATH"].endswith(":")


def test_asr_child_env_windows_usa_path_y_dll_dirs(tmp_path):
    python = make_env(tmp_path / "asr", "Windows")
    child = paths.asr_child_env(python, "Windows", {"PATH": "lo-que-había"})
    parts = child["PATH"].split(os.pathsep)
    assert parts[-1] == "lo-que-había"
    assert [Path(p).parent.name for p in parts[:-1]] == ["cublas", "cudnn"]
    assert child["TRANSCRIBER_DLL_DIRS"].split(os.pathsep) == parts[:-1]
    assert "LD_LIBRARY_PATH" not in child


def test_asr_child_env_sin_librerias_no_cambia_nada(tmp_path):
    python = make_env(tmp_path / "asr", "Linux", libs=())
    assert paths.asr_child_env(python, "Linux", {"PATH": "/usr/bin"}) == {"PATH": "/usr/bin"}


@pytest.mark.parametrize("system", ["Linux", "Windows", "Darwin"])
def test_find_env_orden_de_busqueda(tmp_path, system):
    home, tools = tmp_path / "home", tmp_path / "uv-tools"
    env = {"TRANSCRIBER_HOME": str(tmp_path / "data")}

    assert paths.find_env("asr", system, env, home, tool_dir=tools) == (None, "falta")

    legacy = make_env(tools / "whisper-ctranslate2", system)
    assert paths.find_env("asr", system, env, home, tool_dir=tools) == (legacy, "uv tool")
    assert paths.find_env("diar", system, env, home, tool_dir=tools) == (None, "falta")

    own = make_env(tmp_path / "data" / "envs" / "asr", system)
    assert paths.find_env("asr", system, env, home, tool_dir=tools) == (own, "setup")

    env["TRANSCRIBER_ASR_PYTHON"] = "/opt/mi python/bin/python"
    assert paths.find_env("asr", system, env, home, tool_dir=tools) == (
        Path("/opt/mi python/bin/python"), "variable")


def test_find_env_fedora_de_la_v01_sigue_andando(tmp_path):
    """Entornos de `uv tool install` en ~/.local/share/uv/tools, sin `setup` nuevo."""
    home = tmp_path / "home"
    tools = home / ".local/share/uv/tools"
    asr = make_env(tools / "whisper-ctranslate2", "Linux")
    diar = make_env(tools / "pyannote-audio", "Linux", libs=())
    assert paths.find_env("asr", "Linux", {}, home, tool_dir=tools) == (asr, "uv tool")
    assert paths.find_env("diar", "Linux", {}, home, tool_dir=tools) == (diar, "uv tool")
    assert paths.has_cuda_libs(paths.env_dir_of(asr), "Linux")
    libs = paths.asr_child_env(asr, "Linux", {})["LD_LIBRARY_PATH"]
    cublas = tools / "whisper-ctranslate2/lib/python3.12/site-packages/nvidia/cublas/lib"
    assert libs.startswith(str(cublas) + ":")


def test_hf_token_variable_o_archivo(tmp_path):
    assert paths.hf_token({}, tmp_path) == (None, "")
    assert paths.hf_token({"HF_TOKEN": "  "}, tmp_path) == (None, "")
    assert paths.hf_token({"HF_TOKEN": " abc \n"}, tmp_path) == ("abc", "variable HF_TOKEN")

    cached = tmp_path / ".cache/huggingface/token"
    cached.parent.mkdir(parents=True)
    cached.write_text("def\n", encoding="utf-8")
    assert paths.hf_token({}, tmp_path) == ("def", str(cached))
    assert paths.hf_token({"HF_TOKEN": "abc"}, tmp_path)[0] == "abc"     # la variable gana


def test_hf_token_file_respeta_hf_home(tmp_path):
    assert paths.hf_token_file({"HF_HOME": "/hf"}, tmp_path) == Path("/hf/token")
    assert paths.hf_token_file({"HF_TOKEN_PATH": "/x/tok", "HF_HOME": "/hf"}, tmp_path) == (
        Path("/x/tok"))
    assert paths.hf_token_file({"XDG_CACHE_HOME": "/c"}, tmp_path) == Path("/c/huggingface/token")


def test_los_workers_viajan_dentro_del_paquete():
    for name in ("_asr.py", "_diar.py", "_refine.py", "context.example.md"):
        assert paths.worker(name).is_file(), name
