import subprocess
from types import SimpleNamespace

import pytest

from transcriber import _refine

RAW = " ".join(f"palabra{i}" for i in range(100))       # 100 palabras
CONTEXT = "- **Ana** — lidera el equipo, con tildes y eñes: señora Muñoz."


@pytest.fixture
def claude(monkeypatch):
    """Simula el CLI: guarda cómo lo llamaron y devuelve lo que se le configure."""
    calls = SimpleNamespace(argv=None, kwargs=None, stdout="", returncode=0, stderr="",
                            raises=None)

    def fake_run(argv, **kwargs):
        calls.argv, calls.kwargs = argv, kwargs
        if calls.raises:
            raise calls.raises
        return SimpleNamespace(returncode=calls.returncode, stdout=calls.stdout,
                               stderr=calls.stderr)

    monkeypatch.setattr(_refine.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(_refine.subprocess, "run", fake_run)
    return calls


def words(n):
    return " ".join(["texto"] * n)


def test_acepta_una_revision_completa(claude):
    claude.stdout = f"\n{words(90)}\n"
    assert _refine.refine(RAW, CONTEXT, "meta") == words(90)


def test_guarda_del_65_por_ciento(claude):
    claude.stdout = words(65)                 # justo en el límite: vale
    assert _refine.refine(RAW, CONTEXT, "meta") == words(65)
    claude.stdout = words(64)                 # resumió: se descarta
    assert _refine.refine(RAW, CONTEXT, "meta") is None


def test_descarta_un_resumen_y_lo_dice(claude, capsys):
    claude.stdout = "Resumen: hablaron de la migración."
    assert _refine.refine(RAW, CONTEXT, "meta") is None
    assert "perdió contenido (5 de 100 palabras)" in capsys.readouterr().err


def test_todo_viaja_por_stdin_y_la_linea_de_comandos_queda_corta(claude):
    claude.stdout = words(100)
    largo = "x" * 20000                       # más que los 8191 caracteres de cmd.exe
    _refine.refine(RAW, CONTEXT + largo, "Archivo: reunión.ogg")
    assert claude.argv == ["/usr/bin/claude", "-p"]
    enviado = claude.kwargs["input"]
    assert _refine.INSTRUCCIONES in enviado
    assert CONTEXT in enviado and largo in enviado
    assert "Archivo: reunión.ogg" in enviado
    assert enviado.rstrip().endswith(RAW)
    assert claude.kwargs["encoding"] == "utf-8"
    assert claude.kwargs["timeout"] == _refine.TIMEOUT == 900
    assert "shell" not in claude.kwargs


def test_sin_claude_en_el_path_no_rompe(monkeypatch):
    monkeypatch.setattr(_refine.shutil, "which", lambda name: None)
    monkeypatch.setattr(_refine.subprocess, "run",
                        lambda *a, **k: pytest.fail("no debería ejecutar nada"))
    assert _refine.refine(RAW, CONTEXT, "meta") is None


def test_sesion_vencida_devuelve_none_y_muestra_el_motivo(claude, capsys):
    claude.returncode, claude.stderr = 1, "aviso\nOAuth session expired\n"
    assert _refine.refine(RAW, CONTEXT, "meta") is None
    assert "OAuth session expired" in capsys.readouterr().err


def test_timeout_devuelve_none(claude):
    claude.raises = subprocess.TimeoutExpired("claude", 900)
    assert _refine.refine(RAW, CONTEXT, "meta") is None


def test_no_poder_ejecutarlo_devuelve_none(claude):
    claude.raises = PermissionError("denegado")
    assert _refine.refine(RAW, CONTEXT, "meta") is None


def test_main_sale_con_2_y_no_toca_el_revisado_anterior(claude, tmp_path):
    raw, ctx, out = tmp_path / "x.speakers.txt", tmp_path / "context.md", tmp_path / "x.rev.txt"
    raw.write_text(RAW, encoding="utf-8")
    out.write_text("de una corrida anterior", encoding="utf-8")
    claude.stdout = "resumen corto"
    assert _refine.main([str(raw), str(ctx), "meta", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == "de una corrida anterior"
    assert "(sin contexto)" in claude.kwargs["input"]


def test_main_escribe_el_revisado_en_utf8(claude, tmp_path):
    raw, ctx, out = tmp_path / "x.speakers.txt", tmp_path / "context.md", tmp_path / "x.rev.txt"
    raw.write_text("[00:00:00] SPEAKER_00: Buen día, señora Muñoz.", encoding="utf-8")
    ctx.write_text(CONTEXT, encoding="utf-8")
    claude.stdout = "[00:00] Ana: Buen día, señora Muñoz."
    assert _refine.main([str(raw), str(ctx), "meta", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "[00:00] Ana: Buen día, señora Muñoz.\n"
    assert "señora Muñoz".encode() in out.read_bytes()
