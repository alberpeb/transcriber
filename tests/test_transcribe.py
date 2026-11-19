import json
import tomllib
from pathlib import Path

import pytest

import transcriber
from transcriber import cli, pipeline, transcribe

RESULT = {
    "language": "es",
    "segments": [
        {"start": 0.0, "end": 4.38, "text": " Buen día, señora Muñoz."},
        {"start": 4.96, "end": 3661.5, "text": " ¿Cómo\tsalió?"},
        {"start": 3662.0, "end": 3663.0, "text": "   "},
    ],
}


def test_txt():
    assert transcribe.render("txt", RESULT) == "Buen día, señora Muñoz.\n¿Cómo\tsalió?\n"


def test_srt():
    assert transcribe.render("srt", RESULT) == (
        "1\n00:00:00,000 --> 00:00:04,380\nBuen día, señora Muñoz.\n\n"
        "2\n00:00:04,960 --> 01:01:01,500\n¿Cómo\tsalió?\n\n"
    )


def test_vtt():
    assert transcribe.render("vtt", RESULT) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:04.380\nBuen día, señora Muñoz.\n\n"
        "00:00:04.960 --> 01:01:01.500\n¿Cómo\tsalió?\n\n"
    )


def test_tsv_en_milisegundos_y_sin_tabs_en_el_texto():
    assert transcribe.render("tsv", RESULT) == (
        "start\tend\ttext\n"
        "0\t4380\tBuen día, señora Muñoz.\n"
        "4960\t3661500\t¿Cómo salió?\n"
    )


def test_json():
    data = json.loads(transcribe.render("json", RESULT))
    assert data["language"] == "es"
    assert data["text"] == "Buen día, señora Muñoz. ¿Cómo\tsalió?"
    assert data["segments"][1] == {"id": 1, "start": 4.96, "end": 3661.5, "text": "¿Cómo\tsalió?"}


def test_formato_desconocido():
    with pytest.raises(ValueError):
        transcribe.render("docx", RESULT)


def test_write_outputs_all_con_espacios_en_el_nombre(tmp_path):
    src = Path("WhatsApp Ptt 2026-11-19 at 11.58.49 AM.ogg")
    written = transcribe.write_outputs(RESULT, src, tmp_path, "all")
    assert [p.name for p in written] == [
        f"WhatsApp Ptt 2026-11-19 at 11.58.49 AM.{f}" for f in ("txt", "srt", "vtt", "tsv", "json")
    ]
    assert "señora Muñoz" in written[0].read_bytes().decode("utf-8")
    assert [p.name for p in transcribe.write_outputs(RESULT, src, tmp_path, "srt")] == [
        "WhatsApp Ptt 2026-11-19 at 11.58.49 AM.srt"]


@pytest.mark.parametrize("main, options", [
    (pipeline.main, ["-o", "-l", "-m", "--min-speakers", "--max-speakers", "--context",
                     "--no-refine", "--cpu"]),
    (transcribe.main, ["-m", "-l", "--language", "-f", "txt", "srt", "vtt", "tsv", "json",
                       "all", "-o", "--cpu", "--gpu"]),
    (cli.main, ["setup", "doctor", "--version"]),
])
def test_help_lista_todas_las_opciones(main, options, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for option in options:
        assert option in out, option


def test_setup_help(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["setup", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--force" in out and "--cpu" in out and "--torch-backend" in out


def test_archivo_inexistente_termina_con_mensaje(tmp_path):
    with pytest.raises(SystemExit) as e:
        pipeline.main([str(tmp_path / "no existe.ogg")])
    assert "No existe" in str(e.value.code)


def test_run_asr_arma_los_argumentos_sin_shell(monkeypatch, tmp_path):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    python = tmp_path / "mi env" / "bin" / "python"
    wav = tmp_path / "con espacios" / "a.wav"
    pipeline.run_asr(python, wav, tmp_path / "o.json", "large-v3-turbo", "cpu", 12.5,
                     language="es", prompt_file=tmp_path / "p.txt")
    argv = seen["argv"]
    assert argv[0] == str(python)
    assert argv[1].endswith("_asr.py")
    assert argv[2] == str(wav)                               # un solo argumento, con espacios
    assert argv[argv.index("--language") + 1] == "es"
    assert argv[argv.index("--duration") + 1] == "12.5"
    assert "shell" not in seen["kwargs"]

    pipeline.run_asr(python, wav, tmp_path / "o.json", "small", "cpu", 1.0)
    assert "--language" not in seen["argv"] and "--prompt" not in seen["argv"]


def test_un_paso_que_falla_corta_con_mensaje(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.subprocess, "run",
                        lambda argv, **k: type("P", (), {"returncode": 3})())
    with pytest.raises(SystemExit) as e:
        pipeline.run_diar(tmp_path / "python", tmp_path / "a.wav", "cpu", None, None, "o.json")
    assert "transcriber doctor" in str(e.value.code)
    assert pipeline.run_worker("python", "_asr.py", [], "x", check=False) == 3


def test_la_version_coincide_con_pyproject():
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["version"] == transcriber.__version__
    assert data["project"]["dependencies"] == []            # el orquestador es solo stdlib
    assert set(data["project"]["scripts"]) == {"transcriber", "transcribe-speakers", "transcribe"}
