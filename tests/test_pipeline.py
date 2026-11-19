import json

from transcriber.pipeline import group, speaker_at, vocabulary, write_outputs
from transcriber.util import hhmmss, srt_time

TURNS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"},
    {"start": 20.0, "end": 25.0, "speaker": "SPEAKER_00"},
]


def test_speaker_at_elige_el_mayor_solapamiento():
    assert speaker_at(TURNS, 1.0, 2.0) == "SPEAKER_00"
    assert speaker_at(TURNS, 4.5, 6.5) == "SPEAKER_01"      # 0.5 s contra 1.5 s
    assert speaker_at(TURNS, 3.0, 5.5) == "SPEAKER_00"      # 2.0 s contra 0.5 s


def test_speaker_at_sin_solapamiento_usa_el_turno_mas_cercano():
    assert speaker_at(TURNS, 10.0, 11.0) == "SPEAKER_01"    # a 1.5 s del fin de 01
    assert speaker_at(TURNS, 18.0, 19.0) == "SPEAKER_00"    # a 1.5 s del turno de 20 s
    assert speaker_at(TURNS, 100.0, 101.0) == "SPEAKER_00"


def test_speaker_at_palabra_de_duracion_cero_no_queda_sin_speaker():
    assert speaker_at(TURNS, 6.0, 6.0) == "SPEAKER_01"


def test_speaker_at_sin_turnos():
    assert speaker_at([], 0.0, 1.0) is None


def test_group_junta_palabras_consecutivas_del_mismo_speaker():
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hola"},
        {"start": 0.5, "end": 1.0, "word": " Ana."},
        {"start": 5.5, "end": 6.0, "word": " ¿Qué"},
        {"start": 6.0, "end": 6.5, "word": " tal?"},
        {"start": 21.0, "end": 21.5, "word": " Bien."},
    ]
    blocks = group(words, TURNS)
    assert [(b["speaker"], b["text"]) for b in blocks] == [
        ("SPEAKER_00", "Hola Ana."),
        ("SPEAKER_01", "¿Qué tal?"),
        ("SPEAKER_00", "Bien."),
    ]
    assert (blocks[0]["start"], blocks[0]["end"]) == (0.0, 1.0)
    assert (blocks[1]["start"], blocks[1]["end"]) == (5.5, 6.5)


def test_group_descarta_bloques_vacios_y_nunca_deja_speaker_sin_asignar():
    words = [
        {"start": 12.0, "end": 12.5, "word": "  "},
        {"start": 13.0, "end": 13.5, "word": " eh"},
    ]
    blocks = group(words, TURNS)
    assert len(blocks) == 1
    assert blocks[0]["speaker"] == "SPEAKER_01"


def test_hhmmss():
    assert hhmmss(0) == "00:00:00"
    assert hhmmss(59.9) == "00:00:59"
    assert hhmmss(3661) == "01:01:01"
    assert hhmmss(36000) == "10:00:00"


def test_srt_time():
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(1.5) == "00:00:01,500"
    assert srt_time(3661.25) == "01:01:01,250"
    assert srt_time(59.9996) == "00:01:00,000"              # redondea, no trunca
    assert srt_time(1.5, ".") == "00:00:01.500"             # WebVTT


def test_vocabulary_saca_nombres_y_terminos():
    context = """# Contexto

## Quiénes participan

- **Ana** — lidera el equipo.
- **Bruno** — back end.

## Vocabulario

Nombres de productos, repos, siglas y jerga que aparecen seguido y que el
reconocimiento suele escribir mal:

staging, rollback, pull request, Ñandú

- "pull rec" → pull request
> una cita, con coma
"""
    assert vocabulary(context).endswith(".")
    terms = vocabulary(context).rstrip(".").split(", ")
    assert terms[:2] == ["Ana", "Bruno"]
    assert {"staging", "rollback", "pull request", "Ñandú"} <= set(terms)
    assert not any("pull rec" in t or "cita" in t for t in terms)


def test_vocabulary_sin_repetidos_con_tope_y_vacio():
    assert vocabulary("") == ""
    assert vocabulary("uno, dos, uno, dos") == "uno, dos."
    many = ", ".join(f"termino{i}" for i in range(200))
    assert len(vocabulary(many).split(", ")) == 60


def test_write_outputs_en_utf8_y_con_espacios_en_el_nombre(tmp_path):
    blocks = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "Buen día, señora Muñoz."},
        {"speaker": None, "start": 3.0, "end": 4.0, "text": "¿Cómo está?"},
    ]
    src = tmp_path / "WhatsApp Ptt 2026-11-19 at 11.58.49 AM.ogg"
    base, payload = write_outputs(blocks, src, tmp_path, "es")

    txt = (tmp_path / "WhatsApp Ptt 2026-11-19 at 11.58.49 AM.speakers.txt").read_text("utf-8")
    assert txt == "[00:00:00] SPEAKER_00: Buen día, señora Muñoz.\n[00:00:03] ?: ¿Cómo está?\n"

    srt = (tmp_path / "WhatsApp Ptt 2026-11-19 at 11.58.49 AM.speakers.srt").read_text("utf-8")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nSPEAKER_00: Buen día")

    data = json.loads((tmp_path / "WhatsApp Ptt 2026-11-19 at 11.58.49 AM.json").read_bytes())
    assert data == payload
    assert data["speakers"] == ["?", "SPEAKER_00"]
    assert data["duration"] == 4.0
    assert data["language"] == "es"
    assert data["blocks"][0]["text"] == "Buen día, señora Muñoz."
