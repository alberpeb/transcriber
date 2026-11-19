"""Las partes puras del paso ASR se importan sin faster-whisper instalado."""

from types import SimpleNamespace

import pytest

from transcriber._asr import (
    detect_language,
    detect_window,
    find_gaps,
    language_windows,
    parse_args,
    vote_language,
)


def w(start, end):
    return {"start": start, "end": end, "word": " x"}


def test_find_gaps_sin_huecos():
    assert find_gaps([w(0, 1), w(30, 31), w(60, 61)], 70) == []


def test_find_gaps_al_principio_en_el_medio_y_al_final():
    words = [w(50, 51), w(52, 53), w(120, 121)]
    assert find_gaps(words, 200) == [(0.0, 50), (53, 120), (121, 200)]


def test_find_gaps_el_umbral_es_estricto():
    assert find_gaps([w(0, 1), w(41, 42)], 82) == []
    assert find_gaps([w(0, 1), w(41.1, 42)], 82.2) == [(1, 41.1), (42, 82.2)]


def test_find_gaps_sin_ninguna_palabra_reintenta_todo():
    assert find_gaps([], 360) == [(0.0, 360)]
    assert find_gaps([], 30) == []


def test_find_gaps_palabras_solapadas_no_hacen_retroceder_el_cursor():
    words = [w(0, 100), w(10, 11), w(141, 142)]
    assert find_gaps(words, 150) == [(100, 141)]


def test_language_windows_audio_corto_usa_una_sola():
    assert language_windows(12) == [0.0]
    assert language_windows(30) == [0.0]


def test_language_windows_reparte_al_10_50_y_90():
    assert language_windows(3600) == [345.0, 1785.0, 3225.0]


def test_language_windows_no_se_salen_ni_se_pisan():
    for duration in (31, 45, 60, 90, 200):
        starts = language_windows(duration)
        assert all(0 <= s <= duration - 30 for s in starts)
        assert all(b - a >= 15 for a, b in zip(starts, starts[1:], strict=False))
    assert language_windows(31) == [0.0]


def test_vote_language_la_musica_del_inicio_no_gana():
    votes = [
        {"en": 0.80, "es": 0.10},           # cortina musical
        {"es": 0.98, "en": 0.01},
        {"es": 0.97, "pt": 0.02},
    ]
    language, confidence, unanimous = vote_language(votes)
    assert language == "es"
    assert confidence == pytest.approx((0.10 + 0.98 + 0.97) / 3)
    assert not unanimous


def test_vote_language_unanime():
    language, confidence, unanimous = vote_language([{"es": 0.99}, {"es": 0.97}])
    assert (language, unanimous) == ("es", True)
    assert confidence == pytest.approx(0.98)


def test_vote_language_sin_votos():
    assert vote_language([]) == (None, 0.0, False)
    assert vote_language([{}, {}]) == (None, 0.0, False)


class FakeModel:
    """Imita la parte de WhisperModel que usa la detección."""

    def __init__(self, answers, multilingual=True):
        self.answers = list(answers)
        self.model = SimpleNamespace(is_multilingual=multilingual)
        self.chunks = []

    def detect_language(self, audio):
        self.chunks.append(len(audio))
        probs = self.answers.pop(0)
        top = max(probs, key=lambda p: p[1])
        return top[0], top[1], probs


def test_detect_language_vota_sobre_las_tres_ventanas():
    model = FakeModel([
        [("en", 0.7), ("es", 0.2)],
        [("es", 0.99), ("en", 0.0)],
        [("es", 0.98), ("en", 0.01)],
    ])
    audio = [0.0] * (16000 * 600)
    language, confidence, unanimous, votes = detect_language(model, audio, 600)
    assert language == "es"
    assert not unanimous
    assert 0.6 < confidence < 0.8
    assert [v["language"] for v in votes] == ["en", "es", "es"]
    assert model.chunks == [16000 * 30] * 3


def test_detect_language_modelo_solo_ingles_no_detecta():
    model = FakeModel([], multilingual=False)
    assert detect_language(model, [0.0] * 16000, 1) == ("en", 1.0, True, [])
    assert model.chunks == []


def test_detect_window_con_faster_whisper_viejo_usa_transcribe():
    class Old:
        def transcribe(self, chunk, vad_filter):
            assert vad_filter is False
            info = SimpleNamespace(language="es", language_probability=0.9,
                                   all_language_probs=[("es", 0.9), ("en", 0.1)])
            return iter(()), info

    assert detect_window(Old(), [0.0]) == {"es": 0.9, "en": 0.1}


def test_parse_args_acepta_rutas_con_espacios_e_idioma_opcional():
    args = parse_args(["C:/a b/audio 1.wav", "C:/a b/out.json", "--duration", "12.5",
                       "--device", "cuda", "--model", "small"])
    assert args.wav == "C:/a b/audio 1.wav"
    assert args.language is None
    assert args.prompt is None
    assert (args.duration, args.device, args.model, args.threads) == (12.5, "cuda", "small", 0)
