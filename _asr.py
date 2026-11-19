"""Paso ASR — corre en el entorno de faster-whisper (ctranslate2, CUDA 12).

Además de transcribir:

- Detecta el idioma votando sobre varias ventanas repartidas en el audio. Whisper
  solo mira los primeros 30 s, y un programa que arranca con música salía
  detectado como inglés (y todo traducido).
- Vuelve a decodificar por separado cualquier tramo largo que haya quedado sin
  texto. Whisper a veces se saltea audio bajo o con ruido, y así no se pierde
  habla real (ya pasó: un tramo de 6 minutos salió casi vacío).

Las funciones puras de arriba no importan faster-whisper: se pueden testear sin
el entorno pesado.
"""

import json
import sys
from pathlib import Path

SAMPLE_RATE = 16000
GAP = 40.0                      # segundos sin ninguna palabra que disparan el reintento
WINDOW = 30.0                   # lo que Whisper mira para detectar el idioma
POSITIONS = (0.1, 0.5, 0.9)     # dónde se paran las ventanas, como fracción del audio
LOW_CONFIDENCE = 0.6


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def find_gaps(words, duration, gap=GAP):
    """Tramos (inicio, fin) de más de `gap` segundos sin una sola palabra."""
    gaps, cursor = [], 0.0
    for w in words:
        if w["start"] - cursor > gap:
            gaps.append((cursor, w["start"]))
        cursor = max(cursor, w["end"])
    if duration - cursor > gap:
        gaps.append((cursor, duration))
    return gaps


def language_windows(duration, window=WINDOW, positions=POSITIONS):
    """Inicio, en segundos, de cada ventana donde se detecta el idioma."""
    if duration <= window:
        return [0.0]
    starts = []
    for p in positions:
        start = min(max(duration * p - window / 2, 0.0), duration - window)
        if all(abs(start - s) >= window / 2 for s in starts):
            starts.append(round(start, 2))
    return starts


def vote_language(votes):
    """Elige el idioma sumando las probabilidades de todas las ventanas.

    `votes` es una lista con un dict {idioma: probabilidad} por ventana.
    Devuelve (idioma, confianza, unánime). La confianza es la probabilidad media
    del ganador: si una ventana cayó en música y votó otra cosa, baja.
    """
    votes = [v for v in votes if v]
    if not votes:
        return None, 0.0, False
    totals = {}
    for probs in votes:
        for lang, p in probs.items():
            totals[lang] = totals.get(lang, 0.0) + p
    best = max(totals, key=totals.get)
    winners = {max(v, key=v.get) for v in votes}
    return best, totals[best] / len(votes), len(winners) == 1


def detect_window(model, chunk):
    """Probabilidades de idioma de un tramo de hasta 30 s."""
    if hasattr(model, "detect_language"):            # faster-whisper >= 1.1
        _, _, probs = model.detect_language(audio=chunk)
    else:
        # transcribe() detecta el idioma antes de devolver; los segmentos son un
        # generador que no se consume, así que no decodifica nada.
        _, info = model.transcribe(chunk, vad_filter=False)
        probs = info.all_language_probs or [(info.language, info.language_probability)]
    return dict(probs)


def detect_language(model, audio, duration):
    if not model.model.is_multilingual:
        return "en", 1.0, True, []
    votes, all_probs = [], []
    for start in language_windows(duration):
        a = int(start * SAMPLE_RATE)
        probs = detect_window(model, audio[a:a + int(WINDOW * SAMPLE_RATE)])
        top = max(probs, key=probs.get)
        votes.append({"start": start, "language": top, "probability": round(probs[top], 3)})
        all_probs.append(probs)
    language, confidence, unanimous = vote_language(all_probs)
    return language, confidence, unanimous, votes


def main(argv):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    wav, model_name, language, device, out, prompt_path, duration = argv[:7]
    duration = float(duration)

    from faster_whisper import WhisperModel, decode_audio

    # Sesga el reconocimiento hacia los nombres y el vocabulario del proyecto.
    initial_prompt = None
    if prompt_path != "-" and Path(prompt_path).exists():
        initial_prompt = Path(prompt_path).read_text(encoding="utf-8").strip() or None

    compute = "int8_float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute)

    common = dict(
        vad_filter=False,          # el VAD descartaba habla real en grabaciones del celu
        word_timestamps=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2,
        initial_prompt=initial_prompt,
    )

    # El WAV ya viene en 16 kHz mono: se carga una vez y de ahí salen las ventanas
    # de idioma y los recortes de los huecos.
    audio = decode_audio(wav, sampling_rate=SAMPLE_RATE)

    confidence, votes = 1.0, []
    if not language:
        language, confidence, unanimous, votes = detect_language(model, audio, duration)
        detail = ", ".join(f"{v['language']} {v['probability']:.2f}" for v in votes)
        log(f"      Idioma detectado: {language} (confianza {confidence:.2f}; ventanas: {detail})")
        if confidence < LOW_CONFIDENCE or not unanimous:
            log(f"      Aviso: la detección de idioma no es segura. Si el texto sale "
                f"mal o traducido, pasalo a mano: -l {language}")

    def decode(samples, offset=0.0):
        segments, _ = model.transcribe(samples, language=language, **common)
        return [
            {"start": w.start + offset, "end": w.end + offset, "word": w.word}
            for seg in segments
            for w in (seg.words or [])
        ]

    words = decode(audio)

    # Tramos de más de 40 s sin una sola palabra: los reintenta aislados.
    gaps = find_gaps(words, duration)
    recovered = 0
    for start, end in gaps:
        extra = decode(audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)], offset=start)
        recovered += len(extra)
        words.extend(extra)
    if gaps:
        words.sort(key=lambda w: w["start"])

    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": language,
                "language_probability": round(confidence, 3),
                "language_votes": votes,
                "words": words,
                "gaps_checked": len(gaps),
                "words_recovered": recovered,
            },
            f,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
