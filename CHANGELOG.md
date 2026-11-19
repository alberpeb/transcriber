# Changelog

Los cambios importantes de cada versión. El formato sigue
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y las versiones,
[SemVer](https://semver.org/lang/es/).

## [0.2.0] - 2026-11-19

Multiplataforma: Linux, Windows y macOS.

### Agregado

- Paquete instalable con `uv tool install .`. El orquestador usa solo la
  biblioteca estándar; Whisper y pyannote viven en dos entornos aparte.
- `transcriber setup`: crea los dos entornos según el hardware (NVIDIA en Linux y
  Windows, MPS en Apple Silicon, CPU en el resto).
- `transcriber doctor`: revisa ffmpeg, uv, entornos, GPU, token, acceso al modelo
  y `claude`, y dice cómo arreglar lo que falta.
- `install.ps1` para Windows. `install.sh` ahora también sirve en macOS.
- Variables `TRANSCRIBER_CONTEXT`, `TRANSCRIBER_HOME`, `TRANSCRIBER_ASR_PYTHON`,
  `TRANSCRIBER_DIAR_PYTHON` y `TRANSCRIBER_DIAR_BATCH`.
- Se acepta el token guardado por `hf auth login`, además de `HF_TOKEN`.
- Tests con pytest, lint con ruff y CI en Linux, Windows y macOS.

### Cambiado

- El idioma se detecta votando sobre tres ventanas (10%, 50% y 90% del audio) en
  vez de los primeros 30 segundos. Avisa cuando la detección no es segura.
- `transcribe` (solo Whisper) está reescrito en Python sobre el mismo paso ASR del
  pipeline: hereda el idioma votado y el reintento de huecos. Ya no depende de
  `whisper-ctranslate2`, así que **no acepta opciones extra después de `--`**.
- El prompt de la revisión viaja a `claude` por stdin, no como argumento.
- En GPUs de menos de 6 GB el batch de pyannote baja a 8, para evitar los avisos
  de OOM.
- El audio se carga con la biblioteca estándar, sin PyAV ni torchcodec.
- El contexto de ejemplo pasó a `src/transcriber/context.example.md`.
- Los mensajes de `HF_TOKEN` y de ffmpeg dan el comando de cada sistema.

### Arreglado

- Todos los archivos se leen y escriben en UTF-8 (en Windows se rompían las
  tildes y las eñes), y la consola se configura en UTF-8.
- Rutas fijas de Linux (`~/.local/share/uv/tools/...`, `lib/python3.12/...`).
- Instalación nueva rota por la incompatibilidad entre faster-whisper 1.2.1 y las
  versiones recientes de PyAV.

### Compatibilidad

- Si existen los entornos de la v0.1 (`whisper-ctranslate2` y `pyannote-audio` de
  `uv tool`) y todavía no se crearon los nuevos, se siguen usando sin tocar nada.
- `record-call` sigue siendo solo Linux.

## [0.1.0] - 2026-11-19

Versión original: Linux con GPU NVIDIA.

- `transcribe-speakers`: ffmpeg → Whisper → pyannote → revisión con `claude`.
- `transcribe` (solo Whisper) y `record-call`, como scripts de bash.
- `install.sh` con `uv tool install` y enlaces en `~/.local/bin`.

[0.2.0]: https://github.com/alberpeb/transcriber/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/alberpeb/transcriber/releases/tag/v0.1.0
