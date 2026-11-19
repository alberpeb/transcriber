#!/usr/bin/env bash
# Instala los dos entornos que usa el pipeline y enlaza los comandos en ~/.local/bin.
#
#   ./install.sh
#
# No necesita sudo. Requiere ffmpeg y uv en el PATH.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN_DIR:-$HOME/.local/bin}"

need() { command -v "$1" >/dev/null || { echo "Falta $1 en el PATH."; exit 1; }; }
need uv
need ffmpeg

# ASR y diarización van en entornos separados a propósito: ctranslate2 trae CUDA 12
# y torch trae CUDA 13; en el mismo venv se pisan las librerías de cuDNN.
echo "==> whisper-ctranslate2 (ASR)"
uv tool install --python 3.12 --force \
  --with nvidia-cublas-cu12 --with nvidia-cudnn-cu12 \
  whisper-ctranslate2

echo "==> pyannote-audio (separación de voces)"
uv tool install --python 3.12 --force --with pydub pyannote.audio

mkdir -p "$BIN"
for cmd in transcribe transcribe-speakers record-call; do
  ln -sf "$REPO/bin/$cmd" "$BIN/$cmd"
  echo "==> $BIN/$cmd"
done

CFG="$HOME/.config/transcriber"
if [[ ! -f "$CFG/context.md" ]]; then
  mkdir -p "$CFG"
  cp "$REPO/context.example.md" "$CFG/context.md"
  echo "==> $CFG/context.md (copiado del ejemplo: editalo)"
fi

cat <<'MSG'

Listo. Falta una sola cosa:

  1. Aceptá las condiciones del modelo en
     https://huggingface.co/pyannote/speaker-diarization-community-1
  2. Creá un token de lectura en https://huggingface.co/settings/tokens
  3. export HF_TOKEN=hf_...   (ponelo en ~/.bashrc)

Después:  transcribe-speakers grabacion.ogg
MSG
