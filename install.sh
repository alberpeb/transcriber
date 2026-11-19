#!/usr/bin/env bash
# Atajo para Linux y macOS: instala uv si falta, los comandos con `uv tool
# install` y los dos entornos con `transcriber setup`.
#
#   ./install.sh            # lo normal
#   ./install.sh --force    # rehace los entornos (las opciones pasan a `setup`)
#
# No necesita sudo. ffmpeg lo instalás vos (dnf, apt o brew).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN_DIR:-$HOME/.local/bin}"

if ! command -v uv >/dev/null; then
  echo "==> Instalando uv (instalador oficial)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Los enlaces que dejaba la v0.1 apuntan al clon: se sacan para que no choquen
# con los comandos que instala uv.
for cmd in transcribe transcribe-speakers; do
  if [[ -L "$BIN/$cmd" ]]; then rm "$BIN/$cmd"; fi
done

echo "==> transcriber, transcribe-speakers, transcribe"
uv tool install --force --reinstall "$REPO"

# record-call sigue siendo bash + PulseAudio/PipeWire: solo Linux.
if [[ "$(uname -s)" == "Linux" ]]; then
  mkdir -p "$BIN"
  ln -sf "$REPO/bin/record-call" "$BIN/record-call"
  echo "==> $BIN/record-call"
fi

# Si ya están los entornos de `uv tool` de la v0.1, `setup` los deja como están.
"$(uv tool dir --bin)/transcriber" setup "$@"
