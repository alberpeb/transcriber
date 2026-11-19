# transcriber

[![CI](https://github.com/alberpeb/transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/alberpeb/transcriber/actions/workflows/ci.yml)

Pasa una grabación a un transcript legible con los nombres de quién habla.
Funciona en Linux, Windows y macOS.

```bash
transcribe-speakers daily.ogg
```

Deja, al lado del archivo de entrada (o en `-o DIR`):

| Archivo | Qué es |
|---|---|
| `X.revisado.txt` | **El bueno.** Nombres reales, turnos partidos donde corresponde, texto limpio. |
| `X.speakers.txt` | Crudo, con las etiquetas automáticas `SPEAKER_00`. Sirve como evidencia. |
| `X.speakers.srt` | Subtítulos. |
| `X.json` | Bloques con tiempos, para procesar. |

Sirve para reuniones, llamadas, entrevistas y audios largos de WhatsApp: todo
corre en tu máquina, salvo el último paso (los nombres), que usa el CLI de `claude`
y es opcional.

## Los cuatro pasos

1. **Audio** — ffmpeg lo pasa a WAV 16 kHz mono. Entra cualquier formato: mp4, mkv, ogg, opus, mp3.
2. **Transcripción** — Whisper `large-v3-turbo` con [faster-whisper](https://github.com/SYSTRAN/faster-whisper), con marcas de tiempo por palabra.
3. **Separación de voces** — [pyannote](https://github.com/pyannote/pyannote-audio) marca cuándo cambia el que habla.
4. **Revisión** — un modelo de lenguaje (`claude -p`) asigna los nombres reales leyendo el contenido.

Después, cada palabra se asigna a la voz que más se solapa con ella en el tiempo
y las palabras seguidas de la misma voz se juntan en un bloque.

## Instalación

Hace falta `ffmpeg` y [`uv`](https://docs.astral.sh/uv/) (los instaladores lo bajan
si no está). Para que valga la pena, una GPU NVIDIA; sin GPU funciona igual, unas
20 veces más lento. Nada pide permisos de administrador.

### Linux

```bash
sudo dnf install ffmpeg        # o: sudo apt install ffmpeg
git clone https://github.com/alberpeb/transcriber.git
cd transcriber
./install.sh
```

### macOS

```bash
brew install ffmpeg
git clone https://github.com/alberpeb/transcriber.git
cd transcriber
./install.sh
```

### Windows

```powershell
winget install Gyan.FFmpeg
git clone https://github.com/alberpeb/transcriber.git
cd transcriber
powershell -ExecutionPolicy Bypass -File install.ps1
```

Abrí una terminal nueva al terminar, para que tome el `PATH`.

### Qué hacen los instaladores

Son atajos de dos comandos, que también podés correr a mano:

```bash
uv tool install .      # los comandos; no arrastra torch ni nada pesado
transcriber setup      # los dos entornos con Whisper y pyannote
```

`transcriber setup` mira el hardware y decide qué instalar (ver la tabla de
[plataformas](#plataformas)). Los entornos quedan en:

| Sistema | Entornos | Contexto |
|---|---|---|
| Linux | `~/.local/share/transcriber/` | `~/.config/transcriber/context.md` |
| macOS | `~/Library/Application Support/transcriber/` | `~/Library/Application Support/transcriber/context.md` |
| Windows | `%LOCALAPPDATA%\transcriber\` | `%APPDATA%\transcriber\context.md` |

Opciones de `setup`: `--force` rehace los entornos, `--cpu` no instala soporte de
GPU, `--torch-backend cu128` elige a mano el build de torch (por defecto, en
Windows con NVIDIA, lo elige uv según el driver).

### El modelo de voces: `HF_TOKEN`

El modelo de pyannote es *gated*: gratis, pero hay que pedir acceso una vez.

1. Entrá con tu cuenta a
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   y aceptá las condiciones (pide empresa o universidad y para qué lo vas a usar).
2. Creá un token de lectura en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Cargalo como variable de entorno:

   | Sistema | Comando |
   |---|---|
   | Linux | `echo 'export HF_TOKEN=hf_...' >> ~/.bashrc` |
   | macOS | `echo 'export HF_TOKEN=hf_...' >> ~/.zshrc` |
   | Windows | `setx HF_TOKEN hf_...` y abrí una terminal nueva |

   También sirve el token que guarda `hf auth login`. El token no va nunca en un
   archivo del proyecto.

### Revisar que quedó bien

```bash
transcriber doctor
```

Chequea ffmpeg, uv, los dos entornos, la GPU y el dispositivo que va a usar cada
paso, el token, el acceso al modelo y `claude`. Una línea por cosa, con lo que
falta y cómo arreglarlo.

### Actualizar

```bash
git pull
./install.sh           # Windows: powershell -ExecutionPolicy Bypass -File install.ps1
```

No rehace los entornos si ya existen. Si venís de la v0.1 (entornos
`whisper-ctranslate2` y `pyannote-audio` de `uv tool`), se siguen usando tal
cual; `transcriber setup --force` crea los nuevos cuando quieras.

## Uso

```bash
transcribe-speakers reunion.mp4                      # todo por defecto
transcribe-speakers "WhatsApp Ptt 2026-11-19 at 11.58.49 AM.ogg" -l es
transcribe-speakers entrevista.mkv -o salidas --min-speakers 2 --max-speakers 2
transcribe-speakers clase.mp3 --no-refine            # sin el paso 4
```

```
-l es                     idioma (por defecto lo detecta)
-o DIR                    dónde dejar los archivos
-m MODELO                 tiny | base | small | medium | large-v3 | large-v3-turbo
--min-speakers / --max-speakers   cuántas voces esperar
--context ARCHIVO         otro contexto que el de por defecto
--no-refine               saltear la revisión
--cpu                     forzar CPU (~20x más lento)
```

### Los otros comandos

| Comando | Para qué |
|---|---|
| `transcribe` | Solo Whisper, sin separar voces. Más rápido y no necesita token. |
| `transcriber setup` / `doctor` | Instalar los entornos y diagnosticar. |
| `record-call` | Graba una llamada mezclando la salida del sistema con tu micrófono. **Solo Linux.** |

```bash
transcribe clase.mp4                       # texto junto al vídeo
transcribe -l es -f srt entrevista.mkv     # subtítulos en español
transcribe -f all -o salidas *.ogg         # txt, srt, vtt, tsv y json
transcribe -m medium --cpu podcast.mp3
```

## Contexto

El archivo de contexto dice quiénes participan y qué vocabulario se usa. Se usa
dos veces: para sesgar a Whisper hacia los nombres propios y la jerga, y para que
el paso 4 sepa a quién asignarle cada turno. Cuanto mejor esté, mejor salen los
nombres y menos errores de términos. Actualizalo cuando cambie el equipo.

`transcriber setup` copia un [ejemplo](src/transcriber/context.example.md) la
primera vez. Con `--context` usás otro para una corrida, y con la variable
`TRANSCRIBER_CONTEXT` cambiás el de por defecto.

## Por qué está hecho así

Cada una de estas decisiones salió de un bug real. Si algo parece raro, es por esto.

**El VAD está apagado (`vad_filter=False`).** El detector de voz de faster-whisper
tiraba habla real en grabaciones hechas con el celu. Medido sobre el mismo tramo:
119 palabras con VAD, 711 sin VAD. Queda apagado siempre, también en `transcribe`.

**Los huecos se reintentan.** Todo tramo de más de 40 s sin ninguna palabra se
recorta y se vuelve a decodificar aislado. Ya pasó que 6 minutos salieron casi
vacíos. El resumen dice cuántos tramos se reintentaron y cuántas palabras se
recuperaron.

**`condition_on_previous_text=False` y `hallucination_silence_threshold=2`.**
Evitan que Whisper se cuelgue repitiendo la misma frase o invente texto en los
silencios.

**`initial_prompt` con el vocabulario del contexto.** Sesga el reconocimiento
hacia los nombres propios y la jerga, que es justo lo que Whisper escribe mal.

**El idioma se vota sobre tres ventanas.** Whisper detecta el idioma en los
primeros 30 segundos, y un programa en castellano que arranca con música salió
detectado como inglés, todo traducido y roto. Ahora se detecta al 10%, al 50% y
al 90% del audio y se suman las probabilidades. (`language_detection_segments`
de faster-whisper no sirve para esto: toma segmentos consecutivos desde el
principio.) Si la confianza es baja o las ventanas no coinciden, avisa y sugiere `-l`.

**Whisper y pyannote corren en entornos y procesos separados.** ctranslate2
necesita CUDA 12 (`libcublas.so.12`, cuDNN 9) y el torch que trae pyannote viene
con otro CUDA. En el mismo entorno explotan con
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` o `libcudnn.so.9 not found`. Son dos
entornos en las tres plataformas, aunque en alguna no haga falta: un solo diseño
es más fácil de mantener. El orquestador usa solo la biblioteca estándar.

**pyannote va a la GPU a mano** (`pipeline.to(torch.device("cuda"))`). En CPU
anda a ~1x tiempo real (una hora de audio, una hora de espera); en GPU, ~20x.

**El audio se carga sin PyAV ni torchcodec.** Como ffmpeg ya dejó un WAV de
16 kHz mono, los dos pasos lo leen con la biblioteca estándar. faster-whisper
1.2.1 se rompe con las versiones nuevas de PyAV, y torchcodec necesita las
librerías compartidas de FFmpeg, que en Windows y macOS suelen no estar.

**Una palabra nunca queda sin voz.** Si no se solapa con ningún turno, se le
asigna el turno más cercano: no aparecen fragmentos con speaker `?`.

**El paso 4 nunca rompe el pipeline.** Si `claude` no está, venció la sesión
("OAuth session expired") o tarda más de 15 minutos, queda la versión cruda y lo
dice. Además tiene una guarda contra resúmenes: si la revisión trae menos del 65%
de las palabras del crudo, se descarta. Y si había un `.revisado.txt` de una
corrida anterior, avisa que ese archivo no es de esta corrida (no lo borra).

**El prompt va por stdin.** En Windows `claude` puede ser `claude.cmd`, y cmd.exe
corta la línea de comandos en 8191 caracteres: instrucciones, contexto y
transcripción viajan por la entrada estándar.

### Por qué hace falta el paso 4

La separación de voces se basa en el timbre. Cuando la grabación es pobre —el celu
grabando el parlante de la notebook, con la compresión de la llamada encima— los
timbres se parecen demasiado y el modelo mezcla personas: dos bajo una etiqueta, o
una misma persona partida en varias. Forzar la cantidad de voces no alcanza.

Lo que sí funciona es leer el contenido: quién es nombrado, quién reporta qué tema,
quién pregunta y quién responde. Eso hace el paso 4, usando las etiquetas automáticas
solo como pista.

**Grabando desde la compu** (`record-call`, en Linux) el audio es mucho mejor y los
pasos 2 y 3 solos ya dan buen resultado.

## Plataformas

| | Linux + NVIDIA | Windows + NVIDIA | macOS (Apple Silicon) | Sin GPU |
|---|---|---|---|---|
| Whisper | GPU | GPU | CPU | CPU |
| Separación de voces | GPU | GPU | MPS; si falla, CPU | CPU (~1x tiempo real) |
| Revisión con `claude` | sí | sí | sí | sí |
| `transcribe` | GPU | GPU | CPU | CPU |
| `record-call` | sí | no | no | solo Linux |

- **macOS**: ctranslate2 no usa la GPU de Apple, así que Whisper va por CPU
  (`int8`). En una Mac Intel todo va por CPU.
- **Windows**: el torch de PyPI es solo CPU; `setup` lo reemplaza por el build con
  CUDA del índice de PyTorch.
- **`record-call`** es bash + `pactl` + ffmpeg sobre PulseAudio/PipeWire. Portarlo
  a Windows (loopback por WASAPI) y a macOS (necesita BlackHole) queda pendiente.

Qué está probado de verdad: Linux (Fedora, GPU NVIDIA de 4 GB) es el uso diario
desde la v0.1. En Windows 11 con una RTX 5070 Ti se probaron para la v0.2 la
instalación (`setup`, `doctor`), Whisper en GPU, `transcribe` y la revisión con
`claude`. En macOS solo corren los tests unitarios del CI: no se probó en una Mac
real.

### Variables de entorno

| Variable | Para qué |
|---|---|
| `HF_TOKEN` | Token de HuggingFace para el modelo de voces. |
| `TRANSCRIBER_CONTEXT` | Otro archivo de contexto por defecto. |
| `TRANSCRIBER_HOME` | Otra carpeta para los entornos. |
| `TRANSCRIBER_ASR_PYTHON`, `TRANSCRIBER_DIAR_PYTHON` | Usar un intérprete que ya tenés, en vez del entorno de `setup`. |
| `TRANSCRIBER_DIAR_BATCH` | Batch de pyannote, si querés bajarlo o subirlo a mano. |

## Cuando algo falla

Primero, `transcriber doctor`. El pipeline nunca aborta a mitad de camino por la
revisión: si no puede correr, deja la versión cruda y lo dice.

- **"Falta el token de HuggingFace"** o **"No pude bajar el modelo de voces"** —
  falta `HF_TOKEN` o falta aceptar las condiciones del modelo (ver
  [arriba](#el-modelo-de-voces-hf_token)).
- **"Failed to authenticate"** / **"OAuth session expired"** — la sesión del CLI
  venció. Abrí una terminal, corré `claude` una vez, y volvé a lanzar el comando.
- **"la revisión perdió contenido"** — el modelo resumió en vez de transcribir.
  Se descarta y queda la cruda. Volvé a correrlo.
- **Voces mezcladas en el crudo** — es esperable con grabaciones del celu; para eso
  está el paso 4.
- **Sale traducido o en otro idioma** — mirá la línea "Idioma detectado" del log.
  Si avisa que la detección no es segura, pasá el idioma a mano: `-l es`.
- **Faltan tramos** — no debería: los tramos de más de 40 s sin texto se reintentan
  por separado y el resumen dice cuántas palabras se recuperaron.
- **Líneas `CUDACachingAllocator ... OOM` en GPUs chicas** — con menos de 6 GB de
  VRAM el batch de pyannote se baja solo a 8 para que no aparezcan. Si igual
  aparecen y la corrida termina, son inofensivas: torch reintenta con menos
  memoria. Se puede bajar más con `TRANSCRIBER_DIAR_BATCH=4`.
- **Va por CPU teniendo GPU** — `transcriber doctor` lo marca; se arregla con
  `transcriber setup --force`.
- **Aviso de `triton not found` en Windows** — es de torch y no afecta.

## Desarrollo

```bash
uv sync
uv run ruff check .
uv run pytest
```

Los tests cubren lo puro (asignación de voces, tiempos, huecos, votación de idioma,
la guarda de la revisión, rutas por plataforma, elección de dispositivo) y corren
en Linux, Windows y macOS sin instalar Whisper ni bajar modelos. Los cambios están
en el [CHANGELOG](CHANGELOG.md).

## Licencias

- **Este código**: [MIT](LICENSE).
- **Whisper** (los pesos de OpenAI, `large-v3-turbo`): MIT. faster-whisper,
  CTranslate2 y pyannote.audio (las librerías) también son MIT.
- **`pyannote/speaker-diarization-community-1`** (el modelo de voces): la ficha del
  modelo en HuggingFace lo publica como `cc-by-4.0` y dice: *"This pipeline is
  released under the CC-BY-4.0 license and will always remain freely
  accessible."* Es gated: hay que compartir tus datos de contacto para bajarlo.
  CC-BY pide atribución si redistribuís el modelo o algo derivado de él.
- **ffmpeg** no se distribuye con este proyecto; lo instalás aparte con su propia
  licencia.
