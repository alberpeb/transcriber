# transcriber

Pasa una grabación a un transcript legible con los nombres de quién habla.

```bash
transcribe-speakers daily.ogg
```

Deja, al lado del archivo de entrada:

| Archivo | Qué es |
|---|---|
| `X.revisado.txt` | **El bueno.** Nombres reales, turnos partidos donde corresponde, texto limpio. |
| `X.speakers.txt` | Crudo, con las etiquetas automáticas `SPEAKER_00`. Sirve como evidencia. |
| `X.speakers.srt` | Subtítulos. |
| `X.json` | Bloques con tiempos, para procesar. |

Necesita `HF_TOKEN` en el entorno (modelo de pyannote) y la sesión de `claude`
activa para el último paso.

## Instalación

Hace falta `ffmpeg`, [`uv`](https://docs.astral.sh/uv/) y, para que valga la pena,
una GPU NVIDIA. Sin GPU funciona igual, unas 20 veces más lento.

```bash
git clone https://github.com/<usuario>/transcriber.git ~/Projects/transcriber
cd ~/Projects/transcriber
./install.sh
```

`install.sh` arma los dos entornos, enlaza los comandos en `~/.local/bin` y copia
un contexto de ejemplo. Después hay que habilitar el modelo de voces:

1. Aceptar las condiciones en
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Crear un token de lectura en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. `export HF_TOKEN=hf_...` en el `~/.bashrc`.

El paso 4 (los nombres) usa el CLI de `claude`; si no está disponible, el pipeline
deja igual la versión cruda.

### Los comandos que quedan

| Comando | Para qué |
|---|---|
| `transcribe-speakers` | El pipeline completo, con quién habla. |
| `transcribe` | Solo Whisper, sin separar voces. Más rápido. |
| `record-call` | Graba una llamada mezclando la salida del sistema con tu micrófono. |

## Los cuatro pasos

1. **Audio** — ffmpeg lo pasa a 16 kHz mono. Entra cualquier formato: mp4, mkv, ogg, opus, mp3.
2. **Transcripción** — Whisper `large-v3-turbo` por GPU, ~20x tiempo real.
3. **Separación de voces** — pyannote marca cuándo cambia el que habla.
4. **Revisión** — un modelo de lenguaje asigna los nombres reales leyendo el contenido.

## Por qué hace falta el paso 4

La separación de voces se basa en el timbre. Cuando la grabación es pobre —el celu
grabando el parlante de la notebook, con la compresión de la llamada encima— los
timbres se parecen demasiado y el modelo mezcla personas: dos bajo una etiqueta, o
una misma persona partida en varias. Forzar la cantidad de voces no alcanza.

Lo que sí funciona es leer el contenido: quién es nombrado, quién reporta qué tema,
quién pregunta y quién responde. Eso hace el paso 4, usando las etiquetas automáticas
solo como pista.

**Grabando desde la compu** (`record-call`) el audio es mucho mejor y los pasos 2 y 3
solos ya dan buen resultado.

## Contexto

`~/.config/transcriber/context.md` dice quiénes participan y qué vocabulario se usa.
Cuanto mejor esté, mejor salen los nombres y menos errores de términos. Actualizalo
cuando cambie el equipo.

## Opciones

```
-l en                     idioma (por defecto lo detecta)
-o DIR                    dónde dejar los archivos
-m MODELO                 tiny | base | small | medium | large-v3 | large-v3-turbo
--min-speakers / --max-speakers   cuántas voces esperar
--context ARCHIVO         otro contexto que el de por defecto
--no-refine               saltear la revisión
--cpu                     forzar CPU (~20x más lento)
```

## Cuando algo falla

El pipeline nunca aborta a mitad de camino: si la revisión no puede correr, deja la
versión cruda y lo dice.

- **"Failed to authenticate"** — la sesión del CLI venció. Abrí una terminal, corré
  `claude` una vez, y volvé a lanzar el comando.
- **"la revisión perdió contenido"** — el modelo resumió en vez de transcribir.
  Se descarta y queda la cruda. Volvé a correrlo.
- **Voces mezcladas en el crudo** — es esperable con grabaciones del celu; para eso
  está el paso 4.
- **Sale traducido o en otro idioma** — el idioma se detecta sobre los primeros
  30 segundos, y si ahí hay música o una cortina, se equivoca. Se arregla pasando
  el idioma a mano: `-l es`.
- **Faltan tramos** — no debería: los tramos de más de 40 s sin texto se reintentan
  por separado y el resumen dice cuántas palabras se recuperaron.

## Detalle de implementación

`_asr.py` y `_diar.py` corren en entornos separados a propósito: ctranslate2 usa
CUDA 12 y torch usa CUDA 13, y juntos en un proceso se pisan las librerías de cuDNN.
