# transcriber

[![CI](https://github.com/alberpeb/transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/alberpeb/transcriber/actions/workflows/ci.yml)

*[Leer en castellano](README.es.md)*

Turns a recording into a readable transcript with the names of who is speaking.
Works on Linux, Windows and macOS.

```bash
transcribe-speakers daily.ogg
```

It leaves, next to the input file (or in `-o DIR`):

| File | What it is |
|---|---|
| `X.revisado.txt` | **The good one.** Real names, turns split where they should be, clean text. |
| `X.speakers.txt` | Raw, with the automatic `SPEAKER_00` labels. Useful as evidence. |
| `X.speakers.srt` | Subtitles. |
| `X.json` | Blocks with timestamps, for further processing. |

Good for meetings, calls, interviews and long WhatsApp voice notes: everything
runs on your machine except the last step (the names), which uses the `claude`
CLI and is optional.

The tool's own messages and the review prompt are in Spanish. Audio in any
language Whisper supports works.

## The four steps

1. **Audio** — ffmpeg converts it to 16 kHz mono WAV. Any format goes in: mp4, mkv, ogg, opus, mp3.
2. **Transcription** — Whisper `large-v3-turbo` through [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with word-level timestamps.
3. **Speaker diarization** — [pyannote](https://github.com/pyannote/pyannote-audio) marks when the speaker changes.
4. **Review** — a language model (`claude -p`) assigns the real names by reading the content.

After that, each word goes to the voice that overlaps it the most in time, and
consecutive words from the same voice are joined into one block.

## Installation

You need `ffmpeg` and [`uv`](https://docs.astral.sh/uv/) (the installers download
uv if it is missing). An NVIDIA GPU makes it worthwhile; without one it still
works, about 20 times slower. Nothing asks for administrator rights.

### Linux

```bash
sudo dnf install ffmpeg        # or: sudo apt install ffmpeg
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

Open a new terminal when it finishes, so it picks up the `PATH`.

### What the installers do

They are shortcuts for two commands that you can also run by hand:

```bash
uv tool install .      # the commands; does not pull torch or anything heavy
transcriber setup      # the two environments with Whisper and pyannote
```

`transcriber setup` looks at the hardware and decides what to install (see the
[platforms](#platforms) table). The environments live in:

| System | Environments | Context |
|---|---|---|
| Linux | `~/.local/share/transcriber/` | `~/.config/transcriber/context.md` |
| macOS | `~/Library/Application Support/transcriber/` | `~/Library/Application Support/transcriber/context.md` |
| Windows | `%LOCALAPPDATA%\transcriber\` | `%APPDATA%\transcriber\context.md` |

`setup` options: `--force` rebuilds the environments, `--cpu` skips GPU support,
`--torch-backend cu128` picks the torch build by hand (by default, on Windows
with NVIDIA, uv picks it from the driver).

### The diarization model: `HF_TOKEN`

The pyannote model is *gated*: free, but you have to request access once.

1. Sign in and open
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
   then accept the conditions (it asks for your company or university and what
   you will use it for).
2. Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Set it as an environment variable:

   | System | Command |
   |---|---|
   | Linux | `echo 'export HF_TOKEN=hf_...' >> ~/.bashrc` |
   | macOS | `echo 'export HF_TOKEN=hf_...' >> ~/.zshrc` |
   | Windows | `setx HF_TOKEN hf_...` and open a new terminal |

   The token saved by `hf auth login` works too. The token never goes in a
   project file.

### Check that it works

```bash
transcriber doctor
```

It checks ffmpeg, uv, both environments, the GPU and the device each step will
use, the token, access to the model and `claude`. One line per item, with what
is missing and how to fix it.

### Updating

```bash
git pull
./install.sh           # Windows: powershell -ExecutionPolicy Bypass -File install.ps1
```

It does not rebuild the environments if they already exist. If you come from
v0.1 (the `whisper-ctranslate2` and `pyannote-audio` environments from
`uv tool`), they keep being used as they are; `transcriber setup --force`
creates the new ones whenever you want.

## Usage

```bash
transcribe-speakers meeting.mp4                      # all defaults
transcribe-speakers "WhatsApp Ptt 2026-11-19 at 11.58.49 AM.ogg" -l es
transcribe-speakers interview.mkv -o out --min-speakers 2 --max-speakers 2
transcribe-speakers lecture.mp3 --no-refine          # without step 4
```

```
-l es                     language (detected by default)
-o DIR                    where to write the files
-m MODEL                  tiny | base | small | medium | large-v3 | large-v3-turbo
--min-speakers / --max-speakers   how many voices to expect
--context FILE            a context other than the default one
--no-refine               skip the review
--cpu                     force CPU (~20x slower)
```

### The other commands

| Command | What for |
|---|---|
| `transcribe` | Whisper only, no diarization. Faster and needs no token. |
| `transcriber setup` / `doctor` | Install the environments and diagnose. |
| `record-call` | Records a call by mixing the system output with your microphone. **Linux only.** |

```bash
transcribe lecture.mp4                     # text next to the video
transcribe -l es -f srt interview.mkv      # Spanish subtitles
transcribe -f all -o out *.ogg             # txt, srt, vtt, tsv and json
transcribe -m medium --cpu podcast.mp3
```

## Context

The context file says who takes part and what vocabulary is used. It is used
twice: to bias Whisper towards proper names and jargon, and so that step 4 knows
who to assign each turn to. The better it is, the better the names come out and
the fewer term errors you get. Update it when the team changes.

`transcriber setup` copies an [example](src/transcriber/context.example.md) the
first time. `--context` uses another one for a single run, and the
`TRANSCRIBER_CONTEXT` variable changes the default.

## Why it is built this way

Each of these decisions came from a real bug. If something looks odd, this is why.

**VAD is off (`vad_filter=False`).** faster-whisper's voice detector was dropping
real speech in recordings made with a phone. Measured on the same stretch:
119 words with VAD, 711 without. It stays off always, in `transcribe` too.

**Gaps are retried.** Any stretch longer than 40 s without a single word is cut
out and decoded again on its own. It has happened that 6 minutes came out almost
empty. The summary says how many stretches were retried and how many words were
recovered.

**`condition_on_previous_text=False` and `hallucination_silence_threshold=2`.**
They keep Whisper from getting stuck repeating the same sentence or making up
text during silences.

**`initial_prompt` with the vocabulary from the context.** It biases recognition
towards proper names and jargon, which is exactly what Whisper gets wrong.

**The language is voted over three windows.** Whisper detects the language in
the first 30 seconds, and a Spanish show that starts with music was detected as
English, all translated and broken. Now it is detected at 10%, 50% and 90% of
the audio and the probabilities are added up. (faster-whisper's
`language_detection_segments` does not help here: it takes consecutive segments
from the start.) If confidence is low or the windows disagree, it warns and
suggests `-l`.

**Whisper and pyannote run in separate environments and processes.** ctranslate2
needs CUDA 12 (`libcublas.so.12`, cuDNN 9) and the torch that pyannote brings
ships with a different CUDA. In the same environment they blow up with
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` or `libcudnn.so.9 not found`. There
are two environments on all three platforms, even where one would do: a single
design is easier to maintain. The orchestrator uses only the standard library.

**pyannote is moved to the GPU by hand** (`pipeline.to(torch.device("cuda"))`).
On CPU it runs at ~1x real time (an hour of audio, an hour of waiting); on GPU,
~20x.

**The audio is loaded without PyAV or torchcodec.** Since ffmpeg already left a
16 kHz mono WAV, both steps read it with the standard library. faster-whisper
1.2.1 breaks with recent PyAV versions, and torchcodec needs FFmpeg's shared
libraries, which are usually missing on Windows and macOS.

**A word is never left without a voice.** If it overlaps no turn, it gets the
nearest one: no fragments show up with speaker `?`.

**Step 4 never breaks the pipeline.** If `claude` is missing, the session expired
("OAuth session expired") or it takes more than 15 minutes, the raw version
stays and the tool says so. It also has a guard against summaries: if the review
comes back with less than 65% of the words of the raw text, it is discarded. And
if there was a `.revisado.txt` from an earlier run, it warns that the file is
not from this run (it does not delete it).

**The prompt goes through stdin.** On Windows `claude` can be `claude.cmd`, and
cmd.exe cuts the command line at 8191 characters: instructions, context and
transcript travel through standard input.

### Why step 4 is needed

Diarization relies on timbre. When the recording is poor (a phone recording a
laptop speaker, with call compression on top) the timbres are too similar and
the model mixes people up: two under one label, or one person split into
several. Forcing the number of voices is not enough.

What does work is reading the content: who gets named, who reports on which
topic, who asks and who answers. That is what step 4 does, using the automatic
labels only as a hint.

**Recording from the computer** (`record-call`, on Linux) gives much better
audio, and steps 2 and 3 alone already give a good result.

## Platforms

| | Linux + NVIDIA | Windows + NVIDIA | macOS (Apple Silicon) | No GPU |
|---|---|---|---|---|
| Whisper | GPU | GPU | CPU | CPU |
| Diarization | GPU | GPU | MPS; CPU if it fails | CPU (~1x real time) |
| Review with `claude` | yes | yes | yes | yes |
| `transcribe` | GPU | GPU | CPU | CPU |
| `record-call` | yes | no | no | Linux only |

- **macOS**: ctranslate2 does not use the Apple GPU, so Whisper runs on CPU
  (`int8`). On an Intel Mac everything runs on CPU.
- **Windows**: the torch on PyPI is CPU only; `setup` replaces it with the CUDA
  build from the PyTorch index.
- **`record-call`** is bash + `pactl` + ffmpeg on PulseAudio/PipeWire. Porting it
  to Windows (WASAPI loopback) and macOS (needs BlackHole) is pending:
  [issue #1](https://github.com/alberpeb/transcriber/issues/1).

What has really been tested: Linux (Fedora, 4 GB NVIDIA GPU) has been the daily
driver since v0.1. On Windows 11 with an RTX 5070 Ti the whole path was tested
for v0.2: installation (`setup`, `doctor`), `transcribe` and the four steps of
`transcribe-speakers` on GPU with a short two-voice recording, which came out
correctly separated. On macOS only the CI unit tests run: it was not tested on a
real Mac.

### Environment variables

| Variable | What for |
|---|---|
| `HF_TOKEN` | HuggingFace token for the diarization model. |
| `TRANSCRIBER_CONTEXT` | A different default context file. |
| `TRANSCRIBER_HOME` | A different folder for the environments. |
| `TRANSCRIBER_ASR_PYTHON`, `TRANSCRIBER_DIAR_PYTHON` | Use an interpreter you already have instead of the `setup` environment. |
| `TRANSCRIBER_DIAR_BATCH` | pyannote batch size, if you want to lower or raise it by hand. |

## When something fails

First, `transcriber doctor`. The pipeline never aborts halfway because of the
review: if it cannot run, it leaves the raw version and says so.

The messages below are quoted as the tool prints them, in Spanish.

- **"Falta el token de HuggingFace"** or **"No pude bajar el modelo de voces"** —
  `HF_TOKEN` is missing, or the model conditions have not been accepted (see
  [above](#the-diarization-model-hf_token)).
- **"Failed to authenticate"** / **"OAuth session expired"** — the CLI session
  expired. Open a terminal, run `claude` once, and launch the command again.
- **"la revisión perdió contenido"** — the model summarized instead of
  transcribing. The review is discarded and the raw version stays. Run it again.
- **Mixed-up voices in the raw file** — expected with phone recordings; that is
  what step 4 is for.
- **It comes out translated or in another language** — look at the "Idioma
  detectado" line in the log. If it warns that detection is not reliable, pass
  the language by hand: `-l es`.
- **Stretches are missing** — they should not be: stretches longer than 40 s
  with no text are retried separately, and the summary says how many words were
  recovered.
- **`CUDACachingAllocator ... OOM` lines on small GPUs** — with less than 6 GB of
  VRAM the pyannote batch is lowered to 8 automatically (if it was higher). If
  they still show up and the run finishes, they are harmless: torch retries with
  less memory. You can lower it further with `TRANSCRIBER_DIAR_BATCH=4`.
- **It runs on CPU although there is a GPU** — `transcriber doctor` flags it; fix
  it with `transcriber setup --force`.
- **`triton not found` warning on Windows** — it comes from torch and has no effect.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest
```

The tests cover the pure parts (speaker assignment, timestamps, gaps, language
voting, the review guard, per-platform paths, device selection) and run on
Linux, Windows and macOS without installing Whisper or downloading models.
Changes are listed in the [CHANGELOG](CHANGELOG.md) (in Spanish).

## Licenses

- **This code**: [MIT](LICENSE).
- **Whisper** (OpenAI's weights, `large-v3-turbo`): MIT. faster-whisper,
  CTranslate2 and pyannote.audio (the libraries) are MIT too.
- **`pyannote/speaker-diarization-community-1`** (the diarization model): its
  HuggingFace model card publishes it as `cc-by-4.0` and says: *"This pipeline is
  released under the CC-BY-4.0 license and will always remain freely
  accessible."* It is gated: you have to share your contact details to download
  it. CC-BY requires attribution if you redistribute the model or something
  derived from it.
- **ffmpeg** is not distributed with this project; you install it separately
  under its own license.
