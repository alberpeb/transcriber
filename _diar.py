"""Paso diarización — corre en el entorno de pyannote (torch, CUDA 13)."""

import json
import os
import sys

import torch
from pyannote.audio import Pipeline

wav, device, min_spk, max_spk, out = sys.argv[1:6]

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", token=os.environ["HF_TOKEN"]
)
pipeline.to(torch.device(device))

kwargs = {}
if min_spk != "-":
    kwargs["min_speakers"] = int(min_spk)
if max_spk != "-":
    kwargs["max_speakers"] = int(max_spk)

ann = pipeline(wav, **kwargs).speaker_diarization
turns = [
    {"start": seg.start, "end": seg.end, "speaker": label}
    for seg, _, label in ann.itertracks(yield_label=True)
]

with open(out, "w") as f:
    json.dump(turns, f)
