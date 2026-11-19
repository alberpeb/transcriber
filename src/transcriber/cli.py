"""Comando `transcriber`: instalación y diagnóstico."""

import argparse

from . import __version__
from .util import utf8_console


def main(argv=None):
    utf8_console()
    p = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcribe grabaciones e identifica quién habla. "
                    "Para transcribir: transcribe-speakers (con voces) o transcribe (solo texto).",
    )
    p.add_argument("--version", action="version", version=f"transcriber {__version__}")
    sub = p.add_subparsers(dest="command", metavar="{setup,doctor}")

    s = sub.add_parser(
        "setup", help="Crea los dos entornos (Whisper y pyannote) según el hardware",
    )
    s.add_argument("--cpu", action="store_true",
                   help="No instalar soporte de GPU aunque haya una NVIDIA")
    s.add_argument("--force", action="store_true",
                   help="Rehacer los entornos aunque ya existan (o existan los de la v0.1)")
    s.add_argument("--torch-backend", metavar="BACKEND",
                   help="Build de torch para las voces: auto, cu128, cu130, cpu... "
                        "(por defecto: auto en Windows con NVIDIA)")

    sub.add_parser("doctor", help="Revisa la instalación y dice qué falta")

    args = p.parse_args(argv)
    if args.command == "setup":
        from . import envs
        return envs.main(args)
    if args.command == "doctor":
        from . import doctor
        return doctor.main(args)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
