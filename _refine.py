"""Revisión final: reasigna los speakers por contenido y limpia el texto.

La separación automática de voces se equivoca cuando la grabación es pobre (todas
las voces salen por el mismo parlante y pierden el timbre). Un modelo de lenguaje
sí puede resolverlo leyendo el contenido: quién es nombrado, quién reporta qué.
Este paso llama al CLI de Claude; si no está disponible, el pipeline sigue sin él.
"""

import shutil
import subprocess
import sys
from pathlib import Path

TIMEOUT = 900

INSTRUCCIONES = """\
Sos un editor de transcripciones. Recibís una transcripción con las voces separadas
automáticamente (SPEAKER_00, SPEAKER_01...). Esa separación SE EQUIVOCA: mezcla dos
personas bajo una etiqueta, parte a una misma persona en varias, y corta turnos en
medio de una frase.

Tu trabajo es reescribirla asignando los nombres reales, usando el contenido como
evidencia: a quién nombran, quién reporta qué tema, quién pregunta y quién responde.
Las etiquetas SPEAKER_XX son solo una pista; si el contenido las contradice, ganan
los nombres.

Reglas, en orden de importancia:

1. NO resumas, no omitas ni acortes turnos. Va el diálogo completo, de principio a fin.
2. Si un bloque contiene a dos personas, partilo en los turnos que correspondan.
3. Si no podés determinar quién habla, poné el nombre más probable seguido de " (?)".
   Si no tenés ni idea, poné "(?)". Nunca inventes una certeza que no tenés.
4. Corregí los errores obvios del reconocimiento usando el vocabulario del contexto.
   No cambies nada más: no reformules ideas ni agregues cosas que no se dijeron.
5. Sacá muletillas y repeticiones de tartamudeo ("um", "uh", "the the") para que se lea
   bien. El contenido tiene que quedar intacto.
6. Mantené las marcas de tiempo que ya vienen, al inicio del turno. Para un turno que
   sale de partir un bloque, usá "[~mm:ss]" estimando, o no pongas marca.

Formato exacto de salida, sin nada antes ni después:

TÍTULO (fecha y duración si se deducen del contexto)

Participantes: lista de los nombres que aparecen.
Nota: una línea diciendo que los speakers fueron asignados por contenido y que (?)
marca lo dudoso.

────────────────────────────────────────────────────────────

[mm:ss] Nombre: texto del turno.

[mm:ss] Otro: texto del turno.

Devolvé solo eso, sin bloques de código ni comentarios tuyos.
"""


def build_prompt(raw_text: str, context: str, meta: str) -> str:
    return (
        f"{INSTRUCCIONES}\n\n## Contexto\n\n{context}\n\n## Datos\n\n{meta}"
        f"\n\n## Transcripción a revisar\n\n{raw_text}"
    )


def refine(raw_text: str, context: str, meta: str) -> str | None:
    # En Windows `claude` puede ser claude.cmd: hay que resolver la ruta completa,
    # y cmd.exe corta la línea de comandos en 8191 caracteres. Por eso todo
    # (instrucciones, contexto y transcripción) viaja por stdin.
    exe = shutil.which("claude")
    if exe is None:
        print("refine: no encuentro 'claude' en el PATH", file=sys.stderr)
        return None
    try:
        proc = subprocess.run(
            [exe, "-p"],
            input=build_prompt(raw_text, context, meta),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT,
        )
    except OSError as e:
        print(f"refine: no pude ejecutar claude ({e})", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("refine: el modelo tardó demasiado, se deja la versión cruda", file=sys.stderr)
        return None

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        print(f"refine: {detail[-1] if detail else 'falló'}", file=sys.stderr)
        return None

    text = proc.stdout.strip()

    # Control de que no haya resumido ni recortado: sacar muletillas quita
    # algo de texto, pero perder más de un tercio significa contenido faltante.
    raw_words, new_words = len(raw_text.split()), len(text.split())
    if new_words < 0.65 * raw_words:
        print(
            f"refine: la revisión perdió contenido ({new_words} de {raw_words} palabras), "
            "se deja la versión cruda",
            file=sys.stderr,
        )
        return None

    return text


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    raw_path, ctx_path, meta, out_path = sys.argv[1:5]
    context = Path(ctx_path).read_text(encoding="utf-8") if Path(ctx_path).exists() else "(sin contexto)"
    result = refine(Path(raw_path).read_text(encoding="utf-8"), context, meta)
    if result is None:
        sys.exit(2)
    Path(out_path).write_text(result + "\n", encoding="utf-8")
