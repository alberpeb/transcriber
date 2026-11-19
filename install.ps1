# Atajo para Windows: instala uv si falta, los comandos con `uv tool install`
# y los dos entornos con `transcriber setup`.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 --force
#
# No pide permisos de administrador. ffmpeg lo instalas vos:
#   winget install Gyan.FFmpeg
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "==> Instalando uv (instalador oficial)"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "==> transcriber, transcribe-speakers, transcribe"
uv tool install --force --reinstall $repo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Deja la carpeta de comandos de uv en el PATH del usuario (para terminales nuevas).
uv tool update-shell

$bin = (uv tool dir --bin).Trim()
& (Join-Path $bin "transcriber.exe") setup @args
exit $LASTEXITCODE
