$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "MamaCare environment not found. Create .venv and install requirements.txt first."
}

Set-Location -LiteralPath $projectRoot
& $pythonPath -m streamlit run "app\app.py"
