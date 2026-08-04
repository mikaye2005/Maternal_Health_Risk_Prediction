#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_path="$project_root/.venv/bin/python"

if [[ ! -x "$python_path" ]]; then
  echo "MamaCare environment not found. Create .venv and install requirements.txt first." >&2
  exit 1
fi

cd "$project_root"
exec "$python_path" -m streamlit run app/app.py
