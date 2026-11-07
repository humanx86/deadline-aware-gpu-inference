#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

/opt/venv/bin/python -m pip install -r requirements-dev.txt
/opt/venv/bin/python -m pip check

if [[ ! -x .venv-vllm/bin/python ]]; then
    /usr/bin/python3.12 -m venv .venv-vllm
fi

.venv-vllm/bin/python -m pip install -r requirements-vllm.txt
.venv-vllm/bin/vllm download-kernels
.venv-vllm/bin/python -m pip check
