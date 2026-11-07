# CUDA Development Container

## Environment

- NVIDIA CUDA 13.4.1 development image
- Ubuntu 24.04
- Python 3.12 development virtual environment at `/opt/venv`
- Separate vLLM virtual environment at `.venv-vllm` in the repository
- GCC/G++
- CMake + Ninja
- Rootless Docker
- namespaced container `root` mapped to the unprivileged host user
- GPU access through NVIDIA Container Toolkit
- default Docker seccomp profile
- `umask 0022`

The exact CUDA image is pinned by digest in the `Dockerfile`.

## Files

```text
containers/cuda/
    - Dockerfile
    - README.md

.devcontainer/cuda/
    - devcontainer.json
```

- `Dockerfile` defines the CUDA development environment
- `devcontainer.json` configures VS Code, GPU access, and persistent caches
- `scripts/setup_python_envs.sh` installs each requirements file into its own virtual environment

## Prerequisites

Host requirements:

- Rootless Docker
- NVIDIA driver
- NVIDIA Container Toolkit configured for Rootless Docker
- VS Code + Dev Containers

## Usage

Open the repository in VS Code and run:

```text
Dev Containers: Reopen in Container
```

After changing the Dockerfile or Dev Container configuration:

```text
Dev Containers: Rebuild Container
```

## Python environments

`postCreateCommand` runs `bash scripts/setup_python_envs.sh`

| Environment | Requirements | Purpose |
| --- | --- | --- |
| `/opt/venv` | `requirements-dev.txt` | Fast vision, tests, and analysis |
| `.venv-vllm` | `requirements-vllm.txt` | vLLM server and VLM tests |

## Verify

From the repository root inside the container:

```bash
whoami
umask
python --version
python -c "import sys; print(sys.executable)"
g++ --version
nvcc --version
nvidia-smi
python -m pip check
.venv-vllm/bin/python -m pip check
```
Expected:

```text
whoami  -> root
umask   -> 0022
sys.executable -> /opt/venv/bin/python
```

Container `root` is namespaced by Rootless Docker and maps to the ordinary host user.

## Security

The development container uses:

- Rootless Docker
- default seccomp
- no `--privileged`
- no extra Linux capabilities by default
- explicit GPU access
- minimal mounts

This is a development environment, not a production inference image.
