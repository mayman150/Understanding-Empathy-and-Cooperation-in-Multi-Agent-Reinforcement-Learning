#!/bin/bash
# One-time environment setup on a Compute Canada / Alliance LOGIN node (compute nodes on most
# clusters have no internet, so the virtualenv must be built here and reused by the jobs).
#
#   VENV=~/envs/empathy scripts/slurm/setup_env.sh                 # PD / debug env only
#   WITH_MELTINGPOT=1 VENV=~/envs/empathy scripts/slurm/setup_env.sh   # + dm-meltingpot (Linux x86_64, needs ~2 GB)
#
# Packages available in the Alliance wheelhouse are installed with --no-index (fast, cluster-optimised
# builds); the rest comes from PyPI.  Adjust PY_MODULE to a version present on your cluster
# (`module avail python`).
set -euo pipefail

VENV=${VENV:-$HOME/envs/empathy}
PY_MODULE=${PY_MODULE:-python/3.11}
STDENV=${STDENV:-StdEnv/2023}
WITH_MELTINGPOT=${WITH_MELTINGPOT:-0}

if command -v module >/dev/null 2>&1; then module load "$STDENV" "$PY_MODULE"; fi
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# from the Alliance wheelhouse when available (torch picks the cluster's CUDA build automatically)
pip install --no-index torch numpy tensorboard || pip install torch numpy tensorboard==2.18.0

# from PyPI
pip install gymnasium==0.29.1 pettingzoo==1.24.3 supersuit==3.9.3 tyro==0.9.2 pytest==8.3.4 imageio==2.36.1
if [ "$WITH_MELTINGPOT" = "1" ]; then
  pip install "shimmy[meltingpot]==1.3.0"
fi

python - <<'EOF'
import torch, gymnasium, pettingzoo, supersuit, tyro
print("torch", torch.__version__, "cuda build:", torch.version.cuda, "| gymnasium", gymnasium.__version__,
      "| pettingzoo", pettingzoo.__version__, "| supersuit", supersuit.__version__)
EOF
echo "venv ready: $VENV   (jobs use it via VENV=$VENV, see scripts/slurm/run_grid.sh)"
