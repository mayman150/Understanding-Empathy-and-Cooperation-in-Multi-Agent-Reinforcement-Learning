#!/bin/bash
# One-time environment setup on a Compute Canada / Alliance LOGIN node (compute nodes on most
# clusters have no internet, so the virtualenv must be built here and reused by the jobs).
#
#   scripts/slurm/setup_env.sh                       # PD / debug env only
#   WITH_MELTINGPOT=1 scripts/slurm/setup_env.sh     # + dm-meltingpot (Linux x86_64, ~2 GB)
#
# VENV / PY_MODULE / STDENV come from scripts/slurm/cluster.env (default venv lives in project
# space, ~/projects/aip-machado/$USER/envs/empathy).  Packages available in the Alliance
# wheelhouse are installed with --no-index (cluster-optimised builds); the rest comes from PyPI.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/cluster.env"
WITH_MELTINGPOT=${WITH_MELTINGPOT:-0}

if command -v module >/dev/null 2>&1; then module load "$STDENV" "$PY_MODULE"; fi

# supersuit depends on `tinyscaler`, a C extension built from source. Use the StdEnv gcc for it,
# regardless of CC/CXX exported in the shell (e.g. CC=clang from other projects' setups).
export CC=gcc CXX=g++

mkdir -p "$(dirname "$VENV")"
python -m venv "$VENV"   # idempotent: re-running the script resumes an interrupted install
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1   # ignore ~/.local packages while installing / verifying
pip install --no-index --upgrade pip

# from the Alliance wheelhouse when available (torch picks the cluster's CUDA build automatically)
pip install --no-index torch numpy tensorboard || pip install torch numpy tensorboard==2.18.0

# from PyPI
pip install gymnasium==0.29.1 pettingzoo==1.24.3 supersuit==3.9.3 tyro==0.9.2 pytest==8.3.4 imageio==2.36.1 matplotlib
if [ "$WITH_MELTINGPOT" = "1" ]; then
  pip install "shimmy[meltingpot]==1.3.0"
fi

python - <<'EOF'
import torch, gymnasium, pettingzoo, supersuit, tyro
print("torch", torch.__version__, "cuda build:", torch.version.cuda, "| gymnasium", gymnasium.__version__,
      "| pettingzoo", pettingzoo.__version__, "| supersuit", supersuit.__version__)
EOF
echo "venv ready: $VENV"
echo "next:  source $VENV/bin/activate && cd $PROJECT_DIR && python -m pytest tests -q   (use python -m pytest, not a ~/.local pytest)"
