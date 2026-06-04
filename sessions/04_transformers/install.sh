#!/usr/bin/env bash
#
# install.sh — set up the Python environment for the ml4fp transformer tutorials.
#
# Creates a local virtual environment in ./.venv and installs every package the
# three tutorial notebooks (tutorial/*.ipynb) and particle_transformer/dataloader.py
# need, plus Jupyter so you can run them.
#
# Usage:
#   ./install.sh                # create .venv and install everything
#   source .venv/bin/activate   # then activate it
#   jupyter lab                 # ...and open the notebooks in tutorial/
#
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo ">> Using interpreter: $("$PYTHON" --version 2>&1) ($(command -v "$PYTHON"))"

if [ ! -d "$VENV_DIR" ]; then
    echo ">> Creating virtual environment in $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ">> Upgrading pip"
python -m pip install --upgrade pip

echo ">> Installing dependencies"
python -m pip install \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    tqdm \
    requests \
    awkward \
    uproot \
    vector \
    jupyter \
    nbconvert \
    ipykernel

echo ">> Verifying imports"
python - <<'PY'
import torch, numpy, scipy, matplotlib, sklearn, tqdm, requests, awkward, uproot, vector
print("All tutorial dependencies import OK")
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
PY

cat <<'EOF'

Done. To use the environment:

    source .venv/bin/activate
    jupyter lab            # open tutorial/*.ipynb

The notebooks expect the dataset at data/JetClass_example_100k.root.
If it is missing, the first notebook downloads it automatically.
EOF
