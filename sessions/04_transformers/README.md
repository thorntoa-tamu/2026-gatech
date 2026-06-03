# ML4FP 2026 — Transformers for Particle Physics

Hands-on tutorial material: building a transformer jet classifier from scratch,
the cost of attention (and Linformer), and the Particle Transformer with real
attention interpretability.

Three notebooks in [`tutorial/`](tutorial/):

| Notebook | Topic |
|---|---|
| [`01_transformers_and_self_attention.ipynb`](tutorial/01_transformers_and_self_attention.ipynb) | Self-attention from scratch; a tiny binary jet classifier |
| [`02_complexity_and_linformer.ipynb`](tutorial/02_complexity_and_linformer.ipynb) | Why attention is `O(n²)`, and Linformer's linear-cost trick on the same small binary task |
| [`03_particle_transformer.ipynb`](tutorial/03_particle_transformer.ipynb) | Physics-informed attention; inspecting a pre-trained ParT model |

---

## Setup

### Prerequisites

- **Python 3.11–3.14.** The install pins `torch>=2.10`, which ships wheels for
  these versions. 3.11–3.13 are the most widely tested; 3.14 also works.
- **macOS or Linux.** Runs on CPU — no GPU required (each notebook finishes in
  well under a minute on a laptop).
- **~3 GB free disk** (PyTorch + the example dataset) and a network connection
  for the first install and the dataset download.

### Quick start

From the repository root:

```bash
./install.sh                 # creates ./.venv and installs everything
source .venv/bin/activate    # activate the environment

# if on laptop
#jupyter lab                  # open the notebooks in tutorial/ 
```

`install.sh` creates a virtual environment in `./.venv`, installs every package
the notebooks need, and prints `All tutorial dependencies import OK` when it
succeeds.

To use a specific interpreter or a different environment location:

```bash
PYTHON=python3.12 ./install.sh        # pick the interpreter
VENV_DIR=.myenv   ./install.sh        # pick where the venv lives
```

### Manual install (alternative)

If you prefer to manage the environment yourself:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.10" numpy scipy matplotlib scikit-learn \
            tqdm requests awkward uproot vector \
            jupyter nbconvert ipykernel
```

- `torch`, `numpy`, `matplotlib`, `scikit-learn`, `tqdm`, `requests` — the core
  modelling and plotting stack.
- `uproot`, `awkward`, `vector` — read the JetClass `.root` data files.
- `jupyter`, `nbconvert`, `ipykernel` — run the notebooks.

---

## The dataset

The notebooks use `data/JetClass_example_100k.root` (~130 MB, 100k jets, 10
classes). If it is already in `data/`, nothing is downloaded. If it is missing,
the tutorial helper used by all three notebooks fetches it automatically from
CERN:

```
https://hqu.web.cern.ch/datasets/JetClass/example/JetClass_example_100k.root
```

To pre-download it manually:

```bash
mkdir -p data
curl -L -o data/JetClass_example_100k.root \
  https://hqu.web.cern.ch/datasets/JetClass/example/JetClass_example_100k.root
```

---

## Running the notebooks

After `source .venv/bin/activate`:

```bash
jupyter lab          # or: jupyter notebook
```

then open `tutorial/01_...`, `02_...`, `03_...` in order and run all cells.

To execute a notebook headless (e.g. to check it runs end-to-end):

```bash
cd tutorial
../.venv/bin/jupyter nbconvert --to notebook --execute \
  --output 01_executed.ipynb 01_transformers_and_self_attention.ipynb
```

### Notebook 3 specifics

Notebook 3 runs the **pre-trained Particle Transformer** without needing the
`weaver` framework. Everything it needs lives in `tutorial/`:

- [`tutorial/dataloader.py`](tutorial/dataloader.py) — lightweight JetClass
  loader used by all three notebooks.
- [`tutorial/tutorial_data.py`](tutorial/tutorial_data.py) — shared dataset
  download + the small binary-task preprocessing used in Notebooks 1 and 2.
- [`tutorial/part_model.py`](tutorial/part_model.py) — the ParT model, vendored
  with the `weaver` dependency removed and attention-weight capture added.
- [`tutorial/part_utils.py`](tutorial/part_utils.py) — builds the model inputs
  from the `.root` file and runs a forward pass.
- [`tutorial/models/ParT_full.pt`](tutorial/models/ParT_full.pt) and
  [`tutorial/models/ParT_kin.pt`](tutorial/models/ParT_kin.pt) — pre-trained
  weights.

---

## Repository layout

```
ml4fp/
├── install.sh                 # environment setup
├── README.md                  # this file
├── data/
│   └── JetClass_example_100k.root
├── tutorial/
│   ├── 01_transformers_and_self_attention.ipynb
│   ├── 02_complexity_and_linformer.ipynb
│   ├── 03_particle_transformer.ipynb
│   ├── dataloader.py          # local JetClass loader
│   ├── part_model.py          # standalone ParT (used by NB3)
│   ├── part_utils.py          # NB3 input building + inference
│   ├── tutorial_data.py       # shared dataset/task helpers
│   └── models/
│       ├── ParT_full.pt
│       └── ParT_kin.pt
```

---

## Troubleshooting

- **`pip` can't find a `torch` wheel** — your Python is likely too new or too
  old. Use Python 3.11–3.13: `PYTHON=python3.12 ./install.sh`.
- **`ModuleNotFoundError: No module named 'dataloader'`** — run the notebooks
  from inside `tutorial/` (or open them there in Jupyter) so the local helper
  modules resolve correctly.
- **Jupyter uses the wrong Python** — make sure you `source .venv/bin/activate`
  before launching, or select the `.venv` kernel inside Jupyter.
- **Dataset download is slow or blocked** — pre-download it with the `curl`
  command above and place it in `data/`.
- **Transient `pip` network timeouts** — re-run `./install.sh`; it is safe to
  run again and resumes into the same `.venv`.
