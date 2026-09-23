# Running This Project in VS Code — Full Setup Guide

## Why this fixes the Colab RAM crash

Colab (and any single Jupyter notebook) keeps **every variable from every
cell alive** in one Python process for the whole session. IEEE-CIS (590K
rows), Elliptic (203K nodes), Credit Card (284K rows), PaySim (6.3M rows),
and every trained model — all stayed resident in RAM at once, which is
exactly the kind of thing that exhausts free Colab's ~12GB.

**The fix baked into these scripts:** each week is a standalone `.py` file.
When it finishes running, Python releases all its memory automatically —
you don't have to remember to delete anything. A winning model is saved to
a small `.pkl` file on disk, so the next script loads *just that model*,
not the whole previous script's memory. You run each week as a separate
terminal command, not as cells in one long-lived kernel.

---

## Step 1 — Install VS Code + extensions

1. Install [VS Code](https://code.visualstudio.com/) if you don't have it.
2. Install these extensions (Extensions panel, `Ctrl+Shift+X` / `Cmd+Shift+X`):
   - **Python** (Microsoft) — required
   - **Jupyter** (Microsoft) — optional, only needed if you want to view
     `.ipynb` files or run cells interactively; not required for the `.py`
     scripts below

## Step 2 — Set up a virtual environment

Open the project folder in VS Code (`File > Open Folder`), then open the
integrated terminal (`` Ctrl+` `` / `` Cmd+` ``):

```bash
python -m venv .venv

# activate it:
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

VS Code should prompt "Select Python Interpreter" — pick the one inside
`.venv`. If it doesn't prompt, open the Command Palette (`Ctrl+Shift+P`),
run **"Python: Select Interpreter"**, and pick `.venv`.

**If `pip install torch_geometric` fails:** install torch first, wait for
it to finish, then install torch_geometric separately — installing them
in the same command sometimes fails because torch_geometric needs to
detect the already-installed torch version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

(CPU-only torch is fine here — Week 5's models are small; slower than a
GPU but it will finish in reasonable time on a laptop.)

## Step 3 — Kaggle API credentials (real ones, not Colab secrets)

1. Go to **kaggle.com/settings** → scroll to API → **"Create New Token"**.
   This downloads a `kaggle.json` file containing your username and key.
2. Open that file — it looks like:
   ```json
   {"username":"your_actual_username","key":"a1b2c3d4e5f6..."}
   ```
3. In the project folder, copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Open `.env` in VS Code and fill in the two values from `kaggle.json`:
   ```
   KAGGLE_USERNAME=your_actual_username
   KAGGLE_KEY=a1b2c3d4e5f6...
   ```
5. **Never commit `.env` to git** — it's already in `.gitignore`.

You also need to **accept the competition rules** for IEEE-CIS once,
manually, in your browser (Kaggle requires this for competition datasets,
API access alone isn't enough): go to
kaggle.com/competitions/ieee-fraud-detection/rules and click "I Understand
and Accept."

## Step 4 — Run each week as its own terminal command

This is the important part for avoiding the RAM crash — run these **one
at a time**, as separate commands, not as cells accumulating in one kernel:

```bash
cd scripts

python 01_data_acquisition.py     # downloads all 6 datasets
python 02_eda_pipeline.py         # EDA, one dataset at a time, released after each
python 03_tabular_ablation.py     # LightGBM/XGBoost/CatBoost on IEEE-CIS
python 04_graph_construction.py   # Elliptic graph + temporal split
python 05_graph_ablation.py       # GraphSAGE/GAT/GCN (needs torch_geometric)
python 06_nlp_phishing_module.py  # SMS + phishing URL classifiers
python 07_explainability.py       # SHAP + GNNExplainer on the actual winners
```

Each script prints its own results to the terminal and writes CSVs/PNGs/
JSON to `notebooks/<week>_report/`. Open those output folders in VS Code's
file explorer to view plots (click a `.png`) or tables (click a `.csv` —
VS Code shows a basic table view, or install the "Excel Viewer" /
"Data Wrangler" extension for a nicer one).

## If you still hit memory issues

Your machine's RAM ceiling might just be lower than Colab's ~12GB. If a
script still fails on `MemoryError`:

- **IEEE-CIS (Week 3)** is the biggest single risk — 590K rows × ~394
  columns. If it fails, add `dtype` hints to `pd.read_csv()` in
  `03_tabular_ablation.py` the same way `02_eda_pipeline.py` already does
  for PaySim, or read only a subset of the `V1`-`V339` columns (they're
  the least individually interpretable anyway).
- **PaySim (Week 2 EDA only)** — already dtype-optimized in
  `02_eda_pipeline.py`; if it's still too much, add `nrows=1_000_000` to
  `pd.read_csv()` for EDA purposes (Week 2 is exploratory, not a modeling
  step, so a sample is fine there).
- Close other applications, especially Chrome — browsers are often the
  biggest hidden RAM consumer on a laptop.
- Check actual usage while a script runs: Task Manager (Windows) / Activity
  Monitor (Mac) / `htop` (Linux terminal).

## Order matters — dependencies between scripts

```
01 (acquisition) ──► 02 (EDA, informational only, nothing downstream depends on it)
01 ──► 03 (tabular ablation) ──► saves notebooks/ablation_report/winner_model.pkl
01 ──► 04 (graph construction) ──► saves notebooks/graph_pipeline/*.npy
04 ──► 05 (graph ablation) ──► saves notebooks/graph_ablation_report/winner_model.pkl
01 ──► 06 (NLP module, fully standalone)
03 + 05 ──► 07 (explainability — loads BOTH winner_model.pkl files)
```

02 and 06 can run any time after 01. 07 needs both 03 and 05 to have
completed and saved their pickles first.
