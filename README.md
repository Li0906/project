# Hierarchical Path Planning with Diffusion and PSO

This repository contains a grid-based hierarchical path planning project. The high-level planner uses a diffusion model to predict intermediate waypoints, and the low-level planner uses PSO/continuous distance-field optimization to refine collision-free paths.

The main project code is organized around three command-line scripts and eight core source files:

- Entry scripts: `scripts/gendata_grid.py`, `scripts/train.py`, `scripts/eval.py`
- Core source files: `src/data_gen_fixed.py`, `src/dataset_fixed.py`, `src/diffusion.py`, `src/eval_map.py`, `src/model_map.py`, `src/pso_optimizer.py`, `src/train_map.py`, `src/utils.py`

## Project Structure

```text
.
+-- src/                    # Core model, dataset, diffusion, evaluation and optimizers
|   +-- data_gen_fixed.py    # HDF5 dataset generation for fixed/grid maps
|   +-- dataset_fixed.py     # PyTorch dataset for angle-step prediction
|   +-- diffusion.py         # Gaussian diffusion utilities
|   +-- eval_map.py          # Hierarchical planner and evaluation metrics
|   +-- model_map.py         # Angle denoising network
|   +-- pso_optimizer.py     # Vectorized PSO path optimizer
|   +-- train_map.py         # Training loop
|   +-- utils.py             # A*, collision checks and path helpers
+-- scripts/                 # Command-line entry points
|   +-- gendata_grid.py      # Generate training/test HDF5 data
|   +-- train.py             # Train diffusion angle model
|   +-- eval.py              # Evaluate Diffusion+PSO or pure PSO baseline
+-- data/                    # Small demo data or generated HDF5 files
+-- runs/                    # Model checkpoints and evaluation outputs, ignored by git
+-- requirements.txt
+-- README.md
```

## Environment

Python 3.8 or newer is recommended. A CUDA GPU is recommended for training and PSO evaluation. CPU can be used for small debugging runs by passing `--device cpu`.

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you need a specific CUDA version of PyTorch, install PyTorch first from the official PyTorch command generator, then install the remaining dependencies:

```bash
pip install numpy h5py matplotlib scipy
```

## Quick Start

Run all commands from the repository root, the directory that contains `src/` and `scripts/`.

### 1. Generate Dataset

Generate a 64x64 grid dataset:

```bash
python scripts/gendata_grid.py --out data/testgrid64.h5 --n 50000 --H 64 --W 64 --T 10 --radius 16.0 --pmin 0.05 --pmax 0.15 --seed 1234
```

For a smaller smoke test:

```bash
python scripts/gendata_grid.py --out data/debug_grid64.h5 --n 200 --H 64 --W 64 --T 10 --radius 16.0
```

### 2. Train Model

Train the angle prediction diffusion model:

```bash
python scripts/train.py --h5 data/testgrid64.h5 --ckpt runs/testmodel64.pt --epochs 200 --batch 512 --lr 2e-4 --workers 4 --device cuda --T_steps 25
```

For CPU or quick debugging:

```bash
python scripts/train.py --h5 data/debug_grid64.h5 --ckpt runs/debug_model.pt --epochs 2 --batch 32 --workers 0 --device cpu --T_steps 25
```

### 3. Evaluate Diffusion + PSO

Evaluate the trained model and save visualizations:

```bash
python scripts/eval.py --h5 data/testgrid64.h5 --ckpt runs/testmodel64.pt --n_eval 200 --device cuda --viz --out_dir runs/viz_eval
```

Important evaluation parameters:

```bash
python scripts/eval.py \
  --h5 data/testgrid64.h5 \
  --ckpt runs/testmodel64.pt \
  --n_eval 200 \
  --device cuda \
  --radius 16.0 \
  --max_steps 12 \
  --pso_n_particles 80 \
  --pso_max_iter 50 \
  --w_col 10000 \
  --w_clear 1.0 \
  --w_smooth 8.0 \
  --safe_dist 1.1 \
  --viz \
  --out_dir runs/viz_eval
```

### 4. Evaluate Pure PSO Baseline

Use `--baseline` to disable the diffusion model and run direct PSO:

```bash
python scripts/eval.py --h5 data/testgrid64.h5 --n_eval 200 --device cuda --baseline --viz --out_dir runs/viz_purepso
```

## Outputs

Evaluation saves visualizations and JSON metrics under the selected output directory, for example:

```text
runs/viz_eval/
├── eval_map_success_0.png
├── eval_map_collision_1.png
└── results_diffusion_pso.json
```

The JSON metrics include success rate, path collision rate, normalized path length and average generation time.

## Uploading to GitHub

Use `GP/repo` as the GitHub repository root. Do not upload `.venv/`, `__pycache__/`, large `runs/` outputs, old experiment folders, or large generated datasets.

### Option A: GitHub Desktop

1. Open GitHub Desktop.
2. Choose `File -> Add local repository`.
3. Select this folder: `GP/repo`.
4. Commit all files.
5. Click `Publish repository`.

### Option B: Command Line

Install Git first if `git --version` does not work.

```bash
cd GP/repo
git init
git branch -M main
git add README.md requirements.txt .gitignore
git add scripts/gendata_grid.py scripts/train.py scripts/eval.py
git add src/data_gen_fixed.py src/dataset_fixed.py src/diffusion.py src/eval_map.py
git add src/model_map.py src/pso_optimizer.py src/train_map.py src/utils.py
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub account and repository name.

If you also want to include the small demo dataset, add it explicitly before committing:

```bash
git add data/grid32.h5
```

Do not commit generated checkpoints or visualizations such as `runs/*.pt` and `runs/viz/*.png`.

### Option C: GitHub Web Upload

1. Open your empty GitHub repository in the browser.
2. Click `uploading an existing file`.
3. Drag only the contents of `GP/repo` into the page.
4. Avoid dragging `.venv`, `runs`, `__pycache__`, and old experiment folders.
5. Commit the upload.

## Reproducibility Notes

- The scripts set random seeds in key training/evaluation paths.
- CUDA and PyTorch versions may still cause small numerical differences.
- Generated data depends on dataset size, grid size, obstacle density and random seed.
- The checkpoint path used by evaluation must match the checkpoint path used during training.
