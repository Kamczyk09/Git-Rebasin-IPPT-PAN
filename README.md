# Git Rebasin IPPT PAN

Research utilities for training, evaluating, explaining, and experimenting with neural network models on CIFAR-10 and medical images. This repository collects model definitions, explanation methods (Grad-CAM, RISE), weight interpolation and matching tools, and training/evaluation utilities used during experiments.

## Quick summary
- Models: `models/CNN.py`, `models/MLP.py`, `models/ResNet18.py`
- Explanations: `apply_grad_cam.py` (Grad-CAM), `rise/apply_rise.py` (RISE), and a local `pytorch_grad_cam/` implementation
- Weight operations: `linear_interpolation.py`, `weight_matching_mod.py`
- Training / eval helper: `main.py`
- Example checkpoints: `models_checkpoints/`
- Datasets: `data/` (CIFAR-10 raw batches and `medical_images/`)

This README explains what is in the repository and how to run the main scripts.

---

## Requirements & installation
Recommended: Python 3.12+ in a virtual environment.

Install a minimal set of commonly used packages (adjust versions if needed):

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib scikit-image opencv-python tqdm
```

If you prefer a requirements file, create one from your environment or add a `requirements.txt` to the repo and run:

```bash
pip install -r requirements.txt
```

Notes:
- If you need GPU support, install a PyTorch build that matches your CUDA version (see https://pytorch.org).
- Some scripts may use additional utilities included in `utils_module/` or `rise/`.

---

## Project layout (key files)
- `main.py` — high-level script for training and evaluation (check CLI flags with `-h`).
- `apply_grad_cam.py` — wrapper to apply Grad-CAM explanations to images/inputs.
- `rise/apply_rise.py` — wrapper to apply the RISE explanation method.
- `linear_interpolation.py` — create interpolated weight checkpoints between two saved models.
- `weight_matching_mod.py` — utilities to match/merge model weights.
- `models/` — model definitions used by scripts.
- `models_checkpoints/` — example pretrained and raw weights shipped with the repo.
- `data/` — datasets (CIFAR-10 batches, `medical_images/` with `train/` and `test/`).
- `pytorch_grad_cam/` — implementation of several CAM algorithms used by `apply_grad_cam.py`.
- `utils_module/`, `rise/`, `metrics/`, `feature_factorization/` — supporting utilities and research code.

---

## How to run (examples)
Run `-h` on each script to see supported flags; the repository uses simple CLI patterns. The examples below are templates — always check `-h` to ensure flags match your version.

1) Inspect CLI options

```bash
python main.py -h
python apply_grad_cam.py -h
python rise/apply_rise.py -h
python linear_interpolation.py -h
python weight_matching_mod.py -h
```

2) Train or evaluate with `main.py` (example)

```bash
# Train (example flags — replace with flags supported by your local main.py)
python main.py --mode train --model resnet18 --dataset cifar10 --epochs 50 --batch-size 128 --save-dir models_checkpoints/

# Evaluate
default checkpoint usage may vary; point to the appropriate file in models_checkpoints:
python main.py --mode eval --checkpoint models_checkpoints/resnet18_pretrained.pth --dataset cifar10
```

3) Apply Grad-CAM

```bash
# Example (adjust flags to your local script)
python apply_grad_cam.py --checkpoint models_checkpoints/resnet18_pretrained.pth --image data/medical_images/test/your_image.png --output out_gradcam.png
```

4) Apply RISE

```bash
python rise/apply_rise.py --checkpoint models_checkpoints/cnn_pretrained.pth --image data/medical_images/test/your_image.png --output out_rise.png
```

5) Linear interpolation between two weight files

```bash
python linear_interpolation.py --w1 models_checkpoints/model_a.pth --w2 models_checkpoints/model_b.pth --steps 11 --outdir interpolations/
```

6) Weight matching / merging

```bash
python weight_matching_mod.py --model1 models_checkpoints/mlp_raw.pth --model2 models_checkpoints/mlp_pretrained.pth --output matched.pth
```

---

## Using checkpoints & data
- Checkpoints are stored in `models_checkpoints/`. Load them by passing their path to the script's `--checkpoint` / `--weights` or equivalent flag.
- CIFAR-10 raw batches are in `data/cifar-10-batches-py/` (original format). The helper `data/cifar10.py` contains loading utilities.
- Medical images are in `data/medical_images/` organized into `train/` and `test/` subfolders used by scripts that expect image folders.

---

## GPU / CUDA
To choose device(s):

```bash
# Use specific GPU (device 0)
export CUDA_VISIBLE_DEVICES=0
python main.py --mode train ...

# Force CPU
export CUDA_VISIBLE_DEVICES=""
python main.py --mode eval ...
```

Inside scripts, device selection may be controlled by `torch.cuda.is_available()` or explicit CLI flags; check the script source if you need a different behavior.

---

## Troubleshooting
- Import errors: ensure the virtual environment is active and the working directory is the repository root.
- Package missing: install the missing package via `pip`.
- Checkpoint mismatch errors: confirm the model class you instantiate matches the checkpoint architecture.
- If a script fails because a CLI option is different than shown here, run `python <script>.py -h` to view the current options.

---

## Tips & next steps
- Add a `requirements.txt` with pinned versions for reproducibility.
- Add small example scripts that show end-to-end usage (train → eval → explain) with exact flags used in experiments.
- Consider adding notebooks that render example Grad-CAM / RISE outputs.

---

If you want, I can now: (A) inspect `main.py`, `apply_grad_cam.py`, and `rise/apply_rise.py` and update the README examples to show exact CLI flags used by each script, or (B) add a `requirements.txt` generated from the environment. Which would you prefer?

