# YOLO Training Setup

Simple guide to train YOLO on your custom dataset.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- NVIDIA GPU with CUDA drivers (run `nvidia-smi` to verify)

## Quick Start

### 1. Install uv (if not already installed)

```bash
pip install uv
```

### 2. Set Up the Environment

```bash
uv sync
```

This creates a `.venv` virtual environment and installs all dependencies (including PyTorch with CUDA 13.0 support) as defined in `pyproject.toml`.

### 3. Get Your Dataset

- Download your dataset from [Roboflow](https://roboflow.com/) in **YOLO format**
- Extract the zip file to this directory
- Make sure you have: `train/`, `valid/`, `test/` folders and `data.yaml`

**Important:** Need **30+ images minimum** for usable results!

### 4. Train

```bash
uv run python train_yolo26.py
```

Training takes 10-60 minutes. Press ENTER when prompted at each step.

### 5. Check Results

After training, check: `runs/train/yolo26_trained/`

- `results.png` - Training performance graphs
- `weights/best.pt` - Your trained model
- `val_batch0_pred.jpg` - Sample predictions

### 6. Test

Put test images in `test/images/` folder, then:

```bash
uv run python test_model.py
```

Results saved to `test_results/`

## Common Issues

**"No detections"** - Need more training data (30+ images)

**"Out of memory"** - Edit `train_yolo26.py`, change `batch_size = 16` to `batch_size = 8`

**"CUDA not available"** - Make sure you ran `uv sync` and are using `uv run` to execute scripts. Check that `nvidia-smi` works and shows your GPU.

## Files

- `train_yolo26.py` - Main training script
- `test_model.py` - Test your trained model
- `verify_model.py` - Debug script to check model detection
- `use_pretrained.py` - Test with pretrained YOLO11 model
- `data.yaml` - Dataset config
- `pyproject.toml` - Project dependencies and uv configuration

---

**Need 30+ labeled images for usable results!**