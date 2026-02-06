# YOLO Training Setup

Simple guide to train YOLO on your custom dataset.

## Quick Start

### 1. Install Dependencies
```bash
pip install ultralytics torch torchvision opencv-python matplotlib pyyaml numpy
```

### 2. Get Your Dataset
- Download your dataset from [Roboflow](https://roboflow.com/) in **YOLO format**
- Extract the zip file to this directory
- Make sure you have: `train/`, `valid/`, `test/` folders and `data.yaml`

**Important:** Need **30+ images minimum** for usable results!

### 3. Train
```bash
python train_yolo26.py
```
Training takes 10-60 minutes. Press ENTER when prompted at each step.

### 4. Check Results
After training, check: `runs/train/yolo26_bottle/`
- `results.png` - Training performance graphs
- `weights/best.pt` - Your trained model
- `val_batch0_pred.jpg` - Sample predictions

### 5. Test
Put test images in `test/images/` folder, then:
```bash
python test_model.py
```
Results saved to `test_results/`

## Common Issues

**"No detections"** → Need more training data (30+ images)

**"Out of memory"** → Edit `train_yolo26.py`, change `batch_size = 16` to `batch_size = 8`

## Files You'll Use

- `train_yolo26.py` - Main training script
- `test_model.py` - Test your trained model
- `verify_model.py` - Debug script to check model detection
- `use_pretrained.py` - Test with pretrained YOLO11 model
- `data.yaml` - Dataset config

---

**Need 30+ labeled images for usable results!**
