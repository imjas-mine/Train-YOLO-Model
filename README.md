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

**Important:** Need **100+ images minimum** for good results!

### 3. Update File Paths ⚠️
**The scripts have hardcoded paths!** You need to change them:

Open these files and update paths to match your directory:
- `1_prepare_dataset.py` - Line 13: `base_dir = r"d:\Capstone\train"`
- `train_yolo26.py` - Line 67: `data_yaml = r'd:\Capstone\train\data.yaml'`
- `test_model.py` - Line 33: `test_images_dir = r'd:\Capstone\train\test\images'`

**Change to YOUR project directory!**

### 4. Split Dataset (if needed)
If you only have a `train` folder:
```bash
python 1_prepare_dataset.py
```

### 5. Train
```bash
python train_yolo26.py
```
Training takes 10-60 minutes. Press ENTER when prompted at each step.

### 6. Check Results
After training, check: `runs/train/yolo26_bottle/`
- `results.png` - Training performance graphs
- `weights/best.pt` - Your trained model
- `val_batch0_pred.jpg` - Sample predictions

### 7. Test
Put test images in `test/images/` folder, then:
```bash
python test_model.py
```
Results saved to `test_results/`

## Common Issues

**"No detections"** → Need more training data (100+ images)

**"Out of memory"** → Edit `train_yolo26.py`, change `batch_size = 16` to `batch_size = 8`

**"File not found"** → Update hardcoded paths in the scripts!

## Files You'll Use

- `1_prepare_dataset.py` - Split data into train/valid/test
- `train_yolo26.py` - Main training script
- `test_model.py` - Test your trained model
- `data.yaml` - Dataset config (auto-generated)

---

**Need 100+ labeled images for good results!**
