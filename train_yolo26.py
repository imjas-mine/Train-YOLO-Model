"""
YOLO26n Bottle Detection Training Script
Learn what each part does by running this step-by-step
"""

# ============================================
# STEP 1: Import Required Libraries
# ============================================
print("=" * 60)
print("STEP 1: Importing libraries...")
print("=" * 60)

from ultralytics import YOLO  # YOLO framework
import torch                   # PyTorch for deep learning
import os                      # File operations

print("✓ Libraries imported successfully!")


# ============================================
# STEP 2: Check GPU Availability
# ============================================
print("\n" + "=" * 60)
print("STEP 2: Checking for GPU...")
print("=" * 60)

if torch.cuda.is_available():
    device = '0'  # Use GPU
    print(f"✓ GPU Found: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print("✓ Training will be FAST! ⚡")
else:
    device = 'cpu'  # Use CPU
    print("⚠️  No GPU found - using CPU")
    print("⚠️  Training will be slower 🐢")

input("\n👉 Press ENTER to continue to Step 3...")


# ============================================
# STEP 3: Load YOLO26n Model
# ============================================
print("\n" + "=" * 60)
print("STEP 3: Loading YOLO26n model...")
print("=" * 60)

try:
    model = YOLO('yolo26n.pt')  # YOUR SPECIFIC MODEL!
    print("✓ YOLO26n model loaded successfully!")
    print("  (If this is first time, it will download ~6-10MB)")
except Exception as e:
    print(f"❌ Error loading YOLO26n: {e}")
    print("\n💡 Tip: YOLO26 might not exist yet.")
    print("   Available models: yolo11n.pt, yolov8n.pt, yolov9n.pt")
    exit()

input("\n👉 Press ENTER to continue to Step 4...")


# ============================================
# STEP 4: Set Up Training Parameters
# ============================================
print("\n" + "=" * 60)
print("STEP 4: Setting up training parameters...")
print("=" * 60)

# Dataset configuration
data_yaml = r'd:\Capstone\train\data.yaml'

# Training parameters
epochs = 50           # Number of training cycles
batch_size = 16       # Images per batch
img_size = 320        # Image size (320x320 pixels)
patience = 10         # Stop if no improvement for 10 epochs

print(f"✓ Dataset: {data_yaml}")
print(f"✓ Epochs: {epochs}")
print(f"✓ Batch Size: {batch_size}")
print(f"✓ Image Size: {img_size}x{img_size}")
print(f"✓ Early Stopping Patience: {patience} epochs")
print(f"✓ Device: {device}")

input("\n👉 Press ENTER to START TRAINING (Step 5)...")


# ============================================
# STEP 5: Train the Model
# ============================================
print("\n" + "=" * 60)
print("STEP 5: TRAINING STARTED!")
print("=" * 60)
print("\n⏰ This will take 10-60 minutes depending on your hardware")
print("💡 You'll see progress for each epoch\n")

results = model.train(
    data=data_yaml,           # Your dataset config
    epochs=epochs,            # Train for 50 epochs
    batch=batch_size,         # 16 images per batch
    imgsz=img_size,           # 320x320 image size
    patience=patience,        # Stop early if no improvement
    device=device,            # GPU or CPU
    project=r'd:\Capstone\train\runs\train',  # Save location
    name='yolo26_bottle',     # Run name
    
    # Learning rate parameters
    lr0=0.01,                 # Initial learning rate
    lrf=0.01,                 # Final learning rate
    
    # Other settings
    plots=True,               # Generate training plots
    save_period=10,           # Save checkpoint every 10 epochs
    verbose=True              # Show detailed output
)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)


# ============================================
# STEP 6: Save and Show Results
# ============================================
print("\n" + "=" * 60)
print("STEP 6: Saving results...")
print("=" * 60)

# Results location
results_dir = r'd:\Capstone\train\runs\train\yolo26_bottle'
best_model = os.path.join(results_dir, 'weights', 'best.pt')

print(f"\n📊 Training results saved to:")
print(f"   {results_dir}")
print(f"\n🏆 Best model saved at:")
print(f"   {best_model}")
print(f"\n📈 Check these files:")
print(f"   - results.png (training graphs)")
print(f"   - confusion_matrix.png")
print(f"   - val_batch0_pred.jpg (sample predictions)")

print("\n" + "=" * 60)
print("🎉 ALL DONE! Your model is trained!")
print("=" * 60)