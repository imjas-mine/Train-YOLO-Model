"""
Test YOLO26n Trained Model
Test your trained model on new images
"""

# ============================================
# STEP 1: Import and Load Model
# ============================================
print("=" * 60)
print("STEP 1: Loading your trained model...")
print("=" * 60)

from ultralytics import YOLO
import os

# Load your trained model
model_path = 'runs/detect/runs/train/yolo26_trained/weights/best.pt'
model = YOLO(model_path)

print(f"✓ Model loaded from: {model_path}")

input("\n👉 Press ENTER to continue to Step 2...")


# ============================================
# STEP 2: Get Test Images
# ============================================
print("\n" + "=" * 60)
print("STEP 2: Finding test images...")
print("=" * 60)

test_images_dir = 'test/images'
image_files = [f for f in os.listdir(test_images_dir) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

print(f"✓ Found {len(image_files)} test images")
for img in image_files:
    print(f"   - {img}")

input("\n👉 Press ENTER to run detection (Step 3)...")


# ============================================
# STEP 3: Run Detection
# ============================================
print("\n" + "=" * 60)
print("STEP 3: Running object detection...")
print("=" * 60)

# Confidence threshold (how sure the model should be)
confidence = 0.25  # 25% minimum confidence

for img_file in image_files:
    img_path = os.path.join(test_images_dir, img_file)
    
    print(f"\n📸 Processing: {img_file}")
    
    # Run detection
    results = model(img_path, conf=confidence)
    
    # Print detections
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            print(f"   ✓ Found {len(boxes)} objects:")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                print(f"      - {class_name}: {conf*100:.1f}% confident")
        else:
            print(f"   ⚠️  No objects detected")

input("\n👉 Press ENTER to save visualizations (Step 4)...")


# ============================================
# STEP 4: Save Visualized Results
# ============================================
print("\n" + "=" * 60)
print("STEP 4: Saving detection results...")
print("=" * 60)

# Create output directory
output_dir = 'test_results'
os.makedirs(output_dir, exist_ok=True)

for img_file in image_files:
    img_path = os.path.join(test_images_dir, img_file)
    
    # Run detection and save
    results = model(img_path, conf=confidence)
    
    # Save annotated image
    output_path = os.path.join(output_dir, f'detected_{img_file}')
    results[0].save(output_path)
    
    print(f"✓ Saved: {output_path}")

print(f"\n📁 All results saved to: {output_dir}")

print("\n" + "=" * 60)
print("🎉 TESTING COMPLETE!")
print("=" * 60)
print("\n💡 Next: Open the 'test_results' folder to see your detections!")
print("=" * 60)