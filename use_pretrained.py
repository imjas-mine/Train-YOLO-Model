"""
Use YOLO's pretrained model (already trained on 80 common objects including bottles)
"""

from ultralytics import YOLO
import os

# Load pretrained YOLO model (trained on COCO dataset with 80 classes)
print("Loading pretrained YOLO model...\n")
model = YOLO("yolo11n.pt")  # Pretrained on COCO dataset

# COCO dataset includes class 39 = 'bottle'
print("This model can detect 80 objects, including bottles!\n")

# Test on your images
test_dir = "test/images"

for img_file in os.listdir(test_dir):
    if img_file.lower().endswith((".jpg", ".png")):
        img_path = os.path.join(test_dir, img_file)
        print(f"Testing: {img_file}")

        results = model(img_path, conf=0.25)

        # Show detections
        for r in results:
            if len(r.boxes) > 0:
                print(f"  Found {len(r.boxes)} objects:")
                for box in r.boxes:
                    cls_name = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    print(f"    - {cls_name}: {conf * 100:.1f}%")
            else:
                print("  No objects detected")

        # Save result
        output_path = os.path.join(test_dir, f"PRETRAINED_{img_file}")
        results[0].save(output_path)
        print(f"  Saved to: PRETRAINED_{img_file}\n")

print("Done! Check test/images/ folder for PRETRAINED_* files")
