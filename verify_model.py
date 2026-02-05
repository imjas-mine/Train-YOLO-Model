"""
Verify if the model can detect ANYTHING at all
"""

from ultralytics import YOLO

# Load model
model = YOLO(r'd:\Capstone\train\runs\train\yolo26_bottle\weights\best.pt')

# Test on TRAINING image (should definitely detect since it trained on these)
print("Testing on a TRAINING image (the model saw this during training):\n")

train_img = r'd:\Capstone\train\train\images'
import os
train_images = os.listdir(train_img)

if train_images:
    test_path = os.path.join(train_img, train_images[0])
    print(f"Image: {train_images[0]}\n")
    
    for conf in [0.5, 0.25, 0.1, 0.01]:
        results = model(test_path, conf=conf, verbose=False)
        det = len(results[0].boxes)
        print(f"Confidence {conf}: {det} detections")
        
        if det > 0:
            for box in results[0].boxes:
                cls = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                print(f"  - {cls}: {confidence*100:.1f}%")
            print()

print("\n" + "="*60)
print("Now testing on your test images:\n")

test_img = r'd:\Capstone\train\test\images\photo-1770269059845_jpg.rf.b9cc44918bd49c87b381fa0462892a40.jpg'

for conf in [0.5, 0.25, 0.1, 0.05, 0.01]:
    results = model(test_img, conf=conf, verbose=False)
    det = len(results[0].boxes)
    print(f"Confidence {conf}: {det} detections")
    
    if det > 0:
        for box in results[0].boxes:
            cls = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            print(f"  - {cls}: {confidence*100:.1f}%")