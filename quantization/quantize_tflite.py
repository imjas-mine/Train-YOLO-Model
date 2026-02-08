"""
YOLO26n INT8 Quantization Pipeline for Android Deployment

Converts the trained YOLO26n model to TFLite INT8 format
optimized for Android devices (2023-2025 era phones).

Quantization details:
- Format: TFLite with full INT8 quantization
- Image size: 320x320 (optimized for mobile inference speed)
- Calibration: Uses training dataset for representative data
"""

from ultralytics import YOLO
import os
import shutil
import sys


# ============================================
# Configuration
# ============================================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "runs",
    "detect",
    "runs",
    "train",
    "yolo26_trained",
    "weights",
    "best.pt",
)
DATA_YAML = os.path.join(os.path.dirname(__file__), "..", "data.yaml")
IMG_SIZE = 320
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def validate_paths():
    """Validate that required files exist before starting quantization."""
    model_path = os.path.normpath(MODEL_PATH)
    data_path = os.path.normpath(DATA_YAML)

    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at: {model_path}")
        print("Make sure training has completed before running quantization.")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"Error: Dataset config not found at: {data_path}")
        sys.exit(1)

    print(f"Model path: {model_path}")
    print(f"Dataset config: {data_path}")
    return model_path, data_path


def quantize_model(model_path, data_path):
    """Export the trained model to TFLite INT8 format."""
    print("\n" + "=" * 60)
    print("Loading trained YOLO26n model...")
    print("=" * 60)

    model = YOLO(model_path)

    print(f"Model loaded successfully")
    print(f"Model task: {model.task}")
    print(f"Number of classes: {len(model.names)}")
    print(f"Classes: {model.names}")

    print("\n" + "=" * 60)
    print("Exporting to TFLite INT8...")
    print("=" * 60)
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Quantization: INT8")
    print(f"Calibration data: {data_path}")
    print("This may take several minutes...\n")

    exported_path = model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        int8=True,
        data=data_path,
    )

    return exported_path


def move_to_output(exported_path):
    """Move the exported TFLite model to the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if exported_path and os.path.exists(exported_path):
        filename = os.path.basename(exported_path)
        dest_path = os.path.join(OUTPUT_DIR, filename)

        shutil.copy2(exported_path, dest_path)

        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)

        print("\n" + "=" * 60)
        print("Quantization Complete")
        print("=" * 60)
        print(f"Output: {os.path.normpath(dest_path)}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Format: TFLite INT8")
        print(f"Input size: {IMG_SIZE}x{IMG_SIZE}")
        print(f"\nThis model is ready for Android deployment.")
        return dest_path
    else:
        print(f"Error: Export did not produce expected output at: {exported_path}")
        sys.exit(1)


def verify_exported_model(tflite_path):
    """Load and verify the exported TFLite model runs correctly."""
    print("\n" + "=" * 60)
    print("Verifying exported model...")
    print("=" * 60)

    tflite_model = YOLO(tflite_path, task="detect")

    test_images_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "test", "images")
    )

    if not os.path.exists(test_images_dir):
        print("No test images found, skipping inference verification.")
        return

    image_files = [
        f
        for f in os.listdir(test_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not image_files:
        print("No test images found, skipping inference verification.")
        return

    test_image = os.path.join(test_images_dir, image_files[0])
    print(f"Running inference on: {image_files[0]}")

    results = tflite_model(test_image, conf=0.25, device="cpu", verbose=False)

    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            print(f"Detected {len(boxes)} objects:")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = tflite_model.names[cls_id]
                print(f"  - {class_name}: {conf * 100:.1f}%")
        else:
            print("No objects detected in test image (this may be normal).")

    print("\nVerification complete. Model is functional.")


if __name__ == "__main__":
    print("=" * 60)
    print("YOLO26n INT8 Quantization Pipeline")
    print("Target: Android devices (TFLite INT8, 320x320)")
    print("=" * 60)

    model_path, data_path = validate_paths()
    exported_path = quantize_model(model_path, data_path)
    tflite_path = move_to_output(exported_path)
    verify_exported_model(tflite_path)

    print("\n" + "=" * 60)
    print("Pipeline finished.")
    print("=" * 60)
