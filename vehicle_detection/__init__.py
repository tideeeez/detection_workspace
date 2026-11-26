"""Ultralytics YOLOv8s vehicle detection."""

VEHICLE_CLASSES = (2, 3, 5, 7)  # COCO car, motorcycle, bus, truck


class VehicleDetector:
    def __init__(self, weights, device, image_size):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.options = {
            "classes": VEHICLE_CLASSES,
            "conf": 0.10,  # Retain ByteTrack's low-confidence association inputs.
            "iou": 0.70,
            "imgsz": image_size,
            "device": device,
            "half": device.startswith("cuda"),
            "verbose": False,
        }

    def detect(self, image):
        result = self.model.predict(image, **self.options)[0]
        return result.boxes.cpu().numpy()
