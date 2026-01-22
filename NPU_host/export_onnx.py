import os
from ultralytics import YOLO


def export_to_onnx(weight_path: str = "yolov8s.pt"):
    if not os.path.exists(weight_path):
        return
    model = YOLO(weight_path)
    model.export(format="onnx", dynamic=True)


if __name__ == "__main__":
    export_to_onnx()
