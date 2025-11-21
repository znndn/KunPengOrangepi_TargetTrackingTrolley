#将官方 YOLO 模型导出为具有动态批大小和图像大小的 ONNX。

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="onnx", dynamic=True)