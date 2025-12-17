# 将官方 YOLO 模型导出为具有动态批大小和图像大小的 ONNX
# 增加入口保护与权重存在性检查，避免导入时误触发或因文件缺失报错

import os
from ultralytics import YOLO


def export_to_onnx(weight_path: str = "yolov8s.pt"):
    """导出 YOLO 权重为 ONNX，若文件不存在则给出友好提示。"""
    if not os.path.exists(weight_path):
        print(f"指定的权重文件 {weight_path} 不存在，请将占位符替换为实际路径后再执行。")
        return

    model = YOLO(weight_path)
    model.export(format="onnx", dynamic=True)
    print(f"已将 {weight_path} 导出为 ONNX 格式")


if __name__ == "__main__":
    export_to_onnx()
