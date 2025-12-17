import os
from ultralytics import YOLO


def print_model_names(weight_path: str = "yolov8s.pt"):
    """打印模型类别名称，添加入口保护与权重缺失提示。"""
    if not os.path.exists(weight_path):
        print(f"指定的权重文件 {weight_path} 不存在，请将占位符替换为实际路径后再执行。")
        return

    model = YOLO(weight_path)
    print(model.names)


if __name__ == "__main__":
    print_model_names()
