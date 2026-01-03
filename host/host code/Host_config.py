import argparse
import configparser
import os

import host_process
import host_send


def parse_args():
    parser = argparse.ArgumentParser(description="KunPeng YOLO target tracking")
    parser.add_argument(
        "--serial-port",
        default=os.getenv("SERIAL_PORT", "/dev/ttyUSB0"),
        help="串口号（如 /dev/ttyUSB0 或 COM5，可通过命令行或环境变量SERIAL_PORT配置）",
    )
    parser.add_argument(
        "--config",
        help="可选配置文件路径，支持 [camera] 段的 index/width/height 参数",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=int(os.getenv("CAMERA_INDEX", 0)),
        help="摄像头索引通过命令行环境变量CAMERA_INDEX或配置文件",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.getenv("CAMERA_WIDTH", 640)),
        help="期望宽度通过命令行环境变量CAMERA_WIDTH或配置文件",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.getenv("CAMERA_HEIGHT", 480)),
        help="期望高度通过命令行环境变量CAMERA_HEIGHT或配置文件",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无界面",
    )
    return parser.parse_args()


def merge_camera_config(args):
    if args.config:
        config = host_process.read_config(args.config)
    else:
        config = {}

    if isinstance(config, configparser.ConfigParser) and config.has_section("camera"):
        camera_index = config.getint("camera", "index", fallback=args.camera_index)
        width = config.getint("camera", "width", fallback=args.width)
        height = config.getint("camera", "height", fallback=args.height)
    else:
        camera_index = args.camera_index
        width = args.width
        height = args.height

    return camera_index, width, height

def read_config(config_path: str):
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用命令行/默认参数。")
        return {}

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return config

def main():
    args = parse_args()
    camera_index, width, height = merge_camera_config(args)
    host_process.start_recognition(
        args.serial_port,
        camera_index,
        width,
        height,
        args.headless,
    )

if __name__ == "__main__":
    main()

