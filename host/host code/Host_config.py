import argparse
import os

import host_process


def parse_args():
    parser = argparse.ArgumentParser(description="KunPeng YOLO target tracking")
    parser.add_argument(
        "--serial-port",
        default=os.getenv("SERIAL_PORT", "/dev/ttyUSB0"),
        help="串口号（如 /dev/ttyUSB0 或 COM5，可通过命令行或环境变量SERIAL_PORT配置）",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.getenv("CAMERA_WIDTH", 640)),
        help="期望宽度，可通过命令行或环境变量CAMERA_WIDTH配置",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.getenv("CAMERA_HEIGHT", 480)),
        help="期望高度，可通过命令行或环境变量CAMERA_HEIGHT配置",
    )
    return parser.parse_args()


def merge_camera_config(args):
    width = args.width
    height = args.height

    return width, height

def main():
    args = parse_args()
    width, height = merge_camera_config(args)
    host_process.start_recognition(
        args.serial_port,
        width,
        height,
    )

if __name__ == "__main__":
    main()

