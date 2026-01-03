import argparse

import host_process


def parse_args():
    parser = argparse.ArgumentParser(description="KunPeng YOLO target tracking")
    parser.add_argument(
        "--serial-port",
        default="/dev/ttyUSB0",
        help="串口号（如 /dev/ttyUSB0 或 COM5）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    host_process.start_recognition(
        args.serial_port,
    )

if __name__ == "__main__":
    main()

