import argparse
import configparser
import os

import cv2
import serial
import struct
from ultralytics import YOLO

# 注意画幅已经被强制
def SendDataToStm32(x_offset, size,ser):
    x_offset = max(-320, min(320, int(x_offset)))
    size = max(0, min(307200, int(size)))
    packet = struct.pack('<BBhIB', 0xB3, 0x01, x_offset, size, 0x5B)
    # 小端，永远显式地加上 < 或 >
    ser.write(packet)

def start_recognition(
    serial_port: str, camera_index: int, desired_width: int, desired_height: int, headless: bool
):
    model = YOLO('yolov8s.pt')

    cap = cv2.VideoCapture(camera_index)
    # 确认开发板也是0哦,表示首选

    # 强制设置为 640x480 (宽x高)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # 输出当前画幅设置
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"当前摄像头分辨率: {actual_width} x {actual_height}")

    if not cap.isOpened() or int(actual_width) != int(desired_width) or int(actual_height) != int(desired_height):
        print(
            "摄像头打开失败或分辨率未生效，请核对 /dev/video* 设备是否正确，"
            "或将配置中的摄像头索引/分辨率占位符替换为实际值。"
        )
        cap.release()
        cv2.destroyAllWindows()
        exit()

    print("摄像头运行中\n按下Q退出")
    UnableToSendData = False
    ser = None

    display_enabled = not headless

    if serial_port.startswith("/"):
        if not os.path.exists(serial_port):
            print(f"串口设备 {serial_port} 不存在，请检查连接或使用 --serial-port 指定实际串口号")
            cap.release()
            cv2.destroyAllWindows()
            return
        if not os.access(serial_port, os.R_OK | os.W_OK):
            print(f"没有访问串口设备 {serial_port} 的权限，请检查权限或使用 sudo 运行")
            cap.release()
            cv2.destroyAllWindows()
            return

    try:
        ser = serial.Serial(serial_port, 115200, timeout=0.1)
        # 防止串口连接时重置/挂起 STM32（将占位符替换为实际串口号）
        ser.setRTS(False)
        ser.setDTR(False)

        print(f"串口 {serial_port} 连接成功")
    except serial.SerialException as e:
        print(f"串口连接失败: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return

    try:
        while True:

            # 表示逐帧获取
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break

            # 如果单纯需要确认颜色的方块，可以不用模型，OPENCV就行了

            frame_width = frame.shape[1]
            # 彩色 BGR 图像（OpenCV 默认）：frame.shape == (height, width, 3)，参数都是int
            # 其中3是通道数，通常为3
            frame_center_x = frame_width // 2

            results = model(frame, classes=[67], verbose=False)
            # https://docs.ultralytics.com/zh/modes/predict/
            # 参考YOLO文档，对于每一帧直接调用模型，classes是识别的内容代号，verbose是否在控制台打印

            command = "undetected"
            # 没有满足时的默认文本

            if len(results[0].boxes) == 1:
                # results[0] 是对应这次传入的那一帧的检测结果对象
                # 每个经由model预测并返回的对象都具有boxes、masks、keypoints等属性可以调用。
                # 参考https://docs.ultralytics.com/zh/modes/predict/#key-features-of-predict-mode
                # results[0].boxes 是这个帧上所有检测到的边界框集合（对象数量）

                box = results[0].boxes[0]
                xyxy = box.xyxy[0].cpu().numpy()
                # xyxy是box对象的属性，代表四个坐标点(x_min, y_min, x_max, y_max)
                # 加上[0]确保获取一维坐标
                # 从NPU转移到CPU运行，从tensor格式转化为numpy数组

                object_center_x = (xyxy[0] + xyxy[2]) / 2

                error = object_center_x - frame_center_x
                # 负数偏左，正数偏右

                size = abs(xyxy[0]-xyxy[2])*abs(xyxy[1]-xyxy[3])
                # 用来确定距离

                dead_zone = 50
                # 决定中间的范围有多大

                # 保留死区：当偏差落在 dead_zone 内时不触发 PID 纠偏，向下位机发送 0 纠偏量
                pid_error = 0 if abs(error) <= dead_zone else error

                if (ser is not None and UnableToSendData==False):
                    try:
                        SendDataToStm32(pid_error, size,ser)
                    except Exception as e:
                        print("无法发送数据至单片机\n")
                        UnableToSendData = True

                command="error "+str(error)+"  "+"size "+str(size)

                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                # 用法是这样：
                # Parameters
                # img       Image.
                # pt1       Vertex of the rectangle.
                # pt2       Vertex of the rectangle opposite to pt1 .
                # color     Rectangle color or brightness (grayscale image).
                # thickness Thickness of lines that make up the rectangle. Negative values, like FILLED, mean that the function has to draw a filled rectangle.
                # lineType  Type of the line. See LineTypes
                # shift     Number of fractional bits in the point coordinates.

            elif len(results[0].boxes) > 1:
                command = "stop"
                if (ser is not None and UnableToSendData==False):
                    try:
                        packet = struct.pack('<BBhIB', 0xB3, 0x00, 0, 0, 0x5B)
                        ser.write(packet)
                    except Exception as e:
                        print("无法发送数据至单片机\n")
                        UnableToSendData = True

            else:
                command = "stop"
                if (ser is not None and UnableToSendData==False):
                    try:
                        packet = struct.pack('<BBhIB', 0xB3, 0x00, 0, 0, 0x5B)
                        ser.write(packet)
                    except Exception as e:
                        print("无法发送数据至单片机\n")
                        UnableToSendData = True

            cv2.putText(frame, command, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if display_enabled:
                try:
                    cv2.imshow('YOLOv8s Test', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    print("检测到无法创建显示窗口，已自动切换为无界面模式，仅进行推理和串口发送。")
                    display_enabled = False
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None and ser.is_open:
            ser.close()

def read_config(config_path: str):
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，将使用命令行/默认参数。")
        return {}

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return config


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
        help="摄像头索引，可通过命令行、环境变量CAMERA_INDEX或配置文件指定",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.getenv("CAMERA_WIDTH", 640)),
        help="期望宽度，可通过命令行、环境变量CAMERA_WIDTH或配置文件指定",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.getenv("CAMERA_HEIGHT", 480)),
        help="期望高度，可通过命令行、环境变量CAMERA_HEIGHT或配置文件指定",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无界面模式，跳过画面显示，仅进行推理与串口发送",
    )
    return parser.parse_args()


def merge_camera_config(args):
    config = read_config(args.config) if args.config else {}

    if isinstance(config, configparser.ConfigParser) and config.has_section("camera"):
        camera_index = config.getint("camera", "index", fallback=args.camera_index)
        width = config.getint("camera", "width", fallback=args.width)
        height = config.getint("camera", "height", fallback=args.height)
    else:
        camera_index = args.camera_index
        width = args.width
        height = args.height

    return camera_index, width, height


def main():
    args = parse_args()
    camera_index, width, height = merge_camera_config(args)
    start_recognition(args.serial_port, camera_index, width, height, args.headless)

if __name__ == "__main__":
    main()

