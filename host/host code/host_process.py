import cv2
import serial
import os
import struct
import time
import threading
from ultralytics import YOLO

import host_send


CAMERA_DEVICE_PATH = "/dev/v4l/by-id/usb-ZC_USB_Camera_200901010001-video-index0"
SERIAL_DEVICE_PATH = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


class LatestFrameReader:
    def __init__(self, cap):
        self._cap = cap
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            frame = self._frame
            self._frame = None
        if frame is None:
            return None
        return frame.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1)

def _open_camera():
    if not os.path.exists(CAMERA_DEVICE_PATH):
        print(f"摄像头设备 {CAMERA_DEVICE_PATH} 不存在或未连接")
        return None

    cap = cv2.VideoCapture(CAMERA_DEVICE_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"摄像头设备 {CAMERA_DEVICE_PATH} 无法打开")
        cap.release()
        cv2.destroyAllWindows()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"当前摄像头分辨率: {actual_width} x {actual_height}")

    if int(actual_width) != int(CAMERA_WIDTH) or int(actual_height) != int(CAMERA_HEIGHT):
        print(
            f"摄像头打开失败或分辨率未生效，请确认 {CAMERA_DEVICE_PATH} 是否存在且可用"
        )
        cap.release()
        cv2.destroyAllWindows()
        return None

    return cap


def start_recognition():
    model = YOLO('../yolov8s.pt')

    cap = _open_camera()
    if cap is None:
        return

    print("摄像头运行中")
    UnableToSendData = False
    ser = None

    try:
        if not os.path.exists(SERIAL_DEVICE_PATH):
            print(f"串口设备 {SERIAL_DEVICE_PATH} 不存在或未连接")
            cap.release()
            cv2.destroyAllWindows()
            return
        if not os.access(SERIAL_DEVICE_PATH, os.R_OK | os.W_OK):
            print(f"没有访问串口设备 {SERIAL_DEVICE_PATH} 的权限")
            cap.release()
            cv2.destroyAllWindows()
            return

        ser = serial.Serial(SERIAL_DEVICE_PATH, 115200, timeout=0.1)
        # 固定为 RTS=低、DTR=高，避免复位或进入 Bootloader
        ser.setRTS(False)
        ser.setDTR(True)

        print(f"串口 {SERIAL_DEVICE_PATH} 连接成功 (RTS=低, DTR=高)")
    except serial.SerialException as e:
        print(f"串口连接失败: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return

    try:
        last_log_time = time.time()
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

            log_message = "未检测到物品"
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
                        host_send.SendDataToStm32(pid_error, size, ser)
                    except Exception as e:
                        print("无法发送数据至单片机\n")
                        UnableToSendData = True

                command="error "+str(error)+"  "+"size "+str(size)
                log_message = f"物品检测: x_offset={error:.2f}, size={size:.2f}"

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

                log_message = "检测到多个物品，暂停控制"
            else:
                command = "stop"
                if (ser is not None and UnableToSendData==False):
                    try:
                        packet = struct.pack('<BBhIB', 0xB3, 0x00, 0, 0, 0x5B)
                        ser.write(packet)
                    except Exception as e:
                        print("无法发送数据至单片机\n")
                        UnableToSendData = True

                log_message = "未检测到物品"

            if time.time() - last_log_time >= 0.5:
                print(log_message)
                last_log_time = time.time()

            cv2.putText(frame, command, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None and ser.is_open:
            ser.close()
