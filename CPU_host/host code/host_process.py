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
        # self等同于this指针，指向LFR所对应的实例，__init__相当于构造函数
        self._cap = cap
        self._frame = None
        # 初始化帧的缓冲区
        self._lock = threading.Lock()
        # 类似std::mutex的线程锁，与读线程避开
        self._running = True
        # 创建线程对象。从reader开始基于主线程
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            # retval, image = cv2.VideoCapture.read()
            # image 是返回的捕获到的帧，如果没有帧被捕获到，则该值为空。
            # retval 表示帧捕获是否成功，如果成功，retval为True，失败为False。
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                # 等同于lock_guard，而不是异常处理
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
        print(f"路径 {CAMERA_DEVICE_PATH} 不存在或摄像头未连接")
        return None

    cap = cv2.VideoCapture(CAMERA_DEVICE_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"该路径的摄像头设备 {CAMERA_DEVICE_PATH} 无法打开")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"当前摄像头分辨率: {actual_width} x {actual_height}")

    if int(actual_width) != int(CAMERA_WIDTH) or int(actual_height) != int(CAMERA_HEIGHT):
        print(f"设定的分辨率与实际分辨率不一致，请确认 {CAMERA_DEVICE_PATH} 是否存在且可用")
        cap.release()
        return None

    return cap


def start_recognition():
    model = YOLO('../yolov8s.pt')
    print(f"当前运行设备: {model.device}")
    cap = _open_camera()
    if cap is None:
        print("cap的结果是none，已终止")
        return

    print("摄像头运行中")
    UnableToSendData = False
    ser = None
    frame_reader = None

    try:
        if not os.path.exists(SERIAL_DEVICE_PATH):
            print(f"串口设备 {SERIAL_DEVICE_PATH} 不存在或未连接，已终止")
            cap.release()
            return
        if not os.access(SERIAL_DEVICE_PATH, os.R_OK | os.W_OK):
            print(f"没有访问串口设备 {SERIAL_DEVICE_PATH} 的权限，已终止")
            cap.release()
            return

        ser = serial.Serial(SERIAL_DEVICE_PATH, 115200, timeout=0.1)
        # 固定为 RTS=低、DTR=高，避免复位或进入 Bootloader
        ser.setRTS(False)
        ser.setDTR(True)

        print(f"串口 {SERIAL_DEVICE_PATH} 连接成功 (RTS=低, DTR=高)")
    except serial.SerialException as e:
        print(f"串口连接失败: {e}")
        cap.release()
        return

    try:
        frame_reader = LatestFrameReader(cap)
        last_log_time = time.time()
        process_count = 0

        while True:
            # 表示逐帧获取
            frame = frame_reader.read()
            if frame is None:
                time.sleep(0.005)
                # 防止空转
                continue

            frame_width = frame.shape[1]
            # 彩色 BGR 图像（OpenCV 默认）：frame.shape == (height, width, 3)，参数都是int
            # 其中3是通道数，通常为3
            frame_center_x = frame_width // 2

            results = model(frame, classes=[67], verbose=False, conf=0.15)
            # 降低一点置信度
            # https://docs.ultralytics.com/zh/modes/predict/
            # 参考YOLO文档，对于每一帧直接调用模型，classes是识别的内容代号，verbose是否在控制台打印

            process_count+=1

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
                if (ser is not None and UnableToSendData==False):
                    try:
                        packet = struct.pack('<BBhIB', 0xB3, 0x00, 0, 0, 0x5B)
                        ser.write(packet)
                    except Exception as e:
                        print("无法发送数据至单片机\n")
                        UnableToSendData = True

                log_message = "检测到多个物品，暂停控制"
            else:
                if (ser is not None and UnableToSendData==False):
                    try:
                        packet = struct.pack('<BBhIB', 0xB3, 0x00, 0, 0, 0x5B)
                        ser.write(packet)
                    except Exception as e:
                        print("无法发送数据至单片机\n")
                        UnableToSendData = True

                log_message = "未检测到物品"

            current_time = time.time()
            time_delta = current_time - last_log_time

            if time_delta >= 0.5:
                fps = process_count / time_delta

                print(f"FPS: {fps:.1f} | {log_message}")

                last_log_time = current_time
                process_count = 0


    finally:
        if frame_reader is not None:
            frame_reader.stop()
        cap.release()
        if ser is not None and ser.is_open:
            ser.close()

def main():
    start_recognition()

if __name__ == "__main__":
        main()
