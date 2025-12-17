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

def start_recognition():
    model = YOLO('yolo11n.pt')

    cap = cv2.VideoCapture(0)
    # 确认开发板也是0哦,表示首选

    # 强制设置为 640x480 (宽x高)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 输出当前画幅设置
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"当前摄像头分辨率: {actual_width} x {actual_height}")

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    print("摄像头运行中\n按下Q退出")
    UnableToSendData = False
    ser = None

    try:
        ser = serial.Serial("COM5", 115200, timeout=0.1)
        # 防止串口连接时重置/挂起 STM32
        ser.setRTS(False)
        ser.setDTR(False)

        print("串口 COM5 连接成功")
    except serial.SerialException as e:
        print(f"串口连接失败: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return

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

            if (ser is not None and UnableToSendData==False):
                try:
                    SendDataToStm32(error, size,ser)
                except Exception as e:
                    print("无法发送数据至单片机\n")
                    UnableToSendData = True

            dead_zone = 50
            # 决定中间的范围有多大

            command="error "+str(error)+"  "+"size "+str(size)

            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            # 用法是这样：
            # Parameters
            # img	Image.
            # pt1	Vertex of the rectangle.
            # pt2	Vertex of the rectangle opposite to pt1 .
            # color	Rectangle color or brightness (grayscale image).
            # thickness	Thickness of lines that make up the rectangle. Negative values, like FILLED, mean that the function has to draw a filled rectangle.
            # lineType	Type of the line. See LineTypes
            # shift	Number of fractional bits in the point coordinates.

        elif len(results[0].boxes) > 1:
            command = "stop"

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
        cv2.imshow('YOLOv11 PC Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    start_recognition()

if __name__ == "__main__":
    main()

