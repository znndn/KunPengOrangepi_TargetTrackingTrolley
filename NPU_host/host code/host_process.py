import ctypes
import os
import struct
import threading
import time

import acl
import cv2
import numpy as np
import serial

import host_send


CAMERA_DEVICE_PATH = "/dev/v4l/by-id/usb-ZC_USB_Camera_200901010001-video-index0"
SERIAL_DEVICE_PATH = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MODEL_PATH = "/root/agv_project/NPU_host/yolov8s_310B4.om"
MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 640
CLASS_ID = 67
CONF_THRESHOLD = 0.15
NMS_IOU_THRESHOLD = 0.45
DEAD_ZONE = 50


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


def _check_acl(ret, message):
    if ret != 0:
        raise RuntimeError(f"{message} failed, ret={ret}")


def _acl_dtype_to_numpy(acl_dtype):
    dtype_map = {
        acl.ACL_FLOAT: np.float32,
        acl.ACL_FLOAT16: np.float16,
        acl.ACL_INT8: np.int8,
        acl.ACL_INT16: np.int16,
        acl.ACL_INT32: np.int32,
        acl.ACL_UINT8: np.uint8,
        acl.ACL_UINT16: np.uint16,
        acl.ACL_UINT32: np.uint32,
        acl.ACL_INT64: np.int64,
        acl.ACL_UINT64: np.uint64,
    }
    if acl_dtype not in dtype_map:
        raise ValueError(f"Unsupported ACL dtype: {acl_dtype}")
    return dtype_map[acl_dtype]


class AscendOmRunner:
    def __init__(self, model_path, device_id=0):
        self.model_path = model_path
        self.device_id = device_id
        self._context = None
        self._stream = None
        self._model_id = None
        self._model_desc = None
        self._input_dataset = None
        self._output_dataset = None
        self._input_buffer = None
        self._output_buffer = None
        self._input_size = 0
        self._output_size = 0
        self._output_dims = None
        self._output_dtype = None
        self._input_data_buffer = None
        self._output_data_buffer = None
        self._initialized = False

    def init(self):
        _check_acl(acl.init(), "acl.init")
        _check_acl(acl.rt.set_device(self.device_id), "acl.rt.set_device")
        self._context, ret = acl.rt.create_context(self.device_id)
        _check_acl(ret, "acl.rt.create_context")
        self._stream, ret = acl.rt.create_stream()
        _check_acl(ret, "acl.rt.create_stream")
        self._model_id, ret = acl.mdl.load_from_file(self.model_path)
        _check_acl(ret, "acl.mdl.load_from_file")
        self._model_desc = acl.mdl.create_desc()
        _check_acl(acl.mdl.get_desc(self._model_desc, self._model_id), "acl.mdl.get_desc")

        self._input_size = acl.mdl.get_input_size_by_index(self._model_desc, 0)
        self._output_size = acl.mdl.get_output_size_by_index(self._model_desc, 0)
        self._output_dims = acl.mdl.get_output_dims(self._model_desc, 0)
        self._output_dtype = acl.mdl.get_output_data_type(self._model_desc, 0)

        self._input_buffer, ret = acl.rt.malloc(self._input_size, acl.MEM_MALLOC_HUGE_FIRST)
        _check_acl(ret, "acl.rt.malloc(input)")
        self._output_buffer, ret = acl.rt.malloc(self._output_size, acl.MEM_MALLOC_HUGE_FIRST)
        _check_acl(ret, "acl.rt.malloc(output)")

        self._input_dataset = acl.mdl.create_dataset()
        self._output_dataset = acl.mdl.create_dataset()

        self._input_data_buffer = acl.create_data_buffer(self._input_buffer, self._input_size)
        _check_acl(acl.mdl.add_dataset_buffer(self._input_dataset, self._input_data_buffer), "acl.mdl.add_dataset_buffer(input)")

        self._output_data_buffer = acl.create_data_buffer(self._output_buffer, self._output_size)
        _check_acl(acl.mdl.add_dataset_buffer(self._output_dataset, self._output_data_buffer), "acl.mdl.add_dataset_buffer(output)")
        self._initialized = True

    def execute(self, input_tensor):
        if not self._initialized:
            raise RuntimeError("AscendOmRunner has not been initialized")
        if input_tensor.nbytes != self._input_size:
            raise ValueError(f"Input tensor size mismatch: {input_tensor.nbytes} vs {self._input_size}")

        input_ptr = acl.util.numpy_to_ptr(input_tensor)
        _check_acl(
            acl.rt.memcpy(self._input_buffer, self._input_size, input_ptr, self._input_size, acl.MEMCPY_HOST_TO_DEVICE),
            "acl.rt.memcpy(input)"
        )

        _check_acl(
            acl.mdl.execute_async(self._model_id, self._input_dataset, self._output_dataset, self._stream),
            "acl.mdl.execute_async"
        )
        _check_acl(acl.rt.synchronize_stream(self._stream), "acl.rt.synchronize_stream")

        host_ptr, ret = acl.rt.malloc_host(self._output_size)
        _check_acl(ret, "acl.rt.malloc_host(output)")
        _check_acl(
            acl.rt.memcpy(host_ptr, self._output_size, self._output_buffer, self._output_size, acl.MEMCPY_DEVICE_TO_HOST),
            "acl.rt.memcpy(output)"
        )
        output_bytes = ctypes.string_at(host_ptr, self._output_size)
        _check_acl(acl.rt.free_host(host_ptr), "acl.rt.free_host")
        output_shape = tuple(self._output_dims.get("dims", []))
        if not output_shape:
            raise ValueError("Output shape is empty; check model output dims")
        output = np.frombuffer(output_bytes, dtype=_acl_dtype_to_numpy(self._output_dtype)).reshape(output_shape)
        return output

    def close(self):
        if not self._initialized:
            return
        if self._output_dataset is not None:
            acl.mdl.destroy_dataset(self._output_dataset)
            self._output_dataset = None
        if self._input_dataset is not None:
            acl.mdl.destroy_dataset(self._input_dataset)
            self._input_dataset = None
        if self._output_data_buffer is not None:
            acl.destroy_data_buffer(self._output_data_buffer)
            self._output_data_buffer = None
        if self._input_data_buffer is not None:
            acl.destroy_data_buffer(self._input_data_buffer)
            self._input_data_buffer = None
        if self._output_buffer is not None:
            acl.rt.free(self._output_buffer)
            self._output_buffer = None
        if self._input_buffer is not None:
            acl.rt.free(self._input_buffer)
            self._input_buffer = None
        if self._model_desc is not None and self._model_id is not None:
            acl.mdl.destroy_desc(self._model_desc)
            self._model_desc = None
        if self._model_id is not None:
            acl.mdl.unload(self._model_id)
            self._model_id = None
        if self._stream is not None:
            acl.rt.destroy_stream(self._stream)
            self._stream = None
        if self._context is not None:
            acl.rt.destroy_context(self._context)
            self._context = None
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        self._initialized = False

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


def _letterbox(image, new_shape=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH), color=(114, 114, 114)):
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def _preprocess(frame):
    resized, ratio, (dw, dh) = _letterbox(frame)
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img), ratio, (dw, dh)


def _nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def _postprocess(output, ratio, pad, original_shape):
    predictions = output[0].transpose(1, 0)
    scores = predictions[:, 4 + CLASS_ID]
    mask = scores >= CONF_THRESHOLD
    if not np.any(mask):
        return []
    filtered_boxes = predictions[mask, :4]
    filtered_scores = scores[mask]

    xyxy = np.zeros_like(filtered_boxes)
    xyxy[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
    xyxy[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
    xyxy[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
    xyxy[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2

    dw, dh = pad
    xyxy[:, [0, 2]] -= dw
    xyxy[:, [1, 3]] -= dh
    xyxy /= ratio

    height, width = original_shape[:2]
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, width - 1)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, width - 1)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, height - 1)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, height - 1)

    keep = _nms(xyxy, filtered_scores, NMS_IOU_THRESHOLD)
    return xyxy[keep]


def _send_stop_packet(serial_handle):
    if serial_handle is None:
        return
    packet = struct.pack('<BBhIB', 0xB3, 0x00, 0, 0, 0x5B)
    serial_handle.write(packet)


def start_recognition():
    if not os.path.exists(MODEL_PATH):
        print(f"模型文件 {MODEL_PATH} 不存在，已终止")
        return

    model = AscendOmRunner(MODEL_PATH)
    model.init()
    print("当前运行设备: Ascend NPU (OM)")
    cap = _open_camera()
    if cap is None:
        model.close()
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

            input_tensor, ratio, pad = _preprocess(frame)
            output = model.execute(input_tensor)
            boxes = _postprocess(output, ratio, pad, frame.shape)

            process_count+=1

            log_message = "未检测到物品"
            if len(boxes) == 1:
                xyxy = boxes[0]
                object_center_x = (xyxy[0] + xyxy[2]) / 2.0

                error = object_center_x - frame_center_x
                # 负数偏左，正数偏右

                size = abs(xyxy[0]-xyxy[2]) * abs(xyxy[1]-xyxy[3])
                # 用来确定距离

                # 保留死区：当偏差落在 dead_zone 内时不触发 PID 纠偏，向下位机发送 0 纠偏量
                pid_error = 0 if abs(error) <= DEAD_ZONE else error

                if (ser is not None and UnableToSendData is False):
                    try:
                        host_send.SendDataToStm32(pid_error, size, ser)
                    except Exception as e:
                        print("无法发送数据至单片机\n")
                        UnableToSendData = True

                log_message = f"物品检测: x_offset={pid_error:.2f}, size={size:.2f}"

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

            elif len(boxes) > 1:
                if (ser is not None and UnableToSendData is False):
                    try:
                        _send_stop_packet(ser)
                    except Exception as e:
                        print("无法发送数据至单片机\n")
                        UnableToSendData = True

                log_message = "检测到多个物品，暂停控制"
            else:
                if (ser is not None and UnableToSendData is False):
                    try:
                        _send_stop_packet(ser)
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
        model.close()
        if ser is not None and ser.is_open:
            ser.close()

def main():
    start_recognition()

if __name__ == "__main__":
        main()
