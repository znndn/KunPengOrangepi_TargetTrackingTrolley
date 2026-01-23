# NPU 侧运行说明（Ascend 310B4）

## 必须的环境准备

1. 激活 NPU 专用虚拟环境（禁止使用全局 Python）：

   ```bash
   source /root/agv_project/NPU_host/venv/bin/activate
   ```

2. 加载 CANN 环境变量：

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

## 模型导出与转换

### 导出 ONNX（固定参数）

```bash
cd /root/agv_project/NPU_host
python3 export_onnx.py
```

导出参数固定为：

- dynamic=False
- imgsz=640
- simplify=True
- opset=13

### ATC 转 OM（Ascend 310B4）

```bash
cd /root/agv_project/NPU_host
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc \
  --framework=5 \
  --model=yolov8s.onnx \
  --output=yolov8s_310B4 \
  --input_format=NCHW \
  --input_shape="images:1,3,640,640" \
  --soc_version=Ascend310B4 \
  --log=info \
  2>&1 | tee atc_yolov8s.log
```

成功标准：

- 输出包含 `ATC run success`
- 生成 `yolov8s_310B4.om`

## ATC 可用性验证流程

### 1. 基础检查

```bash
source /root/agv_project/NPU_host/venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
which atc
atc --version
echo "$ASCEND_OPP_PATH"
```

标准：

- `which atc` 指向 `/usr/local/Ascend/ascend-toolkit/latest/.../atc`
- `ASCEND_OPP_PATH` 指向 `/usr/local/Ascend/ascend-toolkit/latest/opp`

### 2. Python 依赖检查

```bash
python3 - <<'PY'
import decorator, attr, cloudpickle, tornado, synr, absl
print("python deps ok")
PY
```

### 3. 最小 ONNX 编译回归

```bash
python3 - <<'PY'
import onnx
from onnx import helper, TensorProto
X = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1,3,4,4])
Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1,3,4,4])
node = helper.make_node('Relu', ['x'], ['y'])
g = helper.make_graph([node], 'relu_graph', [X], [Y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
onnx.save(m, 'relu.onnx')
print("saved relu.onnx")
PY

atc \
  --framework=5 \
  --model=relu.onnx \
  --output=relu_310B4 \
  --input_format=NCHW \
  --input_shape="x:1,3,4,4" \
  --soc_version=Ascend310B4 \
  --log=info
```

标准：

- 输出包含 `ATC run success`
- 生成 `relu_310B4.om`

## 业务约束（不可变）

- 仅检测 COCO 类别 `class=67`，置信度阈值 `conf=0.15`。
- 每帧检测到 **恰好 1 个**目标：发送运动包；检测到 0 个或多于 1 个：发送停止包。
- `error = 目标框中心x - 画面中心x`，当 `abs(error) <= 50` 时下发 `x_offset=0`。
- 串口帧格式：

  - 运动包：`struct.pack('<BBhIB', 0xB3,0x01,x_offset,size,0x5B)`
  - 停止包：`struct.pack('<BBhIB', 0xB3,0x00,0,0,0x5B)`
