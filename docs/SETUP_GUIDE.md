# Camera Tutor — Jetson Orin 系统环境搭建指南

> **目标平台**: NVIDIA Jetson AGX Orin 64GB Developer Kit
> **系统镜像**: JetPack 6.1 (Ubuntu 22.04 LTS + CUDA 12.4 + TensorRT 10.x)
> **目标**: 完成从裸机到 Qwen2.5-Omni-7B 可运行的完整环境

---

## 1. 初始系统烧录

### 1.1 准备

```bash
# 在 Ubuntu 宿主机上下载 NVIDIA SDK Manager
# https://developer.nvidia.com/sdk-manager
sudo apt install sdkmanager

# 或者直接下载 JetPack 镜像手动烧录（推荐，更可控）
# 下载地址: https://developer.nvidia.com/embedded/jetson-linux
```

### 1.2 烧录步骤

```bash
# 1. 进入 Force Recovery 模式
#    - 用跳线帽短接 FC_REC 和 GND (J14 Pin 9 & 10)
#    - 连接电源和 USB-C 到宿主机
#    - 设备应显示为 NVIDIA Corp. APX

# 2. 使用 SDK Manager 烧录
sdkmanager --cli

# 选择:
#   - Jetson AGX Orin
#   - JetPack 6.1
#   - 组件: CUDA + TensorRT + cuDNN + OpenCV + VPI

# 3. 完成初次启动设置
#    - 设置用户名/密码
#    - 连接 WiFi 或网线
#    - 运行系统更新
```

### 1.3 安装后配置

```bash
# 扩展存储（NVMe SSD）
sudo mkfs.ext4 /dev/nvme0n1
sudo mkdir /mnt/ssd
sudo mount /dev/nvme0n1 /mnt/ssd
echo '/dev/nvme0n1 /mnt/ssd ext4 defaults 0 2' | sudo tee -a /etc/fstab

# 设置功耗模式（最大化性能）
sudo nvpmodel -m 0    # Max-N mode
sudo jetson_clocks    # Lock clocks at maximum

# 增加 swap（处理峰值内存需求）
sudo fallocate -l 16G /mnt/ssd/swapfile
sudo chmod 600 /mnt/ssd/swapfile
sudo mkswap /mnt/ssd/swapfile
sudo swapon /mnt/ssd/swapfile

# 验证安装
nvcc --version            # CUDA 版本
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. Python 深度学习环境

### 2.1 基础依赖

```bash
# 系统包
sudo apt update
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    libopenblas-dev libopenmpi-dev \
    libportaudio2 portaudio19-dev \
    libsndfile1 ffmpeg \
    cmake build-essential

# 创建虚拟环境
python3 -m venv ~/camera-tutor-env
source ~/camera-tutor-env/bin/activate
```

### 2.2 PyTorch for Jetson

```bash
# 安装 Jetson 专用 PyTorch wheel
# 从 NVIDIA 官方下载预编译版本
# 参考: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

# JetPack 6.1 对应 PyTorch 2.4+
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.4.0a0+3b8965e5e6.nv24.08.14519422-cp310-cp310-linux_aarch64.whl
pip install torch-2.4.0a0+3b8965e5e6.nv24.08.14519422-cp310-cp310-linux_aarch64.whl

# TorchVision
pip install torchvision --no-deps
# 从 NVIDIA 下载对应版本的预编译 torchvision wheel
```

### 2.3 Transformers + 推理加速

```bash
# HuggingFace Transformers (支持 Qwen-Omni)
pip install transformers>=4.45.0
pip install accelerate>=0.30.0

# TensorRT-LLM (可选，用于极致推理优化)
# 从 NVIDIA 官方 GitHub 编译
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
# 按官方文档编译 for aarch64
# 注意: TensorRT-LLM 对 Qwen-Omni 的支持可能在最新版本中

# 量化工具
pip install auto-gptq  # GPTQ 量化
pip install bitsandbytes  # 8-bit 量化
```

### 2.4 视觉与音频依赖

```bash
# OpenCV (JetPack 自带，验证即可)
python3 -c "import cv2; print(cv2.__version__)"

# 音频处理
pip install pyaudio sounddevice numpy
pip install silero-vad  # 语音活动检测

# HTTP & API
pip install httpx aiohttp fastapi uvicorn[standard]

# 视频流处理
pip install av  # PyAV

# YOLO / 轻量检测 (可选，用于快速目标检测预筛选)
# pip install ultralytics  # YOLOv8 on Jetson
```

---

## 3. 部署 Qwen2.5-Omni-7B

### 3.1 模型下载

```bash
# 方式 1: HuggingFace (推荐)
pip install modelscope  # 国内下载速度更快
modelscope download --model Qwen/Qwen2.5-Omni-7B \
    --local_dir ~/models/Qwen2.5-Omni-7B

# 方式 2: HuggingFace CLI
huggingface-cli download Qwen/Qwen2.5-Omni-7B \
    --local-dir ~/models/Qwen2.5-Omni-7B

# 验证下载
ls -lh ~/models/Qwen2.5-Omni-7B/
# 预期大小: ~15GB (FP16)
```

### 3.2 基础推理测试

```python
# test_omni_load.py — 验证模型能成功加载并推理

from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
import torch
import time

MODEL_PATH = "/home/camera-tutor/models/Qwen2.5-Omni-7B"

print("Loading processor...")
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

print("Loading model (FP16)...")
start = time.time()
model = Qwen2_5OmniModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
print(f"Model loaded in {time.time() - start:.1f}s")

# 简单文本推理
inputs = processor(
    text="Describe what you see in English, in one simple sentence.",
    return_tensors="pt"
).to("cuda")

print("Running inference...")
start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
result = processor.decode(outputs[0])
print(f"Inference: {time.time() - start:.2f}s")
print(f"Output: {result}")
```

### 3.3 模型量化 (INT4)

```bash
# 使用 auto-gptq 量化到 INT4
# 目标: 显存 6-8GB, 推理速度 < 1s

python3 scripts/quantize_omni.py \
    --model_path ~/models/Qwen2.5-Omni-7B \
    --output_path ~/models/Qwen2.5-Omni-7B-INT4 \
    --bits 4 \
    --group_size 128
```

### 3.4 建立推理服务 (FastAPI)

```python
# omni_server.py — 本地推理服务骨架

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
import torch
import base64
import io
from PIL import Image

app = FastAPI()

model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2_5OmniModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )

class VisionRequest(BaseModel):
    image_base64: str  # Base64 encoded JPEG frame
    audio_base64: str | None = None  # Optional audio input
    prompt: str = "What do you see?"

@app.post("/api/vision")
async def vision_endpoint(req: VisionRequest):
    # Decode image
    image_bytes = base64.b64decode(req.image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    # Build multimodal input
    inputs = processor(
        text=req.prompt,
        images=image,
        return_tensors="pt"
    ).to("cuda")

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    result = processor.decode(outputs[0])
    return {"text": result, "success": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
```

---

## 4. 外围设备配置

### 4.1 摄像头验证

```bash
# 确认摄像头被识别
lsusb | grep -i "webcam\|camera"
ls /dev/video*

# 测试捕获
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = cap.read()
print(f'Frame shape: {frame.shape}' if ret else 'Capture failed!')
cap.release()
"
```

### 4.2 麦克风阵列验证

```bash
# 确认 ReSpeaker 被识别
arecord -l
# 应显示: seeed-4mic-voicecard

# 录音测试
arecord -D plughw:2,0 -f S16_LE -r 16000 -c 4 -d 5 test.wav

# PyAudio 验证
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f'{i}: {info[\"name\"]} - {info[\"maxInputChannels\"]} channels')
"
```

### 4.3 音箱验证

```bash
# 测试播放
speaker-test -t sine -f 440 -l 1
aplay /usr/share/sounds/alsa/Front_Center.wav
```

---

## 5. 性能基线测试

```bash
# 运行性能基准测试
python3 scripts/benchmark.py --model ~/models/Qwen2.5-Omni-7B
```

关键性能指标:

| 指标 | 目标 (FP16) | 目标 (INT4) |
|------|-----------|-----------|
| 模型加载时间 | < 30s | < 20s |
| 文本推理 (50 tokens) | < 5s | < 2s |
| 视觉推理 (1 帧 + 50 tokens) | < 8s | < 3s |
| 视觉+音频推理 | < 10s | < 5s |
| GPU 显存占用 | < 20GB | < 10GB |
| 空闲功耗 | < 20W | < 15W |
| 满载功耗 | < 50W | < 35W |

---

## 6. 启动脚本

```bash
# ~/camera-tutor/start.sh

#!/bin/bash
source ~/camera-tutor-env/bin/activate

# 设置 GPU 高性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 启动服务
python3 omni_server.py &
echo "Omni Server: PID $!"

python3 camera_service.py &
echo "Camera Service: PID $!"

python3 audio_service.py &
echo "Audio Service: PID $!"

echo "=== Camera Tutor services started ==="
wait
```

---

> **下一步**: 硬件能力验证 — 实时视频流、远场拾音、清晰度、功耗测试
