# Camera Tutor — 部署实施指南

> 从零到 Emma 开口说话，全程操作手册。
> 两条路径：MacBook（云端 API，10 分钟跑通）| Orin（全本地，2-4 小时）。
> 预计首次启动：MacBook 10 分钟 | Orin 2-4 小时。

---

## 0. 选择你的路径

```
你有 MacBook Pro + Brio 100 + Poly Sync 20？
  → 走 §A: MacBook 快速启动（10 分钟，云端 API，几乎免费）

你有 x86 台式机 + NVIDIA GPU (RTX 3060 12GB+)？
  → 走 §C: 桌面 GPU 本地部署（30 分钟，全本地，零云端依赖）

你有 Jetson AGX Orin + 全套外设？
  → 走 §B: Orin 全本地部署（2-4 小时，数据不出门）
```

---

## A. MacBook 快速启动（推荐先走这条）

> 适用：MacBook Pro + USB 摄像头 + USB 麦克风/音箱
> 推理：云端 DashScope API（Qwen-Omni）
> 时间：10 分钟
> 费用：新用户免费额度，够测一个月

### A.1 插上设备

```
MacBook USB-C ←── Logitech Brio 100（或 USB-A 转接头）
MacBook USB-C ←── Poly Sync 20
```

### A.2 安装依赖

```bash
cd ~/workspace/english-tutor

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# Mac 需要 portaudio（PyAudio 的底层 C 库，用来操作麦克风和音箱）
# Linux 跳过这步
brew install portaudio

# 一条命令装完所有 Python 依赖
pip install -r requirements.txt
```

> **portaudio 是什么？** PyAudio 需要它。装一次就行，之后 pip 就能编译 pyaudio。Linux 上 `sudo apt install portaudio19-dev` 代替 brew。

### A.3 验证设备

```bash
# 摄像头
python3 -c "import cv2; cap=cv2.VideoCapture(0); ret,_=cap.read(); print('✅ CAM OK' if ret else '❌ CAM FAIL')"

# 麦克风
python3 -c "import pyaudio; p=pyaudio.PyAudio(); print(p.get_default_input_device_info()['name'])"

# 音箱 (Mac)
python3 -c "import os; os.system('afplay /System/Library/Sounds/Ping.aiff')"
# 音箱 (Linux): speaker-test -t sine -f 440 -l 1
```

> **多设备 / 设备被占用？** 用 `--select-devices` 启动时菜单式选择麦克风、扬声器、摄像头，选择会自动保存复用。详见 README「[设备选择（麦克风 / 扬声器 / 摄像头）](#设备选择麦克风--扬声器--摄像头)」小节。

三条都绿 → 进入下一步。

### A.4 申请 API Key（免费）

1. 打开 [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com)
2. 注册/登录 → 开通百炼 → 获取 API Key
3. 新用户送几十万 token 免费额度，够开发测试一个月

### A.5 配置 .env

```bash
cd ~/workspace/english-tutor
cat > .env << 'EOF'
DASHSCOPE_API_KEY=你的API-Key
LLM_BASE_URL=https://你的实例ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
EOF
```

### A.6 跑 Mock 验证逻辑

```bash
python3 camera_tutor/demo.py --mock
# 输入 hello、stop 等测试决策引擎
# 确认 5 个场景都走通
```

### A.7 跑场景验证

```bash
python3 camera_tutor/scenario_demos.py
# 预期: 6/6 scenarios passed ✅
```

### A.8 云端模式跑完整 Demo

```bash
# 打开摄像头 + 麦克风，Emma 通过云端 API 对话
python3 camera_tutor/demo.py
```

### Mac 上不能做的（留给 Orin）

```
❌ 本地跑 Qwen-Omni 7B（没 CUDA）
❌ 离线使用（必须联网）
❌ Live2D C++ 渲染（可以编译但没 Orin GPU 快）

✅ 其他所有功能：摄像头、VAD、决策引擎、对话、报告、Dashboard 都在 Mac 上能跑
```

---

## B. Orin 全本地部署

> 适用：Jetson AGX Orin 64GB + 全套外设
> 推理：本地 Qwen2.5-Omni-7B
> 时间：2-4 小时

### B.0 物料清单

| # | 设备 | 型号 | 数量 |
|---|------|------|------|
| 1 | 计算平台 | Jetson AGX Orin 64GB Developer Kit | 1 |
| 2 | NVMe SSD | 1TB M.2 2280 (Samsung 980 / WD SN570) | 1 |
| 3 | 摄像头 | Logitech Brio 100 | 1 |
| 4 | 麦克风+音箱 | Poly Sync 20 | 1 |
| 5 | HDMI 屏幕 + 键鼠 | 开发调试用（任何型号） | 1 |

---

### B.1 硬件组装

### B.1.1 安装 NVMe SSD

```bash
# 1. 断开电源
# 2. 打开 Orin 底部盖板（两颗螺丝）
# 3. 将 SSD 斜插入 M.2 插槽，按下，拧紧固定螺丝
# 4. 盖回盖板
# 5. 接通电源，开机
```

### B.1.2 格式化 SSD

```bash
# 首次开机后，查看 SSD 是否被识别
lsblk
# 应看到 nvme0n1

# 格式化
sudo mkfs.ext4 /dev/nvme0n1

# 挂载
sudo mkdir /mnt/ssd
sudo mount /dev/nvme0n1 /mnt/ssd

# 永久挂载
echo '/dev/nvme0n1 /mnt/ssd ext4 defaults 0 2' | sudo tee -a /etc/fstab

# 设置用户权限
sudo chown $USER:$USER /mnt/ssd
```

### B.1.3 连接外设

```
Orin USB-C(左)  ←── 电源线
Orin USB 3.0(蓝) ←── Logitech Brio 100
Orin USB 3.0(蓝) ←── Poly Sync 20
Orin HDMI       ←── 调试屏幕
Orin USB        ←── 键鼠（调试用）
```

### B.1.4 验证连接

```bash
# 摄像头
ls /dev/video*                    # 应至少显示 /dev/video0
python3 -c "import cv2; cap=cv2.VideoCapture(0); ret,_=cap.read(); print('CAM OK' if ret else 'CAM FAIL')"

# 麦克风（Poly Sync 20）
arecord -l | grep -i poly         # 应有 Poly 设备
python3 -c "import pyaudio; p=pyaudio.PyAudio(); print(p.get_default_input_device_info()['name'])"

# 音箱
speaker-test -t sine -f 440 -l 1  # 应听到"嘀"一声

# SSD
df -h /mnt/ssd                     # 应显示 ~1TB 可用
```

---

### B.2 系统环境配置

### B.2.1 性能模式

```bash
# 设置最大性能
sudo nvpmodel -m 0
sudo jetson_clocks

# 增加 Swap（应对峰值内存）
sudo fallocate -l 16G /mnt/ssd/swapfile
sudo chmod 600 /mnt/ssd/swapfile
sudo mkswap /mnt/ssd/swapfile
sudo swapon /mnt/ssd/swapfile
echo '/mnt/ssd/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### B.2.2 基础依赖

```bash
sudo apt update && sudo apt install -y \
    python3-pip python3-venv python3-dev \
    libopenblas-dev cmake build-essential \
    libportaudio2 portaudio19-dev \
    libsndfile1 ffmpeg \
    git

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装 Python 依赖
cd ~/camera-tutor  # 或你的项目目录
pip install -r requirements.txt
pip install fastapi uvicorn httpx python-dotenv
pip install pyaudio numpy opencv-python
```

### B.2.3 PyTorch for Jetson

```bash
# 确认 JetPack 版本
cat /etc/nv_tegra_release
# R36.x → JetPack 6.x → PyTorch 2.4+

# 从 NVIDIA 官方下载预编译 PyTorch
# 参考: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

# 示例（JetPack 6.1）:
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.4.0-cp310-cp310-linux_aarch64.whl
pip install torch-2.4.0-cp310-cp310-linux_aarch64.whl

# 验证
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# 应显示: CUDA: True
```

---

### B.3 部署 Qwen2.5-Omni-7B（量化）

> Orin 64GB 可选 FP16（原始精度）或 INT4 量化。量化后推理速度提升 2-3x，
> VRAM 占用从 ~19GB 降至 ~8GB，对低延迟场景至关重要。

### B.3.1 选择量化级别

| 量化 | 模型大小 | 推理速度 | 英语质量 | 适合 |
|------|---------|---------|---------|------|
| **FP16（无量化）** | ~19 GB | 基准 | 最佳 | AGX Orin 64GB 开发调优 |
| **Q8_0** | ~13 GB | 1.5x | 几乎无损 | AGX Orin 64GB 生产 |
| **Q4_K_M（推荐）** | ~8.2 GB | 2-3x | <1% 损失 | Orin NX 16GB / AGX Orin 低功耗 |
| **IQ3_M** | ~7.0 GB | 2.5-3.5x | ~2-3% 损失 | Orin Nano 8GB（极限） |

### B.3.2 下载模型

```bash
# 安装 modelscope（国内下载更快）
pip install modelscope

# 下载到 SSD（模型约 15GB）
modelscope download --model Qwen/Qwen2.5-Omni-7B \
    --local_dir /mnt/ssd/models/Qwen2.5-Omni-7B

# 或从 HuggingFace
# pip install huggingface_hub
# huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir /mnt/ssd/models/Qwen2.5-Omni-7B

# 验证
ls -lh /mnt/ssd/models/Qwen2.5-Omni-7B/
# 应有: config.json, model.safetensors, tokenizer.json 等
```

### B.3.3 量化模型

**方式 A：使用项目量化脚本（推荐）**

```bash
# INT4 量化（产出一个 .gguf 文件）
python3 scripts/quantize_omni.py \
    --model_path /mnt/ssd/models/Qwen2.5-Omni-7B \
    --output_path /mnt/ssd/models/Qwen2.5-Omni-7B-Q4_K_M.gguf \
    --bits 4

# 验证大小
ls -lh /mnt/ssd/models/Qwen2.5-Omni-7B-Q4_K_M.gguf
# 预期: ~6.2 GB
```

**方式 B：手动用 llama.cpp convert + quantize**

```bash
# 编译 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# 转换 HF 模型 → GGUF FP16
python3 convert_hf_to_gguf.py \
    /mnt/ssd/models/Qwen2.5-Omni-7B \
    --outtype f16 \
    --outfile /mnt/ssd/models/Qwen2.5-Omni-7B-FP16.gguf

# 量化 FP16 → Q4_K_M
./build/bin/llama-quantize \
    /mnt/ssd/models/Qwen2.5-Omni-7B-FP16.gguf \
    /mnt/ssd/models/Qwen2.5-Omni-7B-Q4_K_M.gguf \
    Q4_K_M
```

### B.3.4 基础加载测试

```python
# test_load.py — Transformers 加载（FP16，非量化）
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
import torch, time

MODEL_PATH = "/mnt/ssd/models/Qwen2.5-Omni-7B"

print("Loading...")
t0 = time.time()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
model = Qwen2_5OmniModel.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
print(f"Loaded in {time.time()-t0:.1f}s")

# 文本推理
inputs = processor(text="Describe a red car in one simple sentence.", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(outputs[0]))
```

```bash
python3 test_load.py
# 预期: 30s 内加载完成，输出一句英文描述

# 量化模型测试（llama.cpp）
./build/bin/llama-cli \
    -m /mnt/ssd/models/Qwen2.5-Omni-7B-Q4_K_M.gguf \
    --mmproj /mnt/ssd/models/qwen2.5-omni-vision.gguf \
    -p "Describe a red car in one simple sentence." \
    -ngl 99 \
    -c 4096 \
    -n 50
# 预期: 3-5s 内输出一句描述（Jetson AGX Orin）
```

### B.3.5 启动本地推理服务

```bash
# 方式 A：项目自带的 omni_server.py（基于 Transformers，FP16 推理）
python3 camera_tutor/omni_server.py &

# 方式 B：llama.cpp server（量化模型，性能更优，推荐）
./build/bin/llama-server \
    -m /mnt/ssd/models/Qwen2.5-Omni-7B-Q4_K_M.gguf \
    --mmproj /mnt/ssd/models/qwen2.5-omni-vision.gguf \
    --host 0.0.0.0 --port 8100 \
    -ngl 99 \
    -c 4096 \
    --mlock &

# 验证
curl http://localhost:8100/api/health
# 应返回: {"status":"ok"}
```

> **性能提示**：`-ngl 99` 将所有层加载到 GPU（AGX Orin 64GB UMA）。
> Orin NX 16GB 推荐 `-ngl 40`，`--mlock` 锁定内存防止 swap。

---

### B.4 安装 Force Aligner（嘴型同步）

```bash
# 安装 MFA
conda install -c conda-forge montreal-forced-aligner

# 下载英语声学模型
mfa model download acoustic english_mfa

# 下载英语发音词典
mfa model download dictionary english_mfa

# 验证
mfa version
# 应显示版本号
```

---

### B.5 部署 Camera Tutor 项目

### B.5.1 拉取代码

```bash
cd ~
git clone <你的仓库地址> camera-tutor
cd camera-tutor
```

### B.5.2 配置

```bash
# 创建 .env 文件
cat > .env << 'EOF'
OMNI_LOCAL_URL=http://localhost:8100
DASHSCOPE_API_KEY=your_key_here  # 云端模式需要，本地模式可跳过
CAMERA_TUTOR_DATA_DIR=/mnt/ssd/camera-tutor-data
LIVE2D_MODEL_PATH=/mnt/ssd/models/live2d/Haru.moc3
EOF

# 创建数据目录
mkdir -p /mnt/ssd/camera-tutor-data
```

### B.5.3 安装项目依赖

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install mediapipe  # 手势追踪（Pro 版）
```

---

### B.6 首次运行验证

### B.6.1 Mock 模式（不需要模型，验证逻辑）

```bash
python3 camera_tutor/demo.py --mock
```

键入一些对话内容，确认决策引擎、对话流正常。

### B.6.2 场景验证

```bash
python3 camera_tutor/scenario_demos.py
# 预期: 6/6 scenarios passed
```

### B.6.3 硬件模式（需要 Qwen-Omni 已启动）

```bash
# 确认服务可用
curl http://localhost:8100/api/health

# 启动完整 Demo
python3 camera_tutor/demo.py
```

---

### B.7 Live2D 安装（可选，Mid/Pro 版）

```bash
# 一键构建
chmod +x scripts/build_live2d_linux.sh
./scripts/build_live2d_linux.sh

# 验证
./build/live2d_renderer --help
```

---

### B.8 家长 Dashboard 启动

```bash
python3 camera_tutor/dashboard_server.py &

# 从浏览器访问
# http://<Orin IP>:8200
```

---

### B.9 常见问题

### Q: 摄像头打不开
```bash
# 检查 UVC 驱动
lsusb | grep -i logitech
# 看看是否有其他程序占用
sudo fuser /dev/video0
```

### Q: 麦克风没声音
```bash
# 检查 Poly Sync 20 是否被识别
arecord -l
# 设置默认设备
export ALSA_INPUT_DEVICE=plughw:2,0  # 根据 arecord -l 的输出调
```

### Q: CUDA 不可用
```bash
# 重新安装 JetPack 对应版本的 PyTorch
# 确认 JetPack 版本
cat /etc/nv_tegra_release
# 去 NVIDIA 官方论坛找到对应的 PyTorch wheel
```

### Q: 模型加载 OOM
```bash
# 启用 INT4 量化
python3 scripts/quantize_omni.py \
    --model_path /mnt/ssd/models/Qwen2.5-Omni-7B \
    --output_path /mnt/ssd/models/Qwen2.5-Omni-7B-INT4 \
    --bits 4
```

---

### B.10 快速检查清单

```
[ ] NVMe SSD 挂载到 /mnt/ssd，可用 >800GB
[ ] 摄像头 cv2.VideoCapture(0) 可读帧
[ ] Poly Sync 20 麦克风可录音、音箱可播放
[ ] nvcc --version 显示 CUDA 12.x
[ ] torch.cuda.is_available() == True
[ ] Qwen2.5-Omni-7B 加载成功
[ ] demo.py --mock 通过
[ ] scenario_demos.py 6/6 通过
[ ] curl localhost:8100/api/health 返回 ok
[ ] demo.py（硬件模式）Emma 说出第一句话
```

---

> 全部绿色打勾 → Camera Tutor 已就绪。下一步：家庭 Alpha 测试（参见 `USER_TEST_PLAN.md`）。

---

## C. x86 桌面 GPU 本地部署（推荐开发阶段）

> 适用：x86 台式机 + NVIDIA GPU (RTX 3060 12GB+, RTX 4060 Ti 16GB+)
> 推理：本地 Qwen2.5-Omni-7B Q4_K_M 量化，llama.cpp server
> 时间：~30 分钟 | 费用：电费 ~¥15-30/月
> 优势：完全本地，零云端，儿童数据不出门

### C.1 硬件要求

| 最低配置 | 推荐配置 |
|---------|---------|
| RTX 3060 12GB | RTX 4060 Ti 16GB |
| 16 GB 系统内存 | 32 GB 系统内存 |
| Ubuntu 22.04 / Arch Linux | Ubuntu 22.04 |
| USB 摄像头 + USB 麦克风/音箱 | Logitech Brio 100 + Poly Sync 20 |

### C.2 安装 NVIDIA 驱动与 CUDA

```bash
# Ubuntu 22.04
sudo apt update
sudo apt install -y nvidia-driver-535 cuda-toolkit-12-4
sudo reboot

# 验证
nvidia-smi
# 应显示: Driver 535.x, CUDA 12.4, VRAM 12288MiB (3060) / 16384MiB (4060 Ti)
```

### C.3 部署 Qwen2.5-Omni-7B 量化模型

```bash
# 1. 下载原始模型
pip install modelscope
modelscope download --model Qwen/Qwen2.5-Omni-7B \
    --local_dir ~/models/Qwen2.5-Omni-7B

# 2. 编译 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# 3. 转换 HF → GGUF FP16
python3 convert_hf_to_gguf.py ~/models/Qwen2.5-Omni-7B \
    --outtype f16 --outfile ~/models/Qwen2.5-Omni-7B-FP16.gguf

# 4. 量化 FP16 → Q4_K_M
./build/bin/llama-quantize \
    ~/models/Qwen2.5-Omni-7B-FP16.gguf \
    ~/models/Qwen2.5-Omni-7B-Q4_K_M.gguf \
    Q4_K_M

# 验证：~6.2 GB
ls -lh ~/models/Qwen2.5-Omni-7B-Q4_K_M.gguf
```

### C.4 启动推理服务

```bash
./build/bin/llama-server \
    -m ~/models/Qwen2.5-Omni-7B-Q4_K_M.gguf \
    --mmproj ~/models/qwen2.5-omni-vision.gguf \
    --host 0.0.0.0 --port 8100 \
    -ngl 99 \
    -c 4096 \
    --mlock &

# 验证
curl http://localhost:8100/health

# 查看 VRAM
nvidia-smi
# 预期: ~8.5-9.0 GB VRAM used (模型 8.2GB + overhead)
# 剩余 ~3 GB，可同时跑 Kokoro TTS 或 Live2D 渲染
```

### C.5 配置与安装

```bash
# .env — 纯本地模式
cat > .env << 'EOF'
OMNI_LOCAL_URL=http://localhost:8100
DASHSCOPE_API_KEY=
CAMERA_TUTOR_DATA_DIR=~/.camera-tutor-data
EOF

# Python 依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### C.6 验证运行

```bash
python3 camera_tutor/demo.py --mock     # 逻辑验证
python3 camera_tutor/scenario_demos.py  # 场景验证（预期 6/6）
python3 camera_tutor/demo.py            # 完整 Demo
```

### C.7 桌面 GPU vs Jetson Orin

| 维度 | RTX 3060 12GB | Jetson AGX Orin 64GB |
|------|:--:|:--:|
| **部署时间** | ~30 分钟 | 2-4 小时 |
| **推理速度** | ⭐⭐⭐ 更快 | ⭐⭐ 低功耗 ARM |
| **功耗** | 170W | 15-60W |
| **VRAM** | 12GB（必须量化） | 64GB（可跑 FP16） |
| **噪音/体积** | 风扇 + 大机箱 | 几乎无声 + 巴掌大 |
| **适合** | **开发迭代首选** | **量产部署目标** |

> **建议**：开发阶段用 RTX 3060 快速迭代，确认效果后移植 Jetson Orin。两套环境共用 `omni_client.py`（改 `OMNI_LOCAL_URL` 即可切换）。

### C.8 一键启动脚本

```bash
#!/bin/bash
# start-camera-tutor-desktop.sh

set -e
LLAMA_CPP_DIR="$HOME/llama.cpp"
MODEL="$HOME/models/Qwen2.5-Omni-7B-Q4_K_M.gguf"
MMPROJ="$HOME/models/qwen2.5-omni-vision.gguf"

echo "🔍 检查 GPU..."
nvidia-smi | head -n 1 || { echo "❌ 未检测到 NVIDIA GPU"; exit 1; }

echo "🚀 启动 llama.cpp server..."
cd "$LLAMA_CPP_DIR"
./build/bin/llama-server -m "$MODEL" --mmproj "$MMPROJ" \
    --host 0.0.0.0 --port 8100 -ngl 99 -c 4096 --mlock &
SERVER_PID=$!

echo "⏳ 等待模型加载..."
for i in $(seq 1 30); do
    curl -s http://localhost:8100/health > /dev/null 2>&1 && break
    sleep 2
done
echo "✅ 推理服务就绪"

echo "🎤 启动 Camera Tutor..."
source .venv/bin/activate
cd ~/workspace/english-tutor
python3 camera_tutor/demo.py

kill $SERVER_PID 2>/dev/null
echo "👋 已关闭"
```

```bash
chmod +x start-camera-tutor-desktop.sh
./start-camera-tutor-desktop.sh
```
