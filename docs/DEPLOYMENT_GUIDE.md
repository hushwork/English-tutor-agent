# Camera Tutor — 部署实施指南

> 从零到 Emma 开口说话，全程操作手册。
> 两条路径：MacBook（云端 API，10 分钟跑通）| Orin（全本地，2-4 小时）。
> 预计首次启动：MacBook 10 分钟 | Orin 2-4 小时。

---

## 0. 选择你的路径

```
你有 MacBook Pro + Brio 100 + Poly Sync 20？
  → 走 §A: MacBook 快速启动（10 分钟，云端 API，几乎免费）

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
cd ~/workspace/english-tutor   # 或你 clone 代码的目录

# 创建虚拟环境
python3 -m venv ~/camera-tutor-env
source ~/camera-tutor-env/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install fastapi uvicorn httpx python-dotenv pyaudio numpy opencv-python

# Mac 上 PyAudio 可能需要 portaudio
# 如果报错: brew install portaudio && pip install pyaudio
```

### A.3 验证设备

```bash
# 摄像头 — 应显示 CAM OK
python3 -c "import cv2; cap=cv2.VideoCapture(0); ret,_=cap.read(); print('✅ CAM OK' if ret else '❌ CAM FAIL')"

# 麦克风 — 应显示 Poly Sync 20
python3 -c "import pyaudio; p=pyaudio.PyAudio(); print(p.get_default_input_device_info()['name'])"

# 音箱 — 应听到"嘀"一声
python3 -c "import os; os.system('afplay /System/Library/Sounds/Ping.aiff')"
```

三条都绿 → 进入下一步。

### A.4 申请 DashScope API Key（免费）

```bash
# 1. 打开 https://dashscope.console.aliyun.com
# 2. 注册/登录阿里云账号
# 3. 开通"模型服务灵积" → 获取 API Key
# 4. 新用户赠送几十万 token 免费额度
```

### A.5 配置环境变量

```bash
cd ~/workspace/english-tutor
cat > .env << 'EOF'
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
OMNI_LOCAL_URL=http://localhost:8100
CAMERA_TUTOR_DATA_DIR=~/.camera-tutor-data
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
python3 -m venv ~/camera-tutor-env
source ~/camera-tutor-env/bin/activate

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

### B.3 部署 Qwen2.5-Omni-7B

### B.3.1 下载模型

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

### B.3.2 基础加载测试

```python
# test_load.py
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
```

### B.3.3 启动本地推理服务

```bash
# 使用项目中的推理服务脚本
python3 camera_tutor/omni_server.py &
# 验证
curl http://localhost:8100/api/health
# 应返回: {"status":"ok"}
```

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
source ~/camera-tutor-env/bin/activate
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
