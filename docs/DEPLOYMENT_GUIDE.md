# Camera Tutor — 部署实施指南

> 从裸机 Orin 到 Emma 开口说话，全程操作手册。
> 适用人员：有 Linux 基础、可 SSH 或键盘操作设备。
> 预计全程：2-4 小时（含模型下载）。

---

## 0. 物料清单

| # | 设备 | 型号 | 数量 |
|---|------|------|------|
| 1 | 计算平台 | Jetson AGX Orin 64GB Developer Kit | 1 |
| 2 | NVMe SSD | 1TB M.2 2280 (Samsung 980 / WD SN570) | 1 |
| 3 | 摄像头 | Logitech Brio 100 | 1 |
| 4 | 麦克风+音箱 | Poly Sync 20 | 1 |
| 5 | HDMI 屏幕 + 键鼠 | 开发调试用（任何型号） | 1 |

---

## 1. 硬件组装

### 1.1 安装 NVMe SSD

```bash
# 1. 断开电源
# 2. 打开 Orin 底部盖板（两颗螺丝）
# 3. 将 SSD 斜插入 M.2 插槽，按下，拧紧固定螺丝
# 4. 盖回盖板
# 5. 接通电源，开机
```

### 1.2 格式化 SSD

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

### 1.3 连接外设

```
Orin USB-C(左)  ←── 电源线
Orin USB 3.0(蓝) ←── Logitech Brio 100
Orin USB 3.0(蓝) ←── Poly Sync 20
Orin HDMI       ←── 调试屏幕
Orin USB        ←── 键鼠（调试用）
```

### 1.4 验证连接

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

## 2. 系统环境配置

### 2.1 性能模式

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

### 2.2 基础依赖

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

### 2.3 PyTorch for Jetson

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

## 3. 部署 Qwen2.5-Omni-7B

### 3.1 下载模型

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

### 3.2 基础加载测试

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

### 3.3 启动本地推理服务

```bash
# 使用项目中的推理服务脚本
python3 camera_tutor/omni_server.py &
# 验证
curl http://localhost:8100/api/health
# 应返回: {"status":"ok"}
```

---

## 4. 安装 Force Aligner（嘴型同步）

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

## 5. 部署 Camera Tutor 项目

### 5.1 拉取代码

```bash
cd ~
git clone <你的仓库地址> camera-tutor
cd camera-tutor
```

### 5.2 配置

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

### 5.3 安装项目依赖

```bash
source ~/camera-tutor-env/bin/activate
pip install -r requirements.txt
pip install mediapipe  # 手势追踪（Pro 版）
```

---

## 6. 首次运行验证

### 6.1 Mock 模式（不需要模型，验证逻辑）

```bash
python3 camera_tutor/demo.py --mock
```

键入一些对话内容，确认决策引擎、对话流正常。

### 6.2 场景验证

```bash
python3 camera_tutor/scenario_demos.py
# 预期: 6/6 scenarios passed
```

### 6.3 硬件模式（需要 Qwen-Omni 已启动）

```bash
# 确认服务可用
curl http://localhost:8100/api/health

# 启动完整 Demo
python3 camera_tutor/demo.py
```

---

## 7. Live2D 安装（可选，Mid/Pro 版）

```bash
# 一键构建
chmod +x scripts/build_live2d_linux.sh
./scripts/build_live2d_linux.sh

# 验证
./build/live2d_renderer --help
```

---

## 8. 家长 Dashboard 启动

```bash
python3 camera_tutor/dashboard_server.py &

# 从浏览器访问
# http://<Orin IP>:8200
```

---

## 9. 常见问题

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

## 10. 快速检查清单

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
