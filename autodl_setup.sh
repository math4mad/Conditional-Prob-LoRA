#!/bin/bash
# ============================================================
# AutoDL 一键部署脚本
# 在 AutoDL 4090 实例上运行，完成：
#   1. 安装依赖
#   2. 克隆代码仓库
#   3. 下载 Qwen2.5-Math-7B-Instruct 模型
#   4. 启动 LoRA 正式训练
#
# 用法：
#   chmod +x autodl_setup.sh
#   ./autodl_setup.sh
#
# AutoDL 创建实例时选择：
#   - GPU: RTX 4090 24GB
#   - 镜像: PyTorch 2.1 + CUDA 11.8 + Python 3.10
# ============================================================

set -e

echo "============================================================"
echo "  Conditional-Prob-LoRA AutoDL 部署"
echo "============================================================"

# --- 配置国内 pip 镜像（阿里云）---
echo ""
echo "配置 pip 国内镜像（阿里云）..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF
# 同时配置 conda 镜像（清华）
cat > ~/.condarc << 'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
echo "✅ pip 镜像: https://mirrors.aliyun.com/pypi/simple/"
echo "✅ conda 镜像: https://mirrors.tuna.tsinghua.edu.cn/anaconda"

# --- 0. 环境检查 ---
echo ""
echo "[0/4] 环境检查..."
echo "GPU 信息:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo "Python: $(python3 --version)"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found, using system CUDA')"

# GPU 显存检查
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
echo "GPU 显存: ${GPU_MEM} MB"
if [ "$GPU_MEM" -lt 20000 ]; then
    echo "⚠️ 显存 < 20GB，将使用 QLoRA 4-bit 模式"
    USE_QLORA=1
else
    echo "✅ 显存 ≥ 20GB，使用 LoRA FP16 模式"
    USE_QLORA=0
fi

# --- 1. 安装依赖 ---
echo ""
echo "[1/4] 安装 Python 依赖..."
# 使用阿里云镜像加速（AutoDL 国内环境）
# 固定兼容版本：transformers 4.49 + trl 0.14 + peft 0.14 兼容 PyTorch 2.5
pip install -q \
  "transformers==4.49.0" \
  "trl==0.14.0" \
  "peft==0.14.0" \
  "accelerate>=0.26" \
  "datasets>=2.16" \
  "bitsandbytes>=0.43" \
  $PIP_MIRROR 2>&1 | tail -5
echo "✅ 依赖安装完成（transformers 4.49 / trl 0.14 / peft 0.14）"

# --- 2. 克隆代码仓库 ---
echo ""
echo "[2/4] 克隆代码仓库..."
REPO_URL="https://github.com/math4mad/Conditional-Prob-LoRA.git"
PROJECT_DIR="$HOME/Conditional-Prob-LoRA"

if [ -d "$PROJECT_DIR" ]; then
    echo "仓库已存在，拉取最新代码..."
    cd "$PROJECT_DIR"
    git pull
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

echo "代码结构:"
ls -la data/ scripts/
echo "✅ 代码就绪"

# --- 3. 下载模型 ---
echo ""
echo "[3/4] 下载 Qwen2.5-Math-7B-Instruct 模型..."
MODEL_DIR="$HOME/models/Qwen2.5-Math-7B-Instruct"

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
    echo "模型已存在: $MODEL_DIR"
else
    echo "使用 modelscope 下载（国内更快）..."
    pip install -q modelscope $PIP_MIRROR
    python3 -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-Math-7B-Instruct', cache_dir='$HOME/models')
print(f'模型已下载到: {model_dir}')
" 2>&1 | tail -10
fi
echo "✅ 模型就绪"

# --- 4. 启动训练 ---
echo ""
echo "[4/4] 启动 LoRA 训练..."
cd "$PROJECT_DIR"

if [ "$USE_QLORA" -eq 1 ]; then
    echo "使用 QLoRA 4-bit 模式 (显存不足 20GB)"
    python3 scripts/train_lora.py --mode formal --model_path "$MODEL_DIR" --lora_r 32 2>&1 | tee train_log.txt
else
    echo "使用 LoRA FP16 模式"
    python3 scripts/train_lora.py --mode formal --model_path "$MODEL_DIR" --lora_r 32 2>&1 | tee train_log.txt
fi

echo ""
echo "============================================================"
echo "  训练完成！"
echo "============================================================"
echo ""
echo "模型输出: $PROJECT_DIR/output/formal/"
echo "训练日志: $PROJECT_DIR/train_log.txt"
echo ""
echo "评估模型:"
echo "  python3 scripts/evaluate_lora.py --model_path $MODEL_DIR --adapter_path $PROJECT_DIR/output/formal --num_samples 20"
echo ""
echo "下载 LoRA adapter 到本地:"
echo "  scp -P <port> root@<autodl_ip>:$PROJECT_DIR/output/formal/adapter_model.safetensors ./"
