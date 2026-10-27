#!/bin/bash
# autodl_train_516.sh
# 明早 AutoDL 一键训练：516条数据 × 10 epoch
# 用法: bash autodl_train_516.sh
set -e

echo "============================================================"
echo "  Conditional-Prob-LoRA: 516条数据训练"
echo "============================================================"

# --- 配置国内 pip 镜像 ---
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF
echo "✅ pip 镜像: 阿里云"

# --- 1. 安装依赖 ---
echo ""
echo "[1/3] 安装 Python 依赖..."
pip install -q \
  "transformers==4.49.0" \
  "trl==0.14.0" \
  "peft==0.14.0" \
  "accelerate>=0.26" \
  "datasets>=2.16" \
  "bitsandbytes>=0.43" \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com 2>&1 | tail -3
echo "✅ 依赖安装完成"

# --- 2. 检查 GPU ---
echo ""
echo "[2/3] GPU 检查..."
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

# --- 3. 检查数据 ---
echo "[3/3] 数据检查..."
DATA_FILE="/root/autodl-tmp/Conditional-Prob-LoRA/data/sft_train_580.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 数据文件不存在: $DATA_FILE"
    echo "   请先上传 data/sft_train_580.jsonl"
    exit 1
fi
wc -l "$DATA_FILE"
echo "✅ 数据就绪"

# --- 训练 ---
echo ""
echo "============================================================"
echo "  开始训练: Qwen2.5-Math-7B QLoRA 4-bit"
echo "  数据: 516条 × 10 epoch"
echo "  预计: ~20分钟 (4090)"
echo "============================================================"

cd /root/autodl-tmp/Conditional-Prob-LoRA

# 修改 train_lora.py 使用 580 数据文件
python3 -c "
import re
with open('scripts/train_lora.py', 'r') as f:
    code = f.read()
# 替换数据文件路径
code = code.replace('sft_train_130.jsonl', 'sft_train_580.jsonl')
with open('scripts/train_lora_580.py', 'w') as f:
    f.write(code)
print('✅ 创建 train_lora_580.py (使用516条数据)')
"

# 训练
nohup python3 scripts/train_lora_580.py --mode formal \
  --model_path /root/autodl-tmp/models/models/Qwen--Qwen2.5-Math-7B-Instruct/snapshots/master \
  --lora_r 32 \
  > /root/autodl-tmp/train516.log 2>&1 &

TRAIN_PID=$!
echo "训练 PID: $TRAIN_PID"
echo "日志: /root/autodl-tmp/train516.log"

# 等待训练完成
echo ""
echo "等待训练完成..."
wait $TRAIN_PID
echo "✅ 训练完成！"

# 评估
echo ""
echo "============================================================"
echo "  开始评估 (10条样本)"
echo "============================================================"
python3 scripts/evaluate_lora.py \
  --model_path /root/autodl-tmp/models/models/Qwen--Qwen2.5-Math-7B-Instruct/snapshots/master \
  --adapter_path output/formal \
  --num_samples 10 \
  2>&1 | tee /root/autodl-tmp/eval516.log

echo ""
echo "============================================================"
echo "  全部完成！"
echo "  训练日志: /root/autodl-tmp/train516.log"
echo "  评估日志: /root/autodl-tmp/eval516.log"
echo "  评估报告: output/eval_report.json"
echo "============================================================"
