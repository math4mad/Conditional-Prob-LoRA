#!/bin/bash
# autodl_phase5.sh
# Phase5 完整测试流水线（AutoDL 4090）
#   1. 用修复 Mermaid 语法后的 546 条数据重训 7B
#   2. P0-c: OOD 泛化评估（58 条分布外测试集）
#   3. P2-a: 无 GraphRAG vs 有 GraphRAG 对比（OOD 测试集）
#   4. P1: 对模型输出做 Mermaid 语义对齐检查
#
# 用法: bash autodl_phase5.sh
set -e

ROOT=/root/autodl-tmp/Conditional-Prob-LoRA
MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-Math-7B-Instruct/snapshots/master
cd "$ROOT"

echo "============================================================"
echo "  Phase5 测试流水线"
echo "============================================================"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

# --- 环境检查 ---
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""
for f in data/sft_train_580.jsonl data/test_ood.jsonl scripts/train_lora.py \
         scripts/evaluate_lora.py scripts/evaluate_graphrag.py \
         scripts/check_mermaid_alignment.py; do
  if [ ! -f "$f" ]; then
    echo "❌ 缺少文件: $f"
    echo "   请先上传: git clone / scp"
    exit 1
  fi
done
echo "✅ 文件检查通过"
wc -l data/sft_train_580.jsonl data/test_ood.jsonl

# --- 生成 580 数据版本的训练/评估脚本 ---
sed 's/sft_train_130.jsonl/sft_train_580.jsonl/g' scripts/train_lora.py   > scripts/train_lora_580.py
sed 's/sft_train_130.jsonl/sft_train_580.jsonl/g' scripts/evaluate_lora.py > scripts/evaluate_lora_580.py

# ============================================================
echo ""
echo "============================================================"
echo "  [1/4] 重训 7B（546 条数据 × 10 epoch）"
echo "============================================================"
rm -rf output/formal
python scripts/train_lora_580.py --mode formal \
  --model_path "$MODEL" --lora_r 32 2>&1 | tail -30

# ============================================================
echo ""
echo "============================================================"
echo "  [2/4] P0-c: OOD 泛化评估（58 条分布外测试集）"
echo "============================================================"
python scripts/evaluate_lora_580.py \
  --model_path "$MODEL" --adapter_path output/formal \
  --test_file data/test_ood.jsonl --num_samples 58 \
  --report_name eval_ood --max_tokens 768 2>&1 | tail -40

# ============================================================
echo ""
echo "============================================================"
echo "  [3/4] P2-a: 无 GraphRAG vs 有 GraphRAG（OOD 测试集）"
echo "============================================================"
python scripts/evaluate_graphrag.py \
  --model_path "$MODEL" --adapter_path output/formal \
  --test_file data/test_ood.jsonl --num_samples 58 \
  --mode both --report_name eval_graphrag_ood --max_tokens 768 2>&1 | tail -45

# ============================================================
echo ""
echo "============================================================"
echo "  [4/4] P1: Mermaid 语义对齐检查"
echo "============================================================"
echo "--- OOD 评估结果 ---"
python scripts/check_mermaid_alignment.py --mode report \
  --file output/eval_ood.json --output output/alignment_ood.json 2>&1 | head -25
echo ""
echo "--- GraphRAG 对比结果 ---"
python scripts/check_mermaid_alignment.py --mode report \
  --file output/eval_graphrag_ood.json --output output/alignment_graphrag.json 2>&1 | head -25

echo ""
echo "============================================================"
echo "  全部完成！"
echo "============================================================"
ls -la output/*.json | awk '{print $5, $9}'
echo ""
echo "下载:"
echo "  scp -P 31224 root@connect.cqa1.seetacloud.com:$ROOT/output/{eval_ood.json,eval_graphrag_ood.json,alignment_ood.json,alignment_graphrag.json} ./"
