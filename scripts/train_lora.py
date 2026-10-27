#!/usr/bin/env python3
"""
train_lora.py
LoRA 微调训练脚本，支持两种模式：

1. 冒烟模式 (smoke): Qwen2.5-0.5B-Instruct, 本地 MPS, 少量数据快速验证
2. 正式模式 (formal): Qwen2.5-Math-7B, 需 CUDA GPU (AutoDL 4090)

用法:
  python3 scripts/train_lora.py --mode smoke
  python3 scripts/train_lora.py --mode formal --model_path /path/to/Qwen2.5-Math-7B
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(__file__).resolve().parent.parent
SFT_FILE = BASE_DIR / "data" / "sft_train_130.jsonl"

# 模型路径
LOCAL_05B = "/Users/mac/Programming/code-2026/GrandFather/models/models/Qwen--Qwen2.5-0.5B-Instruct"
QWEN_MATH_7B = "Qwen/Qwen2.5-Math-7B-Instruct"  # HuggingFace ID，正式训练时下载


def load_sft_data(file_path: str, limit: int = None, tokenizer=None) -> Dataset:
    """加载 SFT JSONL 数据，转为文本格式（用 chat template 格式化）。"""
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
    if limit:
        records = records[:limit]

    # 如果有 tokenizer，用 chat template 格式化为文本；否则用 messages 格式
    if tokenizer is not None:
        formatted = []
        for r in records:
            messages = [
                {"role": "system", "content": r["instruction"]},
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": r["output"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted.append({"text": text})
    else:
        # messages 格式（TRL >= 0.15 支持）
        formatted = [
            {"messages": [
                {"role": "system", "content": r["instruction"]},
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": r["output"]},
            ]}
            for r in records
        ]

    return Dataset.from_list(formatted)


def build_lora_config(r: int = 32) -> LoraConfig:
    """构建 LoRA 配置。"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=r * 2,  # alpha = 2*r 是常见设置
        lora_dropout=0.05,
        target_modules="all-linear",  # q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        bias="none",
    )


def train_smoke():
    """冒烟测试：0.5B 本地 MPS。"""
    print("=" * 60)
    print("冒烟测试模式: Qwen2.5-0.5B-Instruct (MPS)")
    print("=" * 60)

    model_path = LOCAL_05B
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        sys.exit(1)

    # 检查 MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"设备: {device}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    if device == "mps":
        model = model.to("mps")
    model.config.use_cache = False

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params / 1e6:.1f}M")

    # LoRA 配置 (冒烟用小 r)
    lora_config = build_lora_config(r=8)
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable / 1e6:.3f}M ({trainable / total_params * 100:.2f}%)")

    # 加载数据（冒烟只用20条）
    print("加载数据...")
    dataset = load_sft_data(str(SFT_FILE), limit=20, tokenizer=tokenizer)
    print(f"训练样本: {len(dataset)}")

    # 训练参数（冒烟：少 epoch，小 batch）
    # 注意：SFTConfig 的序列长度参数名随 TRL 版本变化
    #   TRL>=0.15: max_length
    #   TRL<0.15:  max_seq_length
    import inspect as _inspect
    _sft_params = list(_inspect.signature(SFTConfig.__init__).parameters.keys())
    _seq_key = "max_seq_length" if "max_seq_length" in _sft_params else "max_length"
    training_args = SFTConfig(
        output_dir=str(BASE_DIR / "output" / "smoke"),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=2,
        save_strategy="epoch",
        **{_seq_key: 1024},  # 0.5B 冒烟用短序列
        dataset_text_field="text",  # 预格式化文本字段
        packing=False,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 训练
    print("\n开始训练...")
    trainer.train()

    # 保存
    output_dir = BASE_DIR / "output" / "smoke"
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n✅ 冒烟训练完成，模型保存到: {output_dir}")

    # 快速推理测试（移到 CPU 避免 MPS generate 崩溃）
    print("\n--- 推理测试 ---")
    model = model.to("cpu")
    test_inference(model, tokenizer, dataset[0])

    return model, tokenizer


def train_formal(model_path: str = None, lora_r: int = 32):
    """正式训练：7B CUDA GPU。"""
    print("=" * 60)
    print("正式训练模式: Qwen2.5-Math-7B (CUDA)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ 正式训练需要 CUDA GPU（推荐 AutoDL 4090 24GB）")
        print("   本地无 NVIDIA GPU，无法进行正式训练。")
        print("   请在 AutoDL 环境运行此脚本。")
        sys.exit(1)

    # GPU 信息
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    model_path = model_path or QWEN_MATH_7B

    # 评估内存需求
    # 7B 模型 FP16 ≈ 14GB, LoRA 训练 forward+backward+optimizer 需 ~10GB
    # 4090 24GB 用 FP16 会 OOM，必须用 4-bit QLoRA (~5GB 模型 + ~10GB 训练 = ~15GB)
    use_4bit = True
    print("使用 4-bit QLoRA 量化（4090 24GB 必需）")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    print("加载模型...")
    from transformers import BitsAndBytesConfig
    # transformers 版本兼容: 4.49 用 torch_dtype, 5.x 用 dtype
    import transformers as _tf
    _tf_ver = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    _dtype_key = "dtype" if _tf_ver >= (5, 0) else "torch_dtype"
    load_kwargs = {
        _dtype_key: torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params / 1e9:.2f}B")

    # LoRA 配置
    lora_config = build_lora_config(r=lora_r)
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable / 1e6:.1f}M ({trainable / total_params * 100:.2f}%)")

    # 加载全部数据
    dataset = load_sft_data(str(SFT_FILE), tokenizer=tokenizer)
    print(f"训练样本: {len(dataset)}")

    # 训练参数
    # TRL 版本兼容: <0.15 用 max_seq_length, >=0.15 用 max_length
    import inspect as _inspect2
    _sft_params2 = list(_inspect2.signature(SFTConfig.__init__).parameters.keys())
    _seq_key2 = "max_seq_length" if "max_seq_length" in _sft_params2 else "max_length"
    training_args = SFTConfig(
        output_dir=str(BASE_DIR / "output" / "formal"),
        num_train_epochs=10,  # 增加 epoch 弥补小数据集
        per_device_train_batch_size=1,  # QLoRA 减小 batch
        gradient_accumulation_steps=16,  # 补偿 batch=1
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        **{_seq_key2: 1536},  # QLoRA 用较短序列节省显存
        dataset_text_field="text",
        packing=False,  # TRL 0.14 packing 不支持 messages 格式
        bf16=True,
        report_to="none",
        gradient_checkpointing=False,  # 与 LoRA 兼容性更好
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n开始训练...")
    trainer.train()

    output_dir = BASE_DIR / "output" / "formal"
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n✅ 正式训练完成，模型保存到: {output_dir}")

    return model, tokenizer


def test_inference(model, tokenizer, sample):
    """快速推理测试。"""
    messages = sample["messages"]
    # 只用 system + user
    input_messages = messages[:2]

    text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # 确保模型在 FP32 推理（避免 FP16 在 CPU 上产生 NaN）
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # 贪婪解码，避免采样 NaN
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print(f"题目: {messages[1]['content'][:60]}...")
    print(f"生成前200字: {generated[:200]}")
    print(f"参考答案前200字: {messages[2]['content'][:200]}")


def main():
    parser = argparse.ArgumentParser(description="LoRA 训练脚本")
    parser.add_argument(
        "--mode",
        choices=["smoke", "formal"],
        default="smoke",
        help="训练模式: smoke=0.5B冒烟, formal=7B正式",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="正式模式下的模型路径（默认 Qwen2.5-Math-7B-Instruct）",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA r 值（冒烟默认8，正式默认32）",
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        if args.lora_r is not None:
            print(f"注意：冒烟模式固定 r=8，忽略 --lora_r={args.lora_r}")
        train_smoke()
    else:
        train_formal(args.model_path, lora_r=args.lora_r)


if __name__ == "__main__":
    main()
