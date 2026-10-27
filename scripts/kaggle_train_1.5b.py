#!/usr/bin/env python3
"""
kaggle_train_1.5b.py
Kaggle Notebook: Qwen2.5-Math-1.5B-Instruct LoRA 训练

Kaggle 用法:
  1. 创建 Notebook, GPU = T4 x2 或 P100
  2. Internet = ON
  3. 将此文件和 data/sft_train_130.jsonl 上传为 Dataset
  4. 运行全部 cell

本地也可运行（有 CUDA GPU 时）。
"""
import os
import sys
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# --- 配置 ---
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
SFT_FILE = "sft_train_130.jsonl"  # Kaggle Dataset 挂载路径
OUTPUT_DIR = "output_1.5b"
LORA_R = 32
NUM_EPOCHS = 10
MAX_LENGTH = 1536
BATCH_SIZE = 2       # 1.5B 模型小，可以用 batch=2
GRAD_ACCUM = 8       # effective batch = 16
LR = 1e-4

# Kaggle 路径适配
KAGGLE_INPUT = "/kaggle/input"
if os.path.exists(KAGGLE_INPUT):
    # Kaggle 环境：从 Dataset 目录找 sft 文件
    for d in os.listdir(KAGGLE_INPUT):
        candidate = os.path.join(KAGGLE_INPUT, d, "sft_train_130.jsonl")
        if os.path.exists(candidate):
            SFT_FILE = candidate
            break
    print(f"Kaggle 环境, SFT 数据: {SFT_FILE}")

BASE_DIR = Path(SFT_FILE).parent


def load_sft_data(file_path: str, tokenizer, limit: int = None):
    """加载 SFT JSONL, 用 chat template 格式化为 text。"""
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
    if limit:
        records = records[:limit]

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

    return Dataset.from_list(formatted)


def main():
    print("=" * 60)
    print(f"Qwen2.5-Math-1.5B-Instruct LoRA 训练")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"LoRA r={LORA_R}, epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print("=" * 60)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 模型 (FP16, 1.5B 不需要 4-bit) ---
    print("\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # --- LoRA ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 数据 ---
    dataset = load_sft_data(SFT_FILE, tokenizer)
    print(f"训练样本: {len(dataset)}")

    # --- SFTConfig (版本兼容) ---
    import inspect
    sft_params = list(inspect.signature(SFTConfig.__init__).parameters.keys())
    seq_key = "max_seq_length" if "max_seq_length" in sft_params else "max_length"

    # transformers 版本兼容
    import transformers
    tf_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if tf_ver >= (5, 0) else "torch_dtype"

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        **{seq_key: MAX_LENGTH},
        dataset_text_field="text",
        packing=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        gradient_checkpointing=False,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n开始训练...")
    trainer.train()

    # --- 保存 ---
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ 训练完成，模型保存到: {OUTPUT_DIR}")

    # --- 快速推理测试 ---
    print("\n--- 推理测试 ---")
    model.eval()
    sample = dataset[0]
    inputs = tokenizer(sample["text"].split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n",
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
    print(f"生成: {generated[:300]}")

    return model, tokenizer


if __name__ == "__main__":
    main()
