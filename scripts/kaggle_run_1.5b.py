#!/usr/bin/env python3
"""
Qwen2.5-Math-1.5B-Instruct LoRA 训练 + 评估 (Kaggle)
"""
import os, sys, json, re, inspect, subprocess
from pathlib import Path

# --- 安装依赖（Kaggle script 模式不会自动跑 pip）---
def ensure_deps():
    deps = ["transformers>=4.44", "peft>=0.12", "trl>=0.11", "accelerate>=0.26", "datasets>=2.16"]
    missing = []
    try:
        import transformers, peft, trl, accelerate, datasets
    except ImportError:
        missing = deps
    if missing:
        print(f"安装依赖: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

ensure_deps()

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
SFT_FILE = "sft_train_130.jsonl"
OUTPUT_DIR = "output_1.5b"
LORA_R = 32
NUM_EPOCHS = 10
MAX_LENGTH = 1536
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 1e-4

def download_data():
    """查找 SFT 数据文件：Kaggle input 或 GitHub 下载"""
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        # Kaggle 环境：从挂载的 dataset 中找
        for d in os.listdir(kaggle_input):
            candidate = os.path.join(kaggle_input, d, "sft_train_130.jsonl")
            if os.path.exists(candidate):
                global SFT_FILE
                SFT_FILE = candidate
                print(f"Kaggle dataset: {SFT_FILE}")
                break
    if not os.path.exists(SFT_FILE):
        # 本地/GitHub 下载
        os.system(f"wget -q https://raw.githubusercontent.com/math4mad/Conditional-Prob-LoRA/main/data/sft_train_130.jsonl -O {SFT_FILE}")
    print(f"数据: {SFT_FILE} ({sum(1 for _ in open(SFT_FILE))} lines)")

def load_sft_data(file_path, tokenizer, limit=None):
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]
    formatted = []
    for r in records:
        messages = [
            {"role": "system", "content": r["instruction"]},
            {"role": "user", "content": r["input"]},
            {"role": "assistant", "content": r["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted.append({"text": text})
    return Dataset.from_list(formatted)

def train():
    print("=" * 60)
    print(f"Qwen2.5-Math-1.5B-Instruct LoRA 训练")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"LoRA r={LORA_R}, epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_sft_data(SFT_FILE, tokenizer)
    print(f"训练样本: {len(dataset)}")

    sft_params = list(inspect.signature(SFTConfig.__init__).parameters.keys())
    seq_key = "max_seq_length" if "max_seq_length" in sft_params else "max_length"

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

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer)
    print("\n开始训练...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ 训练完成，模型保存到: {OUTPUT_DIR}")

    # 保存训练日志
    log_history = trainer.state.log_history
    with open("train_log_1.5b.json", "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"📄 训练日志: train_log_1.5b.json")

    return model, tokenizer

def generate_response(model, tokenizer, instruction, user_input, max_new_tokens=512):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def parse_structure(text):
    steps = ["明确变量", "建立模型", "计算", "验证"]
    return [s for s in steps if s in text]

def count_mermaid(text):
    return 1 if "mermaid" in text.lower() or "graph" in text.lower() else 0

def count_latex(text):
    return 1 if "$" in text and "\\frac" in text else 0

def extract_numbers(text):
    return re.findall(r'P\s*[=（(]\s*([^)）]+)', text)

def evaluate(model, tokenizer, num_samples=10):
    print("\n" + "=" * 60)
    print("Qwen2.5-Math-1.5B LoRA 评估")
    print("=" * 60)

    records = []
    with open(SFT_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    data = records[:num_samples]
    print(f"评估样本: {len(data)}\n")

    results = []
    for i, r in enumerate(data):
        gen = generate_response(model, tokenizer, r["instruction"], r["input"])
        ref = r["output"]
        structure = parse_structure(gen)
        mermaid = count_mermaid(gen)
        latex = count_latex(gen)
        gen_nums = extract_numbers(gen)
        ref_nums = extract_numbers(ref)
        struct_score = len(structure)
        struct_pass = struct_score >= 4
        print(f"  [{i+1:2d}] 结构: {struct_score}/4 {'✅' if struct_pass else '❌'} "
              f"Mermaid:{'✅' if mermaid else '❌'} LaTeX:{'✅' if latex else '❌'} 数值:{len(gen_nums)}个")
        results.append({"idx": i+1, "structure": struct_score, "structure_pass": struct_pass,
                        "mermaid": mermaid, "latex": latex, "gen_nums": gen_nums, "ref_nums": ref_nums,
                        "generated": gen[:500]})

    n = len(results)
    four_step_rate = sum(1 for r in results if r["structure_pass"]) / n
    avg_struct = sum(r["structure"] for r in results) / n
    mermaid_rate = sum(r["mermaid"] for r in results) / n
    latex_rate = sum(r["latex"] for r in results) / n
    num_extract = sum(1 for r in results if r["gen_nums"]) / n

    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"\n四步法遵循率 (≥4/4项): {four_step_rate*100:.0f}%")
    print(f"结构项平均分: {avg_struct:.1f}/4")
    print(f"\nMermaid 覆盖率: {mermaid_rate*100:.0f}%")
    print(f"LaTeX 覆盖率: {latex_rate*100:.0f}%")
    print(f"\n数值提取率: {num_extract*100:.0f}%")

    report = {
        "model": MODEL_ID, "adapter": OUTPUT_DIR, "num_samples": n,
        "four_step_rate": four_step_rate, "avg_structure": avg_struct,
        "mermaid_rate": mermaid_rate, "latex_rate": latex_rate,
        "num_extract_rate": num_extract, "details": results,
    }
    with open("eval_report_1.5b.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 评估报告: eval_report_1.5b.json")

if __name__ == "__main__":
    download_data()
    model, tokenizer = train()
    evaluate(model, tokenizer, num_samples=10)
    print("\n✅ 全部完成！输出文件: output_1.5b/, eval_report_1.5b.json, train_log_1.5b.json")
