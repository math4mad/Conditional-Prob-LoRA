#!/usr/bin/env python3
"""
kaggle_evaluate_1.5b.py
评估 Qwen2.5-Math-1.5B LoRA 模型
"""
import os
import sys
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
ADAPTER_PATH = "output_1.5b"
SFT_FILE = "sft_train_130.jsonl"
NUM_SAMPLES = 10

KAGGLE_INPUT = "/kaggle/input"
if os.path.exists(KAGGLE_INPUT):
    for d in os.listdir(KAGGLE_INPUT):
        candidate = os.path.join(KAGGLE_INPUT, d, "sft_train_130.jsonl")
        if os.path.exists(candidate):
            SFT_FILE = candidate
            break

BASE_DIR = Path(SFT_FILE).parent


def load_eval_data(file_path: str, num_samples: int):
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
    return records[:num_samples]


def generate_response(model, tokenizer, instruction, user_input, max_new_tokens=512):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def parse_structure(text):
    """解析四步法结构"""
    steps = ["明确变量", "建立模型", "计算", "验证"]
    found = []
    for s in steps:
        if s in text:
            found.append(s)
    return found


def count_mermaid(text):
    return 1 if "mermaid" in text.lower() or "graph" in text.lower() else 0


def count_latex(text):
    return 1 if "$" in text and "\\frac" in text else 0


def extract_numbers(text):
    return re.findall(r'P\s*[=（(]\s*([^)）]+)', text)


def evaluate():
    print("=" * 60)
    print("Qwen2.5-Math-1.5B LoRA 评估")
    print("=" * 60)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型 FP16
    print(f"\n加载模型: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"加载 LoRA adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    # 加载评估数据
    data = load_eval_data(SFT_FILE, NUM_SAMPLES)
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
              f"Mermaid:{'✅' if mermaid else '❌'} "
              f"LaTeX:{'✅' if latex else '❌'} "
              f"数值:{len(gen_nums)}个")

        results.append({
            "idx": i + 1,
            "structure": struct_score,
            "structure_pass": struct_pass,
            "mermaid": mermaid,
            "latex": latex,
            "gen_nums": gen_nums,
            "ref_nums": ref_nums,
        })

    # 汇总
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)

    n = len(results)
    four_step_rate = sum(1 for r in results if r["structure_pass"]) / n
    avg_struct = sum(r["structure"] for r in results) / n
    mermaid_rate = sum(r["mermaid"] for r in results) / n
    latex_rate = sum(r["latex"] for r in results) / n
    num_extract = sum(1 for r in results if r["gen_nums"]) / n

    print(f"\n四步法遵循率 (≥4/4项): {four_step_rate*100:.0f}%")
    print(f"结构项平均分: {avg_struct:.1f}/4")
    print(f"\nMermaid 覆盖率: {mermaid_rate*100:.0f}%")
    print(f"LaTeX 覆盖率: {latex_rate*100:.0f}%")
    print(f"\n数值提取率: {num_extract*100:.0f}%")

    # 保存报告
    report = {
        "model": MODEL_ID,
        "adapter": ADAPTER_PATH,
        "num_samples": n,
        "four_step_rate": four_step_rate,
        "avg_structure": avg_struct,
        "mermaid_rate": mermaid_rate,
        "latex_rate": latex_rate,
        "num_extract_rate": num_extract,
        "details": results,
    }
    report_path = "eval_report_1.5b.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 评估报告: {report_path}")

    return report


if __name__ == "__main__":
    evaluate()
