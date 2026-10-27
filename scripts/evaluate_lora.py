#!/usr/bin/env python3
"""
evaluate_lora.py
结构化评估解析器：评估 LoRA 微调模型的输出质量。

两项核心指标：
1. 四步法遵循率：检查输出是否包含 ## 1. ~ ## 4. 四个步骤标题
2. 数值准确率：提取最终概率分数，与正确答案比对
3. Mermaid 覆盖率：检查是否包含 Mermaid 代码块
4. LaTeX 覆盖率：检查是否包含数学公式
"""

import json
import re
import argparse
from pathlib import Path
from fractions import Fraction

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent.parent
SFT_FILE = BASE_DIR / "data" / "sft_train_130.jsonl"


def load_test_data(limit: int = 20):
    """加载测试数据。"""
    records = []
    with open(SFT_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records[:limit]


def extract_final_prob(text: str) -> Fraction | None:
    """从文本中提取最终概率分数。"""
    # 匹配 = a/b
    matches = re.findall(r"=\s*(\d+)/(\d+)", text)
    if matches:
        n, d = matches[-1]
        return Fraction(int(n), int(d))
    # 匹配 = 整数（$$结尾）
    matches_int = re.findall(r"=\s*(\d+)\s*\$\$", text)
    if matches_int:
        return Fraction(int(matches_int[-1]), 1)
    return None


def evaluate_structure(text: str) -> dict:
    """评估四步法结构遵循度。"""
    checks = {
        "step1": bool(re.search(r"##\s*1\.", text)),
        "step2": bool(re.search(r"##\s*2\.", text)),
        "step3": bool(re.search(r"##\s*3\.", text)),
        "step4": bool(re.search(r"##\s*4\.", text)),
        "mermaid": "```mermaid" in text,
        "latex": "$$" in text or "$" in text,
        "table": "|" in text and "---" in text,
    }
    score = sum(checks.values())
    return {"checks": checks, "score": score, "total": len(checks)}


def evaluate_accuracy(generated: str, reference: str) -> dict:
    """评估数值准确率。"""
    gen_prob = extract_final_prob(generated)
    ref_prob = extract_final_prob(reference)
    return {
        "generated_prob": str(gen_prob) if gen_prob else None,
        "reference_prob": str(ref_prob) if ref_prob else None,
        "correct": gen_prob is not None and ref_prob is not None and gen_prob == ref_prob,
        "extracted": gen_prob is not None,
    }


def run_inference(model, tokenizer, instruction: str, input_text: str, max_tokens: int = 512):
    """运行推理生成。"""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated


def evaluate(model_path: str, adapter_path: str = None, num_samples: int = 20):
    """完整评估流程。"""
    print("=" * 60)
    print("LoRA 模型结构化评估")
    print("=" * 60)

    # 加载模型
    device = "cpu"  # 评估用 CPU 避免 MPS 问题
    print(f"\n加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)

    if adapter_path:
        print(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    # 加载测试数据
    test_data = load_test_data(num_samples)
    print(f"测试样本数: {len(test_data)}\n")

    # 评估统计
    stats = {
        "structure_scores": [],
        "structure_details": [],
        "accuracy_correct": 0,
        "accuracy_extracted": 0,
        "accuracy_total": 0,
    }

    print("--- 逐条评估 ---")
    for i, r in enumerate(test_data):
        # 生成
        generated = run_inference(
            model, tokenizer, r["instruction"], r["input"], max_tokens=512
        )

        # 结构评估
        struct = evaluate_structure(generated)
        stats["structure_scores"].append(struct["score"])

        # 准确率评估
        acc = evaluate_accuracy(generated, r["output"])
        stats["accuracy_total"] += 1
        if acc["extracted"]:
            stats["accuracy_extracted"] += 1
        if acc["correct"]:
            stats["accuracy_correct"] += 1

        # 打印
        struct_pct = struct["score"] / struct["total"] * 100
        acc_str = "✅" if acc["correct"] else ("❓" if acc["extracted"] else "❌")
        print(f"  [{i+1:2d}] 结构:{struct_pct:3.0f}% {acc_str} "
              f"gen={acc['generated_prob']} ref={acc['reference_prob']}")

        stats["structure_details"].append({
            "source": r["source"],
            "struct": struct["checks"],
            "accuracy": acc,
        })

    # 汇总
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)

    avg_struct = sum(stats["structure_scores"]) / len(stats["structure_scores"])
    struct_pass_rate = sum(1 for s in stats["structure_scores"] if s >= 4) / len(stats["structure_scores"])

    print(f"\n四步法遵循率 (≥4/7项): {struct_pass_rate*100:.1f}%")
    print(f"结构项平均分: {avg_struct:.2f}/7")

    print(f"\n数值准确率:")
    print(f"  答案提取率: {stats['accuracy_extracted']}/{stats['accuracy_total']} "
          f"({stats['accuracy_extracted']/stats['accuracy_total']*100:.1f}%)")
    print(f"  正确率: {stats['accuracy_correct']}/{stats['accuracy_total']} "
          f"({stats['accuracy_correct']/stats['accuracy_total']*100:.1f}%)")

    # Mermaid 和 LaTeX 覆盖率
    mermaid_count = sum(1 for d in stats["structure_details"] if d["struct"]["mermaid"])
    latex_count = sum(1 for d in stats["structure_details"] if d["struct"]["latex"])
    print(f"\nMermaid 覆盖率: {mermaid_count}/{len(test_data)} ({mermaid_count/len(test_data)*100:.1f}%)")
    print(f"LaTeX 覆盖率: {latex_count}/{len(test_data)} ({latex_count/len(test_data)*100:.1f}%)")

    # 保存详细结果
    report_path = BASE_DIR / "output" / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "samples": len(test_data),
                "structure_avg": avg_struct,
                "structure_pass_rate": struct_pass_rate,
                "accuracy_extracted": stats["accuracy_extracted"],
                "accuracy_correct": stats["accuracy_correct"],
                "mermaid_coverage": mermaid_count,
                "latex_coverage": latex_count,
            },
            "details": stats["structure_details"],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 评估报告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA 模型评估")
    parser.add_argument(
        "--model_path",
 type=str,
        default="/Users/mac/Programming/code-2026/GrandFather/models/models/Qwen--Qwen2.5-0.5B-Instruct",
        help="基础模型路径",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=str(BASE_DIR / "output" / "smoke"),
        help="LoRA adapter 路径",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="评估样本数",
    )
    args = parser.parse_args()

    evaluate(args.model_path, args.adapter_path, args.num_samples)


if __name__ == "__main__":
    main()
