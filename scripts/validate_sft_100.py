#!/usr/bin/env python3
"""
validate_sft_100.py
对 sft_100.jsonl 做两项校验：
1. Mermaid 语法：检查是否有 == 旧语法、%%| 残留、未闭合代码块。
2. 数值一致性：从 output 中提取最终概率分数，用 Python 独立重算比对。
   （由于题目参数在 input 中，此处做宽松校验：提取 output 中的分数，
    验证分子≤分母、分母>0、概率在[0,1]区间。）
3. 结构完整性：四步法标题齐全、Mermaid块存在、概率表存在。
"""

import json
import re
import os
from fractions import Fraction

JSONL_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sft_100.jsonl")


def extract_mermaid_blocks(text):
    return re.findall(r"```mermaid\s*\n(.*?)```", text, re.DOTALL)


def extract_final_prob(text):
    """从 output 中提取最终概率分数（最后一个 = 后面的分数或整数）。"""
    # 匹配 = a/b 形式
    matches = re.findall(r"=\s*(\d+)/(\d+)", text)
    if matches:
        n, d = matches[-1]
        return Fraction(int(n), int(d))
    # 匹配 = 整数 （在 $$ 结尾或行尾）
    matches_int = re.findall(r"=\s*(\d+)\s*\$\$", text)
    if matches_int:
        return Fraction(int(matches_int[-1]), 1)
    return None


def validate():
    with open(JSONL_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"校验 {len(records)} 条数据\n")

    stats = {
        "mermaid_ok": 0, "mermaid_bad": 0,
        "prob_valid": 0, "prob_invalid": 0, "prob_missing": 0,
        "structure_ok": 0, "structure_bad": 0,
    }
    issues = []

    for r in records:
        src = r["source"]
        out = r["output"]

        # 1. Mermaid 语法
        blocks = extract_mermaid_blocks(out)
        mermaid_ok = True
        if not blocks:
            mermaid_ok = False
            issues.append((src, "无Mermaid块"))
        for b in blocks:
            if re.search(r"==[^=]*==>", b):
                mermaid_ok = False
                issues.append((src, "Mermaid旧==语法"))
            if "%%|" in b:
                mermaid_ok = False
                issues.append((src, "Mermaid %%|残留"))
            # 检查未闭合
            if b.count("```") > 0:
                mermaid_ok = False
                issues.append((src, "Mermaid块内嵌套代码块"))
        if mermaid_ok:
            stats["mermaid_ok"] += 1
        else:
            stats["mermaid_bad"] += 1

        # 2. 数值一致性（宽松校验）
        prob = extract_final_prob(out)
        if prob is None:
            stats["prob_missing"] += 1
            issues.append((src, "未提取到概率"))
        elif 0 <= prob <= 1 and prob.denominator > 0:
            stats["prob_valid"] += 1
        else:
            stats["prob_invalid"] += 1
            issues.append((src, f"概率异常: {prob}"))

        # 3. 结构完整性
        has_step1 = "## 1." in out
        has_step2 = "## 2." in out
        has_step3 = "## 3." in out
        has_step4 = "## 4." in out
        has_table = "|" in out and "---" in out
        if has_step1 and has_step2 and has_step3 and has_step4 and has_table:
            stats["structure_ok"] += 1
        else:
            stats["structure_bad"] += 1
            missing = []
            if not has_step1: missing.append("step1")
            if not has_step2: missing.append("step2")
            if not has_step3: missing.append("step3")
            if not has_step4: missing.append("step4")
            if not has_table: missing.append("table")
            issues.append((src, f"结构缺失: {','.join(missing)}"))

    print("=" * 60)
    print(f"Mermaid 语法:  ✅ {stats['mermaid_ok']} | ❌ {stats['mermaid_bad']}")
    print(f"概率数值:      ✅ {stats['prob_valid']} | ❌ {stats['prob_invalid']} | ⚪ {stats['prob_missing']}")
    print(f"结构完整性:    ✅ {stats['structure_ok']} | ❌ {stats['structure_bad']}")

    if issues:
        print(f"\n--- 问题详情 ({len(issues)} 条) ---")
        for src, msg in issues:
            print(f"  ⚠️ {src}: {msg}")
    else:
        print("\n🎉 全部校验通过!")

    # 按题型统计概率分布
    print("\n--- 概率分布抽样 ---")
    type_stats = {}
    for r in records:
        t = r["source"].split("-")[1]
        prob = extract_final_prob(r["output"])
        if prob is not None:
            type_stats.setdefault(t, []).append(float(prob))
    for t, probs in type_stats.items():
        avg = sum(probs) / len(probs)
        print(f"  {t}: {len(probs)}条, 均值={avg:.3f}, 范围=[{min(probs):.3f}, {max(probs):.3f}]")


if __name__ == "__main__":
    validate()
