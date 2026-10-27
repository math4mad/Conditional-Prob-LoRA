#!/usr/bin/env python3
"""
data_audit.py
对 cold_start_sft.jsonl 做质量审计：
1. 统计 Mermaid / Julia / 分数 覆盖率。
2. 列出缺失 Mermaid 的题目（需用 LLM 补全可视化）。
3. 列出缺失 Julia 代码的题目（需人工补充验证锚点）。
4. 检查 input/output 长度分布。
"""

import json
import re
import os

JSONL_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "cold_start_sft.jsonl")


def main():
    records = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"总记录数: {len(records)}\n")

    no_mermaid = []
    no_julia = []
    no_fraction = []
    short_input = []
    short_output = []

    for r in records:
        src = r.get("source", "?")
        out = r["output"]
        inp = r["input"]

        has_mermaid = bool(re.search(r"```(?:\{)?mermaid", out))
        has_julia = bool(re.search(r"```(?:\{)?julia", out))
        has_fraction = "//" in out

        if not has_mermaid:
            no_mermaid.append(src)
        if not has_julia:
            no_julia.append(src)
        if not has_fraction:
            no_fraction.append(src)
        if len(inp) < 30:
            short_input.append((src, len(inp)))
        if len(out) < 100:
            short_output.append((src, len(out)))

    print("=" * 50)
    print("缺失 Mermaid 图 (需 LLM 补全):")
    for s in no_mermaid:
        print(f"  - {s}")
    print(f"  共 {len(no_mermaid)} 条")

    print("\n" + "=" * 50)
    print("缺失 Julia 代码 (需补充验证锚点):")
    for s in no_julia:
        print(f"  - {s}")
    print(f"  共 {len(no_julia)} 条")

    print("\n" + "=" * 50)
    print("缺失分数形式 (//):")
    for s in no_fraction:
        print(f"  - {s}")
    print(f"  共 {len(no_fraction)} 条")

    print("\n" + "=" * 50)
    print("input 过短 (<30字，题干可能提取不完整):")
    for s, n in short_input:
        print(f"  - {s} ({n}字)")
    print(f"  共 {len(short_input)} 条")

    print("\n" + "=" * 50)
    print("output 过短 (<100字，解答可能不完整):")
    for s, n in short_output:
        print(f"  - {s} ({n}字)")
    print(f"  共 {len(short_output)} 条")

    # 长度分布
    print("\n" + "=" * 50)
    print("长度分布:")
    in_lens = [len(r["input"]) for r in records]
    out_lens = [len(r["output"]) for r in records]
    print(f"  input:  min={min(in_lens)} max={max(in_lens)} avg={sum(in_lens)//len(in_lens)}")
    print(f"  output: min={min(out_lens)} max={max(out_lens)} avg={sum(out_lens)//len(out_lens)}")


if __name__ == "__main__":
    main()
