#!/usr/bin/env python3
"""
mermaid_filler.py
为 cold_start_sft.jsonl 中缺失 Mermaid 图的题目生成补全所需的 Prompt 文件。

策略（来自对话方案）：
- 以 ex3.40（含完整高质量 Mermaid 概率路径图）作为 few-shot 示范。
- 对每个缺图题目，提取其 input(题干) + output(解答摘要前500字) 作为待补全输入。
- 输出 prompts/mermaid_fill_<source>.txt，供批量调用 LLM API 使用。
"""

import json
import re
import os

JSONL_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "cold_start_sft.jsonl")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

SYSTEM_PROMPT_TEMPLATE = """【System Prompt】
你是一个概率论可视化专家。你的任务是：根据给定的概率问题解答文本，生成一个标准的 Mermaid flowchart 概率路径图。
要求：
1. 必须使用 ```mermaid 代码块包裹
2. 节点标签包含中文描述和关键数值（如 "B: 30种(点数不同)"）
3. 边标签标注转移概率或筛选条件
4. 目标事件节点用 style 高亮 (fill:#d4edda,stroke:#28a745)
5. 仅输出 mermaid 代码块，不要任何额外解释

【Few-shot Example】
来源: afcp-ex3.40 (Polya Urn 模型)
输入解答摘要：
"坛子里有5个白球和7个红球，每次取出一个放回并加一个同色球。路径：Vase->w1(5/12)->ww2(6/13)->www3(7/14)。三个全白概率=5/12*6/13*7/14。"

输出：
```mermaid
flowchart LR
    A[("⚪️⚪️⚪️⚪️⚪️\\n🔴🔴🔴🔴🔴🔴🔴")]==w1:5/12==>B[("⚪️⚪️⚪️⚪️⚪️⚪️\\n🔴🔴🔴🔴🔴🔴🔴")]
    A[("⚪️⚪️⚪️⚪️⚪️\\n🔴🔴🔴🔴🔴🔴🔴")]==r1:7/12==>C[("⚪️⚪️⚪️⚪️⚪️⚪️\\n🔴🔴🔴🔴🔴🔴🔴🔴")]
    B==ww2:6/13==>D[("⚪️⚪️⚪️⚪️⚪️⚪️⚪️\\n🔴🔴🔴🔴🔴🔴🔴")]
    B==wr2:7/13==>F[("⚪️⚪️⚪️⚪️⚪️⚪️\\n🔴🔴🔴🔴🔴🔴🔴🔴")]
    style B fill:#d4edda,stroke:#28a745
    style D fill:#d4edda,stroke:#28a745
```

【当前任务】
来源: {source}
题干：
{question}

解答摘要：
{solution_summary}
"""

INSTRUCTION = """请为上面的概率问题生成 Mermaid 概率路径图。只输出 ```mermaid 代码块。"""


def main():
    records = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    os.makedirs(PROMPTS_DIR, exist_ok=True)
    count = 0

    for r in records:
        out = r["output"]
        has_mermaid = bool(re.search(r"```(?:\{)?mermaid", out))
        if has_mermaid:
            continue

        source = r["source"].replace(".qmd", "")
        question = r["input"]
        # 解答摘要：取 output 前500字，去除代码块以聚焦文本推理
        summary = re.sub(r"```[^\n]*\n.*?```", "", out, flags=re.DOTALL).strip()
        summary = summary[:500]

        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            source=r["source"],
            question=question,
            solution_summary=summary,
        ) + "\n" + INSTRUCTION

        fname = f"mermaid_fill_{source}.txt"
        with open(os.path.join(PROMPTS_DIR, fname), "w", encoding="utf-8") as f:
            f.write(prompt)
        count += 1
        print(f"  📝 {fname}")

    print(f"\n生成 {count} 个 Mermaid 补全 Prompt 到 prompts/ 目录")


if __name__ == "__main__":
    main()
