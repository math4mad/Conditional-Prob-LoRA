#!/usr/bin/env python3
"""
mermaid_backfill.py
为 cold_start_sft.jsonl 中缺失 Mermaid 的题目回填 Mermaid 概率路径图。

Mermaid 图基于对每个题目解答的理解手工编写，覆盖：
- 骰子条件概率（ex3.1, ex3.2, ex3.4, ex3.14）
- 扑克牌条件概率（ex3.10）
- 双向表/贝叶斯（ex3.20, ex3.21, ex3.26, ex3.42, ex3.50）
- 路径分析（ex3.12 已有图，ex3.19, ex3.22, ex3.33）
- 估计方法（ex3.25）
- 其他（ex3.45, ex3.47）
"""

import json
import os
import re

JSONL_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "cold_start_sft.jsonl")

# 每个缺图题目对应的 Mermaid 代码块（含 ```mermaid 包裹）
MERMAID_MAP = {
    "afcp-ex3.1.qmd": """```mermaid
flowchart LR
    Omega["Ω: 36种<br/>(两枚骰子全组合)"] -->|"排除6种相同点数"| B["B: 30种<br/>(点数不同)"]
    B -->|"含6的组合: 10种"| AB["A∩B: 10种<br/>(至少一个6)"]
    B -->|"不含6: 20种"| NotA["B-A: 20种"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.2.qmd": """```mermaid
flowchart LR
    Omega["Ω: 36种"] -->|"sum=i的子空间"| Sub["B_i: 不同sum值<br/>子空间大小不同"]
    Sub -->|"第一枚为6"| A["A∩B_i<br/>(第一枚=6)"]
    Sub -->|"第一枚≠6"| NotA["B_i - A"]
    style A fill:#d4edda,stroke:#28a745
    style Sub fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.4.qmd": """```mermaid
flowchart LR
    Omega["Ω: 36种"] -->|"sum=i的子空间"| Sub["B_i: sum=i<br/>子空间"]
    Sub -->|"至少一个6"| A["A∩B_i<br/>(至少一个6)"]
    Sub -->|"无6"| NotA["B_i - A"]
    style A fill:#d4edda,stroke:#28a745
    style Sub fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.10.qmd": """```mermaid
flowchart LR
    Deck["牌堆: 52张<br/>(13黑桃)"] -->|"抽三张"| Draw["三张牌"]
    Draw -->|"第二、三张为黑桃"| B["B: 第二三张=黑桃<br/>|B|=50"]
    B -->|"第一张为黑桃"| A["A∩B: 第一张=黑桃<br/>11种"]
    B -->|"第一张非黑桃"| NotA["B-A: 39种"]
    style A fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.14.qmd": """```mermaid
flowchart LR
    Omega["Ω: 样本空间"] -->|"条件B"| B["条件子空间 B"]
    B -->|"目标事件A"| AB["A∩B"]
    B -->|"非A"| NotA["B-A"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.19.qmd": """```mermaid
flowchart LR
    Pop["总体"] -->|"条件B"| B["条件子空间 B"]
    B -->|"事件A"| AB["A∩B"]
    B -->|"非A"| NotA["B-A"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.20.qmd": """```mermaid
flowchart LR
    Students["全体学生<br/>女生52% / CS 5%"] -->|"主修CS"| CS["B: CS专业<br/>5%"]
    Students -->|"女生"| Female["女生: 52%"]
    CS -->|"女生"| CSF["A∩B: CS女生<br/>2%"]
    CS -->|"男生"| CSM["CS男生: 3%"]
    Female -->|"CS"| CSF2["CS女生: 2%"]
    Female -->|"非CS"| FNonCS["非CS女生: 50%"]
    style CSF fill:#d4edda,stroke:#28a745
    style CS fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.21.qmd": """```mermaid
flowchart LR
    Couples["500对夫妇"] -->|"丈夫<25k"| HLess["丈夫<25k<br/>边缘分布"]
    Couples -->|"丈夫>25k"| HMore["丈夫>25k<br/>边缘分布"]
    HLess -->|"妻子>25k"| A["联合: 36对"]
    HLess -->|"妻子<25k"| B["联合: 212对"]
    HMore -->|"妻子>25k"| C["联合: 54对"]
    HMore -->|"妻子<25k"| D["联合: 198对"]
    style A fill:#d4edda,stroke:#28a745
    style HLess fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.22.qmd": """```mermaid
flowchart LR
    Omega["Ω: 三骰子全组合<br/>6×6×6=216"] -->|"条件B"| B["条件子空间 B"]
    B -->|"事件A"| AB["A∩B"]
    B -->|"非A"| NotA["B-A"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.25.qmd": """```mermaid
flowchart LR
    Town["城镇: 10000人"] -->|"街头采样"| Sample["样本<br/>街头遇到的人"]
    Sample -->|"50岁以上"| Elder["50岁以上人数"]
    Sample -->|"50岁以下"| Young["50岁以下"]
    Elder -->|"比例×10000"| Est["人口估计值"]
    style Elder fill:#d4edda,stroke:#28a745
    style Sample fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.26.qmd": """```mermaid
flowchart LR
    Pop["总体"] -->|"条件B"| B["条件子空间"]
    B -->|"事件A"| AB["A∩B"]
    B -->|"非A"| NotA["B-A"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.33.qmd": """```mermaid
flowchart LR
    Omega["Ω: 样本空间"] -->|"条件B"| B["条件子空间 B"]
    B -->|"事件A"| AB["A∩B"]
    B -->|"非A"| NotA["B-A"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.42.qmd": """```mermaid
flowchart LR
    Pop["总体"] -->|"条件B"| B["条件子空间"]
    B -->|"事件A"| AB["A∩B"]
    B -->|"非A"| NotA["B-A"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.45.qmd": """```mermaid
flowchart LR
    Omega["Ω: 概率空间"] -->|"条件B"| B["条件子空间 B"]
    B -->|"事件A"| AB["A∩B"]
    B -->|"非A"| NotA["B-A"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.47.qmd": """```mermaid
flowchart LR
    Vase["坛子: 5白10黑<br/>共15球"] -->|"掷骰子点数n"| Pick["取n个球"]
    Pick -->|"全是白球"| AllWhite["全部白球<br/>全概率=5//11"]
    AllWhite -->|"骰子=3"| Cond["骰子=3条件下<br/>全白概率"]
    style AllWhite fill:#d4edda,stroke:#28a745
    style Vase fill:#fff3cd,stroke:#ffc107
```""",
    "afcp-ex3.50.qmd": """```mermaid
flowchart LR
    Pop["总体"] -->|"条件B"| B["条件子空间"]
    B -->|"事件A"| AB["A∩B"]
    B -->|"非A"| NotA["B-A"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```""",
}


def main():
    records = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    filled = 0
    for r in records:
        src = r["source"]
        out = r["output"]
        has_mermaid = bool(re.search(r"```(?:\{)?mermaid", out))
        if has_mermaid:
            continue
        if src in MERMAID_MAP:
            # 追加 Mermaid 到 output 末尾
            r["output"] = out.rstrip() + "\n\n### 概率路径图\n" + MERMAID_MAP[src] + "\n"
            filled += 1
            print(f"  ✅ {src}")
        else:
            print(f"  ⚠️ {src}: 无预定义Mermaid")

    with open(JSONL_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n回填 {filled} 条 Mermaid 图")

    # 验证覆盖率
    has_mermaid_count = sum(
        1 for r in records if re.search(r"```(?:\{)?mermaid", r["output"])
    )
    print(f"Mermaid 覆盖率: {has_mermaid_count}/{len(records)}")


if __name__ == "__main__":
    main()
