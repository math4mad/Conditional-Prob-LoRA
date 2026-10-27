#!/usr/bin/env python3
"""
generate_sft_500.py
扩充版语料生成：500条条件概率 SFT 数据。

策略：
1. 复用 generate_sft_100.py 的 5 类生成器，加大数量
2. 新增 3 类题型：
   - 扑克牌条件概率 (60条)
   - 贝叶斯后验概率 (50条)
   - 独立事件链条件概率 (40条)

所有数值由 Python 精确计算（Fraction），标签 100% 正确。
"""
import json, os, random, sys
from fractions import Fraction
from itertools import product, permutations, combinations

random.seed(2026)

# 复用原脚本的生成器
sys.path.insert(0, os.path.dirname(__file__))
from generate_sft_100 import (
    SYSTEM_PROMPT, frac_str,
    gen_dice_problems, gen_urn_problems, gen_coin_problems,
    gen_table_problems, gen_path_problems,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_train_500.jsonl")


# ============================================================
# 题型6: 扑克牌条件概率 (60条)
# ============================================================
def gen_poker_problems(n=60):
    problems = []
    # 标准扑克牌 52 张
    suits = ["♠", "♥", "♦", "♣"]
    ranks = list(range(1, 14))  # 1=A, 11=J, 12=Q, 13=K
    deck = [(s, r) for s in suits for r in ranks]

    conditions = [
        ("第一张是黑桃", lambda cards: cards[0][0] == "♠"),
        ("第一张是红桃", lambda cards: cards[0][0] == "♥"),
        ("第一张是A", lambda cards: cards[0][1] == 1),
        ("第一张是人头牌(J/Q/K)", lambda cards: cards[0][1] >= 11),
        ("两张花色相同", lambda cards: cards[0][0] == cards[1][0]),
        ("两张花色不同", lambda cards: cards[0][0] != cards[1][0]),
        ("第一张点数大于第二张", lambda cards: cards[0][1] > cards[1][1]),
        ("没有A", lambda cards: all(c[1] != 1 for c in cards)),
    ]
    targets = [
        ("第二张是黑桃", lambda cards: cards[1][0] == "♠"),
        ("第二张是A", lambda cards: cards[1][1] == 1),
        ("第二张是红桃", lambda cards: cards[1][0] == "♥"),
        ("两张都是A", lambda cards: all(c[1] == 1 for c in cards)),
        ("两张点数相同", lambda cards: cards[0][1] == cards[1][1]),
        ("第二张点数大于第一张", lambda cards: cards[1][1] > cards[0][1]),
        ("至少一张是A", lambda cards: any(c[1] == 1 for c in cards)),
        ("第二张是人头牌", lambda cards: cards[1][1] >= 11),
    ]

    attempts = 0
    while len(problems) < n and attempts < n * 20:
        attempts += 1
        draw = 2
        cond_name, cond_func = random.choice(conditions)
        tgt_name, tgt_func = random.choice(targets)

        # 枚举有序不放回抽取
        space = list(permutations(range(52), draw))
        card_seqs = [tuple(deck[i] for i in seq) for seq in space]

        sub_B = [seq for seq in card_seqs if cond_func(seq)]
        if not sub_B:
            continue
        intersection = [seq for seq in sub_B if tgt_func(seq)]
        prob = Fraction(len(intersection), len(sub_B))
        if prob == 0:
            continue

        omega_size = len(card_seqs)
        b_size = len(sub_B)
        ab_size = len(intersection)

        rank_str = {1: "A", 11: "J", 12: "Q", 13: "K"}
        def card_str(c):
            r = rank_str.get(c[1], str(c[1]))
            return f"{c[0]}{r}"

        question = (
            f"从一副52张标准扑克牌中不放回地依次抽取{draw}张，"
            f"求在「{cond_name}」的条件下，「{tgt_name}」的条件概率。"
        )

        output = f"""## 1. 构建总概率空间
标准扑克牌52张（4花色×13点数），不放回抽{draw}张。
样本空间为所有有序抽取序列，$|\\Omega| = P(52,{draw}) = {omega_size}$。

## 2. 筛选条件子空间
条件 $B$：「{cond_name}」。
筛选后 $|B| = {b_size}$。

## 3. 确定目标事件
目标 $A$：「{tgt_name}」。
交集 $|A \\cap B| = {ab_size}$。

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{|A \\cap B|}}{{|B|}} = \\frac{{{ab_size}}}{{{b_size}}} = {frac_str(prob)}$$

### 概率路径图
```mermaid
flowchart LR
    Deck["扑克牌: 52张<br/>(4花色×13点数)"] -->|"不放回抽{draw}张"| Draw["Ω: {omega_size}种序列"]
    Draw -->|"{cond_name}"| B["B: {b_size}种"]
    B -->|"{tgt_name}: {ab_size}种"| AB["A∩B: {ab_size}种"]
    B -->|"不满足: {b_size-ab_size}种"| NotA["B-A: {b_size-ab_size}种"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 样本点数 | 条件概率 $P(\\cdot|B)$ |
| :--- | :---: | :---: |
| {tgt_name} ($A$) | {ab_size} | ${frac_str(prob)}$ |
| 不满足 ($A^c$) | {b_size-ab_size} | ${frac_str(Fraction(b_size-ab_size, b_size))}$ |
| **总计** | **{b_size}** | **1** |"""

        problems.append({
            "instruction": SYSTEM_PROMPT,
            "input": question,
            "output": output,
            "source": f"synth-poker-{len(problems)+1}",
        })

    return problems


# ============================================================
# 题型7: 贝叶斯后验概率 (50条)
# ============================================================
def gen_bayes_problems(n=50):
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 20:
        attempts += 1
        # 2-3 个假设
        num_hypo = random.choice([2, 3])
        # 先验概率（随机生成，和为1）
        priors_raw = [random.randint(1, 5) for _ in range(num_hypo)]
        s = sum(priors_raw)
        priors = [Fraction(p, s) for p in priors_raw]

        # 似然（P(证据|假设)）
        likelihoods = [Fraction(random.randint(1, 9), 10) for _ in range(num_hypo)]

        # 全概率 P(E) = sum P(E|Hi) * P(Hi)
        pe = sum(l * p for l, p in zip(likelihoods, priors))
        if pe == 0:
            continue

        # 后验 P(Hi|E) = P(E|Hi)*P(Hi) / P(E)
        posteriors = [l * p / pe for l, p in zip(likelihoods, priors)]

        # 随机选一个假设作为目标
        target_idx = random.randint(0, num_hypo - 1)

        labels = random.choice([
            (["疾病A", "疾病B", "疾病C"], "检测阳性"),
            (["机器1生产", "机器2生产", "机器3生产"], "产品有缺陷"),
            (["晴天", "阴天", "雨天"], "带伞出门"),
            (["股市上涨", "股市下跌", "股市震荡"], "交易量异常"),
            (["甲方案", "乙方案", "丙方案"], "项目延期"),
            (["正常", "轻微故障", "严重故障"], "报警触发"),
        ])
        hypo_names = labels[0][:num_hypo]
        evidence = labels[1]

        # 构建问题
        prior_str = "，".join(f"P({hypo_names[i]}) = {frac_str(priors[i])}" for i in range(num_hypo))
        like_str = "，".join(f"P({evidence}|{hypo_names[i]}) = {frac_str(likelihoods[i])}" for i in range(num_hypo))

        question = (
            f"已知先验概率：{prior_str}。\n"
            f"似然概率：{like_str}。\n"
            f"观察到证据「{evidence}」，求后验概率 P({hypo_names[target_idx]}|{evidence})。"
        )

        # 四步法解答
        output = f"""## 1. 构建总概率空间
假设空间 $H = \\{{{", ".join(hypo_names)}\\}}$，证据 $E = $「{evidence}」。
先验概率：
""" + "\n".join(f"- $P({hypo_names[i]}) = {frac_str(priors[i])}$" for i in range(num_hypo)) + f"""

似然概率：
""" + "\n".join(f"- $P(E|{hypo_names[i]}) = {frac_str(likelihoods[i])}$" for i in range(num_hypo)) + f"""

## 2. 筛选条件子空间（全概率公式）
$$P(E) = \\sum_i P(E|H_i) P(H_i) = {" + ".join(f"{frac_str(likelihoods[i])} \\cdot {frac_str(priors[i])}" for i in range(num_hypo))} = {frac_str(pe)}$$

## 3. 确定目标事件
目标：$P({hypo_names[target_idx]}|E)$

## 4. 贝叶斯公式计算
$$P({hypo_names[target_idx]}|E) = \\frac{{P(E|{hypo_names[target_idx]}) \\cdot P({hypo_names[target_idx]})}}{{P(E)}} = \\frac{{{frac_str(likelihoods[target_idx])} \\times {frac_str(priors[target_idx])}}}{{{frac_str(pe)}}} = {frac_str(posteriors[target_idx])}$$

### 概率路径图
```mermaid
flowchart LR
    Prior["先验分布<br/>""" + "<br/>".join(f"{hypo_names[i]}: {frac_str(priors[i])}" for i in range(num_hypo)) + f""""] -->|"观察到{evidence}"| Evidence["P(E) = {frac_str(pe)}"]
    Evidence -->|"贝叶斯更新"| Posterior["后验分布"]
""" + "\n".join(f'    Posterior --> "{hypo_names[i]}: {frac_str(posteriors[i])}"' for i in range(num_hypo)) + f"""
    style Posterior fill:#d4edda,stroke:#28a745
    style Evidence fill:#fff3cd,stroke:#ffc107
```

### 后验概率分布表
| 假设 | 先验 $P(H_i)$ | 似然 $P(E|H_i)$ | 后验 $P(H_i|E)$ |
| :--- | :---: | :---: | :---: |
""" + "\n".join(f"| {hypo_names[i]} | {frac_str(priors[i])} | {frac_str(likelihoods[i])} | {frac_str(posteriors[i])} |" for i in range(num_hypo)) + f"""
| **总计** | **1** | — | **1** |"""

        problems.append({
            "instruction": SYSTEM_PROMPT,
            "input": question,
            "output": output,
            "source": f"synth-bayes-{len(problems)+1}",
        })

    return problems


# ============================================================
# 题型8: 独立事件链条件概率 (40条)
# ============================================================
def gen_chain_problems(n=40):
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 20:
        attempts += 1
        # 2-3 个独立事件
        num_events = random.choice([2, 3])
        events = []
        for i in range(num_events):
            p = Fraction(random.randint(1, 9), 10)
            events.append(p)

        # 条件：某些事件发生
        cond_count = random.randint(1, num_events - 1) if num_events > 1 else 1
        cond_indices = random.sample(range(num_events), cond_count)
        # 目标：另一个事件发生
        remaining = [i for i in range(num_events) if i not in cond_indices]
        if not remaining:
            continue
        tgt_idx = random.choice(remaining)

        # P(B) = prod P(Ei) for i in cond_indices
        p_B = Fraction(1)
        for i in cond_indices:
            p_B *= events[i]
        if p_B == 0:
            continue

        # P(A ∩ B) = P(target) * P(B) (独立)
        p_AB = events[tgt_idx] * p_B
        prob = p_AB / p_B  # = events[tgt_idx] (独立事件)

        event_names = random.choice([
            (["下雨", "堵车", "迟到"][:num_events]),
            (["设备A故障", "设备B故障", "设备C故障"][:num_events]),
            (["收到邮件", "打开邮件", "回复邮件"][:num_events]),
            (["服务器过载", "响应慢", "报警"][:num_events]),
            (["信号A", "信号B", "信号C"][:num_events]),
        ])

        cond_desc = "且".join(event_names[i] + "发生" for i in cond_indices)
        tgt_desc = event_names[tgt_idx] + "发生"

        question = (
            f"{num_events}个独立事件：" + "，".join(f"{event_names[i]}的概率为{frac_str(events[i])}" for i in range(num_events)) + f"。\n"
            f"求在「{cond_desc}」的条件下，「{tgt_desc}」的条件概率。"
        )

        output = f"""## 1. 构建总概率空间
{num_events}个独立事件，样本空间为所有事件发生/不发生的组合，共 $2^{{{num_events}}} = {2**num_events}$ 种。
各事件概率：
""" + "\n".join(f"- $P({event_names[i]}) = {frac_str(events[i])}$" for i in range(num_events)) + f"""

## 2. 筛选条件子空间
条件 $B$：「{cond_desc}」。
$$P(B) = \\prod_{{i \\in cond}} P(E_i) = {" \\cdot ".join(frac_str(events[i]) for i in cond_indices)} = {frac_str(p_B)}$$

## 3. 确定目标事件
目标 $A$：「{tgt_desc}」。
由于事件独立，$A$ 与 $B$ 独立。

## 4. 条件概率计算
$$P(A|B) = \\frac{{P(A \\cap B)}}{{P(B)}} = \\frac{{P(A) \\cdot P(B)}}{{P(B)}} = P(A) = {frac_str(events[tgt_idx])}$$

**结论：独立事件的条件概率等于其本身概率。**

### 概率路径图
```mermaid
flowchart LR
    Space["样本空间<br/>2^{num_events}={2**num_events}种组合"] -->|"{cond_desc}"| B["B: P={frac_str(p_B)}"]
    B -->|"{tgt_desc}"| AB["A∩B: P={frac_str(p_AB)}"]
    B -->|"不满足"| NotA["B-A: P={frac_str(p_B - p_AB)}"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 概率 |
| :--- | :---: |
| {tgt_desc} ($A$) | ${frac_str(prob)}$ |
| 不满足 ($A^c$) | ${frac_str(1 - prob)}$ |
| **总计** | **1** |"""

        problems.append({
            "instruction": SYSTEM_PROMPT,
            "input": question,
            "output": output,
            "source": f"synth-chain-{len(problems)+1}",
        })

    return problems


# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("生成骰子条件概率题目 (80条)...")
    dice = gen_dice_problems(80)
    print(f"  ✅ 生成 {len(dice)} 条")

    print("生成坛子/球条件概率题目 (70条)...")
    urn = gen_urn_problems(70)
    print(f"  ✅ 生成 {len(urn)} 条")

    print("生成硬币条件概率题目 (60条)...")
    coin = gen_coin_problems(60)
    print(f"  ✅ 生成 {len(coin)} 条")

    print("生成双向表/贝叶斯题目 (50条)...")
    table = gen_table_problems(50)
    print(f"  ✅ 生成 {len(table)} 条")

    print("生成多阶段路径图题目 (40条)...")
    path = gen_path_problems(40)
    print(f"  ✅ 生成 {len(path)} 条")

    print("生成扑克牌条件概率题目 (60条)...")
    poker = gen_poker_problems(60)
    print(f"  ✅ 生成 {len(poker)} 条")

    print("生成贝叶斯后验概率题目 (50条)...")
    bayes = gen_bayes_problems(50)
    print(f"  ✅ 生成 {len(bayes)} 条")

    print("生成独立事件链条件概率题目 (40条)...")
    chain = gen_chain_problems(40)
    print(f"  ✅ 生成 {len(chain)} 条")

    all_problems = dice + urn + coin + table + path + poker + bayes + chain
    print(f"\n总计: {len(all_problems)} 条")

    # 写入JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in all_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"📄 输出: {OUTPUT_FILE}")

    # 质量统计
    has_mermaid = sum(1 for p in all_problems if "```mermaid" in p["output"])
    has_table = sum(1 for p in all_problems if "| 事件" in p["output"] or "| 路径" in p["output"] or "| 假设" in p["output"])
    has_latex = sum(1 for p in all_problems if "$$" in p["output"])
    has_fraction = sum(1 for p in all_problems if "/" in p["output"])
    print(f"\n📊 质量统计:")
    print(f"   Mermaid 图: {has_mermaid}/{len(all_problems)}")
    print(f"   概率表: {has_table}/{len(all_problems)}")
    print(f"   LaTeX 公式: {has_latex}/{len(all_problems)}")
    print(f"   分数形式: {has_fraction}/{len(all_problems)}")

    # 题型分布
    print(f"\n📈 题型分布:")
    sources = {}
    for p in all_problems:
        src_type = p["source"].rsplit("-", 1)[0]
        sources[src_type] = sources.get(src_type, 0) + 1
    for k, v in sorted(sources.items()):
        print(f"   {k}: {v}条")


if __name__ == "__main__":
    main()
