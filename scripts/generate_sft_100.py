#!/usr/bin/env python3
"""
generate_sft_100.py
程序化合成100条条件概率 SFT 语料。

策略：基于冷启动数据的5大题型，通过参数变异生成新题目。
每条数据的数值答案由 Python 验证器精确计算（枚举样本空间），
确保标签 100% 正确。Mermaid 图和四步法解答由模板自动填充。

题型分布：
  - 骰子条件概率 (30条)
  - 坛子/球条件概率 (25条)
  - 硬币条件概率 (20条)
  - 双向表/贝叶斯 (15条)
  - 多阶段路径图 (10条)
"""

import json
import os
import random
from fractions import Fraction
from itertools import product, combinations

random.seed(42)

SYSTEM_PROMPT = (
    "你是一个概率论教学专家。请严格按照【空间构建->条件筛选->目标事件->计算与可视化】四步法解决概率问题。"
    "所有概率计算必须使用分数形式，避免小数近似。回答中必须包含Mermaid路径图和LaTeX条件概率表。"
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_100.jsonl")


def frac_str(f: Fraction) -> str:
    """分数格式化：1/3 而非 1//3。"""
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


# ============================================================
# 题型1: 骰子条件概率 (30条)
# ============================================================

def gen_dice_problems(n=30):
    problems = []
    sides_choices = [4, 6, 8, 12]
    num_dice_choices = [2, 3]
    conditions = [
        ("点数不同", lambda dice: len(set(dice)) == len(dice)),
        ("至少有一个骰子点数为1", lambda dice: 1 in dice),
        ("至少有一个骰子点数为最大值", lambda dice: max(dice) in dice),
        ("所有骰子点数相同", lambda dice: len(set(dice)) == 1),
        ("点数之和为偶数", lambda dice: sum(dice) % 2 == 0),
        ("没有骰子点数为1", lambda dice: 1 not in dice),
        ("至少有两个骰子点数相同", lambda dice: len(set(dice)) < len(dice)),
    ]
    targets = [
        ("至少有一个骰子点数为{}", lambda dice, v: v in dice),
        ("所有骰子点数都是偶数", lambda dice: all(d % 2 == 0 for d in dice)),
        ("至少有一个骰子点数为奇数", lambda dice: any(d % 2 == 1 for d in dice)),
        ("点数之和大于{}", lambda dice, s: sum(dice) > s),
        ("最大点数为{}", lambda dice, v: max(dice) == v),
        ("最小点数为{}", lambda dice, v: min(dice) == v),
        ("恰好有一个骰子点数为{}", lambda dice, v: sum(1 for d in dice if d == v) == 1),
    ]

    attempts = 0
    while len(problems) < n and attempts < n * 10:
        attempts += 1
        sides = random.choice(sides_choices)
        num_dice = random.choice(num_dice_choices)
        cond_name, cond_func = random.choice(conditions)
        tgt_name, tgt_func = random.choice(targets)

        # 为 target 填充参数
        val = random.randint(1, sides)
        threshold = random.randint(num_dice, num_dice * sides // 2)
        if "{}" in tgt_name:
            if "大于" in tgt_name:
                full_tgt = tgt_name.format(threshold)
                full_tgt_func = lambda dice, s=threshold: tgt_func(dice, s)
            else:
                full_tgt = tgt_name.format(val)
                full_tgt_func = lambda dice, v=val: tgt_func(dice, v)
        else:
            full_tgt = tgt_name
            full_tgt_func = lambda dice: tgt_func(dice)

        # 枚举样本空间
        omega = list(product(range(1, sides + 1), repeat=num_dice))
        sub_B = [d for d in omega if cond_func(d)]
        if not sub_B:
            continue
        intersection = [d for d in sub_B if full_tgt_func(d)]
        prob = Fraction(len(intersection), len(sub_B))

        if prob == 0:
            continue  # 跳过概率为0的平凡情况

        dice_str = "骰子" if sides == 6 else f"{sides}面骰子"
        num_str = "两枚" if num_dice == 2 else "三枚"

        question = (
            f"掷{num_str}均匀的{dice_str}，"
            f"求在「{cond_name}」的条件下，「{full_tgt}」的条件概率。"
        )

        # 构建四步法解答
        omega_size = len(omega)
        b_size = len(sub_B)
        ab_size = len(intersection)

        # 列举部分样本点（最多展示8个）
        def fmt_sample(d):
            return f"({','.join(str(x) for x in d)})"
        show_omega = ", ".join(fmt_sample(d) for d in omega[:6]) + f" ... 共{omega_size}种"
        show_b = ", ".join(fmt_sample(d) for d in sub_B[:8]) + (f" ... 共{b_size}种" if b_size > 8 else f" 共{b_size}种")
        show_ab = ", ".join(fmt_sample(d) for d in intersection[:8]) + (f" ... 共{ab_size}种" if ab_size > 8 else f" 共{ab_size}种")

        sides_list = ",".join(str(i) for i in range(1, sides + 1))
        space_def = f"\\Omega = \\{{{sides_list}\\}}^{{{num_dice}}}"

        output = f"""## 1. 构建总概率空间
掷{num_str}{dice_str}，样本空间 ${space_def}$，共 $|\\Omega| = {omega_size}$ 种等可能基本事件。
示例：{show_omega}

## 2. 筛选条件子空间
条件 $B$：「{cond_name}」。
筛选后子空间 $|B| = {b_size}$。
示例：{show_b}
在此子空间下，每个基本事件概率为 $1/{b_size}$。

## 3. 确定目标事件
目标事件 $A$：「{full_tgt}」。
在子空间 $B$ 中筛选满足 $A$ 的组合，交集 $|A \\cap B| = {ab_size}$。
示例：{show_ab}

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{|A \\cap B|}}{{|B|}} = \\frac{{{ab_size}}}{{{b_size}}} = {frac_str(prob)}$$

### 概率路径图
```mermaid
flowchart LR
    Omega["Ω: {omega_size}种<br/>({num_str}{dice_str})"] -->|"{cond_name}"| B["B: {b_size}种"]
    B -->|"{full_tgt}: {ab_size}种"| AB["A∩B: {ab_size}种"]
    B -->|"不满足: {b_size-ab_size}种"| NotA["B-A: {b_size-ab_size}种"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 样本点数 | 条件概率 $P(\\cdot|B)$ |
| :--- | :---: | :---: |
| {full_tgt} ($A$) | {ab_size} | ${frac_str(prob)}$ |
| 不满足 ($A^c$) | {b_size-ab_size} | ${frac_str(Fraction(b_size-ab_size, b_size))}$ |
| **总计** | **{b_size}** | **1** |"""

        problems.append({
            "instruction": SYSTEM_PROMPT,
            "input": question,
            "output": output,
            "source": f"synth-dice-{len(problems)+1}",
        })

    return problems


# ============================================================
# 题型2: 坛子/球条件概率 (25条)
# ============================================================

def gen_urn_problems(n=25):
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 10:
        attempts += 1
        w = random.randint(2, 8)  # 白球
        b = random.randint(2, 8)  # 黑球
        total = w + b
        draw = random.randint(2, min(4, total - 1))

        # 条件类型
        cond_type = random.choice(["第一球为白", "第一球为黑", "无白球", "至少一白"])
        # 目标类型
        tgt_type = random.choice(["第二球为白", "第二球为黑", "全为白", "全为黑"])

        # 枚举不放回抽样空间
        colors = ["W"] * w + ["B"] * b
        from itertools import permutations
        # 所有可能的抽取序列（不放回，有序）
        space = list(permutations(range(total), draw))
        # 映射为颜色序列
        color_seqs = [tuple(colors[i] for i in seq) for seq in space]

        # 条件
        if cond_type == "第一球为白":
            cond_func = lambda seq: seq[0] == "W"
        elif cond_type == "第一球为黑":
            cond_func = lambda seq: seq[0] == "B"
        elif cond_type == "无白球":
            cond_func = lambda seq: "W" not in seq
        elif cond_type == "至少一白":
            cond_func = lambda seq: "W" in seq

        # 目标
        if tgt_type == "第二球为白":
            tgt_func = lambda seq: seq[1] == "W" if len(seq) > 1 else False
        elif tgt_type == "第二球为黑":
            tgt_func = lambda seq: seq[1] == "B" if len(seq) > 1 else False
        elif tgt_type == "全为白":
            tgt_func = lambda seq: all(c == "W" for c in seq)
        elif tgt_type == "全为黑":
            tgt_func = lambda seq: all(c == "B" for c in seq)

        sub_B = [seq for seq in color_seqs if cond_func(seq)]
        if not sub_B:
            continue
        intersection = [seq for seq in sub_B if tgt_func(seq)]
        prob = Fraction(len(intersection), len(sub_B))
        if prob == 0:
            continue

        question = (
            f"坛子里有{w}个白球和{b}个黑球。不放回地依次取出{draw}个球，"
            f"求在「{cond_type}」的条件下，「{tgt_type}」的条件概率。"
        )

        omega_size = len(color_seqs)
        b_size = len(sub_B)
        ab_size = len(intersection)

        output = f"""## 1. 构建总概率空间
坛子有 $\\{{{w}\\text{{白}}, {b}\\text{{黑}}\\}}$，不放回取{draw}个球。
样本空间为所有有序抽取序列，$|\\Omega| = P({total},{draw}) = {omega_size}$。

## 2. 筛选条件子空间
条件 $B$：「{cond_type}」。
筛选后 $|B| = {b_size}$。
每个序列在 $B$ 下概率为 $1/{b_size}$。

## 3. 确定目标事件
目标 $A$：「{tgt_type}」。
交集 $|A \\cap B| = {ab_size}$。

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{|A \\cap B|}}{{|B|}} = \\frac{{{ab_size}}}{{{b_size}}} = {frac_str(prob)}$$

### 概率路径图
```mermaid
flowchart LR
    Urn["坛子: {w}白{b}黑<br/>共{total}球"] -->|"不放回取{draw}个"| Draw["Ω: {omega_size}种序列"]
    Draw -->|"{cond_type}"| B["B: {b_size}种"]
    B -->|"{tgt_type}: {ab_size}种"| AB["A∩B: {ab_size}种"]
    B -->|"不满足: {b_size-ab_size}种"| NotA["B-A: {b_size-ab_size}种"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 样本点数 | 条件概率 $P(\\cdot|B)$ |
| :--- | :---: | :---: |
| {tgt_type} ($A$) | {ab_size} | ${frac_str(prob)}$ |
| 不满足 ($A^c$) | {b_size-ab_size} | ${frac_str(Fraction(b_size-ab_size, b_size))}$ |
| **总计** | **{b_size}** | **1** |"""

        problems.append({
            "instruction": SYSTEM_PROMPT,
            "input": question,
            "output": output,
            "source": f"synth-urn-{len(problems)+1}",
        })

    return problems


# ============================================================
# 题型3: 硬币条件概率 (20条)
# ============================================================

def gen_coin_problems(n=20):
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 10:
        attempts += 1
        # 多枚硬币，不同偏置
        num_coins = random.randint(2, 4)
        biases = []
        for i in range(num_coins):
            p = Fraction(random.randint(1, 9), 10)
            biases.append(p)

        cond_type = random.choice(["至少一个正面", "至少一个反面", "第一枚为正面", "第一枚为反面"])
        tgt_type = random.choice(["全部为正面", "全部为反面", "恰好一个正面", "恰好一个反面"])

        # 枚举样本空间（每枚硬币 H/T）
        omega = list(product(["H", "T"], repeat=num_coins))
        # 计算每个结果的概率
        def prob_of(outcome):
            p = Fraction(1)
            for i, r in enumerate(outcome):
                p *= biases[i] if r == "H" else (1 - biases[i])
            return p

        # 条件
        if cond_type == "至少一个正面":
            cond_func = lambda o: "H" in o
        elif cond_type == "至少一个反面":
            cond_func = lambda o: "T" in o
        elif cond_type == "第一枚为正面":
            cond_func = lambda o: o[0] == "H"
        elif cond_type == "第一枚为反面":
            cond_func = lambda o: o[0] == "T"

        # 目标
        if tgt_type == "全部为正面":
            tgt_func = lambda o: all(r == "H" for r in o)
        elif tgt_type == "全部为反面":
            tgt_func = lambda o: all(r == "T" for r in o)
        elif tgt_type == "恰好一个正面":
            tgt_func = lambda o: o.count("H") == 1
        elif tgt_type == "恰好一个反面":
            tgt_func = lambda o: o.count("T") == 1

        # 加权概率（非等可能）
        total_prob = sum(prob_of(o) for o in omega)
        b_prob = sum(prob_of(o) for o in omega if cond_func(o))
        if b_prob == 0:
            continue
        ab_prob = sum(prob_of(o) for o in omega if cond_func(o) and tgt_func(o))
        prob = ab_prob / b_prob
        if prob == 0:
            continue

        bias_desc = ", ".join(f"第{i+1}枚正面概率{frac_str(biases[i])}" for i in range(num_coins))
        question = (
            f"有{num_coins}枚硬币（{bias_desc}）。同时抛掷所有硬币，"
            f"求在「{cond_type}」的条件下，「{tgt_type}」的条件概率。"
        )

        output = f"""## 1. 构建总概率空间
{num_coins}枚硬币，{bias_desc}。
样本空间 $\\Omega = \\{{H,T\\}}^{{{num_coins}}}$，共 $2^{{{num_coins}}} = {len(omega)}$ 种结果（非等可能，需加权计算）。

## 2. 筛选条件子空间
条件 $B$：「{cond_type}」。
$$P(B) = {frac_str(b_prob)}$$

## 3. 确定目标事件
目标 $A$：「{tgt_type}」。
$$P(A \\cap B) = {frac_str(ab_prob)}$$

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{P(A \\cap B)}}{{P(B)}} = \\frac{{{frac_str(ab_prob)}}}{{{frac_str(b_prob)}}} = {frac_str(prob)}$$

### 概率路径图
```mermaid
flowchart LR
    Coins["{num_coins}枚硬币<br/>({bias_desc})"] -->|"同时抛掷"| Omega["Ω: {len(omega)}种结果<br/>P={frac_str(total_prob)}"]
    Omega -->|"{cond_type}<br/>P(B)={frac_str(b_prob)}"| B["条件子空间 B"]
    B -->|"{tgt_type}<br/>P(A∩B)={frac_str(ab_prob)}"| AB["A∩B"]
    B -->|"不满足"| NotA["B-A"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 概率 |
| :--- | :---: |
| {tgt_type} ($A$) | ${frac_str(prob)}$ |
| 不满足 ($A^c$) | ${frac_str(Fraction(1) - prob)}$ |
| **总计** | **1** |"""

        problems.append({
            "instruction": SYSTEM_PROMPT,
            "input": question,
            "output": output,
            "source": f"synth-coin-{len(problems)+1}",
        })

    return problems


# ============================================================
# 题型4: 双向表/贝叶斯 (15条)
# ============================================================

def gen_table_problems(n=15):
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 10:
        attempts += 1
        # 2x2 列联表
        a = random.randint(10, 50)  # 左上
        b = random.randint(10, 50)  # 右上
        c = random.randint(10, 50)  # 左下
        d = random.randint(10, 50)  # 右下
        total = a + b + c + d
        row1 = a + b
        row2 = c + d
        col1 = a + c
        col2 = b + d

        # 随机选条件和目标
        direction = random.choice(["row", "col"])
        if direction == "row":
            cond_name = f"属于第1行"
            cond_prob = Fraction(row1, total)
            tgt_val = a
            tgt_name = "属于第1列"
            prob = Fraction(a, row1)
        else:
            cond_name = f"属于第1列"
            cond_prob = Fraction(col1, total)
            tgt_val = a
            tgt_name = "属于第1行"
            prob = Fraction(a, col1)

        if prob == 0:
            continue

        labels = random.choice([
            ("收入高于阈值", "收入低于阈值", "女性", "男性"),
            ("通过考试", "未通过", "理科生", "文科生"),
            ("患病", "未患病", "检测阳性", "检测阴性"),
            ("吸烟", "不吸烟", "患肺癌", "未患肺癌"),
            ("使用产品A", "使用产品B", "满意", "不满意"),
        ])

        question = (
            f"某调查共有{total}人，双向表如下：\n"
            f"| | {labels[0]} | {labels[1]} | 合计 |\n"
            f"| {labels[2]} | {a} | {b} | {row1} |\n"
            f"| {labels[3]} | {c} | {d} | {row2} |\n"
            f"| 合计 | {col1} | {col2} | {total} |\n"
            f"求在「{cond_name}」的条件下，「{tgt_name}」的条件概率。"
        )

        output = f"""## 1. 构建总概率空间
调查总人数 $N = {total}$。
样本空间为所有受访者，每个个体等可能。

双向表：
| | {labels[0]} | {labels[1]} | 合计 |
| :--- | :---: | :---: | :---: |
| {labels[2]} | {a} | {b} | {row1} |
| {labels[3]} | {c} | {d} | {row2} |
| **合计** | **{col1}** | **{col2}** | **{total}** |

## 2. 筛选条件子空间
条件 $B$：「{cond_name}」。
$$P(B) = \\frac{{{row1 if direction=='row' else col1}}}{{{total}}} = {frac_str(cond_prob)}$$

## 3. 确定目标事件
目标 $A$：「{tgt_name}」。
交集 $A \\cap B$ 对应表中单元格值 ${tgt_val}$。
$$P(A \\cap B) = \\frac{{{tgt_val}}}{{{total}}}$$

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{P(A \\cap B)}}{{P(B)}} = \\frac{{{tgt_val}/{total}}}{{{row1 if direction=='row' else col1}/{total}}} = \\frac{{{tgt_val}}}{{{row1 if direction=='row' else col1}}} = {frac_str(prob)}$$

### 概率路径图
```mermaid
flowchart LR
    N["总人数: {total}"] -->|"{cond_name}"| B["B: {row1 if direction=='row' else col1}人"]
    B -->|"{tgt_name}: {tgt_val}人"| AB["A∩B: {tgt_val}人"]
    B -->|"不满足: {(row1 if direction=='row' else col1)-tgt_val}人"| NotA["B-A: {(row1 if direction=='row' else col1)-tgt_val}人"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 人数 | 条件概率 $P(\\cdot|B)$ |
| :--- | :---: | :---: |
| {tgt_name} ($A$) | {tgt_val} | ${frac_str(prob)}$ |
| 不满足 ($A^c$) | {(row1 if direction=='row' else col1)-tgt_val} | ${frac_str(Fraction(1)-prob)}$ |
| **总计** | **{row1 if direction=='row' else col1}** | **1** |"""

        problems.append({
            "instruction": SYSTEM_PROMPT,
            "input": question,
            "output": output,
            "source": f"synth-table-{len(problems)+1}",
        })

    return problems


# ============================================================
# 题型5: 多阶段路径图 (10条)
# ============================================================

def gen_path_problems(n=10):
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 10:
        attempts += 1
        # 2-3阶段路径，每阶段2个分支
        stages = random.randint(2, 3)
        path_probs = []
        for s in range(stages):
            p1 = Fraction(random.randint(2, 8), 10)
            p2 = Fraction(1) - p1
            path_probs.append((p1, p2))

        # 目标：某条特定路径
        chosen_path = tuple(random.choice([0, 1]) for _ in range(stages))
        # 条件：最终到达某个状态
        final_state = random.randint(0, 1)

        # 计算所有路径概率
        all_paths = list(product([0, 1], repeat=stages))
        path_probs_map = {}
        for path in all_paths:
            p = Fraction(1)
            for i, choice in enumerate(path):
                p *= path_probs[i][choice]
            path_probs_map[path] = p

        # 条件：最终状态（最后一阶段的选择）
        b_prob = sum(p for path, p in path_probs_map.items() if path[-1] == final_state)
        if b_prob == 0:
            continue
        # 目标：选定路径
        target_path_prob = path_probs_map[chosen_path]
        # 目标在条件下：选定路径且最终状态匹配
        if chosen_path[-1] != final_state:
            ab_prob = Fraction(0)
        else:
            ab_prob = target_path_prob

        prob = ab_prob / b_prob if b_prob > 0 else Fraction(0)
        if prob == 0:
            continue

        stage_names = random.choice([
            (["第一阶段", "第二阶段", "第三阶段"][:stages], ["成功", "失败"]),
            (["第一次抽取", "第二次抽取", "第三次抽取"][:stages], ["白球", "黑球"]),
            (["第一步", "第二步", "第三步"][:stages], ["向左", "向右"]),
        ])
        stages_names, branch_names = stage_names

        path_desc = " → ".join(f"{stages_names[i]}:{branch_names[chosen_path[i]]}" for i in range(stages))
        cond_desc = f"最终{branch_names[final_state]}"

        question = (
            f"一个{stages}阶段概率过程，每阶段有两个可能结果（{branch_names[0]}/{branch_names[1]}）。"
            f"各阶段转移概率："
            + "；".join(f"{stages_names[i]}：{branch_names[0]}概率{frac_str(path_probs[i][0])}，{branch_names[1]}概率{frac_str(path_probs[i][1])}" for i in range(stages))
            + f"。求在「{cond_desc}」的条件下，「路径[{path_desc}]」的条件概率。"
        )

        # 构建Mermaid路径图
        mermaid_lines = ["flowchart LR"]
        prev_nodes = ["Start"]
        mermaid_lines.append('    Start(["开始"])')
        for i in range(stages):
            curr_nodes = []
            for j in range(2):
                node_id = f"S{i+1}_{j}"
                curr_nodes.append(node_id)
                label = f'{branch_names[j]}<br/>P={frac_str(path_probs[i][j])}'
                mermaid_lines.append(f'    {node_id}["{label}"]')
            # 连接边
            for pk, prev in enumerate(prev_nodes):
                for ck, curr in enumerate(curr_nodes):
                    # 只画合理路径（避免指数爆炸，画主要路径）
                    pass
            # 简化：从上一阶段每个节点画两条边
            for prev in prev_nodes:
                for ck, curr in enumerate(curr_nodes):
                    mermaid_lines.append(f'    {prev} -->|"{frac_str(path_probs[i][ck])}"| {curr}')
            prev_nodes = curr_nodes
        # 高亮目标路径
        mermaid_lines.append(f'    style S1_{chosen_path[0]} fill:#d4edda,stroke:#28a745')
        mermaid_lines.append(f'    style Start fill:#fff3cd,stroke:#ffc107')
        mermaid_block = "```mermaid\n" + "\n".join(mermaid_lines) + "\n```"

        output = f"""## 1. 构建总概率空间
{stages}阶段概率过程，每阶段2个分支，共 $2^{{{stages}}} = {len(all_paths)}$ 条路径。
各阶段转移概率：
""" + "\n".join(f"- {stages_names[i]}：{branch_names[0]} = {frac_str(path_probs[i][0])}，{branch_names[1]} = {frac_str(path_probs[i][1])}" for i in range(stages)) + f"""

## 2. 筛选条件子空间
条件 $B$：「{cond_desc}」。
$$P(B) = {frac_str(b_prob)}$$

## 3. 确定目标事件
目标 $A$：「路径[{path_desc}]」。
$$P(A \\cap B) = {frac_str(ab_prob)}$$

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{P(A \\cap B)}}{{P(B)}} = \\frac{{{frac_str(ab_prob)}}}{{{frac_str(b_prob)}}} = {frac_str(prob)}$$

### 概率路径图
{mermaid_block}

### 条件概率分布表 (给定B)
| 路径 | 概率 |
| :--- | :---: |
| 目标路径 | ${frac_str(prob)}$ |
| 其他路径 | ${frac_str(Fraction(1)-prob)}$ |
| **总计** | **1** |"""

        problems.append({
            "instruction": SYSTEM_PROMPT,
            "input": question,
            "output": output,
            "source": f"synth-path-{len(problems)+1}",
        })

    return problems


# ============================================================
# 主函数
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("生成骰子条件概率题目 (30条)...")
    dice = gen_dice_problems(30)
    print(f"  ✅ 生成 {len(dice)} 条")

    print("生成坛子/球条件概率题目 (25条)...")
    urn = gen_urn_problems(25)
    print(f"  ✅ 生成 {len(urn)} 条")

    print("生成硬币条件概率题目 (20条)...")
    coin = gen_coin_problems(20)
    print(f"  ✅ 生成 {len(coin)} 条")

    print("生成双向表/贝叶斯题目 (15条)...")
    table = gen_table_problems(15)
    print(f"  ✅ 生成 {len(table)} 条")

    print("生成多阶段路径图题目 (10条)...")
    path = gen_path_problems(10)
    print(f"  ✅ 生成 {len(path)} 条")

    all_problems = dice + urn + coin + table + path
    print(f"\n总计: {len(all_problems)} 条")

    # 写入JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in all_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"📄 输出: {OUTPUT_FILE}")

    # 统计
    import re
    has_mermaid = sum(1 for p in all_problems if "```mermaid" in p["output"])
    has_table = sum(1 for p in all_problems if "| 事件" in p["output"] or "| 路径" in p["output"])
    has_latex = sum(1 for p in all_problems if "$" in p["output"])
    has_fraction = sum(1 for p in all_problems if "/" in p["output"])
    print(f"\n📊 质量统计:")
    print(f"   Mermaid 图: {has_mermaid}/{len(all_problems)}")
    print(f"   概率表: {has_table}/{len(all_problems)}")
    print(f"   LaTeX 公式: {has_latex}/{len(all_problems)}")
    print(f"   分数形式: {has_fraction}/{len(all_problems)}")


if __name__ == "__main__":
    main()
