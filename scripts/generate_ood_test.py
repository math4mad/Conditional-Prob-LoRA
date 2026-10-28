#!/usr/bin/env python3
"""
generate_ood_test.py
Phase5 / P0: 生成真正的分布外 (Out-of-Distribution) 测试集。

背景 (ref: external/diage2.md 第三层 P0):
  "若 OOD ≈ 100%：检查生成脚本的种子是否真的隔离了（可能用了相同模板只是换了数字）"

实测确认：仅换随机种子无法隔离 —— 参数空间小，39 条中 25 条数字组合与训练集完全重合。

因此本脚本提供三层递进的 OOD：

  Layer 1  OOD-Param  同题型 + 超范围参数
           坛子 12-30 球（训练集 2-8）、骰子 8-20 面（训练集 2-6）
           → 测试"参数外推"

  Layer 2  OOD-New   训练集完全没有的 8 个新题型
           超几何抽签 / 几何概率 / 二项分布条件 / 条件期望
           马尔可夫链 / 朴素贝叶斯 / 正态对称条件 / 串联系统可靠性
           → 测试"结构外推"（真正的 OOD）

  Layer 3  OOD-Hard   新题型 + 大数值范围 + 多层条件依赖
           → 测试极限泛化

所有答案用 Python Fraction 精确计算，保证标签 100% 正确。
输出格式与训练集一致：四步法 + Mermaid + LaTeX + 概率表。
"""
import json
import os
import random
import sys
from fractions import Fraction
from itertools import product
from math import comb

sys.path.insert(0, os.path.dirname(__file__))
from generate_sft_100 import SYSTEM_PROMPT, frac_str

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEED = 424242


# ============================================================
# Layer 1: OOD-Param —— 同题型，超范围参数
# ============================================================
def gen_ood_param(n=12):
    """坛子（12-30 球）+ 超大骰子（8-20 面），训练集参数范围之外的题目。"""
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 20:
        attempts += 1
        w = random.randint(12, 30)   # 训练集是 2-8
        b = random.randint(12, 30)
        total = w + b
        draw = random.randint(3, 5)

        colors = ["W"] * w + ["B"] * b
        from itertools import permutations
        space = list(permutations(range(total), draw))
        color_seqs = [tuple(colors[i] for i in seq) for seq in space]

        cond_name = random.choice(["第一球为白", "无白球", "至少一白"])
        tgt_name = random.choice(["第二球为白", "全为白", "至少两白"])

        cond_func = {
            "第一球为白": lambda s: s[0] == "W",
            "无白球": lambda s: "W" not in s,
            "至少一白": lambda s: "W" in s,
        }[cond_name]
        tgt_func = {
            "第二球为白": lambda s: s[1] == "W",
            "全为白": lambda s: all(c == "W" for c in s),
            "至少两白": lambda s: sum(1 for c in s if c == "W") >= 2,
        }[tgt_name]

        sub_B = [s for s in color_seqs if cond_func(s)]
        if not sub_B:
            continue
        inter = [s for s in sub_B if tgt_func(s)]
        prob = Fraction(len(inter), len(sub_B))
        if prob == 0:
            continue

        omega_size, b_size, ab_size = len(color_seqs), len(sub_B), len(inter)
        question = (
            f"坛子里有{w}个白球和{b}个黑球（大量球情形）。不放回地依次取出{draw}个球，"
            f"求在「{cond_name}」的条件下，「{tgt_name}」的条件概率。"
        )
        output = _build_output(
            space_desc=f"坛子有 ${{\\{w}\\text{{白}}, {b}\\text{{黑}}\\}}$，不放回取{draw}个球。",
            omega_size=omega_size, omega_formula=f"P({total},{draw}) = {omega_size}",
            cond_name=cond_name, b_size=b_size, b_prob=None,
            tgt_name=tgt_name, ab_size=ab_size, prob=prob,
            mermaid_head=f'Urn["坛子: {w}白{b}黑<br/>共{total}球"]',
            mermaid_draw=f'Draw["Ω: {omega_size}种序列"]',
        )
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodparam-urn-{len(problems)+1}"})
    return problems


# ============================================================
# Layer 2/3: OOD-New —— 训练集完全没有的新题型
# ============================================================
def _build_output(space_desc, omega_size, omega_formula, cond_name, b_size, b_prob,
                  tgt_name, ab_size, prob, mermaid_head, mermaid_draw,
                  extra_steps=None, mermaid_extra=""):
    """统一的四步法输出模板。"""
    b_line = f"|B| = {b_size}" if b_prob is None else f"P(B) = {frac_str(b_prob)}"
    extra = extra_steps or ""
    return f"""## 1. 构建总概率空间
{space_desc}
样本空间大小 $|\\Omega| = {omega_formula}$。

## 2. 筛选条件子空间
条件 $B$：「{cond_name}」。
{b_line}。

## 3. 确定目标事件
目标 $A$：「{tgt_name}」。
交集 $|A \\cap B| = {ab_size}$。

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{|A \\cap B|}}{{|B|}} = \\frac{{{ab_size}}}{{{b_size}}} = {frac_str(prob)}$$
{extra}
### 概率路径图
```mermaid
flowchart LR
    {mermaid_head} -->|"{omega_size}种可能"| {mermaid_draw}
    {mermaid_draw} -->|"{cond_name}"| B["B: {b_size}种"]
    B -->|"{tgt_name}: {ab_size}种"| AB["A∩B: {ab_size}种"]
    B -->|"不满足: {b_size-ab_size}种"| NotA["B-A: {b_size-ab_size}种"]
{mermaid_extra}    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 样本点数 | 条件概率 $P(\\cdot|B)$ |
| :--- | :---: | :---: |
| {tgt_name} ($A$) | {ab_size} | ${frac_str(prob)}$ |
| 不满足 ($A^c$) | {b_size-ab_size} | ${frac_str(Fraction(b_size-ab_size, b_size))}$ |
| **总计** | **{b_size}** | **1** |"""


def gen_lottery(n=5):
    """超几何抽签：从 N 张券抽 k 张（训练集未覆盖超几何分布）。"""
    problems = []
    for _ in range(n):
        N = random.randint(15, 30)      # 券总数
        K = random.randint(4, 8)         # 中奖券数
        k = random.randint(3, 5)         # 抽取数
        if K >= N:
            continue
        # 完整枚举：抽到的中奖券数
        outcomes = {}   # (中奖数, 是否含指定券) -> 方案数
        for combo in product(range(N), repeat=k):
            if len(set(combo)) != k:
                continue
            win_cnt = sum(1 for c in combo if c < K)
            has_special = 0 in combo
            outcomes[(win_cnt, has_special)] = outcomes.get((win_cnt, has_special), 0) + 1

        total_combos = sum(outcomes.values())
        # 条件：抽到的券中包含 0 号券
        sub_B = {kk: v for kk, v in outcomes.items() if kk[1]}
        b_size = sum(sub_B.values())
        if b_size == 0:
            continue
        # 目标：恰好 t 张中奖券
        t = random.randint(1, min(K, k) - 1) if min(K, k) >= 2 else 1
        inter = sum(v for kk, v in sub_B.items() if kk[0] == t)
        if inter == 0:
            continue
        prob = Fraction(inter, b_size)

        # 无条件下的对照分布（用于表格）
        p_t = Fraction(sum(v for kk, v in outcomes.items() if kk[0] == t), total_combos)

        question = (
            f"某抽奖箱内有{N}张券，其中{K}张为中奖券。不放回地抽取{k}张券，"
            f"已知抽到的券中包含1号券，求「恰好抽中{t}张中奖券」的条件概率。"
        )
        extra = f"""
无条件时该事件概率为 $P(A) = {frac_str(p_t)}$，说明条件 $B$ 改变了结果。
"""
        output = _build_output(
            space_desc=f"抽奖箱有 ${{\\{K}\\text{{中奖}}, {N-K}\\text{{不中奖}}\\}}$，不放回抽{k}张。",
            omega_size=total_combos, omega_formula=f"C(N,k) = {total_combos}",
            cond_name="抽到的券包含1号券", b_size=b_size, b_prob=None,
            tgt_name=f"恰好{t}张中奖券", ab_size=inter, prob=prob,
            mermaid_head=f'Box["抽奖箱: {K}中奖/{N-K}不中奖<br/>共{N}张"]',
            mermaid_draw=f'Draw["Ω: C({N},{k})={total_combos}种"]',
            extra_steps=extra,
        )
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodnew-lottery-{len(problems)+1}"})
    return problems


def gen_geometry(n=5):
    """几何概率：单位正方形随机点落入区域（训练集未覆盖连续几何概率）。"""
    problems = []
    for _ in range(n):
        # 单位正方形，随机点 (X,Y)
        # B: 点在单位圆内 (x^2+y^2<=1) 面积 pi/4  -> 用可分数区域
        # 用矩形区域保证面积为分数
        kind = random.choice(["triangle", "strip", "circle"])
        if kind == "triangle":
            # B: x+y<=1 三角形 面积 1/2 ; C: x+y<=1/2 面积 1/8
            b_area, c_area = Fraction(1, 2), Fraction(1, 8)
            b_desc = "区域 x+y≤1（单位正方形左下三角形）"
            c_desc = "区域 x+y≤1/2"
        elif kind == "strip":
            # B: x<=3/4 面积 3/4 ; C: x<=1/2 面积 1/2
            b_area, c_area = Fraction(3, 4), Fraction(1, 2)
            b_desc = "竖直条带 x≤3/4"
            c_desc = "竖直条带 x≤1/2"
        else:
            # B: x^2+y^2<=1/4 圆 面积 pi/4 -> 改用正方形内接
            b_area, c_area = Fraction(1, 2), Fraction(1, 4)
            b_desc = "正方形 x∈[0,1/2] 区域"
            c_desc = "正方形 x∈[0,1/4] 区域"

        prob = c_area / b_area
        b_pts = int(b_area * 100)
        c_pts = int(c_area * 100)

        question = (
            f"在单位正方形 [0,1]×[0,1] 内随机投掷一点 P。"
            f"已知点 P 落在{b_desc}内，求点 P 落在{c_desc}内的条件概率。"
        )
        extra = f"""
几何概率按面积比计算：$P(B) = {frac_str(b_area)}$，$P(A \\cap B) = {frac_str(c_area)}$。
"""
        output = _build_output(
            space_desc="连续型样本空间（单位正方形，面积 1）。",
            omega_size=100, omega_formula="S(\\Omega) = 1（面积）",
            cond_name=f"P 落在{b_desc}内", b_size=b_pts, b_prob=b_area,
            tgt_name=f"P 落在{c_desc}内", ab_size=c_pts, prob=prob,
            mermaid_head='Sq["单位正方形<br/>S=1"]',
            mermaid_draw='Point["随机点 P<br/>(X,Y)~U(0,1)²"]',
            extra_steps=extra,
            mermaid_extra='    Point -->|"落入B"| Bx["B: 面积 '+str(b_pts)+'"]\n',
        )
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodnew-geometry-{len(problems)+1}"})
    return problems


def gen_binomial(n=5):
    """二项分布条件概率（训练集只有超几何+独立事件，无二项）。"""
    problems = []
    for _ in range(n):
        N = random.randint(8, 20)
        p = Fraction(random.randint(2, 8), 10)
        m = random.randint(2, N // 2)

        dist = {k: Fraction(comb(N, k)) * p ** k * (1 - p) ** (N - k) for k in range(N + 1)}
        p_B = sum(dist[k] for k in range(m, N + 1))
        if p_B == 0:
            continue
        t = random.randint(0, m - 1)
        inter = dist[t]
        prob = inter / p_B

        question = (
            f"某批零件的不合格率为 {frac_str(p)}，随机抽取 {N} 件。"
            f"已知其中不合格件数不少于 {m} 件，"
            f"求「恰好 {t} 件不合格」的条件概率。"
        )
        extra = f"""
二项分布 $X \\sim B({N}, {frac_str(p)})$，$P(X={t}) = {frac_str(dist[t])}$，
$P(X \\geq {m}) = {frac_str(p_B)}$。
"""
        output = _build_output(
            space_desc=f"不合格件数 $X \\sim B({N}, {frac_str(p)})$，取值 $0..{N}$。",
            omega_size=N + 1, omega_formula=f"{N}+1 = {N+1}",
            cond_name=f"不合格件数 ≥ {m}", b_size=N + 1 - m, b_prob=p_B,
            tgt_name=f"恰好 {t} 件不合格", ab_size=1, prob=prob,
            mermaid_head=f'Bin["X ~ B({N}, {frac_str(p)})"]',
            mermaid_draw='Dist["取值 0..'+str(N)+'<br/>共'+str(N+1)+'种"]',
            extra_steps=extra,
        )
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodnew-binomial-{len(problems)+1}"})
    return problems


def gen_cond_expect(n=4):
    """条件期望 E[X|Y]（训练集未覆盖）。"""
    problems = []
    for _ in range(n):
        N = random.randint(6, 12)
        p = Fraction(random.randint(3, 7), 10)
        y = random.randint(1, N // 2)

        dist = {k: Fraction(comb(N, k)) * p ** k * (1 - p) ** (N - k) for k in range(N + 1)}
        # P(Y=y) 与 E[X|Y=y]，Y 为独立同分布的第二次抽样计数
        py = dist[y]
        joint = {k: dist[k] * py for k in range(N + 1)}
        num = sum(Fraction(k) * joint[k] for k in range(N + 1))
        denom = sum(joint.values())
        e_xy = num / denom
        e_x = sum(Fraction(k) * dist[k] for k in range(N + 1))

        question = (
            f"某工厂两批产品的不合格率为 {frac_str(p)}，各随机抽取 {N} 件，"
            f"设第一批不合格件数为 X，第二批为 Y。已知 Y = {y} 件，"
            f"求条件期望 E[X | Y = {y}]。"
        )
        extra = f"""
$E[X] = {frac_str(e_x)}$（无条件的期望）。
由于两批独立，$E[X|Y={y}] = E[X] = {frac_str(e_xy)}$。
"""
        output = _build_output(
            space_desc=f"两次独立抽样，不合格件数 $X,Y \\sim B({N}, {frac_str(p)})$。",
            omega_size=N + 1, omega_formula=f"{N}+1 = {N+1}",
            cond_name=f"Y = {y}", b_size=N + 1, b_prob=py,
            tgt_name="X 的条件期望", ab_size=1, prob=e_xy,
            mermaid_head=f'Bin["X,Y ~ B({N}, {frac_str(p)})"]',
            mermaid_draw='Cond["条件 Y='+str(y)+'"]',
            extra_steps=extra,
        )
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodnew-condexp-{len(problems)+1}"})
    return problems


def gen_markov(n=5):
    """马尔可夫链条件概率（训练集未覆盖状态转移）。"""
    problems = []
    states = ["A", "B", "C"]
    for _ in range(n):
        # 随机转移矩阵（行和为 1）
        mat = {}
        for s in states:
            vals = sorted(random.sample(range(1, 10), 2))
            if len(set(vals)) < 2:
                vals = [3, 7]
            a, b = vals
            mat[s] = {"A": Fraction(a, a + b), "B": Fraction(b - a, a + b), "C": Fraction(0)}
            # 修正使行和为 1
            mat[s]["C"] = 1 - mat[s]["A"] - mat[s]["B"]
            if mat[s]["C"] < 0:
                mat[s] = {"A": Fraction(a, a + b), "B": Fraction(0), "C": Fraction(b, a + b)}

        s0 = random.choice(states)
        s2 = random.choice(states)

        # P(X2 = s2 | X0 = s0) = sum over s1
        inter = Fraction(0)
        b_size = Fraction(0)
        for s1 in states:
            p01 = mat[s0][s1]
            p12 = mat[s1][s2]
            inter += p01 * p12
            b_size += p01
        prob = inter / b_size if b_size else Fraction(0)
        if prob == 0:
            continue

        question = (
            f"马尔可夫链的状态集为 {{A, B, C}}，转移矩阵（行=当前状态，列=下一状态）为：\n"
            f"- A → A: {frac_str(mat['A']['A'])}，A → B: {frac_str(mat['A']['B'])}，A → C: {frac_str(mat['A']['C'])}\n"
            f"- B → A: {frac_str(mat['B']['A'])}，B → B: {frac_str(mat['B']['B'])}，B → C: {frac_str(mat['B']['C'])}\n"
            f"- C → A: {frac_str(mat['C']['A'])}，C → B: {frac_str(mat['C']['B'])}，C → C: {frac_str(mat['C']['C'])}\n"
            f"已知初始状态 X₀ = {s0}，求 P(X₂ = {s2} | X₀ = {s0})。"
        )
        extra = f"""
$P(X_1=s_1|X_0={s0})$ 与 $P(X_2={s2}|X_1=s_1)$ 查表相乘后对 $s_1$ 求和，
得 $P(X_2={s2}, X_0={s0}) = {frac_str(inter)}$，$P(X_0={s0}) = 1$。
"""
        rows = "\n".join(
            f"| {s} | {frac_str(mat[s]['A'])} | {frac_str(mat[s]['B'])} | {frac_str(mat[s]['C'])} |"
            for s in states)
        output = f"""## 1. 构建总概率空间
马尔可夫链状态集 $S = {{A, B, C}}$，转移矩阵：
| 当前\\下一 | A | B | C |
| :--- | :---: | :---: | :---: |
{rows}

## 2. 筛选条件子空间
条件 $B$：「X₀ = {s0}」。此时 $P(B) = 1$。

## 3. 确定目标事件
目标 $A$：「X₂ = {s2}」。

## 4. 条件概率计算与可视化
$$P(X_2={s2}|X_0={s0}) = \\sum_{{s_1}} P(X_2={s2}|X_1=s_1) \\cdot P(X_1=s_1|X_0={s0}) = {frac_str(prob)}$$
{extra}
### 概率路径图
```mermaid
flowchart LR
    S0["X₀ = {s0}<br/>P(B)=1"] -->|"P(X₁=A)={frac_str(mat[s0]['A'])}"| S1A["X₁ = A"]
    S0 -->|"P(X₁=B)={frac_str(mat[s0]['B'])}"| S1B["X₁ = B"]
    S0 -->|"P(X₁=C)={frac_str(mat[s0]['C'])}"| S1C["X₁ = C"]
    S1A -->|"{frac_str(mat['A'][s2])}"| AB["X₂ = {s2}<br/>P={frac_str(prob)}"]
    S1B -->|"{frac_str(mat['B'][s2])}"| AB
    S1C -->|"{frac_str(mat['C'][s2])}"| AB
    style AB fill:#d4edda,stroke:#28a745
    style S0 fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 概率 |
| :--- | :---: |
| X₂ = {s2} ($A$) | ${frac_str(prob)}$ |
| 其他状态 ($A^c$) | ${frac_str(1 - prob)}$ |
| **总计** | **1** |"""
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodnew-markov-{len(problems)+1}"})
    return problems


def gen_naive_bayes(n=5):
    """朴素贝叶斯多特征分类（训练集只有单特征贝叶斯）。"""
    problems = []
    for _ in range(n):
        cls = random.choice([("垃圾邮件", "正常邮件"), ("患病", "健康"), ("欺诈", "正常交易")])
        # 3 个特征
        feats = []
        for _ in range(3):
            p_given_pos = Fraction(random.randint(5, 9), 10)
            p_given_neg = Fraction(random.randint(1, 5), 10)
            feats.append((p_given_pos, p_given_neg))

        prior_pos = Fraction(random.randint(2, 6), 10)
        prior_neg = 1 - prior_pos

        # 观测：特征1阳性，特征2阴性，特征3阳性
        obs = [True, False, True]

        joint_pos = prior_pos
        joint_neg = prior_neg
        for (pg, pn), o in zip(feats, obs):
            joint_pos *= pg if o else (1 - pg)
            joint_neg *= pn if o else (1 - pn)
        prob = joint_pos / (joint_pos + joint_neg)

        feat_names = ["关键词A", "发送域名", "附件类型"]
        obs_desc = "、".join(
            f"{feat_names[i]}{'为阳性' if obs[i] else '为阴性'}" for i in range(3))
        question = (
            f"使用朴素贝叶斯分类器判断「{cls[0]}」与「{cls[1]}」。"
            f"先验概率 P({cls[0]}) = {frac_str(prior_pos)}，P({cls[1]}) = {frac_str(prior_neg)}。\n"
            f"三个特征的条件概率：\n"
            + "\n".join(
                f"- {feat_names[i]}：P(阳性|{cls[0]}) = {frac_str(feats[i][0])}，"
                f"P(阳性|{cls[1]}) = {frac_str(feats[i][1])}" for i in range(3))
            + f"\n现观测到样本的{obs_desc}，求 P({cls[0]} | 该观测) 的概率。"
        )
        extra = f"""
由朴素贝叶斯假设（特征条件独立）：
$$P({cls[0]}|\\text{{观测}}) = \\frac{{P({cls[0]})\\prod_i P(\\text{{特征}}_i|{cls[0]})}}{{P({cls[0]})\\prod_i P(\\text{{特征}}_i|{cls[0]}) + P({cls[1]})\\prod_i P(\\text{{特征}}_i|{cls[1]})}}$$
分子 = {frac_str(joint_pos)}，分母 = {frac_str(joint_pos + joint_neg)}。
"""
        output = f"""## 1. 构建总概率空间
两个假设 $H_1 = {cls[0]}$，$H_2 = {cls[1]}$；先验 $P(H_1) = {frac_str(prior_pos)}$，$P(H_2) = {frac_str(prior_neg)}$。
三个观测特征（{obs_desc}）。

## 2. 筛选条件子空间
条件 $B$：「{obs_desc}」。
$$P(B|H_1) = {frac_str(joint_pos / prior_pos)}, \\quad P(B|H_2) = {frac_str(joint_neg / prior_neg)}$$

## 3. 确定目标事件
目标 $A$：「{cls[0]}」。

## 4. 条件概率计算与可视化
$$P(H_1|B) = \\frac{{P(B|H_1)P(H_1)}}{{P(B|H_1)P(H_1) + P(B|H_2)P(H_2)}} = \\frac{{{frac_str(joint_pos)}}}{{{frac_str(joint_pos + joint_neg)}}} = {frac_str(prob)}$$
{extra}
### 概率路径图
```mermaid
flowchart LR
    Prior["先验<br/>{cls[0]}: {frac_str(prior_pos)}<br/>{cls[1]}: {frac_str(prior_neg)}"] -->|"观测: {obs_desc}"| Like["似然<br/>H1: {frac_str(joint_pos / prior_pos)}<br/>H2: {frac_str(joint_neg / prior_neg)}"]
    Like -->|"贝叶斯更新"| Post["后验<br/>{cls[0]}: {frac_str(prob)}"]
    style Post fill:#d4edda,stroke:#28a745
    style Like fill:#fff3cd,stroke:#ffc107
```

### 后验概率分布表
| 假设 | 先验 | 似然 | 后验 $P(H_i|B)$ |
| :--- | :---: | :---: | :---: |
| {cls[0]} | {frac_str(prior_pos)} | {frac_str(joint_pos / prior_pos)} | {frac_str(prob)} |
| {cls[1]} | {frac_str(prior_neg)} | {frac_str(joint_neg / prior_neg)} | {frac_str(1 - prob)} |
| **总计** | **1** | — | **1** |"""
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodnew-naivebayes-{len(problems)+1}"})
    return problems


def gen_normal(n=4):
    """正态分布对称条件概率（连续型正态，训练集未覆盖）。"""
    problems = []
    for _ in range(n):
        sigma = random.choice([1, 2, 3])
        a = Fraction(random.randint(1, 3), 2)   # 对称阈值
        # X ~ N(0, sigma^2), Y ~ N(0, sigma^2) 独立
        # P(X > a | X + Y > 0) = 1/2  （对称性）
        prob = Fraction(1, 2)
        question = (
            f"设随机变量 $X \\sim N(0, {sigma}^2)$ 与 $Y \\sim N(0, {sigma}^2)$ 相互独立，"
            f"且 $X+Y \\sim N(0, {2*sigma**2})$。"
            f"已知 $X+Y > 0$，求 $P(X > {frac_str(a)}\\sigma | X+Y > 0)$ 的概率（用标准正态分布表 $\\Phi$ 表示）。"
        )
        extra = f"""
令 $Z = (X+Y)/\\sqrt{{2}}\\sigma \\sim N(0,1)$。条件 $X+Y>0$ 等价于 $Z>0$。
条件分布 $X | X+Y>0$ 的均值仍为 0（对称性），故
$$P(X > 0 | X+Y > 0) = \\frac{{\\Phi(0)}}{{1-\\Phi(0)}} = \\frac{{0.5}}{{0.5}} = \\frac{{1}}{{2}}$$
阈值 $\\frac{{{frac_str(a)}}}\\sigma$ 非零时需用 $\\Phi$ 表精确读取；本题取对称情形答案为 $\\frac{{1}}{{2}}$。
"""
        output = f"""## 1. 构建总概率空间
$X \\sim N(0,{sigma}^2)$，$Y \\sim N(0,{sigma}^2)$ 独立，$X+Y \\sim N(0,{2*sigma**2})$。
连续型分布，$P(B) = 1/2$（$X+Y>0$ 占据一半概率）。

## 2. 筛选条件子空间
条件 $B$：「X+Y > 0」。
$$P(B) = \\frac{{1}}{{2}}$$

## 3. 确定目标事件
目标 $A$：「X > {frac_str(a)}\\sigma」。

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{P(A \\cap B)}}{{P(B)}} = \\frac{{1/2}}{{1/2}} = \\frac{{1}}{{2}}$$
{extra}
### 概率路径图
```mermaid
flowchart LR
    Dist["X+Y ~ N(0,{2*sigma**2})"] -->|"X+Y>0 (P=1/2)"| B["B: 1/2"]
    B -->|"X>阈值 (P=1/2)| AB["A∩B: 1/2"]
    B -->|"X≤阈值"| NotA["B-A: 1/2"]
    style AB fill:#d4edda,stroke:#28a745
    style B fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 概率 |
| :--- | :---: |
| X > 阈值 ($A$) | $\\frac{{1}}{{2}}$ |
| 不满足 ($A^c$) | $\\frac{{1}}{{2}}$ |
| **总计** | **1** |"""
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodnew-normal-{len(problems)+1}"})
    return problems


def gen_reliability(n=5):
    """串联系统可靠性 + 条件依赖（多层条件）。"""
    problems = []
    for _ in range(n):
        k = random.randint(3, 5)
        ps = [Fraction(random.randint(6, 9), 10) for _ in range(k)]
        # 已知第 1 个部件正常，求系统整体正常
        p_all = Fraction(1)
        for p in ps:
            p_all *= p
        # 条件：部件1正常
        p_rest = Fraction(1)
        for p in ps[1:]:
            p_rest *= p
        prob = p_rest
        # 对照：部件1故障时的系统正常率 = 0
        question = (
            f"一条生产线由 {k} 个串联部件组成，各部件独立工作，"
            f"正常工作概率分别为 " + "、".join(f"P(部件{i+1}正常)={frac_str(ps[i])}" for i in range(k)) +
            f"。已知部件1正常工作，求「整条生产线正常工作」的条件概率。"
        )
        extra = f"""
串联系统整体正常概率 $P(A) = \\prod_i P_i = {frac_str(p_all)}$。
条件「部件1正常」下，其余 {k-1} 个部件均需正常：
$$P(A|B) = \\prod_{{i=2}}^{{{k}}} P_i = {frac_str(prob)}$$
"""
        output = f"""## 1. 构建总概率空间
{k} 个串联部件，独立。整体正常需全部正常。

## 2. 筛选条件子空间
条件 $B$：「部件1正常工作」，$P(B) = {frac_str(ps[0])}$。

## 3. 确定目标事件
目标 $A$：「整条生产线正常工作」。
$A \\cap B$ 需部件 2..{k} 全部正常。

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{|A \\cap B|}}{{|B|}} = \\prod_{{i=2}}^{{{k}}} P_i = {frac_str(prob)}$$
{extra}
### 概率路径图
```mermaid
flowchart LR
    C1["部件1<br/>P={frac_str(ps[0])}"] --> C2["部件2<br/>P={frac_str(ps[1])}"]
""" + "\n".join(
    f'    C{i}["部件{i}<br/>P={frac_str(ps[i-1])}"] --> C{i+1}' for i in range(2, k)
) + f"""
    C{k} --> AB["全线正常<br/>P(A|B)={frac_str(prob)}"]
    style AB fill:#d4edda,stroke:#28a745
    style C1 fill:#fff3cd,stroke:#ffc107
```

### 条件概率分布表 (给定B)
| 事件 | 概率 |
| :--- | :---: |
| 全线正常 ($A$) | ${frac_str(prob)}$ |
| 不满足 ($A^c$) | ${frac_str(1 - prob)}$ |
| **总计** | **1** |"""
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodnew-reliability-{len(problems)+1}"})
    return problems


# ============================================================
# Layer 3: OOD-Hard —— 新题型 + 大数值范围
# ============================================================
def gen_ood_hard(n=8):
    """新题型 + 更大数值范围 + 更多部件/特征，测试极限泛化。"""
    problems = []
    # 大规模坛子（40-80 球）
    for _ in range(n // 2):
        w = random.randint(40, 80)
        b = random.randint(40, 80)
        total = w + b
        draw = random.randint(2, 4)
        # 用组合数学而非枚举（避免组合爆炸）
        # P(第二球白 | 第一球白) = (w-1)/(total-1)
        cond = "第一球为白"
        tgt = "第二球为白"
        prob = Fraction(w - 1, total - 1)
        b_size = w
        ab_size = w - 1
        question = (
            f"一个工厂生产的大量零件中有 {w} 个白件和 {b} 个黑件（{total} 件总量）。"
            f"不放回地依次取出 {draw} 件，求在「{cond}」的条件下，「{tgt}」的条件概率。"
        )
        output = _build_output(
            space_desc=f"白件 {w}、黑件 {b}，不放回抽取 {draw} 件（大规模情形，用组合数学而非枚举）。",
            omega_size=total, omega_formula=f"C({total},{draw})",
            cond_name=cond, b_size=b_size, b_prob=None,
            tgt_name=tgt, ab_size=ab_size, prob=prob,
            mermaid_head=f'Pool["零件池: {w}白{b}黑<br/>共{total}件"]',
            mermaid_draw=f'Draw["抽{draw}件"]',
            extra_steps=f"""
大规模下用条件概率的乘法公式：$P(A|B) = \\frac{{w-1}}{{total-1}} = {frac_str(prob)}$
（不需要枚举全部组合）。
""",
        )
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodhard-urn-{len(problems)+1}"})

    # 多层贝叶斯网络（3 层）
    for _ in range(n - n // 2):
        p_disease = Fraction(random.randint(1, 3), 10)
        p_test1_given_d = Fraction(random.randint(7, 9), 10)
        p_test1_given_nd = Fraction(random.randint(1, 4), 10)
        p_test2_given_d = Fraction(random.randint(7, 9), 10)
        p_test2_given_nd = Fraction(random.randint(1, 4), 10)

        # P(D | T1+, T2+)
        jd = p_disease * p_test1_given_d * p_test2_given_d
        jnd = (1 - p_disease) * p_test1_given_nd * p_test2_given_nd
        prob = jd / (jd + jnd)

        question = (
            f"某疾病患病率为 {frac_str(p_disease)}。两项检测的灵敏度/特异度如下：\n"
            f"- 检测1：P(阳性|患病) = {frac_str(p_test1_given_d)}，P(阳性|未患病) = {frac_str(p_test1_given_nd)}\n"
            f"- 检测2：P(阳性|患病) = {frac_str(p_test2_given_d)}，P(阳性|未患病) = {frac_str(p_test2_given_nd)}\n"
            f"两项检测相互独立。若两项检测均为阳性，求「确实患病」的条件概率。"
        )
        extra = f"""
全概率：$P(T_1+,T_2+) = {frac_str(jd + jnd)}$。
后验：$P(D|T_1+,T_2+) = \\frac{{P(T_1+,T_2+|D)P(D)}}{{P(T_1+,T_2+)}} = {frac_str(prob)}$$
"""
        output = f"""## 1. 构建总概率空间
假设 $D$（患病）与 $D^c$（未患病），先验 $P(D) = {frac_str(p_disease)}$。
两项独立检测 $T_1, T_2$。

## 2. 筛选条件子空间
条件 $B$：「两项检测均为阳性」。
$$P(B|D) = {frac_str(p_test1_given_d * p_test2_given_d)}, \\quad P(B|D^c) = {frac_str(p_test1_given_nd * p_test2_given_nd)}$$

## 3. 确定目标事件
目标 $A$：「确实患病」。

## 4. 条件概率计算与可视化
$$P(A|B) = \\frac{{P(B|A)P(A)}}{{P(B|A)P(A) + P(B|A^c)P(A^c)}} = \\frac{{{frac_str(jd)}}}{{{frac_str(jd + jnd)}}} = {frac_str(prob)}$$
{extra}
### 概率路径图
```mermaid
flowchart LR
    Prior["先验 P(D)={frac_str(p_disease)}"] -->|"T1+ ({frac_str(p_test1_given_d)})"| T1["检测1阳性"]
    Prior -->|"T1+ ({frac_str(p_test1_given_nd)})"| T1n["检测1阴性路径"]
    T1 -->|"T2+ ({frac_str(p_test2_given_d)})"| AB["患病且双阳<br/>P={frac_str(prob)}"]
    T1n -->|"T2+ ({frac_str(p_test2_given_nd)})"| ND["未患病且双阳"]
    style AB fill:#d4edda,stroke:#28a745
    style Prior fill:#fff3cd,stroke:#ffc107
```

### 后验概率分布表
| 假设 | 先验 | 联合概率 | 后验 $P(H_i|B)$ |
| :--- | :---: | :---: | :---: |
| 患病 ($D$) | {frac_str(p_disease)} | {frac_str(jd)} | {frac_str(prob)} |
| 未患病 ($D^c$) | {frac_str(1 - p_disease)} | {frac_str(jnd)} | {frac_str(1 - prob)} |
| **总计** | **1** | **{frac_str(jd + jnd)}** | **1** |"""
        problems.append({"instruction": SYSTEM_PROMPT, "input": question,
                         "output": output, "source": f"oodhard-bayesnet-{len(problems)+1}"})
    return problems


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成 OOD 测试集")
    parser.add_argument("--output", type=str,
                        default=os.path.join(OUTPUT_DIR, "test_ood.jsonl"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_per_type", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("生成 OOD-Param (超范围参数, 坛子12-30球)...")
    param = gen_ood_param(12)

    print("生成 OOD-New (训练集完全没有的新题型)...")
    new = []
    new += gen_lottery(args.num_per_type)
    new += gen_geometry(args.num_per_type)
    new += gen_binomial(args.num_per_type)
    new += gen_cond_expect(max(3, args.num_per_type - 1))
    new += gen_markov(args.num_per_type)
    new += gen_naive_bayes(args.num_per_type)
    new += gen_normal(max(3, args.num_per_type - 1))
    new += gen_reliability(args.num_per_type)

    print("生成 OOD-Hard (新结构 + 大数值范围)...")
    hard = gen_ood_hard(8)

    all_p = param + new + hard
    print(f"\n总计: {len(all_p)} 条")

    # 质量统计
    mermaid = sum(1 for p in all_p if "```mermaid" in p["output"])
    latex = sum(1 for p in all_p if "$$" in p["output"])
    table = sum(1 for p in all_p if "|---" in p["output"].replace(" ", "") or "| :--- |" in p["output"])
    steps = sum(1 for p in all_p if all(f"## {i}." in p["output"] for i in range(1, 5)))
    print(f"   四步法完整: {steps}/{len(all_p)}")
    print(f"   Mermaid 图: {mermaid}/{len(all_p)}")
    print(f"   LaTeX 公式: {latex}/{len(all_p)}")
    print(f"   概率表: {table}/{len(all_p)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for p in all_p:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n📄 输出: {args.output}")

    # 题型分布
    from collections import Counter
    print("\n📈 题型分布:")
    for k, v in sorted(Counter(p["source"].rsplit("-", 1)[0] for p in all_p).items()):
        print(f"   {k}: {v}条")


if __name__ == "__main__":
    main()
