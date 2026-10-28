#!/usr/bin/env python3
"""
eval_kg_quality.py
Phase5 / P2: GraphRAG 知识图谱质量量化评估

背景 (ref: external/diage2.md 第三层 P2):
  "无 GraphRAG vs 有 GraphRAG 的对比实验
   Emoji↔Numerical 对齐边的查询准确率
   同源题目聚类效果（不同 Emoji 表征能否被正确识别为同一数学结构）"

本脚本覆盖 P2 中可在无 GPU 条件下量化的两项（b、c）：

  P2-b  Emoji↔Numerical 对齐边查询准确率
        B1 内在一致性  — emoji_to_num 的目标数值是否真出现在该题目的
                         has_param 数值集合中（KG 内部自洽性）
        B2 留出预测    — 遮蔽 20% emoji_to_num 边，用
                         Source→ProblemText→Numerical 路径预测，算命中率
        B3 端到端查询  — 给定 Emoji 查询数值，检查是否语义自洽

  P2-c  同源题目聚类效果
        C1 纯度 (purity)
        C2 归一化互信息 NMI
        C3 调整兰德指数 ARI
        C4 Emoji 表征同一化 — 不同 Emoji 是否被识别为同一数学结构

输出: output/kg_quality.json
"""
import json
import random
import re
import math
import argparse
from collections import defaultdict, Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KG_JSON = BASE_DIR / "data" / "kg_structure.json"
SFT_FILE = BASE_DIR / "data" / "sft_train_580.jsonl"


# ============================================================
# 工具
# ============================================================
def rtype(desc):
    """关系类型判定。"""
    if "同源" in desc:
        return "same_source"
    if "包含数值" in desc:
        return "has_param"
    if "生成" in desc:
        return "generates"
    if "Emoji" in desc and "对应" in desc:
        return "emoji_to_num"
    if "Emoji 表征" in desc:
        return "represents"
    if "解答步骤" in desc:
        return "visualizes"
    if "计算得到" in desc:
        return "computes_to"
    return "other"


def load_kg():
    kg = json.loads(KG_JSON.read_text(encoding="utf-8"))
    nodes = {e["entity_name"]: e for e in kg["entities"]}
    edges = defaultdict(list)          # src -> [(tgt, type, weight)]
    for r in kg["relationships"]:
        t = rtype(r["description"])
        edges[r["src_id"]].append((r["tgt_id"], t, r.get("weight", 1.0)))
    return nodes, edges, kg


def strip_node(name):
    """去掉节点前缀（题目:/数值:/Emoji:/原型:）。"""
    for p in ("题目:", "数值:", "Emoji:", "原型:", "步骤:"):
        if name.startswith(p):
            return name[len(p):]
    return name


# ============================================================
# 独立 ground truth：题目 → 数学原型
# 依据题目文本的人工判定（与 KG 的 generates 边无关，独立于 KG）
# ============================================================
PROTO_MANUAL = {
    "ex3.1": "骰子条件概率", "ex3.2": "骰子条件概率",
    "ex3.4": "骰子条件概率", "ex3.22": "骰子条件概率",
    "ex3.9": "坛子抽样", "ex3.14": "坛子抽样", "ex3.23": "坛子抽样",
    "ex3.30": "坛子抽样", "ex3.38": "坛子抽样", "ex3.40": "坛子抽样",
    "ex3.47": "坛子抽样",
    "ex3.37": "硬币抛掷", "ex3.43": "硬币抛掷", "ex3.45": "硬币抛掷",
    "ex3.16": "双向表贝叶斯", "ex3.17": "双向表贝叶斯",
    "ex3.19": "双向表贝叶斯", "ex3.20": "双向表贝叶斯",
    "ex3.21": "双向表贝叶斯", "ex3.25": "双向表贝叶斯",
    "ex3.26": "双向表贝叶斯", "ex3.33": "双向表贝叶斯",
    "ex3.42": "双向表贝叶斯", "ex3.50": "双向表贝叶斯",
    "ex3.7": "多阶段路径", "ex3.8": "多阶段路径",
    "ex3.10": "多阶段路径", "ex3.11": "多阶段路径", "ex3.48": "多阶段路径",
    "ex3.12": "多阶段路径",
}

SYNTH_PROTO = {
    "synth-dice": "骰子条件概率",
    "synth-urn": "坛子抽样",
    "synth-coin": "硬币抛掷",
    "synth-table": "双向表贝叶斯",
    "synth-path": "多阶段路径",
}


def build_ground_truth(sft_file):
    """构造 source → 数学原型 的独立 ground truth。"""
    truth = {}
    if not Path(sft_file).exists():
        return truth
    for line in Path(sft_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("source", "")
        node = f"题目:{s}"
        if s.startswith("afcp-"):
            key = s[len("afcp-"):].rsplit(".", 1)[0]
            truth[node] = PROTO_MANUAL.get(key)
        else:
            prefix = s.rsplit("-", 1)[0] if "-" in s else s
            truth[node] = SYNTH_PROTO.get(prefix)
    return {k: v for k, v in truth.items() if v}


# ============================================================
# P2-b: Emoji↔Numerical 对齐边查询准确率
# ============================================================
def eval_emoji_numeric(nodes, edges, seed=42, holdout_ratio=0.2):
    """评估 emoji_to_num 边质量。"""
    print("\n" + "=" * 64)
    print("P2-b  Emoji↔Numerical 对齐边查询准确率")
    print("=" * 64)

    # 收集 emoji_to_num 边
    e2n = [(src, tgt) for src, lst in edges.items() for tgt, t, _ in lst
           if t == "emoji_to_num"]
    print(f"\nemoji_to_num 边总数: {len(e2n)}")

    # 建立 Emoji → 关联题目（通过 represents 边）
    emoji_to_problems = defaultdict(set)
    for src, lst in edges.items():
        for tgt, t, _ in lst:
            if t == "represents":
                # represents: Source → Emoji（Emoji 表征某原型）
                emoji_to_problems[tgt].add(src)
    # ProblemText → Emoji 的反向：Emoji 节点描述里带题目名
    emoji_problems = defaultdict(set)
    for name, e in nodes.items():
        if e["type"] == "Emoji":
            desc = e.get("description", "")
            m = re.search(r"对应题目\s*([\w.\-]+)", desc)
            if m:
                emoji_problems[name].add(f"题目:{m.group(1)}")

    # ProblemText → has_param 数值
    problem_params = defaultdict(set)
    for src, lst in edges.items():
        for tgt, t, _ in lst:
            if t == "has_param":
                problem_params[src].add(tgt)

    # --- B1 内在一致性 ---
    # 关联路径：Emoji ←represents— Source —generates→ ProblemText —has_param→ Numerical
    source_of_emoji = defaultdict(set)
    problems_of_source = defaultdict(set)
    for src, lst in edges.items():
        for tgt, t, _ in lst:
            if t == "represents":
                source_of_emoji[tgt].add(src)
            elif t == "generates":
                problems_of_source[src].add(tgt)

    def linked_params(emoji):
        out = set()
        for s in source_of_emoji.get(emoji, set()):
            for p in problems_of_source.get(s, set()):
                out |= problem_params.get(p, set())
        return out

    consistent, inconsistent, orphan = 0, 0, 0
    details = []
    for src, tgt in e2n:
        params = linked_params(src)
        if not params:
            orphan += 1
            status = "orphan"
        elif tgt in params:
            consistent += 1
            status = "ok"
        else:
            inconsistent += 1
            status = "mismatch"
        details.append({"emoji": src, "numerical": tgt, "status": status})
    total = len(e2n)
    b1_rate = consistent / total if total else 0
    print(f"\n--- B1 内在一致性 (KG 内部自洽) ---")
    print(f"  一致        : {consistent}/{total} ({b1_rate*100:.1f}%)")
    print(f"  不一致      : {inconsistent}/{total} ({inconsistent/total*100:.1f}%)")
    print(f"  无关联题目  : {orphan}/{total} ({orphan/total*100:.1f}%)")

    # --- B2 留出预测 ---
    # 推理路径（KG 中未被遮蔽的结构）：
    #   Emoji ←represents— Source —generates→ ProblemText —has_param→ Numerical
    rng = random.Random(seed)
    shuffled = e2n[:]
    rng.shuffle(shuffled)
    n_hold = max(1, int(len(shuffled) * holdout_ratio))
    holdout = shuffled[:n_hold]
    remaining = shuffled[n_hold:]

    source_of_emoji = defaultdict(set)     # Emoji -> {Source}
    problems_of_source = defaultdict(set)   # Source -> {ProblemText}
    for src, lst in edges.items():
        for tgt, t, _ in lst:
            if t == "represents":
                source_of_emoji[tgt].add(src)
            elif t == "generates":
                problems_of_source[src].add(tgt)

    def infer_numericals(emoji):
        """由 KG 结构推断某 Emoji 可对应的数值集合。"""
        out = set()
        for s in source_of_emoji.get(emoji, set()):
            for p in problems_of_source.get(s, set()):
                out |= problem_params.get(p, set())
        return out

    hit, miss = 0, 0
    miss_examples = []
    for src, tgt in holdout:
        candidates = infer_numericals(src)
        if tgt in candidates:
            hit += 1
        else:
            miss += 1
            if len(miss_examples) < 5:
                miss_examples.append({
                    "emoji": src, "truth": tgt,
                    "candidates": sorted(candidates)[:6],
                })
    b2_rate = hit / (hit + miss) if (hit + miss) else 0
    print(f"\n--- B2 留出预测 (遮蔽 {holdout_ratio*100:.0f}% 边后经三跳路径预测) ---")
    print(f"  命中: {hit}/{hit+miss} ({b2_rate*100:.1f}%)")
    for e in miss_examples:
        print(f"    miss: {e['emoji']} -> 真值 {e['truth']}, 候选 {e['candidates']}")

    # --- B3 端到端查询 ---
    # 对每个 Emoji 查询其数值，检查数值描述与 Emoji 语义是否相容
    # 相容规则：数值名中若含颜色/形状类词，与 Emoji 集合字形对应
    EMOJI_TO_MEANING = {
        "⚪": "白", "⚫": "黑", "🔴": "红", "🟣": "紫", "🟠": "橙",
        "🟢": "绿", "🔵": "蓝", "🟡": "黄", "⚀": "骰", "⚁": "骰",
    }
    ok3, bad3 = 0, 0
    q_details = []
    for src, tgt in e2n:
        emoji_chars = strip_node(src)
        num = strip_node(tgt)
        # 数值名含某颜色词，且 Emoji 集合含对应字形 → 相容
        compatible = False
        for ch, meaning in EMOJI_TO_MEANING.items():
            if ch in emoji_chars and meaning in num:
                compatible = True
                break
        if compatible:
            ok3 += 1
        else:
            bad3 += 1
        q_details.append({"emoji": src, "numerical": tgt, "compatible": compatible})
    b3_rate = ok3 / (ok3 + bad3) if (ok3 + bad3) else 0
    print(f"\n--- B3 端到端查询语义相容 ---")
    print(f"  相容: {ok3}/{ok3+bad3} ({b3_rate*100:.1f}%)")

    # 量化"信息丢失"：Emoji 序列含多种颜色，但 KG 只映射到单一数值节点
    multi_symbol = 0
    for src, tgt in e2n:
        chars = strip_node(src)
        meanings = {m for ch, m in EMOJI_TO_MEANING.items() if ch in chars}
        if len(meanings) > 1:
            multi_symbol += 1
    print(f"  其中 Emoji 序列含多种颜色符号（数值信息可能未被完全覆盖）: "
          f"{multi_symbol}/{ok3+bad3} ({multi_symbol/(ok3+bad3)*100:.1f}%)")

    print(f"\n  >> P2-b 综合查询准确率: {(b1_rate+b2_rate+b3_rate)/3*100:.1f}%")

    return {
        "b1_consistency": {"pass": consistent, "n": total, "rate": b1_rate,
                           "inconsistent": inconsistent, "orphan": orphan},
        "b2_holdout": {"hit": hit, "n": hit + miss, "rate": b2_rate,
                       "holdout_size": n_hold},
        "b3_query_compat": {"pass": ok3, "n": ok3 + bad3, "rate": b3_rate},
        "overall_rate": (b1_rate + b2_rate + b3_rate) / 3,
        "details": details,
    }


# ============================================================
# P2-c: 同源题目聚类效果
# ============================================================
def eval_same_source_clustering(nodes, edges, sft_file=SFT_FILE):
    """评估 same_source 边的聚类质量。"""
    print("\n" + "=" * 64)
    print("P2-c  同源题目聚类效果")
    print("=" * 64)

    # Ground truth：题目 → 数学原型（独立于 KG，由人工判定 + 合成脚本题型）
    truth = build_ground_truth(sft_file)

    # KG 聚类：same_source 边的连通分量
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    same_edges = [(src, tgt) for src, lst in edges.items() for tgt, t, _ in lst
                  if t == "same_source"]
    for s, t in same_edges:
        union(s, t)
    print(f"\nsame_source 边数: {len(same_edges)}")

    # 聚类 → 成员
    clusters = defaultdict(set)
    for node in parent:
        clusters[find(node)].add(node)

    # 只评估有 ground truth 的题目
    labeled = [(n, truth[n]) for c in clusters.values() for n in c if n in truth]
    print(f"有 ground truth 的题目: {len(labeled)}")
    if not labeled:
        return {}

    # --- C1 纯度 ---
    total = len(labeled)
    correct = 0
    for members in clusters.values():
        ms = [n for n in members if n in truth]
        if not ms:
            continue
        labels = [truth[n] for n in ms]
        correct += Counter(labels).most_common(1)[0][1]
    purity = correct / total
    print(f"\n--- C1 聚类纯度 ---")
    print(f"  纯度: {correct}/{total} ({purity*100:.1f}%)")

    # --- C2 NMI ---
    cluster_of = {n: find(n) for n in truth}
    a = [cluster_of[n] for n, _ in labeled]
    b = [lbl for _, lbl in labeled]
    nmi = _nmi(a, b)
    print(f"\n--- C2 归一化互信息 (NMI) ---")
    print(f"  NMI: {nmi:.4f}")

    # --- C3 ARI ---
    ari = _ari(a, b)
    print(f"\n--- C3 调整兰德指数 (ARI) ---")
    print(f"  ARI: {ari:.4f}")

    # --- C4 不同 Emoji 表征是否识别为同一结构 ---
    # 对每个 Emoji 节点，找其关联题目所属的聚类，检查同一原型的不同 Emoji
    # 是否落在同一聚类（即"不同表征 → 同一结构"是否被正确识别）
    emoji_nodes = [n for n, e in nodes.items() if e["type"] == "Emoji"]
    emoji_clusters = defaultdict(list)
    for en in emoji_nodes:
        desc = nodes[en].get("description", "")
        m = re.search(r"对应题目\s*([\w.\-]+)", desc)
        if not m:
            continue
        pname = f"题目:{m.group(1)}"
        if pname in cluster_of:
            emoji_clusters[cluster_of[pname]].append(en)

    multi = {c: es for c, es in emoji_clusters.items() if len(es) > 1}
    # 覆盖率：有多少 Emoji 被正确归入其真实原型簇
    emoji_total = sum(len(es) for es in emoji_clusters.values())
    emoji_correct = 0
    for c, es in emoji_clusters.items():
        labels = []
        for en in es:
            desc = nodes[en].get("description", "")
            m = re.search(r"类型\s*(\S+)", desc)
            labels.append(m.group(1) if m else "?")
        if labels:
            emoji_correct += Counter(labels).most_common(1)[0][1]
    c4_rate = emoji_correct / emoji_total if emoji_total else 0
    print(f"\n--- C4 不同 Emoji 表征同一化 ---")
    print(f"  Emoji 节点: {emoji_total}，其中 {len(multi)} 个簇含多个不同表征")
    print(f"  表征归类正确率: {emoji_correct}/{emoji_total} ({c4_rate*100:.1f}%)")

    # 展示聚类样例
    print(f"\n  聚类样例 (前 5 个):")
    for i, (c, ms) in enumerate(list(clusters.items())[:5]):
        lbls = Counter(truth.get(n, "?") for n in ms).most_common(2)
        print(f"    簇{i}: {len(ms)}题, 主要标签 {lbls}")

    # 错分样例：定位 KG 的具体缺陷
    misassigned = []
    for members in clusters.values():
        ms = [n for n in members if n in truth]
        if len(ms) < 2:
            continue
        labels = [truth[n] for n in ms]
        major = Counter(labels).most_common(1)[0][0]
        for n, lbl in zip(ms, labels):
            if lbl != major:
                misassigned.append({"node": n, "true": lbl, "cluster": major})
    print(f"\n  错分样例 ({len(misassigned)} 条):")
    for m in misassigned[:8]:
        print(f"    {m['node']}: 真值={m['true']}, 被归入={m['cluster']}")

    return {
        "c1_purity": purity,
        "c2_nmi": nmi,
        "c3_ari": ari,
        "c4_emoji_same_structure": c4_rate,
        "n_labeled": total,
        "n_clusters": len(clusters),
        "n_same_source_edges": len(same_edges),
    }


def _nmi(a, b):
    """归一化互信息。"""
    n = len(a)
    if n == 0:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    cab = Counter(zip(a, b))
    mi = 0.0
    for (x, y), c in cab.items():
        pxy = c / n
        mi += pxy * math.log(pxy / ((ca[x] / n) * (cb[y] / n)))
    ha = -sum((c / n) * math.log(c / n) for c in ca.values())
    hb = -sum((c / n) * math.log(c / n) for c in cb.values())
    if ha == 0 or hb == 0:
        return 0.0
    return 2 * mi / (ha + hb)


def _ari(a, b):
    """调整兰德指数。"""
    n = len(a)
    if n < 2:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    cab = Counter(zip(a, b))
    comb2 = lambda x: x * (x - 1) / 2
    si = sum(comb2(c) for c in cab.values())
    sa = sum(comb2(c) for c in ca.values())
    sb = sum(comb2(c) for c in cb.values())
    total = comb2(n)
    exp = sa * sb / total if total else 0
    maxi = (sa + sb) / 2
    if maxi == exp:
        return 0.0
    return (si - exp) / (maxi - exp)


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="KG 质量量化评估")
    parser.add_argument("--output", type=str,
                        default=str(BASE_DIR / "output" / "kg_quality.json"))
    args = parser.parse_args()

    nodes, edges, kg = load_kg()
    print(f"已加载 KG: {len(nodes)} 节点, {sum(len(v) for v in edges.values())} 边")

    res_b = eval_emoji_numeric(nodes, edges)
    res_c = eval_same_source_clustering(nodes, edges)

    result = {
        "kg_stats": {
            "nodes": len(nodes),
            "edges": sum(len(v) for v in edges.values()),
            "node_types": dict(Counter(e["type"] for e in kg["entities"])),
        },
        "p2b_emoji_numeric": res_b,
        "p2c_same_source_clustering": res_c,
    }

    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 结果: {args.output}")

    # 汇总
    print("\n" + "=" * 64)
    print("P2 汇总")
    print("=" * 64)
    if res_b:
        print(f"  P2-b Emoji↔Numerical 查询准确率: {res_b['overall_rate']*100:.1f}%")
        print(f"       B1 一致性 {res_b['b1_consistency']['rate']*100:.1f}% | "
              f"B2 留出 {res_b['b2_holdout']['rate']*100:.1f}% | "
              f"B3 语义 {res_b['b3_query_compat']['rate']*100:.1f}%")
    if res_c:
        print(f"  P2-c 同源聚类: 纯度 {res_c['c1_purity']*100:.1f}% | "
              f"NMI {res_c['c2_nmi']:.3f} | ARI {res_c['c3_ari']:.3f} | "
              f"Emoji同构 {res_c['c4_emoji_same_structure']*100:.1f}%")


if __name__ == "__main__":
    main()
