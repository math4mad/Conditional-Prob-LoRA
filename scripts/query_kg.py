#!/usr/bin/env python3
"""
query_kg.py
基于知识图谱的检索查询接口。

支持三类查询（对应 dialogue1.md 的设计）：
1. 同源题目检索：给定一个题目，找所有同源（same_source边）的其他题目。
2. 表征转换查询：给定 Emoji 或数值，沿 emoji_to_num / has_param 边找到对应题目。
3. 图谱统计：节点/边分布、连通分量、同源聚类。
"""

import pickle
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent.parent
GRAPH_PKL = BASE_DIR / "lightrag_data" / "kg_graph.pkl"
KG_JSON = BASE_DIR / "data" / "kg_structure.json"


def load_graph():
    """加载 NetworkX 图。"""
    if GRAPH_PKL.exists():
        with open(GRAPH_PKL, "rb") as f:
            return pickle.load(f)
    # 从 JSON 重建
    with open(KG_JSON) as f:
        kg = json.load(f)
    G = nx.DiGraph()
    for e in kg["entities"]:
        G.add_node(e["entity_name"], type=e["type"], description=e["description"])
    for r in kg["relationships"]:
        G.add_edge(r["src_id"], r["tgt_id"], description=r["description"], weight=r["weight"])
    return G


def find_same_source(G, problem_name: str) -> list[str]:
    """查询同源题目（沿 same_source 边）。"""
    # 找到节点
    node = None
    for n in G.nodes():
        if problem_name in n:
            node = n
            break
    if not node:
        return []

    # 沿 same_source 边遍历（无向）
    results = set()
    # BFS 同源邻居
    queue = [node]
    visited = {node}
    while queue:
        curr = queue.pop(0)
        for u, v, d in G.edges(curr, data=True):
            desc = d.get("description", "")
            if "同源" in desc:
                neighbor = v if u == curr else u
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if neighbor.startswith("题目:"):
                        results.add(neighbor)
    return sorted(results)


def find_emoji_to_numerical(G, emoji: str) -> list[dict]:
    """Emoji → Numerical 表征转换查询。"""
    results = []
    for n, d in G.nodes(data=True):
        if emoji in n:
            # 找 emoji_to_num 边
            for u, v, ed in G.edges(n, data=True):
                desc = ed.get("description", "")
                if "Emoji" in desc and "对应" in desc:
                    tgt_data = G.nodes[v]
                    results.append({
                        "emoji": emoji,
                        "numerical": v,
                        "description": desc,
                        "node_type": tgt_data.get("type", ""),
                    })
    return results


def find_problems_by_source(G, source_label: str) -> list[str]:
    """按数学原型检索所有题目。"""
    results = []
    for n, d in G.nodes(data=True):
        if d.get("type") == "Source" and source_label in n:
            # 找 generates 边
            for u, v, ed in G.edges(n, data=True):
                desc = ed.get("description", "")
                if "生成" in desc:
                    results.append(v)
    return sorted(results)


def graph_stats(G):
    """图统计信息。"""
    print("=" * 60)
    print("知识图谱统计")
    print("=" * 60)

    # 节点
    node_types = Counter(d.get("type", "?") for _, d in G.nodes(data=True))
    print(f"\n节点总数: {G.number_of_nodes()}")
    for t, c in sorted(node_types.items(), key=lambda x: -x[1]):
        print(f"  {t:15s}: {c}")

    # 边
    edge_types = Counter()
    for _, _, d in G.edges(data=True):
        desc = d.get("description", "")
        if "同源" in desc:
            edge_types["same_source"] += 1
        elif "包含数值" in desc:
            edge_types["has_param"] += 1
        elif "生成" in desc:
            edge_types["generates"] += 1
        elif "Emoji" in desc and "对应" in desc:
            edge_types["emoji_to_num"] += 1
        elif "Emoji 表征" in desc:
            edge_types["represents"] += 1
        elif "解答步骤" in desc:
            edge_types["visualizes"] += 1
        elif "计算得到" in desc:
            edge_types["computes_to"] += 1
        else:
            edge_types["other"] += 1
    print(f"\n边总数: {G.number_of_edges()}")
    for t, c in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {t:15s}: {c}")

    # 连通分量
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    print(f"\n连通分量数: {len(components)}")
    # 最大的5个
    components.sort(key=len, reverse=True)
    for i, comp in enumerate(components[:5]):
        # 统计分量中各类型
        comp_types = Counter(G.nodes[n].get("type", "?") for n in comp)
        print(f"  分量{i+1}: {len(comp)}节点 {dict(comp_types)}")

    # 同源聚类
    print("\n--- 同源题目聚类 ---")
    source_clusters = defaultdict(list)
    for n, d in G.nodes(data=True):
        if d.get("type") == "ProblemText":
            # 找到其 Source
            for u, v, ed in G.in_edges(n, data=True):
                desc = ed.get("description", "")
                if "生成" in desc and G.nodes[u].get("type") == "Source":
                    source_clusters[u].append(n)
    for src, probs in sorted(source_clusters.items(), key=lambda x: -len(x[1])):
        print(f"  {src}: {len(probs)}题")
        for p in probs[:3]:
            print(f"    - {p}")
        if len(probs) > 3:
            print(f"    ... 还有{len(probs)-3}题")


def demo_queries(G):
    """演示三类查询。"""
    print("\n" + "=" * 60)
    print("查询演示")
    print("=" * 60)

    # 1. 同源检索
    print("\n--- 1. 同源题目检索 ---")
    test_problem = "afcp-ex3.1"
    print(f"输入: {test_problem}")
    same = find_same_source(G, test_problem)
    print(f"同源题目: {len(same)} 条")
    for s in same[:5]:
        print(f"  - {s}")
    if len(same) > 5:
        print(f"  ... 还有{len(same)-5}条")

    # 2. Emoji → Numerical
    print("\n--- 2. 表征转换查询 (Emoji → Numerical) ---")
    test_emoji = "⚪"
    print(f"输入: {test_emoji}")
    results = find_emoji_to_numerical(G, test_emoji)
    print(f"找到 {len(results)} 个映射:")
    for r in results[:5]:
        print(f"  {r['emoji']} → {r['numerical']} ({r['description']})")

    # 3. 按原型检索
    print("\n--- 3. 按数学原型检索 ---")
    test_source = "骰子"
    print(f"输入: {test_source}")
    probs = find_problems_by_source(G, test_source)
    print(f"找到 {len(probs)} 道骰子相关题目:")
    for p in probs[:5]:
        print(f"  - {p}")
    if len(probs) > 5:
        print(f"  ... 还有{len(probs)-5}道")


def main():
    G = load_graph()
    print(f"已加载图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边\n")

    # 统计
    graph_stats(G)

    # 演示查询
    demo_queries(G)

    print("\n" + "=" * 60)
    print("✅ 图查询验证完成")


if __name__ == "__main__":
    main()
