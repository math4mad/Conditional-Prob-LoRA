#!/usr/bin/env python3
"""
build_kg.py
为 SFT 语料构建四层异构知识图谱并导入 LightRAG。

图 Schema（基于 dialogue1.md 设计）：
  四层节点：
    1. Source      — 数学原型（骰子/坛子/硬币/双向表/路径图）
    2. Numerical   — 关键数值参数（球数/骰子面数/概率值）
    3. ProblemText — 题目文本
    4. SolutionStep — 解答步骤（四步法）

  边类型：
    - generates:    Source → ProblemText（原型生成具体题目）
    - has_param:    ProblemText → Numerical（题目含某数值参数）
    - computes_to:  SolutionStep → Numerical（步骤计算某数值）
    - visualizes:   ProblemText → SolutionStep（题目对应解答）
    - same_source:  Source ↔ Source（同源原型，用于跨题检索）
"""

import json
import os
import re
import sys
import asyncio
import numpy as np
from pathlib import Path

# LightRAG imports
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

BASE_DIR = Path(__file__).resolve().parent.parent
SFT_FILE = BASE_DIR / "data" / "sft_train_130.jsonl"
WORKING_DIR = BASE_DIR / "lightrag_data"


# ============================================================
# 1. Embedding 函数（确定性 hash embedding，无需外部模型）
# ============================================================

def _hash_embedding(texts, dim=384):
    """确定性 embedding：基于字符 n-gram hash，无需外部模型。"""
    results = []
    for t in texts:
        vec = np.zeros(dim, dtype=np.float32)
        # unigram + bigram
        for i in range(len(t)):
            vec[ord(t[i]) % dim] += 1.0
            if i + 1 < len(t):
                bigram = (ord(t[i]) * 31 + ord(t[i + 1])) % dim
                vec[bigram] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        results.append(vec)
    return results


def make_embedding_func(dim=384):
    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=8192,
        func=lambda texts: _hash_embedding(texts, dim),
    )


# ============================================================
# 2. Dummy LLM 函数（满足 LightRAG 初始化要求，实际查询不走LLM）
# ============================================================

async def dummy_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """占位 LLM 函数。手动构建的 KG 不需要 LLM 做实体抽取。"""
    return "ok"


# ============================================================
# 3. 从 SFT 语料提取节点和边
# ============================================================

# 数学原型分类规则
SOURCE_TYPES = {
    "dice": {
        "label": "骰子条件概率",
        "keywords": ["骰子", "骰", "点数", "sum", "掷"],
        "variants": ["4面骰子", "6面骰子", "8面骰子", "12面骰子", "两枚", "三枚"],
    },
    "urn": {
        "label": "坛子抽样",
        "keywords": ["坛子", "罐子", "球", "白球", "黑球", "红球", "取出"],
        "variants": ["放回", "不放回", "Polya"],
    },
    "coin": {
        "label": "硬币抛掷",
        "keywords": ["硬币", "正面", "反面", "抛", "掷"],
        "variants": ["均匀硬币", "偏置硬币", "两面正面"],
    },
    "table": {
        "label": "双向表贝叶斯",
        "keywords": ["调查", "百分比", "%", "双向表", "列联表", "贝叶斯"],
        "variants": ["2x2表", "条件概率表"],
    },
    "path": {
        "label": "多阶段路径",
        "keywords": ["阶段", "考试", "连续", "路径", "过程", "步骤"],
        "variants": ["2阶段", "3阶段", "考试路径"],
    },
}


def classify_source(input_text: str, output_text: str) -> str:
    """根据题目文本判断数学原型类型。"""
    combined = input_text + " " + output_text
    best_type = "other"
    best_score = 0
    for stype, info in SOURCE_TYPES.items():
        score = sum(1 for kw in info["keywords"] if kw in combined)
        if score > best_score:
            best_score = score
            best_type = stype
    return best_type


def extract_key_numbers(input_text: str, output_text: str) -> list[str]:
    """提取关键数值参数。"""
    numbers = []
    # 从 input 提取明确的数值描述
    # 球数：N个白球/N个黑球
    for m in re.finditer(r"(\d+)个(白|黑|红)球", input_text):
        numbers.append(f"{m.group(2)}球×{m.group(1)}")
    # 骰子面数
    for m in re.finditer(r"(\d+)面骰子", input_text):
        numbers.append(f"{m.group(1)}面骰")
    if "两枚" in input_text and "骰" in input_text:
        numbers.append("两枚骰子")
    if "三枚" in input_text and "骰" in input_text:
        numbers.append("三枚骰子")
    # 硬币数
    for m in re.finditer(r"(\d+)枚硬币", input_text):
        numbers.append(f"{m.group(1)}枚硬币")
    # 概率值
    for m in re.finditer(r"概率为(\d+/\d+)", input_text):
        numbers.append(f"P={m.group(1)}")
    # 分数结果
    for m in re.finditer(r"=\s*(\d+)/(\d+)", output_text):
        numbers.append(f"结果={m.group(1)}/{m.group(2)}")
        break  # 只取最终结果
    if not numbers:
        # 提取通用数字
        raw = re.findall(r"\b(\d+)\b", input_text)
        for n in raw[:3]:
            numbers.append(f"数值{n}")
    return numbers[:5]  # 最多5个


def extract_solution_steps(output_text: str) -> list[dict]:
    """提取四步法的步骤。"""
    steps = []
    step_patterns = [
        (1, r"## 1\.\s*(.*?)(?=## 2\.|$)", re.DOTALL),
        (2, r"## 2\.\s*(.*?)(?=## 3\.|$)", re.DOTALL),
        (3, r"## 3\.\s*(.*?)(?=## 4\.|$)", re.DOTALL),
        (4, r"## 4\.\s*(.*?)$", re.DOTALL),
    ]
    for step_num, pattern, flags in step_patterns:
        m = re.search(pattern, output_text, flags)
        if m:
            content = m.group(1).strip()[:200]  # 截断
            steps.append({"step": step_num, "content": content})
    return steps


# Emoji 映射（来自 dialogue1.md 的设计）
EMOJI_MAP = {
    "白球": "⚪", "黑球": "⚫", "红球": "🔴", "蓝球": "🔵", "绿球": "🟢", "黄球": "🟡",
    "正面": "🟣", "反面": "🟠",
}


def generate_emoji_representation(input_text: str, source_type: str) -> str:
    """为题目生成 Emoji 表征（如果适用）。"""
    if source_type == "urn":
        emojis = []
        for m in re.finditer(r"(\d+)个(白|黑|红)球", input_text):
            count = int(m.group(1))
            color = m.group(2)
            emoji = EMOJI_MAP.get(color + "球", "❓")
            emojis.append(emoji * min(count, 10))
        if emojis:
            return "".join(emojis)
    elif source_type == "coin":
        if "均匀" in input_text:
            return "🟣🟠"  # 正反各半
        for m in re.finditer(r"(\d+)枚硬币", input_text):
            return "🟣" * min(int(m.group(1)), 5)
    elif source_type == "dice":
        if "两枚" in input_text:
            return "⚀⚁"  # 骰子emoji
        elif "三枚" in input_text:
            return "⚀⚁⚂"
    return ""


# ============================================================
# 4. 构建知识图谱
# ============================================================

def build_custom_kg(records: list[dict]) -> dict:
    """
    从 SFT 语料构建自定义知识图谱。
    返回 LightRAG insert_custom_kg 所需格式。
    """
    entities = []
    relationships = []
    seen_entities = set()

    def add_entity(name, etype, description):
        key = f"{etype}:{name}"
        if key not in seen_entities:
            seen_entities.add(key)
            entities.append({
                "entity_name": name,
                "type": etype,
                "description": description,
            })
        return name

    # 用于同源连接
    source_problems = {}  # source_type -> [problem_ids]

    for r in records:
        src = r["source"]
        inp = r["input"]
        out = r["output"]
        stype = classify_source(inp, out)
        slabel = SOURCE_TYPES.get(stype, {}).get("label", "其他")

        # --- 节点 ---
        # 1. Source 节点
        source_node = add_entity(
            f"原型:{slabel}", "Source",
            f"数学原型: {slabel}。包含{len(records)}条相关题目。"
        )
        source_problems.setdefault(stype, []).append(src)

        # 2. ProblemText 节点
        problem_desc = inp[:150].replace("\n", " ")
        problem_node = add_entity(
            f"题目:{src}", "ProblemText",
            problem_desc
        )

        # 3. Numerical 节点
        key_nums = extract_key_numbers(inp, out)
        num_nodes = []
        for num in key_nums:
            num_node = add_entity(
                f"数值:{num}", "Numerical",
                f"数值参数: {num}，出现于题目 {src}"
            )
            num_nodes.append(num_node)

        # 4. Emoji 节点（如果适用）
        emoji_repr = generate_emoji_representation(inp, stype)
        if emoji_repr:
            emoji_node = add_entity(
                f"Emoji:{emoji_repr}", "Emoji",
                f"视觉表征: {emoji_repr}，对应题目 {src}，类型 {slabel}"
            )
        else:
            emoji_node = None

        # 5. SolutionStep 节点
        steps = extract_solution_steps(out)
        step_nodes = []
        for s in steps:
            step_node = add_entity(
                f"步骤:{src}-S{s['step']}", "SolutionStep",
                f"第{s['step']}步: {s['content'][:100]}"
            )
            step_nodes.append(step_node)

        # --- 边 ---
        # generates: Source → ProblemText
        relationships.append({
            "src_id": source_node,
            "tgt_id": problem_node,
            "description": f"原型「{slabel}」生成了题目 {src}",
            "weight": 1.0,
        })

        # has_param: ProblemText → Numerical
        for num_node in num_nodes:
            relationships.append({
                "src_id": problem_node,
                "tgt_id": num_node,
                "description": f"题目 {src} 包含数值参数",
                "weight": 1.0,
            })

        # emoji_to_num: Emoji → Numerical（如果两者都存在）
        if emoji_node and num_nodes:
            # 连接 emoji 到第一个数值节点
            relationships.append({
                "src_id": emoji_node,
                "tgt_id": num_nodes[0],
                "description": f"Emoji {emoji_repr} 对应数值 {key_nums[0]}",
                "weight": 1.0,
            })
            # represents: Source → Emoji
            relationships.append({
                "src_id": source_node,
                "tgt_id": emoji_node,
                "description": f"原型 {slabel} 的 Emoji 表征",
                "weight": 1.0,
            })

        # visualizes: ProblemText → SolutionStep
        for step_node in step_nodes:
            relationships.append({
                "src_id": problem_node,
                "tgt_id": step_node,
                "description": f"题目 {src} 的解答步骤",
                "weight": 1.0,
            })

        # computes_to: SolutionStep → Numerical（步骤4含最终结果）
        if len(step_nodes) >= 4 and num_nodes:
            result_num = num_nodes[-1]  # 最后一个通常是结果
            relationships.append({
                "src_id": step_nodes[-1],
                "tgt_id": result_num,
                "description": f"步骤4计算得到 {key_nums[-1]}",
                "weight": 1.5,
            })

    # same_source: 同源原型之间的连接
    for stype, probs in source_problems.items():
        if len(probs) > 1:
            slabel = SOURCE_TYPES.get(stype, {}).get("label", "其他")
            source_node_name = f"原型:{slabel}"
            # 为同类型的每对题目添加 "same_source" 边（通过它们各自的 ProblemText）
            for i in range(len(probs)):
                for j in range(i + 1, min(i + 5, len(probs))):  # 每个题连5个同源
                    relationships.append({
                        "src_id": f"题目:{probs[i]}",
                        "tgt_id": f"题目:{probs[j]}",
                        "description": f"同源: 两题都属于「{slabel}」原型",
                        "weight": 2.0,
                    })

    return {"entities": entities, "relationships": relationships}


# ============================================================
# 5. 主函数：构建图并导入 LightRAG
# ============================================================

def main():
    # 读取语料
    with open(SFT_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"读取 {len(records)} 条 SFT 语料")

    # 构建自定义KG
    print("\n构建知识图谱...")
    custom_kg = build_custom_kg(records)
    n_entities = len(custom_kg["entities"])
    n_rels = len(custom_kg["relationships"])
    print(f"  节点: {n_entities}")
    print(f"  边: {n_rels}")

    # 打印节点类型分布
    from collections import Counter
    type_dist = Counter(e["type"] for e in custom_kg["entities"])
    print(f"  节点类型分布: {dict(type_dist)}")

    # 打印边类型分布
    edge_types = Counter()
    for r in custom_kg["relationships"]:
        desc = r["description"]
        if "同源" in desc:
            edge_types["same_source"] += 1
        elif "包含数值" in desc:
            edge_types["has_param"] += 1
        elif "生成" in desc:
            edge_types["generates"] += 1
        elif "Emoji" in desc:
            edge_types["emoji_to_num"] += 1
        elif "解答步骤" in desc:
            edge_types["visualizes"] += 1
        elif "计算得到" in desc:
            edge_types["computes_to"] += 1
        else:
            edge_types["other"] += 1
    print(f"  边类型分布: {dict(edge_types)}")

    # 先保存KG为JSON（确保数据不丢失）
    kg_json_path = BASE_DIR / "data" / "kg_structure.json"
    with open(kg_json_path, "w", encoding="utf-8") as f:
        json.dump(custom_kg, f, ensure_ascii=False, indent=2)
    print(f"\n📄 KG结构: {kg_json_path}")

    # 用 NetworkX 构建图并持久化（核心图存储，不依赖 LightRAG flush）
    import networkx as nx
    G = nx.DiGraph()
    for e in custom_kg["entities"]:
        G.add_node(e["entity_name"], type=e["type"], description=e["description"])
    for r in custom_kg["relationships"]:
        # 从 description 推断边类型
        desc = r["description"]
        if "同源" in desc:
            etype = "same_source"
        elif "包含数值" in desc:
            etype = "has_param"
        elif "生成" in desc:
            etype = "generates"
        elif "Emoji" in desc and "对应" in desc:
            etype = "emoji_to_num"
        elif "Emoji 表征" in desc:
            etype = "represents"
        elif "解答步骤" in desc:
            etype = "visualizes"
        elif "计算得到" in desc:
            etype = "computes_to"
        else:
            etype = "other"
        G.add_edge(r["src_id"], r["tgt_id"], type=etype, description=desc, weight=r["weight"])

    # 持久化
    graphml_path = WORKING_DIR / "kg_graph.graphml"
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, graphml_path)
    print(f"📄 GraphML: {graphml_path} ({G.number_of_nodes()}节点, {G.number_of_edges()}边)")

    # 同时保存 pickle 供快速加载
    gpickle_path = WORKING_DIR / "kg_graph.pkl"
    nx.write_gpickle = None  # 新版 networkx 无 write_gpickle
    import pickle
    with open(gpickle_path, "wb") as f:
        pickle.dump(G, f)
    print(f"📄 Pickle: {gpickle_path}")

    # 尝试导入 LightRAG（容错，flush bug 不阻塞）
    try:
        rag = LightRAG(
            working_dir=str(WORKING_DIR / "lightrag_store"),
            embedding_func=make_embedding_func(),
            llm_model_func=dummy_llm_func,
        )
        rag.insert_custom_kg(custom_kg)
        print("✅ LightRAG 导入完成")
    except Exception as e:
        print(f"⚠️ LightRAG 导入跳过（已知flush bug，图已用NetworkX持久化）: {type(e).__name__}")

    return custom_kg


if __name__ == "__main__":
    main()
