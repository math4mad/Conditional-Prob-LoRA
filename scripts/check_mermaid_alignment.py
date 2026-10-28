#!/usr/bin/env python3
"""
check_mermaid_alignment.py
Phase5 / P1: Mermaid 语义对齐检查器

背景 (ref: external/diage2.md 第三层 P1):
  "Mermaid 图中的节点数量是否与文本步骤中的集合大小一致？
   边上的概率标签是否等于对应步骤的计算结果？
   是否存在'图是对的但文不对题'或'文对图错'的情况？"

检查维度：
  C1  Mermaid 语法合法性      — flowchart 结构、节点定义、边定义、style 引用
  C2  节点-集合大小一致性     — 图中 |A∩B|/|B| 数值是否与文本步骤中的数值一致
  C3  概率标签一致性          — 边标签中的分数是否出现在正文的 LaTeX 公式中
  C4  条件/目标名称一致性      — 条件名与目标名是否在图节点标签中出现
  C5  条件概率比例校验        — |A∩B|/|B| 是否等于图中声称的最终概率
  C6  四步法与图的一致性      — 步骤编号 1-4 是否齐全

两种模式：
  --mode reference  对训练集参考输出跑检查（自验证：检查器正确时应 ~100% 通过）
  --mode report     对 eval_report_*.json 中模型生成结果跑检查
"""
import json
import re
import sys
import argparse
from fractions import Fraction
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Mermaid 解析
# ============================================================
MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
# 节点定义（可内联在边行中）：  ID["标签"]  /  ID(标签)  /  ID{标签}
NODE_BRACKET_RE = re.compile(r'([A-Za-z_][\w]*)\s*\[\s*"(.*?)"\s*\]')
NODE_PAREN_RE = re.compile(r'([A-Za-z_][\w]*)\s*\(\s*"(.*?)"\s*\)')
NODE_BRACE_RE = re.compile(r'([A-Za-z_][\w]*)\s*\{\s*"(.*?)"\s*\}')
NODE_PLAIN_RE = re.compile(r'^\s*([A-Za-z_][\w]*)\s*$\s*$', re.MULTILINE)
# 边: A -->|label| B  /  A --> B  /  A -->|label| B -->|label| C
EDGE_RE = re.compile(
    r'([A-Za-z_][\w]*)\s*-->\s*(?:\|"?(.*?)"?\|)?\s*([A-Za-z_][\w]*)'
)
# style: style ID fill:...
STYLE_RE = re.compile(r'^\s*style\s+([A-Za-z_][\w]*)', re.MULTILINE)
# flow 声明
FLOW_RE = re.compile(r'^\s*flowchart\s+(LR|RL|TD|TB|BT)', re.MULTILINE)


def extract_mermaid_blocks(text):
    return MERMAID_RE.findall(text)


def parse_node_labels(block):
    """扫描 mermaid 源码，提取所有节点 id → label。

    支持全部 mermaid 形状语法：
      ID[label]  ID(label)  ID([label])  ID[[label]]  ID[(label)]
      ID((label))  ID>label]  ID{label}  ID{{label}}
    用括号配平扫描，避免正则误切。
    """
    nodes = {}
    spans = []            # (id, start, end) 用于定位边的源/目标
    i = 0
    n = len(block)
    ident = re.compile(r'[A-Za-z_][\w]*')
    while i < n:
        m = ident.match(block, i)
        if not m:
            i += 1
            continue
        node_id = m.group(0)
        j = m.end()
        while j < n and block[j] == ' ':
            j += 1
        if j >= n or block[j] not in '[({':
            i = m.end()
            continue
        # 括号配平扫描
        depth = 0
        k = j
        start = j
        while k < n:
            ch = block[k]
            if ch in '[({':
                depth += 1
            elif ch in '])}':
                depth -= 1
                if depth == 0:
                    break
            elif ch == '\n' and depth == 0:
                break
            k += 1
        if k < n and depth == 0:
            inner = block[start + 1:k]
            # 去掉外层多余括号，如 ([x]) -> [x] -> x
            inner = inner.strip()
            while len(inner) >= 2 and inner[0] in '[({' and inner[-1] in '])}':
                inner = inner[1:-1].strip()
            inner = inner.strip('"').strip()
            if node_id not in nodes:
                nodes[node_id] = inner
                spans.append((node_id, m.start(), k + 1))
            i = k + 1
        else:
            i = m.end()
    return nodes, spans


def parse_edges(block, spans):
    """提取边 (src, label, dst)。

    源节点 = 在 `-->` 之前结束的最后一个节点定义（支持内联定义）；
    目标节点 = 在 `-->` 之后最近的标识符。
    """
    edges = []
    for m in re.finditer(r'-->\s*(?:\|"?(.*?)"?\|)?', block):
        lbl = (m.group(1) or "").strip()
        after = block[m.end():]
        tm = re.match(r'\s*([A-Za-z_][\w]*)', after)
        dst = tm.group(1) if tm else ""
        src = ""
        best_end = -1
        for nid, s, e in spans:
            if e <= m.start() and e > best_end:
                src, best_end = nid, e
        edges.append((src, lbl, dst))
    return edges


def parse_mermaid(block):
    """解析 mermaid flowchart，返回节点/边/样式信息。"""
    info = {
        "valid_syntax": False,
        "has_flowchart": False,
        "nodes": {},        # id -> label
        "edges": [],        # (src, label, dst)
        "styled": set(),
        "errors": [],
    }
    if not FLOW_RE.search(block):
        info["errors"].append("缺少 flowchart 声明")
    else:
        info["has_flowchart"] = True

    # 节点：括号配平扫描，支持内联定义与全部形状语法
    nodes, spans = parse_node_labels(block)
    info["nodes"].update(nodes)
    # 裸节点行:  ID  单独成行
    for line in block.split("\n"):
        s = line.strip()
        if not s or "-->" in s or s.startswith(("flowchart", "style")):
            continue
        m = NODE_PLAIN_RE.match(s)
        if m:
            info["nodes"].setdefault(m.group(1), m.group(1))

    # 边
    info["edges"] = parse_edges(block, spans)

    # style
    for m in STYLE_RE.finditer(block):
        info["styled"].add(m.group(1))

    # 语法校验
    errs = []
    if not info["has_flowchart"]:
        errs.append("no flowchart header")
    if not info["nodes"] and not info["edges"]:
        errs.append("no nodes or edges")
    # style 引用的节点必须存在
    for sid in info["styled"]:
        if sid not in info["nodes"]:
            errs.append(f"style references undefined node: {sid}")
    # 边的端点：可能是节点定义行内联的，也可能是独立定义行
    for src, lbl, dst in info["edges"]:
        if not dst:
            errs.append(f"edge missing target node: {src} --> {lbl or '(no label)'}")
            continue
        for nid in (src, dst):
            if nid not in info["nodes"]:
                # 可能是 `A["x"] --> B["y"]` 这种内联
                if not re.search(rf'\b{re.escape(nid)}\s*(?:\[|\(|\{{)', block):
                    errs.append(f"edge references undefined node: {nid}")
    info["errors"] = errs
    info["valid_syntax"] = len(errs) == 0
    return info


def label_numbers(label):
    """从节点标签中提取所有数字（含分数 a/b）。"""
    nums = set()
    for m in re.finditer(r'(\d+)\s*/\s*(\d+)', label):
        fr = _safe_frac(m.group(1), m.group(2))
        if fr is not None:
            nums.add(fr)
    for m in re.finditer(r'(?<![\d/])(\d+)(?![\d/])', label):
        nums.add(Fraction(int(m.group(1))))
    return nums


# ============================================================
# 正文解析
# ============================================================
def _safe_frac(a, b):
    """安全构造 Fraction，分母为 0 时返回 None。"""
    try:
        b = int(b)
        if b == 0:
            return None
        return Fraction(int(a), b)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def extract_fracs(text):
    """提取正文所有 a/b 形式的分数。"""
    out = set()
    for m in re.finditer(r'(\d+)\s*/\s*(\d+)', text):
        fr = _safe_frac(m.group(1), m.group(2))
        if fr is not None:
            out.add(fr)
    return out


def extract_ints(text):
    """提取正文所有独立整数（排除分数的分母/分子，避免重复计数）。"""
    # 先移除 a/b 形式，再取整数
    stripped = re.sub(r'(\d+)\s*/\s*\d+', ' ', text)
    return {Fraction(int(m.group(0)))
            for m in re.finditer(r'(?<![\d.])\d+(?![\d.])', stripped)}


def extract_set_sizes(text):
    """提取 |A|, |B|, |A∩B| 等集合大小（支持 LaTeX \\cap 与 Unicode ∩）。"""
    sizes = {}
    # |A∩B| = 30   |A \cap B| = 30   |B| = 30
    pattern = r'\|\s*(A\s*(?:\\cap|∩)\s*B|B|A)\s*\|\s*=\s*(\d+)'
    for m in re.finditer(pattern, text):
        key = re.sub(r'\s+', '', m.group(1)).replace("\\cap", "∩")
        sizes[key] = int(m.group(2))
    return sizes


def extract_final_prob(text):
    """提取正文最后的条件概率结论（与 evaluate_lora.py 一致）。"""
    matches = re.findall(r"=\s*(\d+)/(\d+)", text)
    if matches:
        n, d = matches[-1]
        return _safe_frac(n, d)
    matches_int = re.findall(r"=\s*(\d+)\s*\$\$", text)
    if matches_int:
        return Fraction(int(matches_int[-1]), 1)
    return None


# ============================================================
# 核心检查
# ============================================================
def check_alignment(text, strict=True):
    """对一段解答做 Mermaid 语义对齐检查。返回逐项结果。"""
    result = {
        "checks": {},
        "details": {},
        "score": 0,
        "total": 6,
    }
    blocks = extract_mermaid_blocks(text)

    # C1 Mermaid 语法
    if not blocks:
        result["checks"]["c1_syntax"] = False
        result["details"]["c1_syntax"] = "无 mermaid 代码块"
    else:
        infos = [parse_mermaid(b) for b in blocks]
        ok = all(i["valid_syntax"] for i in infos)
        result["checks"]["c1_syntax"] = ok
        result["details"]["c1_syntax"] = (
            "全部合法" if ok else
            "; ".join(f"block{i}: {e}" for i, inf in enumerate(infos) for e in inf["errors"])
        )

    # C6 四步法齐全（兼容 ## 与 ### 标题层级）
    steps_present = all(re.search(rf"#{{2,3}}\s*{i}\s*[\.、]", text) for i in range(1, 5))
    result["checks"]["c6_steps"] = steps_present
    result["details"]["c6_steps"] = "步骤 1-4 齐全" if steps_present else "缺少步骤标题"

    if not blocks:
        for k in ("c2_node_consistency", "c3_prob_labels", "c4_names",
                  "c5_ratio"):
            result["checks"][k] = False
            result["details"][k] = "无 mermaid，跳过"
        result["score"] = sum(1 for v in result["checks"].values() if v)
        return result

    info = parse_mermaid(blocks[0])
    text_fracs = extract_fracs(text)
    sizes = extract_set_sizes(text)

    # C2 节点标签中的数值是否与正文一致
    node_nums = set()
    for lbl in info["nodes"].values():
        node_nums |= label_numbers(lbl)
    # 只关心"有意义"的数值：真分数（概率）+ 大规模计数(>10)
    # 排除平凡小整数（1-10），它们在正文随处可见会造成误判
    significant = {n for n in node_nums
                   if n.denominator > 1 or n > 10}
    if not significant:
        result["checks"]["c2_node_consistency"] = True
        result["details"]["c2_node_consistency"] = "节点标签无可比对数值"
    else:
        body_nums = set(text_fracs) | extract_ints(text)
        matched = sum(1 for n in significant if n in body_nums)
        ratio = matched / len(significant)
        result["checks"]["c2_node_consistency"] = ratio >= 0.8
        result["details"]["c2_node_consistency"] = (
            f"{matched}/{len(significant)} 节点数值可在正文找到")

    # C3 边标签概率是否出现在正文公式中
    edge_labels = [lbl for _, lbl, _ in info["edges"] if lbl and re.search(r'\d+\s*/\s*\d+', lbl)]
    if not edge_labels:
        result["checks"]["c3_prob_labels"] = True
        result["details"]["c3_prob_labels"] = "边标签无分数（不适用）"
    else:
        ok = 0
        for lbl in edge_labels:
            fr = set()
            for m in re.finditer(r'(\d+)\s*/\s*(\d+)', lbl):
                fr.add(Fraction(int(m.group(1)), int(m.group(2))))
            if fr & text_fracs or not fr:
                ok += 1
        result["checks"]["c3_prob_labels"] = ok == len(edge_labels)
        result["details"]["c3_prob_labels"] = f"{ok}/{len(edge_labels)} 边标签概率与正文一致"

    # C4 条件/目标名称在图中出现
    cond_names = set(re.findall(r'「(.+?)」', text))
    labels_all = " ".join(info["nodes"].values()) + " " + " ".join(
        l for _, l, _ in info["edges"])
    if not cond_names:
        result["checks"]["c4_names"] = True
        result["details"]["c4_names"] = "正文无「」标记（不适用）"
    else:
        def name_in_graph(name):
            """名称在图中出现：直接包含 / 分支词全含 / 实质性后缀命中。

            路径型题目的条件形如 "第一阶段:成功 → 第二阶段:失败"，
            图中用节点 "成功"/"失败" 表示；又如 "最终成功" 对应节点 "成功"。
            """
            if name in labels_all:
                return True
            # 路径型：按分支词判定
            if "→" in name:
                parts = [p.split(":")[-1].strip() for p in name.split("→")]
                parts = [p for p in parts if p]
                if parts and all(p in labels_all for p in parts):
                    return True
            # 实质性后缀命中（长度≥2 的中文/字母后缀）
            for L in range(min(len(name), 8), 1, -1):
                suffix = name[-L:]
                if suffix in labels_all:
                    return True
            return False

        hit = sum(1 for c in cond_names if name_in_graph(c))
        result["checks"]["c4_names"] = hit >= max(1, len(cond_names) - 1)
        result["details"]["c4_names"] = f"{hit}/{len(cond_names)} 条件/目标名出现在图中"

    # C5 比例校验 |A∩B|/|B| == 最终概率
    final = extract_final_prob(text)
    ab = sizes.get("A∩B")
    b = sizes.get("B")
    if ab is None or b is None or b == 0 or final is None:
        result["checks"]["c5_ratio"] = None
        result["details"]["c5_ratio"] = "集合大小或最终概率缺失（不适用）"
    else:
        calc = Fraction(ab, b)
        result["checks"]["c5_ratio"] = (calc == final)
        result["details"]["c5_ratio"] = f"|A∩B|/|B| = {calc} vs 正文 {final}"

    result["score"] = sum(1 for v in result["checks"].values() if v is True)
    applicable = sum(1 for v in result["checks"].values() if v is not None)
    result["total"] = applicable if applicable else 6
    return result


# ============================================================
# 批量运行
# ============================================================
def run_reference(data_file, limit=None):
    """自验证：对参考输出跑检查（检查器正确时应接近 100% 通过）。"""
    records = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break

    agg = {}
    n_pass_full = 0
    rows = []
    for r in records:
        res = check_alignment(r["output"])
        for k, v in res["checks"].items():
            if v is None:
                continue
            agg.setdefault(k, {"pass": 0, "n": 0})
            agg[k]["n"] += 1
            if v:
                agg[k]["pass"] += 1
        if all(v is not False for v in res["checks"].values()):
            n_pass_full += 1
        rows.append({"source": r["source"], "score": res["score"],
                     "total": res["total"], "checks": res["checks"]})

    print("=" * 64)
    print(f"参考输出自验证: {data_file}  (n={len(records)})")
    print("=" * 64)
    for k in sorted(agg):
        p, n = agg[k]["pass"], agg[k]["n"]
        print(f"  {k:24s} {p:4d}/{n:4d}  ({p/n*100:5.1f}%)")
    print(f"\n  全部检查项通过: {n_pass_full}/{len(records)} ({n_pass_full/len(records)*100:.1f}%)")

    # 列出失败样本
    bad = [r for r in rows if any(v is False for v in r["checks"].values())]
    if bad:
        print(f"\n  未完全通过的样本 ({len(bad)} 条), 前 10:")
        for r in bad[:10]:
            fails = [k for k, v in r["checks"].items() if v is False]
            print(f"    {r['source']:32s} score={r['score']}/{r['total']} 失败项: {fails}")

    return {
        "mode": "reference", "data_file": str(data_file), "n": len(records),
        "per_check": {k: {"pass": v["pass"], "n": v["n"]} for k, v in agg.items()},
        "all_pass": n_pass_full,
        "all_pass_rate": n_pass_full / len(records),
    }


def run_report(report_file):
    """对模型生成结果跑检查。

    兼容三种报告格式：
      1. eval_ood.json 格式          —— 顶层 details[]，含 generated
      2. eval_graphrag_ood.json 格式 —— results.{baseline,graphrag}.details[]
      3. 旧 eval_report_*.json 格式  —— 只有结构化标记，无原文
    """
    report = json.loads(Path(report_file).read_text(encoding="utf-8"))

    def collect(details):
        out = []
        for d in details:
            if "generated" in d and d["generated"]:
                out.append((d.get("source", d.get("idx", "?")), d["generated"]))
            elif "struct" in d:
                out.append((d.get("source", "?"), None))
        return out

    texts = []
    if "results" in report:                      # GraphRAG 对比报告
        for variant, blk in report["results"].items():
            for src, txt in collect(blk.get("details", [])):
                texts.append((f"{variant}/{src}", txt))
    else:
        texts = collect(report.get("details", []))

    agg = {}
    n_full = 0
    n_eval = 0
    rows = []
    for src, txt in texts:
        if not txt:
            continue
        n_eval += 1
        res = check_alignment(txt)
        for k, v in res["checks"].items():
            if v is None:
                continue
            agg.setdefault(k, {"pass": 0, "n": 0})
            agg[k]["n"] += 1
            if v:
                agg[k]["pass"] += 1
        if all(v is not False for v in res["checks"].values()):
            n_full += 1
        rows.append({"source": src, "score": res["score"], "total": res["total"],
                     "checks": res["checks"], "details": res["details"]})

    print("=" * 64)
    print(f"模型输出 Mermaid 对齐检查: {report_file}")
    print("=" * 64)
    if n_eval == 0:
        print("  该报告无生成原文，无法做语义检查")
        return {"mode": "report", "report_file": str(report_file), "note": "no generated text"}
    for k in sorted(agg):
        p, n = agg[k]["pass"], agg[k]["n"]
        print(f"  {k:24s} {p:4d}/{n:4d}  ({p/n*100:5.1f}%)")
    print(f"\n  全部检查项通过: {n_full}/{n_eval} ({n_full/n_eval*100:.1f}%)")

    bad = [r for r in rows if any(v is False for v in r["checks"].values())]
    if bad:
        print(f"\n  未完全通过 ({len(bad)} 条), 前 10:")
        for r in bad[:10]:
            fails = [k for k, v in r["checks"].items() if v is False]
            print(f"    {str(r['source']):30s} score={r['score']}/{r['total']} 失败: {fails}")
            for k in fails:
                print(f"        {k}: {r['details'].get(k)}")

    return {
        "mode": "report", "report_file": str(report_file), "n": n_eval,
        "per_check": {k: {"pass": v["pass"], "n": v["n"]} for k, v in agg.items()},
        "all_pass": n_full, "all_pass_rate": n_full / n_eval,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Mermaid 语义对齐检查器")
    parser.add_argument("--mode", choices=["reference", "report"], required=True)
    parser.add_argument("--file", type=str, default=None, help="数据文件或报告文件")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    if args.mode == "reference":
        f = args.file or str(BASE_DIR / "data" / "sft_train_580.jsonl")
        res = run_reference(f, args.limit)
    else:
        f = args.file or str(BASE_DIR / "output" / "eval_report_516.json")
        res = run_report(f)

    if args.output:
        Path(args.output).write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📄 结果: {args.output}")


if __name__ == "__main__":
    main()
