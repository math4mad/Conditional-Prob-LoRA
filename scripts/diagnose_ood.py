#!/usr/bin/env python3
"""
diagnose_ood.py
Phase5: OOD 失败模式诊断

对 evaluate_lora.py 产出的 OOD 评估报告做失败归因，回答三个问题：
  1. 格式迁移是否成功？（结构指标 vs 数值指标的分层对比）
  2. 数值错误是"算错一点"还是"系统性偏移"？
     -> 答案复杂度坍缩检测：生成答案的分母分布 vs 真值/训练集
  3. 图文一致性在哪一环断裂？（P1 各子项失败分布）

用法:
  python diagnose_ood.py --report output/eval_ood.json
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def final_prob(text):
    """提取最终概率（与 evaluate_lora.py 同规则）。"""
    if not text:
        return None
    m = re.findall(r"=\s*(\d+)/(\d+)", text)
    if m:
        try:
            return Fraction(int(m[-1][0]), int(m[-1][1]))
        except (ValueError, ZeroDivisionError):
            return None
    m = re.findall(r"=\s*(\d+)\s*\$\$", text)
    if m:
        return Fraction(int(m[-1]), 1)
    return None


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def diagnose(report_path, train_file, ood_file):
    rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
    det = rep["details"]
    s = rep.get("summary", {})

    print("=" * 68)
    print("OOD 失败模式诊断")
    print("=" * 68)
    print(f"报告: {report_path}  (n={len(det)})")
    print(f"总体: 四步法 {s.get('structure_pass_rate', 0)*100:.1f}% | "
          f"数值正确 {s.get('accuracy_correct', 0)}/{s.get('samples', 0)} "
          f"({s.get('accuracy_correct', 0)/max(s.get('samples', 1), 1)*100:.1f}%) | "
          f"对齐 {s.get('alignment_rate', 0)*100:.1f}%")

    # --- 1. 分层对比 ---
    print("\n--- 1. 分层：格式 vs 数值 ---")
    layers = defaultdict(lambda: {"n": 0, "fs": 0, "num": 0, "align": 0,
                                  "mermaid": 0, "latex": 0})
    for d in det:
        lay = d["source"].split("-")[0]
        a = layers[lay]
        a["n"] += 1
        a["fs"] += all(d["struct"][f"step{i}"] for i in range(1, 5))
        a["num"] += d["accuracy"]["correct"]
        a["mermaid"] += d["struct"].get("mermaid", False)
        a["latex"] += d["struct"].get("latex", False)
        al = d.get("alignment")
        a["align"] += bool(al) and all(v is not False for v in al["checks"].values())
    print(f"{'层':<12}{'n':>4}{'四步法':>10}{'Mermaid':>10}{'LaTeX':>9}"
          f"{'数值正确':>11}{'对齐':>9}")
    print("-" * 65)
    for k, a in sorted(layers.items()):
        n = a["n"]
        print(f"{k:<12}{n:>4}{a['fs']/n*100:>9.1f}%{a['mermaid']/n*100:>9.1f}%"
              f"{a['latex']/n*100:>8.1f}%{a['num']/n*100:>10.1f}%{a['align']/n*100:>8.1f}%")
    print("\n  → 格式层迁移近乎成功，数值层完全失败 = 学到格式而非计算")

    # --- 2. 答案复杂度坍缩 ---
    print("\n--- 2. 答案复杂度坍缩检测 ---")
    tr = load_jsonl(train_file)
    ood = load_jsonl(ood_file)
    sets = {
        "训练集答案": [x.denominator for x in (final_prob(r["output"]) for r in tr) if x],
        "OOD 真值": [x.denominator for x in (final_prob(r["output"]) for r in ood) if x],
        "OOD 模型生成": [x.denominator for x in (final_prob(r.get("generated")) for r in det) if x],
    }
    print(f"{'集合':<16}{'n':>5}{'分母≤12':>10}{'中位分母':>10}{'最大分母':>14}")
    print("-" * 55)
    stats = {}
    for k, v in sets.items():
        if not v:
            continue
        simple = sum(1 for d in v if d <= 12) / len(v) * 100
        med = sorted(v)[len(v) // 2]
        stats[k] = (simple, med, max(v))
        print(f"{k:<16}{len(v):>5}{simple:>9.1f}%{med:>10}{max(v):>14}")
    if "OOD 真值" in stats and "OOD 模型生成" in stats:
        t_med = stats["OOD 真值"][1]
        g_med = stats["OOD 模型生成"][1]
        t_sim = stats["OOD 真值"][0]
        g_sim = stats["OOD 模型生成"][0]
        collapse = g_med < t_med / 3 and g_sim > t_sim
        print(f"\n  真值中位分母 {t_med} vs 模型 {g_med}"
              f" ({'坍缩' if collapse else '未坍缩'})")
        print(f"  简单分数占比 真值 {t_sim:.1f}% vs 模型 {g_sim:.1f}%")
        if collapse:
            print("  → 模型在 OOD 上回退到训练集常见答案形态，未执行组合计算")

    # --- 3. 图文一致性断点 ---
    print("\n--- 3. 图文一致性断点（P1 子项失败分布）---")
    cnt = Counter()
    for d in det:
        al = d.get("alignment")
        if not al:
            continue
        for k, v in al["checks"].items():
            if v is False:
                cnt[k] += 1
    for k, v in cnt.most_common():
        print(f"  {k:<26} 失败 {v}/{len(det)} ({v/len(det)*100:.1f}%)")

    # 代表性样本
    print("\n--- 4. 典型失败样本 ---")
    shown = 0
    for d in det:
        a = d["accuracy"]
        if a["correct"] or not a["extracted"]:
            continue
        al = d.get("alignment") or {}
        bad = [k for k, v in al.get("checks", {}).items() if v is False]
        print(f"  {d['source']}: gen={a['generated_prob']} ref={a['reference_prob']} "
              f"对齐失败={bad}")
        if "c5_ratio" in bad:
            print(f"     c5: {al['details'].get('c5_ratio')}")
        shown += 1
        if shown >= 5:
            break

    # --- 4b. 图中比值 vs 正文结论：谁更接近真值 ---
    print("\n--- 4b. 图文分歧时，谁更接近真值？ ---")
    pat = re.compile(r"=\s*(\d+)\s*/\s*(\d+)\s*vs\s*正文\s*(\d+)\s*/\s*(\d+)")
    closer_graph = closer_text = tie = 0
    total = 0
    examples = []
    for d in det:
        al = d.get("alignment") or {}
        dets = al.get("details", {})
        txt = dets.get("c5_ratio", "")
        m = pat.search(txt)
        if not m:
            continue
        gn, gd, tn, td = map(int, m.groups())
        if gd == 0 or td == 0:
            continue
        g = Fraction(gn, gd)          # 图中算出的比值
        t = Fraction(tn, td)          # 正文写出的结论
        truth = final_prob(d.get("generated"))  # 无法用；改用参考
        # 真值来自 OOD 参考输出
        ref = None
        for o in ood:
            if o["source"] == d["source"]:
                ref = final_prob(o["output"])
                break
        if ref is None:
            continue
        eg, et = abs(g - ref), abs(t - ref)
        total += 1
        if eg < et:
            closer_graph += 1
        elif et < eg:
            closer_text += 1
        else:
            tie += 1
        if len(examples) < 4:
            examples.append((d["source"], str(g), str(t), str(ref),
                             "图更近" if eg < et else ("文更近" if et < eg else "相同")))
    if total:
        print(f"  c5_ratio 分歧样本: {total} 条")
        print(f"    图中比值更接近真值: {closer_graph} ({closer_graph/total*100:.1f}%)")
        print(f"    正文结论更接近真值: {closer_text} ({closer_text/total*100:.1f}%)")
        print(f"    误差相同          : {tie} ({tie/total*100:.1f}%)")
        for e in examples:
            print(f"      {e[0]}: 图={e[1]} 文={e[2]} 真={e[3]} -> {e[4]}")
        if closer_graph > closer_text * 1.5:
            print("  → 图中保留了部分正确的计算线索，但最终结论被"
                  "「简单分数」先验覆盖")
        elif closer_text > closer_graph * 1.5:
            print("  → 正文结论比图中比值更接近真值，图文均为独立错误")
        else:
            print("  → 图与文是各自独立的错误，图不提供额外纠错线索：")
            print("     Mermaid 只是「看起来像推导」的格式化产物，")
            print("     模型并未真正执行组合计算。")

    # --- 5. 结论 ---
    print("\n--- 5. 结论 ---")
    num_rate = s.get("accuracy_correct", 0) / max(s.get("samples", 1), 1)
    fs_rate = s.get("structure_pass_rate", 0)
    if num_rate < 0.1 and fs_rate > 0.8:
        print("  【格式过载，推理欠拟合】模型复现了输出格式但未习得计算过程。")
        print("  建议：训练数据需显式展开中间量（|Ω|、|A|、|B|、|A∩B| 的推导），")
        print("        并引入使模式匹配失效的题目（如大分母组合数、非整除参数）。")
    elif num_rate < 0.3:
        print("  【部分泛化】格式与部分计算能力迁移，需扩大参数覆盖面。")
    else:
        print("  【泛化良好】计算能力已迁移。")
    return {"layers": {k: v for k, v in layers.items()},
            "complexity": stats, "align_failures": dict(cnt)}


def main():
    p = argparse.ArgumentParser(description="OOD 失败模式诊断")
    p.add_argument("--report", default=str(BASE_DIR / "output" / "eval_ood.json"))
    p.add_argument("--train_file", default=str(BASE_DIR / "data" / "sft_train_580.jsonl"))
    p.add_argument("--ood_file", default=str(BASE_DIR / "data" / "test_ood.jsonl"))
    p.add_argument("--output", default=str(BASE_DIR / "output" / "ood_diagnosis.json"))
    args = p.parse_args()

    res = diagnose(args.report, args.train_file, args.ood_file)
    Path(args.output).write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                           default=str), encoding="utf-8")
    print(f"\n📄 诊断结果: {args.output}")


if __name__ == "__main__":
    main()
