#!/usr/bin/env python3
"""
evaluate_graphrag.py
Phase5 / P2-a: 无 GraphRAG vs 有 GraphRAG 的对比实验

设计:
  基线 (baseline)  : 直接给模型题目，生成解答
  GraphRAG 增强     : 先从 KG 检索同数学原型的代表性解答作为 few-shot 上下文，
                     注入 system prompt 后再生成
  对比维度          : 四步法遵循率 / Mermaid 覆盖率 / LaTeX 覆盖率 /
                     数值提取率 / 数值正确率 / Mermaid 语义对齐通过率

用法:
  python evaluate_graphrag.py --model_path <base> --adapter_path <adapter> \
      --test_file data/test_ood.jsonl --num_samples 20 --mode both \
      --report_name eval_graphrag_ood
"""
import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_mermaid_alignment import check_alignment  # P1 检查器


# ============================================================
# KG 检索器
# ============================================================
PROTO_KEYWORDS = [
    ("骰子条件概率", ["骰子", "骰", "点数为", "点数和", "dice"]),
    ("坛子抽样", ["坛子", "罐子", "弹子", "白球", "黑球", "红球", "盒子里", "抽屉",
                  "黑桃", "红桃", "扑克", "52张"]),
    ("硬币抛掷", ["硬币", "正面朝上", "正面为"]),
    ("双向表贝叶斯", ["家庭", "学生", "保险", "剖腹产", "加州", "猫", "狗",
                      "双向表", "婴儿", "戒烟", "夫妇"]),
    ("多阶段路径", ["孩子", "兄妹", "姐妹", "橱柜", "考试", "连续通过"]),
    # OOD 新题型（训练集无对应原型，退化为通用检索）
    ("超几何抽签", ["抽奖", "中奖", "券"]),
    ("几何概率", ["正方形", "随机投掷", "面积"]),
    ("二项分布", ["不合格率", "抽取", "二项"]),
    ("马尔可夫", ["马尔可夫", "转移矩阵"]),
    ("朴素贝叶斯", ["朴素贝叶斯", "分类器"]),
    ("正态分布", ["正态", "N(0"]),
    ("串联系统", ["串联", "生产线", "部件"]),
]


def infer_proto(text):
    """从题目文本推断数学原型。"""
    for proto, kws in PROTO_KEYWORDS:
        for kw in kws:
            if kw in text:
                return proto
    return None


class GraphRAGRetriever:
    """从 KG + 训练集检索原型解答作为上下文。"""

    def __init__(self, kg_file=None, sft_file=None):
        self.kg_file = kg_file or (BASE_DIR / "data" / "kg_structure.json")
        self.sft_file = sft_file or (BASE_DIR / "data" / "sft_train_580.jsonl")
        self.kg = None
        self.protos = {}       # proto -> [训练样例]
        self.emoji_by_proto = {}  # proto -> emoji 表征
        self._load()

    def _load(self):
        if Path(self.kg_file).exists():
            self.kg = json.loads(Path(self.kg_file).read_text(encoding="utf-8"))
            # represents 边：Source → Emoji
            for r in self.kg["relationships"]:
                if "Emoji 表征" in r["description"]:
                    proto = r["src_id"].replace("原型:", "")
                    self.emoji_by_proto.setdefault(proto, []).append(
                        r["tgt_id"].replace("Emoji:", ""))

        # 训练样例按 source 前缀分组
        if Path(self.sft_file).exists():
            for line in Path(self.sft_file).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                s = r.get("source", "")
                prefix = s.rsplit("-", 1)[0] if "-" in s else s
                if prefix.startswith("afcp-"):
                    prefix = "afcp"
                self.protos.setdefault(prefix, []).append(r)

    def proto_for_source(self, prefix):
        """synth-dice → 骰子条件概率"""
        mapping = {
            "synth-dice": "骰子条件概率",
            "synth-urn": "坛子抽样",
            "synth-coin": "硬币抛掷",
            "synth-table": "双向表贝叶斯",
            "synth-path": "多阶段路径",
            "synth-bayes": "双向表贝叶斯",
            "synth-chain": "多阶段路径",
            "synth-poker": "坛子抽样",
        }
        return mapping.get(prefix)

    def retrieve(self, question, k=1, max_chars=1200):
        """检索与问题同原型的代表性解答。"""
        proto = infer_proto(question)
        if not proto:
            return "", None
        # 原型 → 候选训练样例
        cands = []
        for prefix, items in self.protos.items():
            if self.proto_for_source(prefix) == proto:
                cands.extend(items)
        if not cands:
            return "", proto
        # 选取结构完整（四步法齐全）且长度适中的样例
        good = [c for c in cands
                if all(re.search(rf"#{{2,3}}\s*{i}\s*[\.、]", c["output"]) for i in range(1, 5))]
        pool = good or cands
        pool = sorted(pool, key=lambda x: len(x["output"]))
        mid = len(pool) // 2
        picked = [pool[mid]] if pool else []
        for c in picked[:k]:
            snippet = c["output"][:max_chars]
            emoji = self.emoji_by_proto.get(proto, [])
            ctx = (f"【知识图谱检索】该问题属于「{proto}」数学原型"
                   f"（图谱中的 Emoji 表征：{', '.join(emoji) if emoji else '无'}）。\n"
                   f"【参考范例解答】\n{snippet}")
            return ctx, proto
        return "", proto


# ============================================================
# 评估
# ============================================================
def extract_final_prob(text):
    matches = re.findall(r"=\s*(\d+)/(\d+)", text)
    if matches:
        n, d = matches[-1]
        try:
            return Fraction(int(n), int(d))
        except (ValueError, ZeroDivisionError):
            return None
    matches_int = re.findall(r"=\s*(\d+)\s*\$\$", text)
    if matches_int:
        return Fraction(int(matches_int[-1]), 1)
    return None


def evaluate_structure(text):
    checks = {
        "step1": bool(re.search(r"##\s*1\.", text)),
        "step2": bool(re.search(r"##\s*2\.", text)),
        "step3": bool(re.search(r"##\s*3\.", text)),
        "step4": bool(re.search(r"##\s*4\.", text)),
        "mermaid": "```mermaid" in text,
        "latex": "$$" in text,
        "table": "|" in text and "---" in text,
    }
    return checks


def run_inference(model, tokenizer, instruction, input_text, max_tokens=512):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()


def summarize(details):
    n = len(details)
    if n == 0:
        return {}
    def rate(key):
        return sum(1 for d in details if d["struct"][key]) / n
    return {
        "n": n,
        "four_step_rate": sum(1 for d in details
                              if all(d["struct"][f"step{i}"] for i in range(1, 5))) / n,
        "mermaid_rate": rate("mermaid"),
        "latex_rate": rate("latex"),
        "table_rate": rate("table"),
        "num_extract_rate": sum(1 for d in details if d["acc"]["extracted"]) / n,
        "num_correct_rate": sum(1 for d in details if d["acc"]["correct"]) / n,
        "alignment_rate": sum(1 for d in details if d["align_pass"]) / n,
        "avg_align_score": sum(d["align_score"] for d in details) / n,
    }


def evaluate(model_path, adapter_path, test_file, num_samples, mode,
             report_name, max_tokens=768, use_4bit=True, graphrag_k=1):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    records = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records = records[:num_samples]
    print(f"测试集: {test_file}  (样本 {len(records)})")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto",
            trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.config.use_cache = False
    print(f"模型就绪: {model_path}" + (f" + {adapter_path}" if adapter_path else ""))

    retriever = GraphRAGRetriever()
    results = {}

    for variant in (["baseline", "graphrag"] if mode == "both" else [mode]):
        print(f"\n{'='*60}\n运行变体: {variant}\n{'='*60}")
        details = []
        for i, r in enumerate(records):
            instruction = r["instruction"]
            if variant == "graphrag":
                ctx, proto = retriever.retrieve(r["input"], k=graphrag_k)
                if ctx:
                    instruction = instruction + "\n\n" + ctx
            gen = run_inference(model, tokenizer, instruction, r["input"], max_tokens)
            ref = r["output"]
            gp, rp = extract_final_prob(gen), extract_final_prob(ref)
            align = check_alignment(gen)
            align_pass = all(v is not False for v in align["checks"].values())
            details.append({
                "idx": i, "source": r.get("source", "?"),
                "generated": gen,
                "struct": evaluate_structure(gen),
                "acc": {"generated_prob": str(gp) if gp else None,
                        "reference_prob": str(rp) if rp else None,
                        "correct": gp is not None and rp is not None and gp == rp,
                        "extracted": gp is not None},
                "align_pass": align_pass,
                "align_score": align["score"], "align_total": align["total"],
                "align_details": align["details"],
            })
            print(f"  [{i+1}/{len(records)}] {r.get('source','?')[:34]:34s} "
                  f"struct={sum(details[-1]['struct'].values())}/7 "
                  f"num={'✓' if details[-1]['acc']['correct'] else '✗'} "
                  f"align={align['score']}/{align['total']}")

        results[variant] = {"summary": summarize(details), "details": details}

    # 对比
    print(f"\n{'='*60}\n对比 (无 GraphRAG vs 有 GraphRAG)\n{'='*60}")
    if "baseline" in results and "graphrag" in results:
        b, g = results["baseline"]["summary"], results["graphrag"]["summary"]
        print(f"{'指标':<22}{'baseline':>12}{'graphrag':>12}{'Δ':>10}")
        print("-" * 56)
        for k in ["four_step_rate", "mermaid_rate", "latex_rate", "table_rate",
                  "num_extract_rate", "num_correct_rate", "alignment_rate"]:
            bv, gv = b.get(k, 0), g.get(k, 0)
            print(f"{k:<22}{bv*100:>11.1f}%{gv*100:>11.1f}%{(gv-bv)*100:>+9.1f}pp")
        print(f"{'avg_align_score':<22}{b.get('avg_align_score',0):>12.2f}"
              f"{g.get('avg_align_score',0):>12.2f}"
              f"{g.get('avg_align_score',0)-b.get('avg_align_score',0):>+10.2f}")

    out = BASE_DIR / "output" / f"{report_name}.json"
    out.write_text(json.dumps({
        "test_file": str(test_file), "model_path": model_path,
        "adapter_path": adapter_path, "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 结果: {out}")
    return results


def main():
    p = argparse.ArgumentParser(description="GraphRAG 对比实验")
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--test_file", default=str(BASE_DIR / "data" / "sft_train_580.jsonl"))
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--mode", choices=["baseline", "graphrag", "both"], default="both")
    p.add_argument("--report_name", default="eval_graphrag")
    p.add_argument("--max_tokens", type=int, default=768)
    p.add_argument("--graphrag_k", type=int, default=1)
    p.add_argument("--fp16", action="store_true", help="不用 4-bit")
    args = p.parse_args()

    evaluate(args.model_path, args.adapter_path, args.test_file,
             args.num_samples, args.mode, args.report_name,
             args.max_tokens, use_4bit=not args.fp16, graphrag_k=args.graphrag_k)


if __name__ == "__main__":
    main()
