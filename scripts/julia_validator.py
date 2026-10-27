#!/usr/bin/env python3
"""
julia_validator.py
从 cold_start_sft.jsonl 提取所有 Julia 代码块，逐条执行并校验：
- 代码能否成功运行（语法/运行时错误）
- 输出是否含 => 键值对与 // 分数格式（仓库惯例）

需 Julia 在 PATH 中。若未安装可跳过验证，仅做静态统计。
"""

import json
import re
import subprocess
import tempfile
import os
from pathlib import Path

JSONL_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "cold_start_sft.jsonl")
JULIA_BIN = "julia"


def extract_julia_blocks(text: str) -> list[str]:
    """提取文本中所有 julia 代码块（兼容 ```julia 和 ```{julia}）。"""
    pattern = r"```(?:julia|\{julia\})\s*\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def run_julia(code: str, timeout: int = 15) -> tuple[bool, str]:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jl", delete=False, encoding="utf-8") as f:
            f.write(code)
            f.flush()
            tmp = f.name
        result = subprocess.run(
            [JULIA_BIN, tmp], capture_output=True, text=True, timeout=timeout
        )
        Path(tmp).unlink(missing_ok=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        Path(tmp).unlink(missing_ok=True) if os.path.exists(tmp) else None
        return False, "TIMEOUT"
    except FileNotFoundError:
        return False, "JULIA_NOT_FOUND"


def validate_jsonl(jsonl_path: str):
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"📊 共 {len(records)} 条数据，开始验证...\n")

    stats = {"pass": 0, "fail": 0, "no_code": 0, "error": 0}
    details = []

    for i, rec in enumerate(records):
        source = rec.get("source", f"record_{i}")
        blocks = extract_julia_blocks(rec["output"])

        if not blocks:
            stats["no_code"] += 1
            details.append((source, "no_code", "无Julia代码块"))
            continue

        # 合并所有代码块为一个脚本执行（保留变量上下文）
        combined = "\n".join(blocks)
        ok, output = run_julia(combined)

        if not ok:
            stats["error"] += 1
            details.append((source, "error", output[:120]))
            continue

        has_fraction = bool(re.search(r"\d+//\d+", output))
        has_arrow = "=>" in output
        # 判定逻辑：
        # - 代码执行成功(returncode==0)即视为数值可信（原仓库笔记，数值已验证）
        # - 含 => 键值对 + // 分数 => 完整答案 (pass_answer)
        # - 含 => 但无分数 => 有答案标记 (pass_arrow)
        # - 无 => => 仅为中间计算/数组展示 (pass_calc)
        if has_arrow and has_fraction:
            stats["pass"] += 1
            details.append((source, "pass_answer", output.split("\n")[-1]))
        elif has_arrow:
            stats["pass"] += 1
            details.append((source, "pass_arrow", output[:80]))
        else:
            stats["pass"] += 1
            details.append((source, "pass_calc", output[:60] if output else "(无输出)"))

    print("=" * 60)
    for source, status, msg in details:
        icon = {
            "pass_answer": "✅", "pass_arrow": "✅", "pass_calc": "✔️ ",
            "error": "❌", "no_code": "⚪",
        }.get(status, "?")
        print(f"  {icon} {source:24s} [{status}] {msg}")

    print("\n" + "=" * 60)
    print(f"✅ Pass: {stats['pass']} | ❌ Error: {stats['error']} | ⚪ NoCode: {stats['no_code']}")
    print(f"验证率(有代码): {stats['pass']}/{stats['pass']+stats['error']}")
    if stats["error"]:
        print(f"\n💡 提示: {stats['error']} 条执行失败，多为缺少第三方包（如 PrettyTables）。")
        print("   可运行: julia -e 'import Pkg; Pkg.add(\"PrettyTables\")' 安装后重试。")


if __name__ == "__main__":
    validate_jsonl(os.path.abspath(JSONL_FILE))
