#!/usr/bin/env python3
r"""
qmd_to_sft.py
将 afcp-exercises 目录下的 .qmd 笔记批量转换为 LoRA SFT (JSONL) 语料。

设计要点（基于对仓库内 30 个 qmd 文件的实际结构分析）：
1. 标题层级不统一：ex3.1 用 `### 1. question`，ex3.40 用 `## 1. question`，还有拼写 `qustion`。
   -> 用 `#{2,3}\s+1\.\s+(question|qustion|问题)` 忽略大小写匹配第一节。
2. question 内容形态多样：
   - 纯文本 / 列表（ex3.1）
   - `>` 引用块（ex3.2, ex3.40, ex3.25）
   - julia 注释块 `#= ... =#`（ex3.10）
   -> 优先取引用块；否则取 julia 注释；否则取标题后到下一个标题前的纯文本（剔除代码块）。
3. 其余所有章节（含 julia 代码块、mermaid、公式）原样拼为 output，不做转义。
4. 输出 ensure_ascii=False，保留中文与 // 分数。
"""

import os
import re
import json
from pathlib import Path

# === 配置区 ===
QMD_DIR = os.path.join(
    os.path.dirname(__file__), "..", "external", "conditional-prob-exercise", "afcp-exercises"
)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "cold_start_sft.jsonl")

SYSTEM_PROMPT = (
    "你是一个概率论教学专家。请严格按照【空间构建->条件筛选->目标事件->计算与可视化】四步法解决概率问题。"
    "所有概率计算必须使用分数形式，避免小数近似。回答中必须包含Mermaid路径图和LaTeX条件概率表。"
)

# 匹配 qmd YAML front matter
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

# 匹配任意级别的 "1. question/qustion/问题" 标题
QUESTION_HEAD_RE = re.compile(
    r"^#{2,3}\s+1\.\s+(question|qustion|问题)\s*$", re.IGNORECASE | re.MULTILINE
)

# 匹配所有数字编号标题（用于切片）
SECTION_HEAD_RE = re.compile(r"^#{2,3}\s+\d+\.\s+", re.MULTILINE)

# 匹配任意中文/非编号标题（如 ex3.12 的 "### 精算师考试"），用于兜底切片
ANY_HEAD_RE = re.compile(r"^#{2,3}\s+\S", re.MULTILINE)


def strip_front_matter(text: str) -> str:
    m = FRONT_MATTER_RE.match(text)
    return text[m.end():] if m else text


def extract_julia_comment(text: str) -> str:
    """提取 julia 块注释 #= ... =# 中的内容。"""
    m = re.search(r"#=\s*\n(.*?)\n\s*=#", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_blockquotes(text: str) -> str:
    """提取 > 引用块内容（去除 > 前缀）。"""
    lines = text.split("\n")
    bq = [ln for ln in lines if ln.lstrip().startswith(">")]
    if not bq:
        return ""
    cleaned = [re.sub(r"^\s*>\s?", "", ln) for ln in bq]
    return "\n".join(cleaned).strip()


def strip_code_blocks(text: str) -> str:
    """移除 fenced code block，返回纯文本部分。"""
    return re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.DOTALL)


def fix_mermaid_syntax(text: str) -> str:
    r"""
    修正 Quarto/Mermaid 旧语法以兼容 Mermaid v10：
    1. 移除 `%%|` 元数据行（fig-width/label/fig-cap 等）。
    2. `A ==label==> B` → `A -->|label| B`
       `A -.label.-> B` 保持不变（已兼容）
    """

    def replace_mermaid(m):
        code = m.group(1)
        # 移除 %%| 元数据行
        code = re.sub(r"^\s*%%\|.*$", "", code, flags=re.MULTILINE)
        # 修正 ==label==> 边语法 → -->|label|
        # 原始格式: NodeEnd == label ==> NodeStart
        # 例如: B(success) ==s2:0.8 ==> D(success)
        # 目标:   B(success) -->|s2:0.8| D(success)
        code = re.sub(r"==\s*(.*?)\s*==>", lambda mm: f"--><u>{mm.group(1)}</u> -->|placeholder|", code)
        # 上面方法仍不正确——需要把标签放在 -->|label| 中间
        # 用更直接的方式：匹配 == ... ==> 替换为 -->|...|  但需确保后面的节点前有 |
        # 实际上 mermaid 中 A -->|label| B 的语法，| 只在标签两侧
        # 所以 ==label==> 应整体替换为 -->|label|
        # 但 == 前的节点已结束(如 ), )，==> 后直接跟节点
        # 故: ==label==>  →  -->|label|  (注意去掉末尾空格再接节点)
        # 重新来：先恢复code，再正确替换
        return code  # placeholder, will redo below

    # 正确实现：直接对 == ... ==> 做替换
    def replace_mermaid_correct(m):
        code = m.group(1)
        # 移除 %%| 元数据行
        code = re.sub(r"^\s*%%\|.*$", "", code, flags=re.MULTILINE)
        # ==label==> → -->|label|
        # 注意 label 中可能含 / : % 等字符，用非贪婪匹配到 ==>
        code = re.sub(r"==\s*([^=]*?)\s*==>", lambda mm: f" -->|{mm.group(1).strip()}| ", code)
        # 清理多余空行和前导空格导致的空行
        code = re.sub(r"\n{3,}", "\n\n", code)
        return f"```mermaid\n{code}```"

    text = re.sub(
        r"```(?:\{)?mermaid(?:\})?\s*\n(.*?)```",
        replace_mermaid_correct,
        text,
        flags=re.DOTALL,
    )
    return text


def strip_callouts(text: str) -> str:
    """
    将 Quarto callout 语法 ::: {.callout-xxx title="..."} ... :::
    转为普通文本：保留标题行和内容，移除 ::: 标记。
    """
    # ::: {.callout-tip title="build two-way table"}  →  **[提示] build two-way table**
    # ::: {.callout-note title="Question"} →  **[笔记] Question**
    # ::: {.callout-warning title="总结"} →  **[警告] 总结**
    # ::: {.callout-note} → (无标题则移除)
    # :::::: {.callout-note ...} (多个冒号)

    def callout_open(m):
        full = m.group(0)
        # 提取类型
        type_m = re.search(r"callout-(\w+)", full)
        ctype = type_m.group(1) if type_m else "note"
        # 提取 title
        title_m = re.search(r'title="([^"]*)"', full)
        title = title_m.group(1) if title_m else ""
        label_map = {"tip": "提示", "note": "笔记", "warning": "注意", "caution": "注意"}
        label = label_map.get(ctype, ctype)
        if title:
            return f"\n**[{label}] {title}**\n"
        return "\n"

    # 匹配开头的 :::（含多个冒号）后跟 {.callout...}
    text = re.sub(r"^\s*:+\s*\{[^}]*callout[^}]*\}.*$", callout_open, text, flags=re.MULTILINE)
    # 匹配闭合 :::
    text = re.sub(r"^\s*:+\s*$", "", text, flags=re.MULTILINE)
    return text


def parse_question(section_body: str) -> str:
    """从 question 章节正文中提取题干文本。"""
    body = section_body.strip()
    # 1) 优先引用块
    bq = extract_blockquotes(body)
    if bq:
        return bq
    # 2) julia 注释块
    cmt = extract_julia_comment(body)
    if cmt:
        return cmt
    # 3) 去掉代码块后的纯文本
    txt = strip_code_blocks(body).strip()
    return txt


def split_sections(content: str) -> list[tuple[str, str]]:
    """
    按标题切片。
    优先用 #{2,3} N. 数字编号标题；若无数字编号，回退到任意 #{2,3} 标题。
    返回 [(header_line, body), ...]，header_line 含原标题行。
    """
    matches = list(SECTION_HEAD_RE.finditer(content))
    if not matches:
        matches = list(ANY_HEAD_RE.finditer(content))
    if not matches:
        return []
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        chunk = content[start:end].strip()
        # 分离标题行与正文
        nl = chunk.find("\n")
        header = chunk[:nl].strip() if nl != -1 else chunk
        body = chunk[nl + 1:].strip() if nl != -1 else ""
        sections.append((header, body))
    return sections


def parse_qmd(filepath: str) -> dict | None:
    try:
        raw = Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ 读取失败 {filepath}: {e}")
        return None

    content = strip_front_matter(raw).strip()
    sections = split_sections(content)
    if not sections:
        print(f"⚠️ 跳过 {filepath}: 未找到编号章节")
        return None

    # 第一个 "1." 编号章节作为 question，其余章节作为 solution。
    # 标题文字可能是 question/qustion/问题/调查项目/从52张牌... 等，统一按编号识别。
    question_body = ""
    solution_parts = []
    first_num_re = re.compile(r"^#{2,3}\s+1\.\s", re.MULTILINE)
    found_question = False
    for header, body in sections:
        if not found_question and first_num_re.match(header):
            question_body = body
            found_question = True
            continue
        solution_parts.append(f"{header}\n{body}")

    if not found_question:
        # 兜底：无 "1." 编号时，第一节当 question，其余为 solution
        header, body = sections[0]
        question_body = body
        solution_parts = [f"{h}\n{b}" for h, b in sections[1:]]

    question = parse_question(question_body)
    if not question:
        question = question_body.strip()

    # question 章节正文中，剔除题干（引用块/julia注释）后若仍含代码块
    # （ex3.47：题干引用块后紧跟 julia 解答代码），则作为 solution 的一部分。
    if question:
        remaining = re.sub(
            r"(?:^\s*>\s?.*\n?)+", "", question_body, flags=re.MULTILINE
        ).strip()
        if "```" in remaining:
            solution_parts.insert(0, remaining)

    solution = "\n\n".join(solution_parts).strip()

    # 清理 Quarto callout 语法 + 修正 Mermaid 旧语法
    question = strip_callouts(question).strip()
    solution = strip_callouts(solution)
    solution = fix_mermaid_syntax(solution)

    if not question or not solution:
        print(f"⚠️ 跳过 {filepath}: 缺少 question 或 solution")
        return None

    return {
        "instruction": SYSTEM_PROMPT,
        "input": question,
        "output": solution,
        "source": os.path.basename(filepath),
    }


def convert_all(qmd_dir: str, output_file: str):
    qmd_files = sorted(Path(qmd_dir).glob("*.qmd"))
    print(f"📂 找到 {len(qmd_files)} 个 .qmd 文件")

    results = []
    for f in qmd_files:
        record = parse_qmd(str(f))
        if record:
            results.append(record)
            print(f"✅ {f.name}  (input {len(record['input'])}字 / output {len(record['output'])}字)")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n🎉 转换完成! 共 {len(results)}/{len(qmd_files)} 条有效数据")
    print(f"📄 输出文件: {output_file}")

    # 统计 mermaid / julia 覆盖
    has_mermaid = sum(1 for r in results if "```mermaid" in r["output"] or "```{mermaid}" in r["output"])
    has_julia = sum(1 for r in results if "```julia" in r["output"] or "```{julia}" in r["output"])
    has_fraction = sum(1 for r in results if "//" in r["output"])
    print(f"\n📊 覆盖率统计:")
    print(f"   Mermaid 图: {has_mermaid}/{len(results)}")
    print(f"   Julia 代码: {has_julia}/{len(results)}")
    print(f"   分数形式:   {has_fraction}/{len(results)}")

    if results:
        print("\n=== 第一条数据预览 ===")
        preview = results[0].copy()
        preview["output"] = preview["output"][:300] + "..."
        print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    convert_all(os.path.abspath(QMD_DIR), os.path.abspath(OUTPUT_FILE))
