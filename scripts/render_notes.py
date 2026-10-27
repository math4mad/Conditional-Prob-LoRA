#!/usr/bin/env python3
"""
render_notes.py
从 cold_start_sft.jsonl 抽样10条代表性题目，渲染为含 Mermaid + LaTeX 的 HTML 可视化笔记。

每条笔记包含：
- 题目编号与来源
- instruction (system prompt)
- input (题干)
- output (解答，含 Mermaid 图渲染 + Julia 代码高亮 + LaTeX 公式)
"""

import json
import os
import html

JSONL_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "cold_start_sft.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "rendered_notes")

# 抽样10条，覆盖不同题型
SAMPLES = [
    "afcp-ex3.1.qmd",    # 骰子条件概率（基础四步法）
    "afcp-ex3.2.qmd",    # 骰子sum=i条件（遍历计算）
    "afcp-ex3.10.qmd",   # 扑克牌条件概率
    "afcp-ex3.12.qmd",   # 精算师考试路径图
    "afcp-ex3.16.qmd",   # 医疗贝叶斯（剖腹产）
    "afcp-ex3.21.qmd",   # 双向表（夫妇收入）
    "afcp-ex3.22.qmd",   # 三骰子
    "afcp-ex3.37.qmd",   # 赌徒硬币悖论
    "afcp-ex3.40.qmd",   # Polya Urn 模型（完整路径图）
    "afcp-ex3.50.qmd",   # 保险风险分类
]


def render_output(output_text: str) -> str:
    """将 output 文本转为 HTML，保留代码块、Mermaid、表格。"""
    # 转义 HTML
    text = html.escape(output_text)

    # Mermaid 代码块 → <pre class="mermaid">
    text = __import__("re").sub(
        r"```(?:\{)?mermaid(?:\})?\s*\n(.*?)```",
        r'<div class="mermaid">\1</div>',
        text, flags=__import__("re").DOTALL,
    )

    # Julia 代码块 → <pre><code class="language-julia">
    text = __import__("re").sub(
        r"```(?:\{)?julia(?:\})?\s*\n(.*?)```",
        r'<pre class="code-block"><code class="language-julia">\1</code></pre>',
        text, flags=__import__("re").DOTALL,
    )

    # 其他代码块
    text = __import__("re").sub(
        r"```([^\n]*)\n(.*?)```",
        r'<pre class="code-block"><code>\2</code></pre>',
        text, flags=__import__("re").DOTALL,
    )

    # Markdown 表格（简单处理）
    lines = text.split("\n")
    result = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                result.append('<table class="prob-table">')
                in_table = True
                # 表头
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                result.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
                continue
            # 跳过分隔行 |:---:|
            if __import__("re").match(r"^\|[\s:|-]+\|$", stripped):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            result.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        else:
            if in_table:
                result.append("</tbody></table>")
                in_table = False
            result.append(line)
    if in_table:
        result.append("</tbody></table>")

    text = "\n".join(result)

    # 标题
    text = __import__("re").sub(r"^### (.+)$", r"<h4>\1</h4>", text, flags=__import__("re").MULTILINE)
    text = __import__("re").sub(r"^## (.+)$", r"<h3>\1</h3>", text, flags=__import__("re").MULTILINE)

    # 行内 LaTeX $...$ → MathJax
    text = __import__("re").sub(r"\$(.+?)\$", r"\\(\1\\)", text)

    # 段落
    paragraphs = text.split("\n\n")
    html_parts = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith("<") or p.startswith("\\("):
            html_parts.append(p)
        else:
            html_parts.append(f"<p>{p.replace(chr(10), '<br/>')}</p>")

    return "\n".join(html_parts)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {{
    mermaid.initialize({{ startOnLoad: true, theme: "default", securityLevel: "loose" }});
  }});
</script>
<style>
  body {{ font-family: -apple-system, "Noto Sans SC", "PingFang SC", sans-serif;
          max-width: 900px; margin: 40px auto; padding: 0 20px;
          background: #fafafa; color: #333; line-height: 1.8; }}
  .note-card {{ background: #fff; border-radius: 12px; padding: 30px;
               box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 30px; }}
  .source-tag {{ display: inline-block; background: #e3f2fd; color: #1565c0;
                 padding: 4px 12px; border-radius: 6px; font-size: 0.85em;
                 font-family: monospace; margin-bottom: 16px; }}
  .section-label {{ font-weight: 700; color: #1565c0; font-size: 0.9em;
                    text-transform: uppercase; letter-spacing: 1px; margin-top: 24px; }}
  .instruction-box {{ background: #f5f5f5; border-left: 4px solid #1565c0;
                      padding: 12px 16px; border-radius: 4px; font-size: 0.9em; color: #555; }}
  .input-box {{ background: #fff8e1; border-left: 4px solid #ff8f00;
                padding: 16px; border-radius: 4px; margin: 12px 0; }}
  .output-content h3, .output-content h4 {{ color: #283593; margin-top: 24px; }}
  .code-block {{ background: #263238; color: #eceff1; padding: 16px;
                 border-radius: 8px; overflow-x: auto; font-family: "JetBrains Mono", monospace;
                 font-size: 0.85em; }}
  .code-block code {{ background: none; color: inherit; }}
  .mermaid {{ text-align: center; margin: 20px 0; }}
  .prob-table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  .prob-table th, .prob-table td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: center; }}
  .prob-table th {{ background: #e3f2fd; font-weight: 600; }}
  .prob-table tr:nth-child(even) {{ background: #f9f9f9; }}
  .index-bar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 30px; }}
  .index-bar a {{ padding: 6px 14px; background: #fff; border: 1px solid #ddd;
                 border-radius: 6px; text-decoration: none; color: #1565c0; font-size: 0.85em; }}
  .index-bar a:hover {{ background: #e3f2fd; }}
  h1 {{ color: #1a237e; }}
</style>
</head>
<body>
<h1>条件概率练习笔记 — 抽样渲染 (10题)</h1>
<p style="color:#666">基于 cold_start_sft.jsonl 冷启动语料，渲染 Mermaid 路径图 + LaTeX 公式 + Julia 代码。</p>
<div class="index-bar">
{index_bar}
</div>
{notes}
</body>
</html>
"""


def main():
    records = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # 索引 by source
    by_source = {r["source"]: r for r in records}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    notes_html = []
    index_links = []
    for src in SAMPLES:
        if src not in by_source:
            print(f"⚠️ 未找到 {src}")
            continue
        r = by_source[src]
        ex_id = src.replace("afcp-", "").replace(".qmd", "")
        title = f"练习 {ex_id}"

        rendered = render_output(r["output"])
        note_html = f"""
<div class="note-card" id="{ex_id}">
  <span class="source-tag">{src}</span>
  <h2>{title}</h2>
  <div class="section-label">Instruction</div>
  <div class="instruction-box">{html.escape(r['instruction'])}</div>
  <div class="section-label">题目 (Input)</div>
  <div class="input-box">{html.escape(r['input']).replace(chr(10), '<br/>')}</div>
  <div class="section-label">解答 (Output)</div>
  <div class="output-content">
{rendered}
  </div>
</div>
"""
        notes_html.append(note_html)
        index_links.append(f'<a href="#{ex_id}">{title}</a>')
        print(f"  ✅ {src}")

    full_html = HTML_TEMPLATE.format(
        title="条件概率笔记抽样渲染",
        index_bar="\n".join(index_links),
        notes="\n".join(notes_html),
    )

    out_path = os.path.join(OUTPUT_DIR, "sample_notes.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"\n📄 渲染完成: {out_path}")
    print(f"   共 {len(notes_html)} 条笔记")


if __name__ == "__main__":
    main()
