
##  任务列表

- [x] **phase1**: 根据仓库qmd完整笔记，完成cold start 训练语料的构建工作
  - 输出: `data/cold_start_sft.jsonl` (30条 SFT 数据)
  - Mermaid 覆盖率: 30/30 (100%)
  - Julia 代码覆盖: 23/30
  - 分数形式: 22/30
  - 数值验证: 20/23 通过 (3条因 PrettyTables API `tf_matrix` 未定义，非数值错误)
  - 工具脚本:
    - `scripts/qmd_to_sft.py` — qmd → JSONL 解析
    - `scripts/data_audit.py` — 数据质量审计
    - `scripts/julia_validator.py` — Julia 代码数值验证
    - `scripts/mermaid_filler.py` — 生成 Mermaid 补全 Prompt
    - `scripts/mermaid_backfill.py` — Mermaid 图回填
  - Prompt 模板: `prompts/mermaid_fill_*.txt` (16个，用于后续 LLM 批量补全)

- [x] **phase2**: 使用 LLM 生成规模语料 (100条)
  - 基于冷启动数据模板，程序化参数变异合成
  - 每条经 Python 验证器（枚举样本空间精确计算）+ Mermaid 语法校验
  - 拒绝采样：跳过概率为0的平凡情况
  - [x] 抽样10题笔记渲染 (人工审核数据质量)
    - 输出: `rendered_notes/sample_notes.html`
    - 工具: `scripts/render_notes.py`
    - 抽样覆盖: 骰子(ex3.1, ex3.2) / 扑克牌(ex3.10) / 精算师路径(ex3.12) /
      医疗贝叶斯(ex3.16) / 双向表(ex3.21) / 三骰子(ex3.22) / 硬币悖论(ex3.37) /
      Polya Urn(ex3.40) / 保险分类(ex3.50)
    - 渲染内容: 12个Mermaid图 / 32个Julia代码块 / 5个概率表 / 30个LaTeX公式
  - [x] 修复 Mermaid 旧语法（`==label==>` → `-->|label|`）+ Quarto callout 渲染
  - [x] 合成100条规模语料
    - 输出: `data/sft_100.jsonl` (100条)
    - 合并训练集: `data/sft_train_130.jsonl` (30冷启动 + 100合成 = 130条)
    - 题型分布: 骰子30 / 坛子25 / 硬币20 / 双向表15 / 路径图10
    - 校验结果: Mermaid 100% / 概率数值 100% / 四步法结构 100%
    - 工具: `scripts/generate_sft_100.py` / `scripts/validate_sft_100.py`

- [x] **phase3**: 使用 LightRAG / nano-graphrag 为语料生成图结构
  -   ref: ./external/dilalogue1.md
  - 框架: LightRAG (lightrag-hku v1.5.7) + NetworkX 持久化
  - 四层异构图 Schema（来源 dialogue1.md）:
    - Source (5) — 数学原型节点（骰子/坛子/硬币/双向表/路径图）
    - ProblemText (130) — 题目文本节点
    - Numerical (114) — 数值参数节点
    - Emoji (29) — 视觉表征节点
    - SolutionStep (454) — 解答步骤节点
  - 六种边类型:
    - generates (130): Source → ProblemText
    - has_param (268): ProblemText → Numerical
    - emoji_to_num (35): Emoji → Numerical
    - represents (29): Source → Emoji
    - visualizes (454): ProblemText → SolutionStep
    - computes_to (100): SolutionStep → Numerical
    - same_source (470): ProblemText ↔ ProblemText (同源聚类)
  - 输出:
    - `data/kg_structure.json` — KG 节点和边（JSON）
    - `lightrag_data/kg_graph.graphml` — GraphML 格式图
    - `lightrag_data/kg_graph.pkl` — NetworkX pickle 快速加载
  - 检索验证:
    - 同源检索: ex3.1 → 35条同源骰子题 ✅
    - Emoji→数值: ⚪ → 23个数值映射 ✅
    - 原型聚合: 5个原型分别聚合36/34/23/23/14题 ✅
    - 全连通: 732节点单连通分量 ✅
  - 工具: `scripts/build_kg.py` / `scripts/query_kg.py`

- [ ] **phase4**: LoRA 训练
  - Base Model: Qwen2.5-Math-7B
  - LoRA r=32/64, target=all-linear, loss masking on output only
  - 评估: 结构化解析器 (四步法遵循率 + 数值准确率)
