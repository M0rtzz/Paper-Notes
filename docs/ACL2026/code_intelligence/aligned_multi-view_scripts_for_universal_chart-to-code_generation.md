---
title: >-
  [论文解读] Aligned Multi-View Scripts for Universal Chart-to-Code Generation
description: >-
  [ACL 2026][代码智能][Chart-to-Code] 把"同一张图表用 Python / R / LaTeX 三种语言写出语义等价脚本"作为新的监督信号，构建了 176K 四元组数据集 Chart2NCode…
tags:
  - "ACL 2026"
  - "代码智能"
  - "Chart-to-Code"
  - "多语言对齐"
  - "LLaVA"
  - "低秩子空间适配器"
  - "MoE 投影器"
---

# Aligned Multi-View Scripts for Universal Chart-to-Code Generation

**会议**: ACL 2026  
**arXiv**: [2604.24559](https://arxiv.org/abs/2604.24559)  
**代码**: [GitHub](https://github.com/Zhihan72/CharLuMA)  
**领域**: 代码智能 / 多模态  
**关键词**: Chart-to-Code, 多语言对齐, LLaVA, 低秩子空间适配器, MoE 投影器

## 一句话总结
把"同一张图表用 Python / R / LaTeX 三种语言写出语义等价脚本"作为新的监督信号，构建了 176K 四元组数据集 Chart2NCode，并提出在 LLaVA 投影器上加一个"语言条件的低秩子空间路由"的轻量适配器 CharLuMA，让一个模型在三种绘图语言上都达到可执行率与视觉保真度双高的水准。

## 研究背景与动机

**领域现状**：Chart-to-code（把图表图像还原成可执行绘图脚本）能让静态图回到可编辑、可重现状态。现有工作几乎都以 Python/matplotlib 为唯一目标语言，最近一年 ChartMimic、Plot2Code、ChartCoder 等都局限单语言。

**现有痛点**：(1) 真实学术界 R (ggplot2) 和 LaTeX (TikZ) 是大量学科的出版标准，单一 Python 输出不够用；(2) 更深层地，同一张图表本可以由不同语言的等价脚本表达——这种跨语言对齐天然是一种**多视图监督信号**，但单语言数据集完全没有利用；(3) 简单地把多语言数据塞进同一个模型，要么需要为每个语言训独立专家（参数翻倍且不共享知识），要么会出现语言之间相互干扰、专项化失衡。

**核心矛盾**：模型既要"共享一份图表语义理解"，又要"按目标语言走对应的语法专门通道"——LLaVA 一个 MLP 投影器同时干两件事会冲突。

**本文目标**：(a) 提供首个跨 Python/R/LaTeX 对齐的图表-代码四元组数据集；(b) 设计参数高效的多语言适配机制，能在共享视觉理解的前提下专项化输出语法。

**切入角度**：把不同语言脚本视为同一图表语义的"互补视图"，按多视图表示学习的思路对齐它们；架构上借鉴 Mixture-of-Subspaces LoRA 的思想，用低秩子空间池 + 语言条件路由代替"独立语言专家"。

**核心 idea**：用"元数据-模板"管线合成跨语言对齐脚本，用"低秩子空间适配器 + 语言路由"在 LLaVA 投影器上轻量插入语言专门化能力。

## 方法详解

### 整体框架

(1) Chart2NCode 数据构建：先收 Python / LaTeX / R 单语言脚本，执行/解析提取统一的"figure / axis / object"三级元数据，再按 object 级模式匹配人工模板池（202 个模板 × 20+ 图表子类型），实例化得到三语言脚本；模板未命中或执行失败的样本走 GPT-4o LLM-assisted debugging 兜底，渲染失败则丢弃；最终 176K 四元组，其中 14.7% 由 LLM 校正。

(2) CharLuMA 模型：SigLIP 视觉编码器 + DeepSeek-Coder LLM 后端 + LLaVA 风格两层 MLP 投影器，在 MLP 之外并联一个"低秩子空间适配器"——该适配器由低秩降维矩阵 $\mathbf{A}$、子空间池 $\{b_i\}_{i=1}^N$、语言专属路由器 $\mathbf{W}^l$ 三部分组成，按输入图像 + 目标语言动态选 top-$r$ 个子空间组合，输出与基础 MLP 相加得到最终视觉 token。训练分两阶段：先 align 预训练只训 MLP，再 instruction tuning 阶段 warm-up 路由器和子空间池、然后联合微调 LLM。

### 关键设计

1. **元数据-模板对齐管线（Chart2NCode 数据构建）**:

    - 功能：把单语言绘图脚本批量"翻译"成跨 Python/R/LaTeX 视觉等价的三语言脚本。
    - 核心思路：用语言原生 API 提取三层元数据（figure 级全局属性、axis 级坐标系、object 级几何 + 样式），通过 object 模式（如矩形等高变宽 → 横向柱状图）匹配人工策划的模板池；模板含跨语言属性映射字典（Python "upper right" ↔ R "right" ↔ LaTeX "north east"；Python bold 字重 ↔ LaTeX bfseries），保证语义在三种语法间统一。失败样本由 GPT-4o 翻译/修复，再次渲染验证。1000 个样本人工 1-5 评分四个维度均 95%+ 通过率（α=0.81）。
    - 设计动机：完全 LLM 翻译成本高且语义易漂移；纯规则模板覆盖率有限。两者结合既保证规模又保证保真度。

2. **语言条件的低秩子空间适配器（Language-conditioned Subspace Adapter）**:

    - 功能：在共享视觉 MLP 基础上注入语言专门化能力，避免独立语言专家的参数冗余和 Mixture-of-MLP 的容量浪费。
    - 核心思路：给 $\mathbf{Z}_v$ 先过共享 MLP 得到 $\mathbf{H}_{\text{base}} = \mathbf{W}\mathbf{Z}_v$；并行地用低秩矩阵 $\mathbf{A}$ 压到 rank-$r$ 表示，再经语言路由 $y^l = \mathrm{top}_r(\mathrm{softmax}(\mathbf{W}^l \overline{\mathbf{Z}}_v))$ 从 $N=32$ 个子空间池里选 $r=16$ 个，拼成 $\mathbf{B}$；语言自适应视觉 token 为 $\mathbf{H}_v = \mathbf{W}\mathbf{Z}_v + \mathbf{A}\mathbf{B}\mathbf{Z}_v$。
    - 设计动机：低秩 + 子空间池让"共享核心 + 语言专项"以参数最经济的方式共存；激活分析显示 1.3B 模型有大约 5/27 个子空间被三语言共享，其余各语言独占，验证了"compact shared core + language-specific capacity"的设计预期。

3. **两阶段渐进训练策略（Alignment Pretrain + Instruction Tuning）**:

    - 功能：先稳定模态对齐，再稳定语言路由，最后让 LLM 学会用语言自适应 token。
    - 核心思路：阶段 1 在 900K Chart-JSON 对上单训 MLP $\mathbf{W}$，冻视觉和 LLM；阶段 2 加入子空间适配器，先 274 步只 warm-up 路由器 $\mathbf{W}^l$ 和子空间池 $\{b_i\}$（MLP / 视觉 / LLM 全冻、$\mathbf{A}$ 随机初始化后全程冻），再解冻 LLM 联合训练（MLP 和 $\mathbf{A}$ 仍冻）。每个 batch 强制包含全部三种语言。
    - 设计动机：$\mathbf{A}$ 全程冻是为了把适配容量逼向"语言差异"而非重复学视觉共性；warm-up 路由器避免路由还没收敛就被 LLM 反向梯度搅乱。

### 损失函数 / 训练策略

标准 next-token cross-entropy，无额外辅助 loss。两阶段学习率：预训练 2e-4、warm-up 2e-4、联合微调 2e-5。CharLuMA-1.3B 训练共 82 GPU 小时（8×L40S），6.7B 约 321 GPU 小时。

## 实验关键数据

### 主实验

在 Chart2NCode test set (1000 样本) 上跨三语言平均，主要指标 ER（可执行率）/ DS（DreamSim 视觉相似度）/ MJ（MLLM-as-Judge）：

| 模型 | Python ER | Python DS | R ER | R DS | LaTeX ER | LaTeX DS |
|------|----------|-----------|------|------|----------|----------|
| GPT-4o | 98.5 | 85.0 | 94.5 | 78.8 | 88.4 | 72.4 |
| Claude-Sonnet-4 | 98.3 | 86.8 | 93.9 | 82.0 | 92.7 | 76.0 |
| Qwen3-VL-8B | 91.1 | 83.7 | 73.6 | 72.7 | 77.3 | 66.8 |
| ChartCoder-7B (Python 专家) | 96.2 | 48.1 | - | - | 17.9 | 39.1 |
| InternVL3.5-8B | 82.5 | 79.6 | 67.0 | 67.6 | 81.1 | 57.1 |
| **CharLuMA-1.3B** | 94.4 | 86.5 | 94.5 | 78.9 | 84.5 | 71.3 |
| **CharLuMA-6.7B** | 98.0 | 88.7 | 96.5 | 81.8 | 89.0 | 72.5 |

6.7B 在 R 上 96.5 ER / 81.8 DS 接近 Claude-Sonnet-4；ChartCoder-7B 这类 Python 专家在 R/LaTeX 上完全崩盘（R 直接 0 可执行），凸显多语言对齐价值。

### 消融实验

**架构对比**（Chart2NCode 三语言平均）：

| 投影器架构 | 1.3B ER | 1.3B DS | 1.3B MJ | 6.7B ER | 6.7B DS | 6.7B MJ |
|----------|---------|---------|---------|---------|---------|---------|
| Linear MLP | 88.1 | 76.9 | 69.5 | 91.0 | 78.2 | 76.3 |
| Mixture-of-MLP | 87.9 | 75.1 | 68.2 | 91.9 | 77.4 | 76.8 |
| **Subspace Adapter (本文)** | **91.1** | **78.9** | **72.3** | **94.5** | **81.0** | **81.1** |

**子空间-路由配置**（1.3B）：

| 子空间总数 | 激活数 | 路由器数 | ER | DS | MJ |
|----------|-------|---------|-----|------|------|
| 16 | 8 | 3 | 88.9 | 77.6 | 70.5 |
| 32 | 16 | 1 (共享) | 86.1 | 75.1 | 67.0 |
| 32 | 32 | 0 | 85.8 | 73.2 | 66.3 |
| **32** | **16** | **3 (语言专属)** | **91.1** | **78.9** | **72.3** |
| w/o warm-up | - | - | 87.1 | 75.6 | 67.9 |
| 解冻 $\mathbf{A}$ | - | - | 90.2 | 78.0 | 70.1 |

**语言多样性**：训三语言 > 训两语言 > 训单语言，即便每张图被分到的训练量减少；不对齐源数据基线偏向 Python，验证了对齐监督的必要性。

### 关键发现
- 三语言对齐训练比单语言总训练量相同的设置在所有语言上都更好——多视图监督能跨语言增强各自表现。
- 32-16 配置 + 3 个语言专属路由是最佳点，把语言路由器换成共享路由器立刻掉 ~4 个 DS，再去掉路由器掉更多——证明路由器是关键。
- 子空间激活分析显示 1.3B 仅 19% 的激活子空间在三语言间共享（5/27），6.7B 类似（18%），说明 scale up 时容量被自动分配给"语言专项"。
- LaTeX 失败模式独特：55.5% 是语法约束（缺花括号）；Python/R 失败主因是维度不匹配和未定义变量（数据-逻辑错误）。

## 亮点与洞察
- 把"多语言绘图脚本"当作"同一图表的多视图"是一个新颖且自然的视角，相当于把跨语言代码 alignment 这种 NLP/code 经典思路搬到 chart-to-code 场景。
- 子空间适配器 + 路由的设计比 MoE-MLP 更参数经济——共享 MLP 学共性 + 子空间池学差异，强制 $\mathbf{A}$ 冻结让适配容量不"内卷"重复视觉特征。
- 元数据-模板管线给了一条"高质量多语言数据"的可重复路径，176K 已经是同类数据中最大的，且语言数量也最丰富。
- 端到端 6.7B 模型直接挑战 Claude-Sonnet-4，多语言绘图领域开源闭源差距大幅缩小。

## 局限与展望
- 模型规模上限到 6.7B，更大 LLM 后端（如 30B+）的潜力未探索。
- SigLIP 输入分辨率 384×384 是瓶颈，信息密集图（多子图、复杂热图）易丢细节，作者已点名要换高分辨率视觉适配器。
- 模板池虽 202 个但仍有限，新颖图表类型（如动态可交互图）可能模板未覆盖。
- 仅做了三种语言；扩展到 D3.js / Vega-Lite / Mermaid 等需要新模板和元数据 schema。

## 相关工作与启发
- **vs ChartCoder-7B (Zhao et al., 2025)**: ChartCoder 是 Python 专家、在 R 上 0 可执行；本文用对齐数据 + 路由实现真正的多语言通才，Python 性能可比但泛化大胜。
- **vs ChartMoE (Xu et al., 2025)**: ChartMoE 用 sparsely-gated MoE 投影器，参数大幅膨胀；本文低秩子空间适配器更紧凑且效果更好。
- **vs DaTikZ / AutomaTikZ (Belouadi et al., 2024)**: 专注 TikZ 单语言；本文把 TikZ 当作三语言之一，并通过对齐让其他语言互相 boost。

## 评分
- 新颖性: ⭐⭐⭐⭐ 多视图对齐 + 语言条件子空间路由是该领域少见的两点结合
- 实验充分度: ⭐⭐⭐⭐⭐ 三 benchmark × 三语言 × 14 baseline + 详细消融
- 写作质量: ⭐⭐⭐⭐ 动机和方法叙事清晰，公式简洁
- 价值: ⭐⭐⭐⭐ 数据集 + 模型 + 范式三方面贡献，开源也具实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi_agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)

</div>

<!-- RELATED:END -->
