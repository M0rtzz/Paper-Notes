---
title: >-
  [论文解读] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection
description: >-
  [ACL 2026][AIGC检测][LLM生成文本检测] 提出 RACE（Rhetorical Analysis for Creator-Editor Modeling），利用修辞结构理论(RST)构建逻辑图来建模文本"创作者"的思维架构，同时提取篇章单元级特征捕获"编辑者"的语言风格…
tags:
  - "ACL 2026"
  - "AIGC检测"
  - "LLM生成文本检测"
  - "修辞结构理论"
  - "创作者-编辑者建模"
  - "细粒度分类"
  - "篇章分析"
---

# Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection

**会议**: ACL 2026  
**arXiv**: [2604.04932](https://arxiv.org/abs/2604.04932)  
**代码**: [https://race.yang-li.cn](https://race.yang-li.cn)  
**领域**: AIGC检测  
**关键词**: LLM生成文本检测, 修辞结构理论, 创作者-编辑者建模, 细粒度分类, 篇章分析

## 一句话总结
提出 RACE（Rhetorical Analysis for Creator-Editor Modeling），利用修辞结构理论(RST)构建逻辑图来建模文本"创作者"的思维架构，同时提取篇章单元级特征捕获"编辑者"的语言风格，实现四类细粒度 LLM 生成文本检测（人写/LLM写/LLM润色人文/人改写LLM文）。

## 研究背景与动机

**领域现状**：LLM 生成文本检测主要是二分类（人写 vs LLM 写），近期有些工作引入第三类"混合文本"做三分类设置。

**现有痛点**：即使是三分类也不够精细——"LLM 润色的人文"和"人改写的 LLM 文"在实际监管中有完全不同的政策后果。前者通常被视为合法的写作辅助，后者则是绕过检测的作弊行为。但二者都属于"混合文本"，传统方法用统一特征无法区分。

**核心矛盾**：两种混合类型的创作者-编辑者协作模式截然不同：LLM 润色人文 = 人的逻辑框架 + LLM 的表达风格；人改写 LLM 文 = LLM 的逻辑框架 + 人的表达扰动。统一特征难以捕获这种分离的双重痕迹。

**本文目标**：设计一个能分别建模"创作者"和"编辑者"贡献的检测框架，实现可靠的四分类细粒度检测。

**切入角度**：创作者的身份深植于文本的逻辑组织和论证推进中（人用层级化推理，LLM 倾向平铺直叙），编辑者的影响主要体现在表面语言表达上。RST 正好能分离这两个层面。

**核心 idea**：用 RST 解析文本得到修辞关系树，将其转化为逻辑图来刻画创作者的思维指纹；同时用 EDU 级语义表示捕获编辑者的语言风格。

## 方法详解

### 整体框架
RACE 的处理流程：原始文本 → RST 解析得到 EDU 序列和修辞关系树 → 构建多关系逻辑图（EDU 为叶节点，修辞关系为内部节点）→ 节点特征初始化（descendant span pooling + information bottleneck projection）→ Rhetoric-Guided Message Passing（RGCN）→ Root Pooling 得到全局表示 → 分类。

### 关键设计

1. **双重痕迹提取 (Dual Trace Extraction)**:

    - 功能：从文本中分离创作者和编辑者的痕迹
    - 核心思路：用端到端 RST parser 将文本解析为二叉成分树。叶节点是 EDU 序列（代表编辑者的语言单元），内部节点带有修辞关系标签（Elaboration、Contrast 等），代表创作者的逻辑组织
    - 设计动机：统计分析显示人类创作者在 Attribution 和 Background 关系上显著过表达（引用来源、建立上下文），LLM 创作者在 Elaboration 和 Evaluation 上过表达（平铺信息）。即使经过编辑，这些结构指纹仍然保持——余弦相似度显示同一创作者的文本在修辞关系频率上始终更接近（>0.89）

2. **逻辑感知图初始化 (Logic-Aware Graph Initialization)**:

    - 功能：将 RST 树转化为可学习的多关系图
    - 核心思路：构建图 $\mathcal{G} = (\mathcal{V}_{edu} \cup \mathcal{V}_{rel}, \mathcal{E}, \mathcal{R})$。EDU 节点用 PLM 的 MeanPool 嵌入初始化；关系节点用 Descendant Span Pooling 递归地用所有子孙 EDU 的语义重心初始化。然后通过信息瓶颈投影降维到 $d_{feat}$ 过滤表面噪声
    - 设计动机：直接用关系标签的 one-hot 编码信息太稀疏，用后代 EDU 的语义重心初始化关系节点能注入更丰富的上下文信息

3. **修辞引导消息传递 (Rhetoric-Guided Message Passing)**:

    - 功能：在修辞关系图上学习人/LLM 创作差异的深层表示
    - 核心思路：用 $L$ 层 RGCN，为每种修辞关系学习独立的变换矩阵。为避免参数爆炸，用基分解正则化 $\mathbf{W}_r^{(l)} = \sum_{k=1}^B \alpha_{rk}^{(l)} \mathbf{V}_k^{(l)}$，共享 $B$ 个基矩阵。最终通过根节点池化得到全局文本表示
    - 设计动机：不同修辞关系承载不同的逻辑功能（因果 vs 对比 vs 阐述），需要关系特异的传播规则

### 损失函数 / 训练策略
联合损失 $\mathcal{L}_{total} = \mathcal{L}_{con} + \mathcal{L}_{ce}$：监督对比损失鼓励紧凑的类内聚类 + 交叉熵损失做分类。骨干网络用 RoBERTa-base 只微调最后一层。

## 实验关键数据

### 主实验
在 HART 数据集上的四分类检测。

| 方法 | AUROC (Avg) | TPR@1%FPR |
|------|------------|-----------|
| RoBERTa | ~85 | 68.06 |
| CoCo | ~86 | - |
| DeTeCtive | ~87 | - |
| **RACE** | **~92** | **~80** |

### 消融实验

| 配置 | AUROC | 说明 |
|------|-------|------|
| Full RACE | **~92** | 完整模型 |
| w/o RST graph (仅 EDU) | ~87 | 去掉创作者建模，掉5个点 |
| w/o contrastive loss | ~90 | 特征空间不够紧凑 |
| w/o basis decomposition | ~91 | 稀疏关系过拟合 |

### 关键发现
- RACE 在 12 个基线中 AUROC 最高，且在低误报率(1% FPR)下保持高召回率
- 创作者建模（RST 图）贡献最大——去掉后掉 5 个点
- 修辞关系频率分析验证了核心假设：人写文本的 RST 结构更深更复杂，LLM 文本更扁平
- 同一创作者的文本经过编辑后修辞关系频率余弦相似度仍 >0.89，证明编辑难以改变创作者指纹

## 亮点与洞察
- **Creator-Editor 双角色框架**概念清晰有力——将"谁是最后的操作者"升维到"谁建立了逻辑框架+谁做了表面修饰"
- **RST 作为创作者指纹**的发现非常有说服力——人类的修辞结构更深、更多引用/背景关系，LLM 偏好扁平的阐述/评价结构
- 低误报率指标(TPR@1%FPR)的引入很实际——在学术不端检测等高风险场景减少冤枉比提高召回更重要

## 局限与展望
- 依赖 RST parser 的质量，当前 parser 在某些文本类型上可能不够准确
- 只在 HART 数据集上评估，跨域泛化能力未知
- 四分类设置假设文本只经过一次编辑，多轮人-LLM 交互的场景更复杂
- 随着 LLM 能力提升其修辞结构可能越来越接近人类

## 相关工作与启发
- **vs DetectAIve**: 也尝试了四分类但用统一特征，RACE 用双角色建模更精细
- **vs CoCo**: CoCo 也考虑篇章信息但未用 RST 的层级结构
- **vs LF-Motifs**: 用词频模式检测，无法捕获逻辑组织层面的差异

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Creator-Editor 框架+RST 创作者指纹是原创性很强的设计
- 实验充分度: ⭐⭐⭐⭐ 12 个基线充分，但单数据集限制了泛化性验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析极具说服力
- 价值: ⭐⭐⭐⭐⭐ 首次将四分类细粒度检测做到实用水平

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](../../ACL2025/aigc_detection/haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[CVPR 2026\] Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks](../../CVPR2026/aigc_detection/fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela.md)
- [\[ACL 2026\] GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization](gigacheck_detecting_llm-generated_content_via_object-centric_span_localization.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)

</div>

<!-- RELATED:END -->
