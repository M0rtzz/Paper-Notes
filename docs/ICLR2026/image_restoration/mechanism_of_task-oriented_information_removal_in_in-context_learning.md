---
title: >-
  [论文解读] Mechanism of Task-oriented Information Removal in In-context Learning
description: >-
  [ICLR 2026][图像恢复][in-context learning] 从"信息移除"的新视角解释 In-context Learning（ICL）的内部机制：发现 LM 在零样本时将查询编码为包含所有可能任务信息的"非选择性表征"（导致随机输出）…
tags:
  - "ICLR 2026"
  - "图像恢复"
  - "in-context learning"
  - "information removal"
  - "去噪"
  - "mechanistic interpretability"
  - "low-rank filter"
---

# Mechanism of Task-oriented Information Removal in In-context Learning

**会议**: ICLR 2026  
**arXiv**: [2509.21012](https://arxiv.org/abs/2509.21012)  
**代码**: 无  
**领域**: 图像复原  
**关键词**: in-context learning, information removal, denoising heads, mechanistic interpretability, low-rank filter

## 一句话总结
从"信息移除"的新视角解释 In-context Learning（ICL）的内部机制：发现 LM 在零样本时将查询编码为包含所有可能任务信息的"非选择性表征"（导致随机输出），而 few-shot ICL 的核心作用是模拟一种"任务导向的信息移除"过程——通过识别出的"Denoising Heads"（去噪注意力头）从纠缠的表征中选择性移除冗余任务信息，引导模型聚焦目标任务。消融实验证实阻断去噪头后 ICL 准确率显著下降。

## 研究背景与动机

**领域现状**：In-context Learning（ICL）是大语言模型的标志性能力——无需微调，仅通过在 prompt 中提供少量示例（demonstrations）就能让模型执行新任务。尽管 ICL 已被广泛应用，但其内部"如何工作"的机制仍不清晰。

**现有痛点**：
   - **现有理论视角有限**：已有解释包括"ICL 是隐式梯度下降"、"ICL 学习贝叶斯推断"、"induction heads 做复制粘贴"等，但这些解释要么在简化模型上验证、要么只覆盖特定类型任务，缺乏统一和深入的理解
   - **零样本为何失败不清楚**：在没有 demonstrations 的零样本场景下，LM 对许多任务的准确率接近零。模型具备知识但输出随机——为什么？
   - **demonstrations 到底做了什么**：few-shot 的 demonstrations 如何改变模型内部表征，引导模型从"什么任务都想做"变成"只做目标任务"？机制不明

**核心矛盾**：LM 的预训练使其拥有处理各种任务的能力，但这些能力以"纠缠"的形式存在于隐藏状态中。零样本时，查询的隐藏状态包含了所有可能任务的信息，导致输出混乱——ICL 的 demonstrations 需要做的不是"添加信息"，而是"移除干扰"。

**本文目标**：从"信息移除"这个全新视角，解释 ICL 的核心机制——demonstrations 如何帮助模型从纠缠的表征中去除冗余任务信息，聚焦目标任务。

**切入角度**：
   - 首先证明零样本时 LM 的隐藏状态是"非选择性"的（包含所有任务信息）
   - 然后用低秩滤波器人工模拟信息移除，验证移除冗余信息确实能提升任务准确率
   - 接着测量 few-shot ICL 的隐藏状态，发现其效果等价于任务导向的信息移除
   - 最后识别执行移除操作的关键注意力头（Denoising Heads）

**核心 idea**：ICL 的机制不是"利用 demonstrations 学习新知识"，而是"利用 demonstrations 从纠缠表征中移除冗余信息"——去噪而非学习。

## 方法详解

### 整体框架
本文是一项机制分析（mechanistic analysis）工作，而非提出新模型。分析框架包含四个递进的发现：

**Discovery 1**：零样本时的非选择性表征  
**Discovery 2**：低秩滤波器可以模拟任务导向的信息移除  
**Discovery 3**：Few-shot ICL 天然模拟信息移除过程  
**Discovery 4**：关键注意力头（Denoising Heads）是信息移除的执行者

### 关键设计

1. **非选择性表征的发现与度量**：

    - 功能：分析 LM 在零样本场景下查询 token 的隐藏状态，证明这些表征包含了所有可能任务的信息
    - 核心思路：设计精确的度量指标来衡量隐藏状态中不同任务信息的存在程度。例如，对于情感分类查询，检查隐藏状态是否同时包含"情感分类"、"主题分类"、"翻译"等多个任务的激活信号
    - 实验发现：零样本时，隐藏状态确实是"非选择性"的——不同任务的信息混杂在一起，模型无法确定应该执行哪个任务，因此输出近乎随机（准确率接近零）
    - 设计动机：这一发现解释了零样本失败的根本原因——不是"模型不会"，而是"模型什么都想做"

2. **低秩滤波器实验**：

    - 功能：设计一个低秩投影操作 $P$，对隐藏状态 $h$ 进行滤波 $h' = P \cdot h$，选择性移除特定任务维度的信息
    - 核心思路：通过 SVD 分解隐藏状态矩阵，识别与不同任务关联的主成分方向，然后投影到任务相关的低秩子空间——等价于移除了该子空间正交方向上的信息
    - 实验发现：对零样本的隐藏状态施加低秩滤波后，模型能够"聚焦"目标任务，准确率显著提升——验证了"信息移除 = 任务导向"的假设
    - 设计动机：低秩滤波器提供了一个可控的信息移除工具，用来验证"如果我们人工移除冗余信息，效果是否等价于 ICL"

3. **Few-shot ICL 的隐藏状态分析**：

    - 功能：对比 few-shot 和零样本的隐藏状态，证明 demonstrations 的作用等价于任务导向的信息移除
    - 核心思路：用精心设计的指标度量 few-shot 隐藏状态的"选择性"程度——测量冗余任务信息是否被压缩、目标任务信息是否被增强
    - 实验发现：随着 demonstrations 数量增加，隐藏状态逐渐变得"选择性"——冗余信息被抑制、目标任务信息占主导。这个过程在定量上吻合低秩滤波器实验的效果
    - 设计动机：直接比较自然 ICL 和人工滤波的效果，证明 ICL 在功能上等价于信息移除

4. **Denoising Heads 的识别与验证**：

    - 功能：在 Transformer 的多头注意力中定位执行信息移除操作的关键注意力头（命名为"Denoising Heads"）
    - 核心思路：
        - 通过分析每个注意力头对隐藏状态"选择性"指标的贡献，筛选出对信息移除贡献最大的头
        - 这些头的注意力模式显示：它们主要关注 demonstrations 中与目标任务相关的部分（如标签 token），并用这些信息来调制查询的隐藏状态
    - 验证（消融实验）：
        - 在推理时"阻断"Denoising Heads（将输出置零或用原始隐藏状态替代） → ICL 准确率显著下降
        - 特别是在"正确标签不在 demonstrations 中"的极端场景下（flip label 设置），阻断 Denoising Heads 后准确率退化更严重——因为此时信息移除更为关键
    - 设计动机：识别执行信息移除的具体组件，将机制从"黑箱功能描述"推进到"组件级因果验证"

### 分析方法论

本文使用的关键分析工具包括：
- **隐藏状态探测（Probing）**：训练线性探测器检测隐藏状态中特定任务信息的存在
- **因果消融（Causal Ablation）**：通过干预特定组件验证其因果作用
- **低秩投影**：SVD 分解 + 低秩近似作为信息移除工具
- **注意力头分析**：逐头定量评估对信息移除的贡献
- **精心设计的对照实验**：如 flipped labels、随机 labels 等，区分不同情形下 ICL 的行为差异

## 实验关键数据

### 实验设置
- **模型**：在多个语言模型上验证（GPT-2 系列、LLaMA 等不同规模）
- **任务**：文本分类（情感分析、主题分类等）——选择这类任务是因为它们有清晰的标签空间，方便度量"任务信息"
- **规模**：87 页论文、90 张图、7 个表——极其详尽的实验

### 主实验

**发现1：非选择性表征**

| 场景 | 准确率 | 隐藏状态选择性 | 说明 |
|------|--------|-------------|------|
| 零样本 | ~0% | 低（多任务信息混杂） | 模型"什么都想做" |
| 人工低秩滤波 | 显著提升 | 高（目标任务信息占优） | 移除冗余信息等价于指导任务 |
| Few-shot ICL (4-shot) | 高 | 高 | demonstrations 自然实现了信息移除 |

**发现2：ICL ≈ 信息移除**
- 低秩滤波器的效果和 few-shot ICL 的效果在定量指标上高度吻合
- 两者都使隐藏状态变得"更选择性"——冗余任务信息被压缩

**发现3：Denoising Heads 消融**

| 配置 | ICL 准确率变化 | 说明 |
|------|---------------|------|
| 正常 ICL | 基线 | — |
| 阻断 Denoising Heads | 显著下降（↓15-30%） | 信息移除被阻断 |
| 阻断非 Denoising Heads | 轻微影响 | 非关键头不影响 ICL |
| Flipped Labels + 阻断 Denoising Heads | 退化最严重 | 无正确标签时信息移除更关键 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不同 demonstration 数量 | 信息移除程度单调增加 | 更多示例 = 更强的去噪 |
| 不同模型规模 | 更大模型有更多 Denoising Heads | 规模↑ → 信息移除能力↑ |
| 不同任务类型 | 信息移除机制一致存在 | 在情感、主题等多种任务上验证 |
| 标签空间大小 | 标签越多越需要信息移除 | 验证：更多可能的任务 = 更多需要移除的冗余信息 |

### 关键发现
- **ICL 不是在"学习新技能"，而是在"过滤干扰"**：这是最核心的发现。LM 已经具备各种任务能力，demonstrations 只是帮助模型"聚焦"到正确的任务
- **Denoising Heads 数量有限但关键**：只有少量注意力头负责信息移除，但阻断它们对 ICL 影响巨大
- **信息移除在 flipped label 场景更关键**：当 demonstrations 的标签被翻转（故意给错标签）时，模型仍能部分工作——说明 demonstrations 的主要作用不是提供正确标签，而是指示"应该做什么任务"（通过移除其他任务信息）
- **不同模型的 Denoising Heads 位置不同但功能一致**：验证了机制的普适性

## 亮点与洞察

- **全新的 ICL 解释视角**：相比"ICL = 隐式梯度下降"或"ICL = 贝叶斯推断"，"ICL = 信息移除"更直观、更具操作性——它告诉我们 demonstrations 的功能不是"教新东西"而是"告诉模型该做什么"
- **非选择性表征的发现**：首次系统性地展示零样本时的隐藏状态包含所有任务信息。这解释了一个长期困惑：为什么具备知识的模型在零样本时输出随机
- **Denoising Heads 的概念**：将信息移除操作定位到具体的注意力头，是 mechanistic interpretability 的重要进展——从"功能描述"到"组件定位"
- **低秩滤波器作为分析工具**：提供了一个优雅的实验框架来人工模拟信息移除，为 ICL 机制研究提供了新的方法论
- **论文的深度和彻底性**：87 页、90 张图、7 张表——作者对每一个发现都进行了多角度验证，极其严谨

## 局限与展望

- **主要在分类任务上验证**：信息移除机制是否适用于生成式任务（如对话、摘要、翻译）尚不清楚。生成任务的"任务信息"更难定义和度量
- **仅使用线性探测和低秩投影**：信息移除可能涉及非线性变换，低秩线性近似可能只捕获了部分机制
- **模型规模限制**：由于分析需要对隐藏状态进行详细探测，实验主要在中等规模模型（GPT-2 系列、较小的 LLaMA）上验证，对超大模型（100B+）的适用性未知
- **Denoising Heads 的形成机制**：论文发现了这些头的存在，但未解释它们是如何在预训练中形成的——这需要对训练动态的进一步研究
- **与其他 ICL 理论的统一**：信息移除视角与"隐式梯度下降"、"贝叶斯推断"等视角之间是互补还是矛盾？缺乏显式的理论统一

## 相关工作与启发

- **Induction Heads**（Olsson et al.）：识别出执行"复制-粘贴"操作的注意力头。Denoising Heads 是另一类功能性注意力头，执行"信息过滤"操作
- **Task Vectors**：发现模型内部存在表征任务方向的向量。信息移除可以理解为将隐藏状态投影到正确的任务向量方向
- **ICL 的贝叶斯视角**（Xie et al. 2022）：ICL 做隐式贝叶斯推断——选择最可能的任务。信息移除可以看作贝叶斯推断的"注意力头级别"实现
- **可解释性研究**（Mechanistic Interpretability）：本文遵循"定位功能性组件→因果验证→消融实验"的标准范式
- **启发**：信息移除视角可能对 prompt engineering 有实用指导——好的 prompt 应该帮助模型"过滤掉不相关的任务解读"，而不只是"提供任务信息"

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images](../../CVPR2025/image_restoration/tokenize_image_patches_global_context_fusion_for_effective_haze_removal_in_large.md)
- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)
- [\[ICLR 2026\] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)
- [\[CVPR 2026\] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](../../CVPR2026/image_restoration/shadow_removal_cascaded_refinement.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](../../ICCV2025/image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)

</div>

<!-- RELATED:END -->
