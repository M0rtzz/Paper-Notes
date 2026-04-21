---
title: >-
  [论文解读] Attribution-Guided Decoding
description: >-
  [ICLR 2026][机器人][attribution method] 提出AGD解码策略，在每步生成时从高概率候选token中选择对用户指定"兴趣区域"（ROI）归因得分最高的token，将归因方法从被动分析工具转变为主动生成引导工具，在指令遵循和事实性任务上均取得显著提升。
tags:
  - ICLR 2026
  - 机器人
  - attribution method
  - LRP
  - instruction following
  - factuality
  - entropy-gating
  - controlled decoding
---

# Attribution-Guided Decoding

**会议**: ICLR 2026  
**arXiv**: [2509.26307](https://arxiv.org/abs/2509.26307)  
**代码**: [GitHub](https://github.com/piotr-komorowski/AGD)  
**领域**: LLM解码 / 可解释性  
**关键词**: attribution method, LRP, instruction following, factuality, entropy-gating, controlled decoding

## 一句话总结

提出AGD解码策略，在每步生成时从高概率候选token中选择对用户指定"兴趣区域"（ROI）归因得分最高的token，将归因方法从被动分析工具转变为主动生成引导工具，在指令遵循和事实性任务上均取得显著提升。

## 研究背景与动机

**领域现状**：LLM解码策略是控制生成质量的关键一环。标准解码方法（贪心、top-k、nucleus采样）控制的是输出的随机性，但无法直接引导生成的语义属性。为了增强指令遵循和事实准确性，研究者提出了两类方法：(1) **干预型（Interventionist）方法**，如激活工程（activation steering）直接修改模型内部表示，对比解码（CAD、DoLA）修改输出logits；(2) **后处理方法**，对输出进行过滤或重排序。

**现有痛点**：干预型方法直接修改模型的前向传播或logit分布，会将模型推入分布外状态，导致困惑度升高、重复输出、文本质量下降。这造成了一个不理想的权衡——用户必须在"更好的控制"和"更高的生成质量"之间做选择。例如，activation steering在提升指令遵循的同时会严重损害输出的流畅性和连贯性。

**核心矛盾**：如何在**不修改模型内部状态或输出分布**的前提下，引导生成朝向期望的行为（如遵循指令、减少幻觉）？需要一种既能有效控制又不破坏生成质量的机制。

**本文目标** (1) 提出一种不干预模型前向传播的解码引导方法；(2) 使该方法灵活适用于多种任务（指令遵循、事实性、上下文检索）；(3) 减少引导带来的计算开销和质量损失。

**切入角度**：作者提出将**归因方法（attribution methods）**从事后解释工具转变为前向引导工具。归因方法可以量化每个候选token对输入特定部分的"依赖程度"。如果候选token A对用户指令的归因分数高于候选token B，说明A更"听从"指令——那么选择A即可实现指令引导,而不需要修改模型的任何内部状态。

**核心 idea**：将解码过程重新定义为"在候选token中寻找对指定兴趣区域归因最大的token"，用归因方法的选择机制替代概率最大化机制。

## 方法详解

### 整体框架

AGD的解码流程分为四步：(1) 定义**兴趣区域（ROI）** $R$——可以是输入中的指令部分、特定知识处理注意力头、或上下文文档的token嵌入；(2) 标准前向传播生成概率分布，选出top-k高概率候选token构成候选集 $\mathcal{C}_t$，并过滤掉概率低于阈值 $\pi_{\min}$ 的token；(3) 对每个候选token $c \in \mathcal{C}_t$ 用归因方法（LRP）回溯计算其对ROI中每个组件的归因分数，求和得到总归因分数 $S(c, R)$；(4) 选择归因分数最高的候选token作为本步生成结果。整个过程不修改模型的前向传播、不改变logit值——是"选择型"而非"干预型"方法。

### 关键设计

1. **基于LRP的归因评分机制**:

    - 功能：量化每个候选token对指定ROI的依赖程度，为token选择提供有原则的依据
    - 核心思路：对候选token $c$ 的pre-softmax logit使用层逐传播（LRP）方法反向传播，得到模型各组件 $\omega \in \Omega$ 的归因分数 $r_\omega$。ROI内组件的归因分数求和得总分 $S(c, R) = \sum_{\omega \in R} r_\omega$。分数越高表示token $c$ 越依赖于ROI中的信息。选择LRP（特别是AttnLRP变体）是因为它在Transformer中处理self-attention和layer normalization等非线性组件时比简单梯度方法（I×G）更稳定忠实，且计算效率与梯度方法相当——仅需一次反向传播。
    - 设计动机：归因方法天然提供了"一个输出是由输入的哪部分决定的"这一信息，且产生有符号的分数（正/负归因），正归因帮助选择依赖指令的token，负归因帮助避免违反禁止约束的token。这是纯概率方法无法提供的丰富信号。

2. **灵活的ROI定义**:

    - 功能：通过改变ROI的定义，使AGD适用于指令遵循、闭卷事实性、开卷上下文检索等多种任务
    - 核心思路：对于**指令遵循**，ROI $R_I$ 定义为输入中指令部分（如system prompt）对应的token嵌入集合。对于**闭卷事实性**，ROI $R_P$ 定义为预识别的参数知识（parametric knowledge）注意力头集合。对于**开卷检索**，ROI可以定义为上下文文档token嵌入 $R_C$，或上下文检索（in-context retrieval）注意力头 $R_{IC}$。所有任务使用相同的AGD算法框架，只需切换ROI即可。
    - 设计动机：将ROI抽象为模型可归因组件的任意子集，使方法从特定任务的解决方案上升为通用框架。这种"归因组件 ↔ 控制目标"的对应关系使AGD具有极强的灵活性，可以扩展到任何可以用归因来量化的控制目标。

3. **基于熵的自适应门控（Entropy-Gating）**:

    - 功能：仅在模型不确定时施加归因引导，降低计算开销并保护输出质量
    - 核心思路：计算每步输出分布的Shannon熵 $H(\mathbf{p}_t)$。当 $H(\mathbf{p}_t) < \tau$ 时（模型自信），直接使用贪心解码；当 $H(\mathbf{p}_t) \geq \tau$ 时（模型不确定），激活AGD。阈值 $\tau$ 设为IHEval上token级熵的第80百分位数（$\tau = 1.734$）。这样AGD仅在模型犹豫不决的"关键分叉点"施加引导——正是这些点决定了生成轨迹的走向。
    - 设计动机：每步都做归因计算（需要多次反向传播）计算开销大。更重要的是，当模型已经很确定时强制引导反而会破坏已有的高质量输出——相当于"给已经知道答案的学生强加提示反而搞混了"。自适应门控在效果和效率之间取得了优秀的平衡。

### 损失函数 / 训练策略

AGD是纯推理时方法，**无需训练或微调**。固定参数：$k=5$（候选集大小）、$\pi_{\min}=0.05$（最小概率阈值）、$\tau=1.734$（熵门控阈值）。所有实验使用相同超参数，无需针对不同模型或任务调整。

## 实验关键数据

### 指令遵循实验

在3个模型（Llama 3.1 8B、Qwen 2.5 7B、Gemma 3 4B）上评测IHEval和SysBench两个基准。

| 模型 | 方法 | PLA (IHEval) | QS | PLA*QS | SSR (SysBench) |
|------|------|-------------|-----|--------|----------------|
| Llama 3.1 | Greedy | 66.0 | 81.3 | 53.7 | 26.0 |
| Llama 3.1 | CAD | 73.9 | 72.6 | 53.7 | 32.3 |
| Llama 3.1 | AGD_LRP | **79.1** | 73.2 | **57.9** | 32.2 |
| Llama 3.1 | AGD_LRP_e | 74.5 | 76.4 | 56.9 | **33.9** |
| Qwen 2.5 | Greedy | 63.2 | 74.1 | 46.8 | 27.1 |
| Qwen 2.5 | AGD_LRP_e | **70.4** | 70.6 | **49.7** | **29.9** |
| Gemma 3 | Greedy | 84.7 | 82.3 | 69.7 | 33.3 |
| Gemma 3 | AGD_LRP_e | **86.7** | 81.4 | 70.6 | **36.0** |

### 事实性与上下文检索实验 (Llama 3.1 8B)

| 设置 | 方法 | TriviaQA | NQ | HotPotQA |
|------|------|----------|-----|----------|
| 闭卷 | Greedy | 81.4 | 63.6 | 34.6 |
| 闭卷 | DoLA | 81.2 | 63.8 | 34.3 |
| 闭卷 | AGD_LRP_h | **82.4** | 63.0 | **39.6** |
| 开卷 | Greedy | 89.4 | 83.5 | 81.3 |
| 开卷 | CAD | 87.9 | 84.6 | 83.7 |
| 开卷 | AGD_LRP_c | **91.4** | **87.9** | **87.9** |

### 关键发现

- **LRP远优于I×G**：使用LRP归因的AGD在指令遵循上一致性地大幅超越使用I×G的版本。LRP在Transformer中处理self-attention的规则（AttnLRP）提供了更忠实的归因分数，这直接转化为更好的引导效果。
- **负归因信号是关键**：对于"禁止包含某些词"类型的约束，违禁候选token会在指令部分产生**负归因分数**，帮助模型主动避开这些token。这是AGD优于简单概率操控方法的独特优势。
- **熵门控显著改善质量-遵循权衡**：在Llama 3.1上，完整AGD的PLA为79.1但QS降至73.2；熵门控版本PLA降至74.5但QS恢复至76.4，综合指标PLA*QS仅差1%。在SysBench多轮对话中，熵门控版本的SSR（33.9）甚至超过完整AGD（32.2），说明只在关键点引导反而效果更好。
- **开卷QA提升显著**：在HotPotQA（含80%干扰文档）上，AGD比贪心解码提升6.6个点，说明归因机制能帮助模型在噪声上下文中锁定相关段落。

## 亮点与洞察

- **"从解释到引导"的范式转变**：将归因方法从事后分析工具转变为生成过程中的主动引导信号，这是一个深刻的视角转换。归因方法几十年来一直用于"解释模型为什么这样做"，这篇论文首次将其用于"决定模型应该怎么做"。
- **选择型 vs 干预型**：AGD不修改模型的前向传播或logit分布——它只是在模型已经认为可行的候选中做"更明智的选择"。这保证了选出的token始终处于模型的正常分布范围内，从根本上避免了activation steering等方法导致的质量退化问题。
- **ROI的统一抽象**：通过将控制目标统一抽象为"可归因组件的子集"，AGD成为一个通用框架。只要目标可以表示为输入token或注意力头的集合，就可以用AGD引导——从指令遵循到事实性到上下文检索，切换只需改ROI定义。

## 局限与展望

- **候选集限制**：AGD作为选择机制，无法生成候选集中不存在的token。如果"正确答案"不在模型的top-k候选中，AGD无法发挥作用。
- **计算开销**：每个候选token都需要一次反向传播来计算归因（虽然熵门控缓解了这一问题），对于长文本生成仍然有可观的额外延迟。
- **ROI定义依赖先验知识**：指令遵循的ROI相对自然（system prompt），但知识头、上下文检索头的识别需要预先分析，可迁移性受限。
- **仅在≤8B模型上验证**：实验模型最大为8B参数，对更大规模模型的扩展性（尤其是归因计算的内存需求）未知。

## 相关工作与启发

- **vs CAD (Context-aware Decoding)**: CAD通过对比有/无指令时的logit来修改分布，属于干预型方法。AGD不修改logit，只在概率高的候选中选择归因最高的。AGD在指令遵循上超越CAD（Llama 3.1: 79.1 vs 73.9 PLA），说明归因信号比对比logit差值更有效。
- **vs Activation Steering**: Steering直接修改内部表示，虽然控制力强但质量退化严重。AGD的"不干预"设计从根本上避免了这个问题。
- **vs DoLA**: DoLA用层间logit对比减少幻觉，也是干预型方法。AGD在闭卷HotPotQA上显著超越DoLA（39.6 vs 34.3），归因信号比层间对比更精确地捕获了知识存储位置。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将归因方法从解释工具转变为生成引导工具，启发性极强
- 实验充分度: ⭐⭐⭐⭐ 覆盖3种任务、3个模型、多个基准，消融和案例分析充分
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，图示精美，方法动机阐述到位
- 价值: ⭐⭐⭐⭐⭐ 通用性强的免训练解码框架，对LLM可控生成方向有深远影响

<!-- RELATED:START -->

## 相关论文

- [Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)
- [A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning](../../NeurIPS2025/robotics/a_snapshot_of_influence_a_local_data_attribution_framework_f.md)
- [UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](../../AAAI2026/robotics/urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)
- [Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](../../AAAI2026/robotics/affordance-guided_coarse-to-fine_exploration_for_base_placem.md)
- [Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)

<!-- RELATED:END -->
