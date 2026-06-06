---
title: >-
  [论文解读] Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding
description: >-
  [AAAI 2026][预训练][过程化视频理解] 提出 Task-Step-State（TSS）三层语义框架，在传统的任务-步骤层次中引入"状态"作为视觉锚定层，并设计渐进式预训练策略（Task→Step→State→Step→Task）逐步展开 TSS 层次…
tags:
  - "AAAI 2026"
  - "预训练"
  - "过程化视频理解"
  - "状态锚定"
  - "层次化学习"
  - "渐进式预训练"
  - "视频表征"
---

# Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding

**会议**: AAAI 2026  
**arXiv**: [2511.20073](https://arxiv.org/abs/2511.20073)  
**代码**: [https://github.com/zhao-jinghan/TSS-unfolding](https://github.com/zhao-jinghan/TSS-unfolding)  
**领域**: LLM预训练  
**关键词**: 过程化视频理解, 状态锚定, 层次化学习, 渐进式预训练, 视频表征

## 一句话总结
提出 Task-Step-State（TSS）三层语义框架，在传统的任务-步骤层次中引入"状态"作为视觉锚定层，并设计渐进式预训练策略（Task→Step→State→Step→Task）逐步展开 TSS 层次，在 COIN 和 CrossTask 数据集上的任务识别、步骤识别和步骤预测任务上全面超越 SOTA。

## 研究背景与动机

理解和执行面向目标的过程化活动（如教学视频中的操作步骤）是智能体的核心能力。现有方法通过将视觉内容与任务/步骤层级的文本描述对齐来学习过程化视频表征——例如用 wikiHow 中的任务名称（"制作橙汁"）和步骤描述（"切橙子"）作为监督信号。

然而，这种两层方法存在**关键限制**：
- 任务和步骤描述高度抽象，难以与视觉数据中具体可观察的细节形成鲁棒对齐
- "切橙子"这样的抽象指令与实际展示该动作的原始像素之间存在**巨大的语义鸿沟**
- 在训练视觉-文本对齐时，这种鸿沟使模型难以将抽象过程锚定在实际可见的内容上

核心矛盾：抽象文本→具体视觉之间的语义鸿沟。作者的洞察：引入"**状态**"（state）——对象配置的文本快照（如"橙子不再完整，果肉裸露"）——作为视觉锚定的语义层，将抽象过程锚定到模型实际可见的内容上。

从逻辑角度看，状态构成了任何过程化任务的骨架：任务是从初始状态到最终状态的宏观转换，步骤是驱动中间状态转换的动作。这形成了 Task-Step-State 三层框架。

## 方法详解

### 整体框架
方法包含两个核心贡献：
1. **TSS 框架构建**：用 LLM 为每个步骤生成前/中/后三种状态描述，形成三层语义结构
2. **渐进式预训练策略**：按 Task→Step→State→Step→Task 的 U 型路径逐阶段训练

### 关键设计

1. **TSS 框架构建（Task-Step-State Framework）**:

    - 功能：在传统的任务-步骤两层结构中增加"状态"层，为每个步骤 $s_{i,j}$ 关联三种状态
        - **前状态 $c_{i,j}^b$**：动作开始前的物体配置
        - **中间状态 $c_{i,j}^m$**：动作进行中的状态
        - **后状态 $c_{i,j}^a$**：动作完成后的状态
    - 核心思路：用 GPT-4o-mini 和 CoT 提示策略自动从 wikiHow 的任务+步骤描述生成状态文本，形成丰富的三层知识库
    - 规模：1,053 个任务、10,588 个步骤、31,764 个状态描述
    - 设计动机：状态描述了可观察的物体配置（如形状、属性、空间关系），与视觉内容的语义距离远小于抽象的步骤描述

2. **伪标签生成**:

    - 功能：通过视频-文本特征对齐生成训练监督信号
    - 核心思路：
        - 用冻结的 S3D 视觉编码器和 Sentence-BERT 文本编码器分别提取特征
        - 对文本特征进行聚合聚类（步骤节点 10,038 个，状态节点约 9,000-10,000 个）
        - 通过余弦相似度匹配视频片段到文本节点，选 top-3 节点作为多分类伪标签
    - 五种伪标签：TaskVNM、StepVNM、StateVNM、StepNRL（节点关系学习）、StepTCL（任务上下文学习）
    - 设计动机：利用预训练编码器的大规模弱监督，无需人工时序标注

3. **渐进式预训练策略**:

    - 功能：按特定路径分阶段训练，每阶段聚焦一个语义层
    - 最佳路径：**Task→Step→State→Step→Task**（Path-5/6）
    - 模型架构：冻结的 S3D 视觉编码器 + 可训练的瓶颈适配器（512→128→512）+ 随机初始化的任务头
    - 知识传递机制：每阶段完成后保留适配器权重传递到下一阶段，丢弃任务头重新初始化
    - 设计动机：先自上而下分析（Task→Step→State），再自下而上综合（State→Step→Task），形成完整的分析-综合循环。关键发现：
        - State→Task 直接跳跃效果不好（Path-4），证实步骤层是必要的中间桥梁
        - 联合训练（Mix_Train）不如渐进式训练，因为无法捕捉层级间的因果关系
        - 最后的 Step→Task 回溯（Path-6 vs Path-5）边际收益很小

### 损失函数 / 训练策略
- BCEWithLogitsLoss 多分类损失
- Adam 优化器，lr=1e-4，batch_size=256
- 训练数据：410 万视频片段（来自 HowTo100M 的 85K 视频子集）
- 每个 epoch 约 90 秒（Step/State 阶段）或 30 秒（Task 阶段），共 1500 epochs
- 8×H200 GPU 训练

## 实验关键数据

### 主实验

**与 SOTA 对比（Path-5，COIN 数据集）**:

| 方法 | TR(MLP) | SR(MLP) | SF(MLP) | TR(Trans) | SR(Trans) | SF(Trans) |
|------|---------|---------|---------|-----------|-----------|-----------|
| No pretrain | 2.09 | 1.37 | 0.84 | 78.31 | 39.23 | 35.43 |
| Paprika (SOTA) | 81.54 | 42.39 | 34.10 | 82.83 | 41.19 | 38.93 |
| **Ours (Path-5)** | **83.78** | **44.54** | **38.07** | **83.11** | **42.42** | **40.40** |
| 提升 vs Paprika | +2.24 | +2.15 | +3.97 | +0.28 | +1.23 | +1.47 |

**CrossTask 数据集**:

| 方法 | TR(MLP) | SR(MLP) | SF(MLP) | TR(Trans) | SR(Trans) | SF(Trans) |
|------|---------|---------|---------|-----------|-----------|-----------|
| Paprika | 89.65 | 56.21 | 55.77 | 90.27 | 55.57 | 55.67 |
| **Ours (Path-5)** | 89.44 | **57.92** | **57.13** | 89.44 | **57.08** | **56.50** |

### 消融实验

| 配置 | COIN TR(MLP) | COIN SR(MLP) | COIN SF(MLP) | 说明 |
|------|-------------|-------------|-------------|------|
| Path-1 (Task only) | 73.31 | 34.18 | 23.67 | 仅任务级预训练 |
| Path-2 (Task→Step) | 82.45 | 43.06 | 36.04 | 加入步骤层有提升 |
| Path-3 (→State) | 80.73 | 41.28 | 34.35 | 直接到 State 反而下降 |
| Path-4 (→State→Task) | 77.74 | 37.51 | 24.84 | State→Task 跨越太大 |
| **Path-5 (→State→Step)** | **83.78** | **44.54** | **38.07** | **U 型回溯最优** |
| Path-6 (→Step→Task) | 83.30 | 44.04 | 36.94 | 额外 Task 回溯边际收益小 |
| Mix_Train (联合) | 77.74 | 38.43 | 29.79 | 联合训练不如渐进式 |
| Fusion-AvgPool | 83.11 | 44.35 | 36.88 | 特征融合也有效但弱于渐进 |

### 关键发现
- **状态层是关键驱动力**：没有状态（Path-2）的最好结果是 43.06 SR，加入状态后的最优结果是 **44.54** SR
- **渐进式训练优于联合训练**：Mix_Train 的 SR（38.43）远低于 Path-5（44.54），说明层级间的因果关系很重要
- **步骤层是必要的中间桥梁**：Path-4（State→Task 直接跳跃）性能骤降，Path-5（State→Step 中间过渡）效果最佳
- **U 型路径模拟分析-综合过程**：先自上而下分析到最具体的状态，再自下而上综合回到抽象步骤
- 简单 MLP 下游头的提升更显著（+3.97 SF），说明预训练表征质量确实更高
- 特征融合（AvgPool/Concat）也有效，验证了状态信息的互补性

## 亮点与洞察
- **"状态"作为视觉锚定层的概念创新**：将抽象的过程化知识通过可观察的物体配置锚定到视觉数据
- **渐进式预训练路径的系统探索**：不是随机选择训练顺序，而是通过消融实验系统验证了 6 种路径
- **低成本设计**：冻结视觉编码器，仅训练瓶颈适配器（512→128→512），参数效率极高
- LLM 生成状态描述的方法实用且可扩展
- Path-4 vs Path-5 的对比精确揭示了语义鸿沟的存在和步骤层的桥梁作用

## 局限与展望
- 依赖 S3D 编码器和 wikiHow 语料，可能限制在更广泛视频类型上的适用性
- LLM 生成的状态描述可能不够准确，尤其对于抽象或复杂操作
- 仅在 COIN 和 CrossTask 上评估，这两个数据集规模相对有限
- 固定 9.6 秒的视频分段可能不适合所有步骤（有的步骤很短、有的很长）
- 未探索视频到状态的直接视觉预测，仅通过文本-视觉对齐间接利用

## 相关工作与启发
- "状态"概念连接了过程化学习和物体中心视频理解两个方向
- 渐进式/课程学习的思想在体系结构化知识中特别有效
- LLM 作为知识增强工具（生成状态描述）的范式可推广到其他需要中间语义层的任务
- 分析-综合（U 型）学习路径为多层次知识的预训练策略设计提供了新思路
- 适配器微调策略使得大规模预训练的计算成本可控

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — TSS 三层框架和 U 型渐进式训练路径均为原创
- 实验充分度: ⭐⭐⭐⭐⭐ — 6 种路径的系统消融 + 融合策略对比 + SOTA 比较
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑链严密，消融分析深受启发
- 价值: ⭐⭐⭐⭐ — 状态锚定思想有推广价值，但数据集规模限制了影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](../../NeurIPS2025/llm_pretraining/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)
- [\[AAAI 2026\] Learning Time in Static Classifiers](learning_time_in_static_classifiers.md)
- [\[ECCV 2024\] Cross-Domain Learning for Video Anomaly Detection with Limited Supervision](../../ECCV2024/llm_pretraining/cross-domain_learning_for_video_anomaly_detection_with_limited_supervision.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](../../ICLR2026/llm_pretraining/explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence.md)

</div>

<!-- RELATED:END -->
