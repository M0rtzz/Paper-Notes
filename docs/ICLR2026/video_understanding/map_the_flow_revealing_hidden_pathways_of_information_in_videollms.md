---
title: >-
  [论文解读] Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs
description: >-
  [ICLR 2026][视频理解][VideoLLM] 首次用机制可解释性工具（Attention Knockout + Logit Lens）系统逆向工程VideoLLM的时序推理过程，揭示出"早中层跨帧交互→中层视频-语言整合→中后层答案生成"的三阶段信息流蓝图，并证明仅保留42%注意力边即可几乎无损保持VideoQA性能。
tags:
  - ICLR 2026
  - 视频理解
  - VideoLLM
  - 信息流分析
  - 机制可解释性
  - 注意力剪枝
  - 时序推理
---

# Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs

**会议**: ICLR 2026  
**arXiv**: [2510.13251](https://arxiv.org/abs/2510.13251)  
**代码**: [项目页面](https://map-the-flow.github.io)  
**领域**: 视频理解 / 可解释AI  
**关键词**: VideoLLM, 信息流分析, 机制可解释性, 注意力剪枝, 时序推理

## 一句话总结

首次用机制可解释性工具（Attention Knockout + Logit Lens）系统逆向工程VideoLLM的时序推理过程，揭示出"早中层跨帧交互→中层视频-语言整合→中后层答案生成"的三阶段信息流蓝图，并证明仅保留42%注意力边即可几乎无损保持VideoQA性能。

## 研究背景与动机

**领域现状**：VideoLLM的标准范式是将视频帧通过视觉编码器patch化为token序列，与文本token拼接后送入因果注意力的LLM进行自回归生成。研究社区的工作主要集中在模型的"外部设计"上——包括扩大视频指令微调数据集、关键帧选择策略、以及视频token压缩方法。但对于模型内部"如何"从扁平化的帧token序列中提取时序信息、"在哪里"完成视频和语言的语义整合，几乎没有系统研究。

**现有痛点**：图像MLLM的可解释性研究（如Neo 2025）已发现了一些结构化行为模式，但这些发现能否推广到视频场景完全未知。视频与图像有根本区别——VideoQA需要从多帧中聚合时间维度的信息。具体而言有三个核心问题：(1) VideoLLM如何从扁平化的帧token序列中编码时间顺序？(2) 时间概念（如"先""后"）怎样从视频token传播到文本token？(3) 模型在哪一层"准备好"生成正确答案？

**核心矛盾**：视频帧经过patchify后变成一维token序列，时间结构被隐式编码在位置中。模型必须通过某种内部机制重新发现和利用这些时序关系，而现有研究只关注提升性能，对这个"黑箱内部发生了什么"一无所知。这阻碍了有针对性的架构改进和推理加速。

**本文目标** 提供一张VideoLLM时序推理的完整蓝图：信息从哪里提取、在哪些层整合、在哪个阶段准备好答案。进而验证这些关键路径是否sufficiently代表了模型的推理过程。

**切入角度**：作者从机制可解释性（mechanistic interpretability）出发，使用因果干预工具（Attention Knockout断开特定注意力边测影响）和探针工具（Logit Lens投影中间层到词汇空间读语义），将VideoLLM的推理过程分解为可检验的阶段。

**核心 idea**：用Attention Knockout和Logit Lens逆向工程VideoLLM的注意力路径，发现时序推理遵循"跨帧交互→时间关键词对齐→答案生成"三阶段pattern，且大部分注意力边是冗余的。

## 方法详解

### 整体框架

本文不是提出新模型，而是一项分析研究。整体pipeline是：(1) 在VideoQA任务上运行VideoLLM → (2) 用Attention Knockout系统性断开不同类型的注意力边（跨帧、视频→问题、问题→最后token等），观测预测概率变化 → (3) 用Logit Lens探针视频token中间层的语义内容 → (4) 综合发现概括出三阶段信息流蓝图 → (5) 验证：仅保留关键路径，剪枝其余注意力边，在benchmark上测性能。

实验基座主要是LLaVA-NeXT-7B经过VideoChat2-IT数据集微调3个epoch得到的LLaVA-NeXT-7B-Video-FT，8帧采样，每帧144个token。分析同时覆盖5种时序推理任务（动作反义、动作顺序、场景转换、运动方向、物体计数），均来自TVBench基准。

### 关键设计

1. **Attention Knockout——因果追踪注意力贡献**:

    - 功能：选择性断开特定token对之间的注意力连接（将注意力mask对应位置设为$-\infty$），然后测量对模型预测的概率影响
    - 核心思路：对每层$l$，以窗口$k=9$为中心断开目标注意力边（如跨帧的video-to-video attention），计算相对概率变化$((p_{\text{knockout}} - p_{\text{base}})/p_{\text{base}}) \times 100$。窗口设为9是因为太窄时信息可通过残差连接绕过干预
    - 设计动机：相比观察性分析（如直接看attention weight），因果干预能准确量化"如果没有这条信息路径，模型行为会怎么变"。这是机制可解释性领域的标准方法，源自Geva et al. 2023

2. **Logit Lens——探针中间层语义内容**:

    - 功能：将中间层hidden state投影到语言模型head上得到logits，读出视频token在每层"看起来像什么词"
    - 核心思路：对视频token在每层的表示过LM head，统计空间关键词（物体、颜色）和时间关键词（before/after/first等）出现的频率和位置分布。用LLaVA-NeXT-13B-Video-FT在Action Sequence任务上进行
    - 设计动机：Attention Knockout告诉我们"哪些路径重要"，但不告诉我们"路径上流动的是什么信息"。Logit Lens填补这个gap，揭示视频token中时间概念何时涌现

3. **信息流阶段分解与验证**:

    - 功能：将注意力路径分为6类（跨帧video-video、video→question、video→last、question→last、last→last、question→video），分析各类路径在不同层的贡献
    - 核心思路：分别knockout每类路径在每层的注意力边，绘制层-概率变化曲线。路径角色清晰后，只保留关键层范围内的关键路径类型（如L6-15保留跨帧交互，L6-20保留video→question，L16-25保留question→last），禁用所有其余路径
    - 设计动机：从发现到验证的闭环——先分析出哪些路径关键，再用"只保留关键路径"的端到端实验证明分析的正确性

### 四大核心发现

**发现1：VideoQA微调诱导早中层跨帧交互。** 对比LLaVA-NeXT-7B（仅图像训练）和LLaVA-NeXT-7B-Video-FT（视频微调后），在早中层knockout跨帧注意力时，视频微调模型性能大幅下降，而图像模型几乎无反应。这说明视频微调特异性地在早中层建立了跨帧交互能力，且这是图像预训练中不存在的新能力。在前半层（L1-16）完全禁用跨帧注意力导致准确率下降18%-60.8%，且模型会生成语义相反的答案。

**发现2：视频token中时间概念的涌现与选择性传播。** Logit Lens显示空间概念（物体、颜色）在非常早的层就出现在前景token上，而时间概念（before/after等）在中层才涌现，且占据的是前景token之外的位置——模型先在前景token上稳定空间表示，再在剩余token位置编码时间动态。当保持正常跨帧注意力时，问题中的时间关键词（如"begins""ends"）会精确attend到视频中语义对应的时间段；禁用跨帧注意力后退化为基于位置proximity的注意力。

**发现3：答案生成在中后层启动。** 追踪最后token位置每层的正确答案概率，约第20层（中层）后概率急剧上升，正好与video→question信息流完成的时间一致。且正确选项快速压倒其他选项，不存在长时间的选项竞争。

**发现4：关键信息路径只占总注意力边的42%。** 保留关键路径、禁用其余58%注意力边后，在TVBench和TOMATO两个benchmark上性能几乎无损。随机剪枝同比例则性能崩溃。

## 实验关键数据

### 有效信息路径剪枝——多模型验证

| 模型 | 视频Token数 | 注意力边保留比例 | TVBench | TOMATO | vs Full Causal |
|------|------------|----------------|---------|--------|----------------|
| LLaVA-NeXT-7B-Video-FT | 8×12×12 | **42%** (10.8M/25.7M) | 51.2 | 29.2 | -0.3 / -1.0 |
| LLaVA-NeXT-7B-Video-FT (随机剪枝) | 同上 | 42% | 40.1 | 23.1 | -11.4 / -7.1 |
| LLaVA-NeXT-13B-Video-FT | 8×12×12 | **37%** (14.3M/32.2M) | 54.6 | 27.4 | -0.5 / +0.2 |
| Mini-InternVL-4B-Video-FT | 8×16×16 | **40%** (29.6M/74.6M) | 56.0 | 31.2 | 0.0 / -1.0 |
| VideoLLaMA3-7B | 8×12×12 | **58%** (11.4M/19.9M) | 57.2 | 28.7 | +2.0 / +0.7 |

保留有效路径的剪枝在4个不同架构/规模的模型上一致生效。VideoLLaMA3甚至剪枝后反超baseline，说明部分注意力边是干扰性噪声。

### 跨帧注意力禁用消融——按任务

| 任务 | 禁用前半层跨帧注意力后准确率下降 | 典型错误 |
|------|-------------------------------|---------|
| Action Antonym | -24.1% | "stand up" → "sit on chair"（语义相反） |
| Action Sequence | -20.2% | "open bag" → "put bag in microwave"（顺序完全错误） |
| Scene Transition | -18.0% | "bedroom→street" → "street→different location"（方向颠倒） |
| Moving Direction | -44.8% | "move right" → "move left"（方向相反） |
| Object Count | -60.8% | "zero moving objects" → "three"（数量完全错误） |

此表清晰展示跨帧注意力的不可或缺——禁用后模型不是"不确定"而是给出语义相反的答案，说明没有跨帧交互时模型回退到了静态偏置。

### 关键发现

- **VideoQA微调的独特贡献**：通过对比同架构的ImageLLM和VideoLLM，证实跨帧注意力交互是视频微调独有的习得能力，早中层的跨帧交互模式仅在视频微调后出现。这回答了"视频微调到底教了什么"这个基本问题
- **时间概念的"涌现"特性**：视频token中的时间概念不是视觉编码器直接产出的，而是在LLM的中间层自发涌现。空间概念先稳定（前景token），时间概念后涌现（剩余token），两类概念在token空间上互不覆盖
- **时间关键词的"信息检查站"角色**：问题中的时间关键词（选项中的动作词、时间副词）充当信息整合检查站。在不同任务中，视频信息到达这些检查站的路径不同：简单任务走直接路径(video→option)，需要目标识别的任务走间接路径(video→non-option question→option)
- **失败案例的机制诊断**：分析错误预测样本发现，跨模态整合路径（中层→后层）的模式与正确样本一致，说明失败源头在更早的视频表示阶段——要么是虚假的跨帧注意力带偏了表示(Case 1)，要么是模型回退到了静态场景偏置(Case 2)

## 亮点与洞察

- **完整的三阶段推理蓝图**：将VideoLLM的黑箱推理过程分解为可检验、可操作的三个阶段。这不只是描述性分析——通过"只保留关键路径"的端到端验证形成了分析→假设→验证的闭环，增强了发现的可信度
- **58%注意力边可安全剪枝**：这一发现直接指向实际应用——可以构建更高效的VideoLLM推理pipeline。不同于启发式的token压缩方法，本文从机制层面解释了为什么这些token交互是冗余的，为注意力稀疏化提供了理论依据
- **"时间概念涌现"的发现特别精彩**：视频经过空间编码器处理后，token本身不直接包含时间语义。但在LLM处理过程中，时间概念在中间层自发涌现在非前景token位置上。这暗示LLM的自注意力机制具有从位置编码序列中"发明"时间语义的能力
- **失败模式的两种机制**：Case 1（虚假跨帧注意力）和Case 2（静态偏置回退）的区分，为针对性改进提供了方向——前者需要改进跨帧交互质量，后者需要减少训练数据的静态场景偏置

## 局限与展望

- **任务覆盖有限**：主要分析在TVBench（多选QA）上进行，虽Appendix补充了开放式QA和长视频，但信息流模式在视频描述、视频摘要等生成任务中可能有本质不同
- **模型规模局限**：最大分析模型是13B参数。70B+级别模型是否也遵循同样的三阶段模式？更深的网络可能有更复杂的信息路由
- **Attention Knockout的窗口粒度**：使用$k=9$的层窗口是为了防止残差绕过，但这使得分析的层精度较粗。更细粒度的因果干预（如单层+MLP分析）可能揭示更精细的机制
- **剪枝层范围是人工经验设定的**：有效路径的层范围（如L6-15用于跨帧交互）基于分析观察人工确定。自适应地学习每个样本的最优路径范围可能进一步提升剪枝效率
- **静态分析缺乏动态自适应**：当前分析是数据集级别的统计模式，但每个视频样本的最优路径可能不同。开发sample-adaptive的路径选择可能是有价值的研究方向

## 相关工作与启发

- **vs 图像MLLM可解释性 (Neo 2025, Zhang 2025c)**：他们发现图像MLLM有结构化的视觉-语言信息流模式，本文将分析范式扩展到视频域并发现了全新的能力——跨帧时序交互。本文的关键贡献是证明视频微调引入了图像预训练中不存在的新计算路径
- **vs Token压缩方法 (FastV, LLaVA-PruMerge等)**：这些方法从效率角度启发式地删减视频token，本文从机制层面解释了为什么某些token交互可以安全删除——它们不在关键信息路径上。二者可以结合：用本文发现指导更有原则的token/attention压缩
- **vs 早退策略 (Elbayad 2020, Schuster 2022)**：本文发现答案在中层之后就准备好了，直接暗示了early exit的可行性——中后层的计算有很大比例是冗余的。与传统基于confidence的early exit不同，本文提供了基于信息流完成度的structural early exit依据

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个VideoLLM时序推理的完整机制分析，填补了视频领域可解释性的空白
- 实验充分度: ⭐⭐⭐⭐ 5种任务、4个模型验证、分析→假设→验证闭环完整，但任务类型主要限于多选QA
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题分解清晰，图示直观（尤其Fig.1的流程蓝图），发现递进式展现有很强的叙事逻辑
- 价值: ⭐⭐⭐⭐ 对VideoLLM架构设计、注意力稀疏化、推理加速有直接指导意义，失败模式分析指出了改进方向

<!-- RELATED:START -->

## 相关论文

- [Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously](../../CVPR2025/video_understanding/video_streaming_thinking_videollms_can_watch_and_think_simultaneously.md)
- [ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding](../../AAAI2026/video_understanding/reason_reinforced_causal_search_with_information_bottleneck_for_video_understand.md)
- [Causality Matters: How Temporal Information Emerges in Video Language Models](../../AAAI2026/video_understanding/causality_matters_how_temporal_information_emerges_in_video_language_models.md)
- [ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs](../../ACL2025/video_understanding/icr_probe_tracking_hidden_state_dynamics_for_reliable_hallucination_detection_in.md)

<!-- RELATED:END -->
