---
title: >-
  [论文解读] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation
description: >-
  [ICLR 2026][代码智能][Chart-to-Code] 针对图表到代码生成任务中SFT的性能瓶颈问题，提出多模态结构化强化学习（MSRL），通过文本+视觉双层奖励函数和两阶段RL策略，在ChartMimic和ReachQA上分别提升6.2%和9.9%的高层指标，达到开源SOTA并媲美GPT-4o。
tags:
  - "ICLR 2026"
  - "代码智能"
  - "Chart-to-Code"
  - "强化学习"
  - "SFT瓶颈"
  - "多粒度奖励"
  - "GRPO"
---

# Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation

**会议**: ICLR 2026  
**arXiv**: [2508.13587](https://arxiv.org/abs/2508.13587)  
**代码**: [GitHub](https://github.com/DocTron-hub/MSRL)  
**领域**: 多模态VLM / 代码生成  
**关键词**: Chart-to-Code, 强化学习, SFT瓶颈, 多粒度奖励, GRPO

## 一句话总结
针对图表到代码生成任务中SFT的性能瓶颈问题，提出多模态结构化强化学习（MSRL），通过文本+视觉双层奖励函数和两阶段RL策略，在ChartMimic和ReachQA上分别提升6.2%和9.9%的高层指标，达到开源SOTA并媲美GPT-4o。

## 研究背景与动机
**领域现状**: 多模态大模型（MLLM）在视觉问答等任务上表现出色，但在处理信息密集型图像（如图表）并生成结构化输出（如代码）时能力仍然有限。Chart-to-code生成任务要求模型深度理解可视化图表并生成准确的绘图代码，具有重要的实际价值。

**现有痛点**: 已有方法依赖SFT或DPO在合成数据上训练，数据模式单一（合成数据缺乏真实复杂度），泛化能力有限。而且SFT存在固有缺陷——它对目标序列中每个token赋予相同重要性，而绘图代码中大量是模板代码（如`plt.plot`），关键信息（数据值、样式参数）出现频率很低。

**核心矛盾**: SFT数据量扩大到一定规模后收益递减，出现**性能平台效应**。实验证明从200k扩展到2.8M样本后，超过2M之后性能不再增长。这意味着靠堆数据无法突破上限。

**本文目标**: (1) 系统验证SFT的性能天花板；(2) 设计有效的RL策略突破这个天花板；(3) 构建大规模真实数据的Chart-to-code训练语料。

**切入角度**: 作者观察到SFT的均匀权重分配机制无法重视关键token，而RL可以通过定制奖励函数重点优化关键内容的准确性。同时，纯文本奖励忽视整体视觉结构，需要引入视觉反馈形成多粒度奖励。

**核心 idea**: 用多粒度（文本+视觉）奖励函数驱动两阶段GRPO强化学习，突破Chart-to-code任务中SFT的性能瓶颈。

## 方法详解

### 整体框架
输入是一张图表图像，输出是一段可执行的 Matplotlib 绘图代码。MSRL 把训练拆成三步走：先用大规模真实数据做 SFT 预训练，把"看懂图表、写出能跑的代码"这个基础能力顶到 SFT 的天花板；然后进入两阶段 RL——第一阶段只用文本奖励抠代码细节，第二阶段再叠加视觉奖励提升整体视觉保真度。底座模型是 Qwen2.5-VL-7B，RL 算法采用 GRPO（用组内相对优势做优化，省掉 critic 模型）。

### 关键设计

**1. 大规模真实数据构建：用真实表格替合成数据，把 SFT 的瓶颈彻底压出来。**

已有数据集有两个硬伤——纯合成数据趋势单调、缺乏多样性，规模也不足以暴露 SFT 的真实上限。MSRL 因此从 arXiv 论文里爬取 2023 年及之前的真实表格，用 Gemini-2.5-Flash 根据表格和示例代码生成绘图代码，再经执行验证与过滤，最终得到约 300 万对的训练语料——目前最大的 chart-to-code 数据集，覆盖 24 种图表类型、1555 种 Matplotlib API。真实表格让数据分布更贴近实际场景，也才撑得起后面对 SFT 平台效应的量化验证。

**2. RL 数据精细化筛选：从 300 万里精挑 33k，逼模型探索而不是背 SFT 格式。**

RL 既要高质量样本，又要和 SFT 数据分离，否则模型会过拟合 SFT 的输出格式、失去探索能力。筛选分两阶段：第一阶段按代码内容过滤（图表类型、数据定义格式），用树结构解析识别复杂图表类型，只保留一维数组和非嵌套字典格式的样本，筛到 45k；第二阶段用 GPT-4.1-mini 做视觉质量评估（人工验证一致性 >90%），最终留下 33k。这批数据规模不大但干净，专供 RL 训练。

**3. 文本奖励（Textual Reward）：先把代码归一化，再分维度算细粒度正确性。**

绘图代码风格极其多样——同一个语义能有很多种写法，直接抽取关键信息会被语法噪声淹没。MSRL 的关键适配是先做**格式归一化**消除语法变体，再从五个维度评估：数据值用软匹配（允许 ±5% 误差）、图表类型用硬匹配、布局用硬匹配、标题标签等文本元素用编辑距离。执行奖励单独算，是个二值信号（代码能否跑通）。归一化这一步是 RLVR 用在结构化代码生成上的核心 trick，没有它奖励函数会被语法变体淹没、基本没法用。

**4. 视觉奖励（Visual Reward）：把代码渲染成图，让 MLLM 判"像不像"。**

文本奖励只盯细粒度代码细节，看不到整体视觉结构和风格。视觉奖励补这个盲区：把生成代码执行渲染成图像，再用 Qwen2.5-VL-72B 当评估模型，从图表类型、布局、文本内容、数据、样式、清晰度六个维度打分并归一化；渲染不出来的代码直接得 0。这一步把"代码对不对"映射成"图像像不像"这个更直观的维度，恰好和文本奖励互补。

**5. 两阶段 RL 策略：先低成本文本奖励铺量，再高成本视觉奖励精修。**

视觉奖励要渲染图像加大模型打分，开销很大，全程开着不划算。MSRL 因此把奖励拆成两阶段，总奖励写成

$$R = w_t R_\text{text} + w_v R_\text{vis} + w_e R_\text{exec}$$

第一阶段令 $w_v=0$，只用文本奖励在 22k 样本上训练，以低成本拿到大部分提升（约 5%）；第二阶段引入混合奖励，在 11k 样本上微调，再补约 1.5%。全程用 GRPO，靠组内相对优势做策略优化。这样既保住了性能，又把视觉奖励的算力花在刀刃上。

### 损失函数 / 训练策略
- SFT 阶段：标准自回归负对数似然损失。
- RL 阶段：GRPO，组内采样多个响应算相对优势，无需额外的 critic 模型。
- 两阶段课程训练：先文本奖励（约 240 GPU hours）→ 再混合奖励（约 336 GPU hours）。

## 实验关键数据

### 主实验

| 模型 | 参数量 | ChartMimic Exec. | ChartMimic High | ReachQA Exec. | ReachQA High |
|------|--------|-------------------|-----------------|---------------|--------------|
| GPT-4o | - | 93.2 | 83.5 | 92.8 | 84.0 |
| Qwen2.5-VL-7B | 7B | 73.2 | 41.6 | 62.2 | 37.6 |
| ChartCoder | 7B | 91.4 | 74.0 | 83.8 | 69.4 |
| MSRL-SFT | 7B | 93.2 | 77.6 | 92.2 | 80.0 |
| **MSRL** | **7B** | **96.5** | **83.8** | **98.2** | **89.9** |

MSRL以7B参数量超越所有开源模型，ChartMimic高层指标83.8超过GPT-4o的83.5，ReachQA高层指标89.9大幅超过GPT-4o的84.0。

### 消融实验

| 配置 | ChartMimic Exec. | Low-Level | High-Level | 说明 |
|------|-------------------|-----------|------------|------|
| Baseline (无SFT/RL) | 73.2 | 44.6 | 41.6 | Qwen2.5-VL-7B原始模型 |
| SFT only | 93.2 | 73.0 | 77.6 | SFT达到的天花板 |
| RL only (无SFT) | 93.8 | 65.6 | 62.3 | 直接RL不如SFT |
| SFT + RL (文本奖励) | 97.0 | 78.1 | 82.7 | 突破SFT瓶颈 |
| SFT + RL (两阶段) | 96.5 | 78.6 | 83.8 | 最终版，加入视觉奖励 |

奖励策略对比：纯视觉RL效果最好但需1344 GPU hours，两阶段策略以576 GPU hours达到相近性能。

### 关键发现
- **SFT瓶颈确认**：数据从200k增到2.8M，超过2M后性能不再增长，证明SFT有固有天花板
- **RL突破效果显著**：在饱和的SFT模型上，RL仍能带来ChartMimic高层指标+6.2%、ReachQA +9.9%的提升
- **两阶段策略高效**：第一阶段文本奖励贡献约5%提升，第二阶段视觉奖励额外贡献1.5%，但计算开销大幅降低
- **跨库泛化**：MSRL在Seaborn和Plotly测试集上也展示了泛化能力（训练仅用Matplotlib），Plotly上执行率从62.7%提升至90.0%

## 亮点与洞察
- **系统化的SFT瓶颈分析**：通过六个数据规模（200k-2.8M）的对照实验，首次在chart-to-code领域给出了SFT性能天花板的量化证据，这种方法论可以迁移到其他结构化生成任务。
- **代码格式归一化是RLVR的关键适配**：绘图代码的语法多样性（同一语义可以用不同写法），如果不做归一化，奖励函数会被噪声淹没。这个trick对所有涉及代码生成的RLVR任务都有参考价值。
- **视觉奖励作为结构化输出的全局校验**：将代码渲染为图像再评估，巧妙地把"代码是否正确"映射到了"图像是否相似"这个更直观的维度，适用于任何代码输出可被可视化的场景（如LaTeX生成、网页代码生成等）。
- **先文本后视觉的课程设计**：先用低成本奖励建立基础能力，再引入高成本奖励精细调整，是一个实用的资源-性能权衡策略。

## 局限与展望
- 仅在Matplotlib风格上训练，虽然展示了对Seaborn/Plotly的泛化，但泛化提升有限（Plotly高层仅35.9），多库联合训练值得探索
- 视觉奖励依赖Qwen2.5-VL-72B打分，成本高且可能引入MLLM自身偏差，能否用更轻量的度量（如SSIM、LPIPS）替代？
- 数据构建使用2023年前论文，可能不覆盖新型可视化风格
- RL数据仅33k，是否更大规模RL数据能进一步突破？论文显示RL也有性能平台（22k左右收敛）

## 相关工作与启发
- **vs ChartCoder**: ChartCoder也做chart-to-code但只用SFT+DPO在合成数据上训练，MSRL通过真实数据+RLVR大幅超越（高层指标74.0→83.8）
- **vs Chart-R1/BigCharts-R1**: 这些用RLVR提升图表推理QA，但不涉及代码生成。MSRL是首个将RLVR应用于chart-to-code的结构化代码生成任务。
- **vs DeepSeek-R1/Vision-R1**: R1系列关注通用推理的RL训练，MSRL在结构化输出生成上提出了针对性的多粒度奖励设计，展示了RLVR在不同任务范式下的适配方法。

## 评分
- 新颖性: ⭐⭐⭐⭐ SFT瓶颈的系统化验证和多粒度奖励设计有新意，但整体框架是GRPO+定制奖励的标准范式
- 实验充分度: ⭐⭐⭐⭐⭐ 数据规模实验、多维度消融、跨库泛化、与GPT-4o对比，实验非常充分
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表信息丰富，动机阐述有说服力
- 价值: ⭐⭐⭐⭐ 在chart-to-code这个细分领域建立了强SOTA，方法论对其他结构化代码生成任务有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ACL 2026\] Aligned Multi-View Scripts for Universal Chart-to-Code Generation](../../ACL2026/code_intelligence/aligned_multi-view_scripts_for_universal_chart-to-code_generation.md)
- [\[ICLR 2026\] Learning to Reason without External Rewards](learning_to_reason_without_external_rewards.md)

</div>

<!-- RELATED:END -->
