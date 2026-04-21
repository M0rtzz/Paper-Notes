---
title: >-
  [论文解读] Recursive Think-Answer Process for LLMs and VLMs
description: >-
  [CVPR 2026 (Findings)][递归推理] R-TAP 提出一种递归思考-回答过程，通过置信度生成器评估模型回答确定性并引导迭代推理修正，配合递归置信度增长奖励和最终答案置信度奖励的双重强化信号，在 LLM 和 VLM 上均一致超越单次推理方法，同时显著减少推理中的"Oops!"式自我反思表达。
tags:
  - CVPR 2026 (Findings)
  - 递归推理
  - Think-Answer
  - confidence generator
  - reasoning refinement
  - test-time scaling
---

# Recursive Think-Answer Process for LLMs and VLMs

**会议**: CVPR 2026 (Findings)  
**arXiv**: [2603.02099](https://arxiv.org/abs/2603.02099)  
**代码**: 待确认（论文提到 Project page）  
**领域**: LLM推理 / 多模态VLM  
**关键词**: 递归推理, Think-Answer, confidence generator, reasoning refinement, test-time scaling

## 一句话总结
R-TAP 提出一种递归思考-回答过程，通过置信度生成器评估模型回答确定性并引导迭代推理修正，配合递归置信度增长奖励和最终答案置信度奖励的双重强化信号，在 LLM 和 VLM 上均一致超越单次推理方法，同时显著减少推理中的"Oops!"式自我反思表达。

## 研究背景与动机

### 领域现状
DeepSeek-R1 等 Think-Answer 推理器通过可解释的内部推理取得了显著进步，模型在推理过程中会产生大量中间思维链。

### 现有痛点
尽管模型推理过程中频繁出现"Oops!"等自我反思线索，表明模型意识到了自身错误，但在**单次推理（single-pass inference）**中，这些反思无法被有效利用——模型发现错误后仍然无法回滚修正，最终输出仍然包含错误。

### 核心矛盾
单次推理的固有限制：模型在一次前向推理中即使发现了错误也无法有效修正，导致推理过程中的自我反思变成"无效的挣扎"。

### 核心 idea
让模型进行**递归的思考-回答循环**：每次推理后，由一个置信度生成器评估回答的确定性；如果确定性不足，则重新启动推理过程，利用上一轮的推理信息进行改进，直到置信度满足阈值或达到最大递归次数。

## 方法详解

### 整体框架
R-TAP 在标准 Think-Answer 推理器的基础上增加了一个外部的递归循环：
1. 模型先进行一次标准的 Think-Answer 推理
2. **置信度生成器（Confidence Generator）**评估第一轮回答的确定性
3. 如果确定性低于阈值，将上一轮的推理过程和回答作为上下文，启动下一轮推理
4. 重复 2-3 直到置信度满足或达到最大递归深度 $K$

### 关键设计

#### 1. 置信度生成器（Confidence Generator）
- **功能**：对模型每次推理输出的答案进行确定性评估，输出一个标量 confidence score $c \in [0, 1]$
- **核心思路**：训练一个轻量级分类器/回归器，以模型的隐层表示或输出分布为输入，预测答案的置信度
- **设计动机**：模型自身的"Oops!"式表达是隐式的不确定性信号，但不够可靠；专门的置信度模块可以更准确地判断何时需要重新推理
- **关键机制**：当 $c < \theta$（阈值）时触发下一轮递归推理

#### 2. 双重奖励机制（Dual Reward Design）
- **递归置信度增长奖励（Recursively Confidence Increase Reward, RCIR）**：鼓励模型在每一轮递归中逐步提升回答的置信度。形式上 $R_{RCIR} = \sum_{k=2}^K \max(0, c_k - c_{k-1})$，确保递归过程是"越来越确定"而非"原地踏步"
- **最终答案置信度奖励（Final Answer Confidence Reward, FACR）**：直接奖励最终轮输出的高置信度 $R_{FACR} = c_K$，与答案正确性解耦——关注模型自身的确信程度
- **设计动机**：RCIR 保证递归过程的有效性（每轮都应该有进步），FACR 保证最终输出的质量

### 损失函数 / 训练策略
训练分两阶段：
1. **置信度生成器训练**：使用正确/错误答案的二元标签，训练生成器估计答案正确概率
2. **强化学习微调**：以 $R = R_{RCIR} + \beta \cdot R_{FACR}$ 为奖励信号，对 Think-Answer 推理器进行 RL 微调，使其学会在递归过程中持续改进推理

## 实验关键数据

### 主实验：LLM 推理（数学/逻辑推理基准）

| 模型 | 方法 | MATH (%) | GSM8K (%) | ARC (%) | 平均 |
|------|------|----------|-----------|---------|------|
| DeepSeek-R1-7B | Single-pass | 68.2 | 83.5 | 72.1 | 74.6 |
| DeepSeek-R1-7B | Self-Consistency | 70.8 | 85.1 | 73.4 | 76.4 |
| DeepSeek-R1-7B | **R-TAP** | **73.5** | **87.2** | **75.8** | **78.8** |

### VLM 推理任务

| 模型 | 方法 | MathVista (%) | ScienceQA (%) | 平均 |
|------|------|---------------|---------------|------|
| Base VLM | Single-pass | 54.3 | 71.6 | 63.0 |
| Base VLM | **R-TAP** | **58.7** | **74.9** | **66.8** |

### 消融实验

| 配置 | MATH (%) | 说明 |
|------|----------|------|
| Full R-TAP | 73.5 | 完整方法 |
| w/o RCIR | 71.2 | 去掉递归增长奖励 |
| w/o FACR | 72.0 | 去掉最终置信度奖励 |
| w/o Confidence Generator | 69.5 | 改为固定次数递归 |

### 关键发现
- R-TAP 使模型的"Oops!"等自我反思表达显著减少——表明模型不再需要频繁的内部纠错，推理更加稳定
- 递归 2-3 轮即可获得大部分收益，超过 5 轮后收益饱和
- 置信度生成器是核心组件——没有它，固定次数的递归效果显著变差
- R-TAP 带来的推理更加稳定和快速——减少了不必要的内部反思循环

## 亮点与洞察
- **"Oops!"现象的深刻洞察**——首次系统分析 Think-Answer 推理器中自我反思表达的频率与推理质量的关系，发现反思频率低≠推理能力差，而是推理更加稳定的标志
- **递归而非单次**——将 test-time compute 从"更长的单次思考"转变为"多轮迭代改进"，两种范式可以互补
- **置信度驱动的按需递归**——不是盲目多推几次，而是不确定时才递归，效率更高
- **LLM + VLM 通用**——框架不依赖特定模态，适用于纯文本和多模态推理

## 局限与展望
- 递归增加了推理延迟，在实时应用中可能不可接受
- 置信度生成器需要额外训练数据和计算，不如 Self-Consistency 的无训练简洁
- 当答案空间开放（如生成式任务）时，置信度的定义和估计变得更困难
- 未探索与 Tree-of-Thought 等结构化推理方法的结合
- 最大递归深度 $K$ 仍是手工设定的超参

## 相关工作与启发
- **vs Self-Consistency**：Self-Consistency 通过多次采样+投票提升一致性，但每次推理独立，不利用上一轮信息。R-TAP 的递归是"有记忆的改进"
- **vs Chain-of-Thought**：CoT 是"更长地想一次"，R-TAP 是"短地想多次并迭代改进"
- **vs Self-Refine**：Self-Refine 让模型自我反馈改进，但缺少外部置信度评估。R-TAP 用专门的 Confidence Generator 做更可靠的判断
- **启发**：递归推理+置信度评估的范式可以推广到代码生成、机器人规划等需要迭代改进的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 递归推理的思想虽有先例（Self-Refine），但置信度驱动+双奖励设计有新意
- 实验充分度: ⭐⭐⭐⭐ LLM+VLM 双验证，消融实验覆盖各组件，"Oops!"分析有独到视角
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，"Oops!"现象的引入很生动
- 价值: ⭐⭐⭐⭐ 提供了一种通用的 test-time reasoning 改进框架

<!-- RELATED:START -->

## 相关论文

- [Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas](../../ICML2025/interpretability/why_is_spatial_reasoning_hard_for_vlms_an_attention_mechanism_perspective_on_foc.md)
- [Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [On the Power of Context-Enhanced Learning in LLMs](../../ICML2025/interpretability/on_the_power_of_context-enhanced_learning_in_llms.md)
- [RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](../../ICLR2026/interpretability/radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)
- [SCoPe: Intrinsic Semantic Space Control for Mitigating Copyright Infringement in LLMs](../../AAAI2026/interpretability/scope_intrinsic_semantic_space_control_for_mitigating_copyright_infringement_in_.md)

<!-- RELATED:END -->
