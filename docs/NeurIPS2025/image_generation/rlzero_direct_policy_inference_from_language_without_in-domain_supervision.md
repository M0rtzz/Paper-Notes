---
title: >-
  [论文解读] RLZero: Direct Policy Inference from Language Without In-Domain Supervision
description: >-
  [NeurIPS 2025][图像生成][零样本策略推理] 提出 RLZero 框架，通过"想象 → 投影 → 模仿"三步流程，将自然语言指令转化为目标环境中的行为策略——用视频生成模型从语言"想象"观测序列，将其投影到目标域，最终由无监督预训练的 RL 智能体通过闭合形式解即时模仿，整个过程无需任何域内监督或标注轨迹。
tags:
  - NeurIPS 2025
  - 图像生成
  - 零样本策略推理
  - 语言条件RL
  - 视频生成模型
  - 无监督RL
  - 跨embodiment迁移
---

# RLZero: Direct Policy Inference from Language Without In-Domain Supervision

**会议**: NeurIPS 2025  
**arXiv**: [2412.05718](https://arxiv.org/abs/2412.05718)  
**代码**: 无  
**领域**: 强化学习 / 语言条件控制  
**关键词**: 零样本策略推理, 语言条件RL, 视频生成模型, 无监督RL, 跨embodiment迁移

## 一句话总结

提出 RLZero 框架，通过"想象 → 投影 → 模仿"三步流程，将自然语言指令转化为目标环境中的行为策略——用视频生成模型从语言"想象"观测序列，将其投影到目标域，最终由无监督预训练的 RL 智能体通过闭合形式解即时模仿，整个过程无需任何域内监督或标注轨迹。

## 研究背景与动机

**领域现状**：奖励假说认为所有目标都可以表示为标量奖励信号的最大化，但实践中定义合适的奖励函数极其困难。自然语言为指导 RL 智能体提供了直觉替代方案，但已有的语言条件 RL 方法要么需要昂贵的域内监督（标注轨迹、语言-动作配对），要么需要在获得新语言指令后进行测试时训练。

**现有痛点**：(1) 传统语言条件 RL 需要大量人工标注的语言-轨迹对，收集成本高；(2) 基于奖励函数学习的方法仍需域内训练数据来桥接语言和环境；(3) 测试时训练（test-time training）的方法虽然减少了预标注需求，但每次新指令都需要训练，无法实现即时执行。

**核心矛盾**：语言指令空间是开放的且与环境无关，而 RL 策略必须扎根于具体环境的动力学——如何在不对特定任务进行任何监督的前提下桥接这两个空间？

**本文目标** (1) 实现完全零样本的语言到行为转换——无域内监督、无标注轨迹、无测试时训练；(2) 让预训练的 RL 智能体能响应任意自然语言指令；(3) 支持跨 embodiment 迁移，包括从 YouTube 视频到机器人的迁移。

**切入角度**：将问题分解为三个可独立解决的子问题——语言→视觉（利用视频生成模型的强大 text-to-video 能力），视觉→跨域（domain transfer），以及观测→动作（利用无监督 RL 预训练的闭合形式模仿）。

**核心 idea**：用视频生成模型做语言到视觉的翻译，然后让无监督预训练的 RL 智能体直接模仿翻译出的观测序列，从而绕过域内监督的需求。

## 方法详解

### 整体框架

RLZero 的 pipeline 由三个独立模块顺序组成：(1) **Imagine** — 给定自然语言指令，调用预训练的 text-to-video 模型生成一段符合指令描述的观测序列（想象中的任务执行过程）；(2) **Project** — 将生成的视频帧通过域迁移方法投影到目标环境的视觉空间（消除源域与目标域之间的 domain gap）；(3) **Imitate** — 在目标环境中，使用无监督 RL 预训练的智能体，通过闭合形式解（closed-form solution）即时计算出模仿投影后观测序列的策略，无需任何额外训练。

### 关键设计

1. **Imagine: 视频生成模型做语言→视觉翻译**:

    - 功能：将任意自然语言指令转化为视觉化的任务执行序列
    - 核心思路：利用大规模预训练的 text-to-video 生成模型（如扩散模型），输入语言描述，输出一段包含环境中智能体执行对应任务的视频帧序列。视频生成模型的通用知识充当了语言-行为之间的"桥梁"
    - 设计动机：视频模型在互联网规模数据上预训练，已具备丰富的世界知识和语言-视觉关联，无需针对特定 RL 环境做任何微调

2. **Project: 跨域视觉投影**:

    - 功能：消除生成视频帧与目标环境观测之间的域差距
    - 核心思路：将 Imagine 阶段生成的视频帧（可能是卡通风格、模拟器渲染或真实世界视角）映射到目标环境的观测空间中，使后续模仿阶段能将其作为合法的环境观测处理。可用的技术包括风格迁移、视觉编码器的特征对齐等
    - 设计动机：视频生成模型产生的图像风格和视角通常与目标环境不一致，直接使用会导致模仿失败

3. **Imitate: 闭合形式无监督 RL 模仿**:

    - 功能：从投影后的观测序列即时推断出可执行的策略
    - 核心思路：在预训练阶段，智能体通过无监督 RL（如 goal-conditioned RL 或 successor features）在目标环境中进行无任务标签的探索性训练，学习到通用的状态-动作映射能力。在推理阶段，给定投影后的目标观测序列，智能体通过闭合形式解（如求解最小二乘或线性规划）即时计算出模仿该序列所需的动作，无需梯度下降或额外训练
    - 设计动机：闭合形式解保证了零延迟的策略生成，使系统能真正实现"语言输入→即时行为输出"

### 损失函数 / 训练策略

预训练阶段使用无监督 RL 的目标函数（具体取决于所用的无监督 RL 方法，如互信息最大化或 goal-conditioned reward）。推理阶段无需训练。

## 实验关键数据

### 主实验

| 环境 | 模态 | 任务类型 | RLZero 成功率 | 基线对比 | 说明 |
|------|------|----------|--------------|----------|------|
| 多种连续控制 | 语言→行为 | 多种操作任务 | 有效 | 首个零样本方法 | 无域内监督 |
| 跨 embodiment | YouTube视频→行为 | 人形机器人 | 有效 | — | 从YouTube到humanoid |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 完整 RLZero | 最佳 | 三步协同 |
| 无 Project 步骤 | 下降 | 域差距导致模仿偏差 |
| 替换不同视频模型 | 有变化 | 生成质量影响下游性能 |

### 关键发现

- **首个零样本语言→行为方法**：RLZero 是第一个在多种任务和环境中展示直接语言到行为生成能力的方法，且完全不需要域内监督
- **跨 embodiment 迁移可行**：不仅能处理语言指令，还能从 YouTube 上的跨 embodiment 视频（如人类演示）推断出人形机器人的策略
- **模块化设计的优势**：三个模块可独立升级——更好的视频模型、更强的域适应方法、更优的无监督 RL 算法都能直接提升整体性能

## 亮点与洞察

- **三步解耦策略**将一个极难的端到端问题分解为三个相对独立的子问题，每个子问题都有成熟工具可用，这种"divide and conquer"式的系统设计非常巧妙。可以迁移到其他需要跨模态转换的机器人规划问题
- **闭合形式模仿**实现了真正的零延迟推理，这比需要测试时训练的方法具有质的优势——在实际部署中，每条新指令都需要重新训练是无法接受的
- **视频生成模型作为"世界模型"**的新用法值得关注：不是用来预测环境动力学，而是用来将语言转化为可模仿的行为参考

## 局限与展望

- **视频生成质量的瓶颈**：系统性能上限受限于视频生成模型的质量，当语言指令描述复杂或罕见动作时，生成视频可能不准确
- **投影步骤的鲁棒性**：域迁移可能丢失关键的空间信息或时序信息，尤其在源域和目标域视觉差异巨大时
- **无监督 RL 预训练的覆盖范围**：如果预训练阶段的探索未覆盖到任务所需的状态-动作组合，闭合形式模仿可能产生不合理的动作
- **缺少定量对比**：由于是首个零样本方法，难以与需要监督的方法做公平对比；论文定量实验的规模和深度有待加强
- **长视界任务**：视频生成模型产生的序列长度有限，对需要长时间规划的任务可能不适用

## 相关工作与启发

- **vs SayCan / Language-to-Reward**: SayCan 需要预训练的技能库，Language-to-Reward 需要定义奖励函数，两者都需要一定程度的域内监督。RLZero 完全绕过了这些需求
- **vs UniPi**: UniPi 也使用视频生成模型做规划，但它需要在目标域内训练逆动力学模型来将视频转化为动作。RLZero 用无监督 RL + 闭合形式解替代了此步骤
- **vs DALL-E-Bot / SuSIE**: 这些方法用图像生成模型产生子目标图像，但仍需要目标条件策略的域内训练。RLZero 的无监督预训练更加通用
- 论文展示了基础模型（视频生成、无监督 RL）组合使用的巨大潜力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "想象-投影-模仿"的系统架构解决了一个长期开放问题，真正实现零样本语言→行为
- 实验充分度: ⭐⭐⭐ 展示了多种环境和跨 embodiment 能力，但定量对比和消融可更深入
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，动机论述有说服力
- 价值: ⭐⭐⭐⭐⭐ 首个零样本语言条件 RL 方法，开辟了全新研究方向

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[NeurIPS 2025\] Flattening Hierarchies with Policy Bootstrapping](flattening_hierarchies_with_policy_bootstrapping.md)
- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](../../ICML2025/image_generation/performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[NeurIPS 2025\] Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising](two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)
- [\[ICLR 2026\] Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations](../../ICLR2026/image_generation/translate_policy_to_language_flow_matching_generated_rewards_for_llm_explanation.md)

<!-- RELATED:END -->
