---
title: >-
  [论文解读] Visual Grounding for Object-Level Generalization in Reinforcement Learning
description: >-
  [ECCV2024][visual grounding] 利用视觉语言模型 (MineCLIP) 的 visual grounding 能力生成目标物体的 confidence map，通过奖励设计和任务表征两条路径将 VLM 知识迁移到强化学习中，实现对未见物体和指令的零样本泛化。
tags:
  - ECCV2024
  - visual grounding
  - zero-shot generalization
  - VLM
  - Minecraft
  - CLIP
  - intrinsic reward
---

# Visual Grounding for Object-Level Generalization in Reinforcement Learning

**会议**: ECCV2024  
**arXiv**: [2408.01942](https://arxiv.org/abs/2408.01942)  
**代码**: [PKU-RL/COPL](https://github.com/PKU-RL/COPL)  
**领域**: reinforcement_learning  
**关键词**: visual grounding, zero-shot generalization, VLM, Minecraft, CLIP, intrinsic reward

## 一句话总结
利用视觉语言模型 (MineCLIP) 的 visual grounding 能力生成目标物体的 confidence map，通过奖励设计和任务表征两条路径将 VLM 知识迁移到强化学习中，实现对未见物体和指令的零样本泛化。

## 背景与动机
在 Minecraft 等开放世界环境中，智能体需要根据自然语言指令与多种物体交互。然而训练数据的覆盖范围总是有限的，智能体在评估时会遇到训练中从未见过的物体名称。现有方法存在两个关键不足：

1. **MineCLIP 奖励对距离不敏感**：MineCLIP 通过图像序列与文本的相似度作为内在奖励，但该相似度与智能体到目标物体的实际距离不相关。智能体倾向于远远"盯着"目标而不接近，无法完成需要多次交互的困难技能（如捕猎）。
2. **语言嵌入作为任务表征泛化能力不足**：传统方法直接用语言嵌入作为策略输入，面对训练集之外的物体名称时，策略网络无法理解其含义。

## 核心问题
如何以最小代价将 VLM 中的视觉-语言知识迁移到 RL，使智能体既能高效学习基本技能，又能对训练中未见过的目标物体实现零样本泛化？

## 方法详解
提出 **COPL (CLIP-guided Object-grounded Policy Learning)**，包含三个核心模块：

### 1. Visual Grounding：生成 Confidence Map
- 首先用 GPT-4 从自然语言指令中提取目标物体名称（如从"hunt a cow in plains with a diamond sword"中提取"cow"）
- 对 MineCLIP 的图像编码器做 MaskCLIP 风格的修改：**移除最后一个 ViT block 中 multi-head attention 的 scaled dot-product attention，只保留 value-embedding 变换**，使每个 patch 位置的特征可单独使用
- 将各 patch embedding 单独通过 MineCLIP 的 temporal transformer（序列长度设为 1），保证与 MineCLIP 嵌入空间对齐
- 文本端不做修改，编码目标名称和一组负样本词汇，计算各 patch 与文本 embedding 的余弦相似度，经 softmax 得到每个 patch 上目标存在的概率
- 最终输出一个二维 confidence map，尺寸等于 patch 数量

### 2. Transfer via Reward：Focal Reward
设计 **focal reward**，在每个时间步 $t$ 计算：

$$r_t^f = \text{mean}(m_t^c \circ m^k)$$

其中 $m_t^c$ 为目标 confidence map，$m^k$ 为以视野中心为均值的二维高斯核（$\sigma_1 = H/3, \sigma_2 = W/3$），$\circ$ 为 Hadamard 积。

- **面积代理距离**：目标越近 → 占据像素越多 → 奖励越大
- **高斯核鼓励居中**：目标越靠近视野中心 → 奖励越大，解决多目标时智能体不知该追谁的问题
- **去噪处理**：(1) 负样本词概率最高的 patch 置零；(2) 低于阈值 $\tau=0.2$ 的置零、高于阈值的置一

最终训练奖励为 $r_t = r_t^{env} + \lambda r_t^f$，$\lambda=5$。

### 3. Transfer via Representation：Confidence Map 作为任务表征
- 不再使用语言嵌入作为策略输入，而是将 confidence map 作为统一的二维任务表征
- 策略网络采用 MineAgent 架构，增加一个分支编码 confidence map，通过拼接融合多模态特征
- 关键优势：面对未见物体时，MineCLIP 的 open-vocabulary 特性仍能产生合理的 confidence map，策略网络可以理解这个基于视觉的二维表征并据此行动
- 使用 PPO 进行多任务 RL 训练

## 实验关键数据

### 单任务实验（4 个 hunting 技能）

| 任务 | Focal | MineCLIP | NDCLIP | Sparse |
|------|-------|----------|--------|--------|
| hunt a cow | **71.3±9.7** | 3.8±4.8 | 3.5±3.0 | 0.0±0.0 |
| hunt a sheep | **68.8±25.3** | 5.3±2.9 | 28.8±23.0 | 2.5±3.0 |
| hunt a pig | **58.3±7.8** | 2.3±1.7 | 0.3±0.5 | 0.5±0.6 |
| hunt a chicken | **29.5±10.9** | 0.0±0.0 | 4.8±1.5 | 0.5±0.6 |

Focal reward 是唯一能掌握全部四个困难技能的方法。

### 多任务零样本泛化（Hunting 域，未见目标）

| 未见目标 | COPL | LCRL[t] | LCRL[i] |
|----------|------|---------|---------|
| llama | **48.8±6.5** | 14.5±10.4 | 24.5±12.7 |
| horse | **49.0±5.5** | 2.5±1.3 | 5.5±4.7 |
| spider | **54.5±12.7** | 9.8±3.5 | 18.3±12.0 |
| mushroom cow | **40.3±11.2** | 19.3±20.5 | 0.0±0.0 |
| **平均** | **48.1** | 11.5 | 12.1 |

在未见目标上，COPL 平均成功率约为语言条件方法的 **4 倍**（hunting 域）和 **2 倍**（harvest 域）。

## 亮点
1. **思路简洁有效**：仅对 MineCLIP 做轻量修改（无需微调）即可获得 visual grounding 能力，计算开销极低
2. **Focal reward 设计精巧**：高斯核同时解决了距离引导和多目标聚焦两个问题，比 MineCLIP reward 更符合任务需求
3. **两条迁移路径互补**：reward 路径解决技能学习效率，representation 路径解决泛化能力
4. **实验对比全面**：单任务/多任务/泛化实验层层递进，同时与模仿学习方法（VPT 系列）做了参考对比

## 局限与展望
1. **仅适用于物体为中心的任务**：对于 "dig a hole"、"build a house" 等非物体中心任务，难以定义明确的目标物体用于 grounding
2. **不考虑动作泛化**：只支持目标物体级别的泛化，无法泛化到训练中未见的行为模式
3. **依赖 LLM 提取目标**：需要 GPT-4 从指令中提取目标物体名，增加了系统依赖
4. **confidence map 存在噪声**：虽有去噪处理，但原始 map 质量受 MineCLIP 视觉编码器的限制

## 与相关工作的对比
- **vs MineCLIP reward**：MineCLIP reward 对距离不敏感，focal reward 通过像素面积代理距离解决了这一问题
- **vs 模仿学习方法 (VPT, STEVE-1)**：模仿学习依赖大规模标注数据，泛化能力受限于训练数据覆盖；COPL 通过 VLM 的 open-vocabulary 能力绕过了这一限制
- **vs 语言条件 RL (LCRL)**：LCRL 直接用语言嵌入作为策略输入，面对未见词汇时陷入困境；COPL 将语言映射到视觉 confidence map，提供更统一可理解的表征
- **vs MaskCLIP/CLIPSurgery**：在 Minecraft 领域使用领域特定的 MineCLIP 效果优于通用 CLIP 模型，且可与现有计算流程复用

## 启发与关联
- **VLM → RL 知识迁移范式**：通过 reward 和 representation 两条路径实现迁移是一种通用思路，可推广到机器人操作等场景
- **visual grounding 作为中间表征**：将语言指令转化为二维视觉概率图作为策略输入，比直接使用语言嵌入更具可解释性和泛化性
- **领域特定 VLM 的价值**：通用 CLIP 在 Minecraft 中效果不佳，MineCLIP 经过领域微调后显著提升，这提示在特定领域应优先选用领域适配的 VLM
- **去噪策略的通用性**：confidence map 的阈值化和负样本过滤思路可迁移到其他基于 CLIP 的 dense prediction 任务中

## 评分
- 新颖性: ⭐⭐⭐⭐ — Focal reward 和 confidence map 作为任务表征的设计均有新意
- 实验充分度: ⭐⭐⭐⭐ — 单任务/多任务/泛化的实验设计层层递进，消融充分
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、方法阐述详细
- 价值: ⭐⭐⭐⭐ — 为 VLM 与 RL 结合提供了实用且有启发性的范式

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding](../../AAAI2026/reinforcement_learning/start_small_think_big_curriculum-based_relative_policy_optimization_for_visual_g.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Human-Inspired Multi-Level Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/human-inspired_multi-level_reinforcement_learning.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](../../AAAI2026/reinforcement_learning/object-centric_latent_action_learning.md)
- [\[AAAI 2026\] Object-Centric World Models for Causality-Aware Reinforcement Learning](../../AAAI2026/reinforcement_learning/object-centric_world_models_for_causality-aware_reinforcement_learning.md)

<!-- RELATED:END -->
