---
title: >-
  [论文解读] PreferThinker: Reasoning-based Personalized Image Preference Assessment
description: >-
  [ICLR2026][personalized preference assessment] 提出 PreferThinker，通过引入通用视觉偏好画像（preference profile）连接不同用户，采用 predict-then-assess 的 CoT 推理范式进行可解释的个性化图像偏好评估，结合冷启动 SFT + GRPO 强化学习及 similarity-aware 预测奖励，7B 模型超越 GPT-4o（+5.2%）和 Claude 3.7（+5.1%）。
tags:
  - ICLR2026
  - personalized preference assessment
  - reasoning
  - GRPO
  - predict-then-assess
  - visual preference profile
  - CoT
---

# PreferThinker: Reasoning-based Personalized Image Preference Assessment

**会议**: ICLR2026  
**arXiv**: [2511.00609](https://arxiv.org/abs/2511.00609)  
**代码**: [项目页面](https://preferthinker.github.io/)  
**领域**: reinforcement_learning  
**关键词**: personalized preference assessment, reasoning, GRPO, predict-then-assess, visual preference profile, CoT  

## 一句话总结
提出 PreferThinker，通过引入通用视觉偏好画像（preference profile）连接不同用户，采用 predict-then-assess 的 CoT 推理范式进行可解释的个性化图像偏好评估，结合冷启动 SFT + GRPO 强化学习及 similarity-aware 预测奖励，7B 模型超越 GPT-4o（+5.2%）和 Claude 3.7（+5.1%）。

## 背景与动机
- **个性化偏好评估面临两大难题**：
  1. 每个用户的个性化数据极为稀少且不可大规模扩展，不同于可共享评价标准的通用偏好数据
  2. 个性化偏好跨越多个维度（艺术风格、色彩、介质等），复杂且多样
- **CLIP-based 方法**（PickScore、ImageReward 等）：依赖大规模通用偏好数据训练，无法处理个性化场景，且仅输出数值分数缺乏可解释性
- **MLLM-based 方法**（UnifiedReward 等）：需要大量 VQA pairs 微调，个性化图像数量不足以支撑
- **ViPer**：现有唯一的个性化方法，但仅隐式利用参考图像做分数回归，缺乏可解释推理步骤
- **核心 insight**：虽然每个用户偏好独特，但构成偏好的基本视觉元素（art style、color、detail、art medium、saturation）是通用的，可作为跨用户的桥梁

## 方法详解

### 整体框架：Predict-then-Assess 范式
给定用户的个性化参考图像（喜欢/不喜欢）和两张候选图像，PreferThinker 通过两阶段 CoT 推理：
1. **Profile Prediction**：根据参考图像预测用户的视觉偏好画像和非偏好画像
2. **Multi-dimensional Assessment**：基于预测画像对候选图像进行多维可解释评分，得出最终结果

### 关键设计 1：视觉偏好画像（Visual Preference Profile）
- 从 Lexica 平台的文本提示词中识别 15 个最常见的视觉元素
- 100 人用户研究投票选出 top-5：**art style、color、detail、art medium、saturation**
- 收集 288 个相关词汇确保画像多样性
- 画像的三大优势：描述复杂偏好、跨用户知识共享、支持可解释多维评估

### 关键设计 2：PreferImg-CoT 大规模数据集
- **PreferImg 构建**：80K 模拟用户（含 20K 多偏好用户），1.36M 图像
    - 随机采样 5 个视觉偏好元素分配画像
    - 使用 T2I 模型生成参考图像和候选图像
    - 190K 初始 prompt 覆盖 Lexica、DiffusionDB、COCO
- **CoT 标注**：Claude 3.7 生成 predict-then-assess 格式的推理链
- **质量过滤**：去除逻辑不一致或答案不匹配的样本
- 最终得到 60K 高质量 CoT 样本

### 关键设计 3：两阶段训练 + Similarity-aware 预测奖励
**Stage 1 - 冷启动 SFT**：
- 基座模型 Qwen2.5-VL-7B
- 标准自回归交叉熵损失：$\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{CoT}}\sum_{t=1}^{T}\log P(y_t|x,y_{<t};\theta)$

**Stage 2 - GRPO 强化学习**：
- 每个输入生成 G 个 CoT 输出，计算组内归一化优势 $A_i$
- PPO-clip 目标 + KL 散度正则

**Similarity-aware Prediction Reward**：
- **文本相似度**：SBERT 计算预测画像与 GT 画像的语义相似度 $s_{text}$
- **图像相似度**：基于预测/GT 画像分别生成图像，DreamSim 计算视觉相似度 $s_{img}$
- 预测奖励 $r_{predict} = w_{img}s_{img} + w_{text}s_{text}$
- 混合奖励：$r = w_p r_{predict} + w_f r_{format} + w_a r_{accuracy}$（权重 0.7/0.3/1.0）

## 实验

### 主实验结果（评估准确率，%）

| 方法 | 参数量 | PreferImg Seen-SP | Seen-MP | Unseen-SP | Unseen-MP | PickaPic | 平均 |
|------|--------|----------|---------|-----------|-----------|---------|------|
| PickScore | 986M | 49.6 | 48.4 | 51.2 | 56.4 | 67.9 | 54.7 |
| ViPer | 8B | 92.4 | 78.0 | 93.4 | 80.0 | 62.2 | 81.2 |
| GPT-4o | - | 94.2 | 80.4 | 92.2 | 85.2 | 65.7 | 83.5 |
| Claude 3.7 | - | 93.8 | 83.2 | 90.2 | 86.0 | 64.9 | 83.6 |
| **PreferThinker** | **7B** | **96.6** | **92.0** | **96.4** | **92.8** | 65.7 | **88.7** |

### 消融实验

| 配置 | Seen-SP Acc | Seen-SP Pred | Unseen-MP Acc | Unseen-MP Pred |
|------|-------------|---------|------------|---------|
| Base (Qwen2.5-VL-7B) | 75.4 | 70.4 | 64.8 | 71.1 |
| + SFT | 92.0 | 84.2 | 81.6 | 74.2 |
| + SFT + RL | 93.8 | 85.0 | 88.4 | 79.5 |
| + SFT + RL + PR (完整) | **96.6** | **87.5** | **92.8** | **83.1** |

### 关键发现
1. **7B 模型超越所有闭源模型**：PreferThinker 在 PreferImg 上全面超越 GPT-4o 和 Claude 3.7
2. **多偏好（MP）场景改进最显著**：相比 SOTA 提升 +8.8%（Seen-MP），说明 profile 机制有效应对复杂偏好
3. **RL 阶段显著增强泛化性**：RL 在 unseen 用户上的提升（+6.8%）大于 seen 用户（+4.6%）
4. **预测奖励是关键**：画像预测越准确，后续评估越合理（无 PR 时预测准确性下降→评估错误）
5. **个性化画像可迁移到图像生成**：预测的偏好画像可引导个性化图像生成

## 亮点
- 提出了连接不同用户的偏好画像（preference profile）概念，优雅地解决个性化数据稀缺问题
- Predict-then-assess 范式实现了可解释的多维评估，不再是黑盒打分
- Similarity-aware prediction reward 设计巧妙，同时利用文本和图像空间的相似度信号
- 7B 开源模型超越 GPT-4o 和 Claude 3.7 等商业模型

## 局限性
- PreferImg 数据集基于模拟用户（T2I 生成），与真实用户偏好分布可能存在差异
- 在 PickaPic 真实用户数据集上表现一般（65.7%），因为 PickaPic 标注的是通用偏好而非个性化偏好
- 画像的 5 个视觉元素固定，可能不覆盖所有个性化维度（如构图、情感）
- 训练需要 T2I 模型生成图像来计算图像相似度奖励，训练成本较高

## 相关工作
- **图像偏好评估**：CLIP-based（PickScore、ImageReward、HPSv2）→ MLLM-based（UnifiedReward、LLaVA-Reward）
- **个性化偏好**：ViPer（ECCV2024）首次尝试，但缺乏可解释性
- **推理型 MLLM**：DeepSeek-R1 启发的 GRPO 后训练范式
- **偏好数据集**：ImageRewardDB、PickaPic、HPD_v2 主要面向通用偏好

## 评分
⭐⭐⭐⭐ (4/5)

方法设计完整，从数据构建到训练都有创新点。偏好画像桥接概念简洁有效。主要的担忧是模拟数据与真实个性化偏好之间的 gap，PickaPic 上的表现也证实了这一点。

<!-- RELATED:START -->

## 相关论文

- [Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)
- [DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)
- [Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](../../CVPR2026/reinforcement_learning/reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)
- [Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)

<!-- RELATED:END -->
