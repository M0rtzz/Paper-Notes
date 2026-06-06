---
title: >-
  [论文解读] Bias at the End of the Score: Demographic Biases in Reward Models for T2I
description: >-
  [CVPR 2026][图像生成][奖励模型] 对文本到图像生成中广泛使用的奖励模型（PickScore、ImageReward、HPS 等）进行大规模人口统计偏差审计，发现奖励引导优化会不成比例地性化女性图像、使人口统计收敛到白人、且奖励分数与现实世界的人口频率先验相关。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "奖励模型"
  - "文本到图像"
  - "人口统计偏差"
  - "超性化"
  - "公平性"
---

# Bias at the End of the Score: Demographic Biases in Reward Models for T2I

**会议**: CVPR 2026  
**arXiv**: [2604.13305](https://arxiv.org/abs/2604.13305)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 奖励模型, 文本到图像, 人口统计偏差, 超性化, 公平性

## 一句话总结

对文本到图像生成中广泛使用的奖励模型（PickScore、ImageReward、HPS 等）进行大规模人口统计偏差审计，发现奖励引导优化会不成比例地性化女性图像、使人口统计收敛到白人、且奖励分数与现实世界的人口频率先验相关。

## 研究背景与动机

**领域现状**：奖励模型（RM）在 T2I 管线中无处不在——数据集过滤、评估指标、参数优化的监督信号和生成后过滤。PickScore、ImageReward、HPS 等基于人类偏好数据训练。

**现有痛点**：RM 被设计和部署为"质量度量"，但其在人口统计偏差方面的鲁棒性和公平性几乎未被研究。训练 RM 的数据、人类偏好和模型归纳偏差都可能注入偏差。

**核心矛盾**：RM 作为"质量"代理被广泛使用，但它们可能隐式编码了人口统计偏差，导致偏差通过 T2I 管线的各个环节指数级放大。

**本文目标**：系统审计 RM 在微调和评估中的人口统计偏差行为。

**切入角度**：使用 ReNO 框架进行奖励引导优化，观察优化前后图像的人口统计变化；使用反事实数据集分析评分层面的偏差。

**核心 idea**：RM 不仅评估图像质量，还隐式奖励符合其训练数据中主导人口统计特征的图像。

## 方法详解

### 整体框架

两部分分析：Part I（优化实验）使用 ReNO 框架的奖励引导噪声优化，观察 RM 如何改变生成图像的人口统计属性和性化程度。Part II（评分实验）使用三个反事实数据集（CausalFace、SocialCounterfactuals、PAIRS），通过线性回归和排名分析检验 RM 评分的系统性人口偏差。

### 关键设计

1. **奖励引导优化实验 (Part I)**:

    - 功能：揭示 RM 梯度对图像人口统计属性的系统性影响
    - 核心思路：使用 ReNO 框架 $\varepsilon^* = \arg\max_\varepsilon R(G_\theta(\varepsilon, p), p)$ 优化初始噪声向量。测量优化前后的 NSFW 分类率、皮肤暴露度、人口统计分类器输出的变化。使用包含和不包含人口标识符的两种 prompt 集
    - 设计动机：如果 RM 是中立的质量度量，优化不应系统性地改变人口统计属性或增加性化内容

2. **反事实评分分析 (Part II)**:

    - 功能：直接检验 RM 的评分是否因人口属性而系统性不同
    - 核心思路：使用仅在人口属性（种族、性别）上不同的匹配图像集。OLS 回归：$s^R_{I,p} \approx \beta_0 + \beta_1 \rho_I + \beta_2 \gamma_I + \beta_3(\rho_I \times \gamma_I) + \epsilon_I$。统计显著的系数表明 RM 对特定人口属性有系统性偏好。排名分析补充相对偏好排序
    - 设计动机：反事实设计控制了除人口属性外的所有变量，分离出纯粹的人口偏差

3. **现实频率相关性分析**:

    - 功能：揭示 RM 编码了现实世界的人口分布先验
    - 核心思路：将 RM 对职业 prompt 的评分与美国劳工统计局报告的各职业女性就业比例进行相关性分析
    - 设计动机：如果 RM 仅评估质量，分数不应与现实世界的人口频率相关

### 损失函数 / 训练策略

本文是审计/分析论文，不涉及新模型训练。使用 ReNO 的默认超参数进行优化实验。评分归一化到零均值和单位方差以确保跨模型可比性。

## 实验关键数据

### 主实验

| 发现 | RM | 效应量 |
|------|-----|-------|
| 超性化放大 | PickScore | 女性 NSFW 率增加 19% vs 男性 7% (2.7×) |
| 人口收敛 | ImageReward/HPS | >80% 黑人图像优化后被分类为白人 |
| 性别翻转 | ImageReward | 39% 女性图像优化后被分类为男性 |
| 种族评分偏差 | HPS/ImageReward | 白人图像系统性获得最高评分 |
| VQAScore 反转 | VQAScore | 正面 prompt 偏好白人，负面 prompt 偏好黑人 |

### 消融实验

| RM | 白人排名 | 黑人排名 | 差距 |
|----|---------|---------|------|
| HPS | 1.2 | 3.1 | 最大偏差 |
| ImageReward | 1.4 | 2.8 | 显著偏差 |
| CLIP | 2.5 | 3.5 | 黑人始终最低 |
| PickScore | 1.8 | 2.3 | 中等偏差 |

### 关键发现

- PickScore 的超性化效应最强：女性受影响程度是男性的 2.7 倍
- ImageReward 和 HPS 导致最严重的人口收敛：超过 80% 的黑人图像优化后被分类为白人
- RM 评分与美国职业性别比例显著相关，说明 RM 学到了现实世界的频率先验
- VQAScore 表现出"刻板印象强化"模式：正面描述偏好白人，负面描述偏好黑人

## 亮点与洞察

- 这是对 T2I 奖励模型最系统的公平性审计：揭示了 RM 远不是中立的质量度量
- "人口收敛"现象（优化使多样化图像收敛到白人）的发现非常重要：说明 RM 可能成为多样性的敌人
- RM 编码的不是"质量"而是"主导人口符合度"的结论对 RM 的设计和使用有深远影响

## 局限与展望

- 仅使用 ReNO 一种优化方法，其他优化策略可能有不同行为
- 依赖自动分类器判断人口属性，存在测量噪声
- 未深入分析偏差的来源（训练数据 vs 标注者偏好 vs 架构）
- 需要开发去偏差的 RM 训练方法

## 相关工作与启发

- **vs Concept2Concept**: C2C 发现 Pick-a-Pic 数据集包含 CSAM，本文聚焦 RM 的系统性人口偏差
- **vs T2I 公平性研究**: 之前的研究关注生成模型本身的偏差，本文揭示 RM 作为评估和优化工具的偏差同样严重

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性审计 T2I RM 的人口偏差
- 实验充分度: ⭐⭐⭐⭐⭐ 5个RM×3个反事实数据集×多种分析
- 写作质量: ⭐⭐⭐⭐ 发现阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 对 AI 安全和公平性有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models](autodebias_automated_framework_for_debiasing_text-to-image_models.md)
- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](dcw_snr_t_bias_diffusion.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[NeurIPS 2025\] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models](../../NeurIPS2025/image_generation/fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)

</div>

<!-- RELATED:END -->
