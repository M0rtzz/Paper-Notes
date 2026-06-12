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

这篇论文不提新模型，而是给一类被默认当作"中立质量尺子"的奖励模型（RM）做体检：当我们用 PickScore、ImageReward、HPS 这些 RM 去引导 T2I 生成时，它们到底在悄悄奖励什么样的人？整篇分析分两条腿走。一条是**优化端**（Part I）：把 RM 当成可微的优化目标，看它沿梯度往上爬的过程会不会系统性地改变图像里人的种族、性别和性化程度——如果一把尺子是中立的，往它的高分方向优化不该让黑人变白、让女性变得更暴露。另一条是**评分端**（Part II）：抛开优化，直接拿"只在人口属性上不同、其它都一样"的反事实图像喂给 RM，看它给的分数本身会不会因肤色或性别而系统性地高低不等；评分端还要再追一步，把这种"分数偏差"和现实世界的人口频率先验对上号，说清偏到底从何而来。两条腿一起，把偏差既归因到"优化怎么放大"，也归因到"评分本身就偏、且偏向主导人口分布"。

### 关键设计

**1. 奖励引导优化实验（Part I）：让 RM 自己暴露它偏爱谁**

直接质问 RM"你公平吗"是问不出来的，所以作者改用 ReNO 框架把 RM 推到极致：固定生成器 $G_\theta$，只优化初始噪声向量去最大化奖励，

$$\varepsilon^* = \arg\max_\varepsilon\; R\big(G_\theta(\varepsilon, p), p\big),$$

然后对比优化前后同一张图的变化。测量的不是美学分，而是一组人口统计信号——NSFW 分类率、皮肤暴露面积、人口属性分类器的输出。prompt 分成"带人口标识符"和"不带"两套，用来分辨偏差是 prompt 里写明的还是 RM 自己脑补的。逻辑很硬：一把真正中立的尺子，往它的高分方向爬不该改变图里人的种族或性别，更不该单方面增加女性的性化内容；一旦优化后这些指标系统性漂移，漂移的方向就是 RM 的偏好方向。

**2. 反事实评分分析（Part II）：把"评分偏差"从混杂因素里剥出来**

优化实验能看到偏差被放大，但放大可能来自生成器而非 RM。为了把账算到 RM 头上，作者用三个反事实数据集（CausalFace、SocialCounterfactuals、PAIRS）构造"配对图像"——除了种族 $\rho_I$ 和性别 $\gamma_I$，其它一切（姿态、构图、光照）都保持一致。在这些配对上对 RM 分数 $s^R_{I,p}$ 做 OLS 回归，

$$s^R_{I,p} \approx \beta_0 + \beta_1 \rho_I + \beta_2 \gamma_I + \beta_3(\rho_I \times \gamma_I) + \epsilon_I,$$

只要 $\beta_1, \beta_2$ 或交互项 $\beta_3$ 统计显著，就说明在控制掉一切其它变量后，RM 仍然只因人口属性就给出不同的分。回归之外再补一层排名分析，看同一组人不同肤色版本在 RM 眼里谁排前谁排后，给出相对偏好的直观排序。反事实设计的价值正在于此：它把"图像质量"这个混杂变量摁住，剩下的分数差只能归因于人口属性本身。

**3. 现实频率相关性分析：检验 RM 是不是把"人口分布"误当成了"质量"**

前两个设计证明 RM 有偏，这个设计追问偏从何来。作者把 RM 对各类职业 prompt 的评分，和美国劳工统计局公布的各职业真实女性就业比例做相关性分析。推理同样干净：如果 RM 评的纯粹是图像质量，它的分数没理由和现实世界里"哪个职业女性多"这种社会统计量相关；可一旦两者显著相关，就说明 RM 实际上在奖励"符合训练数据里主导人口分布"的图像——它把世界本来的样子记成了"好的样子"，质量度量退化成了人口频率先验的复读机。

### 损失函数 / 训练策略

本文是审计/分析论文，不训练新模型。优化实验沿用 ReNO 的默认超参数；为了让 PickScore、ImageReward、HPS 等量纲不同的 RM 能横向比较，所有评分先归一化到零均值、单位方差再做回归与排名。

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
