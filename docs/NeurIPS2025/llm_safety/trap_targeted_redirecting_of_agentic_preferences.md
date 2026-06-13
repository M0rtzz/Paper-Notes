---
title: >-
  [论文解读] TRAP: Targeted Redirecting of Agentic Preferences
description: >-
  [NeurIPS 2025][LLM安全][对抗攻击] TRAP 提出了一种基于扩散模型的语义注入对抗框架，通过在 CLIP 嵌入空间中优化图像语义，在黑盒条件下以视觉自然的方式系统性地误导多个主流 VLM 智能体的决策偏好，在 LLaVA-34B、GPT-4o 等六个模型上实现了高达 100% 的攻击成功率。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "对抗攻击"
  - "视觉语言模型"
  - "语义注入"
  - "智能体安全"
  - "扩散模型"
---

# TRAP: Targeted Redirecting of Agentic Preferences

**会议**: NeurIPS 2025  
**arXiv**: [2505.23518](https://arxiv.org/abs/2505.23518)  
**代码**: [https://github.com/uiuc-focal-lab/TRAP](https://github.com/uiuc-focal-lab/TRAP)  
**领域**: AI安全  
**关键词**: 对抗攻击, 视觉语言模型, 语义注入, 智能体安全, 扩散模型

## 一句话总结

TRAP 提出了一种基于扩散模型的语义注入对抗框架，通过在 CLIP 嵌入空间中优化图像语义，在黑盒条件下以视觉自然的方式系统性地误导多个主流 VLM 智能体的决策偏好，在 LLaVA-34B、GPT-4o 等六个模型上实现了高达 100% 的攻击成功率。

## 研究背景与动机

随着基于视觉语言模型（VLM）的自主智能体系统逐步走向现实部署，其跨模态推理能力引入了新的攻击面。现有对抗攻击方法主要依赖可见的像素扰动或需要对模型/环境的特权访问，这在现实中既不隐蔽也不实用。传统像素级攻击（如 FGSM、PGD）注重低层噪声注入，但现代多模态系统对像素噪声具有一定鲁棒性，而真正的脆弱点在于语义层面的跨模态对齐。

核心矛盾在于：自主智能体天然信任其感知输入，而这种信任可以被语义层面的微妙操纵所利用。本文的切入角度是：利用扩散模型在 CLIP 共享嵌入空间中进行语义优化，生成视觉自然但语义经过操纵的对抗图像，从而在黑盒条件下重定向智能体的选择偏好。

核心 idea：结合负提示降质和正向语义优化，通过孪生语义网络和空间布局掩码，在嵌入空间内进行语义级对抗操纵。

## 方法详解

### 整体框架

TRAP 分四个阶段运作：(1) 提取目标图像和对抗提示的 CLIP 嵌入；(2) 利用孪生语义网络和提示对齐引导，配合空间布局掩码，在嵌入空间中迭代优化图像嵌入；(3) 应用感知损失和语义损失以保持图像的身份和真实感；(4) 通过 Stable Diffusion 解码器将优化后的嵌入解码为最终对抗图像。

整个过程完全在黑盒条件下运作——攻击者无需访问目标模型的权重、参数或梯度，只能替换自己控制的图像并观察智能体的最终选择。

### 关键设计

1. **语义对齐损失 (ℒ_sem)**：通过最小化对抗嵌入 e_adv 与正向提示嵌入 e_pos 之间的余弦距离，将高层语义概念注入图像表示。这利用了 CLIP 的联合嵌入空间——语义相近的内容嵌入距离更近。攻击者选择的正向提示（如 "luxury"、"premium quality"）作为语义代理，对一系列用户查询具有泛化性。

2. **独特特征保持损失 (ℒ_dist) + 孪生语义网络**：仅优化语义对齐会导致图像丧失独特身份。孪生网络将嵌入分解为"共同成分"和"独特成分"两个分支。通过惩罚独特分支的变化，迫使优化器将语义修改集中在共同分支上，从而在注入语义的同时保持图像身份。这种"推-拉"动态是方法的核心创新之一。

3. **感知相似度损失 (ℒ_LPIPS) + 空间布局掩码**：为确保解码图像视觉合理，使用可微的解码管道：首先利用轻量 MLP 编解码器从提示和图像嵌入生成语义布局掩码 A，再与 DeepLabv3 分割模型的前景掩码 F_seg 相乘得到精细化掩码 A_final。这确保了语义编辑仅限于图像主体区域，通过 LPIPS 约束解码图像与原图的感知距离。

### 损失函数 / 训练策略

总损失为三项加权和：

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{sem} + \lambda_2 \mathcal{L}_{dist} + \lambda_3 \mathcal{L}_{LPIPS}$$

优化使用 Adam 优化器（学习率 0.005），每次迭代包含 K=20 外循环和 T=20 内步梯度下降。对扩散强度 [0.3, 0.8] 和 CFG [2.0, 12.0] 进行网格搜索。优化变量仅为 e_adv，嵌入空间优化而非像素空间优化是核心策略选择。

## 实验关键数据

### 主实验

在 COCO 数据集上的 100 个图像-描述对上评估，模拟黑盒 N-way 选择场景。

| 方法 | LLaVA-34B | Gemma3-8B | Mistral-3.1-24B | Mistral-3.2-24B | GPT-4o | CogVLM |
|------|-----------|-----------|-----------------|-----------------|--------|--------|
| 初始坏图像 | 21% | 17% | 14% | 6% | 0% | 8% |
| SPSA | 36% | 27% | 22% | 11% | 1% | 18% |
| Bandit | 6% | 2% | 1% | 0% | 0% | 0% |
| SSA_CWA | 65% | 42% | 28% | 18% | 8% | 4% |
| SA_AET | 85% | 67% | 61% | 55% | 12% | 42% |
| **TRAP** | **100%** | **100%** | **100%** | **99%** | **63%** | **94%** |

### 防御鲁棒性

| 方法 | LLaVA-34B | Gemma3 | Mistral-3.1 | Mistral-3.2 | Robust-LLaVA |
|------|-----------|--------|-------------|-------------|-------------|
| TRAP | 100% | 100% | 100% | 97% | 92% |
| TRAP + 高斯噪声 | 100% | 100% | 100% | 96% | 92% |
| TRAP + CIDER | 100% | 100% | 96% | 90% | 85% |
| TRAP + MirrorCheck | 100% | 98% | 88% | 82% | 74% |

### 消融实验

| 配置变化 | LLaVA-34B | Gemma3 | Mistral-3.1 | Mistral-3.2 |
|---------|-----------|--------|-------------|-------------|
| Distinctive Loss 0.3→0.8 | 88% | 70% | 72% | 65% |
| Semantic Loss 0.5→0.0 | 90% | 82% | 77% | 70% |
| Perceptual Loss 1.0→1.5 | 100% | 100% | 100% | 98% |

### 关键发现

- TRAP 在所有六个评估模型上均大幅超越所有基线，包括开源和闭源模型
- 攻击可迁移到非对比式架构（CogVLM）和完全闭源的 GPT-4o
- 对系统提示变体的鲁棒性很强，ASR 偏差仅在低个位数范围
- 在不同采样温度（T=0.1 和 T=0.7）下攻击效果稳定
- 过度强调独特性损失会损害跨模型迁移性
- 移除语义损失项性能下降最显著（ASR 降至 70-90%）

## 亮点与洞察

- 首次系统性地证明了语义级跨模态操纵对自主智能体的威胁，超越了传统像素级攻击范式
- 嵌入空间优化策略巧妙——不直接修改像素，而是在 CLIP 空间中操作高层语义，实现模型无关的攻击迁移
- 孪生网络的"共同-独特"分解是关键创新，解决了语义注入与身份保持之间的矛盾
- 布局感知掩码设计使语义编辑精确限制在前景区域，增强了攻击的隐蔽性
- 黑盒威胁模型的设定现实且有意义——攻击者只能控制自己的图像，不需要环境或模型访问

## 局限与展望

- 假设智能体依赖对比式视觉语言相似度评分，未来非对比架构可能减弱攻击效果
- 攻击成功依赖辅助组件（布局掩码、扩散模型）的质量，在边缘案例上可能退化
- 计算成本较高（每样本约 520 秒），是像素级攻击的数倍，实时场景下的可扩展性仍是挑战
- 未深入探讨有效的防御策略，仅展示了现有防御的不足
- 电商网页场景下成功率显著下降（51%），提示真实部署场景中的挑战更大

## 相关工作与启发

- 与传统像素级攻击（FGSM、PGD、C&W）相比，TRAP 在语义层面操作，更隐蔽且更难检测
- 与已有扩散模型语义操纵工作（AdvDiff、Instruct2Attack）不同，TRAP 仅使用模型嵌入，无需访问扩散模型参数
- 启发方向：需要开发嵌入空间级防御和语义级鲁棒性标准，而非仅依赖像素空间鲁棒性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Reverse Engineering Human Preferences with Reinforcement Learning](reverse_engineering_human_preferences_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory](llm_strategic_reasoning_agentic_study_through_behavioral_gam.md)
- [\[ICML 2025\] Targeted Unlearning with Single Layer Unlearning Gradient](../../ICML2025/llm_safety/targeted_unlearning_with_single_layer_unlearning_gradient.md)
- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](../../AAAI2026/llm_safety/the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)
- [\[AAAI 2026\] LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users](../../AAAI2026/llm_safety/llm_targeted_underperformance_disproportionately_impacts_vulnerable_users.md)

</div>

<!-- RELATED:END -->
