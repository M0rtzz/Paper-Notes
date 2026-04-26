---
title: >-
  [论文解读] Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards
description: >-
  [CVPR 2026][图像生成][自信度奖励] 提出 SOLACE，一种利用文本-图像生成模型自身去噪自信度作为内在奖励的后训练框架，无需外部奖励模型即可在组合生成、文字渲染和文图对齐上获得一致提升，且可与外部奖励互补缓解 reward hacking。
tags:
  - CVPR 2026
  - 图像生成
  - 自信度奖励
  - 后训练
  - Flow Matching
  - GRPO
  - 文本到图像
---

# Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards

**会议**: CVPR 2026  
**arXiv**: [2603.00918](https://arxiv.org/abs/2603.00918)  
**代码**: [项目页面](https://wookiekim.github.io/SOLACE/)  
**领域**: 图像生成 / 扩散模型后训练  
**关键词**: 自信度奖励, 后训练, Flow Matching, GRPO, 文本到图像

## 一句话总结

提出 SOLACE，一种利用文本-图像生成模型自身去噪自信度作为内在奖励的后训练框架，无需外部奖励模型即可在组合生成、文字渲染和文图对齐上获得一致提升，且可与外部奖励互补缓解 reward hacking。

## 研究背景与动机

文本到图像 (T2I) 生成模型的后训练（通过强化学习优化外部奖励）已成为提升图像质量的有效范式。然而现有方法存在三大痛点：

**奖励定义困难**：好图像需满足组合性、文字渲染、美学、文图对齐等多个弱对齐标准，它们的重要性因场景而异

**Reward hacking**：优化单一外部奖励容易导致过拟合，在非目标能力上退化（如 PickScore 上升但组合能力下降）

**运维成本高**：外部奖励需在训练时运行额外评估器（偏好/OCR/安全模型），增加流程复杂度

本文的核心洞察是：**大规模预训练赋予了扩散模型关于真实图像和文图对齐的先验**，模型对自身生成结果的"自信程度"本身就是一个有意义的奖励信号。受 Score Distillation Sampling 启发，让模型"自评"自己的生成——如果模型能准确恢复注入其生成结果中的噪声，说明它对该结果很有信心。

## 方法详解

### 整体框架

SOLACE 基于 Flow-GRPO 框架。给定文本提示 $c$，采样 $G$ 张图像的潜变量 $\{z_0^{(i)}\}_{i=1}^G$；对每个潜变量进行**重新加噪**（re-noising），然后测量模型恢复注入噪声的能力，将恢复精度转化为标量奖励；最后通过 GRPO 策略梯度优化生成模型。整个过程在潜空间完成，无需解码。

### 关键设计

1. **自信度奖励计算 (Self-Confidence Reward)**：对生成的潜变量 $z_0^{(i)}$，使用 $K$ 个共享噪声探针 $\epsilon^{(m)} \sim \mathcal{N}(0, I)$ 进行重新加噪：

$$z_t^{(i,m)} = (1-t) z_0^{(i)} + t \epsilon^{(m)}, \quad t \in \mathcal{T}$$

然后用模型速度场恢复噪声估计 $\hat{\epsilon}_\theta = v_\theta(z_t^{(i,m)}, t, c) + z_0^{(i)}$，计算恢复误差：

$$\text{MSE}_{i,t} = \frac{1}{K} \sum_{m=1}^K \|\hat{\epsilon}_\theta(z_t^{(i,m)}, t, c) - \epsilon^{(m)}\|_2^2$$

通过负对数变换得到自信度分数 $S_{i,t} = -\log(\text{MSE}_{i,t} + \delta)$，最终加权聚合为标量奖励 $R_{\text{SOLACE}}$。设计动机：小误差=高自信=高奖励，负对数变换压缩异常值并近似高斯对数似然。使用**反对噪声对** $(\epsilon^{(m+K/2)} = -\epsilon^{(m)})$ 保证探针均值为零。

2. **训练稳定化技术**：

    - **后缀时间步训练**：仅在去噪轨迹的后 $\rho$ 比例时间步上优化 GRPO 损失（$|\mathcal{T}_{\text{train}}| = \lceil \rho |\mathcal{T}| \rceil$），避免在早期时间步上过度优化导致崩溃（生成无纹理空白图像）
    - **无 CFG 自信度计算**：采样用 CFG，但自信度计算不用 CFG，因为 CFG 混合场会让自信度评估的是引导代理而非基础条件模型
    - **在线自信度**：用正在训练的模型 $\pi_\theta$（而非固定基础模型 $\pi_{\text{ref}}$）计算自信度，随模型改进获得更强信号

3. **与外部奖励的互补集成**：SOLACE 可在外部奖励后训练（如 FlowGRPO + PickScore）之后叠加使用。外部奖励优化的目标维度与内在自信度关注的维度不同（美学 vs. 组合性/文字/对齐），两者互补可缓解 reward hacking。

### 损失函数 / 训练策略

采用 Flow-GRPO 目标函数，优势函数使用组内归一化：

$$\hat{A}_t^i = \frac{R(z_0^i, c) - \text{mean}(\{R(z_0^i, c)\}_{i=1}^G)}{\text{std}(\{R(z_0^i, c)\}_{i=1}^G)}$$

训练使用 10 步去噪（SD3.5 推理用 40 步），配合 clipped 重要性采样比率和 KL 正则化。

## 实验关键数据

### 主实验

SD3.5-Medium 上 SOLACE 后训练效果：

| 指标 | SD3.5-M (基线) | + SOLACE | 提升 | 说明 |
|------|-------------|----------|------|------|
| GenEval (组合) | 0.65 | **0.71** | +0.06 | 组合生成能力显著提升 |
| OCR (文字渲染) | 0.61 | **0.67** | +0.06 | 文字渲染能力显著提升 |
| CLIPScore (对齐) | 0.282 | **0.288** | +0.006 | 文图对齐提升 |
| Aesthetic | 5.36 | 5.39 | +0.03 | 美学质量小幅提升 |
| PickScore | 22.34 | 22.41 | +0.07 | 人类偏好小幅提升 |

SOLACE 叠加外部奖励（FlowGRPO + GenEval 奖励 + SOLACE）：GenEval 达到 0.95→维持高水平，同时 OCR 从 0.65 回升。

### 消融实验

| 配置 | GenEval | OCR | CLIPScore | 说明 |
|------|---------|-----|-----------|------|
| K=4 噪声探针 | 0.69 | 0.66 | 0.286 | 探针数偏少 |
| **K=8 噪声探针** | **0.71** | **0.67** | **0.288** | **最优配置** |
| K=16 噪声探针 | 0.70 | 0.66 | 0.287 | 收益递减，成本增加 |
| 有 CFG 自信度 | 0.69 | 0.65 | 0.285 | CFG 引入代理偏差 |
| Offline 自信度 | 0.68 | 0.65 | 0.282 | 静态奖励次优 |
| **Online 自信度** | **0.71** | **0.67** | **0.288** | **自适应奖励更优** |

### 关键发现

- 内在自信度与组合生成、文字渲染、文图对齐强相关，但与人类偏好弱相关
- 训练时间步过多（$\rho > 0.6$）或采样不用 CFG 会导致训练崩溃（reward hacking → 无纹理图像）
- SOLACE 与外部奖励互补：叠加使用可缓解外部奖励的 reward hacking 问题
- 用户研究（1800 响应/20 人）证实 SOLACE 在视觉真实感和文图对齐上均优于基线

## 亮点与洞察

1. **自我评估范式**：首次将扩散/流模型的去噪自信度形式化为内在奖励信号，提供了一种"模型自评"的全新后训练范式
2. **无需外部依赖**：不需要额外的奖励模型、标注数据或评估器，降低了后训练的资源门槛和流程复杂度
3. **缓解 Reward Hacking**：与外部奖励叠加使用时，SOLACE 作为正则化力量恢复组合性和文字渲染能力，具有很好的互补性
4. **潜空间直接计算**：自信度奖励完全在潜空间计算，无需解码到像素空间，计算高效

## 局限与展望

1. 内在自信度与人类偏好的相关性较弱，无法单独替代人类偏好奖励
2. 无法针对特定对齐目标（如安全性）进行定向优化
3. 训练稳定性对时间步选择敏感（$\rho$ 需仔细调节），可能需要更鲁棒的自信度估计方法
4. 仅在 SD3.5 上验证，对其他架构（如 DiT、自回归模型）的适用性待探索
5. 未来可扩展到视频/3D 生成中的时序/多视角一致性评估

## 相关工作与启发

- **Flow-GRPO**：本文的基础 RL 后训练框架，将 GRPO 应用于 Flow Matching 模型
- **Score Distillation Sampling (SDS)**：启发了"用生成模型评价自身生成"的核心思路
- **Intuitor (LLM 领域)**：LLM 中使用自信度作为内在奖励的先驱，本文将其拓展到连续去噪轨迹
- **启发**：自信度信号可能在其他生成任务（视频、3D、音频）中同样有效——去噪能力强 ≈ 对生成质量有信心的直觉具有通用性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将扩散模型的去噪自信度形式化为内在奖励是非常原创的idea，打开了自监督后训练的新方向
- 实验充分度: ⭐⭐⭐⭐ 多指标评估 + 用户研究 + 消融分析 + 与外部奖励叠加实验，验证充分
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，动机阐述有说服力，结构组织良好
- 价值: ⭐⭐⭐⭐⭐ 提供了一种零成本内在奖励信号，可广泛应用于 T2I 后训练，且与外部奖励互补

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Self-Corrected Image Generation with Explainable Latent Rewards](self-corrected_image_generation_with_explainable_latent_rewards.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] Disentangling to Re-couple: Resolving the Similarity-Controllability Paradox in Subject-Driven Text-to-Image Generation](disentangling_to_re-couple_resolving_the_similarity-controllability_paradox_in_s.md)

<!-- RELATED:END -->
