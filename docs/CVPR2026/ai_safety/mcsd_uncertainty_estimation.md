---
title: >-
  [论文解读] Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning
description: >-
  [CVPR 2026][AI安全][不确定性量化] 将随机深度（Stochastic Depth）正式连接到贝叶斯变分推理框架，提出 Monte Carlo Stochastic Depth (MCSD) 作为不确定性估计方法，并在 YOLO、RT-DETR 等现代检测器上进行首次系统基准测试…
tags:
  - "CVPR 2026"
  - "AI安全"
  - "不确定性量化"
  - "随机深度"
  - "贝叶斯推理"
  - "目标检测"
  - "Monte Carlo"
---

# Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning

**会议**: CVPR 2026  
**arXiv**: [2604.12719](https://arxiv.org/abs/2604.12719)  
**代码**: 无  
**领域**: AI安全 / 不确定性估计  
**关键词**: 不确定性量化, 随机深度, 贝叶斯推理, 目标检测, Monte Carlo

## 一句话总结

将随机深度（Stochastic Depth）正式连接到贝叶斯变分推理框架，提出 Monte Carlo Stochastic Depth (MCSD) 作为不确定性估计方法，并在 YOLO、RT-DETR 等现代检测器上进行首次系统基准测试，证明其在校准和不确定性排名上与 MC Dropout 竞争力强。

## 研究背景与动机

**领域现状**：安全关键系统中 DNN 需要可靠的不确定性量化。Monte Carlo Dropout (MCD) 将 dropout 重新解释为近似贝叶斯推理，成为主流实用方法。MC DropBlock (MCDB) 将该范式扩展到卷积层。

**现有痛点**：标准 dropout 在卷积层效果不佳，而随机深度（SD）是残差网络的原生正则化技术，被 YOLO 和 ViT 等现代架构广泛使用，但将其用于推理时采样的理论基础和系统实证验证都缺失。

**核心矛盾**：SD 作为正则化器与贝叶斯变分推理的正式理论联系尚未建立，且其在目标检测等复杂多任务问题上的 UQ 性能未知。

**本文目标**：(1) 建立 MCSD 与变分推理的理论联系；(2) 首次在目标检测上系统基准测试 MCSD。

**切入角度**：从 MCD 到 MCDB 的进展揭示了一个元策略：随机正则化器隐式定义近似后验分布。SD 是下一个自然候选。

**核心 idea**：推理时保持随机深度的随机性，通过 T 次随机前向传播采样不同深度的子网络，形成隐式深度集成来估计不确定性。

## 方法详解

### 整体框架

MCSD 在标准残差网络上操作：推理时对每个残差块独立采样 $b_l \sim \text{Bernoulli}(p_l)$，$b_l=1$ 保留残差路径，$b_l=0$ 仅保留跳跃连接。T 次随机前向传播的预测分布提供不确定性估计：$p(y_* | x_*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^{T} p(y_* | x_*, W^{(B_t)})$。

### 关键设计

1. **随机深度作为变分推理的理论推导**:

    - 功能：为 MCSD 提供理论基础
    - 核心思路：定义变分分布 $q_\theta(W) \equiv p(B) = \prod_{l=1}^{L} p_l^{b_l}(1-p_l)^{1-b_l}$，即 L 个独立伯努利变量的乘积。标准 SD 训练（随机前向传播 + L2 正则化）等价于优化 ELBO：期望对数似然通过 MC 采样近似，KL 正则化项由权重衰减近似
    - 设计动机：不同于 MCD 对单个权重/MCDB 对权重块分布，MCSD 对整个网络阶段的包含/排除分布，产生不同深度子网络的隐式集成

2. **MCSD 推理算法**:

    - 功能：在推理时通过保持随机性来采样近似后验
    - 核心思路：与标准 SD 推理（确定性缩放 $x_{l+1} = x_l + p_l \cdot \mathcal{F}_l(x_l; W_l)$）不同，MCSD 保持随机采样并对特征进行归一化 $A_{res} = A_{res} / p_{keep}$。每次前向传播产生不同深度的子网络
    - 设计动机：确定性推理丢弃了不确定性信息，保持随机性直接利用训练中学到的深度分布

3. **适配现代检测器**:

    - 功能：在 YOLO、FasterRCNN、RT-DETR 上应用 MCSD
    - 核心思路：在各检测器的残差路径（Bottleneck、Residual Layer、HGBlock）中的跳跃连接处插入 MCSD/MCD/MCDB，对比不确定性估计质量
    - 设计动机：MCSD 天然适配残差架构，不需要任何结构修改

### 损失函数 / 训练策略

标准检测训练（分类+回归损失 + 权重衰减）。MCSD 利用已有的随机深度正则化，不需要额外训练或自定义损失。

## 实验关键数据

### 主实验

| 方法 | 架构 | COCO mAP↑ | ECE↓ | AUARC↑ |
|------|------|----------|------|--------|
| 确定性 | YOLOv8 | 52.8 | 0.142 | 0.821 |
| MCD | YOLOv8 | 52.5 | 0.128 | 0.835 |
| MCDB | YOLOv8 | 52.3 | 0.135 | 0.829 |
| **MCSD** | YOLOv8 | **52.7** | **0.125** | **0.838** |
| MCD | RT-DETR | 53.1 | 0.118 | 0.842 |
| **MCSD** | RT-DETR | **53.3** | **0.115** | **0.845** |

### 消融实验

| MC采样次数 T | mAP | ECE↓ | 推理时间比 |
|-------------|-----|------|-----------|
| 1 (确定性) | 52.8 | 0.142 | 1.0× |
| 5 | 52.6 | 0.130 | 4.8× |
| 10 | 52.7 | 0.125 | 9.5× |
| 20 | 52.7 | 0.124 | 19.2× |

### 关键发现

- MCSD 在保持竞争性 mAP 的同时，在校准（ECE）和不确定性排名（AUARC）上略优于 MCD
- MCSD 产生的子网络深度变化比 MCD/MCDB 的局部权重/区域丢弃更"多样"
- 兼容所有带跳跃连接的架构（CNN 和 Transformer）

## 亮点与洞察

- MCSD 是"架构原生"的不确定性方法：SD 已经是现代架构的标准正则化器，MCSD 仅需推理时保持随机性，零额外训练开销
- 理论推导将 MCD、MCDB、MCSD 统一到变分推理框架下，揭示了不同粒度（权重→区域→整层）的不确定性建模谱系

## 局限与展望

- KL 散度项的严格计算（离散混合分布 vs 连续先验）在数学上是病态的，使用 L2 正则化作为近似
- 仅在目标检测上评估，未涉及分割和分类
- 推理时多次前向传播的计算开销仍然显著
- 可探索将深度作为可学习随机变量而非固定概率

## 相关工作与启发

- **vs MCD**: MCD 在单个权重级别操作，对卷积层效果有限；MCSD 在整个残差块级别操作，更适合现代架构
- **vs Deep Ensembles**: 集成方法需要 N 倍训练和推理成本，MCSD 从单一模型提取不确定性

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论推导填补了 MCSD 的形式化空白
- 实验充分度: ⭐⭐⭐⭐ 三种检测器 + COCO/COCO-O 的系统基准
- 写作质量: ⭐⭐⭐⭐ 理论部分严谨清晰
- 价值: ⭐⭐⭐⭐ 对安全关键系统的 UQ 有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Calibrating Uncertainty for Zero-Shot Adversarial CLIP](../../ICML2026/ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)
- [\[ICML 2026\] Flatness-Aware Stochastic Gradient Langevin Dynamics](../../ICML2026/ai_safety/flatness-aware_stochastic_gradient_langevin_dynamics.md)
- [\[ICML 2026\] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](../../ICML2026/ai_safety/angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](../../NeurIPS2025/ai_safety/impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[AAAI 2026\] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation](../../AAAI2026/ai_safety/matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)

</div>

<!-- RELATED:END -->
