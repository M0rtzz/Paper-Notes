---
title: >-
  [论文解读] Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference
description: >-
  [CVPR 2025][图像质量评估] 将全参考图像质量评估（FR-IQA）形式化为反事实推断问题，通过结构因果模型（SCM）区分深度特征中与感知质量因果相关的成分和噪声成分，实现无需训练、可跨骨干网络的鲁棒质量预测，在多个基准数据集上取得竞争性性能。
tags:
  - CVPR 2025
  - 图像质量评估
  - 因果推理
  - 反事实推断
  - 全参考IQA
  - 感知质量
---

# Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference

**会议**: CVPR 2025  
**arXiv**: [2412.16939](https://arxiv.org/abs/2412.16939)  
**代码**: https://anonymous.4open.science/r/DeepCausalQuality-25BC (有)  
**领域**: 因果推理  
**关键词**: 图像质量评估, 因果推理, 反事实推断, 全参考IQA, 感知质量

## 一句话总结
将全参考图像质量评估（FR-IQA）形式化为反事实推断问题，通过结构因果模型（SCM）区分深度特征中与感知质量因果相关的成分和噪声成分，实现无需训练、可跨骨干网络的鲁棒质量预测，在多个基准数据集上取得竞争性性能。

## 研究背景与动机

**领域现状**：全参考图像质量评估通常分为特征分解、特征比较和感知分数映射三个阶段。深度学习方法（如 LPIPS、DISTS、DeepWSD）利用预训练网络提取深度特征并计算参考图与失真图之间的距离来预测质量分数。

**现有痛点**：现有方法依赖统计相关性而非因果机制——它们只能量化失真对特征相似性的影响，无法解释失真如何影响人类感知。这导致具有相似特征距离但感知影响截然不同的失真无法被有效区分，跨数据集泛化能力受限。

**核心矛盾**：深度特征中既包含与感知质量因果相关的信息，也包含与质量无关的噪声信息。现有方法将所有特征等同对待，未能分离因果特征和噪声特征，降低了评估准确性。

**本文目标**：建立因果推理框架，从深度特征中识别和提取与感知质量因果相关的特征，排除噪声特征的干扰。

**切入角度**：引入反事实推理——如果对参考图和失真图的特征施加相同的干预（如添加混淆因子），观察感知距离是否发生变化，以此验证特征的因果性。

**核心 idea**：通过结构因果模型定义因果特征 $\gamma$ 和噪声特征 $\eta$，利用干预实验（do 操作）和混淆因子字典筛选出对感知质量具有稳定因果效应的特征通道，仅用因果特征计算质量分数。

## 方法详解

### 整体框架
输入参考图 $I$ 和失真图 $D$，通过预训练深度网络（VGG/ResNet/EfficientNet）提取多层特征。然后构建结构因果模型（SCM），引入外生变量 $U$ 作为混淆因子，对特征进行不同强度的干预。通过观察干预前后感知距离的变化，筛选出因果特征通道（记录在混淆因子字典 $\Gamma$ 中）。最终仅用因果特征计算参考图与失真图之间的因果传输代价（Causal Optimal Transport），作为质量预测分数。

### 关键设计

1. **结构因果模型（SCM）与因果特征分离**:

    - 功能：将深度特征参数 $\theta$ 分解为因果参数 $\gamma$（与质量因果相关）和噪声参数 $\eta$（与质量无关），实现 $\eta \perp \gamma$
    - 核心思路：定义质量分数 $Q_S = m(\phi_\gamma(I, D))$，即质量仅由因果特征决定。通过在所有可能分布上最小化最坏情况预测误差 $\min_{\theta \to \gamma} \sup_{P \in \mathcal{P}} \mathbb{E}_P[l(\cdot)]$ 来优化因果特征的选择
    - 设计动机：现有方法直接用全部预训练特征（含噪声）评估质量，导致虚假相关和泛化不足。分离因果/噪声特征可以从根本上解决这个问题

2. **深度因果测量（Deep Causal Measurement）**:

    - 功能：通过干预实验验证和筛选因果特征通道
    - 核心思路：对参考图和失真图的深度特征施加 do 操作，分别得到干预前特征 $\mathbf{f}_I, \mathbf{f}_D$ 和干预后特征 $\mathbf{f}'_I, \mathbf{f}'_D$。计算感知距离差 $\Delta = Dis(\mathbf{f}_I, \mathbf{f}_D) - Dis(\mathbf{f}'_I, \mathbf{f}'_D)$。若 $\Delta \neq 0$，则该特征具有因果效应。变化干预强度，记录在所有强度下都维持因果性的特征通道到混淆因子字典 $\Gamma(\mathbf{f}_I, \mathbf{f}_D)$
    - 设计动机：仅通过观测相关性无法区分因果关系和虚假关联。干预实验（反事实推理）是验证因果性的经典手段

3. **因果传输代价（Causal Optimal Transport）**:

    - 功能：基于因果特征计算参考图与失真图的感知质量差异
    - 核心思路：$COT(P_X, P_Y) = \inf_{g \in G(P_X, P_Y)} \int \Gamma(x,y) \cdot c(x,y) \, dg(x,y)$，其中 $c(x,y)$ 为 L2 范数距离，$\Gamma(x,y)$ 为因果混淆因子字典起加权作用——只有因果相关的特征通道才参与距离计算
    - 设计动机：传统特征距离（如 LPIPS 的加权 L2）对所有通道等权或学习权重，未考虑因果性。因果传输代价从理论上保证了只有真正影响感知的特征才被计入

### 损失函数 / 训练策略
本方法完全无需训练（training-free）。直接使用预训练网络（VGG-16、ResNet、EfficientNet）的权重，通过因果干预实验筛选特征通道。这是其核心优势之一——无需在特定 IQA 数据集上训练，天然具备跨数据集泛化能力。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Our-VGG | Our-EffNet | DISTS | DeepWSD | LPIPS |
|--------|------|---------|-----------|-------|---------|-------|
| LIVE | PLCC/SRCC | 0.929/0.932 | 0.927/0.932 | 0.924/0.925 | 0.904/0.925 | 0.866/0.863 |
| CSIQ | PLCC/SRCC | 0.949/0.952 | 0.933/0.938 | 0.919/0.920 | 0.941/0.950 | 0.891/0.895 |
| TID2013 | PLCC/SRCC | 0.909/0.884 | 0.899/0.879 | 0.854/0.830 | 0.894/0.874 | 0.713/0.713 |
| KADID | PLCC/SRCC | 0.898/0.899 | 0.905/0.907 | 0.886/0.886 | 0.887/0.888 | 0.838/0.837 |

### 消融实验

| 配置 | LIVE PLCC/SRCC | CSIQ PLCC/SRCC | TID2013 PLCC/SRCC |
|------|---------------|---------------|-------------------|
| $\phi = \phi_\theta$（全部预训练权重） | 0.901/0.915 | 0.913/0.916 | 0.884/0.867 |
| $\phi = \phi_\gamma$（因果特征） | 0.929/0.932 | 0.949/0.952 | 0.909/0.884 |
| $\phi = \phi_\eta$（噪声特征） | 0.843/0.866 | 0.803/0.831 | 0.786/0.789 |

### 关键发现
- 因果特征 $\phi_\gamma$ 的性能显著优于使用全部预训练权重 $\phi_\theta$，说明去除噪声特征确实有助于提升质量预测
- 噪声特征 $\phi_\eta$ 仍有一定感知相关性（非零性能），但远不如因果特征，证实了因果/噪声分离的合理性
- 方法可跨骨干网络（VGG、ResNet、EfficientNet）通用，Our-VGG 和 Our-EffNet 各有优势
- 无需训练即可在大多数数据集上达到或超越需训练方法（TOPIQ-FR 除外），泛化能力出色

## 亮点与洞察
- **将 IQA 形式化为反事实推断问题**：这一视角转换是核心贡献，为理解"为什么某些方法在某些场景好/差"提供了理论框架。其他回归性质的感知任务也可以借鉴此思路
- **完全 training-free**：不依赖任何 IQA 数据集训练，天然具备跨数据集/跨失真类型的泛化能力。这对于实际部署（无需收集训练数据）非常有价值
- **骨干网络无关性**：因果分析过程独立于网络架构，可以即插即用地应用于任何预训练网络，灵活性很高

## 局限与展望
- 在 PIPAL 数据集（GAN 生成失真）上性能不如 TOPIQ-FR，说明对复杂算法失真的因果建模仍需改进
- 混淆因子字典的构建依赖于干预强度的选择，不同强度范围可能影响结果稳定性
- 因果干预在每个通道独立进行，未考虑通道间的交互因果效应
- 方法理论性较强但工程实现细节（如干预方式、字典构建算法）在论文中表述不够清晰

## 相关工作与启发
- **vs LPIPS**: LPIPS 用学习的线性权重加权各层特征距离，本文则用因果筛选替代学习权重，无需训练且更具可解释性
- **vs DISTS**: DISTS 结合结构相似性和纹理相似性，但仍是统计度量。本文从因果角度区分"哪些特征真正影响感知"，理论基础更扎实
- **vs DeepWSD**: DeepWSD 用 Wasserstein 距离度量特征分布差异，本文用因果传输代价替代，额外引入了因果加权

## 评分
- 新颖性: ⭐⭐⭐⭐ 将因果推理引入 FR-IQA 的视角新颖，理论贡献显著
- 实验充分度: ⭐⭐⭐ 6 个数据集评估全面，但消融实验较简单，缺少对干预策略的深入分析
- 写作质量: ⭐⭐⭐ 理论阐述详细但部分内容冗长，公式符号较多增加阅读难度
- 价值: ⭐⭐⭐⭐ 提供了新的理论视角和 training-free 方案，对 IQA 领域有启发意义

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[NeurIPS 2025\] Transferring Causal Effects using Proxies](../../NeurIPS2025/causal_inference/transferring_causal_effects_using_proxies.md)
- [\[ICML 2025\] Causal Abstraction Inference under Lossy Representations](../../ICML2025/causal_inference/causal_abstraction_inference_under_lossy_representations.md)
- [\[ICML 2025\] Isolated Causal Effects of Natural Language](../../ICML2025/causal_inference/isolated_causal_effects_of_natural_language.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/causal_inference/conformal_prediction_for_causal_effects_of_continuous_treatments.md)

<!-- RELATED:END -->
