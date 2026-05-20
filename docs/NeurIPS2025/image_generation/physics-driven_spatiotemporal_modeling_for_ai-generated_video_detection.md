---
title: >-
  [论文解读] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection
description: >-
  [NeurIPS 2025][图像生成][AI生成视频检测] 提出基于物理守恒定律的AI生成视频检测范式，定义归一化时空梯度（NSG）统计量来捕获空间概率梯度与时间密度变化的比率，利用预训练扩散模型估计NSG并通过MMD进行检测，在Recall上超越SOTA 16%、F1超越10.75%。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "AI生成视频检测"
  - "概率流守恒"
  - "归一化时空梯度"
  - "扩散模型"
  - "MMD"
---

# Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection

**会议**: NeurIPS 2025  
**arXiv**: [2510.08073](https://arxiv.org/abs/2510.08073)  
**作者**: Shuhai Zhang, Zihao Lian, Jiahao Yang, Daiyuan Li (SCUT), Guoxuan Pang (USTC), Feng Liu (U Melbourne), Bo Han (HKBU), Shutao Li (HNU), Mingkui Tan (SCUT)  
**代码**: [ZSHsh98/NSG-VD](https://github.com/ZSHsh98/NSG-VD)  
**领域**: 图像生成  
**关键词**: AI生成视频检测, 概率流守恒, 归一化时空梯度, 扩散模型, MMD  

## 一句话总结

提出基于物理守恒定律的AI生成视频检测范式，定义归一化时空梯度（NSG）统计量来捕获空间概率梯度与时间密度变化的比率，利用预训练扩散模型估计NSG并通过MMD进行检测，在Recall上超越SOTA 16%、F1超越10.75%。

## 研究背景与动机

### 问题背景
AI视频生成技术（如Sora）已达到近乎完美的视觉真实感，检测AI生成视频成为维护数字媒体信任的紧迫需求。核心挑战在于：(1) 视频包含复杂的空间纹理结构和时间运动轨迹，需要联合建模框架；(2) AI生成视频在视觉外观和时间演化上的差异越来越细微。

### 已有工作的不足
- **基于伪影的方法**（光流建模、外观一致性分析）依赖特定生成器的伪影特征，对Sora等高质量生成模型失效
- **DeMamba**在HotShot上仅40.60% Recall，Sora上仅48.21% Recall
- **STIL**在关键场景完全崩溃（HotShot 1.40% Recall，Sora 1.79% Recall）
- **TALL**在Sora上仅25.00% Recall
- 现有方法忽略了自然视频固有的物理约束驱动的时空演化规律

### 核心动机
自然视频天然遵循运动连贯性、纹理连续性等物理定律，而AI生成视频常表现出违反物理规律的系统性不一致。本文提出：能否通过物理守恒定律来建模自然视频的内在时空动力学，从而暴露合成异常？

## 方法详解

### 概率流速度场建模
将视频演化建模为类流体力学过程。定义概率流密度 $\mathbf{J}(\mathbf{x},t) = p(\mathbf{x},t) \cdot \mathbf{v}(\mathbf{x},t)$，其中 $p(\mathbf{x},t)$ 为概率密度，$\mathbf{v}(\mathbf{x},t)$ 为引导概率质量流动的速度场。概率质量守恒隐含连续性方程：

$$\frac{\partial p(\mathbf{x},t)}{\partial t} + \nabla_{\mathbf{x}} \cdot \mathbf{J}(\mathbf{x},t) = 0$$

将 $\mathbf{J}$ 代入并取对数，利用不可压缩流近似（散度项 $\nabla_\mathbf{x} \cdot \mathbf{v}$ 为次主项），得到：

$$\mathbf{v}(\mathbf{x},t) \cdot \nabla_{\mathbf{x}} \log p(\mathbf{x},t) \approx -\partial_t \log p(\mathbf{x},t)$$

### 归一化时空梯度（NSG）
由于速度场 $\mathbf{v}$ 的解不唯一，定义其对偶场——归一化时空梯度：

$$\mathbf{g}(\mathbf{x},t) = \frac{\nabla_{\mathbf{x}} \log p(\mathbf{x},t)}{-\partial_t \log p(\mathbf{x},t) + \lambda}$$

其中 $\lambda > 0$ 防止数值不稳定。NSG满足 $\mathbf{v} \cdot \mathbf{g} \approx 1$，绕过了 $\mathbf{v}$ 求逆的病态问题，同时保留时空梯度动力学的关键信息。

**物理含义**：NSG量化每单位时间变化的概率流方向灵敏度，同时捕获空间不规则性（通过 $\nabla_\mathbf{x} \log p$）和时间不一致性（通过 $\partial_t \log p$）。

### 基于扩散模型的NSG估计
利用预训练扩散模型的梯度估计能力：

- **空间梯度**：直接用扩散模型的score网络 $\mathbf{s}_\theta$ 近似 $\nabla_\mathbf{x} \log p(\mathbf{x},t) \approx \mathbf{s}_\theta(\mathbf{x}_t)$
- **时间导数**：基于亮度恒常假设（光流约束），$\partial_t \log p(\mathbf{x},t) \approx -\nabla_\mathbf{x} \log p(\mathbf{x},t) \cdot \frac{\Delta\mathbf{x}}{\Delta t}$

最终NSG估计器：

$$\mathbf{g}(\mathbf{x},t) \approx \frac{\mathbf{s}_\theta(\mathbf{x}_t)}{\mathbf{s}_\theta(\mathbf{x}_t) \cdot \frac{\mathbf{x}_{t+\Delta t} - \mathbf{x}_t}{\Delta t} + \lambda}$$

无需显式光流计算，仅需扩散模型单次前向传播加帧差即可。

### NSG-VD检测方法
1. 汇聚视频所有帧的NSG特征 $\mathbf{G}(\mathbf{x}) = \{\mathbf{g}(\mathbf{x},t)\}_{t=1}^T$
2. 用深度核MMD计算测试视频NSG与参考真实视频集NSG的分布差异
3. 设阈值 $\tau$ 判决：$\widehat{\text{MMD}}_b^2 > \tau$ 则判为Fake

核心深度核采用可学习特征映射 $\phi_\mathbf{G}$ 与高斯核的组合，通过多群体感知优化（MPP）最大化检测能力。

### 理论保证
假设真实视频 $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \sigma(t)^2\mathbf{I}_d)$，生成视频 $\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}, \sigma(t)^2\mathbf{I}_d)$，证明NSG特征距离上界随分布偏移 $\varphi = \|\boldsymbol{\mu}\|^2/\sigma(t)^2$ 增大而增大。这保证了真实视频间MMD小于真实与生成视频间MMD，奠定NSG-VD的理论基础。

## 实验关键数据

### 实验1：标准评估（Pika训练）

在GenVideo基准上，用Kinetics-400（真实）+ Pika（生成）各10,000视频训练。

| 方法 | Avg Recall | Avg Accuracy | Avg F1 | Avg AUROC |
|------|-----------|-------------|--------|-----------|
| DeMamba | 72.02 | 84.21 | 80.12 | 93.88 |
| NPR | 57.35 | 77.96 | 68.39 | 93.02 |
| TALL | 60.78 | 79.85 | 72.63 | 95.67 |
| STIL | 27.02 | 63.51 | 35.82 | 93.49 |
| **NSG-VD** | **88.02** | **91.46** | **90.87** | **96.14** |

关键对比：在Sora上NSG-VD达78.57% Recall（DeMamba 48.21%），HotShot上92.50%（DeMamba 40.60%）。

### 实验2：数据不平衡场景

仅用1,000生成视频（SEINE）+ 10,000真实视频训练——模拟实际中生成样本稀缺。

| 方法 | Avg Recall | Avg Accuracy | Avg F1 | Avg AUROC |
|------|-----------|-------------|--------|-----------|
| DeMamba | 64.09 | 81.60 | 76.44 | 94.85 |
| NPR | 32.71 | 66.09 | 46.54 | 87.10 |
| TALL | 36.08 | 67.95 | 51.40 | 91.96 |
| STIL | 46.78 | 73.21 | 61.43 | 90.20 |
| **NSG-VD** | **93.21** | **89.16** | **89.48** | **94.91** |

NSG-VD在仅1/10生成训练数据下，Recall仍达93.21%，超DeMamba 29.12%，在Sora上82.14%（DeMamba 33.93%）。

### 消融：空间梯度 vs 时间导数

| 组件 | Recall | Accuracy | F1 | AUROC |
|------|--------|----------|-----|-------|
| 仅空间梯度 | 87.99 | 82.84 | 83.40 | 91.85 |
| 仅时间导数 | 60.35 | 71.09 | 66.97 | 78.95 |
| **NSG-VD（两者结合）** | **88.02** | **91.46** | **90.87** | **96.14** |

空间梯度是主要贡献者，但时间导数与其结合后F1从83.40%提升至90.87%（+7.47%），验证了物理守恒原理下二者协同的必要性。

## 亮点

- **物理驱动的新范式**：首次将概率流守恒定律引入AI生成视频检测，通过NSG统计量建模自然视频的内在时空动力学，而非依赖特定伪影
- **优雅的估计器设计**：利用扩散模型score函数估计空间梯度+亮度恒常约束估计时间导数，避免复杂光流计算，仅需单次前向传播
- **强泛化性**：在10种不同生成器（含闭源Sora）上均显著优于SOTA，尤其在数据不平衡（1/10生成数据）下仍保持93%+ Recall
- **理论支撑完备**：严格证明真实/生成视频NSG特征距离与分布偏移的定量关系，为检测有效性提供理论保证
- **阈值鲁棒**：$\tau \in [0.7, 1.1]$ 范围内性能稳定，无需精细调参

## 局限与展望

- **高斯分布假设**：理论分析（Theorem 1）基于高斯分布假设，实际视频分布远比高斯复杂，理论上界可能不够紧
- **不可压缩流近似**：散度项的忽略是启发式的，对快速场景切换或剧烈运动可能不成立
- **扩散模型依赖**：需要预训练扩散模型作为score估计器，计算开销较传统方法大
- **亮度恒常假设**：光照剧变、遮挡等场景下该假设可能失效，影响时间导数估计
- **参考集依赖**：检测需要维护真实视频参考集，实际部署时参考集的选择和规模会影响性能
- **Accuracy略低于Recall**：在SEINE训练设定下Accuracy (86.05%) 低于部分baseline的Accuracy，存在一定误报

## 与相关工作的对比

- **DeMamba (Chen et al., 2024)**：基于Mamba的时空关系建模，依赖大规模监督训练，对未见生成器泛化不足（Sora 48.21% Recall vs NSG-VD 78.57%）
- **TALL (Xu et al., 2023)**：通过缩略图布局进行时空建模，但在闭源模型上表现不稳定（Sora 25.00% Recall）
- **STIL (Gu et al., 2021)**：分别建模空间和时间不一致性，但在新型生成器上完全崩溃（HotShot 1.40% Recall）
- **NPR (Tan et al., 2024)**：基于CNN上采样操作的深度伪造检测，性能波动大（Accuracy 57.20%~98.20%）
- **DIRE (Wang et al., 2023)**：利用扩散模型重建误差检测生成图像，但未涉及时空动力学建模
- **Score-based检测 (Song et al., 2025; Zhang et al., 2024)**：用score统计量检测AI生成文本/图像，本文将其扩展到视频域并引入物理约束

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将流体力学概率流守恒引入视频检测，NSG统计量的定义优雅而具物理直觉
- 实验充分度: ⭐⭐⭐⭐ — 10种生成器、3种训练设定、消融全面，但缺少更多backbone和不同扩散模型的消融
- 写作质量: ⭐⭐⭐⭐⭐ — 从物理建模到统计量定义到估计器推导到理论保证，逻辑链完整流畅
- 价值: ⭐⭐⭐⭐⭐ — 为AI生成视频检测开辟了物理驱动的新方向，实际性能提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Epistemic Uncertainty for Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)
- [\[CVPR 2025\] Where's the Liability in the Generative Era? Recovery-Based Black-Box Detection of AI-Generated Content](../../CVPR2025/image_generation/wheres_the_liability_in_the_generative_era_recovery-based_black-box_detection_of.md)
- [\[NeurIPS 2025\] Is Artificial Intelligence Generated Image Detection a Solved Problem?](is_artificial_intelligence_generated_image_detection_a_solved_problem.md)
- [\[NeurIPS 2025\] UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback](unilumos_fast_and_unified_image_and_video_relighting_with_physics-plausible_feed.md)
- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](../../CVPR2025/image_generation/any-resolution_ai-generated_image_detection_by_spectral_learning.md)

</div>

<!-- RELATED:END -->
