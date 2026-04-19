---
title: >-
  [论文解读] Rectified Noise: A Generative Model Using Positive-incentive Noise
description: >-
  [AAAI 2026][图像生成][Rectified Flow] 提出 Rectified Noise（ΔRN），通过正向激励噪声（π-noise）框架学习一组有益噪声并注入预训练 Rectified Flow 模型的速度场中，以仅 0.39% 的额外参数在 ImageNet-1k 上将 FID 从 10.16 降低到 9.05。
tags:
  - AAAI 2026
  - 图像生成
  - Rectified Flow
  - 正向激励噪声
  - 流匹配
  - SiT
  - 生成模型
---

# Rectified Noise: A Generative Model Using Positive-incentive Noise

**会议**: AAAI 2026  
**arXiv**: [2511.07911](https://arxiv.org/abs/2511.07911)  
**代码**: [https://github.com/simulateuser538/Rectified-Noise](https://github.com/simulateuser538/Rectified-Noise)  
**领域**: 图像生成  
**关键词**: Rectified Flow, 正向激励噪声, 流匹配, SiT, 生成模型

## 一句话总结

提出 Rectified Noise（ΔRN），通过正向激励噪声（π-noise）框架学习一组有益噪声并注入预训练 Rectified Flow 模型的速度场中，以仅 0.39% 的额外参数在 ImageNet-1k 上将 FID 从 10.16 降低到 9.05。

## 研究背景与动机

### 领域现状

Rectified Flow (RF) 是一种高效的生成建模方法，通过直线路径连接源分布和目标分布来学习速度场。RF 直接参数化连续时间传输映射，不引入额外随机性，训练目标简单：

$$\mathcal{L}_{velocity}(\theta) = \mathbb{E}_{x_*, \epsilon, t}\left[\|\mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{x}_* + \epsilon\|^2\right]$$

SiT（Scalable Interpolant Transformers）作为基于 DiT 的 RF 模型家族，通过系统探索设计空间取得了优秀的生成性能。

### 现有痛点

**一个有趣的发现**：虽然 RF 基于概率流 ODE，但最近研究（SiT）发现，使用反时间 SDE 在采样时注入随机噪声反而能**提升**生成性能（更低的 FID）。这意味着：
- RF 的确定性采样并非最优
- 某种特定的噪声对 RF 是有益的

### 核心问题

这引发了两个关键问题：

**什么样的随机噪声**能为 RF 带来性能增益？

**如何**将有益噪声引入 RF？

### 切入角度

π-noise（正向激励噪声）框架提供了理论基础——通过最大化任务与噪声之间的互信息来学习有益噪声：

$$\max_{\mathcal{E}} MI(\mathcal{T}, \mathcal{E}) = H(\mathcal{T}) - H(\mathcal{T}|\mathcal{E})$$

本文将 π-noise 框架与 RF 建立联系，设计 π-noise 生成器来自动学习最优噪声。

## 方法详解

### 整体框架

Rectified Noise 管道包含两个阶段：
1. **预训练 RF 模型**获得最优参数 $\psi^*$
2. **训练 π-noise 生成器**：冻结 RF 参数，附加可训练 SiT 块预测 π-noise，注入到速度场中

推理时：标准 RF 推理 + 将 π-noise 加到预测的速度场上。

### 关键设计

#### 1. **通过 RF 损失定义任务熵**

**核心思路**：需要度量 RF 模型在给定数据集上的学习复杂度。引入辅助随机变量 $\alpha$ 连接 RF 损失和信息熵：

$$\alpha | \mathbf{x}, t \sim \mathcal{N}(0, \exp(\mathcal{L}(\mathbf{x}, t; \psi^*)))$$

其中 $\mathcal{L}(\mathbf{x}, t; \psi^*)$ 是最优 RF 模型在样本 $\mathbf{x}$ 和时间 $t$ 上的损失值。损失越大 → 辅助分布方差越大 → 信息熵越高 → 任务越困难。

任务熵定义为：

$$H(\mathcal{T}) = \frac{1}{2}\mathbb{E}_{\mathbf{x},t}\mathcal{L}(\mathbf{x}, t; \psi^*) + \frac{1}{2}\ln(2\pi e)$$

**设计动机**：通过辅助高斯分布巧妙地将 RF 的回归损失与信息熵联系起来，为 π-noise 框架在生成模型中的应用奠定基础。

#### 2. **将 π-noise 注入 RF 模型**

**核心推导**：最大化互信息等价于最小化条件熵 $H(\mathcal{T}|\mathcal{E})$。定义带噪声的辅助分布：

$$\mathcal{L}(\mathbf{x}, \epsilon, t, \psi^*) = \|\mathbf{v}_{\psi^*} + \epsilon(\mathbf{x}_t, t) - \mathbf{x}_* + \mathbf{x}_0\|^2$$

**关键洞察**：当 $p(\epsilon|\mathbf{x}, t) \rightarrow \delta(\epsilon)$（狄拉克函数，即噪声恒为 0）时，优化目标退化为标准 RF 损失。这意味着**标准 RF 是 ΔRN 的一个特例**（π-noise 始终为 0）。

最终优化目标简化为：

$$\max_\theta \mathbb{E}_{\mathbf{x}, t, \epsilon \sim \epsilon_\theta} \mathcal{L}(\mathbf{x}, \epsilon, t; \psi^*)$$

用神经网络 $\epsilon_\theta$ 参数化 π-noise，最大化带噪声的 RF 损失。

#### 3. **两种优化策略**

**策略一：同时优化 $\theta$ 和 $\psi$**

通过重参数化技巧统一两组参数。以高斯分布为例：

$$\hat{\mathbf{v}} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) + \boldsymbol{\sigma}_\theta(\mathbf{x}_t, t) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\hat{\mu}_\theta = \mathbf{v}_{\psi^*} + \mu_\theta$ 可被单个网络预测。

**策略二：冻结 $\psi^*$，仅优化 $\theta$（推荐）**

从预训练 RF 模型的中间特征层提取输入，附加新的 SiT 块作为 π-noise 生成器，最终线性层初始化为零以确保初始输出匹配原始 RF。

**关键发现**：策略一训练不稳定（引入随机噪声导致收敛困难），策略二（微调方式）更优。

### π-noise 分布假设

探索了三种可重参数化分布：
- **高斯分布**：$\mathbf{z} = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
- **Gumbel 分布**：$\mathbf{z} = \mu - \beta \odot \log(-\log(\epsilon)), \quad \epsilon_i \sim U(0,1)$
- **均匀分布**：$\mathbf{z} = \mathbf{a} + (\mathbf{b}-\mathbf{a}) \odot \epsilon, \quad \epsilon_i \sim U(0,1)$

### 训练策略

- 冻结预训练 RF 模型参数
- 附加 0-4 个额外 SiT 块作为 π-noise 生成器
- 仅训练 π-noise 生成器参数（额外参数量 0.39%-14.56%）
- ImageNet 使用 6M 步预训练 RF + 100K 步微调 ΔRN
- AFHQ/CelebA-HQ 用 100K/200K 步预训练 + 10K 步微调

## 实验关键数据

### 主实验

ImageNet-1k 256×256（无 CFG）：

| 模型 | 噪声设置 | 额外 SiT 块 | 额外参数 | FID↓ | IS↑ | sFID↓ | Prec.↑ | Rec.↑ |
|------|---------|-----------|----------|------|-----|-------|--------|-------|
| SiT-XL/2 | - | - | - | 10.16 | 123.86 | 12.02 | 0.50 | 0.62 |
| +ΔRN | $\mathcal{N}(\mu,\Sigma)$ | 0 | 0.39% | 9.06 | 130.21 | 11.18 | 0.52 | 0.61 |
| +ΔRN | $\mathcal{N}(\mu,\Sigma)$ | 1 | 3.93% | **9.05** | **132.10** | 11.23 | 0.52 | 0.62 |

跨数据集结果：

| 数据集 | 基线 FID | ΔRN FID | FID 提升 |
|--------|---------|---------|---------|
| ImageNet-1k | 10.16 | 9.05 | **-1.11** |
| AFHQ | 12.33 | 10.44 | **-1.89** |
| CelebA-HQ | 11.25 | 7.73 | **-3.52** |

### 消融实验

不同噪声分布假设（ImageNet-1k）：

| 噪声分布 | FID↓ | IS↑ | sFID↓ | Prec.↑ | Rec.↑ |
|---------|------|-----|-------|--------|-------|
| 无（基线） | 10.16 | 123.86 | 12.02 | 0.50 | 0.62 |
| **高斯** | **9.05** | **132.10** | **11.23** | **0.52** | 0.62 |
| Gumbel | 9.42 | 129.73 | 11.42 | 0.52 | 0.61 |
| 均匀 | 10.02 | 124.40 | 11.63 | 0.51 | 0.62 |

额外 SiT 块数量的影响（$\mathcal{N}(\mu,\Sigma)$）：

| 额外块数 | 参数比例 | FID↓ | 说明 |
|---------|----------|------|------|
| 0 | 0.39% | 9.06 | 仅线性层即有效 |
| 1 | 3.93% | **9.05** | 最优 |
| 2 | 7.48% | 9.08 | 收益饱和 |
| 4 | 14.56% | 9.15 | 参数过多反而略降 |

### 关键发现

- **极少参数即可生效**：仅 0.39% 额外参数（不加 SiT 块，只有线性层）即可将 FID 从 10.16 降至 9.06
- **高斯分布最优**：三种噪声分布中高斯效果最好，可能因为与 RF 的正向过程（高斯噪声）天然匹配
- **CelebA-HQ 提升最大**：FID 降低 3.52，可能因为面部数据的分布更集中，π-noise 更容易学习
- **微调策略优于联合训练**：同时训练 θ 和 ψ 导致 FID 收敛更慢且不稳定
- **SiT 块数量边际收益递减**：0-1 个额外块已足够，更多块反而可能引入过拟合

## 亮点与洞察

1. **理论优雅**：通过辅助高斯变量将 RF 损失与信息熵建立联系，推导过程严谨且简洁，最终揭示标准 RF 是 ΔRN 在 π-noise 为 0 时的特例
2. **参数效率极高**：0.39% 额外参数即可获得显著提升，这在当前模型日益膨胀的趋势下非常有价值
3. **即插即用**：不改变预训练 RF 模型的架构和权重，仅需附加轻量级 π-noise 生成器
4. **π-noise 可视化**：论文展示了 π-noise 随时间步的变化，揭示了有益噪声的时空结构特征
5. **通用性**：三个不同数据集上都有稳定提升，说明方法不依赖于特定数据分布

## 局限与展望

- 仅在 SiT（RF 的特定实现）上验证，未测试其他 Flow Matching 架构（如 Flux、SD3）
- 实验仅使用 256×256 分辨率，高分辨率生成的效果未知
- 未探索与 CFG（Classifier-Free Guidance）的交互效果
- 联合训练策略失效的原因分析不够深入
- π-noise 的可解释性有限——有益噪声到底编码了什么信息？
- 10K 步微调的收敛性和超参数敏感性未充分探讨

## 相关工作与启发

- **与 SDE 采样的联系**：SiT 发现 SDE 采样优于 ODE 采样，ΔRN 可以看作是对"什么噪声最优"的进一步回答——不是随机噪声，而是学习到的 π-noise
- **π-noise 在其他任务的成功**：VPN 增强经典神经网络、PiNI 增强视觉语言模型，本文将其扩展到生成模型
- **启发**：预训练模型 + 轻量级增强模块的范式非常高效，可以推广到其他生成任务（文本生成、视频生成等）
- 与 LoRA 类方法的关系值得探讨——两者都是以极少参数增强预训练模型

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （π-noise 与 RF 的理论联系是全新发现，推导优雅）
- 实验充分度: ⭐⭐⭐⭐ （三个数据集验证，消融较全面，但缺少高分辨率/CFG 实验）
- 写作质量: ⭐⭐⭐⭐ （理论推导清晰，实验呈现规范）
- 价值: ⭐⭐⭐⭐ （揭示了 RF 中有益噪声的存在并提供了学习方法，有后续研究空间）

<!-- RELATED:START -->

## 相关论文

- [Straighten Viscous Rectified Flow via Noise Optimization](../../ICCV2025/image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)
- [TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](../../CVPR2026/image_generation/taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)
- [Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](../../ICLR2026/image_generation/flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation](../../ICML2025/image_generation/editable_noise_map_inversion_encoding_target-image_into_noise_for_high-fidelity_.md)
- [Improved Noise Schedule for Diffusion Training](../../ICCV2025/image_generation/improved_noise_schedule_for_diffusion_training.md)

<!-- RELATED:END -->
