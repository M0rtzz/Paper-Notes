---
title: >-
  [论文解读] Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis
description: >-
  [ICCV 2025][图像生成][扩散模型] Omegance 提出仅通过一个参数 $\omega$ 缩放扩散模型去噪步骤中的噪声预测，即可无需重训练地实现对生成图像/视频细节粒度的全局、空间和时序精细控制，方法与架构无关且兼容 SDXL、SD3、FLUX 等多种模型。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "扩散模型"
  - "粒度控制"
  - "噪声缩放"
  - "无训练"
  - "细节增强/抑制"
---

# Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis

**会议**: ICCV 2025  
**arXiv**: [2411.17769](https://arxiv.org/abs/2411.17769)  
**代码**: [https://github.com/itsmag11/Omegance](https://github.com/itsmag11/Omegance)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 扩散模型, 粒度控制, 噪声缩放, 无训练, 细节增强/抑制

## 一句话总结
Omegance 提出仅通过一个参数 $\omega$ 缩放扩散模型去噪步骤中的噪声预测，即可无需重训练地实现对生成图像/视频细节粒度的全局、空间和时序精细控制，方法与架构无关且兼容 SDXL、SD3、FLUX 等多种模型。

## 研究背景与动机

**领域现状**：扩散模型已成为高质量图像合成的主流范式，但对生成结果的细节粒度（detail granularity）缺乏直接的精细控制。用户在实际创作中经常需要选择性地调整不同区域的细节丰富程度。

**现有痛点**：(a) 文本难以精确描述细节程度（如"减少背景细节同时保持主体高细节"很难用 prompt 表达）；(b) 现有质量增强方法（如 FreeU、SAG/PAG）only 全局增强，无法空间精细控制；(c) RLHF 微调方法成本高且不灵活；(d) FreeU 与 U-Net 架构紧耦合，不适用于新架构如 DiT。

**核心矛盾**：扩散模型的均匀去噪过程不允许对同一图像中不同区域施加不同级别的细节控制，且 SNR 调度在生成过程中是固定的。

**本文目标** 用最简单的方式实现：(a) 全局细节增强/抑制；(b) 空间区域指定的粒度控制；(c) 时序阶段相关的粒度控制。

**切入角度**：噪声缩放是扩散模型中的基础操作，但从未被系统性地探索为粒度控制手段。作者发现对噪声预测做简单缩放就能有效改变 SNR，从而控制高频/低频信息的保留。

**核心 idea**：在每步去噪中将噪声预测乘以 $\omega$——$\omega < 1$ 保留更多高频信息生成更丰富细节，$\omega > 1$ 去除更多高频产生更平滑效果。

## 方法详解

### 整体框架
Omegance 在扩散模型反向去噪过程的每一步，对噪声预测施加一个缩放因子 $\omega$。标准去噪步为 $z_{t-1} = \delta_t \cdot z_t + \zeta_t \cdot \epsilon_\theta(z_t, t)$，Omegance 修改为 $z'_{t-1} = \delta_t \cdot z_t + \zeta_t \cdot \epsilon_\theta(z_t, t) \cdot \omega$。该方法不修改网络架构、不需要重训练，计算开销几乎为零。

### 关键设计

1. **全局 Omega 控制**:

    - 功能：统一地增强或抑制整幅图像的细节
    - 核心思路：以 DDIM 为例，修改后的 SNR 为：$\text{SNR}(t-1)' = \frac{\alpha_{t-1}}{[\frac{\sqrt{\alpha_{t-1}}\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}} + \omega(\frac{\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}} - \sqrt{\alpha_{t-1}}\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}})]^2}$。由于 $\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}} - \sqrt{\alpha_{t-1}}\sqrt{1-\alpha_t}$ 恒为负，$\omega < 1$ 时 $\text{SNR}' < \text{SNR}$（保留更多高频），$\omega > 1$ 时 $\text{SNR}' > \text{SNR}$（去除更多高频）
    - 设计动机：通过修改有效 SNR 实现对高频信息的控制，这是一个物理上可解释的机制

2. **Omega Mask（空间控制）**:

    - 功能：为图像不同区域指定不同的 $\omega$ 值
    - 核心思路：$\omega_{i,j} = \mathcal{M}(i,j)$，其中 $\mathcal{M} \in \mathbb{R}^{H' \times W'}$ 是与去噪潜变量同尺寸的 mask。利用去噪过程的局部性，一个区域的 $\omega$ 调整不会影响邻域的 SNR
    - Mask 来源：用户手绘 strokes、分割掩码、ControlNet 信号（如 pose 骨架→人物 mask，depth→前后景 mask）、或连续深度值
    - 设计动机：实际应用中经常需要主体高细节+背景低细节，或反之。空间 mask 提供了直觉且灵活的接口

3. **Omega Schedule（时序控制）**:

    - 功能：在不同去噪阶段使用不同的 $\omega$ 值
    - 核心思路：$\omega_t = \mathcal{S}(t)$，利用扩散模型的去噪动力学特性——早期阶段（$t \in [T, \tau]$）形成布局结构，晚期阶段（$t \in [\tau, 0]$）细化细节纹理。早期低 $\omega$ 增加布局复杂度，晚期低 $\omega$ 增强纹理细节。布局形成阶段仅占前 ~10 步（50 步总数中）
    - 设计动机：layout 和 texture 分属不同去噪阶段，时序控制允许独立调节两者的复杂度

4. **适配不同调度器**:

    - DDIM/Euler：直接对 $\epsilon_\theta$ 乘 $\omega$
    - Flow matching（如 FLUX）：需要 mean-preserving 操作避免颜色偏移，公式为 $z'_{t-dt} = z_t + [(dt \cdot v_\theta(z_t,t) - m) \cdot \omega + m]$，其中 $m = \mathbb{E}[dt \cdot v_\theta(z_t,t)]$

### 实用技巧
实际使用中 $\omega$ 通过 $\omega = \mathcal{R}(\varpi)$ 重缩放，使输入 $\varpi \in (-\infty, \infty)$ 的中心为 0，方便用户调节。

## 实验关键数据

### 主实验（SDXL T2I 定量评估）

| 方法 | FID↓ | IS↑ | CLIP↑ | Q-Align↑ | PickScore↑ |
|------|------|-----|-------|----------|-----------|
| SDXL (baseline) | 162.18 | 13.23 | 32.88 | 4.68 | 0.1468 |
| + FreeU | 167.22 | 12.25 | 31.76 | 4.64 | 0.0967 |
| + Cosine Sch. | 182.06 | 11.38 | 30.78 | 2.88 | 0.0376 |
| + Rescaled Sch. | 163.29 | 10.88 | 28.80 | 3.25 | 0.0295 |
| + **Omegance** $\varpi(6.0)$ | **157.47** | **13.82** | 32.70 | 4.64 | 0.1149 |
| + **Omegance** $\varpi(-6.0)$ | 170.52 | 13.01 | 32.81 | 4.67 | **0.1601** |
| + EXP1 schedule | 173.49 | 12.67 | 32.70 | 4.64 | 0.1578 |
| + COS1 schedule | 159.87 | 13.25 | 32.64 | 4.60 | 0.0962 |

Omegance 在所有关键维度上均优于 FreeU、Cosine/Rescaled scheduler 等方法。

### 频域分析（细节变化量化）

| 设置 | SSIM↑ | HFE 变化 |
|------|-------|---------|
| SDXL baseline | 1.0 | 0 |
| $\varpi(6.0)$ 细节抑制 | 0.8124 | -204.2 |
| $\varpi(-6.0)$ 细节增强 | 0.7940 | +520.7 |
| EXP1 (layout+detail增强) | 0.7087 | +1113.1 |
| EXP2 (layout增强+detail抑制) | 0.6926 | +205.8 |
| COS1 (layout+detail抑制) | 0.8183 | -154.7 |
| COS2 (layout抑制+detail增强) | 0.7311 | -546.6 |

HFE（高频能量）变化完全符合 $\omega$ 设计预期，验证了方法的可控性。

### 用户研究

| 评估维度 | Omegance 得分 |
|---------|-------------|
| 粒度控制准确度 | 93.94% |
| 输出质量偏好 | 81.38% (67.62%更好 + 13.76%同等) |

101 名参与者的研究表明 Omegance 在准确控制粒度的同时不影响基础模型质量。

### 关键发现
- **细节抑制提升图像质量**：$\varpi=6.0$ 获得最低 FID (157.47) 和最高 IS (13.82)，说明适度平滑可以减少低质量模型的 artifacts
- **细节增强改善美学**：$\varpi=-6.0$ 获得最高 PickScore (0.1601)，说明更丰富的细节更符合人类审美偏好
- **SSIM 保持较高**证明 Omegance 不改变整体布局结构
- **对 FLUX 的互补作用**：FLUX 生成的图像tends to过度平滑，Omegance 的细节增强模式能恢复精细纹理，提升真实感
- 方法成功泛化到视频生成（Mochi、Hunyuan）且保持时序一致性

## 亮点与洞察
- **极致简洁**是最大亮点：一个参数、不改架构、不需训练、零额外计算——这种改动看似微不足道，但通过 SNR 分析给出了严谨的理论解释。简单到令人惊讶的方法往往有最大的实际价值
- **空间+时序的组合控制**非常实用：layout 和 texture 分别受控的能力可以直接用于 ControlNet/IP-Adapter 的后处理流程
- **架构无关性**使其成为即插即用的工具：从 U-Net 到 DiT 到 Flow Matching，只需修改几行代码
- **频域分析**提供了理解扩散模型去噪过程的有价值视角

## 局限性
- Omegance 不能从根本上提升基础模型的生成质量，只能在已有能力范围内调节粒度
- $\omega$ 的最优值依赖于具体模型和场景，用户需要手动调参
- 对于 flow matching 模型需要额外的 mean-preserving 操作，不如对 DDIM 那样优雅
- 论文没有讨论 $\omega$ 范围的边界——过大或过小时生成结果会如何退化

## 相关工作与启发
- **vs FreeU**: FreeU 通过调整 U-Net 骨干特征和 skip connection 的缩放因子来提升质量，与 U-Net 紧耦合且需要双参数。Omegance 单参数、架构无关，更通用简洁
- **vs SAG/PAG**: 这些方法替换 CFG 中的 null-text 预测来全局增强质量，缺乏空间精细控制能力。Omegance 通过 omega mask 实现了真正的区域级控制
- **vs Noise Scheduling**: 传统的 cosine/linear/rescaled scheduler 需要重训练且不支持局部控制，Omegance 在推理时即可灵活调整

## 评分
- 新颖性: ⭐⭐⭐⭐ 思路极简但此前未被系统探索，理论分析到位
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 T2I/I2I/T2V、多模型、定量+用户研究+频域分析
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，公式推导完整，figure 质量高
- 价值: ⭐⭐⭐⭐ 实用性极高的即插即用工具，但不是方法论突破

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] PolarAnything: Diffusion-based Polarimetric Image Synthesis](polaranything_diffusion-based_polarimetric_image_synthesis.md)
- [\[ICML 2025\] Reimagining Parameter Space Exploration with Diffusion Models](../../ICML2025/image_generation/reimagining_parameter_space_exploration_with_diffusion_models.md)
- [\[ICCV 2025\] Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy](disrupting_model_merging_a_parameter-level_defense_without_sacrificing_accuracy.md)
- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [\[ICML 2025\] Learning Single Index Models with Diffusion Priors](../../ICML2025/image_generation/learning_single_index_models_with_diffusion_priors.md)

</div>

<!-- RELATED:END -->
