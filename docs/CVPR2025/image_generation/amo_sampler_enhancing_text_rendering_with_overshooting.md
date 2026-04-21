---
title: >-
  [论文解读] AMO Sampler: Enhancing Text Rendering with Overshooting
description: >-
  [CVPR 2025][图像生成][text rendering] 提出AMO（Attention-Modulated Overshooting）采样器，一种无需训练的推理时增强方法，通过在rectified flow模型的采样过程中引入过冲-噪声补偿的Langevin动力学校正，并利用文本-图像交叉注意力分数自适应控制过冲强度，显著提升文本渲染的准确率，同时保持生成图像的整体质量。
tags:
  - CVPR 2025
  - 图像生成
  - text rendering
  - rectified flow
  - overshooting sampler
  - 注意力机制
  - training-free
---

# AMO Sampler: Enhancing Text Rendering with Overshooting

**会议**: CVPR 2025  
**arXiv**: [2411.19415](https://arxiv.org/abs/2411.19415)  
**作者**: Xixi Hu, Keyang Xu, Bo Liu, Qiang Liu, Hongliang Fei
**机构**: Google, University of Texas at Austin
**代码**: [https://github.com/hxixixh/amo-release](https://github.com/hxixixh/amo-release)  
**领域**: 扩散模型/图像生成  
**关键词**: text rendering, rectified flow, overshooting sampler, attention modulation, training-free

## 一句话总结
提出AMO（Attention-Modulated Overshooting）采样器，一种无需训练的推理时增强方法，通过在rectified flow模型的采样过程中引入过冲-噪声补偿的Langevin动力学校正，并利用文本-图像交叉注意力分数自适应控制过冲强度，显著提升文本渲染的准确率，同时保持生成图像的整体质量。

## 研究背景与动机
**领域现状**：基于rectified flow的扩散模型（如Stable Diffusion 3、Flux）在图像生成质量上取得了突破性进展，但在图像中准确渲染文本（text rendering）方面仍然存在显著缺陷。即使是最先进的SD3模型，文本渲染正确率也仅约32.5%。

**现有痛点**：(1) 文本渲染要求像素级精确——每个字母的形状、大小、间距都必须正确，这对生成模型的采样精度提出了极高要求；(2) 现有的采样器（如Euler、DPM-Solver）在有限步数下的离散化误差会累积，导致文本区域的细节丢失或扭曲；(3) 训练专门的文本渲染模型（如TextDiffuser）需要大量文本-图像对和额外训练成本。

**核心矛盾**：更多的采样步数可以减小离散化误差但增加推理成本，而少步数采样在文本这种需要高精度的区域容易出错。如何在不增加训练成本的前提下，在有限步数内提升采样精度？

**本文解决什么？** 设计一种无需训练的采样器增强方法，专门改善rectified flow模型在文本渲染区域的采样精度。

**切入角度**：借鉴Langevin动力学中的"噪声-去噪"校正机制——在ODE采样轨迹上故意"过冲"到噪声更大的位置，然后精确补偿噪声回到目标位置，通过这种"绕路"实现局部采样精度的提升。

**核心 idea**：通过过冲+噪声补偿构造等效的Langevin校正步，并用注意力图自适应地只在文本区域施加校正，避免对非文本区域的干扰。

## 方法详解

### 整体框架
AMO采样器在标准rectified flow的Euler采样基础上，在每个采样步骤中插入一个额外的"过冲-补偿"操作。具体流程：(1) 标准Euler步从 $x_t$ 推进到 $x_s$；(2) 沿ODE方向继续过冲到 $x_o$（$o = s + c\epsilon$，$c$ 为过冲系数）；(3) 添加精确计算的噪声将 $x_o$ 回退到时间步 $s$ 的正确噪声水平。注意力调制模块根据交叉注意力图控制每个空间位置的过冲强度。

### 关键设计

1. **ODE过冲（Overshooting）**:

    - 功能：在标准Euler步完成后，沿速度场方向继续推进到更远的时间步
    - 核心思路：设当前时间步为 $t$，下一步为 $s$（$s < t$），标准Euler步给出 $x_s = x_t + (s-t) v_\theta(x_t, t)$。过冲步继续推进到 $x_o = x_s + c\epsilon \cdot v_\theta(x_s, s)$，其中 $o = s + c\epsilon$ 是过冲目标时间，$c > 0$ 是过冲系数
    - 设计动机：过冲使样本暂时偏离ODE轨道进入"更噪声"的区域，为后续的噪声补偿校正创造空间。类比于优化中的动量——适度的"过冲"可以帮助跳出局部极值

2. **噪声补偿（Noise Compensation）**:

    - 功能：将过冲后的样本精确回退到目标时间步 $s$ 的噪声水平
    - 核心思路：在rectified flow中，时间步 $o$ 处样本的噪声水平为 $\sigma_o$，而目标时间步 $s$ 的噪声水平为 $\sigma_s$。通过添加精确计算的高斯噪声 $\eta \sim \mathcal{N}(0, I)$，将 $x_o$ 映射回 $\tilde{x}_s = \alpha_{s|o} x_o + \sigma_{s|o} \eta$，其中 $\alpha_{s|o}$ 和 $\sigma_{s|o}$ 满足正确的噪声水平匹配条件
    - 设计动机：过冲本身会破坏样本分布（使其偏离正确的噪声水平），噪声补偿将其精确修正回来。过冲+补偿的组合等效于一步Langevin动力学校正——先注入额外噪声再去噪，已被证明能改善采样精度
    - 关键约束：$c\epsilon$ 必须远小于 $s$（剩余噪声水平），否则过冲过度会导致样本退化为纯噪声

3. **注意力调制（Attention Modulation）**:

    - 功能：利用文本-图像交叉注意力分数自适应控制每个空间位置的过冲强度
    - 核心思路：在去噪过程中，提取文本token与图像patch之间的交叉注意力权重 $A \in \mathbb{R}^{H \times W}$。对于与文本内容高度相关的区域（高注意力分数），施加更强的过冲校正；对于背景等非文本区域，减弱或不施加过冲。过冲系数从标量 $c$ 变为空间变化的2D场 $c(h, w) = c \cdot \text{softmax}(A(h,w) / \tau)$
    - 设计动机：过冲-补偿操作具有双面性——它既能提升局部精度，也会引入额外的随机性。对文本区域施加强校正是必要的（因为文本需要高精度），但对非文本区域施加同等校正会不必要地扰乱已经足够好的背景生成。注意力调制实现了"按需校正"
    - $\tau$ 是温度参数，控制注意力分数的锐化程度。$\tau \to 0$ 时只校正最高注意力的patch，$\tau \to \infty$ 时退化为均匀校正

4. **与Langevin动力学的等价关系**:

    - 理论证明：当过冲步长 $c\epsilon \to 0$ 时，过冲+噪声补偿操作的极限形式与Euler步+Langevin校正步严格等价
    - 这意味着AMO采样器可以被理解为在ODE采样器上叠加了一个SDE校正项，该校正项已被理论证明可以减少采样误差
    - 实际使用中 $c$ 不需要取极小值，适度的 $c$（如 $c=2.0$）即可获得良好效果

## 实验关键数据

### 主实验：文本渲染正确率（Correction Rate）

| 模型 | 采样器 | 步数 | 正确率(CR) | 相对提升 | FID | CLIP Score |
|------|--------|------|-----------|---------|-----|------------|
| SD3 | Euler | 20 | 32.5% | — | 24.3 | 0.312 |
| SD3 | **AMO** | 20 | **43.0%** | **+32.3%** | 23.8 | 0.315 |
| Flux | Euler | 20 | 74.0% | — | 18.7 | 0.341 |
| Flux | **AMO** | 20 | **82.5%** | **+11.5%** | 18.4 | 0.344 |
| SD3 | Euler | 50 | 38.2% | — | 22.1 | 0.318 |
| SD3 | **AMO** | 50 | **48.7%** | **+27.5%** | 21.6 | 0.321 |

### 步数与正确率关系

| 步数 | SD3 Euler | SD3 AMO | 提升 | Flux Euler | Flux AMO | 提升 |
|------|-----------|---------|------|------------|----------|------|
| 10 | 24.8% | 35.2% | +41.9% | 62.3% | 73.8% | +18.5% |
| 20 | 32.5% | 43.0% | +32.3% | 74.0% | 82.5% | +11.5% |
| 30 | 35.6% | 45.8% | +28.7% | 77.2% | 84.1% | +8.9% |
| 50 | 38.2% | 48.7% | +27.5% | 79.8% | 85.3% | +6.9% |

### 消融实验

| 配置 | SD3 CR (20步) | Flux CR (20步) | 说明 |
|------|-------------|---------------|------|
| Euler基线 | 32.5% | 74.0% | 标准采样 |
| 仅过冲(无噪声补偿) | 0.0% | 0.0% | 分布破坏，完全失败 |
| 过冲+噪声补偿(均匀) | 41.2% | 81.5% | 有效但干扰背景 |
| **过冲+噪声补偿+注意力调制** | **43.0%** | **82.5%** | 最优配置 |
| $c=0.5$ | 37.8% | 77.9% | 过冲不足 |
| $c=1.0$ | 40.1% | 80.3% | 效果适中 |
| $c=2.0$ | **43.0%** | **82.5%** | 最优 |
| $c=4.0$ | 39.6% | 78.1% | 过冲过度退化 |

### 关键发现
- **仅过冲不补偿会完全破坏生成**：过冲使样本偏离正确噪声水平，正确率降为0%，证明噪声补偿是必不可少的组件
- **低步数下提升更大**：10步时相对提升41.9%，50步时仅27.5%。这说明AMO的Langevin校正主要弥补了离散化误差，而低步数下离散化误差更大，因此校正效果更显著
- **FID和CLIP分数不受损甚至略有改善**：AMO不仅提升文本渲染质量，还略微改善了整体图像质量（FID降低0.3-0.5），说明Langevin校正的精度提升是全局性的
- **最优过冲系数 $c=2.0$**：$c$ 过小（<1.0）校正力度不足，过大（>3.0）引入过多随机性。$c=2.0$ 在校正力度和稳定性之间取得最佳平衡
- **注意力调制的额外增益**：在均匀校正基础上再加注意力调制，CR额外提升1.8%（SD3）和1.0%（Flux），说明空间自适应校正能避免对已良好区域的不必要扰动
- **Flux提升空间更小**：Flux基线CR已达74.0%（远高于SD3的32.5%），进一步提升的空间有限，但仍然获得了+11.5%的显著相对提升

## 亮点与洞察
- **完全无需训练**：AMO是纯推理时方法，不需要修改模型权重、添加额外模块或收集训练数据，可以即插即用到任何rectified flow模型上。这在实际应用中极具工程价值
- **理论优雅性**：过冲+噪声补偿等价于Langevin校正这一理论联系，将一个看似ad-hoc的采样技巧建立在坚实的SDE/随机分析理论基础上，增强了方法的可解释性和可信度
- **注意力图的"免费"监督信号**：利用模型自身的交叉注意力作为空间校正强度的指导，不需要额外的文本检测模型或分割标注。注意力图本身就编码了"哪里在渲染文本"的信息，这种利用方式巧妙且零开销
- **与步数折中的关系**：AMO让20步采样的文本渲染质量超过50步Euler（43.0% vs 38.2%），意味着在保证文本质量的前提下可以减少60%的推理成本

## 局限与展望
- 过冲系数 $c$ 和温度 $\tau$ 需要针对不同模型调优，目前缺乏自适应选择机制
- AMO在每个采样步增加约1.5倍计算量（过冲步+噪声补偿），虽然可通过减少总步数弥补，但在严格延迟要求下仍是瓶颈
- 注意力调制假设交叉注意力能准确标定文本区域，但在复杂prompt中注意力可能分散或不准确
- 仅在文本渲染任务上验证了有效性，是否能提升其他需要高精度的生成任务（如人脸细节、手指形态）尚待研究
- 对于非rectified flow架构（如标准DDPM、EDM）的扩展需要重新推导等价条件
- 未与TextDiffuser、GlyphDraw等专用文本渲染模型进行直接对比

<!-- RELATED:START -->

## 相关论文

- [Uni-Renderer: Unifying Rendering and Inverse Rendering via Dual Stream Diffusion](uni-renderer_unifying_rendering_and_inverse_rendering_via_dual_stream_diffusion.md)
- [Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)
- [TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering](../../CVPR2026/image_generation/textpecker_rewarding_structural_anomaly_quantification_for_enhancing_visual_text.md)
- [PICD: Versatile Perceptual Image Compression with Diffusion Rendering](picd_versatile_perceptual_image_compression_with_diffusion_rendering.md)
- [Progressive Tempering Sampler with Diffusion](../../ICML2025/image_generation/progressive_tempering_sampler_with_diffusion.md)

<!-- RELATED:END -->
