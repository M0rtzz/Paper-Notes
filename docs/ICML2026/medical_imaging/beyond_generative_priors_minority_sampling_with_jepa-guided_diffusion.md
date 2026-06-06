---
title: >-
  [论文解读] Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion
description: >-
  [ICML2026][医学图像][少数样本生成] 提出 JEPA Guidance，利用 JEPA（如 DINOv2）编码器的隐式密度信号引导扩散模型采样，将少数样本（minority sample）的定义从"生成模型先验下的低密度"转变为"世界先验下的低密度"…
tags:
  - "ICML2026"
  - "医学图像"
  - "少数样本生成"
  - "扩散模型"
  - "JEPA"
  - "世界模型先验"
  - "随机SVD"
---

# Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion

**会议**: ICML2026  
**arXiv**: [2605.24631](https://arxiv.org/abs/2605.24631)  
**代码**: https://github.com/soobin-um/jepa-guidance  
**领域**: 图像生成  
**关键词**: 少数样本生成, 扩散模型, JEPA, 世界模型先验, 随机SVD  

## 一句话总结

提出 JEPA Guidance，利用 JEPA（如 DINOv2）编码器的隐式密度信号引导扩散模型采样，将少数样本（minority sample）的定义从"生成模型先验下的低密度"转变为"世界先验下的低密度"，在无条件、类条件和文生图场景均实现更具语义意义的稀有样本生成。

## 研究背景与动机

**领域现状**：少数样本生成（minority sampling）旨在生成数据流形上低密度区域的实例，在医学诊断、异常检测和创意 AI 等领域有广泛需求。扩散模型凭借对复杂分布的建模能力，已成为该任务的主要框架，现有方法包括 classifier-guided、self-contained minority guidance 等。

**现有痛点**：所有现有方法都将"少数样本"定义为生成模型自身学到的密度 $p_\theta$ 下的低密度样本。这种 **generator-centric** 定义本质上绑定于训练数据，导致生成的"稀有样本"仅在特定模型下罕见（如白色背景上的狗），并不对应真实世界中语义上的稀有。

**核心矛盾**：生成器先验 $p_\theta$ 只捕捉了特定训练集的分布，无法反映更广泛的真实世界语义。当我们需要"世界级别"的稀有样本时（如隐身飞机、非典型人物），generator-centric 方法完全无能为力。

**本文目标**：将少数样本的定义从 generator-centric 转变为 **world-centric**——用独立于生成模型的世界先验来衡量稀有性，并在扩散采样中实现这一目标。

**切入角度**：JEPA（Joint-Embedding Predictive Architecture）如 DINOv2 在大规模数据上训练，其表征隐式编码了数据密度（JEPA-SCORE），可作为世界先验的自然代理。

**核心 idea**：用 JEPA 编码器的 Jacobian 奇异值来估计世界先验下的密度，通过随机 SVD 高效近似并以梯度引导扩散采样走向低密度区域，实现 world-centric minority sampling。

## 方法详解

### 整体框架

给定预训练扩散模型 $\epsilon_\theta$ 和 JEPA 编码器 $f_\phi$（如 DINOv2），在扩散逆过程采样时，间歇性地计算去噪估计 $\hat{x}_{0|t}$ 的 JEPA-SCORE 近似值，并用其负梯度引导采样走向世界先验下的低密度区域。整个过程无需训练，仅在推理时使用预训练模型。采样修改为：$x_{t-1} = \mu_\theta(x_t, t) + \Sigma_\theta^{1/2} z - \eta_t \nabla_{x_t} \text{JS}^*(\hat{x}_{0|t})$。

### 关键设计

1. **JEPA-SCORE 的随机 SVD 近似**:

    - 功能：将计算代价极高的 JEPA-SCORE 降到可实用的水平
    - 核心思路：JEPA-SCORE 定义为编码器 Jacobian $J_f(x) \in \mathbb{R}^{d \times n}$ 所有奇异值对数之和 $\text{JS}(x) = \sum_{i=1}^r \log(\sigma_i(J_f(x)))$，直接 SVD 代价过高。方法使用随机 SVD 构造低秩投影矩阵 $Q \in \mathbb{R}^{d \times l}$（$l \ll d$），将 Jacobian 压缩为 $\tilde{J}_f = Q^\top J_f \in \mathbb{R}^{l \times n}$，仅用前 $k$ 个奇异值近似。论文证明了近似误差的上界 $\text{JS} - \bar{\text{JS}} \leq \mathcal{E}_{\text{RSVD}} + \mathcal{E}_{\text{Trunc}}$，实验表明 $k \approx 10$ 即足够有效
    - 设计动机：原始 JEPA-SCORE 需要对大矩阵做 SVD，在扩散采样的每步迭代中不可行；随机 SVD 将复杂度从 $O(dn)$ 降至 $O(ln)$

2. **包络定理加速梯度计算**:

    - 功能：消除对随机 SVD 内部优化过程的反向传播，大幅降低内存和计算开销
    - 核心思路：直接对 $\bar{\text{JS}}(\hat{x}_{0|t})$ 关于 $x_t$ 求梯度需要通过 $Q$ 的计算图反传，因为 $Q$ 依赖于 $J_f(\hat{x}_{0|t})$ 从而间接依赖 $x_t$。利用包络定理（Envelope Theorem），在内层随机 SVD 达到最优时，可将最优投影 $Q^*$ 视为常数（stop-gradient），即 $\text{JS}^* = \sum_{i=1}^k \log(\tilde{\sigma}_i(\text{sg}(Q^{*\top}) J_f))$，保证一阶梯度正确的同时避免了对 SVD 过程的反传
    - 设计动机：朴素实现中 $Q$ 保留计算图导致内存爆炸；包络定理提供了理论保证，使得 stop-gradient 不损失梯度准确性

3. **延迟引导（Deferred Guidance）**:

    - 功能：解决 JEPA 编码器与扩散中间态之间的域差距，并使方法扩展到条件生成
    - 核心思路：将 JEPA 引导延迟到中间时步 $\tau T$（如 $\tau = 0.8$）之后才开始。早期采样步中 $\hat{x}_{0|t}$ 过于模糊噪声，与 JEPA 编码器期望的干净输入差距大。延迟引导让条件扩散模型先自由采样建立条件结构，再在后期引导走向低密度区域
    - 设计动机：实验表明不延迟（$\tau = 1.0$）时质量和文本对齐严重下降；延迟引导还天然解决了 JEPA 无法感知条件信息的问题——条件信息已在前期采样中融入

## 实验关键数据

### 主实验——无条件与类条件生成

| 数据集 | 方法 | cFID ↓ | sFID ↓ | Prec ↑ | Rec ↑ | JEPA-SCORE ↓ |
|--------|------|--------|--------|--------|-------|--------------|
| CelebA 64² | ADM | 12.11 | 6.35 | 0.85 | 0.57 | -221.67 |
| CelebA 64² | SGMS | 61.76 | 20.42 | 0.62 | 0.84 | -171.85 |
| CelebA 64² | **Ours** | **8.50** | **4.94** | **0.82** | 0.65 | **-300.79** |
| ImageNet 256² | ADM | 26.44 | 9.70 | 0.95 | 0.51 | -102.01 |
| ImageNet 256² | BnS | 32.01 | 10.61 | 0.92 | 0.56 | -125.77 |
| ImageNet 256² | **Ours** | **18.33** | **7.62** | 0.92 | **0.68** | **-241.62** |

### 文生图生成

| 模型 | 方法 | CLIP ↑ | PickScore ↑ | ImageReward ↑ | JEPA-SCORE ↓ |
|------|------|--------|-------------|---------------|--------------|
| SDv1.5 | DDIM | 31.52 | 21.49 | 0.21 | -292.27 |
| SDv1.5 | MinorityPrompt | 31.56 | 21.32 | 0.24 | -322.33 |
| SDv1.5 | **Ours** | 31.46 | **21.50** | 0.22 | **-355.40** |
| SDXL-Lightning | DDIM | 31.57 | 22.68 | 0.73 | -283.04 |
| SDXL-Lightning | MinorityPrompt | 31.36 | 22.62 | 0.71 | -302.17 |
| SDXL-Lightning | **Ours** | 31.52 | 22.63 | 0.73 | **-337.88** |

### 消融实验

| 配置 | CLIP ↑ | PickScore ↑ | JEPA-SCORE ↓ | 说明 |
|------|--------|-------------|--------------|------|
| $\tau = 1.0$（不延迟） | 31.26 | 21.33 | -356.22 | 质量严重下降 |
| $\tau = 0.9$ | 31.31 | 21.42 | -356.72 | 略有改善 |
| $\tau = 0.8$（默认） | 31.40 | 21.46 | -360.82 | 质量与稀有性最佳平衡 |
| $k = 3$ | 31.56 | 22.59 | -325.35 | 秩不足 |
| $k = 9$（默认） | 31.52 | 22.59 | -344.85 | 足够有效 |
| $k = 15$ | 31.53 | 22.58 | -335.28 | 边际收益递减 |

### 下游应用——数据增强分类

| 训练数据 | Acc ↑ | F1 ↑ | Prec ↑ | Rec ↑ | 增强量 |
|----------|-------|------|--------|-------|--------|
| CelebA trainset | 0.898 | 0.746 | 0.815 | 0.710 | — |
| + SGMS | 0.903 | 0.757 | 0.822 | 0.724 | 50K |
| + BnS | 0.902 | 0.755 | 0.819 | 0.723 | 50K |
| + **Ours** | **0.902** | **0.775** | **0.824** | **0.731** | **30K** |

## 亮点与洞察

- **范式转换**：将 minority sampling 从"在生成器分布下找罕见"重新定义为"在世界先验下找罕见"，概念上更合理——generator-centric 的稀有可能只是训练集偏差，world-centric 的稀有才反映真实语义
- **理论与工程并重**：随机 SVD 近似有严格误差上界（Proposition 4.1），包络定理保证梯度正确性，不是 ad-hoc hack
- **condition-agnostic 设计**：JEPA 编码器不需要知道条件信息（文本/类别），通过延迟引导自然兼容条件生成，设计极为优雅
- **数据效率**：下游分类中仅用 30K 增强样本超越 50K 的 baseline，说明世界先验下的稀有样本信息量更高

## 局限性 / 可改进方向

- 每个引导步需要计算 Jacobian + 随机 SVD，引入额外计算开销，可探索摊销化或更高效近似
- 世界先验的质量取决于 JEPA 编码器的训练数据和能力，换用不同编码器会改变"稀有"的定义
- 仅探索了 DINOv2/MetaCLIP 等编码器，未验证 V-JEPA 等视频模型或其他模态
- 反转引导方向可生成高密度样本强化偏见，存在双用途风险

## 相关工作与启发

- **Minority Sampling 系列**（Sehwag et al., Um et al.）：从 classifier-guided → self-contained → guidance-free 的演进，本文突破了"只能在生成器先验下定义少数"的根本限制
- **JEPA-SCORE**（Balestriero et al., 2025）：证明 JEPA 表征隐式编码数据密度，本文将其从后验排序工具升级为在线采样引导信号
- **DINOv2**（Oquab et al., 2023）：1.42 亿图像训练的 ViT 编码器，充当世界先验的代理
- **启发**：可将此框架推广到其他需要"定义什么是稀有"的场景，如公平性、鲁棒性测试、创意内容生成

## 评分
- 新颖性: 9/10 — 从 generator-centric 到 world-centric 的范式转换，概念贡献突出
- 实验充分度: 8/10 — 覆盖无条件/类条件/文生图/下游应用，消融详尽
- 写作质量: 9/10 — 概念清晰，理论推导严谨，图示直观
- 价值: 8/10 — 为 minority sampling 打开新方向，但计算开销限制实际规模化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[CVPR 2026\] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](../../CVPR2026/medical_imaging/solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_imaging/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](../../CVPR2026/medical_imaging/multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)

</div>

<!-- RELATED:END -->
