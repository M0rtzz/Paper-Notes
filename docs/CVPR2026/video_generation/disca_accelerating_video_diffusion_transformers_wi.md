---
title: >-
  [论文解读] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching
description: >-
  [CVPR 2026][视频生成][特征缓存] DisCa 首次将可学习特征缓存与步蒸馏统一为兼容框架，用轻量神经预测器（<4% 模型参数）替代手工缓存策略，配合 Restricted MeanFlow 稳定大规模视频 DiT 蒸馏，在 HunyuanVideo 上实现 11.8× 近无损加速。
tags:
  - "CVPR 2026"
  - "视频生成"
  - "特征缓存"
  - "步蒸馏"
  - "MeanFlow"
  - "可学习预测器"
  - "HunyuanVideo"
---

# DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching

**会议**: CVPR 2026  
**arXiv**: [2602.05449](https://arxiv.org/abs/2602.05449)  
**代码**: 即将公开  
**领域**: 视频生成 / 扩散模型加速  
**关键词**: 特征缓存, 步蒸馏, MeanFlow, 可学习预测器, HunyuanVideo

## 一句话总结

DisCa 首次将可学习特征缓存与步蒸馏统一为兼容框架，用轻量神经预测器（<4% 模型参数）替代手工缓存策略，配合 Restricted MeanFlow 稳定大规模视频 DiT 蒸馏，在 HunyuanVideo 上实现 11.8× 近无损加速。

## 研究背景与动机

**领域现状**：视频扩散模型（如 HunyuanVideo）生成质量已达 SOTA 水平，但推理极慢——以 HunyuanVideo 为例，50 步 CFG 推理生成一段 5 秒 704×704 视频需 1155 秒。现有加速手段主要有两条路线：减少采样步数的**步蒸馏**（如 MeanFlow），以及跳过冗余计算的**特征缓存**（如 TaylorSeer、TeaCache）。

**现有痛点**：步蒸馏方面，MeanFlow 在图像生成上表现优秀，但直接应用于大规模视频 DiT 时，JVP 运算的数值误差加上原始设计过于激进（目标一步生成），导致训练发散和严重伪影——10 步 MeanFlow 语义分直降 17.1%。特征缓存方面，传统方法依赖步间特征相似性进行复用或 Taylor 展开预测，但在步蒸馏后采样轨迹变得稀疏，相邻步特征差异剧增，简单手工策略完全失效——TaylorSeer 在高加速比场景下语义分也下降 13.3%。

**核心矛盾**：两种加速路线各有局限且难以兼容——蒸馏后的稀疏轨迹恰好破坏了缓存方法所依赖的步间冗余假设，简单叠加两者反而比单独使用更差。

**本文目标** 如何让步蒸馏和特征缓存两种加速策略真正兼容并互补，在大规模视频 DiT 上实现极致加速而不牺牲质量。

**切入角度**：用可学习的神经网络预测器替代手工缓存公式来捕捉高维特征演化；同时通过限制 MeanFlow 的压缩范围来稳定蒸馏过程。

**核心 idea**：蒸馏后的特征演化虽然超出 Taylor 展开等手工方法的建模能力，但轻量神经网络仍可准确学到这种高维演化规律。

## 方法详解

### 整体框架

DisCa 采用三级级联加速流水线：(1) **CFG 蒸馏**将双分支推理合为单分支（2× 加速）；(2) **Restricted MeanFlow** 步蒸馏进一步压缩采样步数（从 50 步到 10 步，约 5× 加速）；(3) **Learnable Feature Caching** 用轻量预测器在蒸馏后的稀疏轨迹上进一步跳步推理。三级叠加最终实现 11.8× 加速。推理时每 $N$ 步做一次完整 DiT 前向计算初始化缓存，中间 $N-1$ 步仅用轻量预测器快速生成输出。

### 关键设计

1. **Restricted MeanFlow（保守步蒸馏）**：

    - 功能：将原始 50 步采样压缩到 10-20 步，同时保持生成质量稳定
    - 核心思路：原始 MeanFlow 将平均速度区间 $\mathcal{I}=(t-r)$ 采样至 $[0,1]$，目标是一步生成。Restricted MeanFlow 引入限制因子 $\mathcal{R} \in (0,1)$，将区间约束为 $\mathcal{I} \in [0, \mathcal{R}]$，直接剪掉压缩比过高的训练样本。实验中 $\mathcal{R}=0.2$ 效果最佳
    - 设计动机：大规模视频 DiT 的高复杂度导致 MeanFlow 中 JVP 运算的数值误差放大，而过大的时间区间（高压缩比）使误差进一步累积。与其强行学习全局平均速度，不如稳定地学习局部平均速度，通过多步串联实现高质量生成

2. **Learnable Feature Caching（可学习特征缓存）**：

    - 功能：在蒸馏后的稀疏采样轨迹上，用预测器替代完整 DiT 计算，进一步加速推理
    - 核心思路：设计一个仅含 2 个 DiT Block 的轻量 Predictor $\mathcal{P}$（<4% 模型参数），接收上一步完整计算的缓存 $\mathcal{C}$ 和当前噪声输入 $x_{t'}$，预测当前步的平均速度输出。与 TaylorSeer 等需要维护多层多阶导数缓存（额外 33.5GB VRAM）不同，DisCa 仅保留最后一层的单个缓存张量（额外仅 0.43GB），学习能力替代了复杂缓存结构
    - 设计动机：蒸馏后步间特征差异巨大，超出 Taylor 展开等手工方法的建模能力上限。数据驱动的神经网络天然适合捕捉这种高维非线性演化趋势

3. **Memory-Efficient 缓存设计**：

    - 功能：大幅降低缓存的显存开销，使方法在高分辨率长视频场景中可用
    - 核心思路：不再为 DiT 每层维护多张量缓存（如 TaylorSeer），仅保留模型最终输出的单个张量作为缓存传入 Predictor
    - 设计动机：在实际分布式并行（sequence parallel size=4）环境下，多层缓存方案需要跨 GPU 同步缓存数据，通信开销反而抵消了计算节省。单张量缓存完全避免了这一问题

### 损失函数 / 训练策略

Predictor 训练采用 MSE + GAN 的两阶段策略：

- **MSE 阶段**（500 iter）：最小化预测器输出与大模型真实输出的 L2 距离
  $$\mathcal{L}_\mathcal{P} = \mathbb{E}\|\mathcal{M}_{\theta_M}(x_{t'}, r', t') - \mathcal{P}_{\theta_p}(\mathcal{C}, x_{t'}, r', t')\|_2^2$$
- **GAN 阶段**（1000 iter）：引入多尺度谱归一化判别器 $\mathcal{D}$，使用 Hinge Loss 进行对抗训练，强制预测器输出保留高频细节和视觉保真度。以大模型本身作为特征提取器 $\mathcal{F}$，在特征空间进行对抗
- 超参：预测器学习率 $10^{-4}$，判别器学习率 $10^{-2}$，对抗损失权重 $\lambda=1.0$

## 实验关键数据

### 主实验

实验在 HunyuanVideo 上进行，生成 704×704 分辨率、129 帧、5 秒视频，使用 VBench 评测。

**Restricted MeanFlow 对比**（与原始 MeanFlow 基线比）：

| 方法 | 步数 | 加速比 | 语义分↑ | 质量分↑ | 总分↑ |
|------|------|--------|---------|---------|-------|
| Original 50 步 | 50×2 | 1.0× | 73.5% | 81.5% | 79.9% |
| MeanFlow 20 步 | 20 | 4.96× | 66.6% | 81.8% | 78.8% |
| Restricted MeanFlow (R=0.2) 20 步 | 20 | 4.97× | **70.4% (+5.7%)** | 81.8% | 79.5% |
| MeanFlow 10 步 | 10 | 9.68× | 60.9% | 80.6% | 76.7% |
| Restricted MeanFlow (R=0.2) 10 步 | 10 | 9.68× | **68.2% (+12.0%)** | 81.3% | 78.7% |

**DisCa 与现有加速方法全面对比**：

| 方法 | 加速比 | Peak VRAM | 语义分↑ | 质量分↑ | 总分↑ |
|------|--------|-----------|---------|---------|-------|
| Original 50 步 | 1.0× | 99.23GB | 73.5% | 81.5% | 79.9% |
| Δ-DiT (N=8) | 4.55× | 97.68GB | 42.7% (-41.9%) | 70.9% | 65.2% |
| PAB (N=8) | 6.46× | 121.3GB | 56.3% (-23.4%) | 76.1% | 72.1% |
| TeaCache (l=0.4) | 9.22× | 97.70GB | 62.1% (-15.5%) | 78.7% | 75.4% |
| TaylorSeer (N=6) | 6.96× | 130.7GB | 63.7% (-13.3%) | 79.9% | 76.7% |
| FORA (N=6) | 8.01× | 124.6GB | 57.5% (-21.8%) | 76.4% | 72.6% |
| **DisCa (R=0.2, N=2)** | **7.56×** | **97.64GB** | **70.8% (-3.7%)** | **81.9%** | **79.7%** |
| **DisCa (R=0.2, N=3)** | **8.84×** | **97.64GB** | **70.3% (-4.4%)** | **81.8%** | **79.5%** |
| **DisCa (R=0.2, N=4)** | **11.8×** | **97.64GB** | **69.3% (-5.7%)** | **81.1%** | **78.8%** |

### 消融实验

| Restricted MeanFlow | Learnable Predictor | GAN Training | 语义分↑ | 质量分↑ | 总分↑ |
|:---:|:---:|:---:|---------|---------|-------|
| ✔ | ✔ | ✔ | 69.3% (+0.0%) | 81.1% (+0.0%) | 78.7% |
| ✘ | ✔ | ✔ | 65.2% (-5.9%) | 80.3% (-1.0%) | 77.3% |
| ✔ | ✘ | — | 67.3% (-2.9%) | 80.5% (-0.7%) | 77.9% |
| ✔ | ✔ | ✘ | 68.5% (-1.2%) | 81.0% (-0.1%) | 78.5% |

### 关键发现

- **Restricted MeanFlow 是基石**：不用 Restricted 直接在原始 MeanFlow 上训练缓存，语义分暴跌 5.9%，生成结果出现"完全不可接受的畸变"
- **可学习预测器 vs 免训练缓存**：即使在 Restricted MeanFlow 加持下，免训练缓存仍损失 2.9% 语义分和 0.7% 质量分——高维特征演化确实需要学习才能捕捉
- **GAN 训练不可或缺**：去掉对抗训练语义分下降 1.2%，说明 MSE 损失+对抗损失的组合对保持语义保真度至关重要
- **显存效率优势明显**：DisCa 仅需 97.64GB（额外 +0.43GB），而 TaylorSeer 需 130.7GB（+33.5GB）、FORA 需 124.6GB（+27.4GB）

## 亮点与洞察

- DisCa 首次证明步蒸馏和特征缓存可以互补而非冲突：关键在于用可学习预测器替代对步间冗余的硬性依赖，从而在蒸馏带来的稀疏采样轨迹上仍能有效加速。这为扩散模型加速开辟了"training-free + training-aware 协同"的新路线。
- Restricted MeanFlow 的设计极其朴素——仅仅限制训练时的时间区间采样范围——却在 10 步场景下带来 12.0% 的语义分提升。这揭示了一个重要直觉：对于大规模复杂模型的蒸馏，放弃极端压缩目标反而能获得全局更优的质量-速度权衡。
- 单张量缓存设计不仅节省显存，还在分布式并行环境下避免了跨 GPU 通信的延迟瓶颈，使得 DisCa 成为唯一在实际部署场景下同时满足显存和延迟约束的方案。

## 局限与展望

- 需要额外训练预测器和判别器（约 1500 iter），不再是完全免训练方案，每换一个基础模型或分辨率都需要重新训练
- 仅在 HunyuanVideo 上验证，对其他视频 DiT（CogVideoX、Wan 等）的迁移性未知
- 限制因子 $\mathcal{R}$ 需要手动调参（实验中 0.2 最优），缺乏自适应选择策略

## 相关工作与启发

- **vs TaylorSeer**：TaylorSeer 用 Taylor 展开预测缓存特征，在蒸馏模型的稀疏轨迹上效果大幅下降（-13.3% 语义分）且显存开销巨大（+33.5GB）。DisCa 用可学习预测器解决了建模能力和显存两个瓶颈
- **vs TeaCache**：TeaCache 用时间步嵌入做自适应缓存决策，但在高加速比下仍损失 15.5% 语义分。DisCa 在更高加速比（11.8× vs 9.22×）下仅损失 5.7%
- **vs MeanFlow**：原始 MeanFlow 为一步生成设计，对大规模视频模型过于激进。Restricted MeanFlow 以极简的区间限制策略实现了稳定蒸馏

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出蒸馏兼容的可学习缓存框架，将两大加速路线统一
- 实验充分度: ⭐⭐⭐⭐⭐ 在 HunyuanVideo 上全面对比 6 种方法，消融清晰，显存/延迟分析完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰、动机强，符号偏多但推导完整
- 价值: ⭐⭐⭐⭐⭐ 11.8× 近无损加速对视频生成实际部署价值极大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation](../../ICLR2026/video_generation/precisecache_precise_feature_caching_for_efficient_and_high-fidelity_video_gener.md)
- [\[ICML 2026\] WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](../../ICML2026/video_generation/worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)
- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](../../ICCV2025/video_generation/fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)
- [\[CVPR 2026\] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](../../ICML2026/video_generation/light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)

</div>

<!-- RELATED:END -->
