---
title: >-
  [论文解读] LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration
description: >-
  [CVPR 2026][图像生成][扩散模型加速] 提出 LESA 框架，用 KAN（Kolmogorov-Arnold Network）作为可学习时序预测器，结合多阶段多专家架构和两阶段训练策略，在 FLUX 上实现 5× 加速仅 1.0% 质量下降，在 Qwen-Image 上 6.25× 加速比 TaylorSeer 质量提升 20.2%，在 HunyuanVideo 上 5× 加速 PSNR 提升 24.7%。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型加速"
  - "特征缓存"
  - "KAN"
  - "阶段感知"
  - "DiT"
  - "文生图"
  - "文生视频"
---

# LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration

**会议**: CVPR 2026  
**arXiv**: [2602.20497](https://arxiv.org/abs/2602.20497)  
**领域**: 图像生成 / 扩散模型加速  
**关键词**: 扩散模型加速, 特征缓存, KAN, 阶段感知, DiT, 文生图, 文生视频

## 一句话总结
提出 LESA 框架，用 KAN（Kolmogorov-Arnold Network）作为可学习时序预测器，结合多阶段多专家架构和两阶段训练策略，在 FLUX 上实现 5× 加速仅 1.0% 质量下降，在 Qwen-Image 上 6.25× 加速比 TaylorSeer 质量提升 20.2%，在 HunyuanVideo 上 5× 加速 PSNR 提升 24.7%。

## 研究背景与动机

**领域现状**：DiT 架构在图像/视频生成中取得卓越效果，但深层 Transformer + 多步去噪的推理开销巨大。特征缓存（Feature Caching）利用相邻时步间的时序冗余加速推理，是近年热门方向。

**现有方案分类**：(a) Cache-Then-Reuse（直接复用上一步特征，如 PAB、DBCache）——忽略时序演化致误差累积；(b) Cache-Then-Forecast（用 Taylor 展开预测，如 TaylorSeer、FORA）——假设特征平滑连续演化，但实际扩散过程是**阶段依赖**的。

**关键观察**：通过余弦相似度和 PCA 轨迹分析，扩散过程特征动态呈三阶段：高噪声阶段变化剧烈不稳定，中间阶段稳定连续，低噪声阶段精修细节。单一固定预测策略无法适应。

**核心idea**：用**可学习的** KAN 代替训练无关的多项式预测，并为每个噪声阶段分配**专用专家预测器**，显式建模阶段依赖动态。

## 方法详解

### 整体框架

LESA 想解决的是 DiT 扩散推理太慢、但相邻时步特征又高度冗余的矛盾。它的做法是在去噪轨迹上每隔 N 步才让 DiT 正式算一次（锚点），其余时步交给一个轻量的可学习预测器，根据缓存的历史 K 步特征直接外推出当前特征、跳过整次 DiT 前向。关键在于这个预测器不是一个固定的多项式，而是按噪声阶段分工的多个 KAN 专家，从而显式贴合扩散过程「高噪声剧变、中段平稳、低噪声精修」的非均匀动态。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["去噪时步 t<br/>+ 缓存历史 K 步特征"] --> B{"每隔 N 步<br/>是否锚点步"}
    B -->|是| C["DiT 完整前向<br/>算真实特征并缓存"]
    B -->|否, 走预测器| D["阶段感知多专家<br/>按噪声水平选专家"]
    D -->|高噪声 K=4| E["KAN 时序建模<br/>线性投影压空间方向 z<br/>KAN 吃 Δt 出标量 α<br/>残差预测 ĥ = h + α·z"]
    D -->|中噪声| E
    D -->|低噪声 K=8| E
    C --> G["输出特征 → 下一时步"]
    E --> G
    subgraph T["两阶段训练（离线）"]
        direction TB
        H["GT-Guided：真实特征 L1 监督"] --> I["CL-AR 闭环自回归<br/>带误差预测喂回当输入"]
    end
    T -. 训好的专家权重 .-> D
```

### 关键设计

**1. KAN 时序建模：用可学习函数代替训练无关的 Taylor 外推**

Cache-Then-Forecast 一类方法（TaylorSeer、FORA）默认特征沿时间平滑连续、可用 Taylor 展开预测，但扩散特征实际是阶段依赖的，固定多项式贴合不了。LESA 把预测拆成空间和时间两个正交方向：先用线性投影在空间域把缓存的 K 步特征压成一个特征方向 $\mathbf{z} = \mathbf{W}[\mathbf{h}_{t+K-1},...,\mathbf{h}_t] + b$，再让 KAN 在时间域吃相对时步偏移 $\Delta t$、输出一个标量时序调制因子 $\alpha = f_{KAN}(\Delta t_{L-1},...,\Delta t_0) = \sum_{m=1}^{M} w_m \phi_m(a_m^\top \Delta \mathbf{t})$，最后做残差预测 $\hat{\mathbf{h}}_{t-1} = \mathbf{h}_t + \alpha \mathbf{z}$。这种「线性投影管空间、KAN 管时间」的分离变量让 KAN 借 Kolmogorov-Arnold 表示定理用一组可学习的单变量基函数 $\phi_m$ 拟合时序规律，比 Taylor 更灵活、又比 MLP 参数省得多——每个专家只有线性投影加一个小 KAN 模块，额外参数远小于基础 DiT。

**2. 阶段感知多专家：一个预测器管不了整条去噪轨迹**

作者用余弦相似度和 PCA 轨迹分析发现，扩散特征动态明显分三段：高噪声段剧烈不稳、中段平稳连续、低噪声段精修细节，单一预测策略顾此失彼。LESA 因此按噪声水平把去噪过程切成三段、各配一个独立专家：高噪声专家用较短窗口 $K=4$ 跟上初期剧变，中噪声专家负责平稳段的连续去噪，低噪声专家用较长窗口 $K=8$ 稳住细节精修。每段都用最贴合自己动态的窗口和参数，避免了「短窗口在平稳段浪费、长窗口在剧变段滞后」的两难。

**3. 两阶段训练：先学对、再学稳**

预测器若只用真值监督训练，推理时一旦自己的预测带误差就会逐步漂移、误差累积。LESA 先做 GT-Guided Training：拿基础模型无加速跑出的真实中间特征当监督、用 L1 损失把预测器训到「算得准」；再做 CL-AR（闭环自回归）训练：让预测器把自己上一步带误差的预测重新喂回当输入，主动模拟推理时的误差累积链路，逼它学会对预测漂移鲁棒。这一步是工程关键——只用 GT 训练的模型推理时退化很快，闭环训练才把误差累积压住。

## 实验关键数据

### 主实验：FLUX.1-dev 文生图

| 方法 | FLOPs加速 | ImageReward↑ | CLIP Score↑ | PSNR↑ | LPIPS↓ |
|------|----------|-------------|-------------|-------|--------|
| 原始 50步 | 1.00× | 0.99 | 32.64 | ∞ | 0.00 |
| TaylorSeer (N=6,O=2) | 4.99× | 1.02 | 32.53 | 28.94 | 0.40 |
| TeaCache (l=1.0) | 4.54× | 0.84 | 31.88 | 28.61 | 0.48 |
| **LESA (N=7)** | **5.00×** | **0.98** | **32.88** | **30.17** | **0.32** |
| TaylorSeer (N=9,O=2) | 6.24× | 0.86 | 32.04 | 28.38 | 0.51 |
| **LESA (N=10)** | **6.25×** | **0.91** | **32.65** | **29.65** | **0.40** |

### 主实验：Qwen-Image 文生图

| 方法 | FLOPs加速 | ImageReward↑ | PSNR↑ | LPIPS↓ |
|------|----------|-------------|-------|--------|
| TaylorSeer (N=6,O=2) | 5.00× | 1.01 | 28.58 | 0.46 |
| **LESA (N=7)** | **5.00×** | **1.15** | **30.18** | **0.25** |
| TaylorSeer (N=8,O=2) | 6.24× | 0.84 | 28.14 | 0.68 |
| **LESA (N=10)** | **6.25×** | **1.01** | **29.23** | **0.34** |

### 主实验：HunyuanVideo 文生视频

| 方法 | FLOPs加速 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|----------|-------|-------|--------|
| TaylorSeer (N=5,O=1) | 5.00× | 17.29 | 0.55 | 0.42 |
| TeaCache (l=0.4) | 4.55× | 18.25 | 0.61 | 0.38 |
| **LESA (N=7)** | **5.00×** | **21.43** | **0.72** | **0.29** |
| **LESA (N=8)** | **5.56×** | **21.05** | **0.70** | **0.32** |

### 消融实验（FLUX, N=5）

| 阶段分割 | 时步模块 | PSNR↑ | LPIPS↓ |
|----------|----------|-------|--------|
| ✗ | MLP | 30.29 | 0.31 |
| ✗ | KAN | 30.77 | 0.25 |
| ✔ | MLP | 30.76 | 0.25 |
| **✔** | **KAN** | **30.96** | **0.24** |

### 关键发现
- 5× 加速下 FLUX ImageReward 仅下降 1.0%（0.99→0.98），**几乎无损加速**
- Qwen-Image 上 6.25× 加速 ImageReward 反超 TaylorSeer 20.2%（0.84→1.01）
- HunyuanVideo 上 5× 加速 PSNR 超 TaylorSeer 24.7%（17.29→21.43）
- KAN + 阶段分割的组合效果最佳，两者缺一不可
- 蒸馏模型（FLUX-schnell、Qwen-Lightning）同样有效
- 高加速比（6×+）下 training-free 方法严重退化（†），LESA 仍保持竞争力

## 亮点与洞察
- **KAN 做时序建模**：利用 KAR 定理的函数表示能力做可学习时序外推，比 Taylor 更灵活、比 MLP 更参数高效
- **阶段感知设计**显式建模扩散过程的非均匀动态，符合直觉且实验有效
- **闭环自回归训练**是工程关键：GT 训练的模型推理时误差快速累积，闭环训练显著提升鲁棒性
- 跨模型（FLUX/Qwen/HunyuanVideo）和跨任务（T2I/T2V）泛化能力强

## 局限性
- 需预先运行基础模型收集特征轨迹做训练数据
- 每个新模型需重新训练预测器，不完全即插即用
- 阶段划分边界需人工设定，未探索自适应分割

## 评分
- 新颖性: ⭐⭐⭐⭐ KAN+阶段感知预测器是新组合
- 实验充分度: ⭐⭐⭐⭐⭐ 3个T2I模型+1个T2V模型+蒸馏模型+完整消融
- 写作质量: ⭐⭐⭐⭐ 分析扎实图表清晰
- 实用价值: ⭐⭐⭐⭐⭐ 实际推理加速效果显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration](tc-padé_trajectory-consistent_padé_approximation_for_diffusion_acceleration.md)
- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[CVPR 2026\] Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models](flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)
- [\[AAAI 2026\] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration](../../AAAI2026/image_generation/procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)

</div>

<!-- RELATED:END -->
