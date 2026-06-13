---
title: >-
  [论文解读] From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers
description: >-
  [ICCV 2025][图像生成][扩散模型加速] 提出 TaylorSeer，将扩散模型特征缓存范式从"缓存-重用"升级为"缓存-预测"——利用 Taylor 级数展开基于历史特征的高阶有限差分来预测未来时间步的中间特征，在 FLUX 上实现近乎无损的 4.99× 加速、在 HunyuanVideo 上实现 5.00× 加速，且完全无需额外训练。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "扩散模型加速"
  - "特征缓存"
  - "Taylor展开"
  - "免训练加速"
  - "DiT"
  - "FLUX"
  - "HunyuanVideo"
---

# From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers

**会议**: ICCV 2025  
**arXiv**: 2503.06923  
**代码**: [https://github.com/Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)  
**领域**: 图像生成  
**关键词**: 扩散模型加速, 特征缓存, Taylor展开, 免训练加速, DiT, FLUX, HunyuanVideo

## 一句话总结

提出 TaylorSeer，将扩散模型特征缓存范式从"缓存-重用"升级为"缓存-预测"——利用 Taylor 级数展开基于历史特征的高阶有限差分来预测未来时间步的中间特征，在 FLUX 上实现近乎无损的 4.99× 加速、在 HunyuanVideo 上实现 5.00× 加速，且完全无需额外训练。

## 研究背景与动机

Diffusion Transformer（DiT）在高保真图像和视频生成上取得了革命性进展，但其巨大的计算需求仍是实时应用的主要瓶颈。

**特征缓存方法的自然局限**：现有特征缓存方法（如 FORA、Δ-DiT、TeaCache 等）遵循"缓存-重用"范式——将前一时间步计算的特征直接复用到后续时间步。这在相邻时间步间效果不错，但存在一个根本性限制：**随着时间步间距增大，特征相似度指数级下降**，导致直接重用引入的误差急剧增加，严重损害生成质量。这把特征缓存方法锁死在低加速比范围内。

**关键观察**：通过 PCA 可视化扩散模型不同时间步的特征，作者发现：
1. 特征在不同时间步形成**稳定的轨迹**（trajectory），说明未来特征是可预测的
2. 特征的**导数**（即轨迹上的速度）在相邻时间步也具有高度稳定性和连续性

这意味着预测未来特征并非复杂问题，甚至可以用非参数方法解决。

## 方法详解

### 整体框架：从"重用"到"预测"

TaylorSeer 将扩散模型的特征缓存从直接复制升级为基于 Taylor 级数的轨迹预测。核心思想是利用多步历史特征来近似计算各阶导数，然后通过 Taylor 展开预测未来时间步的特征。

### 阶层预测公式

对于 $(m+1)$ 次可微的特征函数 $\mathcal{F}(x_t^l)$，未来时间步 $t-k$ 的特征可通过 Taylor 展开表示：

$$\mathcal{F}(x_{t-k}^l) = \mathcal{F}(x_t^l) + \sum_{i=1}^{m} \frac{\mathcal{F}^{(i)}(x_t^l)}{i!}(-k)^i + R_{m+1}$$

为避免显式计算高阶导数，使用**有限差分**递归近似：

$$\Delta^i \mathcal{F}(x_t^l) = \Delta^{i-1}\mathcal{F}(x_{t+N}^l) - \Delta^{i-1}\mathcal{F}(x_t^l)$$

其中 $i$ 阶有限差分近似 $i$ 阶导数缩放 $N^i$：$\Delta^i \mathcal{F}(x_t^l) \approx N^i \mathcal{F}^{(i)}(x_t^l)$

代入 Taylor 展开得到 **$m$ 阶预测公式**：

$$\mathcal{F}_{\text{pred},m}(x_{t-k}^l) = \mathcal{F}(x_t^l) + \sum_{i=1}^{m} \frac{\Delta^i \mathcal{F}(x_t^l)}{i! \cdot N^i}(-k)^i$$

只需 $(m+1)$ 个完整计算的时间步 $\{t+mN, \dots, t+N, t\}$ 即可预测中间时间步特征。

### 统一视角

- **$m=0$**：退化为朴素特征缓存（直接重用）
- **$m=1$**：线性预测，使用一阶有限差分捕获线性趋势
- **$m \geq 2$**：高阶预测，捕获非线性轨迹动态，减少长距离误差

### 误差界分析

预测误差有严格的理论上界：

$$E_m(k) \leq \frac{M_{m+1}}{(m+1)!}|k|^{m+1} + \sum_{i=1}^{m}\frac{C_i}{i! \cdot |N|^{i-1}}|k|^i$$

揭示了阶数与误差的基本权衡：更高阶有效减少主误差项，但引入额外的有限差分近似误差。

## 实验关键数据

### 主实验：FLUX 文本到图像生成 (Image Reward)

| 方法 | 加速比 | Image Reward ↑ | CLIP ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|
| FLUX 原始 (50步) | 1.00× | 0.9898 | 19.604 | — | — | — |
| Δ-DiT (N=3) | 1.95× | 0.8561 | 18.833 | 28.794 | 0.6665 | 0.4133 |
| FORA (N=3) | 2.82× | 0.9227 | 18.950 | 30.652 | 0.7666 | 0.2450 |
| DuCa (N=5) | 3.45× | 0.9896 | 19.595 | 29.413 | 0.7142 | 0.3082 |
| **TaylorSeer (N=3,O=2)** | **2.82×** | **1.0181** | **19.397** | **30.762** | **0.7818** | **0.2300** |
| FORA (N=6) | 4.99× | 0.7761 | 17.986 | 28.360 | 0.6001 | 0.5177 |
| DuCa (N=6) | 4.56× | 0.9470 | 19.082 | 28.672 | 0.6228 | 0.4182 |
| **TaylorSeer (N=6,O=2)** | **4.99×** | **1.0039** | **19.427** | **28.945** | **0.6556** | **0.4020** |

在 4.99× 加速下，TaylorSeer 的 Image Reward **超过原始模型**，而所有竞争方法均严重退化。

### 消融/对比：DiT-XL/2 类条件图像生成 (FID-50k)

| 方法 | 加速比 | FID ↓ | sFID ↓ | IS ↑ |
|---|---|---|---|---|
| DDIM-50步 | 1.00× | 2.32 | 4.32 | 241.25 |
| FORA (N=3) | 2.77× | 3.55 | 6.36 | 229.02 |
| DuCa (N=3) | 2.48× | 2.88 | 4.66 | 233.37 |
| **TaylorSeer (N=3,O=3)** | **2.77×** | **2.34** | **4.69** | **238.42** |
| FORA (N=5) | 4.53× | 6.58 | 11.29 | 193.01 |
| DuCa (N=5) | 3.78× | 6.06 | 6.72 | 198.46 |
| **TaylorSeer (N=5,O=3)** | **4.53×** | **2.65** | **5.36** | **231.59** |

在 4.53× 加速下，TaylorSeer 的 FID 仅 2.65（对比 FORA 的 6.58 和 DuCa 的 6.06），**比之前 SOTA 低 3.41**。

### HunyuanVideo 视频生成

| 方法 | 加速比 | VBench ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| 原始 (50步) | 1.00× | 80.66 | — | — | — |
| FORA (N=5) | 5.00× | 78.83 | 16.072 | 0.6334 | 0.3457 |
| TeaCache (l=0.4) | 4.55× | 79.36 | 16.072 | 0.6216 | 0.4377 |
| **TaylorSeer (N=5,O=1)** | **5.00×** | **79.93** | **16.796** | **0.7039** | **0.2691** |

### 关键发现

1. **高加速比下的绝对优势**：在 >4× 加速下，所有"缓存-重用"方法明显退化，而 TaylorSeer 仍保持接近原始质量
2. **效率优于质量损失**：比之前 SOTA 的质量损失降低 **36 倍**
3. **在 6× 加速范围仍可用**：所有先前方法在 >6× 时完全失败，TaylorSeer 仍能生成可接受结果
4. **Image Reward 可超越原始模型**：在某些配置下（如 N=4,O=2），TaylorSeer 的生成质量反而优于未加速的原始模型

## 亮点与洞察

1. **范式创新**：从"缓存-重用"到"缓存-预测"的范式转换是本文最大贡献——不仅是增量改进，而是开辟了新方向
2. **数学优雅性**：Taylor 级数展开为特征预测提供了统一且优美的数学框架，从直接缓存到高阶预测都纳入同一公式
3. **完全免训练**：无需搜索、无需额外训练成本，即插即用
4. **图像+视频双验证**：在 FLUX（图像）和 HunyuanVideo（视频）上都实现了近乎无损的 ~5× 加速，说明方法的通用性

## 局限性

1. **需要额外缓存空间**：高阶预测需要存储多个时间步的特征和有限差分，内存开销随阶数增加
2. **Assumption 1 的强度**：依赖特征函数可微且高阶导数有界的假设，当扩散过程在某些时间步存在不连续变化时可能失效
3. **阶数选择需要调参**：不同模型和加速比下最优的 Taylor 阶数不同（如 FLUX 用 O=2、DiT 用 O=3/4）
4. **未在高分辨率超长视频上验证**：HunyuanVideo 实验的分辨率和帧数有限

## 相关工作与启发

- **与 ODE 求解器加速的互补**：TaylorSeer 加速的是去噪网络的计算，与 DPM-Solver 等减少采样步数的方法正交，可组合使用
- **特征轨迹的连续性**：特征在时间步间的光滑变化是一个深刻的物理直觉——扩散模型的 SDE 本身就定义了光滑的正反过程，特征的连续性只是其自然结果
- **对未来缓存方法的启示**：未来的缓存方法可能从纯数值方法升级为学习型预测（如用小型 MLP 预测特征变化），进一步提升精度

## 评分

⭐⭐⭐⭐⭐ (5/5)

- **创新性**: ⭐⭐⭐⭐⭐ — 范式级创新，优雅简洁
- **实验完整性**: ⭐⭐⭐⭐⭐ — 覆盖图像（FLUX、DiT）和视频（HunyuanVideo），多种加速比、多种基线
- **实用性**: ⭐⭐⭐⭐⭐ — 免训练、即插即用，5× 加速近乎无损
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，图表直观

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Accelerating Diffusion Sampling via Exploiting Local Transition Coherence](accelerating_diffusion_sampling_via_exploiting_local_transition_coherence.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[AAAI 2026\] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](../../AAAI2026/image_generation/specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)
- [\[CVPR 2026\] SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](../../CVPR2026/image_generation/seacache_spectral-evolution-aware_cache_for_accelerating_diffusion_models.md)
- [\[ICML 2025\] Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization](../../ICML2025/image_generation/modulated_diffusion_accelerating_generative_modeling_with_modulated_quantization.md)

</div>

<!-- RELATED:END -->
