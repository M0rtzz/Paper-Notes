---
title: >-
  [论文解读] LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation
description: >-
  [NeurIPS 2025][扩散模型] 提出 LeMiCa，一种免训练的扩散视频生成加速框架，将缓存调度建模为有向无环图上的字典序极小极大路径优化问题，通过全局误差控制实现速度和质量的双重提升（Latte 上 2.9× 加速，Open-Sora 上 LPIPS 低至 0.05）。
tags:
  - NeurIPS 2025
  - 扩散模型
  - video generation
  - caching
  - DAG optimization
  - lexicographic minimax
---

# LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation

**会议**: NeurIPS 2025  
**arXiv**: [2511.00090](https://arxiv.org/abs/2511.00090)  
**代码**: [GitHub](https://github.com/UnicomAI/LeMiCa)  
**领域**: video_understanding  
**关键词**: diffusion acceleration, video generation, caching, DAG optimization, lexicographic minimax

## 一句话总结

提出 LeMiCa，一种免训练的扩散视频生成加速框架，将缓存调度建模为有向无环图上的字典序极小极大路径优化问题，通过全局误差控制实现速度和质量的双重提升（Latte 上 2.9× 加速，Open-Sora 上 LPIPS 低至 0.05）。

## 研究背景与动机

- **扩散模型推理代价高**：DiT 架构虽大幅提升视频质量，但多步去噪导致计算量大、延迟高，限制了交互式应用。
- **现有缓存方法的局限 — 局部贪心策略**：TeaCache 等 SOTA 方法基于相邻时步输出的局部差异（$L1_{rel}$）和固定阈值决定是否缓存。但扩散去噪过程具有**时序异质性**——早期步骤塑造全局结构，晚期步骤细化细节——统一阈值会导致语义不一致。
- **局部误差累积导致全局退化**：最小化相邻步差异（Local-Greedy error）忽略了小误差沿去噪轨迹的累积效应，最终导致视频质量和内容一致性的双重损失。
- **现有加速方法需要训练或精巧工程**：蒸馏、剪枝、量化等方法需大量数据和重训练，缓存方法作为免训练替代方案有天然优势。

## 方法详解

### 重新审视扩散采样中的缓存

**局部贪心误差**（现有方法）：基于相邻步输出的相对 L1 距离做决策：

$$L1_{rel}(O, t) = \frac{\|O_t - O_{t+1}\|_1}{\|O_{t+1}\|_1}$$

**全局结果感知误差**（本文提出）：度量每个缓存段对最终输出的影响：

$$L1_{glob}(i \to j) = \frac{1}{N} \|x_0^{\text{cache}(i \to j)} - x_0^{\text{original}}\|_1$$

关键发现：① 全局误差传播是非均匀且时间依赖的——早期缓存引起的下游误差被指数放大；② 缓存段的时序位置比长度更重要。

### 字典序极小极大路径缓存（LeMiCa）

**图构建**：构造有向无环图（DAG），每条边代表一个候选缓存段，边权值为全局结果感知误差。为控制复杂度，设最大跳跃长度限制。使用多个 prompt 和随机种子离线构建静态图（70 个 prompt × 10 种种子，取平均）。

**图优化**：在固定推理预算 $B$（模型前向步数）下，寻找从源到汇的最优路径。采用字典序极小极大准则：

$$\min_{P \in \mathcal{P}_{s \to t}^{(B)}} \text{LexMax}\big(\text{sort\_desc}(\{w(e) \mid e \in P_{\text{cache}}\})\big)$$

即在所有可行路径中，首先最小化最大缓存误差；若相同则比较次大误差，依此类推。这比最短路径策略更鲁棒——因为缓存误差非独立累加，极端误差会对结果产生不成比例的破坏。

**为何不用最短路径**：早期缓存误差在去噪过程中指数放大，传统最短路径最小化累加代价、无法控制单步峰值误差。字典序极小极大显式约束最坏情况退化。

### 关键特性

- **免训练**：缓存策略离线预计算，无运行时开销
- **模型无关**：适用于 Open-Sora、Latte、CogVideoX 等不同架构
- **静态缓存策略鲁棒**：仅需 1-20 个样本即可构建高质量 DAG

## 实验

### 表1：推理效率与视觉质量对比（单 GPU）

| 方法 | 模型 | 加速比↑ | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| TeaCache-slow | Open-Sora | 1.50× | 79.20% | 0.134 | 0.837 | 23.50 |
| **LeMiCa-slow** | Open-Sora | **1.52×** | **79.26%** | **0.050** | **0.923** | **31.32** |
| TeaCache-fast | Open-Sora | 2.10× | 78.24% | 0.252 | 0.743 | 19.03 |
| **LeMiCa-fast** | Open-Sora | **2.44×** | **78.34%** | **0.187** | **0.798** | **21.76** |
| TeaCache-slow | Latte | 1.65× | 77.40% | 0.195 | 0.775 | 21.52 |
| **LeMiCa-slow** | Latte | **1.69×** | **77.45%** | **0.091** | **0.865** | **27.65** |
| TeaCache-fast | Latte | 2.60× | 76.09% | 0.318 | 0.674 | 18.04 |
| **LeMiCa-fast** | Latte | **2.93×** | **76.75%** | **0.273** | **0.700** | **19.43** |
| TeaCache-slow | CogVideoX | 1.70× | 76.79% | 0.053 | 0.928 | 31.07 |
| **LeMiCa-slow** | CogVideoX | **1.72×** | **76.89%** | **0.023** | **0.958** | **35.93** |

LeMiCa 在所有模型上一致超越 TeaCache。LeMiCa-slow 在 Open-Sora 上 LPIPS 从 0.134 降至 0.050（2.7× 改进），在 CogVideoX 上 LPIPS 仅 0.023。LeMiCa-fast 在 Latte 上达到 2.93× 加速（从 TeaCache 的 2.60×）。

### 表2：DAG 构建所需样本量

| 样本数 | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 78.58 | 0.164 | 0.838 | 24.51 |
| 5 | 78.70 | 0.161 | 0.843 | 24.57 |
| 10 | 78.95 | 0.158 | 0.844 | 24.56 |
| 350 | 79.27 | 0.143 | 0.851 | 24.67 |

仅 1 个样本即可获得强结果（PSNR 24.51 vs 350 样本的 24.67），证明静态缓存策略的鲁棒性。

### 表3：路径策略消融

| 路径策略 | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|:---|:---:|:---:|:---:|:---:|
| 最短路径 | 76.04 | 0.203 | 0.809 | 22.90 |
| **极小极大路径** | **79.27** | **0.143** | **0.851** | **24.67** |

极小极大路径全面优于最短路径，验证了"控制峰值误差"比"最小化累加误差"更重要的核心论点。

### 轨迹鲁棒性

改变采样调度的 scale 参数（0.5~1.5），LeMiCa 在所有轨迹上 LPIPS 一致优于 TeaCache，证明方法在不同去噪路径下保持有效。

## 亮点

- **理论视角新颖**：将缓存调度形式化为 DAG 上的字典序极小极大路径问题，为扩散加速提供了新的优化范式
- **全局误差控制**：全局结果感知误差消除了时序异质性的影响，字典序极小极大约束最坏情况
- **极低感知退化**：Open-Sora 上 LPIPS 0.05、CogVideoX 上 LPIPS 0.023，接近无损加速
- **零额外运行时开销**：缓存策略完全离线预计算，推理时直接查表

## 局限性

- **DAG 构建需离线计算**：需要完整去噪轨迹 + 多 prompt 评估，初次构建有一定计算成本
- **假设固定采样路径**：对于训练好的模型采样路径稳定的假设在某些自适应调度器下可能不成立
- **仅验证文本到视频**：未在图生视频、视频编辑等更多任务上验证
- **最大跳跃长度需手动设定**：基于启发式，未给出自动确定方法

## 相关工作

- **扩散加速**（DDIM [Song+ 2020]、一致性蒸馏 [Song+ 2023]、对抗蒸馏 [Sauer+ 2024]）：需要训练，LeMiCa 免训练
- **扩散缓存**（DeepCache [Ma+ 2023]、PAB [Zhao+ 2024]、TeaCache [Liu+ 2024]）：基于局部贪心策略，LeMiCa 提出全局优化替代
- **Δ-DiT [Chen+ 2024]、T-GATE [Zhang+ 2024]**：图像生成加速方法，扩展到视频时效果有限

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 字典序极小极大路径优化是扩散缓存领域的全新视角
- 实验充分度: ⭐⭐⭐⭐ — 三个模型 + 多项消融 + 样本效率 + 轨迹鲁棒性
- 写作质量: ⭐⭐⭐⭐ — 局部 vs 全局误差的分析深入，图示清晰
- 价值: ⭐⭐⭐⭐⭐ — 为扩散视频生成提供了强大的免训练加速基线

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation](../../ICLR2026/video_generation/precisecache_precise_feature_caching_for_efficient_and_high-fidelity_video_gener.md)
- [Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing](../../ICML2025/video_generation/ca2-vdm_efficient_autoregressive_video_diffusion_model_with_causal_generation_an.md)
- [V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)

<!-- RELATED:END -->
