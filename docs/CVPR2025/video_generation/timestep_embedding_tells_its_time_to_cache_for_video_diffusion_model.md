---
title: >-
  [论文解读] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model
description: >-
  [CVPR 2025][视频生成][视频扩散模型] 本文提出 TeaCache，一种免训练的视频扩散模型缓存加速方法，通过时间步嵌入调制噪声输入来估计相邻时间步模型输出的差异，配合多项式拟合进行缩放校准，从而自适应决定何时缓存/复用输出，在 Open-Sora-Plan 上实现 4.41× 加速且视觉质量几乎无损（VBench 仅降 0.07%）。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频扩散模型"
  - "缓存加速"
  - "时间步嵌入"
  - "免训练加速"
  - "DiT"
---

# Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model

**会议**: CVPR 2025  
**arXiv**: [2411.19108](https://arxiv.org/abs/2411.19108)  
**代码**: [https://liewfeng.github.io/TeaCache](https://liewfeng.github.io/TeaCache)  
**领域**: 图像/视频生成 / 扩散模型加速  
**关键词**: 视频扩散模型, 缓存加速, 时间步嵌入, 免训练加速, DiT

## 一句话总结

本文提出 TeaCache，一种免训练的视频扩散模型缓存加速方法，通过时间步嵌入调制噪声输入来估计相邻时间步模型输出的差异，配合多项式拟合进行缩放校准，从而自适应决定何时缓存/复用输出，在 Open-Sora-Plan 上实现 4.41× 加速且视觉质量几乎无损（VBench 仅降 0.07%）。

## 研究背景与动机

**领域现状**：DiT（Diffusion Transformer）已成为视频生成的核心骨干网络，但推理速度慢是阻碍广泛应用的主要瓶颈。去噪过程的序贯性质限制了并行化能力，随着模型参数量增大和视频分辨率/长度提高，问题愈发严重。

**现有痛点**：现有加速方法分两类：(1) 蒸馏/后训练——需要大量额外训练成本；(2) 缓存机制（如 PAB、FORA）——免训练但采用均匀缓存策略，在等间隔的时间步缓存和复用模型输出。然而，相邻时间步之间的输出差异并非均匀分布（在某些时间步输出变化大、某些变化小），均匀缓存策略缺乏灵活性，无法最大化缓存利用率。

**核心矛盾**：要判断某个时间步的输出是否可以复用缓存，需要知道该输出与缓存输出的差异——但这个差异在输出计算完成之前是未知的。这是一个"先有鸡还是先有蛋"的问题。

**本文目标**：不计算模型输出就能预测输出差异的大小，从而智能选择缓存时机。

**切入角度**：模型输入与输出之间存在强相关性。如果能用输入的差异来估计输出的差异，就能以几乎零成本预判缓存决策。关键观察是：扩散模型的三个输入中，文本嵌入恒定不变、噪声输入对时间步不敏感、时间步嵌入变化但独立于输入内容——只有"时间步嵌入调制后的噪声输入"才同时包含时间步信息和内容信息，与输出最强相关。

**核心 idea**：用时间步嵌入调制噪声输入的差异作为模型输出差异的代理估计量，再通过多项式拟合校准缩放偏差，实现自适应的非均匀缓存策略。

## 方法详解

### 整体框架

TeaCache 工作在 DiT 扩散模型的推理阶段。对于每个去噪时间步：(1) 计算当前时间步嵌入调制的噪声输入与上一步的相对 L1 差异；(2) 通过预训练的多项式函数将输入差异映射为输出差异的估计；(3) 累积差异超过阈值 δ 时计算新的模型输出并缓存，否则复用缓存输出。整个过程无需额外训练，是即插即用的加速方案。

### 关键设计

1. **时间步嵌入调制的输入差异估计**:

    - 功能：用几乎零成本的输入差异来估计昂贵的输出差异
    - 核心思路：扩散模型在每个 Transformer block 中，时间步嵌入会通过 AdaLN 调制自注意力层和 FFN 的输入输出幅度。因此"时间步嵌入调制后的噪声输入"（即 Transformer 输入阶段的特征）同时包含了噪声内容信息和时间步信息，与模型输出有最强相关性。用相对 L1 距离衡量输入差异：$L1_{rel}(\mathbf{F}, t) = \|\mathbf{F}_t - \mathbf{F}_{t+1}\|_1 / \|\mathbf{F}_{t+1}\|_1$
    - 设计动机：文本嵌入恒定无法反映变化，单独的时间步嵌入或噪声输入各自不完整。经过调制的输入在实验中展现出与输出差异最强的相关性（在 Open-Sora、Latte、OpenSora-Plan 三个模型上验证）

2. **多项式拟合缩放校准**:

    - 功能：弥合输入差异与输出差异之间的缩放偏差
    - 核心思路：虽然输入差异与输出差异趋势一致，但存在幅度偏差。用简单的多项式拟合 $y = f(x) = a_0 + a_1x + a_2x^2 + \cdots + a_nx^n$ 将输入差异 $x$ 映射为输出差异估计 $y$。拟合数据通过在 70 个 prompt 上运行一次完整推理获得（一次性离线成本）。使用 numpy 的 poly1d 函数即可求解
    - 设计动机：直接用输入差异做判断会导致次优的时间步选择。多项式拟合简单但有效：一阶拟合即可提升 VBench 0.24%，四阶以上收益饱和

3. **累积差异自适应缓存策略**:

    - 功能：自适应决定何时刷新缓存
    - 核心思路：从上次缓存时间步 $t_a$ 开始累积校准后的差异 $\sum_{t=t_a}^{t_b-1} f(L1_{rel}(\mathbf{F}, t))$，当累积值超过阈值 δ 时在 $t_b$ 计算新输出并缓存，否则复用。关键细节是只缓存残差信号（输出减去输入），所以即使复用缓存，模型输出仍会随输入更新
    - 设计动机：与均匀缓存不同，自适应策略可以在输出变化平缓期大量跳过计算（如 U 形曲线中间段），在输出变化剧烈期保持全量计算，大幅提高缓存利用效率

### 损失函数 / 训练策略

TeaCache 完全免训练。多项式拟合系数通过离线采样 70 个 prompt 做一次校准即可，对不同的 base model 各校准一次。阈值 δ 控制速度-质量平衡：slow=0.1，fast=0.2。

## 实验关键数据

### 主实验

**三个 base model 上的加速与质量对比**:

| 模型 | 方法 | FLOPs (P) ↓ | 加速比 ↑ | VBench ↑ | LPIPS ↓ | PSNR ↑ |
|------|------|------------|---------|---------|---------|--------|
| Latte | PAB-fast | 2.52 | 1.34× | 73.13% | 0.3903 | 17.16 |
| Latte | **TeaCache-slow** | **1.86** | **1.86×** | **77.40%** | **0.1901** | **22.09** |
| Latte | **TeaCache-fast** | **1.12** | **3.28×** | 76.69% | 0.3133 | 18.62 |
| Open-Sora | PAB-fast | 2.50 | 1.40× | 76.95% | 0.1743 | 23.58 |
| Open-Sora | **TeaCache-slow** | **2.40** | **1.55×** | **79.28%** | **0.1316** | **23.62** |
| OSP | PAB-fast | 8.35 | 1.56× | 71.81% | 0.5499 | 15.47 |
| OSP | **TeaCache-slow** | **3.13** | **4.41×** | **80.32%** | **0.2145** | **21.02** |
| OSP | **TeaCache-fast** | **2.06** | **6.83×** | 79.72% | 0.3155 | 18.95 |

### 消融实验

| 缓存指标 | 模型 | VBench ↑ | 说明 |
|---------|------|---------|------|
| 时间步嵌入 | Open-Sora | 较低 | 不随内容变化 |
| **调制噪声输入** | Open-Sora | **较高** | 同时包含时间步和内容信息 |

| 多项式阶数 | VBench ↑ | LPIPS ↓ | 说明 |
|-----------|---------|---------|------|
| 无拟合 | 基准 | 基准 | 直接用输入差异 |
| 1阶 | +0.24% | 改善 | 简单线性校正 |
| 4阶 | 饱和 | 饱和 | 高阶无额外收益 |

### 关键发现

- 在 Open-Sora-Plan 上实现了惊人的 4.41×（slow）和 6.83×（fast）加速，说明该模型 150 步采样中存在大量冗余
- 不同模型的输出差异曲线形状差异大：Open-Sora 呈 U 形，Latte 和 OSP 呈翻转 L 形，验证了自适应策略优于均匀策略的必要性
- 缓存机制 vs 减少时间步：减少时间步会粗化 $\alpha_t$ 参数导致质量下降，而缓存保留了完整的调度参数
- TeaCache-slow 在 Latte 上 VBench 与原始模型完全一致（77.40%），实现了无损 1.86× 加速
- 多 GPU 扩展性良好：随 GPU 数量增加，TeaCache 持续优于 PAB

## 亮点与洞察

- **用输入估计输出差异的思路极其巧妙**：几乎零成本的代理指标避免了"要先知道输出才能决定是否计算"的悖论。这个方法论可以迁移到任何有迭代计算的系统
- **时间步嵌入的调制作用**：深入分析了时间步嵌入如何通过 AdaLN 层调制每层的输入输出幅度，这个观察是方法设计的关键基础
- **极简但有效**：多项式拟合只需 numpy 一行代码，70 个 prompt 校准一次。整个方法没有引入任何额外网络参数或训练，是教科书级的简洁

## 局限与展望

- 多项式拟合系数需要对每个 base model 分别校准，切换模型时需要重新拟合
- 缓存策略假设残差信号是好的近似，极端情况下（如场景突变）可能引入伪影
- 阈值 δ 是全局常数，未考虑不同 prompt 复杂度差异对最优阈值的影响
- 可以考虑 per-layer 的缓存策略，不同层的输出冗余程度可能不同
- 与 CFG-aware 缓存（如 FasterCache）结合可能进一步提升加速比

## 相关工作与启发

- **vs PAB**: PAB 按注意力块类型（spatial/temporal/cross）设置不同的均匀缓存间隔，但仍是静态策略。TeaCache 自适应策略在所有模型上全面超越
- **vs Δ-DiT**: Δ-DiT 缓存注意力层间的残差，但加速比有限（仅 1.02×）。TeaCache 的加速比高出 1-2 个数量级
- **vs DeepCache/FORA**: 这些方法针对 UNet 或特定层设计，TeaCache 直接在整个 DiT 模型层面工作，通用性更强

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 用模型输入估计输出差异的思路新颖且优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 base model、多分辨率/帧数、多消融全面验证
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机推导清晰，分析可视化充分
- 价值: ⭐⭐⭐⭐⭐ 免训练、即插即用、大幅加速，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing](../../ICML2025/video_generation/ca2-vdm_efficient_autoregressive_video_diffusion_model_with_causal_generation_an.md)
- [\[CVPR 2025\] Improved Video VAE for Latent Video Diffusion Model](improved_video_vae_for_latent_video_diffusion_model.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)
- [\[CVPR 2025\] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step](videoscene_distilling_video_diffusion_model_to_generate_3d_scenes_in_one_step.md)
- [\[CVPR 2025\] One-Minute Video Generation with Test-Time Training](one-minute_video_generation_with_test-time_training.md)

</div>

<!-- RELATED:END -->
