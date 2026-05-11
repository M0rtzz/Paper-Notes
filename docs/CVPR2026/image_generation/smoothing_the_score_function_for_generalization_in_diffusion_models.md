---
title: >-
  [论文解读] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework
description: >-
  [CVPR 2026][图像生成][扩散模型] 本文从理论上证明扩散模型的记忆化问题源于经验得分函数中 softmax 权重的尖锐性（单个训练样本主导），并提出 Noise Unconditioning 和 Temperature Smoothing 两种平滑方法来缓解记忆化、增强泛化，同时保持生成质量。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型
  - 记忆化
  - 泛化
  - 得分函数平滑
  - 温度缩放
---

# Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework

**会议**: CVPR 2026  
**arXiv**: [2601.19285](https://arxiv.org/abs/2601.19285)  
**代码**: [GitHub](https://github.com/xinyu-zhou/score-smoothing)  
**领域**: 图像生成  
**关键词**: 扩散模型, 记忆化, 泛化, 得分函数平滑, 温度缩放

## 一句话总结
本文从理论上证明扩散模型的记忆化问题源于经验得分函数中 softmax 权重的尖锐性（单个训练样本主导），并提出 Noise Unconditioning 和 Temperature Smoothing 两种平滑方法来缓解记忆化、增强泛化，同时保持生成质量。

## 研究背景与动机

1. **领域现状**：扩散模型已成为图像生成的主流框架，通过逐步添加噪声再学习反转过程实现高质量生成。其核心机制是通过神经网络逼近不同噪声水平下的得分函数。
2. **现有痛点**：大量研究发现扩散模型会"记忆化"训练数据——生成的样本可能与训练样本完全相同。理论分析表明，如果神经网络完美学习了经验得分函数，采样将退化为复制训练数据，无法产生新样本。
3. **核心矛盾**：理论预测的完美记忆化与实践中观察到的泛化能力之间存在根本性矛盾。虽然已知神经网络的有限容量和正则化起了作用，但"为什么神经网络能部分解决记忆化"这一核心问题缺乏系统性的理论解释。
4. **本文目标**：(1) 建立理论框架解释记忆化的根本原因；(2) 解释神经网络为何能实现泛化；(3) 基于理论提出进一步增强泛化的方法。
5. **切入角度**：将经验得分函数分解为高斯分量得分的加权和，发现权重本质上是 softmax 函数。在高维空间中，几何上每个训练样本对应一个"壳层"，采样点在低噪声时只落入单个壳层内，导致单点主导。
6. **核心 idea**：神经网络通过隐式平滑得分函数权重实现泛化，使采样受到局部流形（多个近邻样本）而非单个点的影响；可以通过显式平滑方法进一步增强这一效果。

## 方法详解

### 整体框架
输入是训练数据集 $\{\mu_j\}_{j=1}^M$，通过前向 SDE 添加噪声得到含噪分布 $p_i^*(x)$，然后训练神经网络逼近得分函数。本文的两种方法分别通过去除噪声条件化和引入温度参数来平滑得分函数权重，从而在反向采样过程中实现泛化而非记忆化。

### 关键设计

1. **高维壳层几何分析**:
    - 功能：揭示记忆化的几何本质
    - 核心思路：在高维空间中，高斯分布的概率质量集中在以训练样本 $\mu_j$ 为中心、半径约 $\sigma_i\sqrt{d}$ 的薄壳层上。经验得分函数的权重 $w_{ij}(x) = \text{Softmax}(f(x, \mu_j, \sigma_i))$。有两个关键性质：(1) $\sigma$-主导 — 对于固定中心和位置，存在最优噪声水平 $\sigma_j^*$ 使对应权重远超其他噪声水平（比值可达 $e^6 \approx 403$）；(2) $\mu$-主导 — 最近训练样本的权重随距离差异指数增长。在低噪声采样阶段，壳层缩小到不再重叠，单个训练样本主导得分函数，导致采样坍缩到该点——即记忆化。
    - 设计动机：通过严格的数学分析揭示记忆化的根本原因，为后续平滑方法提供理论基础

2. **Noise Unconditioning（噪声去条件化）**:
    - 功能：允许每个训练样本自适应选择最优噪声水平，增加其他样本的贡献
    - 核心思路：标准扩散模型中得分函数 $s_\theta(x, \sigma_i)$ 以噪声水平为条件，去条件化后 $s_\theta(x)$ 将所有噪声水平的分布统一为一个 $M \times N$ 的高斯混合 $p_{\text{MN}}(x)$，等价于对该统一分布做得分匹配。采样过程可重新解释为对 $\log p_{\text{MN}}(x)$ 的梯度上升，训练点是最优解但"坍缩时间"被延迟。损失函数 $\mathcal{L}_u$ 与标准扩散相同，唯一区别是去除了噪声作为网络输入。采样时由于噪声水平不再作为输入，需要用自适应步长 $\alpha \sigma_{n*}^2$ 替代预设步长。
    - 设计动机：标准扩散中采样点可能不在大多数训练点的最优壳层上，导致这些点的权重被抑制。去条件化让每个训练点都能以最优壳层贡献得分，防止单点主导

3. **Temperature Smoothing（温度平滑）**:
    - 功能：通过温度参数显式控制得分函数权重的平滑度
    - 核心思路：在得分函数权重的 softmax 中引入温度参数 $T_i$，$w_j^*(x;T) = \frac{\exp(f/T_j^*)}{\sum_l \exp(f/T_l^*)}$。设置阈值 $\sigma_{\text{collapse}}$，当 $\sigma_i \leq \sigma_{\text{collapse}}$ 时启用温度缩放（$T_i = \sigma_{\text{collapse}}/\sigma_i$），并用 top-K 最近邻训练样本近似得分函数。高温降低主导比 $a$，减小扩展因子 $\gamma_{ex}$。在特征空间（而非像素空间）中做 KNN 效果更好，因为特征空间局部曲率更小。
    - 设计动机：直接利用 softmax 温度的经典性质来控制权重分布的尖锐度，提供可调节的泛化-质量平衡

### 损失函数 / 训练策略
整体损失函数自适应组合两种方法：当 $\sigma_i > \sigma_{\text{collapse}}$ 时使用 Unconditioning 损失 $\mathcal{L}_u$，当 $\sigma_i \leq \sigma_{\text{collapse}}$ 时使用 Temperature 损失 $\mathcal{L}_T$。训练沿用 VE-SDE 框架，Unconditioning 方法仅去除时间嵌入层，其余结构不变。

## 实验关键数据

### 主实验

| 数据集 | 方法 | FID(G,Train) | FID(G,Test) | 说明 |
|--------|------|-------------|------------|------|
| CIFAR-10 | Conditioning (baseline) | 6.49 | 6.56 | 标准 VE-SDE |
| CIFAR-10 | Unconditioning | 7.33 | 7.34 | FID 微升但泛化增强 |
| CIFAR-10 | Temp T=7/σ, K=100 (feat) | 7.96 | 7.98 | 像素空间 KNN 会崩溃(50.81) |
| CelebA | Conditioning | 7.25 | 7.81 | 标准 VE-SDE |
| CelebA | Unconditioning | 7.07 | 7.34 | FID 反而下降 |
| CelebA | Temp T=10/σ, K=100 (feat) | 8.40 | 8.19 | 特征 KNN 显著优于像素 KNN |

### 消融实验

| 配置 | 扩展因子 γ_ex | 说明 |
|------|-------------|------|
| 经验得分函数 (Conditioning) | ~10³ (低噪声时) | 极度尖锐，记忆化 |
| NN 学习的得分函数 | ~1-2 | 隐式平滑 |
| Unconditioning (经验) | 中等 | 比 Conditioning 平滑 |
| Temperature T=10 | ~1 | 显式平滑效果好 |
| Temperature T=100 | <1 | 接近非扩展 |
| Temperature T=1000 | ≈1 | 极度平滑 |

### 关键发现
- **特征空间 KNN 一致优于像素空间 KNN**：在 CIFAR-10 上 T=7/σ, K=100 时，像素空间 FID 崩溃到 50.81 而特征空间仅 7.96，验证了"局部流形曲率小有助于平滑"的理论
- **Unconditioning 在 CelebA 上反而改善 FID**（7.07 vs 7.25），说明平滑不仅不损害质量，可能还有正面效果
- **ODE 采样器在 Unconditioning 下会失败**，因为预设噪声水平与实际不匹配导致步长爆炸，SDE 采样器的随机项提供自校正机制保持稳定

## 亮点与洞察
- **壳层几何直觉**极其优美：将高维高斯混合的抽象数学分析转化为"薄壳层是否重叠"的几何图景，使记忆化的本质一目了然。这种几何化思维可迁移到其他涉及高斯混合的问题中。
- **统一分布视角**是最大的"啊哈"时刻：Noise Unconditioning 将扩散模型的逐步去噪重新解释为对固定目标函数的梯度上升，这不仅解释了泛化，还开启了用投影梯度法施加约束的可能性（如物理定律约束的视频生成）。
- **温度平滑作为即插即用方法**几乎不增加额外成本，可直接应用于现有扩散模型框架中，具有很强的工程实用性。

## 局限与展望
- 实验主要基于 VE-SDE 框架，未验证在更主流的 VP-SDE 和 Flow Matching 框架上的效果
- Temperature Smoothing 需要 KNN 查询，对大规模数据集有额外开销
- 温度参数和 $\sigma_{\text{collapse}}$ 的选择需要调参，缺少自动化策略
- 未扩展到潜在扩散模型（Latent Diffusion），作者提到这是重要的未来方向
- 未来可探索将此框架与 Consistency Model、Rectified Flow 等新范式结合

## 相关工作与启发
- **vs Carlini et al. (2023) 的记忆化检测工作**: 它们从攻击角度证明记忆化的存在，本文从理论角度解释记忆化的根本原因并提出缓解方法，二者互补
- **vs Bonnaire et al. (2025) 的隐式正则化分析**: 该工作研究训练动态中的隐式正则化如何防止记忆化，本文聚焦于得分函数的结构分析，提供了更直接的几何解释和显式干预方法

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从得分函数权重的 softmax 结构出发建立完整理论框架，视角独特且优美
- 实验充分度: ⭐⭐⭐⭐ 理论验证充分，但实验规模有限（主要在小数据集上）
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，几何直觉清晰，论述逻辑流畅
- 价值: ⭐⭐⭐⭐ 为扩散模型泛化提供了深刻的理论洞察，实用方法简单有效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[CVPR 2026\] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultrafast_trainingfree_highresolution_im.md)
- [\[CVPR 2026\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multiagent_dialogue_framework_for_c.md)

</div>

<!-- RELATED:END -->
