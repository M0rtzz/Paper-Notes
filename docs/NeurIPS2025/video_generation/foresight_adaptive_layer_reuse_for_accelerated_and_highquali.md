---
title: >-
  [论文解读] Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation
description: >-
  [NeurIPS 2025][自适应缓存] 提出 Foresight，一种训练无关的自适应层复用框架，通过在 warmup 阶段建立逐层 MSE 阈值、在 reuse 阶段按阈值动态决策每层是复用缓存还是重新计算，在 5 个视频生成模型上实现了比静态方法更高质量和更快速度的推理加速（最高 2.23×）。
tags:
  - NeurIPS 2025
  - 自适应缓存
  - DiT 加速
  - 特征复用
  - 文本到视频生成
  - 训练无关
---

# Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation

**会议**: NeurIPS 2025  
**arXiv**: [2506.00329](https://arxiv.org/abs/2506.00329)  
**代码**: https://github.com/STAR-Laboratory/foresight  
**领域**: 视频理解 / 扩散模型加速  
**关键词**: 自适应缓存、DiT 加速、特征复用、文本到视频生成、训练无关

## 一句话总结

提出 Foresight，一种训练无关的自适应层复用框架，通过在 warmup 阶段建立逐层 MSE 阈值、在 reuse 阶段按阈值动态决策每层是复用缓存还是重新计算，在 5 个视频生成模型上实现了比静态方法更高质量和更快速度的推理加速（最高 2.23×）。

## 研究背景与动机

**领域现状**：扩散 Transformer（DiT）已成为文本到视频生成的主流架构，OpenSora、CogVideoX、HunyuanVideo 等模型都基于空时 DiT 构建。然而，自注意力的 $O(L^2)$ 复杂度随分辨率和帧数增长，加上通常需要 30-50 步去噪，推理延迟极高。

**现有痛点**：特征缓存（feature caching）是无需训练的加速手段，通过在相邻去噪步骤间复用中间特征来减少计算量。但现有方法（Static、PAB、Δ-DiT、T-GATE、TeaCache）全部采用**静态策略**——以固定间隔对所有层统一复用，忽略了复用潜力在层间、prompt 间和配置间的巨大差异。

**核心矛盾**：作者通过分析 OpenSora 中 28 层 DiT 的空间特征 MSE 热力图，发现三个关键现象：(1) 早期层特征变化小、复用安全，后期层变化大、粗暴复用会严重降质；(2) 场景变化快的 prompt 比静态场景 prompt 的复用潜力低得多；(3) 分辨率从 240p 变到 720p 时，同一层的 MSE 模式发生显著变化。静态方法无法适配这些变化。

**本文目标** 如何在每一步每一层做出自适应的"复用 or 重算"决策，使得速度和质量达到更优的帕累托前沿。

**切入角度**：用特征 MSE 的统计量作为复用指标，在 warmup 阶段自动学习每层的阈值，运行时实时比较，不需要任何训练或架构修改。

**核心 idea**：用运行时 MSE 动态阈值替代静态复用间隔，让每层每步自主决定是否复用。

## 方法详解

### 整体框架

Foresight 将去噪过程分为两个阶段：**Warmup Phase** 和 **Reuse Phase**。Warmup 阶段正常计算所有层，建立缓存和逐层阈值；Reuse 阶段交替进行复用和重计算，每个重计算步更新复用指标，下一步按指标与阈值的比较做出逐层独立决策。整个过程只需要存储每层 DiT block 的两个输出（spatial + temporal），不修改模型权重。

### 关键设计

1. **Warmup 阶段与自适应阈值初始化**:

    - 功能：在前 $W$ 步（默认 15%）正常计算，让特征稳定后为每层建立复用阈值
    - 核心思路：用最后三步的 MSE 几何加权平均作为阈值 $\lambda_x^l = \sum_{t=W-2}^{W} \frac{1}{10^{W-t}} \cdot \text{MSE}^l(t, t-1)$，近期步的权重更大，阈值因层、prompt、分辨率而自然不同
    - 设计动机：静态方法需要人工调参确定复用间隔，Foresight 让阈值从数据中自动生成。后期层 MSE 大则阈值高，早期层 MSE 小则阈值低，自动实现"早期层多复用、后期层谨慎复用"

2. **Reuse 阶段的动态决策机制**:

    - 功能：在每个重计算步更新复用指标 $\delta$，下一步按指标与阈值比较做逐层决策
    - 核心思路：重计算步计算当前特征与缓存的 MSE 作为指标 $\delta_x^l(t)$，若 $\delta \leq \gamma \cdot \lambda_x^l$ 则下一步复用缓存，否则重新计算。缩放因子 $\gamma \in (0, 2]$ 控制速度-质量平衡
    - 设计动机：通过 $\gamma$ 提供简单的旋钮控制——$\gamma=0.25$ 几乎不复用但质量极高（PSNR 38），$\gamma=2.0$ 最大化复用和速度

3. **粗粒度 Block 级缓存**:

    - 功能：缓存整个 DiT Block 输出而非细粒度的 attention/MLP 分别缓存
    - 核心思路：每层只缓存 spatial 和 temporal 两个 block 输出，缓存大小为 $2L \cdot H \cdot W \cdot F$，比 PAB 的 $6L \cdot H \cdot W \cdot F$ 减少 3×
    - 设计动机：相邻步骤的 block 级特征本身就高度相似（余弦相似度 >0.99），细粒度缓存虽理论更灵活但额外的存储和管理开销不值得

### 收敛性分析

作者证明了 Foresight 的复用引入的误差有界且可控。在每个复用层，误差 $\varepsilon_t^l \leq \gamma \cdot \lambda_x^l$，整个去噪链的累积误差满足 $\|\hat{x}_t - x_t^*\| \leq \varepsilon_{\text{tot}} / (1 - \rho)$，其中 $\rho = \max_s \sqrt{1 - \beta_s} < 1$。收紧 $\gamma$ 可以任意接近基线输出。

## 实验关键数据

### 主实验（VBench, 550 prompts）

| 模型 | 方法 | VBench Acc | PSNR↑ | SSIM↑ | FVD↓ | 加速比 |
|------|------|------------|-------|-------|------|--------|
| OpenSora | PAB | 75.32 | 25.67 | 0.85 | 541.53 | 1.26× |
| OpenSora | **Foresight** (N=1,R=2) | **75.90** | **29.67** | **0.90** | **306.66** | **1.28×** |
| OpenSora | **Foresight** (N=2,R=3) | 75.62 | 27.49 | 0.87 | 457.69 | **1.44×** |
| CogVideoX | PAB | 77.89 | 29.04 | 0.91 | 340.24 | 1.37× |
| CogVideoX | **Foresight** (N=1,R=2) | **77.94** | **34.75** | **0.95** | **130.65** | **1.46×** |
| CogVideoX | **Foresight** (N=2,R=3) | 77.84 | 28.45 | 0.87 | 531.99 | **1.63×** |

### 消融实验

| 配置 | 延迟(s) | PSNR↑ | 说明 |
|------|---------|-------|------|
| PAB baseline | 19.88 | 28.12 | 静态基线 |
| γ=0.25 | 20.50 (+0.62) | 38.09 (+9.97) | 极少复用，质量极高 |
| γ=0.5 | 18.70 (−1.17) | 32.38 (+4.26) | 默认配置 |
| γ=2.0 | 16.02 (−3.85) | 29.51 (+1.39) | 最大复用 |
| N=3, R=4 | 14.79 (−5.08) | 29.03 (+0.91) | 更激进复用，仍优于 PAB |

### 关键发现

- **后期层是质量瓶颈**：将层分为 early/middle/late 三组，static 复用 late 层导致最大质量下降，Foresight 自动让 late 层更频繁重计算
- **扩展到最新模型**：HunyuanVideo 上 Foresight 达 1.62× 加速（PSNR 41.79 远超 TeaCache 的 37.31），Wan-2.1 上达 2.23× 加速
- **匹配质量比速度**：在匹配 PAB 输出质量的条件下，Foresight 在 OpenSora/Latte/CogVideoX 上分别达到 1.68×/1.58×/1.95× 加速
- **跨任务泛化**：应用于 FLUX 文本到图像模型也实现了约 2× 加速

## 亮点与洞察

- **自适应阈值设计巧妙**：阈值从 warmup 阶段的 MSE 统计量自动导出，无需人工调参，不同 prompt/分辨率/层自然得到不同阈值。这个 idea 可以迁移到任何需要"是否缓存"决策的场景
- **粗粒度优于细粒度的反直觉发现**：PAB 精心设计了空间/时间/交叉注意力的分层 broadcast 策略，但 Foresight 的 block 级粗粒度复用反而更好——因为决策本身是自适应的，弥补了粒度上的粗糙
- **理论和实践统一**：既有收敛性证明提供理论保证，又在 5 个模型上广泛验证，工程落地性强

## 局限与展望

- 加速比上限受 $N$ 和 $W$ 配置约束，最大约 2.23×，远低于步数压缩/蒸馏的 10-50×（但两者正交可组合）
- 目前采用 block 级复用，细粒度扩展到 attention head 级或 token 级可能进一步提升
- 阈值初始化依赖 warmup 阶段的 MSE 估计质量，极短视频（warmup 步数很少）可能不稳定
- 未探索与步数压缩方法（如 consistency distillation）的组合效果

## 相关工作与启发

- **vs PAB**：PAB 按经验固定不同注意力类型的 broadcast 范围，需要模型特定调参；Foresight 数据驱动、自适应，在所有模型上一套参数统一工作
- **vs TeaCache**：TeaCache 用 timestep embedding 的变化量做缓存判断，Foresight 用实际特征 MSE，后者更直接反映层级特征变化
- **vs Δ-DiT**：Δ-DiT 缓存残差偏移量而非完整特征，仍然是静态方案，无法适应 prompt/配置变化

## 评分

- 新颖性: ⭐⭐⭐ 自适应缓存的思路不算全新，但阈值设计和 warmup-reuse 两阶段框架有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个视频模型 + 1 个图像模型 × 3 个 benchmark × 多配置 × 消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 清晰系统，收敛性分析加分
- 价值: ⭐⭐⭐⭐ 即插即用、训练无关的加速方案，工程价值高

<!-- RELATED:START -->

## 相关论文

- [Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](../../CVPR2026/video_generation/free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)
- [OSV: One Step is Enough for High-Quality Image to Video Generation](../../CVPR2025/video_generation/osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)
- [Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](../../ICCV2025/video_generation/magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)
- [DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation](../../ICCV2025/video_generation/dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)

<!-- RELATED:END -->
