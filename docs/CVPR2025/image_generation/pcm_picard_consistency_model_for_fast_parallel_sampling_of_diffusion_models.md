---
title: >-
  [论文解读] PCM: Picard Consistency Model for Fast Parallel Sampling of Diffusion Models
description: >-
  [CVPR 2025][图像生成][并行采样] PCM 提出了 Picard 一致性模型来加速扩散模型的 Picard 迭代并行采样，通过训练模型直接预测不动点解并引入模型切换机制确保精确收敛，在图像生成和机器人控制任务上实现最高 2.71x 加速。
tags:
  - CVPR 2025
  - 图像生成
  - 并行采样
  - Picard迭代
  - 一致性模型
  - 扩散加速
  - 精确收敛
---

# PCM: Picard Consistency Model for Fast Parallel Sampling of Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2503.19731](https://arxiv.org/abs/2503.19731)  
**代码**: 无  
**领域**: 扩散模型  
**关键词**: 并行采样, Picard迭代, 一致性模型, 扩散加速, 精确收敛

## 一句话总结

PCM 提出了 Picard 一致性模型来加速扩散模型的 Picard 迭代并行采样，通过训练模型直接预测不动点解并引入模型切换机制确保精确收敛，在图像生成和机器人控制任务上实现最高 2.71x 加速。

## 研究背景与动机

**领域现状**：扩散模型在视觉、文本、机器人等领域取得显著进展，但顺序去噪过程导致生成速度慢。加速方法大致分为两类：（1）减少采样步数（如 DDIM、Consistency Model、Progressive Distillation），但牺牲质量或改变输出分布；（2）并行化计算，如 ParaDiGMS 基于 Picard 迭代实现并行采样，保证精确收敛。

**现有痛点**：Picard 迭代虽然保证收敛到原始模型的精确输出，但其收敛速率没有保证，可能实际上很慢。在迭代早期，naive Picard 生成的图像质量极差（如 k=2 时 FID 高达 257），限制了实际加速效果。

**核心矛盾**：精确收敛的保证与快速收敛的需求之间的矛盾——修改模型权重可以加速收敛但会改变输出分布。

**本文目标**：加速 Picard 迭代的收敛速度，同时保证最终结果与原始模型完全一致。

**切入角度**：类比 Consistency Model 的思路——Consistency Model 训练模型从去噪轨迹上的任意点直接预测最终输出 $x_T$；类似地，Picard 迭代也构成一条轨迹（在 $\mathbb{R}^{T \times n}$ 空间中），可以训练模型从轨迹上的任意点直接预测不动点 $X^*$。

**核心 idea**：在 Picard 迭代轨迹上训练一个"Picard 一致性模型"加速收敛，再通过模型切换策略在收敛后期平滑过渡回原始模型，确保精确收敛。

## 方法详解

### 整体框架

首先用原始扩散模型跑 Picard 迭代收集轨迹数据集 $\{X^0, X^1, ..., X^K = X^*\}$。然后训练 PCM：给定轨迹上的任意点 $X^k$，经过一次 Picard 迭代 $\Phi(X^k; \theta_{PCM})$ 后应直接跳到不动点 $X^*$。推理时使用模型切换：早期用 $\theta_{PCM}$ 加速，后期逐渐过渡到 $\theta_{base}$ 确保精确收敛。

### 关键设计

1. **Picard Consistency Training（Picard 一致性训练）**:

    - 功能：训练扩散模型在 Picard 迭代中预测不动点解
    - 核心思路：从轨迹数据集中随机采样 $X^k$ 和对应的 $X^* = X^K$，最小化损失 $\mathcal{L} = \mathbb{E}_{X \sim \mathcal{D}, k \sim \mathcal{U}[0,K-1]} \alpha(k) D(X^*, \Phi(X^k; \theta_{PCM}))$，其中 $\alpha(k) = \frac{1}{\sqrt{\text{Var}(k)}}$ 是加权函数。使用 EMA 更新稳定训练
    - 设计动机：Picard 迭代早期的 $X^k$ 与 $X^*$ 差距大且噪声多，加权函数 $\alpha(k)$ 缓解了不同阶段的方差差异；EMA 平滑了训练过程中过渡点的不稳定性

2. **Feature-space Model Switching（特征空间模型切换）**:

    - 功能：保证 PCM 最终收敛到与原始模型完全一致的输出
    - 核心思路：使用线性插值调度函数 $\lambda(k) = \max(0, \min(1, 1 - s \cdot k/K))$ 在 PCM 和原始模型的输出之间平滑过渡。$X^{k+1} = \lambda(k) \cdot \Phi(X^k, \theta_{PCM}) + (1-\lambda(k)) \cdot \Phi(X^k, \theta_{base})$
    - 设计动机：PCM 修改了权重因此收敛点不同于原模型。如果不切换，PCM 在初期收敛更快但最终误差不为零。模型切换利用了 PCM 的快速初始收敛，同时通过后期的原模型保证精确收敛

3. **Parameter-space Model Switching with LoRA（基于 LoRA 的参数空间切换）**:

    - 功能：减少存储和计算开销
    - 核心思路：只训练 LoRA 参数 $\Delta W$，推理时通过调节 LoRA scale 实现切换：$h^k = (W_0 + \lambda(k) \Delta W) x^k$。$\lambda(k)$ 从 1 衰减到 0，实现从 PCM 到原模型的平滑过渡
    - 设计动机：Feature-space 切换需要双倍模型存储和推理成本；LoRA 方式只需额外存储低秩参数，且权重混合可以离线预计算

### 损失函数 / 训练策略

使用 L2 距离作为度量函数 $D(\cdot, \cdot)$。从预训练权重初始化 PCM，EMA 衰减因子 $\mu = 0.999$。轨迹数据集生成 500 个样本，训练 50 epochs，Adam 优化器，学习率 1e-4。

## 实验关键数据

### 主实验（LDM-CelebA, DDIM）

| 方法 | 顺序步数 | FID↓ | 延迟 | 加速比 |
|------|---------|------|-----|--------|
| Sequential | 18 | 36.09 | 2.83s | 1x |
| Picard | 6 | 36.19 | 1.34s | 2.11x |
| PCM | 6 | 36.67 | 1.87s | 1.51x |
| PCM-LoRA | 6 | 35.97 | 1.34s | 2.11x |
| **极端对比 (k=2)** | | | | |
| Sequential | 2 | 366.92 | 0.32s | - |
| Picard | 2 | 257.83 | 0.44s | - |
| **PCM-LoRA** | **2** | **67.74** | **0.44s** | **-** |

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| PCM w/o model switching | 初期快但最终不收敛到原始输出 |
| PCM w/o EMA | 训练不稳定，过渡点每个 epoch 变化 |
| Feature-space switching | 需要双倍推理，但收敛更稳定 |
| LoRA switching | 推理成本不增加，但需要选择合适的 stiffness |

### 关键发现

- **PCM 在极端少步数下优势巨大**：k=2 时 PCM-LoRA FID 67.74 vs Picard 257.83，差距约 4 倍
- **收敛后 PCM 与原始模型输出完全一致**——model switching 成功保证了精确收敛（通过 Fig.5 的定性对比验证）
- PCM-LoRA 在性能上接近 full PCM 但推理成本不增加
- 在 Stable Diffusion 上，PCM 同样保持更好的 CLIP score，且在 k=8 时 PCM-LoRA 的效果超过 Picard k=9

## 亮点与洞察

- **"Picard 轨迹上的一致性模型"**是非常优雅的类比——将 Consistency Model 的思想从去噪轨迹推广到 Picard 迭代轨迹，是一个很自然但此前没人做的扩展
- **Model switching 解决了修改权重和精确收敛之间的矛盾**——用 PCM 加速初期收敛，用原模型保证最终精度。这种"先粗后细"的策略在很多场景都有借鉴意义
- **LoRA 实现参数空间切换**不仅节省了存储，还使得权重混合可以预计算，不增加额外延迟。这是 LoRA 在推理优化中一个很有创意的应用

## 局限与展望

- 需要预先收集 Picard 轨迹数据集，增加了前置成本
- 模型切换的 stiffness 参数需要手动调节，没有自适应策略
- Feature-space switching 需要双倍推理成本，虽然可并行但在资源受限时不实际
- 在更大规模的扩散模型（如 SDXL、FLUX）上的效果和收益有待验证
- 可以探索与其他加速技术（如步数蒸馏、量化）的组合

## 相关工作与启发

- **vs ParaDiGMS**: ParaDiGMS 首次将 Picard 迭代应用于扩散并行采样，但收敛速度慢；PCM 通过一致性训练显著加速收敛早期阶段
- **vs Consistency Model**: Consistency Model 在去噪轨迹 $x_t \in \mathbb{R}^n$ 上训练直接预测 $x_T$；PCM 在 Picard 轨迹 $X \in \mathbb{R}^{T \times n}$ 上训练预测不动点 $X^*$，维度更高但思路一致
- **vs Progressive Distillation**: 蒸馏方法改变了模型输出分布，不保证精确收敛；PCM 的 model switching 保证最终输出与原模型完全一致

## 评分

- 新颖性: ⭐⭐⭐⭐ Picard 轨迹上的一致性模型是优雅的新思路，model switching 也有创新
- 实验充分度: ⭐⭐⭐⭐ 图像生成和机器人控制两个领域验证，但模型规模偏小
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，motivation 讲解充分
- 价值: ⭐⭐⭐ 理论贡献突出，但并行推理的硬件前提限制了实际应用场景

<!-- RELATED:START -->

## 相关论文

- [See Further When Clear: Curriculum Consistency Model](see_further_when_clear_curriculum_consistency_model.md)
- [Riemannian Consistency Model](../../NeurIPS2025/image_generation/riemannian_consistency_model.md)
- [Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)
- [Accelerating Parallel Diffusion Model Serving with Residual Compression](../../NeurIPS2025/image_generation/accelerating_parallel_diffusion_model_serving_with_residual_compression.md)
- [TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)

<!-- RELATED:END -->
