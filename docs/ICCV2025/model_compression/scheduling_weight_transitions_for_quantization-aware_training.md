---
title: >-
  [论文解读] Scheduling Weight Transitions for Quantization-Aware Training
description: >-
  [ICCV 2025][模型压缩][量化感知训练] 指出传统学习率调度对量化感知训练（QAT）中量化权重的有效步长控制失效，提出转换率（Transition Rate）调度技术，通过自适应学习率（TALR）显式控制量化权重的离散跳变次数，显著提升低比特量化模型性能。
tags:
  - ICCV 2025
  - 模型压缩
  - 量化感知训练
  - 转换率调度
  - 自适应学习率
  - 网络量化
  - 低比特精度
---

# Scheduling Weight Transitions for Quantization-Aware Training

**会议**: ICCV 2025  
**arXiv**: [2404.19248](https://arxiv.org/abs/2404.19248)  
**代码**: [https://cvlab.yonsei.ac.kr/projects/TRS/](https://cvlab.yonsei.ac.kr/projects/TRS/)  
**领域**: model_compression  
**关键词**: 量化感知训练, 转换率调度, 自适应学习率, 网络量化, 低比特精度

## 一句话总结

指出传统学习率调度对量化感知训练（QAT）中量化权重的有效步长控制失效，提出转换率（Transition Rate）调度技术，通过自适应学习率（TALR）显式控制量化权重的离散跳变次数，显著提升低比特量化模型性能。

## 研究背景与动机

QAT 通过在训练中模拟量化来学习低比特权重。其核心机制是：维护全精度的"潜在权重"（latent weights），经量化器离散化后得到量化权重用于前向传播。优化器更新的是潜在权重，量化权重只在潜在权重越过量化器的**转换点**（transition point）时才发生离散跳变。

传统做法直接用手工调度的学习率（LR）更新潜在权重，这在全精度训练中可以有效控制参数变化幅度（有效步长），但作者发现 **在 QAT 中这一假设不成立**：

**量化权重的有效步长与 LR 弱相关**：即使 LR 很小，如果潜在权重集中在转换点附近，大量权重仍会发生跳变，导致有效步长剧烈波动。

**训练后期潜在权重趋向转换点**：这是导致 QAT 中量化权重振荡（oscillation）的根本原因，也是已知问题 [Nagel et al., 2022] 的深层机制。

**无法实现粗到细的优化**：全精度训练中 LR 衰减保证收敛，但 QAT 中量化权重的剧烈变化破坏了这一性质。

因此，需要一种 **专门面向 QAT 的训练调度器**，直接控制量化权重的实际变化。

## 方法详解

### 整体框架

核心思想：不再调度潜在权重的学习率，而是调度量化权重的 **目标转换率**（target transition rate），并用自适应的 TALR 来更新潜在权重，使实际转换率跟踪目标值。

### 关键设计

1. **转换率（Transition Rate, TR）**  
   定义为单次更新中量化权重发生离散跳变的比例：
   $$k^t = \frac{\sum_{i=1}^{N} \mathbb{I}[w_d^t(i) \neq w_d^{t-1}(i)]}{N}$$
   其中 $w_d$ 是离散权重（round/signum 函数输出的整数值）。作者证明量化权重的有效步长近似为 $|\Delta w_q^t| \approx \delta^t \cdot \mathbb{I}[w_d^t \neq w_d^{t-1}]$，即每个量化权重的步长要么为零要么为固定值 $\delta^t$，因此 **平均有效步长主要由转换次数决定**。

2. **运行转换率（Running TR）估计**  
   用指数移动平均平滑当前 TR：
   $$K^t = m \cdot K^{t-1} + (1-m) \cdot k^t$$
   其中动量 $m = 0.99$，减少异常值影响。

3. **转换自适应学习率（TALR）**  
   根据运行 TR 与目标 TR 的差异，自适应调整学习率：
   $$U^t = \max(0, U^{t-1} + \eta(R^t - K^t))$$
   当 $K^t < R^t$ 时 TALR 增大，促使更多潜在权重越过转换点；反之减小。最终用 TALR 更新潜在权重：$\mathbf{w}^{t+1} = \mathbf{w}^t - U^t \mathbf{g}^t$。

4. **目标 TR 调度**  
   目标 TR $R^t$ 使用常规调度器（如余弦退火）从初始值衰减到零，初始值设为 $\lambda \sqrt{b_w}$（$b_w$ 为权重比特宽度），保证高比特宽度有更大的初始 TR。

### 损失函数 / 训练策略

- 训练目标与原始 QAT 相同（交叉熵或蒸馏损失），仅更改优化器的学习率为 TALR。
- 额外计算开销仅为逐元素比较和标量运算，训练时间仅增加约 2%。
- 兼容多种优化器（SGD、Adam、AdamW）和多种量化方案（二值、多比特）。
- 权重量化器的可学习 scale 参数在使用 TR 调度时固定，以避免转换点移动干扰 TR 控制。

## 实验关键数据

### 主实验（表格）

**ImageNet 分类 Top-1 准确率**

| 模型 | 比特宽度 (W/A) | SGD | SGD_T (ours) | Adam | Adam_T (ours) |
|------|-------------|------|-------------|------|--------------|
| MobileNetV2 | 2/2 | 46.9 | **53.6** (+6.7) | 49.6 | **53.8** (+4.2) |
| MobileNetV2 | 3/3 | 65.6 | **67.0** (+1.4) | 66.5 | **67.3** (+0.8) |
| MobileNetV2 | 4/4 | 69.9 | **70.5** (+0.6) | 70.0 | **70.8** (+0.8) |
| ResNet-18 | 1/1 | 55.3 | **55.8** (+0.5) | 56.1 | **56.3** (+0.2) |
| ResNet-18 | 2/2 | 66.8 | **66.9** (+0.1) | 66.7 | **67.2** (+0.5) |
| DeiT-T | 2/2 | - | - | 54.6 | **57.4** (+2.8) |
| DeiT-S | 2/2 | - | - | 68.4 | **71.8** (+3.4) |

轻量模型 + 低比特设置下增益最大（MobileNetV2 2-bit 提升 6.7%），说明 TR 调度对激进量化尤其重要。

### 消融实验（表格）

**不同 TR 因子 $\lambda$ 对性能影响（ResNet-20, CIFAR-100, 2-bit）**

| TR 因子 | 1e-3 | 2e-3 | 3e-3 | 4e-3 | 5e-3 | 6e-3 | 7e-3 | 8e-3 |
|--------|------|------|------|------|------|------|------|------|
| 准确率 | 62.5 | 64.2 | 64.3 | 65.3 | **65.5** | 65.1 | 63.1 | 63.6 |

最佳 TR 因子在 4e-3 到 6e-3 范围内，性能对该超参数敏感度适中（约 3% 范围），且所有值均优于 SGD 基线(64.1)。

### 关键发现

- TALR 在训练后期急剧衰减，这是因为潜在权重聚集在转换点附近，需要极小的更新才能维持低 TR。
- 使用 step decay 调度器时，传统 LR 方案性能严重退化（ResNet-20 从 64.1 降至 61.3），而 TR 调度仅轻微下降（65.5→64.9），鲁棒性更强。
- 目标检测（MS COCO RetinaNet）上也有一致提升（AP 提升 0.27-0.31），验证了方法的泛化能力。

## 亮点与洞察

- 精准诊断了 QAT 中 LR 失效的根因：量化权重的步长由转换决定，而转换同时受 LR 和潜在权重分布影响，仅调度 LR 不够。
- TALR 隐式地考虑了潜在权重的分布，是一种优雅的自适应策略。
- 方法极度轻量（仅增加逐元素比较 + 标量运算），却能在各种架构/优化器/比特宽度上稳定提升。

## 局限与展望

- 初始 TR 因子 $\lambda$ 仍需手动设定，类似 LR 的调参负担转移到了 TR 空间。
- 多比特量化时转换点更多、不同层的 TR 动态可能不同，全局统一调度可能非最优。
- 仅在图像分类和目标检测上验证，NLP 序列模型（如 LLM 量化）的适用性未探索。
- 仅在均匀量化方案上实验，对非均匀量化（如 GPTQ、AWQ）的兼容性有待研究。

## 相关工作与启发

- [Nagel et al., 2022] 发现量化权重振荡问题，通过冻结或正则化缓解；本文从更本质的角度（直接控制跳变次数）解决，避免了冻结权重可能的训练性退化。
- 本文的 TR 调度思想可类比为"量化空间中的学习率调度"，将全精度训练中成熟的 coarse-to-fine 范式推广到离散参数优化。
- 对 LLM 量化（如 QLoRA 后续工作）有潜在启发：大模型量化训练中权重振荡同样严重。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （首次提出面向 QAT 的转换率调度，洞察深刻）
- 实验充分度: ⭐⭐⭐⭐ （多架构/优化器/比特/任务，消融详尽）
- 写作质量: ⭐⭐⭐⭐⭐ （论证逻辑严密，从现象到分析到方案一气呵成）
- 价值: ⭐⭐⭐⭐ （轻量通用，即插即用提升 QAT 性能）

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](../../ICLR2026/model_compression/compute-optimal_quantization-aware_training.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](../../ACL2025/model_compression/efficientqat.md)
- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](../../ICML2025/model_compression/boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](../../ACL2025/model_compression/l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)

<!-- RELATED:END -->
