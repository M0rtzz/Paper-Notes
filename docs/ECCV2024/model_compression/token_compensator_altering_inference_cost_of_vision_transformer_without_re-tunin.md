---
title: >-
  [论文解读] Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning
description: >-
  [ECCV 2024][模型压缩][Token 压缩] 提出 ToCom（Token Compensator），一个模型算术框架的轻量插件，通过快速的参数高效自蒸馏获得，可在推理时直接插入任意下游已训练模型以弥补 token 压缩度不匹配造成的性能损失，无需重新训练。
tags:
  - ECCV 2024
  - 模型压缩
  - Token 压缩
  - Transformer
  - 模型算术
  - 知识蒸馏
  - 推理效率
---

# Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning

**会议**: ECCV 2024  
**arXiv**: [2408.06798](https://arxiv.org/abs/2408.06798)  
**代码**: [有](https://github.com/JieShibo/ToCom)  
**领域**: 模型压缩  
**关键词**: Token 压缩, Vision Transformer, 模型算术, 知识蒸馏, 推理效率

## 一句话总结

提出 ToCom（Token Compensator），一个模型算术框架的轻量插件，通过快速的参数高效自蒸馏获得，可在推理时直接插入任意下游已训练模型以弥补 token 压缩度不匹配造成的性能损失，无需重新训练。

## 研究背景与动机

Token 压缩是加速 Vision Transformer（ViT）训练和推理的重要技术，主要有两种方式：

**Token 剪枝**：移除不重要的 token

**Token 合并**（如 ToMe）：合并相似 token

然而，现有 token 压缩方法面临一个严峻的实际问题：**训练和推理阶段的压缩度不匹配会导致显著性能下降**。具体来说：

- 如果模型在压缩度 r=8 下训练，但推理时需要 r=12（更大压缩以加速），性能会大幅下降
- 反之，如果推理时使用 r=4（减少压缩追求精度），也无法恢复到 r=0 的性能
- 这限制了 token 压缩在已训练好的现成（off-the-shelf）模型上的灵活应用

**核心问题**：如何让用户在推理时自由调整压缩度，而不需要为每个压缩度重新训练模型？

## 方法详解

### 整体框架

ToCom 基于"模型算术"（Model Arithmetic）的思想：

| 阶段 | 操作 | 说明 |
|------|------|------|
| 预训练阶段 | 对预训练模型执行快速自蒸馏 | 仅在 ImageNet 上训练 ~10 epochs |
| 获得插件 | 得到 ToCom 补偿器集合 P | P = {P_0→1, P_1→2, ..., P_15→16} |
| 推理阶段 | 将 ToCom 插入现成模型 | 补偿压缩度不匹配的性能损失 |

核心思想是：不同压缩度的模型之间的差异可以被一个轻量级"补偿器"捕获。通过在预训练模型上学习这种补偿关系，可以将其迁移到任意下游模型。

### 关键设计

**1. 参数高效自蒸馏**

ToCom 的训练流程：
- 选择预训练模型 M̂（如 DeiT-B）
- 随机采样两个不同的压缩度 m 和 n
- 分别以 r=m 和 r=n 应用 ToMe（Token Merging）到 M̂
- 通过知识蒸馏损失学习从一个压缩度到另一个的补偿映射

**2. 可组合的补偿器链**

ToCom 由一系列相邻压缩度之间的补偿器组成：P_{i→(i+1)}。当需要从压缩度 m 补偿到 n 时：
- 若 m < n（增大压缩）：将补偿器 P_{m→(m+1)}, ..., P_{(n-1)→n} 级联并相加（⊕）
- 若 m > n（减小压缩）：将补偿器反向应用并相减（⊖）

这种设计使得 ToCom 可以处理任意压缩度对之间的补偿。

**3. 与知识蒸馏的关系**

训练损失函数为：
$$\mathcal{L} = \mathcal{L}_{KD}\left(\hat{\mathcal{M}}_m \oplus \left(\bigoplus_{i=m}^{n-1} \mathcal{P}_{i\rightarrow(i+1)}\right), \hat{\mathcal{M}}_n\right)$$

即让应用了补偿器的 source 模型输出尽可能接近 target 模型的输出。

### 损失函数 / 训练策略

- **损失函数**：标准知识蒸馏损失（KL 散度）
- **训练数据**：仅使用预训练数据（ImageNet-1K）
- **训练时长**：DeiT-B 仅需 10 个 epochs
- **优化器**：AdamW，学习率 1e-3，权重衰减 0.05
- **批大小**：1024，在 8×V100 GPU 上训练
- 压缩度范围 r ∈ {0, 1, 2, ..., 16}，训练时随机采样

## 实验关键数据

### 主实验（表格）

在 20+ 下游任务上的性能提升（DeiT-B，source r 训练，target r 推理，+ToCom）：

| 评估基准 | 无 ToCom 的降幅 | +ToCom 的最大恢复 | 下游任务数 |
|----------|----------------|-------------------|-----------|
| CIFAR-100 | 显著下降 | 最多恢复 2.3% | 1 |
| FGVC (细粒度视觉分类) | 显著下降 | 最多恢复 1.5% | 4 |
| VTAB-1k | 显著下降 | 最多恢复 2.0% | 19 |

### ADE20k 语义分割（表格）

| 设置 | mIoU | GFLOPs |
|------|------|--------|
| r=0（无压缩） | 48.7 | 106.2 |
| r=8 | 48.0 | 91.8 |
| r=8 + ToCom | **48.3** | 91.8 |
| r=12 | 46.4 | 84.5 |
| r=12 + ToCom | **47.2** | 84.5 |
| r=16 | 41.3 | 77.3 |
| r=16 + ToCom | **43.4** | 77.3 |

### 关键发现

- ToCom 可以在不增加任何计算开销的情况下（GFLOPs 不变），显著恢复 token 压缩导致的性能损失
- 在压缩度越大时（r=16），ToCom 的改善幅度越明显（+2.1 mIoU on ADE20k）
- ToCom 是通用的：在预训练模型上学习一次，可直接插入各种下游任务的模型
- 支持多种 ViT 架构：DeiT-B、DeiT-S、ViT-B (MAE)
- 训练成本低：仅需 10 个 epochs 的 ImageNet 自蒸馏
- 可扩展至密集预测任务（语义分割），此时在 FFN 层前合并、后还原

## 亮点与洞察

1. **即插即用**：ToCom 可直接插入任何已训练好的下游模型，无需重新训练，极大提升了实用性
2. **模型算术的优雅抽象**：将压缩度间的差异建模为可加/减的补偿器，数学上简洁且直观
3. **训练-推理解耦**：首次系统性地解决了 token 压缩中训练/推理压缩度不匹配的问题
4. **经济的训练**：相比为每个压缩度重训模型，ToCom 仅需一次轻量自蒸馏
5. **对密集任务的扩展**：通过 merge-before-FFN/unmerge-after-FFN 策略适配语义分割

## 局限与展望

- 补偿效果有上限，无法完全恢复到无压缩的性能水平
- 当前仅基于 ToMe（Token Merging），未验证对其他 token 压缩方法（如 pruning）的泛化性
- 压缩度范围 r ∈ {0, ..., 16} 是预定义的，更细粒度可能需要更多补偿器
- 组合多个补偿器可能引入累积误差
- 仅在分类和分割任务上验证，检测等其他下游任务待探索
- 可以考虑结合自适应压缩度选择机制

## 相关工作与启发

- **ToMe (Token Merging)**：本文的基础 token 压缩方法，通过合并相似 token 加速 ViT
- **参数高效微调（PET）**：AdaptFormer 等方法在 VTAB-1k 上的成功为本文提供了下游适配的范式
- **知识蒸馏**：ToCom 的训练本质上是不同压缩度间的自蒸馏
- **模型算术**：将模型差异形式化为可加减的操作，灵感可能来源于 task vectors 等工作
- 本文为动态推理效率调节提供了优雅的解决方案，与 elastic inference 方向相关

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 |
| 理论深度 | 3.5 |
| 实验充分度 | 4.5 |
| 实用性 | 4.5 |
| 写作质量 | 4 |
| 总体 | 4 |

<!-- RELATED:START -->

## 相关论文

- [BHViT: Binarized Hybrid Vision Transformer](../../CVPR2025/model_compression/bhvit_binarized_hybrid_vision_transformer.md)
- [Isomorphic Pruning for Vision Models](isomorphic_pruning_for_vision_models.md)
- [Uncertainty-Driven Spectral Compressive Imaging with Spatial-Frequency Transformer](uncertainty-driven_spectral_compressive_imaging_with_spatial-frequency_transform.md)
- [EA-ViT: Efficient Adaptation for Elastic Vision Transformer](../../ICCV2025/model_compression/ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](../../ICML2025/model_compression/orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)

<!-- RELATED:END -->
