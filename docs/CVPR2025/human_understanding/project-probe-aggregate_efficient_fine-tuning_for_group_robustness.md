---
title: >-
  [论文解读] Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness
description: >-
  [CVPR 2025][人体理解][群组鲁棒性] 提出 PPA（Project-Probe-Aggregate）三步方法，通过投影去除类代理信息放大偏差、以组先验校正探测组标签、聚合组权重，仅需不到 0.01% 可训练参数即可在无组标注情况下提升基础模型的群组鲁棒性。
tags:
  - CVPR 2025
  - 人体理解
  - 群组鲁棒性
  - 虚假相关
  - 参数高效微调
  - 偏差消除
  - CLIP
---

# Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness

**会议**: CVPR 2025  
**arXiv**: [2503.09487](https://arxiv.org/abs/2503.09487)  
**代码**: 无  
**领域**: human_understanding  
**关键词**: 群组鲁棒性, 虚假相关, 参数高效微调, 偏差消除, CLIP

## 一句话总结

提出 PPA（Project-Probe-Aggregate）三步方法，通过投影去除类代理信息放大偏差、以组先验校正探测组标签、聚合组权重，仅需不到 0.01% 可训练参数即可在无组标注情况下提升基础模型的群组鲁棒性。

## 研究背景与动机

图像-文本基础模型（如 CLIP）在下游任务中虽然平均误差低，但在存在虚假相关（spurious correlation）时，某些群组上的错误率极高。例如在 Waterbirds 数据集中，95% 的水鸟图像有水背景，模型容易依赖"水背景"而非"鸟的特征"进行分类，导致在"陆地背景的水鸟"群组上严重失败。

现有改善群组鲁棒性的方法面临两大挑战：(1) 需要组标注（annotation cost 高）；(2) 需要重新训练整个模型（computational cost 高）。基于"失败去偏"（failure-based debiasing）的策略——先训练有偏模型识别少数群组，再用推断的组标签训练鲁棒模型——是主流方案，但少数群组识别精度不高，且上加权策略缺乏理论最优性保证。

**核心问题**: 如何利用预训练基础模型的知识，更高效地识别少数群组，并设计有理论保证的去偏训练算法？

## 方法详解

### 整体框架

PPA 是一个三步线性探测框架，冻结 CLIP 骨干网络仅训练线性分类头：Step 1 (Project) 将图像特征投影到类代理的零空间以训练有偏分类器；Step 2 (Probe) 用有偏分类器推断伪组标签，训练组分类器并用组先验偏移校正；Step 3 (Aggregate) 聚合同类组权重得到最终去偏分类器。全程仅需训练两个线性层。

### 关键设计1: 类代理投影 (Project)

**功能**: 放大模型对虚假特征的依赖，提高少数群组识别精度。

**核心思路**: 用 CLIP 文本编码器获取类名嵌入 $Z = [\mathbf{z}_1, ..., \mathbf{z}_K]^T$，计算其零空间投影矩阵 $\Pi = I - Z^T(ZZ^T)^{-1}Z$。将图像特征投影后训练有偏分类器 $f_\mathsf{b}(\mathbf{x}) = W_\mathsf{b} \Pi \mathbf{x}$，配合 logit-adjustment 损失处理类别不平衡。

**设计动机**: 去除核心类别信息后，分类器被迫更多依赖虚假特征（如背景）进行预测。理论证明（Proposition 1）：当虚假特征与被投影去除的核心特征正相关时，虚假特征的权重 $\gamma' > \gamma$。如 Waterbirds 中"水背景"与"水鸟"正相关，去除"水鸟"信息后"水背景"权重增大，使有偏模型更容易在少数群组上犯错。

### 关键设计2: 组先验校正探测 (Probe)

**功能**: 以理论最优方式训练去偏分类器。

**核心思路**: 根据有偏分类器的正确/错误预测定义伪组标签 $\hat{g} = (y, \hat{a})$。训练组分类器 $h_\mathsf{d}(\mathbf{x}) = W_\mathsf{d} \mathbf{x}$，使用组 logit 调整损失：$\ell_\mathsf{gla}(\hat{g}, h_\mathsf{d}(\mathbf{x})) = -\ln \frac{\exp(h_\mathsf{d}(\mathbf{x}) + \tau \cdot \ln \hat{\boldsymbol{\beta}})_{\hat{g}}}{\sum_{g'} \exp(h_\mathsf{d}(\mathbf{x}) + \tau \cdot \ln \hat{\boldsymbol{\beta}})_{g'}}$，其中 $\hat{\boldsymbol{\beta}}$ 为组先验。

**设计动机**: 传统方法通过上加权少数样本来提升群组鲁棒性，但最优权重需要超参搜索。Proposition 2 证明组 logit 调整是最小化平衡组误差（BGE）的贝叶斯最优分类器，提供了理论保证。

### 关键设计3: 权重空间聚合 (Aggregate)

**功能**: 将组分类器转换为类分类器，消除推理时的组推断开销。

**核心思路**: 将同一类的所有组权重向量求和得到类分类器：$f_\mathsf{d}(\mathbf{x})_y = \mathbf{w}_y^T \mathbf{x}$，其中 $\mathbf{w}_y^T = \sum_{g \in \hat{\mathcal{G}}(y)} W_{\mathsf{d},g}$。由于线性分类器的输出空间求和等价于权重空间求和，推理时只需一次前向传播。

**设计动机**: 直接使用组分类器需要先推断组标签再转换为类预测，增加推理复杂度。权重聚合在训练后一次性完成，将 $|G|$ 维输出压缩为 $K$ 维，推理与标准分类器完全相同。

### 损失函数

两阶段损失：(1) 有偏分类器使用 logit-adjusted 交叉熵训练；(2) 组分类器使用 group logit-adjusted 交叉熵训练 $\ell_\mathsf{gla}$，超参 $\tau$ 控制先验校正强度。

## 实验关键数据

### 主实验结果 (Worst-Group Accuracy, CLIP ResNet-50)

| 方法 | 组标注 | Waterbirds WGA | CelebA WGA | MetaShift WGA |
|------|--------|---------------|------------|---------------|
| Zero-Shot CLIP | 无 | 54.2 | 55.0 | 86.2 |
| ERM | 无 | 7.9 | 11.9 | 75.4 |
| JTT | 无 | 61.7 | 60.2 | — |
| CA (Contrastive Adapter) | 无 | — | — | — |
| **PPA (ours)** | **无** | **高于 JTT/CA** | **高于 JTT/CA** | **高于 JTT** |
| GroupDRO | 有 | 75.1 | 84.1 | 83.2 |

### 少数群组识别质量对比 (Waterbirds)

| 方法 | Worst-group Recall(%) | Worst-group Precision(%) |
|------|----------------------|-------------------------|
| CA | 46.4 | 15.6 |
| JTT | 78.6 | 24.1 |
| **PPA** | **80.4** | **44.3** |

### 关键发现

1. **少数群组识别大幅提升**: PPA 的 worst-group precision (44.3%) 比 JTT (24.1%) 高近一倍，recall 也更高 (80.4% vs 78.6%)，证明类代理投影有效放大了对虚假特征的依赖。
2. **极低参数量**: 不到 0.01% 的可训练参数（仅两个线性层），远低于 adapter 或 prompt tuning 方案。
3. **无需组标注**: 在不使用任何组标注的情况下，性能接近甚至超越部分使用组标注的方法（如 S-CS、S-CL）。
4. **架构无关**: 方法在 CLIP ResNet-50 和 CLIP ViT-L/14 上均有效，也适用于 prompt tuning 和 adapter 等 PEFT 范式。
5. **理论保证**: Proposition 1 证明投影增大虚假特征权重，Proposition 2 证明组 logit 调整是 BGE 最优。

## 亮点与洞察

- **理论驱动设计**: 每一步设计都有严格的数学推导支撑，而非经验性组合。
- **利用文本先验**: 巧妙利用 CLIP 文本编码器的类代理信息作为"要去除的核心特征"，将预训练知识转化为去偏工具。
- **权重聚合的巧妙**: 利用线性分类器的可加性，在权重空间完成组到类的转换，零推理开销。

## 局限与展望

- **线性分类器假设**: 理论分析基于线性回归/线性分类器，对非线性复杂分类器的推广性需要验证。
- **二元属性限制**: 当前主要在二元虚假属性（如水/陆背景）上验证，多属性场景的扩展性未知。
- **依赖 CLIP 质量**: 类代理投影依赖 CLIP 文本编码器的质量，若类名嵌入不准确，投影效果可能下降。
- 未来可探索非线性特征空间的投影、多属性/连续属性的去偏。

## 相关工作与启发

- **JTT**: 通过 ERM 模型误分类样本识别少数群组，PPA 进一步用类代理投影增强偏差。
- **Orth-Cali**: 在文本嵌入中去除偏差方向，PPA 则在图像特征空间操作。
- **启发**: "去除核心信息以暴露虚假依赖"的策略可推广到其他公平性和鲁棒性问题。

## 评分

⭐⭐⭐⭐ — 理论扎实、设计优雅、实用性强。三步流程简洁明了，每步都有理论保证。不到 0.01% 参数的极致高效令人赞赏。对线性假设的依赖是唯一明显限制。

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](../../ICML2025/human_understanding/fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)
- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](../../AAAI2026/human_understanding/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/human_understanding/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[CVPR 2025\] Human Motion Instruction Tuning](human_motion_instruction_tuning.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/human_understanding/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)

<!-- RELATED:END -->
