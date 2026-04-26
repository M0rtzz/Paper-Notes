---
title: >-
  [论文解读] Post-hoc Probabilistic Vision-Language Models
description: >-
  [ICLR 2026][多模态][视觉语言模型] 提出一种免训练的后验（post-hoc）不确定性估计方法，对 CLIP/SigLIP 等 VLM 最后几层使用 Laplace 近似，解析推导余弦相似度的不确定性，在不确定性量化和主动学习中取得显著优于基线的效果。
tags:
  - ICLR 2026
  - 多模态
  - 视觉语言模型
  - 不确定性量化
  - 贝叶斯推断
  - Laplace近似
  - 主动学习
---

# Post-hoc Probabilistic Vision-Language Models

**会议**: ICLR 2026  
**arXiv**: [2412.06014](https://arxiv.org/abs/2412.06014)  
**代码**: 有（Project page）  
**领域**: Multimodal VLM / Uncertainty Quantification  
**关键词**: 视觉语言模型, 不确定性量化, 贝叶斯推断, Laplace近似, 主动学习

## 一句话总结

提出一种免训练的后验（post-hoc）不确定性估计方法，对 CLIP/SigLIP 等 VLM 最后几层使用 Laplace 近似，解析推导余弦相似度的不确定性，在不确定性量化和主动学习中取得显著优于基线的效果。

## 研究背景与动机

### 领域现状

**领域现状**：视觉语言模型（VLMs），如 CLIP 和 SigLIP，已在分类、检索、生成等任务中取得了巨大成功。这些模型的核心操作是将图像和文本分别映射到共享的潜在空间，然后使用**余弦相似度**（cosine similarity）评估匹配程度。

然而，这种确定性映射存在一个根本性问题：**无法捕获概念上的不确定性**（uncertainty over concepts）。具体而言：

**领域偏移（domain shift）**：当 VLM 用于下游任务时，训练域与目标域之间的差异导致预测不可靠，但模型无法表达其"不确信"的程度

**分布外样本（OOD）**：对于未见过的图像或概念，VLM 仍然输出单一的确定性嵌入，无法区分"确信正确"和"猜测性预测"

**安全关键应用**：在医疗诊断、自动驾驶等场景中，不确定性估计对于可靠决策至关重要

现有的不确定性估计方法通常需要：

### 现有痛点

**现有痛点**：重新训练整个模型（如 Monte Carlo Dropout、集成学习）

### 核心矛盾

**核心矛盾**：修改模型架构（如概率嵌入方法）

### 解决思路

**解决思路**：大量额外的计算资源

这些方法在大规模 VLM 上不切实际——CLIP 等模型在数十亿图文对上训练，重新训练成本极高。因此，一种**不需要额外训练**的后验不确定性估计方法具有极大的实用价值。

## 方法详解

### 整体框架

BayesVLM（论文提出的方法名称）的核心思想是：

1. 保持 VLM 的所有参数不变
2. 仅对最后几层构建贝叶斯后验近似
3. 通过 Laplace 近似获得参数的高斯后验分布
4. 解析推导余弦相似度在后验分布下的不确定性

### 关键设计

1. **Laplace 后验近似**：

    - 将 VLM 最后几层的权重视为随机变量
    - 使用 Laplace 近似构建权重的高斯后验分布：$p(\theta | D) \approx \mathcal{N}(\theta^*, \Sigma)$
    - 其中 $\theta^*$ 是预训练权重（MAP 估计），$\Sigma$ 通过 Fisher 信息矩阵的逆近似
    - **设计动机**：Laplace 近似在 MAP 估计点展开，天然利用了已训练好的模型参数，无需重新训练

2. **余弦相似度的不确定性解析推导**：

    - 传统 VLM 中，余弦相似度 $s = \frac{f_I \cdot f_T}{\|f_I\| \|f_T\|}$ 是确定性的
    - 当嵌入成为随机变量后，余弦相似度也成为随机变量
    - 论文解析推导了余弦相似度的分布特性（均值和方差），避免了 Monte Carlo 采样的计算开销
    - **设计动机**：解析解不仅计算高效，还避免了采样引入的近似误差

3. **仅处理最后几层**：

    - 不对整个 VLM 做贝叶斯处理，仅处理最后几层
    - 这大幅降低了计算成本——Fisher 信息矩阵的规模从整个模型缩小到最后几层的参数
    - 实践中最后几层对下游任务的适应性影响最大
    - **设计动机**：平衡计算成本与不确定性估计的有效性，feature extractor 部分的不确定性在后验中近似为零

4. **免训练特性**：

    - 方法仅需要一次简单的校准（calibration），不需要微调模型任何参数
    - 校准过程计算 Fisher 信息矩阵（或其近似），可以在少量数据上完成
    - 模型参数完全保持不变，不影响原有性能
    - **设计动机**：使方法可以即插即用地应用于任何预训练 VLM，最大化实用性

### 损失函数 / 训练策略

- **无需额外训练**：方法完全是后验（post-hoc）的
- **校准过程**：使用少量数据计算 Fisher 信息矩阵的对角近似或 Kronecker 分解近似
- **推理过程**：前向传播得到嵌入 → 利用Laplace近似计算不确定性 → 输出均值预测和不确定性估计

## 实验关键数据

### 主实验

论文在两个主要应用场景中验证方法的有效性：

**不确定性量化（Uncertainty Quantification）**

| 设置 | 指标 | BayesVLM | 确定性基线 | 优势 |
|------|------|---------|----------|------|
| ID 数据 | 校准误差 (ECE) | 显著改善 | 过度自信 | 校准更好 |
| OOD 检测 | AUROC | 提升明显 | 无不确定性 | 能识别 OOD |
| 领域偏移 | 预测可靠性 | 更稳健 | 性能下降 | 提供可靠的不确定性信号 |

**主动学习（Active Learning）**

| 数据集 | 指标 | BayesVLM | 随机采样 | 其他基线 |
|--------|------|---------|---------|---------|
| 多个下游任务 | 样本效率 | **最高** | 基准线 | 中等 |
| 标注预算受限 | 准确率 | **最优** | 较差 | 次优 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 处理层数 | 不确定性质量 | 仅最后 1-2 层即可获得良好效果 |
| Fisher 矩阵近似方式 | 校准质量 | 对角近似已足够，Kronecker 分解效果更好 |
| 不同 VLM 骨架 | 通用性 | CLIP 和 SigLIP 上均有效 |

### 关键发现

- **校准良好**：BayesVLM 提供的不确定性估计具有良好的校准性——模型预测"不确信"时确实更可能出错
- **可解释性**：不确定性估计具有直觉上的可解释性——模棱两可或分布外的样本获得更高不确定性
- **主动学习高效**：基于不确定性的样本选择显著优于随机采样，在标注预算有限时价值尤其突出
- **不影响原始性能**：作为后验方法，不修改模型参数，不降低原有的分类/检索性能
- **计算高效**：解析推导避免了 Monte Carlo 采样，推理开销极小

## 亮点与洞察

- **问题选择精准**：VLM 的不确定性估计是一个被忽视但极其重要的问题，特别是在安全关键应用中
- **方法设计简洁**：不需要重新训练、不需要修改架构、不需要大量额外计算，真正的"即插即用"
- **理论-实用平衡**：Laplace 近似有坚实的理论基础，同时解析推导保证了计算效率
- **余弦相似度的概率化处理**：将确定性的余弦相似度转化为具有不确定性的随机变量，是一个优雅的理论贡献
- **下游应用多样**：同时展示了在不确定性量化和主动学习两个实际场景中的价值

## 局限与展望

- **近似质量**：Laplace 近似假设后验为高斯分布，在高维空间中可能不够准确
- **仅处理最后几层**：忽略了 VLM 更深层的不确定性传播，可能低估总体不确定性
- **Fisher 矩阵计算**：对于非常大的模型，即使是对角近似也可能有一定计算开销
- **评估基准有限**：不确定性估计的评估缺乏统一标准，不同数据集上的表现可能差异较大
- **面向分类/检索场景**：未验证在生成式 VLM（如 LLaVA、GPT-4V）上的适用性
- **自回归生成**：方法适用于 CLIP 类的双编码器架构，对于自回归 VLM 架构需要进一步扩展

## 相关工作与启发

- **CLIP** (Radford et al., 2021)：最具代表性的 deterministic VLM，本文方法的主要应用对象
- **SigLIP** (Zhai et al., 2023)：CLIP 的改进版本，使用 Sigmoid 损失，同样适用于本方法
- **Laplace 近似**：经典的贝叶斯近似方法，近年来在深度学习中重新受到关注（Laplace Redux, Daxberger et al., 2021）
- **Monte Carlo Dropout** (Gal & Ghahramani, 2016)：通过 Dropout 近似贝叶斯推理，但需要多次前向传播
- **概率嵌入** (Kirchhof et al., 2023)：将嵌入建模为分布而非点，但需要重新训练
- **主动学习** (Settles, 2009)：基于不确定性的样本选择是主动学习的经典策略

**启发**：后验方法是将贝叶斯不确定性引入大规模预训练模型的务实路径。这一思路可以推广到其他预训练模型（如 LLM、音频模型）的不确定性估计中。余弦相似度的概率化可能催生新的基于不确定性的检索和匹配算法。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 后验 Laplace 近似不新，但在 VLM 余弦相似度上的解析推导是新的贡献
- 实验充分度: ⭐⭐⭐⭐ — 不确定性量化和主动学习双场景验证，多 VLM 骨架测试
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，方法描述简洁易懂
- 价值: ⭐⭐⭐⭐ — 解决了 VLM 部署中的实际需求，安全关键应用前景广阔

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](../../CVPR2025/multimodal_vlm/post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](../../CVPR2026/multimodal_vlm/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[ICLR 2026\] Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?](do_vision-language_models_respect_contextual_integrity_in_location_disclosure.md)
- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)

<!-- RELATED:END -->
