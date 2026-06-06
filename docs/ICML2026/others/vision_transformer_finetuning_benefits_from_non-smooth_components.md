---
title: >-
  [论文解读] Vision Transformer 微调中的非光滑分量优势
description: >-
  [ICML 2026][可塑性] 通过定义"可塑性"度量，本文证明 ViT 中的非光滑分量（注意力和前馈层）具有更高可塑性——在微调时能提供更大梯度范数，实现更好且稳定的迁移学习性能。
tags:
  - "ICML 2026"
  - "可塑性"
  - "Transformer"
  - "微调"
  - "参数高效"
  - "平滑性"
---

# Vision Transformer 微调中的非光滑分量优势

**会议**: ICML 2026  
**arXiv**: [2602.06883](https://arxiv.org/abs/2602.06883)  
**代码**: https://github.com/ambroiseodt/vit-plasticity  
**领域**: 模型压缩 / 迁移学习 / 参数高效微调  
**关键词**: 可塑性, Vision Transformer, 微调, 参数高效, 平滑性

## 一句话总结
通过定义"可塑性"度量，本文证明 ViT 中的非光滑分量（注意力和前馈层）具有更高可塑性——在微调时能提供更大梯度范数，实现更好且稳定的迁移学习性能。

## 研究背景与动机

**领域现状**：ViT 已成为视觉和 NLP 领域标准骨干，普遍采用预训练后在下游任务微调的范式。PEFT 方法已成为行业标准，但对各组件的适应能力缺乏理论理解。

**现有痛点**：当前研究聚焦哪些参数需要更新（注意力、前馈层、归一化层），但缺乏原理性指导。光滑性通常被认为有益（改善泛化、稳定性、对抗鲁棒性），但在迁移学习背景下的作用鲜有探讨。

**核心矛盾**：过度光滑性（低 Lipschitz 常数）虽有利于泛化，但会限制模型对输入变化的响应能力，反而阻碍其对下游数据的适应。

**本文目标**：用"可塑性"（输入变化的平均响应率）取代单纯的光滑性约束，作为微调时选择适应性强组件的指导原则。

**切入角度**：在 ViT 架构分析基础上，提出理论-实证相结合的方法。

**核心 idea**：高可塑性（低平滑性）允许更大梯度范数，加速优化收敛——这与光滑性追求恰恰相反。

## 方法详解

### 整体框架
（1）定义可塑性度量；（2）推导各组件的可塑性上界；（3）在大规模预训练模型上验证理论排序；（4）通过 >1000 次微调实验验证可塑性与性能的对应关系。

### 关键设计

1. **可塑性度量**:

    - 功能：量化 Transformer 组件对输入变化的响应程度。
    - 核心思路：定义 $P(f) = \mathbb{E}_{(x,y) \sim \nu}\left[\frac{\|f(x)-f(y)\|_F}{\|x-y\|_F}\right]$。可塑性与 Lipschitz 常数 $P(f) \leq \text{Lip}(f)$ 相联系但捕捉平均行为而非最坏情况。
    - 设计动机：与输入梯度界有连接；通过 Béthune et al. (2024) 理论，输入-参数光滑性相关，高可塑性允许更大梯度范数。

2. **理论可塑性排序**:

    - 功能：为各组件导出可塑性上界。
    - 核心思路：对 LayerNorm 有 $P(f) \leq \frac{\|\gamma\|_\infty}{\sigma}$；对前馈层 $P(f) \leq \|W\|_2$；对多头注意力 $P(f) \leq \sum_h \|O^h\|_2\|V^h\|_2\sqrt{3n + (12n+3)r^4\|A^h\|_2^2}$。相对排序：MHA > FC1 ≈ FC2 > LN2 ≈ LN1。
    - 设计动机：通过谱范数比较和序列长度依赖性精确刻画为什么注意力和前馈具有更高可塑性。

3. **组件隔离微调**:

    - 功能：在大规模 ViT（86M/307M/632M 参数）上逐个组件微调。
    - 核心思路：每个配置固定其他权重仅更新一类组件；11 个分类基准 × 3 个随机种子 × 4 学习率的网格搜索，共 ~1000 次实验。
    - 设计动机：避免组件交互混杂效应，直接验证可塑性与微调性能对应关系。

## 实验关键数据

### 主实验

| 组件 | Cifar10 | Cifar100 | Clipart | Sketch | 平均准确度 | 关键特性 |
|------|--------|----------|---------|--------|-----------|---------|
| MHA（注意力） | 93.2 | 84.1 | 78.5 | 62.1 | 90.8 | 可塑性最高 |
| FC1（前馈 1） | 93.0 | 83.8 | 78.1 | 61.9 | 90.7 | 可塑性次高 |
| FC2（前馈 2） | 92.6 | 83.2 | 77.6 | 61.5 | 90.3 | 可塑性中等 |
| LN2（归一化 2） | 92.1 | 82.4 | 76.8 | 60.2 | 89.9 | 可塑性低 |
| LN1（归一化 1） | 92.0 | 82.1 | 76.5 | 59.8 | 89.8 | 可塑性最低 |

### 可塑性关联

| 度量 | MHA | FC1 | FC2 | LN2 | LN1 | 说明 |
|------|-----|-----|-----|-----|-----|------|
| 可塑性排序 | 1 | 2 | 3 | 4 | 5 | 理论与实验一致 |
| 梯度范数峰值 | 2.0 | 1.8 | 1.5 | 1.1 | 0.2 | 可塑性越高梯度越大 |
| 验证损失下降速度 | 快 | 快 | 快 | 慢 | 慢 | 优化收敛更快 |
| 学习率鲁棒性 | 高 | 高 | 中 | 低 | 低 | 高可塑性对超参稳定 |

### 关键发现
- 注意力模块和前馈层在大多数基准上显著更优，特别在难数据集（Cifar100、Clipart、Sketch）。
- 梯度范数与可塑性排序一致——高可塑性 → 大梯度 → 快速优化。
- 微调性能对学习率敏感性随可塑性递增而降低。

## 亮点与洞察
- **反传统智慧**：推翻"光滑性总是有益"的经典假设，在迁移学习场景下证明非光滑性（高可塑性）更有优势。
- **理论与实证统一**：从梯度界、可塑性定义到大规模 1000+ 次实验的完整链条。
- **跨架构一致性**：同样规律在 ViT-Base/Large/Huge、DINOv3、GPT-2 上均成立。

## 局限与展望
- 实验限于分类任务，在检测、分割等密集预测任务上推广需验证。
- 组件隔离设置避免交互效应分析，实际多组件协同更新可能产生新动力学。
- 可塑性定义基于均匀分布假设，其他分布或特定领域上的推广待探索。

## 相关工作与启发
- **vs LoRA 等 PEFT**：本文提供理论依据说明直接微调高可塑性组件反而效率更高（28K 参数 vs LoRA 400K，Cifar100 性能持平）。
- **vs 完全微调**：单组件微调往往优于全量微调，表明各部分"可教性"存在差异。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  打破光滑性总是有益的常识。
- 实验充分度: ⭐⭐⭐⭐⭐  超 1000 次微调 + 11 基准 + 跨多模型架构。
- 写作质量: ⭐⭐⭐⭐  理论推导清晰，少量篇幅占用繁琐公式。
- 价值: ⭐⭐⭐⭐⭐  直接指导高效微调的组件选择，对 PEFT 方法设计有实践参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](../../CVPR2026/others/vit3_unlocking_test_time_training_in_vision.md)
- [\[CVPR 2025\] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](../../CVPR2025/others/care_transformer_linear_attention.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](../../AAAI2026/others/enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)
- [\[ECCV 2024\] AttnZero: Efficient Attention Discovery for Vision Transformers](../../ECCV2024/others/attnzero_efficient_attention_discovery_for_vision_transformers.md)

</div>

<!-- RELATED:END -->
