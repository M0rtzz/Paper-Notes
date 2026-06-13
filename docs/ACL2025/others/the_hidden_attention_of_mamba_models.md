---
title: >-
  [论文解读] The Hidden Attention of Mamba Models
description: >-
  [ACL 2025][Mamba模型] 揭示了Mamba（选择性状态空间模型S6）可以被重新表述为一种隐式的因果自注意力机制，并基于此提出了适用于Mamba模型的注意力可视化和可解释性方法（Attention Rollout和Mamba-Attribution），证明其可解释性指标与Transformer相当。
tags:
  - "ACL 2025"
  - "Mamba模型"
  - "状态空间模型"
  - "隐式注意力"
  - "可解释性"
  - "选择性SSM"
---

# The Hidden Attention of Mamba Models

**会议**: ACL 2025  
**arXiv**: [2403.01590](https://arxiv.org/abs/2403.01590)  
**代码**: [https://github.com/AmeenAli/HiddenMambaAttn](https://github.com/AmeenAli/HiddenMambaAttn)  
**领域**: 其他  
**关键词**: Mamba模型, 状态空间模型, 隐式注意力, 可解释性, 选择性SSM

## 一句话总结

揭示了Mamba（选择性状态空间模型S6）可以被重新表述为一种隐式的因果自注意力机制，并基于此提出了适用于Mamba模型的注意力可视化和可解释性方法（Attention Rollout和Mamba-Attribution），证明其可解释性指标与Transformer相当。

## 研究背景与动机

**领域现状**：Mamba模型（基于Selective SSM/S6层）在NLP、计算机视觉、长序列建模等多个领域展现了卓越性能，以线性复杂度实现了与Transformer近似的效果，推理时可切换为高效的RNN模式。然而，对Mamba模型的内部工作机制——特别是token之间的信息流动和依赖关系捕获方式——的理解非常有限。

**现有痛点**：Transformer的注意力矩阵已被广泛用于模型可解释性分析（如Attention Rollout、Transformer Attribution等方法），但SSM模型传统上只有卷积视图和循环视图两种理解方式，缺乏类似注意力矩阵的可解释性工具。这严重制约了Mamba模型在医疗、金融等对可解释性有要求的领域的应用。

**核心矛盾**：Mamba模型在实践中表现优异，但其信息流动机制不透明——既不像传统SSM那样可以用固定卷积核解释（因为S6是时变/数据依赖的），也不像Transformer那样有显式的注意力权重可供分析。

**本文目标** (1) 从理论上揭示Mamba的隐式注意力结构；(2) 基于此构建Mamba模型的可解释性工具；(3) 比较Mamba和Transformer在注意力机制上的异同。

**切入角度**：作者观察到S6层是"数据控制的线性算子"（data-controlled linear operator），通过将时变递推公式展开为矩阵形式，可以得到一个输入依赖的下三角矩阵，其结构与因果自注意力矩阵高度类似。

**核心 idea**：将S6层重新表述为 $y = \tilde{\alpha} x$ 的矩阵乘法形式，其中 $\tilde{\alpha}$ 是一个数据依赖的下三角矩阵，可以视为Mamba的"隐式注意力矩阵"。

## 方法详解

### 整体框架

论文为Mamba模型提供了第三种视角（除了已有的并行扫描视图和RNN递推视图）：注意力视图。通过数学推导，将每个S6通道的输出表示为一个数据依赖的下三角矩阵与输入向量的乘积。基于提取出的隐式注意力矩阵，进一步将Transformer中的Attention Rollout和Transformer Attribution方法适配到Mamba模型。

### 关键设计

1. **隐式注意力矩阵的推导**:

    - 功能：从S6层的递推公式推导出等价的注意力矩阵形式
    - 核心思路：给定时变系统矩阵 $\bar{A}_t$、$\bar{B}_t$、$C_t$，将递推 $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$，$y_t = C_t h_t$ 展开得到 $y_t = C_t \sum_{j=1}^{t} (\prod_{k=j+1}^{t} \bar{A}_k) \bar{B}_j x_j$。写成矩阵形式 $y = \tilde{\alpha} x$，其中 $\tilde{\alpha}_{i,j} = C_i (\prod_{k=j+1}^{i} \bar{A}_k) \bar{B}_j$，这就是隐式注意力矩阵
    - 设计动机：由于 $\bar{A}_t$ 是对角矩阵，可以进一步分解为N个独立的内部注意力矩阵之和。一个S6通道产生N个内部注意力矩阵，D个通道共产生 $D \times N$ 个，远多于Transformer的H个注意力头

2. **Query-Key-History分解**:

    - 功能：将隐式注意力矩阵分解为类似Q/K/V的直观形式
    - 核心思路：将softplus近似为ReLU后，可以得到 $\tilde{\alpha}_{i,j} \approx \tilde{Q}_i \tilde{H}_{i,j} \tilde{K}_j$，其中 $\tilde{Q}_i = S_C(\hat{x}_i)$ 对应query，$\tilde{K}_j = \text{ReLU}(S_\Delta(\hat{x}_j)) S_B(\hat{x}_j)$ 对应key，$\tilde{H}_{i,j} = \exp(\sum_{k=j+1}^{i} S_\Delta(\hat{x}_k)) A$ 对应"历史上下文"项
    - 设计动机：这个分解揭示了Mamba与Transformer注意力的关键区别——Mamba额外引入了 $\tilde{H}_{i,j}$ 来控制历史token的重要性衰减，这可能是Mamba更擅长建模连续历史上下文的原因

3. **Mamba Attention Rollout**:

    - 功能：提供class-agnostic的可解释性方法
    - 核心思路：对每层每通道提取隐式注意力矩阵 $\tilde{\alpha}^{\lambda,d}$，在通道维度取平均，加上恒等矩阵（跳跃连接），然后跨层相乘得到全局注意力 $\rho = \prod_{\lambda=1}^{\Lambda} (\mathbb{I} + \mathbb{E}_{d}[\tilde{\alpha}^{\lambda,d}])$。对于双向Mamba，将两个方向的注意力矩阵相加
    - 设计动机：直接复用Transformer Attention Rollout的框架，仅需适配注意力矩阵的来源

4. **Mamba Attribution**:

    - 功能：提供class-specific的可解释性方法
    - 核心思路：结合隐式注意力矩阵与梯度信息：$\tilde{\beta}^\lambda = \mathbb{I} + (\mathbb{E}_{d}[\nabla \hat{y}'^{\lambda,d}] \odot \mathbb{E}_{d}[\tilde{\alpha}^{\lambda,d}])^+$。与Transformer Attribution不同的是，梯度不是对注意力矩阵求的，而是对S6输出与gate的乘积 $\hat{y}'$ 求的，以同时捕获S6 mixer和gating机制的类别特异性信号
    - 设计动机：直接将LRP得分替换为注意力矩阵效果更好，同时利用gating机制的梯度可以获得更强的class-specific归因

### 损失函数 / 训练策略

本文不涉及训练新模型，而是分析已有预训练模型。视觉实验使用预训练的Vision Mamba (ViM) 和DeiT，NLP实验使用Mamba-130M和Pythia-160M。

## 实验关键数据

### 主实验（扰动测试）

| 方法 | 正向扰动AUC↓ (Mamba) | 正向扰动AUC↓ (Trans.) | 负向扰动AUC↑ (Mamba) | 负向扰动AUC↑ (Trans.) |
|--------|------|------|------|------|
| Raw-Attention | 17.27 | 20.69 | 34.03 | 40.77 |
| Attn-Rollout | 18.81 | 20.59 | 41.86 | 43.53 |
| Attribution | 16.62 | 15.35 | 39.63 | 48.09 |

### 分割测试（ImageNet-Segmentation）

| 模型 | 方法 | 像素精度↑ | mAP↑ | mIoU↑ |
|------|---------|------|------|------|
| Mamba | Raw-Attention | 67.64 | 74.88 | 45.09 |
| Transformer | Raw-Attention | 59.69 | 77.25 | 36.94 |
| Mamba | Attn-Rollout | 71.01 | 80.78 | 51.51 |
| Transformer | Attn-Rollout | 66.84 | 80.34 | 47.85 |
| Mamba | Attribution | 74.72 | 81.70 | 54.24 |
| Transformer | Trans.-Attribution | 79.26 | 84.85 | 60.63 |

### 关键发现

- Mamba的Raw Attention在像素精度和mIoU上显著优于Transformer的Raw Attention，表明隐式注意力矩阵本身已具有较好的可解释性
- 在Attn-Rollout方法下Mamba全面优于Transformer，而在Attribution方法下Transformer更强，说明Mamba-Attribution可能需要进一步针对性设计
- CLS token的位置显著影响视觉Mamba的注意力分布——靠近CLS的patch影响更大，暗示非空间的全局CLS token可能是更好的选择
- Mamba注意力矩阵与Transformer的结构高度相似：浅层关注局部对角线模式，深层捕获远程依赖

## 亮点与洞察

- **理论贡献非常精彩**：证明了单通道S6层可以表达单头Transformer的所有函数，但反之不成立（Theorem 5.2）。这从理论上解释了为什么Mamba在实践中至少不逊于Transformer
- **隐式注意力矩阵数量惊人**：Mamba产生的注意力矩阵数量约为Transformer的 $DN/H \approx 100N$ 倍，但它们共享Q矩阵，仅通过K和H项区分。这种"大量轻量注意力矩阵"的结构或许是Mamba效率的关键
- 论文同时提供了SSM模型注意力机制演化的理论分析（Theorem 5.1）：从S4（固定混合）→ GSS/Hyena（固定混合+对角数据控制）→ Selective SSM（数据控制非对角混合），揭示了"数据控制非对角混合器"是Mamba和Transformer共有的关键能力

## 局限与展望

- Mamba-Attribution的分割性能仍低于Transformer-Attribution，可能是因为直接从Transformer方法改编，未充分利用Mamba特有结构
- 扰动实验中Mamba在负向扰动下一致低于Transformer，可能因为Mamba对patch遮挡更敏感，应尝试模糊而非删除的扰动方式
- 仅分析了Mamba-1（S6），未涉及Mamba-2等后续架构
- 实际应用中提取全部 $D \times N$ 个注意力矩阵的计算开销可能较大，需要高效近似方法

## 相关工作与启发

- **vs Transformer Attention**: Mamba的隐式注意力是下三角的（因果性），不使用softmax（因此不会过平滑），且通过 $\tilde{H}_{i,j}$ 项自然编码了位置关系和历史衰减
- **vs 传统SSM (S4/DSS)**: 传统SSM的注意力矩阵是固定的（不依赖输入），而Selective SSM（Mamba）的注意力是数据依赖的，这是其表达能力显著超越前者的根本原因
- **vs Attention Rollout / Chefer et al.**: 本文将这两种经典Transformer可解释方法适配到Mamba，为SSM系列模型的可解释性研究奠定了基础

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统揭示Mamba的注意力本质，理论推导优美，视角转换非常有洞察力
- 实验充分度: ⭐⭐⭐⭐ 覆盖了视觉和NLP两个领域的可解释性验证，但定量对比可以更深入
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，结构清晰，但公式密度较高可能影响部分读者理解
- 价值: ⭐⭐⭐⭐⭐ 为理解和改进SSM模型提供了重要理论基础，有广泛的后续研究价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)
- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](../../ICCV2025/others/a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[ACL 2025\] Inferring Functionality of Attention Heads from their Parameters](inferring_functionality_of_attention_heads_from_their_parameters.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Segment-Based Attention Masking for GPTs](segment-based_attention_masking_for_gpts.md)

</div>

<!-- RELATED:END -->
