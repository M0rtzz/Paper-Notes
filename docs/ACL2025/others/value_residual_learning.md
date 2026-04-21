---
title: >-
  [论文解读] Value Residual Learning
description: >-
  [ACL 2025][Value残差连接] 提出 ResFormer 和 SVFormer，通过在注意力机制中引入第一层 Value 向量到后续层的残差连接，增强初始 token 级信息在深层网络中的传播，以比标准 Transformer 少 16.11% 的参数和 20.3% 的训练数据达到同等性能，SVFormer 还能减少近一半 KV 缓存。
tags:
  - ACL 2025
  - Value残差连接
  - Transformer
  - 信息传播
  - KV缓存压缩
  - 过平滑
---

# Value Residual Learning

**会议**: ACL 2025  
**arXiv**: [2410.17897](https://arxiv.org/abs/2410.17897)  
**代码**: 无  
**领域**: 其他  
**关键词**: Value残差连接, Transformer架构, 信息传播, KV缓存压缩, 过平滑

## 一句话总结

提出 ResFormer 和 SVFormer，通过在注意力机制中引入第一层 Value 向量到后续层的残差连接，增强初始 token 级信息在深层网络中的传播，以比标准 Transformer 少 16.11% 的参数和 20.3% 的训练数据达到同等性能，SVFormer 还能减少近一半 KV 缓存。

## 研究背景与动机

标准 Transformer 通过隐藏状态残差连接在深层网络中传播信息，但存在一个核心问题：**过平滑（over-smoothing）**。注意力机制的平滑效应使得 token 表征在网络加深时越来越相似——序列级特征逐渐主导，token 级特征被稀释。

已有研究的局限：
- **DenseFormer**：使用可学习稠密连接（类似 DenseNet），学到的连接系数表明深层确实需要更多初始嵌入的注意力，但直接求和初始嵌入和深层隐藏状态可能显著影响高层的注意力分布建模
- **NeuTRENO**：通过添加第一层和当前层 Value 向量的差值来缓解过平滑，但以正则化视角处理，效果有限

作者的关键洞察：初始 token 嵌入（$\mathbf{H}_0$）和第一层的 Value 状态（$\mathbf{V}_1$）虽然都包含局部 token 信息（两者仅差一个线性变换），但通过 Value 残差而非隐藏状态残差传递初始信息，对注意力分布的干扰更小。因为 Value 的残差连接是在注意力矩阵计算之前引入的，与现有注意力矩阵共享，不会改变注意力模式的建模。

## 方法详解

### 整体框架

在标准 Transformer 的基础上，ResFormer 仅添加一个简单的 Value 残差连接：将第一层的 Value 向量 $\mathbf{V}_1$ 与当前层的 Value 向量 $\mathbf{V}_n$ 进行加权融合，然后共用当前层的注意力矩阵。SVFormer 更进一步，让所有层共享第一层的 Value 状态。

### 关键设计

1. **ResFormer 核心公式**：$\mathbf{V}_n = \lambda_{n,1}\mathbf{V}_1 + \lambda_{n,2}\mathbf{H}_{n-1}\mathbf{W}_n^V$。其中 $\lambda$ 可以是固定常数（Constant-ResFormer）、统一值（Identity-ResFormer）、可学习参数（Learnable-ResFormer）或稀疏应用（Sparse-ResFormer）。设计动机是让深层网络能直接访问未被注意力平滑稀释的原始 token 级信息

2. **多种变体设计**：

    - **Identity-ResFormer**：$\lambda_{n,1} = \lambda_{n,2} = 0.5$，最简单的变体
    - **Constant-ResFormer**：手动调优常数，最优为 $\lambda = 2$
    - **Sparse-ResFormer**：仅在后几层应用 Value 残差，实验表明最后 3 层（6-8 层中的 8 层模型）受益最大
    - **Learnable-ResFormer Plus**：深层给予 $\mathbf{V}_1$ 更大权重的自适应初始化

3. **SVFormer**：将 Value 与注意力操作解耦，所有层共享第一层的 Value：$\mathbf{U}_n = \mathbf{A}_n\mathbf{V}_1$。主要优势是仅需计算和存储第一层的 Value 向量，**将 KV 缓存减少近一半**。实验表明共享 Value 的负面影响远小于共享 Key

4. **Dense-ResFormer**：最一般形式，$\mathbf{V}_n = \lambda_{n,n}\mathbf{H}_{n-1}\mathbf{W}_n^V + \sum_{i=1}^{n-1}\lambda_{n,i}\mathbf{V}_i$，允许跨所有前序层的 Value 连接

### 损失函数 / 训练策略

- 使用 AdamW 优化器，权重衰减 0.1，$\beta_1=0.9$, $\beta_2=0.95$
- Batch size 约 2M tokens，序列长度 2048，训练 10000 步
- 线性学习率预热 1200 步，峰值 6e-4，余弦衰减至峰值的 10%
- 训练数据：20B SlimPajama 子采样
- 标准语言建模目标

## 实验关键数据

### 主实验

| 模型 | 参数 | Wiki. PPL | 下游任务 Avg. ACC |
|------|------|-----------|----------------|
| Transformer | 468M | 24.8 | 40.6 |
| NeuTRENO | 468M | 24.3 | 41.4 |
| DenseFormer | 468M | 24.0 | 40.8 |
| Identity ResFormer | 468M | 23.8 | 41.3 |
| Learnable ResFormer | 468M | 23.7 | **42.3** |
| Learnable ResFormer Plus | 468M | **23.2** | 42.0 |

### 缩放实验

| 指标 | ResFormer vs Transformer |
|------|------------------------|
| 参数效率 | 少 16.11% 参数达到同等 valid loss |
| 数据效率 | 少 20.3% 训练数据达到同等 valid loss |
| 1.6B 规模 | 200B token 训练，Value 残差一致提升性能 |

### 消融实验

| 配置 | Valid Loss | 说明 |
|------|----------|------|
| Vanilla Transformer | 2.739 | 基线 |
| Identity-ResFormer ($\lambda=0.5$) | 2.712 | 最简单变体即有显著提升 |
| Constant-ResFormer ($\lambda=2$) | 2.700 | 手动调优更优 |
| Sparse-ResFormer (层6-8, $\lambda=5$) | **2.687** | 仅在后几层应用效果最好 |
| ResFormer-Plus (可学习) | 2.681 | 自适应初始化最优 |
| 额外 Hidden 残差到 $\mathbf{H}_0$ | 2.781 | 反而有害 |
| Query 残差 | 2.742 | 无益 |
| Key 残差 | 2.746 | 无益 |
| Attention 残差 | 2.757 | 有害 |

### 关键发现

1. **为什么是 $\mathbf{V}_1$ 而非 $\mathbf{V}_2$**：只有来自第一层 Value 的跳跃连接能显著提升性能。因为默认隐藏状态残差已经将 $\mathbf{H}_1$ 的信息传播到后续层（$\mathbf{V}_2 = \mathbf{H}_1\mathbf{W}_2^V$），但 $\mathbf{H}_0$ 的信息因为被后续信息稀释而丢失。当删除 $\mathbf{H}_0$ 到 $\mathbf{H}_1$ 的残差连接（使 $\mathbf{V}_2$ 无法通过隐藏残差获取 $\mathbf{H}_0$ 信息）后，$\mathbf{V}_2$ 的跳跃连接才能提供显著收益

2. **后几层受益最大**：对 8 层模型，仅在第 7 层应用 Value 残差效果最好；扩展到 6-8 层效果最佳，但从第 5 层开始收益减弱

3. **Value 残差优于 Hidden 残差**：额外的隐藏状态残差等价于同时对 Q、K、V 应用残差，会干扰注意力分布

4. **学习到的 $\lambda$ 模式**：可学习 ResFormer 自动发现深层需要更多来自 $\mathbf{V}_1$ 的信息，与手动调优的 Sparse-ResFormer 模式一致

## 亮点与洞察

1. **极简设计，深刻洞察**：仅添加一个 Value 残差连接（几乎零额外参数）即可显著提升性能，设计之简洁令人赞叹
2. **彻底的消融分析**：从"为什么是 Value 而非 QKA"、"为什么是 $\mathbf{V}_1$ 而非 $\mathbf{V}_2$"、"为什么不用额外 Hidden 残差"等多个角度验证设计合理性
3. **SVFormer 的实用价值**：减少近一半 KV 缓存对长序列推理部署极为重要，且可与 GQA 等方法组合
4. **不仅是加速训练**：通过不同学习率实验证明性能提升不是因为快捷连接加速训练，而是真正学到了更好的表征
5. **信息传播视角**：揭示了标准 Transformer 中初始 token 级信息在深层被稀释的重要问题

## 局限与展望

1. SVFormer 需要增加 12.2% 参数才能与 Transformer 达到同等 valid loss，高序列长度下表现更好
2. 仅在 82M-1.6B 参数规模验证，更大模型的效果尚不确定
3. Sparse-ResFormer 的最优层配置需要手动搜索，Learnable 版本虽然自动但可能非最优
4. 未探索 Value 残差与其他高效方法（如 MoE、稀疏注意力）的组合
5. 理论解释主要是直觉性的，缺乏严格的数学证明

## 相关工作与启发

- **快捷连接**：从 ResNet 到 DenseNet 到 DenseFormer 的发展脉络
- **KV 缓存压缩**：MQA、GQA、CLA 等方法，本文首次提出仅解耦 Value
- **过平滑**：Zhou et al. 2021（ViT 32 层不如 24 层）、Shi et al. 2022（BERT 的过平滑）
- 启发：可将 Value 残差应用于已有大模型的微调或继续预训练中，作为低成本架构增强

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Value 残差连接的思路极其简洁却有效，SVFormer 的 Value 解耦也是首创
- 实验充分度: ⭐⭐⭐⭐⭐ 消融实验堪称教科书级别，从多个维度验证了设计选择的合理性
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，每个设计选择都有实验支撑，图表精准有信息量
- 价值: ⭐⭐⭐⭐⭐ 方法简单通用、即插即用，对 Transformer 架构有深远影响

<!-- RELATED:START -->

## 相关论文

- [Residual Matrix Transformers: Scaling the Size of the Residual Stream](../../ICML2025/others/residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)
- [TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)
- [Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [Prediction via Shapley Value Regression (ViaSHAP)](../../ICML2025/others/prediction_via_shapley_value_regression.md)

<!-- RELATED:END -->
