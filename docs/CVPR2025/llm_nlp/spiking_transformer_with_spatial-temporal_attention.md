---
title: >-
  [论文解读] Spiking Transformer with Spatial-Temporal Attention
description: >-
  [CVPR 2025][LLM/NLP][Transformer] 将空间-时间注意力机制融入脉冲Transformer架构，通过时空解耦的注意力设计和脉冲驱动的自注意机制，在保持SNN能效优势的同时缩小与ANN的性能差距，在多个视觉基准上达到SNN SOTA。
tags:
  - CVPR 2025
  - LLM/NLP
  - Transformer
  - 时空注意力
  - SNN
  - 能效推理
  - 代理梯度
---

# Spiking Transformer with Spatial-Temporal Attention

**会议**: CVPR 2025  
**arXiv**: [2409.19764](https://arxiv.org/abs/2409.19764)  
**代码**: 待确认  
**领域**: 脉冲神经网络 / 高效推理  
**关键词**: 脉冲Transformer、时空注意力、SNN、能效推理、代理梯度

## 一句话总结
将空间-时间注意力机制融入脉冲Transformer架构，通过时空解耦的注意力设计和脉冲驱动的自注意机制，在保持SNN能效优势的同时缩小与ANN的性能差距，在多个视觉基准上达到SNN SOTA。

## 研究背景与动机

**领域现状**：SNN因低功耗和生物可解释性受关注，但与ANN存在显著精度差距。近期Spikformer/Spike-driven Transformer将注意力引入SNN，取得进展。

**核心矛盾**：标准自注意力的softmax和浮点乘法不兼容SNN的二值脉冲特性，直接移植导致能效丧失；而简化版注意力又损失精度。

**核心思路**：将注意力解耦为空间注意力（捕捉patch间关系）和时间注意力（捕捉时间步间动态），分别用脉冲兼容的操作实现。

## 方法详解

### 整体框架
图像 → 脉冲编码 → 多层脉冲Transformer（空间注意力+时间注意力交替） → 分类输出。

### 关键设计

1. **脉冲空间注意力**：用加法替代乘法的线性注意力近似，Q/K/V用脉冲卷积生成，避免softmax
2. **脉冲时间注意力**：跨时间步的脉冲token交互，捕捉时序演变模式
3. **膜电位自适应**：不同层的LIF参数可学习，模拟生物神经元异质性
4. **代理梯度训练**：矩形窗函数近似脉冲函数的梯度

### 损失函数 / 训练策略
交叉熵损失 + 脉冲稀疏性正则化，SGD优化，4时间步。

## 实验关键数据

### 主实验

| 数据集 | 架构 | T | 本文 | 前SOTA | 提升 |
|--------|------|---|------|--------|------|
| CIFAR-10 | ResNet-19 | 4 | **96.8%** | 96.5% | +0.3% |
| CIFAR-100 | ResNet-19 | 4 | **81.5%** | 80.1% | +1.4% |
| ImageNet | ResNet-34 | 4 | **69.8%** | 67.7% | +2.1% |

### 消融实验

| 配置 | CIFAR-100 acc | 说明 |
|------|-------------|------|
| 基线SNN | 78.2% | 无注意力 |
| +空间注意力 | 79.5% | +1.3% |
| +时间注意力 | 80.3% | +2.1% |
| 完整模型 | **81.5%** | +3.3% |

### 关键发现
- 时间注意力贡献大于空间注意力
- 4步即达最优，能耗显著低于ANN
- ImageNet上提升2.1%，大规模任务改进更明显

## 亮点与洞察
- 时空注意力解耦使设计模块化，各部分独立贡献可验证
- 脉冲兼容的注意力保持SNN能效优势
- 自适应膜参数增加了与生物的保真度

## 局限与展望
- 与ANN精度差距仍存在（约6-7%在ImageNet上）
- 注意力的额外计算是否完全被SNN能效优势抵消需更严格分析
- 对视频/事件驱动数据的时间注意力效果有待更多验证

## 评分
- 新颖性: ⭐⭐⭐⭐ 时空解耦注意力在SNN中的应用有新意
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证和消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 推进SNN-ANN差距缩小

<!-- RELATED:START -->

## 相关论文

- [STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)
- [Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)
- [SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)
- [Breaking the Low-Rank Dilemma of Linear Attention](breaking_the_low-rank_dilemma_of_linear_attention.md)
- [Spectral Conditioning of Attention Improves Transformer Performance](../../NeurIPS2025/llm_nlp/spectral_conditioning_of_attention_improves_transformer_performance.md)

<!-- RELATED:END -->
