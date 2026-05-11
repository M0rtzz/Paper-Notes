---
title: >-
  [论文解读] Unbiased Dynamic Multimodal Fusion
description: >-
  [CVPR 2026][多模态][动态多模态融合] UDML 提出无偏动态多模态学习框架，包含噪声感知不确定性估计器（通过注入可控噪声并预测其强度来实现在低噪和高噪条件下均准确的模态质量评估）和模态依赖计算器（通过 Dropout 量化模型对各模态的固有依赖偏差并融入加权机制），解决了现有方法的双重抑制问题，在多个多模态基准上一致提升性能。
tags:
  - CVPR 2026
  - 多模态
  - 动态多模态融合
  - 不确定性估计
  - 模态依赖偏差
  - 噪声感知
  - 双重抑制
---

# Unbiased Dynamic Multimodal Fusion

**会议**: CVPR 2026  
**arXiv**: [2603.19681](https://arxiv.org/abs/2603.19681)  
**代码**: [https://github.com/shicaiwei123/UDML](https://github.com/shicaiwei123/UDML)  
**领域**: 多模态VLM / 多模态融合  
**关键词**: 动态多模态融合, 不确定性估计, 模态依赖偏差, 噪声感知, 双重抑制

## 一句话总结

UDML 提出无偏动态多模态学习框架，包含噪声感知不确定性估计器（通过注入可控噪声并预测其强度来实现在低噪和高噪条件下均准确的模态质量评估）和模态依赖计算器（通过 Dropout 量化模型对各模态的固有依赖偏差并融入加权机制），解决了现有方法的双重抑制问题，在多个多模态基准上一致提升性能。

## 研究背景与动机

1. **领域现状**：动态多模态学习根据输入数据的模态质量动态调整各模态的贡献权重，主要有基于先验的方法和基于不确定性的方法。
2. **现有痛点**：(1) 不确定性估计偏差：现有经验度量（如能量分数、概率嵌入）在低噪时不敏感（无法检测轻微退化），在高噪时仍给严重损坏的模态分配不可忽略的权重；(2) 双重抑制效应：现有方法假设各模态初始贡献相同，忽视了模型优化过程中产生的模态依赖偏差——难学的模态既被优化偏差抑制，又被高不确定性二次抑制。
3. **核心矛盾**：双重抑制导致动态融合反而不如静态融合，这与动态融合的设计初衷矛盾。
4. **本文目标**：设计一种在各噪声水平下都准确的不确定性估计器，同时量化并补偿模态依赖偏差。
5. **切入角度**：主动注入已知噪声来建立特征损坏与噪声强度的清晰对应关系；用模态 Dropout 量化固有依赖。
6. **核心 idea**：噪声感知估计 + 偏差补偿的双管齐下策略。

## 方法详解

### 整体框架

UDML 是架构无关的通用框架，包含两个核心组件：(1) 噪声感知不确定性估计器，注入可控噪声后从特征预测噪声强度；(2) 模态依赖计算器，通过 Dropout 量化模型对各模态的依赖程度。两者共同决定动态融合权重。

### 关键设计

1. **噪声感知不确定性估计器**:

    - 功能：在所有噪声水平（从无噪声到严重损坏）下准确测量模态质量
    - 核心思路：训练时向模态数据注入已知强度的可控噪声，然后从编码特征预测噪声强度。引入概率表示技术：将每个模态映射为分布（均值编码语义信息，方差反映噪声特性），估计器从方差推导噪声强度。这建立了特征损坏与噪声水平之间的直接监督信号
    - 设计动机：经验度量（能量分数、概率嵌入）缺乏对噪声的直接监督，噪声感知估计器通过显式的噪声预测任务建立准确的对应关系

2. **模态依赖计算器**:

    - 功能：量化多模态网络对各模态的固有依赖偏差并补偿
    - 核心思路：通过模态 Dropout 量化输出对各模态的依赖程度 $\alpha^m$，将依赖度融入权重计算：$w_i^{m_1} = g(\frac{1}{s(z_i^{m_1}) \cdot \alpha^{m_1}})$。高依赖度的模态不会被不确定性过度惩罚，低依赖度的难学模态也不会被双重抑制
    - 设计动机：消除"优化偏差 + 高不确定性"导致的难学模态双重抑制

3. **渐进优化策略**:

    - 功能：在标准训练流程中同时学习多模态表示、噪声估计和主任务
    - 核心思路：渐进式训练，先稳定多模态表示，再逐步引入噪声感知估计和依赖补偿
    - 设计动机：避免多个学习目标之间的干扰

### 损失函数 / 训练策略

总损失 = 主任务损失（分类/检测等）+ 噪声预测损失（MSE）+ KL 散度正则化（概率表示）。

## 实验关键数据

### 主实验

| 数据集 | 任务 | 静态融合 | 动态(PE) | UDML | 提升 vs 动态 |
|--------|------|---------|---------|------|-------------|
| CREMA-D | 音视频分类 | 67.2 | 65.8 | 71.5 | +5.7 |
| Kinetics-Sound | 音视频分类 | 64.1 | 63.5 | 66.8 | +3.3 |
| NYU Depth v2 | RGB-D分割 | 51.2 | 50.8 | 53.1 | +2.3 |

注意：在 CREMA-D 上，PE 动态融合(65.8)反而低于静态融合(67.2)，验证了双重抑制问题的存在。UDML 显著解决了此问题。

### 消融实验

| 配置 | CREMA-D Acc | 说明 |
|------|------------|------|
| 静态融合基线 | 67.2 | 无动态权重 |
| +噪声感知估计器 | 69.8 | 准确估计的贡献 |
| +模态依赖计算器 | 71.5 | 消除双重抑制 |
| w/o 概率表示 | 70.1 | 概率表示帮助泛化 |

### 关键发现

- 噪声感知估计器在所有噪声水平下单调响应，PE 在 $\sigma < 4$ 和 $\sigma > 10$ 时失效
- 模态依赖计算器贡献约 1.7%，说明双重抑制确实是现有方法的重要瓶颈
- UDML 架构无关，在 Concat/Attention/Gating 等多种融合方式上均有提升
- 在高噪声条件下优势更加明显，证明了鲁棒性

## 亮点与洞察

- **双重抑制的发现**：首次清晰地指出"动态融合不如静态融合"的根源是双重抑制，而非方法本身的缺陷
- **噪声注入+预测的估计范式**：比经验度量更有原则性，建立了噪声水平与不确定性的直接因果关系
- **架构无关设计**：所有组件仅操作模态表示，可即插即用到任意多模态模型

## 局限与展望

- 可控噪声注入假设噪声类型已知，实际中退化可能是未知类型
- 模态 Dropout 计算的依赖度是全局统计量，不是逐样本的
- 目前只验证了两模态场景，三模态及以上的扩展性待验证
- 未来可结合更精细的噪声建模（如噪声类型分类）

## 相关工作与启发

- **vs 概率嵌入 (PE)**: PE 经验性地用方差估不确定性，UDML 通过噪声预测任务显式学习
- **vs OGM-GE/Greedy**: 这些方法通过梯度调制解决优化不平衡，但未处理推理时的依赖偏差
- **vs TMC**: TMC 用 Dirichlet 分布建模不确定性，但同样假设模态等贡献

## 评分

- 新颖性: ⭐⭐⭐⭐ 双重抑制的分析深刻，噪声感知估计器设计合理
- 实验充分度: ⭐⭐⭐⭐ 多任务多数据集验证
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，可视化直观
- 价值: ⭐⭐⭐⭐ 对动态多模态融合有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment](the_more_the_merrier_contrastive_fusion_for_higher-order_multimodal_alignment.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)
- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_vision-language_models.md)

</div>

<!-- RELATED:END -->
