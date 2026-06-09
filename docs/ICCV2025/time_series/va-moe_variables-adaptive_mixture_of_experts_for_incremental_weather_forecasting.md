---
title: >-
  [论文解读] VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting
description: >-
  [ICCV 2025][时间序列][天气预报] 提出增量天气预报新范式和VA-MoE框架，通过变量自适应的MoE架构和索引嵌入机制，实现在仅25%可训练参数和50%初始训练数据的条件下达到与全量训练可比的预报精度。
tags:
  - "ICCV 2025"
  - "时间序列"
  - "天气预报"
  - "增量学习"
  - "混合专家模型"
  - "变量自适应"
  - "灾难性遗忘"
---

# VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting

**会议**: ICCV 2025  
**arXiv**: [2412.02503](https://arxiv.org/abs/2412.02503)  
**代码**: [https://github.com/chenhao-zju/VAMoE](https://github.com/chenhao-zju/VAMoE)  
**领域**: 时序预测 / 气象预报  
**关键词**: 天气预报, 增量学习, 混合专家模型, 变量自适应, 灾难性遗忘

## 一句话总结

提出增量天气预报新范式和VA-MoE框架，通过变量自适应的MoE架构和索引嵌入机制，实现在仅25%可训练参数和50%初始训练数据的条件下达到与全量训练可比的预报精度。

## 研究背景与动机

数据驱动的AI天气预报模型（如Pangu-Weather、GraphCast等）已取得显著进展，但面临一个根本性限制：假设所有变量在训练和推理时同步可用。实际中：

**变量异构性**：高空变量（温度廓线等）通过探空/卫星稀疏采样，地表变量（降水、风速等）近实时密集更新

**重训成本高昂**：引入新变量（如卫星气溶胶数据）时需完全重训，Pangu-Weather需64天×192 V100 GPU

**灾难性遗忘**：增量引入新变量时，预训练参数向新分布漂移，导致原变量性能严重退化

本文首次提出增量天气预报（IWF）范式，用VA-MoE解决变量动态扩展问题。

## 方法详解

### 整体框架

采用两阶段训练范式：
- **初始阶段**：在40年数据上训练5类高空变量（Z/Q/U/V/T各13层，共65个通道），每类由一个Channel-Adaptive Expert (CAE)处理
- **增量阶段**：冻结已有专家，新增CAE_SV处理5个地表变量（u10/v10/t2m/msl/sp），使用20年（或原40年的一半）数据训练新增模块

模型基于Transformer架构，在每个Transformer Block中用VA-MoE替换FFN层。

### 关键设计

1. **变量索引嵌入（Variable Index Embedding）**: 引入one-hot索引嵌入 $\mathbf{I}_h \in \mathbb{R}^{5 \times N}$ 指导专家学习变量亲和性。索引嵌入通过Linear层编码到潜空间，在CAE模块中与输入特征做通道级乘法，引导专家发展领域特定的专业化能力。增量阶段索引嵌入从 $\mathbb{R}^{(N \times l) \times N}$ 扩展到 $\mathbb{R}^{(N \times l + M \times r) \times (N+M)}$。

2. **通道自适应专家（Channel-Adaptive Expert, CAE）**: 每个CAE处理特定变量类型。关键流程：索引嵌入与输入特征做通道级乘法 → 通过GateEmbed层 → SoftMax + TopK选择Top-K高排名通道 → 得到GateIndex和GateWeight → 选择并加权输入特征 → 送入Expert MLP处理。公式：$\mathbf{I}_Z^{topk}, \mathbf{W}_Z^{topk} = \text{TOP}_k(\text{SoftMax}(\text{MLP}_Z(\mathbf{X}_h^t \odot \mathbf{I}_Z)))$。这是无辅助损失的(auxiliary-loss-free)设计。

3. **共享专家（Shared Expert）**: 并行处理所有变量的整体特征，与各CAE输出求和后通过上采样Linear层恢复通道维度：$(X')_h^t = \text{Expert}_{shared}(X_h^t) + \text{Linear}_{up}(X_h^{t,fused})$。

### 损失函数 / 训练策略

- **动态预测损失**：引入通道级可学习权重 $\mathbf{w}$，对不同变量动态分配梯度：$Obj_{pred} = (\hat{X}^{t+1} - X^{t+1})^2 / e^{\mathbf{w}} + \mathbf{w}$。快变量（温度）获得更大梯度，慢变量（位势高度）权重逐渐调整
- **重建损失**：$Obj_{recon} = (\hat{X}^t - X^t)^2$，确保编解码器专注于特征编解码
- **总损失**：$Obj_{final} = Obj_{pred} + \lambda \cdot Obj_{recon}$
- AdamW优化器，初始阶段lr=0.0002，增量阶段lr=0.00005，两阶段各训练100 epoch，batch size 16
- 16块A100 GPU训练

## 实验关键数据

### 主实验 — 地表变量预测

ERA5数据集上5个地表变量（T2M/U10/V10/MSL/SP）的RMSE对比：

| 方法 | 训练方式 | T2M-6h | T2M-72h | T2M-120h | U10-72h | U10-120h |
|------|---------|--------|---------|----------|---------|----------|
| Pangu-Weather | 全量 | 0.82 | 1.09 | 1.53 | 1.63 | 2.54 |
| GraphCast | 全量 | 0.51 | 0.94 | 1.37 | 1.51 | 2.37 |
| FuXi | 全量 | 0.55 | 0.99 | 1.41 | 1.50 | 2.36 |
| VA-MoE | 全量 | 0.57 | 1.03 | 1.42 | **1.41** | **2.25** |
| VA-MoE(IL) 40yr | 增量 | 0.58 | 1.05 | 1.45 | 1.47 | 2.33 |
| VA-MoE(IL) 20yr | 增量 | 0.73 | 1.17 | 1.57 | 1.58 | 2.49 |

VA-MoE在U10和V10上取得最佳长期预报性能，增量训练（40年数据+一半迭代）几乎匹配全量训练。

### 消融实验 — 架构影响

高空变量500hPa（1.5°分辨率）架构对比：

| 方法 | 参数量(M) | Z500-6h | Z500-72h | Z500-120h | T500-72h | T500-120h |
|------|----------|---------|----------|-----------|----------|-----------|
| ViT | 307 | 33.38 | 209.4 | 517.81 | 1.18 | 2.40 |
| ViT+MoE(light) | 609 | 37.92 | 207.11 | 405.73 | 1.23 | 2.02 |
| ViT+MoE | 1113 | 28.31 | 169.61 | 356.02 | 1.07 | 1.83 |
| **VA-MoE** | **665** | **20.59** | **139.02** | **302.13** | **0.92** | **1.59** |
| VA-MoE(IL) | 137 | 20.29 | 138.52 | 301.41 | 0.93 | 1.60 |

VA-MoE以665M参数显著优于1113M参数的ViT+MoE。增量训练版本（仅137M可训练参数）与全量版本性能几乎相同。

### 关键发现

- VA-MoE(IL)在增量训练后高空变量Z500的长期预测甚至略微优于初始训练版本，证实了无灾难性遗忘
- 仅使用20年数据（50%初始数据量）和25%迭代次数的增量训练仍保持可接受精度
- 索引嵌入和CAE的组合使专家在无辅助损失的情况下实现领域特定专业化
- 6小时全球预测可视化显示最大绝对误差：Z500为0.08%，T850为0.22%

## 亮点与洞察

- **首次提出增量天气预报范式**：将增量学习引入天气预报领域，建立了定量性的benchmark
- **无辅助损失的专家专业化**：通过索引嵌入驱动专家多样性，避免了传统MoE的负载均衡辅助损失
- **动态预测损失**：考虑不同变量分布特性的自适应权重学习，比统一对待所有变量更合理
- **冻结策略简单有效**：增量阶段仅训练新专家和共享专家，完全冻结原有专家，简洁地避免遗忘

## 局限与展望

- 仅验证了从高空变量到地表变量的一步增量，未测试多步增量场景
- 增量阶段仍需使用原始50%高空变量数据，并非完全免数据的增量
- 地表变量仅5个单层变量，与13层高空变量的复杂度差异较大
- 与GraphCast等最强基线在某些变量（如T2M）上仍有差距
- 未探讨极端天气事件的预测能力

## 相关工作与启发

- 与EWMoE的区别：VA-MoE面向增量学习场景，采用变量级专家分配而非任务级
- Expert Gate和Lifelong-MoE在视觉增量学习中的MoE设计提供了灵感
- 对其他需要动态扩展变量的时空预测任务（如交通、能源预测）有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 增量天气预报范式和变量自适应MoE设计新颖
- **实验充分度**: ⭐⭐⭐⭐ ERA5上全面验证，与多个SOTA对比
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰，框架描述系统
- **综合价值**: ⭐⭐⭐⭐ 为气象AI的可扩展性提供了实用解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](../../AAAI2026/time_series/m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)
- [\[CVPR 2026\] STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](../../CVPR2026/time_series/stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)
- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](../../NeurIPS2025/time_series/omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](../../NeurIPS2025/time_series/graph-based_neural_space_weather_forecasting.md)
- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](../../ICML2026/time_series/dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)

</div>

<!-- RELATED:END -->
