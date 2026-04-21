---
title: >-
  [论文解读] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion
description: >-
  [NeurIPS 2025][自动驾驶][interpretability] 提出 LMD（Layer-Wise Modality Decomposition），一种事后、模型无关的可解释性方法，通过逐层线性化神经网络操作将多模态融合模型的预测精确分解为各传感器模态的贡献，首次实现了自动驾驶感知模型中对单个输入模态的预测归因，并在 camera-radar、camera-LiDAR、camera-radar-LiDAR 多种融合设置下验证了有效性。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - interpretability
  - sensor fusion
  - modality decomposition
  - LRP
---

# Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion

**会议**: NeurIPS 2025  
**arXiv**: [2511.00859](https://arxiv.org/abs/2511.00859)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: interpretability, sensor fusion, autonomous driving, modality decomposition, LRP

## 一句话总结

提出 LMD（Layer-Wise Modality Decomposition），一种事后、模型无关的可解释性方法，通过逐层线性化神经网络操作将多模态融合模型的预测精确分解为各传感器模态的贡献，首次实现了自动驾驶感知模型中对单个输入模态的预测归因，并在 camera-radar、camera-LiDAR、camera-radar-LiDAR 多种融合设置下验证了有效性。

## 背景与动机

1. **多传感器融合中可解释性缺失**：自动驾驶中 camera、radar、LiDAR 的融合提升了感知性能，但融合后各模态信息交织在一起，难以判断每个传感器对预测的贡献，阻碍了故障诊断和系统验证。

2. **安全关键场景对透明决策的刚需**：自动驾驶中即使一次感知错误都可能导致灾难性后果，需要清晰地了解模型决策依据，特别是哪个传感器主导了关键预测。

3. **现有可解释方法难以处理多模态融合**：内在可解释模型（GAM/NAM）因结构约束无法捕获跨模态交互；局部代理模型（LIME/LORE）无法处理高维依赖；fANOVA 在高维多模态场景下可扩展性差。

4. **LRP 等归因方法未考虑模态分离**：现有 LRP/DTD 方法将预测归因到输入像素级别，但没有针对多模态融合场景进行模态级别的分解，无法回答"这个检测主要依赖 camera 还是 radar"这类问题。

5. **需要无需修改架构的事后方法**：高性能融合模型的架构不应为了可解释性而被修改，需要一种不影响原始模型性能的事后分析工具。

6. **Shapley 方法计算成本指数爆炸**：基于 Shapley 值的归因需要 O(2^M) 次前向传播来枚举所有模态子集，在模态数量增多时不可行，且存在模态间独立性假设的局限。

## 方法详解

### 整体框架：通过逐层线性化实现模态分解

- **功能**：将预训练多模态融合模型的每一层非线性操作线性化，使整个网络变为线性系统，从而利用叠加原理将输出精确分解为各模态的独立贡献。
- **为什么**：对于线性系统，F(x_c, x_r) = F(x_c, 0) + F(0, x_r) + F(0, 0)（叠加原理），这意味着可以通过分别置零其他模态来精确提取每个模态的贡献，且保证分解的完整性（三项之和等于原始输出）。
- **怎么做**：
  1. **第一遍前向传播**：用完整多模态输入运行原始模型，记录每层非线性操作的行为（激活层的输入输出比、LayerNorm 的方差）
  2. **第二遍前向传播**：用线性化后的层替代原始非线性层，分别传播各模态输入以获得模态特定特征
  3. 最终输出 = camera 贡献 + radar 贡献 + bias 贡献，且严格等于原始模型输出

### 关键设计 1：激活层线性化

- **功能**：将 ReLU 等非线性激活函数替换为元素级乘法操作，其中比率系数是第一遍前向传播中记录的输出/输入比。
- **为什么**：ReLU 的非线性使得叠加原理不成立（即 f(a+b) ≠ f(a) + f(b)），必须将其线性化才能进行模态分解；使用实际运行时的比率而非梯度可以精确保持原始模型的功能行为。
- **怎么做**：对于 ReLU，比率退化为二值掩码 {0, 1}，记录神经元是否激活；对于一般激活函数，c_j^l = F_j^l(x_c, x_r) / (F_j^{l-1}(x_c, x_r) + ε)，即连接两个工作点的线段斜率。

### 关键设计 2：归一化层线性化与偏置分配策略

- **功能**：将 BatchNorm 和 LayerNorm 分解为模态特定项和偏置项，并通过不同的偏置分配规则（identity/uniform/ratio）实现模态分离。
- **为什么**：归一化层中的偏置项（均值减法和仿射变换的 β）不属于任何特定模态，需要合理分配以保持分离属性；LayerNorm 的方差依赖于当前输入，需要特殊处理。
- **怎么做**：
    - **BatchNorm**：统计量在训练时固定，采用 identity rule 将偏置统一归入 bias 项（δ_c = δ_r = 0, δ_b = 1）
    - **LayerNorm**：方差从第一遍前向传播中缓存为常数，采用 ratio rule——对每个模态单独计算其空间维度上的均值进行中心化
    - 实验证明 identity (BN) + ratio (LN) 组合在模态分离指标上最优

### 关键设计 3：扰动评估指标

- **功能**：提出基于模态替换的定量评估指标——将一个模态的输入替换为不相关样本，检查被扰动模态的预测是否变化、未扰动模态的预测是否保持不变。
- **为什么**：现有研究缺乏评估融合网络中模态分解质量的标准方法，需要新的定量指标来衡量分离属性是否被满足。
- **怎么做**：计算扰动前后各模态预测的 Pearson 相关系数和 MSE，理想情况下被扰动模态的相关系数应低（预测变化大），未扰动模态的相关系数应为 1（预测不变）。

## 实验

### 实验 1：线性化效果与模态分离评估

| 方法 | 激活层 | 归一化层 | Rp/R 相关↓ | Rp/C 相关↑ | Cp/R 相关↑ | Cp/C 相关↓ |
|------|--------|---------|-----------|-----------|-----------|-----------|
| Baseline | ✗ | ✗ | 0.22 | 0.76 | 0.22 | 0.09 |
| 仅激活层 | ✓ | ✗ | 0.56 | 0.80 | 0.22 | 0.12 |
| 仅归一化层 | ✗ | ✓ | 0.50 | 0.99 | 0.93 | 0.38 |
| **LMD (Ratio)** | **✓** | **✓** | **0.05** | **1.00** | **1.00** | **0.15** |

**关键发现**：
- LMD 同时线性化激活层和归一化层后实现了近乎完美的模态分离：扰动 radar 后 camera 预测完全不变（相关=1.00），radar 预测剧烈变化（相关=0.05）
- 单独线性化某一类层都不足以实现良好分离，两者缺一不可
- 在 LiDAR+Camera 设置下同样有效（Rp/R=0.09, Cp/C=0.44）

### 实验 2：LMD 变体对比

| BN-LN 规则 | Rp/R↓ | Rp/C↑ | Cp/C↓ |
|-----------|-------|-------|-------|
| Uniform-Identity | 0.50 | 1.00 | 0.42 |
| Identity-Identity | 0.15 | 1.00 | 0.38 |
| Identity-Uniform | 0.18 | 1.00 | 0.38 |
| **Identity-Ratio** | **0.05** | **1.00** | **0.15** |

**关键发现**：
- Identity (BN) + Ratio (LN) 组合在所有分离指标上最优
- Ratio rule 对 LayerNorm 至关重要，采用 identity 或 uniform 规则处理 LN 会导致模态分离不充分
- 不同 BN 规则对结果影响较小，关键瓶颈在 LN 的处理策略

### 实验 3：LMD + SHAP 组合

| 设置 | 指标 | SHAP | LMD + SHAP |
|------|------|------|-----------|
| Radar+Camera | Rp/C↑ | 0.69 | **0.94** |
| Radar+Camera | Cp/R↑ | 0.67 | **0.89** |
| LiDAR+Camera | Lp/C↑ | 0.71 | **0.92** |
| LiDAR+Camera | Cp/L↑ | 0.72 | **0.91** |

**关键发现**：
- LMD+SHAP 的模态分离质量显著优于单独使用 SHAP，Pearson 相关从 0.67-0.72 提升至 0.89-0.94
- 三模态设置下提升同样显著，证明 LMD 可与其他归因方法互补
- LMD 先分解后归因的两阶段策略比直接归因更有效

## 亮点

- **首创性**：据作者所知是第一个在自动驾驶传感器融合中实现模态级预测归因的方法，填补了重要空白
- **理论严谨**：基于一阶 Taylor 展开推导逐层分解公式，证明线性化后满足功能保持约束和分离属性，数学推导完整
- **完全事后、模型无关**：仅需两次前向传播，不修改原始模型架构和权重，不影响模型性能
- **计算高效**：O(1) 额外内存，O(1) 额外计算（2次前向传播），远优于 Shapley 方法的 O(2^M)

## 局限性

- **一阶近似误差**：Taylor 展开忽略高阶项，对高度非线性网络区域的分解可能不够精确
- **注意力机制中的双线性项**：attention 模块中 QK^T 产生的跨模态双线性交互项无法完全归因到单个模态，这是线性分解的固有局限
- **偏置项包含交互信息**：bias 项不仅包含常数偏置，还包含线性化产生的高阶交互近似，可能掩盖部分模态交互信息
- **仅在 BEV 感知模型上验证**：实验集中在 SimpleBEV 等 BEV 融合架构上，对 Transformer-heavy 的融合架构（如纯注意力融合）适用性还需进一步验证

## 相关工作对比

| 维度 | 本文 (LMD) | LRP (Bach et al., 2015) |
|------|-----------|------------------------|
| 归因粒度 | 模态级（camera/radar/LiDAR） | 像素/特征级 |
| 分解对象 | 多模态融合模型的模态贡献 | 单模态模型的输入变量贡献 |
| 前向传播次数 | 2 次 | 2 次前向 + 1 次反向 |
| 分离属性保证 | 有（理论证明） | 无（不针对多模态设计） |
| 适用场景 | 多传感器融合 | 通用深度网络 |

| 维度 | 本文 (LMD) | SHAP (Shapley-based) |
|------|-----------|---------------------|
| 计算复杂度 | O(1)（2次前向） | O(2^M)（枚举所有模态子集） |
| 内存消耗 | O(1) | O(1) |
| 独立性假设 | 无 | 特征间独立假设 |
| 可组合性 | 可与 SHAP 结合提升效果 | 独立使用 |
| 模态分离质量 | 强（扰动相关 0.94-1.00） | 中（扰动相关 0.67-0.71） |

## 评分

- **新颖性**：⭐⭐⭐⭐⭐ — 首次将逐层分解应用于多模态融合的模态归因问题，提出全新的分析视角和评估指标
- **技术深度**：⭐⭐⭐⭐ — 对激活层、BatchNorm、LayerNorm 分别推导线性化方案并证明保持原始功能，理论体系完整；但核心思想（线性化+叠加）相对直观
- **实验充分性**：⭐⭐⭐⭐ — 覆盖 radar-camera、LiDAR-camera、三模态融合配置，提出新评估指标；但主要在 SimpleBEV 一个基础架构上验证
- **实用价值**：⭐⭐⭐⭐ — 对自动驾驶安全审计和故障诊断有直接应用价值，计算开销极低；代码已开源

<!-- RELATED:START -->

## 相关论文

- [Availability-aware Sensor Fusion via Unified Canonical Space](availability-aware_sensor_fusion_via_unified_canonical_space.md)
- [DBLoss: Decomposition-based Loss Function for Time Series Forecasting](dbloss_decomposition-based_loss_function_for_time_series_forecasting.md)
- [Self-Supervised Sparse Sensor Fusion for Long Range Perception](../../ICCV2025/autonomous_driving/self-supervised_sparse_sensor_fusion_for_long_range_perception.md)
- [SiMO: Single-Modality-Operable Multimodal Collaborative Perception](../../ICLR2026/autonomous_driving/simo_single-modality-operable_multimodal_collaborative_perceptio.md)
- [x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](../../ICLR2026/autonomous_driving/x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)

<!-- RELATED:END -->
