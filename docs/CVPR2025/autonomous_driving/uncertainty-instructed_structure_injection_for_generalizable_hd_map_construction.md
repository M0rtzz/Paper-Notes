---
title: >-
  [论文解读] Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction
description: >-
  [CVPR 2025][自动驾驶][HD地图构建] 提出 UIGenMap，通过不确定性感知的透视图(PV)检测分支获取显式结构特征，并基于不确定性权重构建 PV prompt 注入 BEV 地图解码器，结合 Mimic Query 蒸馏实现实时推理，在地理不相交数据拆分上取得 +5.7 mAP 的泛化性能提升。
tags:
  - CVPR 2025
  - 自动驾驶
  - HD地图构建
  - 不确定性估计
  - 透视图结构注入
  - 泛化能力
  - 蒸馏
---

# Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction

**会议**: CVPR 2025  
**arXiv**: [2503.23109](https://arxiv.org/abs/2503.23109)  
**代码**: https://github.com/xiaolul2/UIGenMap  
**领域**: 自动驾驶  
**关键词**: HD地图构建, 不确定性估计, 透视图结构注入, 泛化能力, 蒸馏

## 一句话总结

提出 UIGenMap，通过不确定性感知的透视图(PV)检测分支获取显式结构特征，并基于不确定性权重构建 PV prompt 注入 BEV 地图解码器，结合 Mimic Query 蒸馏实现实时推理，在地理不相交数据拆分上取得 +5.7 mAP 的泛化性能提升。

## 研究背景与动机

**领域现状**：在线 HD 地图矢量化已成为自动驾驶感知的重要方向，主流方法基于 Transformer 将透视图(PV)图像特征转换为鸟瞰图(BEV)空间，再由解码器预测地图元素。MapTR、MapTRv2、GeMap、StreamMapNet 等方法在标准 benchmark 上性能不断提升。

**现有痛点**：当前公共数据集（如 nuScenes）的训练集与验证集之间存在大量地理重叠，模型更多是记忆相似场景而非真正学习道路结构。在地理不相交(geo-based)的数据拆分下，现有方法性能显著退化。此外，基于学习的 PV-to-BEV 转换不可避免地引入几何误差和纹理细节丢失。

**核心矛盾**：模型对训练数据分布的过度依赖导致泛化能力不足；隐式的 PV-to-BEV 转换丢失了有价值的结构信息。需要一种方法既能适应不同驾驶场景的特征分布变化，又能补偿 BEV 转换中丢失的显式结构信息。

**本文目标**：(1) 利用不确定性建模增强模型对不同场景的动态适应能力；(2) 引入 PV 显式结构信息补偿 BEV 地图预测；(3) 确保推理时的实时性。

**切入角度**：不确定性估计可学习统计均值和方差，实现基于概率分布的动态重采样，使模型在面对不熟悉环境时具有动态适应性。同时，2D 透视图检测能捕获更直观的语义和角度结构信息，可作为 BEV 预测的可靠补偿。

**核心 idea**：用不确定性引导的透视图结构注入策略(UIGenMap)——在 PV 和 BEV 两个空间设计不确定性感知解码器(UA-Decoder)，基于不确定性加权构建 PV prompt 并通过混合注入机制补偿 BEV 地图预测，最后通过轻量 Mimic Query 蒸馏消除推理时 PV 分支的额外开销。

## 方法详解

### 整体框架

输入为车载环视相机图像，输出为 BEV 空间的矢量化地图元素（类别标签 + 有序点序列）。架构包含：(1) 图像骨干网络提取 PV 特征；(2) BEV 特征通过 PV 特征与可学习 BEV query 交互构建；(3) PV 检测分支用 UA-Decoder 获取 PV 实例坐标和不确定性；(4) UI2DPrompt 模块构建 PV prompt；(5) 混合注入将 PV prompt 融入 BEV 特征和 map query；(6) BEV UA-Decoder 预测最终地图；(7) MQ-Distillation 模块在训练时蒸馏 PV prompt 知识。推理时仅用蒸馏后的 Mimic Query 替代 PV 分支。

### 关键设计

1. **不确定性感知解码器 (UA-Decoder)**:

    - 功能：在实例和点级别引入概率建模，使模型具有动态自适应能力
    - 核心思路：在特征层面设计 UA-Attention——将 deformable attention 中的确定性权重 $\alpha_i$ 改为高斯分布采样 $\alpha_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$，其中均值和方差由 MLP 从 query 预测，通过重参数化技巧实现。在输出层面设计 UA-Head——每个点不仅预测坐标 $(\hat{p}_x^i, \hat{p}_y^i)$ 还预测不确定性 $(\sigma_x^i, \sigma_y^i)$，建模为 Laplace 分布。训练时结合 NLL 损失和点回归损失
    - 设计动机：在驾驶场景多样性大的情况下，确定性注意力权重无法适应挑战性场景。概率性采样提供动态调节能力，不确定性输出为后续特征选择提供可靠的置信度指标

2. **UI2DPrompt (不确定性引导的 2D 提示构建)**:

    - 功能：从 PV 检测结果构建可靠的结构性 prompt，用于补偿 BEV 预测
    - 核心思路：先根据分类得分筛选高置信 PV 实例，通过 IPM（逆透视变换）将 PV 坐标转到 BEV 坐标系。将转换后坐标和不确定性参数分别编码并拼接为点级嵌入 $e_{pv}^i$。不确定性作为权重：$\omega_{pv}^i = \exp((\|\sigma_{pv}^i\|_2)^{-1} / \sum(\|\sigma_{pv}^i\|_2)^{-1})$，不确定性越低权重越大。最终增强 PV prompt 为 $\tilde{e}_{pv}^i = \omega_{pv}^i \cdot e_{pv}^i + e_m^i$（其中 $e_m^i$ 是 Mimic Query）
    - 设计动机：直接使用 PV 检测结果会引入误差，通过不确定性加权可以放大可靠信息、抑制不可靠信息

3. **混合注入与 Mimic Query 蒸馏**:

    - 功能：将 PV prompt 高效注入 BEV 预测流程，并通过蒸馏消除推理时的额外计算
    - 核心思路：混合注入包括 P2BEV（点级 PV prompt 通过 cross-attention 融入 BEV 特征）和 P2Q（实例级 PV prompt 通过 cross-attention 注入 map query）。MQ-Distillation 定义可学习 Mimic Query $e_m^i$ 和 MLP 学习器 $h(\cdot)$，用 MSE 蒸馏损失 $\mathcal{L}_{distill} = \|e_{pv}^i - h(e_m^i)\|^2$ 让 Mimic Query 学习 PV prompt 的结构特征。推理时直接用 Mimic Query 替代 PV 分支
    - 设计动机：PV 分支增加了计算开销，通过蒸馏到轻量查询可保持实时推理能力（UIGenMap-d 版本 12.2 FPS vs 完整版 8.2 FPS）

### 损失函数 / 训练策略

- **总损失**：$\mathcal{L}_{map} = \lambda_1 \mathcal{L}_{pts} + \lambda_2 \mathcal{L}_{cls} + \mathcal{L}_{nll} + \mathcal{L}_{distill}$
- $\mathcal{L}_{pts}$：点回归的 Manhattan 距离损失
- $\mathcal{L}_{cls}$：地图分类的 focal loss
- $\mathcal{L}_{nll}$：不确定性训练的负对数似然损失（Laplace 分布）
- $\mathcal{L}_{distill}$：Mimic Query 的 MSE 蒸馏损失
- 推理时仅用 Mimic Query，不确定性学习使推理时支持动态采样

## 实验关键数据

### 主实验 (nuScenes Region-Based / City-Based)

| 方法 | Backbone | Region mAP | City mAP | FPS |
|------|----------|-----------|---------|-----|
| MapTR | R50 | 20.9 | 15.0 | 15.8 |
| MapTRv2 | R50 | 28.9 | 21.8 | 12.9 |
| StreamMapNet | R50 | 34.1 | 19.3 | 13.3 |
| GeMap | R50 | 27.3 | 18.6 | 11.6 |
| **UIGenMap-d** | R50 | **39.3 (+5.2)** | **22.7 (+3.4)** | 12.2 |
| **UIGenMap** | R50 | **39.8 (+5.7)** | **23.6 (+4.3)** | 8.2 |

### 消融实验

| 组件 | Region mAP | 说明 |
|------|-----------|------|
| Baseline (StreamMapNet) | 34.1 | — |
| + UA-Decoder | ~36 | 不确定性建模提升适应性 |
| + PV 分支 + UI2DPrompt | ~38 | PV 结构补偿效果显著 |
| + 混合注入 (P2BEV+P2Q) | ~39 | 双路注入优于单路 |
| + MQ-Distillation | 39.3 | 蒸馏版本接近完整版精度 |

### 关键发现

- 地理不相交拆分下性能提升最为显著（+5.7 mAP），说明方法对泛化能力提升效果明确
- UIGenMap-d（蒸馏版）在 region-based 仅损失 0.5 mAP 但 FPS 从 8.2 提升到 12.2，实用性强
- 行人横道(Pedestrian)类别提升最大（从 32.2 到 40.3），说明 PV 结构补偿对细粒度元素特别有效
- 使用 SwinT backbone 可进一步提升到 40.6 mAP
- Argoverse2 数据集上同样取得了一致的性能提升

## 亮点与洞察

- **泛化导向的实验设计**：不同于多数 HD 地图工作在标准拆分上刷指标，本文专注于更有实际意义的地理不相交拆分
- **不确定性的双重用途**：既用于动态注意力重采样增强适应性，又用于 PV prompt 的置信度加权选择
- **蒸馏策略实用性强**：UIGenMap-d 在推理时不需要 PV 分支，FPS 与 baseline 相当，适合部署
- PV 空间的显式结构信息确实能补偿 BEV 转换中的信息丢失，这是一个有说服力的思路

## 局限与展望

- IPM 假设地面平坦，对坡道等场景可能不准确
- PV 检测分支增加了训练时间和显存占用
- 不确定性估计的准确性依赖于训练数据的充分性
- 未来可探索更强的 PV-to-BEV 转换方案替代 IPM
- 可考虑将时序信息与不确定性估计结合，进一步提升泛化能力

## 相关工作与启发

- **StreamMapNet**：本文的 baseline，UIGenMap 在其基础上引入 PV 分支和不确定性建模
- **BEVFormerv2 / SimMoD**：PV 检测辅助 BEV 感知的思路在 3D 目标检测中已有应用，本文将其扩展到 HD 地图构建
- **MapQR / GeMap**：从解码器设计和几何关系角度优化地图构建，与本文的不确定性视角互补
- 启发：在泛化性问题上，显式结构先验 + 不确定性估计是一个有效的组合策略

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 7 |
| 技术深度 | 8 |
| 实验充分度 | 8 |
| 写作质量 | 7 |
| 实用价值 | 8 |
| 总评 | 7.6 |

<!-- RELATED:START -->

## 相关论文

- [MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [SafeMap: Robust HD Map Construction from Incomplete Observations](../../ICML2025/autonomous_driving/safemap_robust_hd_map_construction_from_incomplete_observations.md)
- [Stream Query Denoising for Vectorized HD-Map Construction](../../ECCV2024/autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)
- [DAMap: Distance-aware MapNet for High Quality HD Map Construction](../../ICCV2025/autonomous_driving/damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)
- [SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)

<!-- RELATED:END -->
