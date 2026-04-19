---
title: >-
  [论文解读] Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement
description: >-
  [CVPR 2026][多模态][多模态配准] 提出 HRNet，通过跨尺度特征解纠缠和自适应投影（CDAP）学习干净的共享表示，并在统一的粗到细管线中非迭代地联合预测刚性和非刚性变换，在四个多模态数据集上达到SOTA。
tags:
  - CVPR 2026
  - 多模态
  - 多模态配准
  - 混合变换
  - 特征解纠缠
  - 跨尺度一致性
  - Mamba
---

# Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement

**会议**: CVPR 2026  
**arXiv**: [2603.19623](https://arxiv.org/abs/2603.19623)  
**代码**: [GitHub](https://github.com/Chunlei0913/HRNet) (有)  
**领域**: Multimodal VLM / 多模态图像配准  
**关键词**: 多模态配准, 混合变换, 特征解纠缠, 跨尺度一致性, Mamba

## 一句话总结
提出 HRNet，通过跨尺度特征解纠缠和自适应投影（CDAP）学习干净的共享表示，并在统一的粗到细管线中非迭代地联合预测刚性和非刚性变换，在四个多模态数据集上达到SOTA。

## 研究背景与动机

**领域现状**: 多模态图像配准（如RGB-热红外、RGB-SAR）是跨模态融合的基础。现有深度学习方法多采用多尺度策略提升精度，但通常限于单一变换类型。

**现有痛点**: (a) 大多数多尺度框架仅支持刚性或非刚性中的一种——刚性无法处理局部形变，非刚性在大全局偏移下扭曲结构完整性；(b) 现有混合配准方法采用串行级联（先刚性后非刚性），刚性和非刚性在不同特征空间估计，难以协调且后级继承前级偏差；(c) 共享特征提取方法虽缓解模态差异，但约束主要作用于共享部分，模态私有信息仍会泄漏到共享空间。

**核心矛盾**: 如何在统一的特征空间中同时估计全局刚性对齐和局部非刚性形变，且不受私有模态信息干扰？

**本文目标**: 设计一个统一框架，同时解决：(a) 共享特征空间的模态私有泄漏问题；(b) 刚性和非刚性变换的协调估计问题。

**切入角度**: "解纠缠-再对齐"——先用CDAP学习干净的多尺度共享表示，再在HPPM中联合预测混合变换。

**核心idea**: 表示解纠缠（跨尺度门控+自适应投影）+ 混合参数预测（统一粗到细管线内刚性+非刚性）= 单次前向传播产出统一形变场。

## 方法详解

### 整体框架
输入固定图像 $I_f$ 和移动图像 $I_m$，经过三个阶段：(1) 共享backbone+MSBN提取多尺度特征 $F, M$；(2) CDAP模块解纠缠得到干净共享特征 $(\hat{F}^s, \hat{M}^s)$；(3) HPPM从粗到细预测混合变换 $\phi$，用于warp $I_m$。

### 关键设计

1. **CDAP：跨尺度解纠缠与自适应投影**:

    - **功能**: 遵循"分解-门控-投影"管线，学习多尺度干净共享表示
    - **核心思路**:
        - **Decompose**: 每个尺度用共享权重提取器 $E_{sh}^i$ 提取模态无关共享分量，模态特定提取器 $E_{pf/m}^i$ 提取私有分量
        - **Gate (ILDA)**: 利用邻近尺度语义做跨尺度注意力门控：$\widetilde{F}_i^s = \alpha_i^s \odot F_i^s - \gamma^i \alpha_i^p \odot F_i^p$，其中 $\alpha_i^s, \alpha_i^p$ 通过cross-scale attention计算
        - **Project (DSS)**: 数据自适应生成近似正交基 $W_i^s = Gen^i(z_i^s)$，投影 $\hat{F}_i^s = \widetilde{F}_i^s W_i^{s\top}$
    - **设计动机**: 仅分解不能阻止私有到共享的泄漏（需要门控显式抑制）；固定投影不够灵活（需要自适应基）；跨尺度注意力利用邻近尺度的语义互补信息

2. **HPPM：混合参数预测模块**:

    - **功能**: 在统一管线中非迭代地联合预测刚性和非刚性变换
    - **核心思路**: 5个尺度从粗到细处理。最粗尺度：HRB估计全局刚性参数 $H$（通过GAP+FC），编码为粗形变场。后续各尺度：上采样前级变换 $\phi_{i-1}$ warp当前移动特征，拼接后送入HRB估计增量形变 $\phi_i' = \text{conv}(f_i)$，累积更新 $\phi_i = \text{upsample}(\phi_{i-1}) + \phi_i'$。HRB内部使用2个RSSB（Residual State Space Block，基于Mamba）建模长程依赖
    - **设计动机**: 不同于串行级联的分离估计，HPPM在同一共享特征空间内联合估计，刚性预测立即编码为flow并在后续尺度渐进精化。Mamba的状态空间模型在低计算开销下捕获长程依赖

3. **结构化正则化**:

    - **功能**: 塑造共享特征空间的三个互补正则
    - **核心思路**:
        - $L_{ccd}$（交叉协方差去相关）: $\|\text{Cov}(\hat{F}_i^s, F_i^p)\|_F^2$ → 减少共享-私有耦合
        - $L_{bo}$（基正交性）: $\|W^{(i)}W^{(i)\top} - I\|_F^2$ → 避免子空间退化
        - $L_{cs}$（跨尺度方向一致性）: $1 - \cos(\hat{F}_i^s, \hat{F}_{i+1}^s)$ → 保持跨尺度语义一致
        - $L_{tri}$（三元组损失）: 拉近跨模态共位置共享特征，推远私有干扰
    - **设计动机**: 四种损失从不同维度约束共享空间质量：解耦、非冗余、一致性、对齐

### 损失函数 / 训练策略
- 总损失: $L = \alpha_r L_r + \alpha_n L_n + \alpha_s L_s + \alpha_{tri} L_{tri} + \alpha_{cs} L_{cs} + \alpha_{ccd} L_{ccd} + \alpha_{bo} L_{bo}$
- **三阶段课程训练**: warmup(10%)→mid(50%)→late(40%)，渐进调整损失权重（如 $\alpha_n$: 6→10→12）
- Adam优化器，lr=1e-4，batch=8，100 epochs，图像resize到256×256

## 实验关键数据

### 主实验（刚性配准）

| 方法 | RGB-NIR RE↓ | RGB-TIR RE↓ | RGB-IR RE↓ | RGB-SAR RE↓ |
|------|-------------|-------------|------------|-------------|
| IHN | 3.887 | 3.006 | 5.684 | 7.087 |
| MMRNet | 3.179 | 2.472 | 4.406 | 7.075 |
| **HRNet (Ours)** | **0.785** | **0.744** | **0.578** | **3.161** |

RE降低：75.3%, 69.9%, 86.9%, 55.3%（相对MMRNet，平均~72%）

### 主实验（非刚性配准）

| 方法 | RGB-NIR RE↓ | RGB-TIR RE↓ | RGB-IR RE↓ | RGB-SAR RE↓ |
|------|-------------|-------------|------------|-------------|
| ADRNet (混合) | - | - | - | - |
| MMRNet | - | - | - | - |
| **HRNet (Ours)** | **最优** | **最优** | **最优** | **最优** |

RE相对ADRNet降低：61.2%, 62.5%, 66.9%, 23.3%

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|----------|------|
| w/o CDAP | 共享特征含私有噪声 | 配准精度下降 |
| w/o ILDA门控 | 私有信息泄漏 | 解纠缠不充分 |
| w/o DSS投影 | 跨模态对齐不稳定 | 特征空间不紧凑 |
| 仅刚性 | 无法处理局部形变 | 结构完整性差 |
| 仅非刚性 | 大偏移下扭曲 | 全局对齐不足 |
| **完整HRNet** | **刚性+非刚性统一** | **全面最优** |

### 关键发现
- **混合配准的巨大优势**: 在RGB-IR上RE从4.406→0.578（86.9%↓），证明联合估计远优于单一范式
- RGB-SAR最具挑战性（模态差异最大），但HRNet仍显著领先
- 三阶段课程训练中渐进增加非刚性权重（$\alpha_n$: 6→12）很关键

## 亮点与洞察
- **统一混合框架**: 首次在单一管线中非迭代联合估计刚性+非刚性变换，产出单一统一形变场
- **解纠缠全面**: CDAP的decompose-gate-project管线 + 4种结构化正则，从根本上解决私有信息泄漏
- **Mamba在配准中的应用**: RSSB提供低开销长程依赖建模，适合配准中的全局结构感知

## 局限与展望
- 当前在256×256分辨率验证，更高分辨率下的效率和性能待测试
- 课程训练的超参数调优可能需要根据不同模态对调整
- 对极端遮挡或完全无重叠区域的鲁棒性未讨论

## 相关工作与启发
- 与ADRNet（串行混合）区别：ADRNet分阶段估计，后级继承前级偏差；HRNet在统一空间联合估计
- 与Shi等（特征解纠缠）区别：现有方法仅约束共享部分，HRNet通过ILDA门控和正则显式抑制私有泄漏
- 启发：跨尺度特征交互（邻近尺度门控）是一个值得在其他多尺度任务中探索的通用思路

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一混合配准和CDAP解纠缠设计有价值，Mamba的引入也有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 四个多模态数据集，刚性+非刚性，详细消融和课程训练分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，但符号密度较高
- 价值: ⭐⭐⭐⭐ 为多模态配准提供了通用模板，但应用场景相对垂直

<!-- RELATED:START -->

## 相关论文

- [Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)
- [Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation](locate-then-sparsify_attribution_guided_sparse_strategy_for_visual_hallucination.md)

<!-- RELATED:END -->
