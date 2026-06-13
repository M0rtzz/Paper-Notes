---
title: >-
  [论文解读] Embracing Large Language Models in Traffic Flow Forecasting
description: >-
  [ACL 2025][自动驾驶][交通流预测] 提出 LEAF 框架，用图分支（pair-wise关系）和超图分支（non-pair-wise关系）的双分支预测器生成候选预测，再用冻结的 LLM 作为选择器（判别而非生成）挑选最优预测，通过 ranking loss 反馈优化预测器，在 PEMS 数据集上取得 SOTA。
tags:
  - "ACL 2025"
  - "自动驾驶"
  - "交通流预测"
  - "LLM判别能力"
  - "图神经网络"
  - "超图"
  - "Ranking Loss"
---

# Embracing Large Language Models in Traffic Flow Forecasting

**会议**: ACL 2025  
**arXiv**: [2412.12201](https://arxiv.org/abs/2412.12201)  
**代码**: [https://github.com/YushengZhao/LEAF](https://github.com/YushengZhao/LEAF)  
**领域**: 自动驾驶  
**关键词**: 交通流预测, LLM判别能力, 图神经网络, 超图, Ranking Loss

## 一句话总结
提出 LEAF 框架，用图分支（pair-wise关系）和超图分支（non-pair-wise关系）的双分支预测器生成候选预测，再用冻结的 LLM 作为选择器（判别而非生成）挑选最优预测，通过 ranking loss 反馈优化预测器，在 PEMS 数据集上取得 SOTA。

## 研究背景与动机

**领域现状**：交通流预测是智能交通系统的核心问题，主流方法使用 GNN/RNN/Transformer 捕获时空关系。最近开始有工作尝试将 LLM 引入交通预测。

**现有痛点**：(1) 现有方法假设训练/测试分布一致，但交通条件会因特殊事件、天气、时代变迁而发生分布偏移，导致性能下降；(2) 图只能捕获 pair-wise 关系，超图只能捕获 non-pair-wise 关系，单一结构不够。

**核心矛盾**：用 LLM 做交通预测的直觉方法是让 LLM 直接"生成"预测值，但交通数据涉及复杂的时空关系，对语言模型的生成能力来说太困难——LLM-MPE 在多个数据集上甚至不如简单的 GNN 方法。

**本文目标**：如何利用 LLM 的泛化和推理能力来增强交通流预测，同时避免让 LLM 直接处理复杂的时空关系？

**切入角度**：不用 LLM 的"生成能力"，而用其"判别能力"——让 LLM 从多个候选预测中选择最合理的一个。

**核心 idea**：用传统时空模型生成候选，用 LLM 做选择器，用选择结果通过 ranking loss 反馈训练预测器。

## 方法详解

### 整体框架
LEAF 由两部分组成：(1) 双分支预测器——graph branch 捕获 pair-wise 时空关系，hypergraph branch 捕获 non-pair-wise 关系；(2) LLM-based selector——冻结的 LLaMA 3 70B 从候选集中选择最优预测。流程：预训练双分支 → 测试时生成预测 → 构建候选集（加变换） → LLM 选择 → ranking loss 微调预测器 → 迭代。

### 关键设计

1. **时空图构建与 Graph Branch**:

    - 功能：将 $T \times N$ 的时空数据展开为时空图，用 GCN 捕获 pair-wise 时空关系
    - 核心思路：构建时空图 $\mathcal{G}^{ST}$，节点为 $TN$ 个时空节点，边包括空间边（同一时刻相邻传感器）和时间边（同一传感器相邻时刻）。用标准 GCN 卷积 $X^{(l)} = \sigma(\hat{A}^{ST} X^{(l-1)} W_G^{(l)})$ 传播信息，7 层
    - 设计动机：图分支擅长建模局部传播效应（如一个路口堵车影响邻近路口）

2. **Hypergraph Branch**:

    - 功能：学习超图关联矩阵，捕获 non-pair-wise 群组关系
    - 核心思路：用可学习的关联矩阵 $I_H = \text{softmax}(X_H^{(l-1)} W_H)$，通过 $X_H^{(l)} = I_H(I_H^\top X_H^{(l-1)} + \sigma(W_E I_H^\top X_H^{(l-1)}))$ 做超图卷积。第一项建模超边内节点交互，第二项建模超边间交互
    - 设计动机：住宅区→商业区的早高峰通勤是典型的 non-pair-wise 关系——一组节点同步变化，不能用 pair-wise 图边表达

3. **候选集构建与 LLM 选择器**:

    - 功能：对双分支预测结果施加多种变换扩展候选集，用 LLM 从中选择
    - 核心思路：变换包括平滑、上升趋势（1-12%线性增）、下降趋势、高估（+5%）、低估（-5%），加上原始预测共 12 个候选。构建包含任务描述、时空信息、历史数据、候选集的 prompt，让 LLaMA 3 70B 选择最优项
    - 设计动机：(1) 更多候选给 LLM 更大空间应对分布偏移（如周一早高峰 LLM 可选上升趋势）；(2) LLM 做选择（判别）远比直接生成数值简单，能利用其常识推理能力

4. **Ranking Loss 反馈**:

    - 功能：用 LLM 的选择结果通过 ranking loss 反向训练预测器
    - 核心思路：$\mathcal{L}^G = [\Delta(y_i^G, \hat{y}_i) - \inf_{y_i' \in \mathcal{C}_i \setminus \{\hat{y}_i\}} \Delta(y_i^G, y_i') + \epsilon]_+$，要求预测器输出更接近被选中的候选而非次优候选
    - 设计动机：因为 LLM 选择的不一定是 ground truth，直接用 MSE/MAE 会引入噪声。Ranking loss 只要求相对排序正确，更鲁棒

### 损失函数 / 训练策略
- 预训练阶段：双分支各自用 MAE loss 训练
- 测试时适应：Ranking loss（Huber distance，margin $\epsilon=0$），每轮更新 $M=5$ 步，prediction-selection 迭代 $K=2$ 轮
- 隐藏维度 $d=64$，7 层，batch 训练

## 实验关键数据

### 主实验

| 方法 | PEMS03 MAE | PEMS04 MAE | PEMS08 MAE | PEMS08 RMSE | PEMS08 MAPE |
|------|-----------|-----------|-----------|-------------|-------------|
| DCRNN (GNN+RNN) | 29.99 | 34.36 | 31.41 | 43.91 | 15.44% |
| STSGNN (GNN) | 28.21 | 33.43 | 29.58 | 41.95 | 12.90% |
| DyHSL (超图) | 27.10 | 33.36 | 27.34 | 39.05 | 11.56% |
| STAEformer (Transformer) | 27.87 | 33.77 | 27.43 | 38.16 | 11.36% |
| LLM-MPE (LLM生成) | 33.82 | 35.63 | 26.42 | 40.02 | 10.61% |
| **LEAF (ours)** | **25.46** | **31.49** | **24.68** | **36.07** | **10.56%** |

### 消融实验（PEMS08）

| 配置 | MAE | RMSE | MAPE |
|------|-----|------|------|
| Graph branch only | 29.12 | 41.36 | 13.54% |
| Hypergraph branch only | 27.94 | 39.11 | 11.82% |
| w/o hypergraph | 26.29 | 38.18 | 12.83% |
| w/o graph | 25.80 | 37.23 | 11.00% |
| w/o transformation | 25.47 | 36.47 | 11.01% |
| w/o ranking loss | 25.41 | 37.00 | 11.34% |
| **LEAF** | **24.68** | **36.07** | **10.56%** |

### 关键发现
- **LLM 做判别远优于做生成**：LLM-MPE（生成）在大网络 PEMS03 上 MAE 33.82，不如简单 GNN；LEAF（判别）MAE 25.46，大幅领先
- **双分支互补**：去掉任一分支均导致性能下降，pair-wise 和 non-pair-wise 关系都重要
- **LLM 选择器的作用显著**：单独 graph branch MAE 29.12 → 加 LLM＝26.29，单独 hypergraph 27.94 → 加 LLM＝25.80
- **Ranking loss 比直接拟合好**：去掉 ranking loss 后 RMSE 从 36.07 上升到 37.00
- **长期预测优势更大**：LEAF 在 12-step 预测中前几步与分支差异不大，但后续步骤误差显著降低

## 亮点与洞察
- **"用 LLM 做选择而非生成"**：这是最核心的设计洞察。LLM 擅长理解语义和常识推理（如"下午 7 点高峰结束"），但不擅长精确数值预测。将其定位为判别器而非生成器是巧妙的能力匹配
- **Ranking loss 容忍选择噪声**：LLM 选择不完美，ranking loss 只要求相对排序正确，优雅地处理了噪声监督信号
- **变换增强候选集**：简单的变换（趋势/平滑/偏移）就能给 LLM 足够的"行动空间"来适应分布偏移

## 局限与展望
- 仅在 PEMS 交通数据集上验证，缺少其他时空预测任务（如气象、能源）
- LLM 未经微调，LoRA 等参数高效微调可能进一步提升选择器质量
- 迭代轮次 K>2 时性能下降，因为没有跨轮上下文传递，会重复考虑相同因素
- 使用 LLaMA 3 70B 推理成本较高，实际部署需考虑效率
- 训练数据仅用 10%，在完整数据下的表现未知

## 相关工作与启发
- **vs LLM-MPE**: LLM-MPE 让 LLM 直接生成预测值，在大网络上表现差。LEAF 改用判别方式，避开了 LLM 处理复杂时空关系的短板
- **vs DyHSL**: DyHSL 是 LEAF 超图分支的前身，LEAF 通过加入图分支和 LLM 选择器进一步提升
- **vs STAEformer**: 纯 Transformer 方案缺少显式图/超图结构建模

## 评分
- 新颖性: ⭐⭐⭐⭐ "LLM 做判别者"的角色定位很有创意，但双分支设计本身是已有工作的组合
- 实验充分度: ⭐⭐⭐ 仅 3 个 PEMS 数据集，10% 训练数据设置较特殊
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，可视化分析有说服力
- 价值: ⭐⭐⭐⭐ 为时空预测中利用 LLM 提供了实用范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Distilling Multi-modal Large Language Models for Autonomous Driving](../../CVPR2025/autonomous_driving/distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [\[AAAI 2026\] Generalising Traffic Forecasting to Regions without Traffic Observations](../../AAAI2026/autonomous_driving/generalising_traffic_forecasting_to_regions_without_traffic_observations.md)
- [\[AAAI 2026\] TimeBill: Time-Budgeted Inference for Large Language Models](../../AAAI2026/autonomous_driving/timebill_time-budgeted_inference_for_large_language_models.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](../../CVPR2026/autonomous_driving/traffic_scene_generation_from_natural_language_description_for_autonomous_vehicl.md)
- [\[ECCV 2024\] Navigation Instruction Generation with BEV Perception and Large Language Models](../../ECCV2024/autonomous_driving/navigation_instruction_generation_with_bev_perception_and_large_language_models.md)

</div>

<!-- RELATED:END -->
