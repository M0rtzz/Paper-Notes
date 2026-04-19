---
title: >-
  [论文解读] Online Time Series Prediction Using Feature Adjustment
description: >-
  [ICLR 2026][视频理解][在线学习] 提出 ADAPT-Z（Automatic Delta Adjustment via Persistent Tracking in Z-space），将在线时序预测的适应目标从模型参数更新转移到特征空间修正，通过轻量 adapter 融合当前特征与历史梯度来应对多步预测中的延迟反馈问题，在13个数据集上一致超越现有在线学习方法。
tags:
  - ICLR 2026
  - 视频理解
  - 在线学习
  - 分布漂移
  - 特征空间适应
  - 延迟反馈
  - 时间序列预测
---

# Online Time Series Prediction Using Feature Adjustment

**会议**: ICLR 2026  
**arXiv**: [2509.03810](https://arxiv.org/abs/2509.03810)  
**代码**: [有](https://github.com/xiannanhuang/ADAPT-Z)  
**领域**: 时间序列  
**关键词**: 在线学习, 分布漂移, 特征空间适应, 延迟反馈, 时间序列预测

## 一句话总结

提出 ADAPT-Z（Automatic Delta Adjustment via Persistent Tracking in Z-space），将在线时序预测的适应目标从模型参数更新转移到特征空间修正，通过轻量 adapter 融合当前特征与历史梯度来应对多步预测中的延迟反馈问题，在13个数据集上一致超越现有在线学习方法。

## 研究背景与动机

时间序列预测面临 **分布漂移**（distribution shift）这一核心挑战：部署阶段数据的分布随时间持续变化。现有在线学习方法围绕两个问题展开：(1) 更新哪些参数，(2) 如何更新。

现有方法的局限：

- **参数选择偏向**：多数方法更新最后一层参数或引入小型 adapter 模块，但这些可能不是适应分布漂移的最优选择
- **延迟反馈问题**：多步预测（如预测未来24步）中，时刻 $t$ 的真值要到 $t+24$ 才到达，基于延迟梯度的更新可能不可靠
- **训练-部署不匹配**：训练时样本随机打乱，部署时数据按时间顺序到达

**核心洞察**：表面的分布漂移源自底层潜在因素（如经济状况、温度等）的变化。模型可分解为编码器 $f$（提取潜在因素特征 $z$）和预测头 $g$。修正特征 $z$ 比修正模型参数更直接对应分布漂移的根因。

## 方法详解

### 整体框架

ADAPT-Z 将预测模型分为编码器 $f$ 和预测头 $g$。目标：找到修正项 $\delta_t$ 使得 $g(z_t + \delta_t) \approx y_t$，其中 $z_t = f(x_t)$ 是当前特征表示。

核心组件：(1) 双路径 adapter 网络，(2) 历史梯度计算，(3) 延迟在线更新机制。

### 关键设计

**1. 特征空间适应范式**

最简方案——特征空间在线梯度下降（fOGD）：

$$\delta_{t+1} = \delta_t - \eta \frac{\partial (g(z_{t-k} + \delta_{t-k}) - y_{t-k})^2}{\partial \delta_{t-k}}$$

其中 $k$ 为预测步数。但此方法效果有限，原因有二：(a) 多步预测延迟导致梯度过时；(b) 最优修正 $\delta_t$ 可能依赖当前上下文 $z_t$，非固定常量。

尽管如此，实验证明即使简单 fOGD 也能媲美甚至超越复杂的参数更新方法——这挑战了"复杂适应机制必要"的常规认知。

**2. Adapter 网络架构（双路径设计）**

直接拼接 $z_t$ 和梯度会因量级差异大而效果差。ADAPT-Z 采用双路径结构：

- **路径1**：线性层独立变换当前特征 $z_t$
- **路径2**：线性层独立变换历史梯度
- **融合**：两路输出相加，经过两个线性层生成最终的 $\delta_t$

输入包括当前特征向量 $z_t$ 和历史梯度信息，输出为特征修正项 $\delta_t$。

**3. 历史梯度计算**

为降低单样本梯度的高方差，使用批量方式计算。给定批大小 $b$ 和预测步长 $k$，在时刻 $t$ 计算时间戳 $t-k-b$ 到 $t-k$ 的平均损失的梯度作为历史梯度输入。

**4. 延迟在线更新**

部署时使用 $k$ 步延迟在线梯度下降：

- 缓存每个时间步的历史梯度、特征和模型输出
- 收到时刻 $t$ 的真值后，计算 $t-k$ 时刻预测的损失
- 反向传播更新 adapter 参数
- 同时在线更新最后线性层参数

### 损失函数 / 训练策略

- 训练：标准 MSE 损失预训练基础模型
- 部署：在线更新 adapter 参数和最后线性层参数
- 增强版本：可用训练集先微调基础模型并训练 adapter（3个 epoch），再在线部署
- 数据划分：60%训练 / 10%验证 / 30%测试（比之前工作的 25/5/70 更现实）

## 实验关键数据

### 主实验

13个数据集（4个 ETT、4个 PEMS、weather、solar、traffic、electricity、exchange），3个基础模型（iTransformer、SOFTS、TimesNet），预测步长12/24/48。

| 数据集 | 原始 | fOGD | DSOF | SOLID | ADCSD | Proceed | **ADAPT-Z** | 提升 |
|-------|------|------|------|-------|-------|---------|------------|------|
| ETTm1 | 0.2211 | 0.2178 | 0.2647 | 0.2166 | 0.2169 | 0.2168 | **0.1937** | 12.42% |
| solar | 0.1084 | 0.1074 | 0.1038 | 0.1083 | 0.1075 | 0.1083 | **0.0948** | 12.61% |
| traffic | 0.4075 | 0.4068 | 0.4060 | 0.4070 | 0.4070 | 0.4079 | **0.3689** | 9.49% |
| PEMS04 | 0.1288 | 0.1263 | 0.1465 | 0.1291 | 0.1280 | 0.1290 | **0.1223** | 5.05% |
| weather | 0.1575 | 0.1573 | 0.1975 | 0.1573 | 0.1564 | 0.1575 | **0.1481** | 5.98% |

ADAPT-Z 在全部13个数据集上均取得最佳成绩。DSOF 方法在某些数据集上反而劣于原始模型。

### 消融实验

使用训练集微调的增强版本结果：

| 版本 | ETTh1 | ETTm1 | PEMS03 | solar | traffic |
|------|-------|-------|--------|-------|---------|
| ADAPT-Z (仅验证集) | 0.2626 | 0.1954 | 0.0974 | 0.0940 | 0.3314 |
| Version1 (微调+在线更新) | 0.2625 | 0.1948 | 0.0936 | 0.0885 | 0.3197 |
| Version2 (微调+冻结) | 0.2680 | 0.2104 | 0.0945 | 0.1141 | 0.3224 |

特征位置分析（iTransformer）：不同层输出作为特征时性能稳定，但直接修改输入一致变差。平均来看，第一个 Transformer block 输出最优。

### 关键发现

- **fOGD 的惊人表现**：仅做特征空间梯度下降就在很多数据集上排名第二，证明特征修正方向正确
- **冻结版本的"学会适应"现象**：Version2 不做任何在线更新也能降低误差，说明模型通过训练学会了利用前一批次信息进行自适应
- **训练-测试风格不匹配**：现有训练独立打乱样本，但部署时数据有时序关系，未来工作应考虑训练时的样本顺序

## 亮点与洞察

1. **范式转移**：从"更新哪些参数"转向"修正哪些特征"，直击分布漂移根因
2. **简洁有力的 baseline**：fOGD 就能打败大多数复杂方法，挑战了领域常规假设
3. **"学会适应"现象**：揭示了训练带梯度信息可让模型获得内在适应能力
4. **实用性**：轻量 adapter + 即插即用，适配多种预测模型

## 局限与展望

- 数据划分（60/10/30）与之前工作（25/5/70）不同，对基线的在线阶段表现可能有影响
- 特征位置的选择（第几个 block 输出）缺乏理论指导，目前靠实验确定
- 仅测试点预测模型，概率预测模型的适配未探索
- "学会适应"现象值得深入理论分析，目前仅作为实验观察

## 相关工作与启发

- 与 DSOF、SOLID 等对比：它们更新 adapter/最后层参数，ADAPT-Z 更新特征空间
- FSNet 的双流 EMA 策略和 ELF 的直接拟合策略是不同方向的尝试
- 启发：在线学习和测试时训练（test-time training）领域中，特征修正可能是被忽视的更优方案

## 评分

- 新颖性: ⭐⭐⭐⭐ （特征空间适应范式新颖，"学会适应"发现有趣）
- 实验充分度: ⭐⭐⭐⭐ （13个数据集、3个基础模型、多个对比和消融）
- 写作质量: ⭐⭐⭐⭐ （思路清晰，动机说服力强，相关工作总结详尽）
- 价值: ⭐⭐⭐⭐ （为在线时序预测提供了新的思路和简洁有效的方案）

<!-- RELATED:START -->

## 相关论文

- [Towards Efficient General Feature Prediction in Masked Skeleton Modeling](../../ICCV2025/video_understanding/towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](../../AAAI2026/video_understanding/distillation_dynamics_towards_understanding_feature-based_di.md)
- [Predicting Video Slot Attention Queries from Random Slot-Feature Pairs](../../AAAI2026/video_understanding/predicting_video_slot_attention_queries_from_random_slot-feature_pairs.md)
- [VideoNSA: Native Sparse Attention Scales Video Understanding](videonsa_native_sparse_attention_scales_video_understanding.md)
- [BlinkTrack: Feature Tracking over 80 FPS via Events and Images](../../ICCV2025/video_understanding/blinktrack_feature_tracking_over_80_fps_via_events_and_images.md)

<!-- RELATED:END -->
