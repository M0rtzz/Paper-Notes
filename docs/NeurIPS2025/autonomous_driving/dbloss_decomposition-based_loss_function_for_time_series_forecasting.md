---
title: >-
  [论文解读] DBLoss: Decomposition-based Loss Function for Time Series Forecasting
description: >-
  [NeurIPS 2025][自动驾驶][时间序列预测] 提出 DBLoss——一种基于指数移动平均分解的通用损失函数，在预测窗口内将预测值与真实值分别分解为季节和趋势分量并分开计算损失，可即插即用替换 MSE 为任意深度学习预测模型带来一致性提升，在 8 个基准数据集 × 8 个 SOTA 模型上全面验证有效性。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 时间序列预测
  - 损失函数
  - 季节性-趋势分解
  - EMA
  - 通用损失
---

# DBLoss: Decomposition-based Loss Function for Time Series Forecasting

**会议**: NeurIPS 2025  
**arXiv**: [2510.23672](https://arxiv.org/abs/2510.23672)  
**代码**: [https://github.com/decisionintelligence/DBLoss](https://github.com/decisionintelligence/DBLoss)  
**领域**: autonomous_driving / 时间序列预测  
**关键词**: 时间序列预测, 损失函数, 季节性-趋势分解, EMA, 通用损失  

## 一句话总结
提出 DBLoss——一种基于指数移动平均分解的通用损失函数，在预测窗口内将预测值与真实值分别分解为季节和趋势分量并分开计算损失，可即插即用替换 MSE 为任意深度学习预测模型带来一致性提升，在 8 个基准数据集 × 8 个 SOTA 模型上全面验证有效性。

## 研究背景与动机

**领域现状**：长期时间序列预测是经济、交通、能源等领域的关键任务。主流深度模型（DLinear、TimesNet、iTransformer 等）普遍在前向传播中使用季节-趋势分解模块来提取有效表征。

**现有痛点**：标准 MSE 损失直接计算预测值与真实值的逐点差异，但无法显式约束预测结果在季节性和趋势两个维度上的准确性。作者观察到三类失败模式：(a) 季节性预测差但趋势可接受；(b) 趋势预测差但季节性可接受；(c) 二者都差。即使模型前向传播中做了分解，损失函数端仍然"一视同仁"。

**核心矛盾**：前向传播中的分解提取了归纳偏置，但损失函数端未利用这个先验——预测窗口内的季节性和趋势没有被独立监督。

**切入角度**：既然分解在前向传播端有用，何不在损失计算端也做分解？在预测窗口内对预测和 GT 分别做 EMA 分解，对趋势和季节分量分别计算损失再加权融合。

**核心idea**：DBLoss = EMA 分解 + 分量级独立损失 + scale alignment 加权机制，零额外参数、可组合任意 backbone。

## 方法详解

### 整体框架
给定任意 backbone 产生的预测 $\hat{Y}$ 和真实值 $Y$（均为 $\mathbb{R}^{N \times F}$，$N$ 通道 $F$ 预测步），DBLoss 在损失计算阶段执行：(1) EMA 分解→ 得到季节和趋势分量；(2) 分别计算季节 loss 和趋势 loss；(3) scale alignment 后加权求和。整个过程不改变模型结构和推理开销。

### 关键设计

1. **EMA 分解模块**：

    - 功能：将预测和 GT 分解为趋势 $Y_T$ 和季节 $Y_S = Y - Y_T$。
    - 核心思路：对时间序列 $X \in \mathbb{R}^{B \times T \times N}$，计算权重 $W = [(1-\alpha)^{T-1}, (1-\alpha)^{T-2}, \ldots, 1]$，其中 $\alpha \in (0,1)$ 为平滑因子。对 $W[1:]$ 乘以 $\alpha$ 后与输入逐元素相乘，做累积和 $C = \text{cumsum}(X \times W)$，再除以因子 $D_{div}$ 得到趋势 $\text{Trend} = C / D_{div}$，残差即为季节分量 $\text{Seasonality} = X - \text{Trend}$。
    - 设计动机：EMA 相比 SMA 对近期变化更敏感，且计算复杂度为 $O(T)$，无需窗口大小选择。

2. **分量级损失计算**：

    - 季节 loss 使用 L2 范数：$\mathcal{L}_S = |\hat{Y}_S - Y_S|_2$
    - 趋势 loss 使用 L1 范数：$\mathcal{L}_T = |\hat{Y}_T - Y_T|_1$
    - 设计动机：季节分量波动大适合 L2 约束；趋势分量相对平滑，L1 更鲁棒。

3. **Scale Alignment 机制**：

    - 功能：防止某一分量的 loss 因尺度差异而主导优化。
    - 核心公式：$\mathcal{L}_T^{\text{aligned}} = \mathcal{L}_T \times \text{stopgrad}\left(\frac{\mathcal{L}_S}{\mathcal{L}_T + \epsilon}\right)$
    - stopgrad 阻断梯度通过对齐比例回传，避免两个 loss 分量之间的梯度干扰。

4. **最终加权损失**：

    - $\mathcal{L} = \beta \cdot \mathcal{L}_S + (1-\beta) \cdot \mathcal{L}_T^{\text{aligned}}$
    - $\beta$ 平衡季节和趋势的权重，可针对不同应用场景调节。

### 训练策略
- 仅替换损失函数，不修改模型结构、优化器或超参数。
- 使用 TFB 统一评测框架确保公平比较，不使用 "Drop Last" 技巧。

## 实验关键数据

### 主实验——与原始损失对比（8 个数据集 × 4 个 SOTA 模型，Avg MSE/MAE）

| 模型 | 原始 MSE/MAE | DBLoss MSE/MAE | 提升 |
|------|-------------|---------------|------|
| iTransformer | 0.439/0.448 (ETTh1 Avg) | 0.423/0.430 | MSE↓3.6% |
| Amplifier | 0.428/0.435 (ETTh1 Avg) | 0.419/0.425 | MSE↓2.1% |
| PatchTST | 0.419/0.436 (ETTh1 Avg) | 0.402/0.420 | MSE↓4.1% |
| DLinear | 0.425/0.439 (ETTh1 Avg) | 0.412/0.425 | MSE↓3.1% |
| PatchTST | 0.351/0.395 (ETTh2 Avg) | 0.337/0.381 | MSE↓4.0% |
| DLinear | 0.470/0.468 (ETTh2 Avg) | 0.409/0.424 | MSE↓13.0% |

在 **所有** backbone 和 **绝大多数** 数据集上 DBLoss 均优于 MSE。DLinear 这种已包含分解模块的模型同样受益，说明前向分解和损失端分解互补而非冲突。

### 基础模型实验

| Foundation Model | 原始 MSE | DBLoss MSE | 说明 |
|-----------------|---------|-----------|------|
| CALF | baseline | improved | LLM-based 方法同样受益 |
| UniTS | baseline | improved | 预训练方法也有提升 |
| TTM | baseline | improved | 轻量基础模型有效 |
| GPT4TS | baseline | improved | GPT 架构同样受益 |

### 关键发现
- DBLoss 在 ETTh2 + DLinear 上取得最大提升（MSE 从 0.470→0.409，↓13%），这恰恰是分解模型+MSE 原本表现最差的组合，说明 MSE 确实无法有效监督分解后的分量。
- 对已有分解模块的模型（DLinear、DUET）同样有效，前向分解和损失分解互补。
- 计算开销可忽略——仅增加一次 EMA 计算和两个范数运算。
- $\beta$ 默认 0.5 即可在多数场景取得良好效果，对超参不敏感。

## 亮点与洞察
- **极致简洁的设计**：整个方法只有一个 EMA 分解 + 两个分量 loss + 一个 scale alignment，代码量极少，却带来一致且显著的提升。这是"好的损失函数胜过复杂模型"的典范。
- **Scale alignment + stopgrad** 是一个可迁移的 trick——任何多分量 loss 都可以用这个机制避免尺度不平衡。
- **前向分解和损失分解互补**而非冗余——这一发现颠覆了直觉，说明损失端的归纳偏置注入是独立于模型设计的有效维度。

## 局限与展望
- 仅验证了长期多变量预测，短期预测和单变量场景未涉及。
- EMA 平滑因子 $\alpha$ 固定，理论上可学习或自适应选择。
- 对非平稳序列的趋势定义可能不够鲁棒——极端分布偏移场景下 EMA 分解的有效性存疑。
- 未探索多头 loss（如加入频域分量约束）的可能性扩展。

## 相关工作与启发
- **vs Soft-DTW / DILATE**：这类 shape-based loss 关注形状对齐但计算复杂度高（$O(T^2)$）；DBLoss 只需 $O(T)$ 且从分解角度切入，正交互补。
- **vs FreDF**：频域损失关注频率依赖关系；DBLoss 从时域分解切入，二者可组合使用。
- **vs PSLoss**：patch 级结构损失关注局部统计量；DBLoss 关注全局季节/趋势分量。
- 对自动驾驶领域：轨迹预测本质也是时间序列预测，DBLoss 可尝试作为运动预测任务的辅助 loss。

## 评分
- 新颖性: ⭐⭐⭐⭐ 思路简洁但有效，从损失函数端注入分解先验是新视角
- 实验充分度: ⭐⭐⭐⭐⭐ 8 数据集 × 8 模型（含 4 基础模型），覆盖全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机图示直观
- 价值: ⭐⭐⭐⭐ 通用即插即用 loss，实用价值高
