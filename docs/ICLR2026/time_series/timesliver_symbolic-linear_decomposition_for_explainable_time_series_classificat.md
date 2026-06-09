---
title: >-
  [论文解读] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification
description: >-
  [ICLR 2026][时间序列][时间归因] 提出TimeSliver——可解释性驱动的深度学习框架,联合利用原始时序数据和符号抽象(分箱)构建保持原始时间结构的表示,每个元素线性编码对应时间段对最终预测的贡献→赋予每个时间点正/负归因分数,在7个数据集上时间归因准确率超越其他方法11%…
tags:
  - "ICLR 2026"
  - "时间序列"
  - "时间归因"
  - "符号抽象"
  - "线性组合"
  - "可解释分类"
  - "正负归因"
---

# TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification

**会议**: ICLR 2026  
**arXiv**: [2601.21289](https://arxiv.org/abs/2601.21289)  
**代码**: [GitHub](https://github.com/pandeyakash23/TimeSliver)  
**领域**: 时间序列/可解释性  
**关键词**: 时间归因, 符号抽象, 线性组合, 可解释分类, 正负归因

## 一句话总结
提出TimeSliver——可解释性驱动的深度学习框架,联合利用原始时序数据和符号抽象(分箱)构建保持原始时间结构的表示,每个元素线性编码对应时间段对最终预测的贡献→赋予每个时间点正/负归因分数,在7个数据集上时间归因准确率超越其他方法11%,同时在26个UEA基准上预测性能持平SOTA。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：DL模型(CNN/LSTM/Transformer)→强分类性能但不可解释。可解释性对高风险应用(医疗/金融/法律)至关重要。

**现有痛点**：

### 现有痛点

**现有痛点**：(1) 事后解释(DeepLift/IntGrad/SHAP)→对基准状态敏感+假设特征独立+跨数据集不泛化

### 核心矛盾

**核心矛盾**：(2) 注意力权重→Transformer中注意力与真正归因不忠实

### 解决思路

**解决思路**：(3) MIL方法→未扩展到多变量

### 补充说明

**补充说明**：(4) 无法区分正/负归因→"推向预测类"vs"推离预测类"

**切入角度**：设计内生可解释的架构→线性组合保证归因可直接计算→不依赖事后方法。

## 方法详解

### 整体框架

TimeSliver 把可解释性写进架构本身：它从原始时序中抽出一条保持时间结构的"段级表示" $Q$，再从分箱后的符号序列抽出一条去噪的"模式级潜向量" $Z$，两者逐元素相乘得到表示 $R=Z\odot Q$，并直接喂给一层线性分类器。因为分类头是线性的，每个时间段对最终预测的贡献都能被精确拆出来，从而无需任何事后近似就读出正/负归因分数。

### 关键设计

**1. 双路编码：原始段表示 $Q$ 与符号潜向量 $Z$ 互补。** 单靠原始信号会把高频噪声当成信号，单靠符号抽象又会丢掉精细的时间定位，TimeSliver 让两者各司其职。第一路把时间序列切成与原始时间位置一一对应的若干段，每段过编码器得到段级表示向量 $Q$，保留"第 $k$ 段对应原始哪一段时间"的对应关系，使归因能落回具体时间点。第二路先对序列做分箱（binning）得到符号化序列，再经编码器得到潜在时间向量 $Z$；分箱本身是一次有损压缩，它抹掉无关的数值抖动、保留模式级结构，等于给模型一个"先看形状再看细节"的归纳偏置。两路在时间维度上对齐，为下一步逐元素组合打好基础。

**2. 符号-线性分解：用 $R=Z\odot Q$ 把"可解释"变成架构保证而非事后近似。** 事后解释之所以不忠实，根源在于真实模型是高度非线性的，任何梯度或扰动都只是局部近似。TimeSliver 直接让表示层做逐元素线性组合 $R=Z\odot Q$，再接线性分类层 $\hat{y}=W\cdot R+b$。这样第 $k$ 段对某预测类的贡献就是闭式的 $W\cdot(Z_k\cdot Q_k)$，没有任何隐藏的非线性纠缠。线性不是为了简单，而是为了让"贡献"这个概念在数学上严格成立——这也意味着归因不再依赖某个人为选定的基准状态，天然规避了 DeepLift/IntGrad 对基准敏感、SHAP 假设特征独立的老问题。

**3. 正/负归因分离：区分"推向"与"推离"预测类。** 多数方法只给一个标量重要性，无法回答"这一段到底是支持还是反对当前判断"。由于贡献项 $W\cdot(Z_k\cdot Q_k)$ 带符号，TimeSliver 直接把第 $k$ 段拆成正归因 $\phi_k^{+}$（对预测类的正贡献部分）和负归因 $\phi_k^{-}$（负贡献部分）。整个归因函数 $f_{att}$ 是非参数的，直接从表示 $R$ 与权重 $W$ 读出，无需再训练或近似。区分正负让解释从"哪里重要"升级到"哪里支持、哪里反对"，在医疗、故障诊断这类需要完整决策图景的场景里信息量明显更高。

下表对比 TimeSliver 与事后方法在归因机制上的根本差异：

| 特性 | 事后方法 | **TimeSliver** |
|------|---------|---------------|
| 归因来源 | 梯度/perturbation | 架构内生 |
| 基准依赖 | 是 | 否 |
| 正/负归因 | 不区分 | **区分** |
| 忠实性 | 存疑 | **保证(线性)** |

## 实验关键数据

### 时间归因质量(7个数据集)


### 主实验

| 方法 | 归因准确率 | 说明 |
|------|----------|------|
| DeepLift | 基线 | 事后 |
| IntGrad | 中 | 事后 |
| Grad-CAM | 低 | 不适合时序 |
| SHAP | 中 | 慢 |
| Attention | 低(不忠实) | 内在 |
| **TimeSliver** | **+11%** | 内生线性 |

### 预测性能(26个UEA基准)


### 消融实验

| 方法 | 平均准确率 | 可解释性 |
|------|----------|---------|
| SOTA(各种) | 最好 | 无 |
| **TimeSliver** | **-2%(追平)** | **强** |

### 关键发现
- 线性组合不损失预测能力→可解释性和性能不矛盾
- 正/负归因→揭示哪些时间段"支持"vs"反对"预测→比单一归因更丰富
- 符号抽象→帮助忽略无关波动→关注结构模式
- 跨域一致→音频/睡眠/故障诊断都work

## 亮点与洞察
- **"线性=可解释的保证"**：不是用复杂方法近似归因→而是用线性架构保证归因精确→设计层面解决问题。
- **"正+负归因"的信息量**：知道某时间段"支持预测"还不够→还知道"反对预测"的时间段→为决策提供完整图景。
- **符号抽象的优雅**：分箱看似简单→但压缩了不必要细节→让模型关注结构模式而非数值→类似人类的时序理解方式。
- **预测-可解释性的帕累托前沿**：TimeSliver在两个轴上都好→不是牺牲一个换另一个。

## 评分
- 新颖性: ⭐⭐⭐⭐ 符号-线性分解的架构创新
- 实验充分度: ⭐⭐⭐⭐⭐ 7归因数据集+26UEA基准+12基线
- 写作质量: ⭐⭐⭐⭐ 可解释性概念清晰
- 价值: ⭐⭐⭐⭐ 对可解释时间序列分析有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](../../AAAI2026/time_series/counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[ICLR 2026\] Weight-Space Linear Recurrent Neural Networks](weight-space_linear_recurrent_neural_networks.md)
- [\[AAAI 2026\] ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition](../../AAAI2026/time_series/probfm_probabilistic_time_series_foundation_model_with_uncertainty_decomposition.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[NeurIPS 2025\] Decomposition of Small Transformer Models](../../NeurIPS2025/time_series/decomposition_of_small_transformer_models.md)

</div>

<!-- RELATED:END -->
