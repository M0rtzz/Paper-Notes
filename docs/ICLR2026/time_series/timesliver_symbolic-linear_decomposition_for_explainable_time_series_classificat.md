---
title: >-
  [论文解读] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification
description: >-
  [ICLR 2026][时间序列][时间归因] 提出TimeSliver——可解释性驱动的深度学习框架,联合利用原始时序数据和符号抽象(分箱)构建保持原始时间结构的表示,每个元素线性编码对应时间段对最终预测的贡献→赋予每个时间点正/负归因分数,在7个数据集上时间归因准确率超越其他方法11%,同时在26个UEA基准上预测性能持平SOTA。
tags:
  - ICLR 2026
  - 时间序列
  - 时间归因
  - 符号抽象
  - 线性组合
  - 可解释分类
  - 正负归因
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

### TimeSliver三模块架构

**模块I: 时间段表示学习(Q)**
- 将时间序列分段→每段用encoder→得到段级表示向量
- 保持段与原始时间位置的对应关系

**模块II: 符号抽象潜在向量(Z)**
- 将时间序列分箱(binning)→符号化表示
- 用encoder→得到潜在时间向量Z
- 符号化→消除高频噪声+捕获模式级信息

**模块III: 线性组合→可解释表示**
- **关键**: R = Z ⊙ Q (元素级线性组合)
- R直接线性传给分类层: ŷ = W·R + b
- 因为线性→每个时间段的贡献=W·(Z_k·Q_k)
- 正贡献→推向预测类; 负贡献→推离预测类

### 归因分数计算(f_att)
- 正归因$\phi_k^+$: W·(Z_k·Q_k)中对预测类正贡献的部分
- 负归因$\phi_k^-$: 对预测类负贡献的部分
- 非参数操作→直接从表示和权重计算→无需近似

### 与事后方法的根本区别

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

## 相关论文

- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](../../AAAI2026/time_series/counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[ACL 2025\] LETS-C: Leveraging Text Embedding for Time Series Classification](../../ACL2025/time_series/lets-c_leveraging_text_embedding_for_time_series_classification.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)

<!-- RELATED:END -->
