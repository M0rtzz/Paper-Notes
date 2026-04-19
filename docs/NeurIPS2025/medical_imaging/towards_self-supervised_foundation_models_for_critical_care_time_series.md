---
title: >-
  [论文解读] Towards Self-Supervised Foundation Models for Critical Care Time Series
description: >-
  [NeurIPS 2025][医学图像][自监督学习] 基于双轴Transformer（BAT）架构，在多个ICU数据集上进行自监督预训练，构建重症监护时间序列基础模型，在小数据集场景下显著优于监督学习基线。
tags:
  - NeurIPS 2025
  - 医学图像
  - 自监督学习
  - 基础模型
  - 重症监护
  - 时间序列
  - 迁移学习
---

# Towards Self-Supervised Foundation Models for Critical Care Time Series

**会议**: NeurIPS 2025  
**arXiv**: [2509.19885](https://arxiv.org/abs/2509.19885)  
**代码**: [GitHub](https://github.com/Katja-Jagd/YAIB)  
**领域**: 医学图像  
**关键词**: 自监督学习, 基础模型, 重症监护, 时间序列, 迁移学习

## 一句话总结

基于双轴Transformer（BAT）架构，在多个ICU数据集上进行自监督预训练，构建重症监护时间序列基础模型，在小数据集场景下显著优于监督学习基线。

## 研究背景与动机

医疗领域的基础模型在NLP、医学影像等方向已有长足发展，但**重症监护时间序列**方向仍严重不足。其核心原因在于：

**数据稀缺与异构性**：公开可用的ICU数据集数量有限且格式各异，不同医院的监测设备和采样频率差异巨大

**可复现性差**：已有模型往往依赖单一数据集、单一监督任务训练，结果难以在新临床环境中复现

**迁移能力弱**：在一个数据集上训练的模型直接迁移到新数据集效果急剧下降（如AUC-ROC下降4-5个点）

**现有工作闭源**：首个ICU基础模型ICareFM的代码和模型均未公开

本文希望在透明、可复现的框架（YAIB）中，通过池化多个ICU数据集并进行自监督预训练，构建首个**开源**的重症监护时间序列基础模型，特别关注资源受限场景（标注数据少）下的表现。

## 方法详解

### 整体框架

整体思路分两阶段：**自监督预训练 → 监督微调**。

- 预训练阶段：将多个ICU数据集（如eICU + MIMIC-IV）池化，用时间序列**预测任务**做自监督学习
- 微调阶段：在未参与预训练的新数据集（如MIMIC-III）上做死亡率预测的二分类微调

### 关键设计

1. **双轴Transformer（BAT）架构**：BAT在时间轴和特征轴上分别做注意力计算，能同时捕获时序依赖和跨特征关系。输入嵌入由三部分组成：观测值（含缺失指示）、学习到的特征身份嵌入、连续时间位置编码，输出经过池化后与静态特征（年龄、性别等）拼接。BAT的核心优势是**原生处理缺失值**和**不规则采样**，无需均值填充即可建模信息性缺失（informative missingness）。

2. **动态窗口采样策略**：训练时动态构建观测窗口和预测窗口。对每个batch随机选取患者和时间索引，通过三重约束确保有效性：(a) 稀疏性检查——窗口内至少有一个观测值；(b) 最短观测长度$L=12$小时；(c) 预测窗口$H=2$小时可用。这种策略让模型暴露于不同时间段和上下文，增强泛化能力。

3. **双任务预测头设计**：BAT被改造为同时支持两种预测头——自监督预训练使用预测头输出$\hat{\mathbf{X}}^{\text{for}} \in \mathbb{R}^{T^{\text{for}} \times D}$，监督微调使用二分类头输出$\hat{y} \in \{0, 1\}$。微调时可选择仅微调分类头（head fine-tuning）或全模型微调（full fine-tuning）。

### 损失函数 / 训练策略

**预训练损失**：带掩码的MSE损失，仅在预测窗口中有观测值的位置计算误差：

$$\mathcal{L}^{\text{Pre}} = \frac{1}{\sum_{k=2}^{K} N_k} \sum_{k=2}^{K} \sum_{i=1}^{N_k} \left\| \mathbf{M}_i^{\text{for}} \odot (\hat{\mathbf{X}}_i^{\text{for}} - \mathbf{X}_i^{\text{for}}) \right\|_F^2$$

**微调损失**：标准二元交叉熵：

$$\mathcal{L}^{\text{Fine}} = -\frac{1}{N_1} \sum_{i=1}^{N_1} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

预训练使用贝叶斯超参数优化（通过YAIB框架），学习率约3-8e-4，batch size 64，最多200个epoch，早停patience 10-15。使用加权损失处理类别不平衡。

## 实验关键数据

### 主实验

实验使用MIMIC-III、MIMIC-IV和eICU三个数据集，52个临床特征（4个静态+48个时变），按留一法进行预训练和微调。

| 微调数据集 | 数据量 | BAT (全微调) | BAT (头微调) | BAT (从头) | Transformer (从头) |
|---|---|---|---|---|---|
| MIMIC-III | 1000 | **36.24±1.63** | 33.98±1.32 | 27.63±2.23 | 21.34±4.58 |
| MIMIC-III | 5000 | **41.89±1.31** | 38.99±0.96 | 36.30±0.31 | 26.14±0.40 |
| MIMIC-III | 9000 | **43.57±0.94** | 40.14±0.70 | 37.09±1.06 | 27.13±0.41 |
| MIMIC-IV | 1000 | **28.98±0.85** | 26.97±1.71 | 26.12±1.95 | 13.06±1.56 |
| MIMIC-IV | 5000 | **38.10±1.36** | 31.31±1.17 | 34.97±1.03 | 18.00±1.11 |
| eICU | 1000 | **28.37±1.13** | 25.39±1.56 | 20.86±3.31 | 6.58±4.00 |
| eICU | 5000 | **33.89±0.49** | 29.09±0.62 | 30.13±1.37 | 14.41±0.42 |

*指标为AUC-PR（%），预训练数据集为未包含微调数据集的其余两个*

### 消融实验

| 配置 | 关键表现 | 说明 |
|---|---|---|
| 全模型微调 vs 仅头微调 | 全模型AUC-PR高2-6个点 | 全模型微调一致更优 |
| 预训练数据量273K vs 100K | 大数据集预训练迁移更好 | eICU+MIMIC-IV（273K）预训练效果最佳 |
| <5000样本 vs ≥5000样本 | 小数据优势更明显 | 预训练在低数据场景优势最大 |
| 跨数据集直接推理 | AUC-ROC下降1-5个点 | 证明直接迁移效果差 |

### 关键发现

- 预训练模型在**所有>500样本**的场景优于从头训练基线，在**<5000样本**时优势最显著
- 最大预训练数据集（MIMIC-IV+eICU，273K样本）产生最强迁移表示
- 仅微调分类头即可接近全模型微调的性能，表明预训练学到了可迁移的通用表示

## 亮点与洞察

- **首个开源可复现的ICU时间序列基础模型**，所有代码和实验基于YAIB框架
- 巧妙地利用掩码损失处理不规则采样和稀疏数据，避免了对缺失值填充的依赖
- 在资源受限的临床场景中潜力巨大：标注数据少时预训练优势尤为突出

## 局限与展望

- 仅使用了三个美国ICU数据集，数据多样性有限
- 模型参数量仅约1M，与NLP/CV领域的基础模型相比规模极小
- 未探索跨领域（如气象、电力）时间序列数据对预训练的增益
- 在大标注数据场景（≥5000），从头训练的BAT有时也能匹敌预训练模型

## 相关工作与启发

- **ICareFM**：首个ICU基础模型，但未开源，本文提供了开源替代方案
- **YAIB框架**：为可复现的临床ML提供端到端基准测试
- 启发：医疗时间序列的基础模型发展仍处于早期阶段，数据规模是关键瓶颈

## 评分

- 新颖性: ⭐⭐⭐ 方法框架较为直接（预训练+微调），但在ICU时间序列领域属首个开源工作
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、多种数据量、多基线对比、交叉验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，附录详尽
- 价值: ⭐⭐⭐⭐ 开源可复现的ICU基础模型对社区有重要价值

<!-- RELATED:START -->

## 相关论文

- [Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing](generating_multi-table_time_series_ehr_from_latent_space_with_minimal_preprocess.md)

<!-- RELATED:END -->
