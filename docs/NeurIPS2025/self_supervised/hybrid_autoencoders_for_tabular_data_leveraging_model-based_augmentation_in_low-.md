---
title: >-
  [论文解读] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings
description: >-
  [NeurIPS 2025][自监督学习][表格数据] 提出 TANDEM（Tree-And-Neural Dual Encoder Model），一种混合自编码器架构，通过联合训练神经网络编码器和遗忘软决策树（OSDT）编码器，并引入样本级随机门控网络作为可学习的数据增强，在低标签表格数据场景下实现了超越强基线（包括树模型和深度学习方法）的性能。
tags:
  - NeurIPS 2025
  - 自监督学习
  - 表格数据
  - 混合自编码器
  - 遗忘软决策树
  - 低标签学习
---

# Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings

**会议**: NeurIPS 2025  
**arXiv**: [2511.06961](https://arxiv.org/abs/2511.06961)  
**代码**: 无  
**领域**: 自监督学习 / 表格数据  
**关键词**: 表格数据, 自监督学习, 混合自编码器, 遗忘软决策树, 低标签学习

## 一句话总结

提出 TANDEM（Tree-And-Neural Dual Encoder Model），一种混合自编码器架构，通过联合训练神经网络编码器和遗忘软决策树（OSDT）编码器，并引入样本级随机门控网络作为可学习的数据增强，在低标签表格数据场景下实现了超越强基线（包括树模型和深度学习方法）的性能。

## 研究背景与动机

**领域现状**：表格数据是医疗、金融等领域的主导数据格式，梯度提升决策树（GBDT/XGBoost/CatBoost）在表格数据上通常优于深度神经网络，是实际应用的首选。

**现有痛点**：(1) 神经网络存在光谱归纳偏置（spectral bias），倾向于拟合平滑的低频函数，难以捕获表格数据中复杂的高频模式；(2) 自监督学习（SSL）在表格域面临数据增强困难——常见增强如噪声注入或特征值交换容易破坏关键特征关系；(3) 掩码自编码器（MAE）在异质表格数据上也有局限。

**核心矛盾**：低标签场景下 SSL 意义重大，但表格数据缺乏有效的增强策略，传统增强方法容易生成不真实样本。

**本文目标**：在低标签表格数据上学习有效的自监督表征，使下游分类/回归超越传统方法。

**切入角度**：从模型增强（model-based augmentation）取代数据增强——利用树模型的归纳偏置引导神经网络学习更好的表征。

**核心 idea**：将 OSDT 编码器作为训练时的"模型增强器"，通过共享解码器和对齐损失把树模型的表格友好归纳偏置传递给神经网络编码器。

## 方法详解

### 整体框架

TANDEM 是一个双编码器-共享解码器的掩码自编码器：
- 输入 $x \in \mathbb{R}^D$ 先经过样本级随机门控网络（STG）得到特征掩码 $g(x) \in [0,1]^D$
- 掩码后的视图 $\tilde{x} = x \odot g(x)$ 并行送入：(i) 全连接神经编码器 → $z^{NN}$，(ii) OSDT 集成编码器 → $z^{OSDT}$
- 共享解码器 $h$ 分别重建 $\hat{x}^{NN}$ 和 $\hat{x}^{OSDT}$
- 预训练后，推理时仅使用神经编码器 + 轻量分类/回归头

### 关键设计

1. **Oblivious Soft Decision Tree (OSDT) 编码器**:

    - **功能**：作为可微分的树编码器，提取表格数据的结构化表征
    - **为什么**：树模型天然适合表格数据，能捕获尖锐的高频模式和条件特征交互，弥补神经网络的光谱偏置
    - **怎么做**：固定深度 $L$ 的遗忘决策树集成，每层共享投影向量 $w_\ell$。软路由概率：$p_{\text{leaf}}(x) = \prod_{\ell=1}^{L} [\sigma_\ell^+(x)]^{b_\ell} \cdot [\sigma_\ell^-(x)]^{1-b_\ell}$。最终表征为所有树的叶分布均值 $z^{OSDT}(x) = \frac{1}{T}\sum_{t=1}^T f_t^{OSDT}(x) \in \mathbb{R}^{2^L}$
    - **区别**：仅在训练时使用，推理时丢弃，避免了树模型的泛化限制

2. **Stochastic Gating Network (SGN)** 作为样本级增强:

    - **功能**：为每个输入样本学习一个特征掩码，实现样本级的特征选择
    - **为什么**：替代传统的固定数据增强（噪声、交换等），提供保持语义结构的可学习输入变换
    - **怎么做**：门控网络 $f_\theta(x)$ 输出参数 $\mu(x)$，通过截断高斯扰动采样 $g(x) = \max(0, \min(1, 0.5 + \mu(x) + \epsilon))$，$\epsilon \sim \mathcal{N}(0, \sigma^2)$
    - **区别**：神经编码器使用单个全局门控，OSDT 编码器在每个树层使用独立门控 $g_\ell^{OSDT}(x)$，支持层级化特征选择

3. **联合训练目标**:

    - **重建损失**：$\mathcal{L}_{\text{recon}} = \frac{1}{N}\sum(\|x - \hat{x}^{OSDT}\|_2^2 + \|x - \hat{x}^{NN}\|_2^2)$
    - **对齐损失**：$\mathcal{L}_{\text{align}} = \frac{1}{N}\sum\|\hat{x}^{OSDT} - \hat{x}^{NN}\|_2^2$（重建输出一致性）
    - **潜在表征相似性损失**：$\mathcal{L}_{\text{LRS}} = \frac{1}{N}\sum(1 - \frac{\langle z^{NN}, z^{OSDT} \rangle}{\|z^{NN}\| \cdot \|z^{OSDT}\|})$（余弦距离）

### 损失函数 / 训练策略

- 预训练 100 个 epoch，batch size 128，RMSprop 优化器
- 超参数通过 Optuna 在 50 次试验中基于验证损失选择
- 下游评估：单层 MLP，编码器冻结 25 epoch 后微调 25 epoch
- 门控网络在微调时冻结

## 实验关键数据

### 主实验

**分类（19 个数据集，400 标签）：**

| 方法 | Mean Accuracy | Mean Rank |
|------|-------------|-----------|
| MLogReg | 0.6380 | 6.16 |
| MLP | 0.6721 | 4.84 |
| XGBoost | 0.6706 | 4.47 |
| CatBoost | 0.6731 | 4.16 |
| TabPFN | 0.7012 | 2.56 |
| **TANDEM** | **0.7124** | **1.58** |

**回归（13 个数据集，400 标签）：**

| 方法 | Mean MSE | Mean Rank |
|------|---------|-----------|
| CatBoost | 0.3318 | 4.00 |
| XGBoost | 0.3405 | 4.15 |
| MLP | 0.3877 | 4.38 |
| **TANDEM** | **0.3234** | **3.38** |

TANDEM 在分类和回归上均取得最佳均值和最佳平均排名。

### 消融实验

**分类消融（400 标签）：**

| 变体 | Mean Accuracy | Mean Rank |
|------|-------------|-----------|
| SS-AE（标准自编码器） | 0.6815 | 4.45 |
| SS-AE + Gating | 0.6941 | 3.61 |
| OSDT AE + Gating（仅树） | 0.6600 | 4.71 |
| TANDEM (no gate) | 0.6966 | 2.92 |
| TANDEM (no LRS + Align) | 0.6971 | 2.79 |
| **TANDEM（完整）** | **0.7124** | **1.74** |

- 移除任何编码器或门控都会降低性能
- 完整 TANDEM 始终最优
- 仅 OSDT 编码器性能最差（0.6600），说明树模型单独用作编码器不够灵活

### 关键发现

- 双编码器架构比单编码器显著更好，证明了互补归纳偏置的价值
- 门控网络作为可学习增强比固定增强更有效
- TANDEM 在 50-1000 标签的宽范围内均表现稳健
- 光谱分析揭示两个编码器捕获了不同且互补的频率成分
- 在 TabPFN 最强的分类基准上仍取得最佳结果

## 亮点与洞察

- **模型增强替代数据增强**：核心洞察是将树模型的归纳偏置作为增强手段注入神经网络，而非依赖不可靠的表格数据增强
- **推理时仅用神经网络**：OSDT 仅训练时使用，不增加推理开销，保持了与下游任务的灵活兼容性
- **门控网络的双重身份**：既是特征选择器又是增强器，在不破坏语义的前提下提供有效的输入变换
- **互补光谱分析**：从频域视角解释了双编码器为何有效——神经网络捕获低频，树捕获高频

## 局限与展望

- 预训练需要 2000 样本/类的无标签数据，在极端小数据场景可能不适用
- OSDT 的深度 L 和树数量 T 是关键超参数，需要仔细调优
- 在个别数据集上 TANDEM 仍不及某些基线（如 BF 回归数据集上 MSE 1.0057 > CatBoost 的 0.6565）
- 未探索与 Transformer-based 表格方法（如 SAINT、FT-Transformer）的结合

## 相关工作与启发

- **NODE (popov2019node)**：遗忘可微决策树的基础工作
- **TabPFN (hollmann2023tabpfn)**：通过合成数据预训练的概率推理分类方法，主要竞争对手
- **VIME / SCARF / SubTab**：现有表格SSL方法，性能均不及 TANDEM
- **STG (yamada2020)**：随机门控网络的原始提出
- 启发：利用异构模型的归纳偏置互补是提升SSL表征质量的有效策略，可推广到其他模型组合

## 评分

- 新颖性: ⭐⭐⭐⭐ 双编码器+模型增强的设计思路新颖，门控作为增强的视角独特
- 实验充分度: ⭐⭐⭐⭐⭐ 19个分类+13个回归数据集，100次重复，50-1000标签范围，消融全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法阐述系统完整
- 价值: ⭐⭐⭐⭐ 对低标签表格数据学习有实际意义，方法通用性强

<!-- RELATED:START -->

## 相关论文

- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [TabArena: A Living Benchmark for Machine Learning on Tabular Data](tabarena_a_living_benchmark_for_machine_learning_on_tabular_data.md)
- [Towards Benchmarking Foundation Models for Tabular Data With Text](../../ICML2025/self_supervised/towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models](../../ICCV2025/self_supervised/to_label_or_not_to_label_palm_-_a_predictive_model_for_evaluating_sample_efficie.md)
- [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)

<!-- RELATED:END -->
