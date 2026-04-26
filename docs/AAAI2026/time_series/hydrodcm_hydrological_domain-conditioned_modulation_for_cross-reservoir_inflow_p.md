---
title: >-
  [论文解读] HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction
description: >-
  [AAAI 2026][时间序列][域泛化] 提出 HydroDCM，首次将域泛化(Domain Generalization)引入水文预测领域，通过空间元属性构建伪域标签指导对抗学习提取不变特征，再用 FiLM 适配器根据目标水库的地理信息调制特征，实现对未见水库的跨域入流预测。
tags:
  - AAAI 2026
  - 时间序列
  - 域泛化
  - 跨水库入流预测
  - 对抗训练
  - FiLM调制
  - 空间元属性
---

# HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction

**会议**: AAAI 2026  
**arXiv**: [2512.03300](https://arxiv.org/abs/2512.03300)  
**代码**: [github.com/humphreyhuu/HydroDCM](https://github.com/humphreyhuu/HydroDCM)  
**领域**: 时间序列  
**关键词**: 域泛化, 跨水库入流预测, 对抗训练, FiLM调制, 空间元属性

## 一句话总结

提出 HydroDCM，首次将域泛化(Domain Generalization)引入水文预测领域，通过空间元属性构建伪域标签指导对抗学习提取不变特征，再用 FiLM 适配器根据目标水库的地理信息调制特征，实现对未见水库的跨域入流预测。

## 研究背景与动机

水库入流预测（Reservoir Inflow Prediction）对防洪调度、水资源分配和水力发电至关重要。深度学习方法在单个水库上已取得良好效果，但面临两大核心挑战：

**域迁移问题**：每个水库的入流模式受独特的气候条件、地理位置和集水区操作影响，在一个水库上训练的模型直接应用到另一个水库时性能严重下降。

**数据稀缺问题**：新建水库或无观测站水库缺乏充足的历史数据，无法独立训练可靠的预测模型。

传统域泛化(DG)方法在水文领域面临两个特殊困难：

- **多域分类难题**：水库网络通常包含数十甚至上百个站点，每个水库应被视为独立域以充分捕获域特异性。但元学习等 DG 方法的计算开销随域数线性增长，对大规模水文系统不切实际。DANN 等方法假设少量潜在域，难以处理细粒度域划分。
- **元属性依赖**：水库拥有丰富的空间和环境元数据（经纬度、海拔等），这些信息对区分水库具有间接但显著的影响。但现有 DG 方法主要关注不变特征学习，忽略了辅助元数据的利用。

**核心动机**：能否设计一个既能高效处理大量域、又能利用空间元属性的 DG 框架？这正是首次将 DG 扩展到水文应用的关键挑战所在。

## 方法详解

### 整体框架

HydroDCM 包含四个模块，分两阶段运行：
- **阶段1**（训练时）：时间特征提取器 $f_\phi$ + 域判别器 $d_\theta$（对抗训练去除域信息）
- **阶段2**（训练+推理）：FiLM 适配器 $m_\beta$（注入域特定信息）+ 预测头 $p_\omega$
- **推理时**：丢弃域判别器，仅保留 $f_\phi$、$m_\beta$、$p_\omega$

### 关键设计

1. **空间元属性驱动的伪域标签**：不同于传统 DG 为每个域分配 one-hot 标签，HydroDCM 使用空间元属性向量 $\mathbf{s}_i \in \mathbb{R}^3$（经度、纬度、海拔）作为伪域标识符。具体做法是将元属性和观测数据拼接后通过线性投影得到伪域嵌入 $\mathbf{v}_i = \mathbf{W}[\mathbf{s}_i : \mathbf{X}_i] + \mathbf{b}$，然后用对比学习损失确保地理相近的水库在嵌入空间中更近：

   $$L_{\text{con}} = -\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_i^+)/\tau)}{\exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_i^+)/\tau) + \sum_{j \in \mathcal{N}} \exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_j^-)/\tau)}$$

   **设计动机**：连续的空间元属性比离散域标签更适合多域场景——它天然编码了域间的相似度关系，同时避免了 one-hot 编码在域数量大时的计算爆炸。

2. **对抗训练泛化**：域判别器 $d_\theta$ 试图从时间特征 $\mathbf{h}_i = f_\phi(\mathbf{X}_i)$ 中预测伪域标签 $\mathbf{v}_i$，而特征提取器通过对抗训练被迫去除域相关信息：

   $$L_{\text{adv}} = -\mathbb{E}_{(\mathbf{x}_i, \mathbf{v})} \|d_\theta(\mathbf{h}_i) - \mathbf{v}_i\|_2^2$$

   梯度仅通过判别器参数 $\theta$ 传播，方向取反以鼓励判别器失败。这样提取的特征 $\mathbf{z}$ 保留了与标签（入流量）相关的水文动力学信息，同时滤除了域特异性噪声。

   **设计动机**：与标准 DANN 不同，使用连续伪域标签而非离散类别标签，可以在高域数下保持计算效率，同时更好地捕获域间的渐变关系。

3. **FiLM 域条件调制**：对抗训练后的不变特征可能丢失了水文意义上有用的域特定信息。FiLM 适配器通过空间元属性生成缩放和平移系数，对不变特征进行仿射变换：

   $$\tilde{\mathbf{z}}_i = \gamma(\mathbf{s}_i) \odot \mathbf{z}_i + \delta(\mathbf{s}_i)$$

   其中 $\gamma$ 和 $\delta$ 由元属性 $\mathbf{s}_i$ 通过轻量 MLP 生成。调制后的特征送入预测头 $p_\omega$ 输出未来 7 天入流量 $\hat{y}_i = \text{MLP}(\tilde{\mathbf{z}}_i)$。

   **设计动机**：这是"先去后加"策略的精髓——先通过对抗训练移除域偏差，再通过 FiLM 以可控方式重新注入域特定信息。相比直接在不变特征上预测，FiLM 调制可以融入地形梯度和气候变异等空间先验。

### 损失函数 / 训练策略

总损失为三项加权和：

$$L_{\text{total}} = \lambda_{\text{con}} L_{\text{con}} + \lambda_{\text{adv}} L_{\text{adv}} + \lambda_{\text{sup}} L_{\text{sup}}$$

其中 $L_{\text{sup}}$ 为 MSE 回归损失。训练分两阶段：
- **前 10 轮**：仅用 $L_{\text{con}} + L_{\text{sup}}$，学习不变表示
- **后续轮次**：加入 $L_{\text{adv}}$ 和 FiLM 调制

超参数设置：$\lambda_{\text{sup}}=1.0$，$\lambda_{\text{adv}}=0.1$，Adam 优化器，初始学习率 $10^{-3}$，ReduceLROnPlateau 调度，梯度裁剪 max norm=1.0。特征提取器采用 2 层 Encoder-Decoder LSTM，隐藏维度 64。

## 实验关键数据

### 主实验

在 Upper Colorado River Basin 的 30 个水库上评估，27 个源域训练，3 个目标域（MCR、JVR、MCP）测试。指标为 Nash-Sutcliffe Efficiency (NSE, %)：

| 方法 | Overall NSE | Day 1 | Day 4 | Day 7 | 类别 |
|------|-------------|-------|-------|-------|------|
| Base (下界) | 78.29 | 87.96 | 78.95 | 67.95 | 仅源域训练 |
| Few-shot | 80.08 | 89.08 | 80.19 | 71.03 | 目标域少量数据 |
| DANN | 78.89 | 88.63 | 79.50 | 68.94 | DG 基线 |
| MLDG | 80.67 | 89.82 | 80.53 | 71.42 | DG 基线 |
| CondAdv | 80.77 | 90.06 | 80.73 | 71.61 | DG 基线 |
| IRM | 78.50 | 88.15 | 79.13 | 68.20 | DG 基线 |
| **HydroDCM** | **82.90** | **92.92** | **82.26** | **73.96** | 本文 |
| Oracle (上界) | 83.93 | 93.79 | 83.25 | 75.29 | 目标域全数据 |

HydroDCM 在所有预测日上一致优于所有 DG 基线，整体 NSE 仅比 Oracle 低 1.03%。

### 消融实验

| 配置 | Overall NSE | 说明 |
|------|-------------|------|
| HydroDCM (完整) | **82.90** | 全部模块 |
| w/o Adversarial Loss | 79.09 | **下降最大**，对抗训练是核心 |
| w/o Contrastive Loss | 80.30 | 域间分离不清晰导致性能下降 |
| w/o FiLM Adaptor | 81.39 | 缺少域适应微调，稳定下降 |
| w/ Spatial Shuffle | 80.63 | 空间元属性被破坏，退化 |

### 关键发现

1. **对抗损失最关键**：去除对抗训练导致最大性能下降（-3.81%），证明去除域协变量是跨水库泛化的核心
2. **HydroDCM 超越 Few-shot**（+2.82%无需目标域监督）：说明 DG 方案在水文场景可能比少量标注数据更有竞争力
3. **长预测日优势更明显**：Day 7 的 NSE 优势达 2.5~5%，表明 FiLM 调制有效抑制了预测不确定性的累积
4. **空间元属性提供有效信号**：Spatial Shuffle 的性能下降证实了地理信息的重要性

## 亮点与洞察

- **首次将 DG 引入水文预测**：填补了水文领域域泛化研究的空白，定义了合理的问题设置和评估协议
- **伪域标签设计精巧**：用连续空间元属性替代离散域标签，天然适配多域场景
- **"先去后加"策略**：对抗训练去除域偏差 + FiLM 重新注入域信息，比单纯的不变特征学习更灵活
- **计算高效**：FiLM 适配器参数极少，推理时仅需一次前传，适合实时预报

## 局限与展望

1. **目标域仅 3 个水库**：评估规模偏小，需要在更大流域（如全美 CAMELS 数据集的 671 个流域）上验证
2. **空间元属性仅 3 维**：仅用经纬度+海拔，未包含更丰富的流域特征（如集水面积、年均降水、土壤类型等）
3. **预测长度固定 7 天**：未评估更长预测范围下的泛化能力
4. **时间特征提取器选择有限**：仅用 LSTM，未尝试 Transformer 等更强的时序编码器

## 相关工作与启发

- **FiLM 调制(Perez et al., 2018)**：源自视觉QA领域，用条件信息调制特征，本文将其创新性地用于水文域适应
- **DANN 族**：标准梯度翻转层的变体，本文用连续伪标签替代离散域标签，值得在其他多域场景中推广
- **水文大模型方向**：Kratzert 等人倡导"不要在单个流域上训练 LSTM"——本文提供了一种从多流域泛化到未见流域的具体方案

## 评分

- 新颖性: ⭐⭐⭐⭐ — DG 框架本身不新，但在水文领域的应用和伪域标签设计有创新
- 实验充分度: ⭐⭐⭐ — 数据集规模偏小（30 水库/3 目标），消融设计合理但缺乏多流域验证
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，动机论证充分，图示直观
- 价值: ⭐⭐⭐⭐ — 对水文领域有重要实用价值，但泛化到其他领域的价值有待验证

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] ResCP: Reservoir Conformal Prediction for Time Series Forecasting](../../ICLR2026/time_series/rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)
- [\[AAAI 2026\] Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)
- [\[AAAI 2026\] Finding Time Series Anomalies using Granular-ball Vector Data Description](finding_time_series_anomalies_using_granular-ball_vector_data_description.md)
- [\[AAAI 2026\] IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?](idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](../../ICLR2026/time_series/adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)

<!-- RELATED:END -->
