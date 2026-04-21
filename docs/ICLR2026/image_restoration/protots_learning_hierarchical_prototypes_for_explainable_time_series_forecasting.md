---
title: >-
  [论文解读] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting
description: >-
  [ICLR 2026][图像恢复][可解释预测] 提出 ProtoTS，通过层级原型学习实现可解释时间序列预测：少量粗粒度原型提供全局模式概览，逐级细分捕捉局部变化，结合多通道嵌入与瓶颈融合处理异质外生变量。在 LOF 数据集上 MSE 降低 48.3%，MAE 降低 20.9%，且支持专家编辑原型以进一步提升性能。
tags:
  - ICLR 2026
  - 图像恢复
  - 可解释预测
  - 层级原型
  - 外生变量
  - 多通道嵌入
  - 专家可调控
---

# ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting

**会议**: ICLR 2026  
**arXiv**: [2509.23159](https://arxiv.org/abs/2509.23159)  
**代码**: [有](https://github.com/SKURA502/ProtoTS)  
**领域**: 时间序列  
**关键词**: 可解释预测, 层级原型, 外生变量, 多通道嵌入, 专家可调控

## 一句话总结

提出 ProtoTS，通过层级原型学习实现可解释时间序列预测：少量粗粒度原型提供全局模式概览，逐级细分捕捉局部变化，结合多通道嵌入与瓶颈融合处理异质外生变量。在 LOF 数据集上 MSE 降低 48.3%，MAE 降低 20.9%，且支持专家编辑原型以进一步提升性能。

## 研究背景与动机

时间序列预测在电力调度、能源管理、天气预报等高利害场景中广泛应用。在这些场景中，仅有准确预测不够——理解预测原因同样关键，以防止巨大财务损失并建立信任。

现有可解释方法的两个核心缺陷：

- **C1（输出侧）**：TFT、DiPE-Linear 等只解释单个时间步的预测，无法解释整体趋势模式（如"为什么电力负荷曲线在中午、下午、晚间出现三个递减峰值"）。电力调度专家需要理解整体模式才能做出是否外购电力的决策
- **C2（输入侧）**：现有解释仅关注部分输入变量（如 CycleNet 仅关注内生变量）。但预测结果由多种异质变量的交互作用决定（如高温+夏季→空调高峰），需理解它们的联合影响

**ProtoTS 的解决思路**：每个原型对应一种典型时间模式（如"春节模式"、"夏季工作日模式"），通过实例与原型的相似度匹配形成预测。少量原型提供全局概览，层级结构支持逐步深入和专家干预。

## 方法详解

### 整体框架

ProtoTS 由两大模块组成：

1. **多通道原型相似度计算模块**：处理异质输入变量并计算实例-原型相似度
2. **层级原型学习模块**：以树结构组织原型，从粗到细学习时间模式

### 关键设计

**1. 多通道嵌入**

针对内生变量、离散外生变量、连续外生变量设计独立编码通道：

- **内生通道**：$\gamma(\mathbf{y}_t)$，通过带激活函数的 MLP 编码
- **离散外生通道**：$\mathbf{E}_j(\mathbf{x}_{t,j}^{\text{dis}})$，使用独立嵌入表
- **连续外生通道**：$\psi_j(\mathbf{x}_{t,j}^{\text{con}})$，使用变量特定非线性投影

时刻 $t$ 的完整嵌入通过加法聚合：

$$\mathbf{Z}_t = \gamma(\mathbf{y}_t) + \sum_{j=1}^{C_{\text{dis}}} \mathbf{E}_j(\mathbf{x}_{t,j}^{\text{dis}}) + \sum_{j=1}^{C_{\text{con}}} \psi_j(\mathbf{x}_{t,j}^{\text{con}})$$

预测窗口内无 $\mathbf{y}_t$，仅用外生变量。

**2. 瓶颈通道融合**

异质变量聚合后可能引入噪声。ProtoTS 在 MLP-Mixer 架构中引入瓶颈层 $\mathbb{R}^d \to \mathbb{R}^{d_{\text{bottle}}} \to \mathbb{R}^d$（$d_{\text{bottle}} \ll d$），分别沿特征维度和时间维度进行融合：

$$\mathbf{Z}_{1:L+H}^{(l+1)} = \text{MLP}_{\text{time}}(\text{MLP}_{\text{feature}}(\mathbf{Z}_{1:L+H}^{(l)})^T)^T$$

最后通过线性聚合时间维度：$\hat{\mathbf{Z}} = \mathbf{Z}_{1:L+H}^T \mathbf{W} \in \mathbb{R}^d$。

**3. 原型相似度计算与预测**

每个原型包含嵌入 $\boldsymbol{\mu} \in \mathbb{R}^d$ 和时间模式 $\mathbf{p} \in \mathbb{R}^T$（均为可学习参数）。通过欧氏距离 + softmax 计算相似度：

$$f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_c) = \frac{\exp(-d(\hat{\mathbf{Z}}, \boldsymbol{\mu}_c))}{\sum_{i=1}^N \exp(-d(\hat{\mathbf{Z}}, \boldsymbol{\mu}_i))}$$

预测为原型时间模式的加权组合：$\hat{\mathbf{Y}} = \sum_{i=1}^N f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_i) \cdot \mathbf{p}_i$。

**4. 层级原型学习**

- **根层级**：少量原型（如6个）捕获粗粒度模式（季节性、假日等），先训练至收敛
- **分裂策略**：根据归一化损失选择需细化的叶原型（top $\alpha$%），每个分裂为 $M$ 个子原型
- **子层级预测**：

$$\hat{\mathbf{Y}} = \sum_{i=1}^N f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_i) \sum_{j=1}^M f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_{i,j}) \cdot \mathbf{p}_{i,j}$$

分裂规则基于各原型关联实例的平均 MAE 损失：高损失原型说明其时间模式不足以代表关联实例，需进一步细化。

**5. 专家可调控**

- 选择性分裂特定原型（如将"春节"原型分为"节前"和"节中"）
- 引入新根层级原型
- 直接编辑原型的时间模式

### 损失函数 / 训练策略

$$\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_1 - \lambda \sum_{i=1}^N f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_i) \log(f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_i))$$

- L1 预测损失 + 熵正则化（鼓励少数原型覆盖大部分预测）
- 分阶段训练：先根层级→收敛后分裂→继续训练子层级

## 实验关键数据

### 主实验

**LOF 数据集**（电力负荷预测，22个外生变量，4个区域）：

| 模型 | RE | YC | EA | PC | Avg MAE | vs ProtoTS |
|------|-----|-----|-----|-----|---------|-----------|
| **ProtoTS** | **0.198** | **0.055** | **0.059** | **0.112** | **0.106** | - |
| TiDE | 0.253 | 0.057 | 0.061 | 0.164 | 0.134 | +21% |
| iTransformer | 0.279 | 0.080 | 0.097 | 0.139 | 0.149 | +29% |
| TimeXer | 0.272 | 0.079 | 0.096 | 0.182 | 0.157 | +32% |
| XGBoost | 0.405 | 0.084 | 0.092 | 0.230 | 0.203 | +48% |

**EPF 数据集**（电价预测，5个市场）：

| 模型 | NP | PJM | BE | FR | DE | Avg MAE |
|------|-----|-----|-----|-----|-----|---------|
| **ProtoTS** | **0.213** | **0.152** | **0.226** | **0.183** | **0.318** | **0.218** |
| TimeXer | 0.240 | 0.173 | 0.241 | 0.192 | 0.343 | 0.238 |

ProtoTS 在 LOF 上 MSE 降低 48.3%、MAE 降低 20.9%；EPF 上 MSE 和 MAE 均降低 8%。

### 消融实验

| 组件 | PC MSE | YC MSE | RE MSE | EA MSE | Avg MAE |
|------|--------|--------|--------|--------|---------|
| w/o bottleneck | 0.044 | 0.013 | 0.089 | 0.129 | 0.143 |
| w/o multi-channel | 0.034 | 0.006 | 0.108 | 0.007 | 0.117 |
| w/o hierarchy | 0.026 | 0.006 | 0.089 | 0.007 | 0.110 |
| **ProtoTS (完整)** | **0.025** | **0.006** | **0.085** | **0.007** | **0.106** |

### 关键发现

- **数据效率高**：训练数据从100%减少到50%时，ProtoTS 性能下降轻微，而 TimeXer、iTransformer 明显劣化
- **根原型数量**：增至12-15个时趋于饱和，说明典型模式数量有限
- **可解释性量化评估**：24名用户参与，ProtoTS 的 User Precision 77%（TFT 64%、NBEATSx 62%），SUS 得分 73.36（大幅领先）
- **专家编辑案例**：将"春节"原型手动分裂为"节前"和"节中"，春节期间 MSE 降低 0.009

## 亮点与洞察

1. **原型即时间模式**：首次将原型解码为输出序列（如96步预测曲线），而非单一类别标签
2. **全局+局部双层解释**：粗粒度原型给出全局理解，细粒度原型提供局部细节
3. **专家在回路**：可解释性不止于"展示"，还支持专家主动编辑原型来优化模型
4. **处理异质外生变量**：多通道嵌入 + 瓶颈去噪，避免噪声变量干扰

## 局限与展望

- 当前原型的"命名"需人工总结（如"春节模式"），可结合 LLM 自动生成语义标签
- 层级深度和分裂策略依赖启发式规则（top $\alpha$% 损失），自适应方案值得探索
- 仅验证了电力负荷和电价数据集，其他高利害场景（医疗、金融）的适用性待验证
- 与 Foundation Model 的结合（如用 ProtoTS 原型解释大模型预测）是有趣方向

## 相关工作与启发

- **原型网络家族**：从分类扩展到回归序列输出，是原型方法的重要进展
- **CycleNet**：仅发现内生变量周期模式，ProtoTS 同时建模外生变量交互
- **TFT 的 attention 解释**：提供局部逐步解释，ProtoTS 的原型提供更直觉的全局视角
- 启发：在时序预测中，"可解释性"不仅是附加功能，通过原型学习可同时提升准确性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （层级原型→时间模式序列输出，专家可编辑设计，开创性工作）
- 实验充分度: ⭐⭐⭐⭐ （LOF + EPF 数据集，消融完整，用户研究加分）
- 写作质量: ⭐⭐⭐⭐⭐ （电力场景案例详实，层级原型可视化极具说服力）
- 价值: ⭐⭐⭐⭐⭐ （兼顾准确性与可解释性，专家可调控设计直击工业需求）

<!-- RELATED:START -->

## 相关论文

- [TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](../../ICML2025/image_restoration/timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)
- [Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement](../../NeurIPS2025/image_restoration/luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)
- [Mechanism of Task-oriented Information Removal in In-context Learning](mechanism_of_task-oriented_information_removal_in_in-context_learning.md)
- [Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [A Flag Decomposition for Hierarchical Datasets](../../CVPR2025/image_restoration/a_flag_decomposition_for_hierarchical_datasets.md)

<!-- RELATED:END -->
