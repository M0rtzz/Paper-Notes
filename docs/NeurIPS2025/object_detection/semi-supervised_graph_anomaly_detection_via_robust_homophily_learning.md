---
title: >-
  [论文解读] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning
description: >-
  [NeurIPS 2025][目标检测][图异常检测] 提出RHO (Robust Homophily Learning)方法，通过自适应频率响应滤波器(AdaFreq)和图正常性对齐(GNA)模块，解决半监督图异常检测中正常节点同质性多样性的问题，在8个真实数据集上超越现有方法。 - 半监督图异常检测(GAD)利用少量标注…
tags:
  - "NeurIPS 2025"
  - "目标检测"
  - "图异常检测"
  - "同质性学习"
  - "自适应频率滤波器"
  - "半监督"
  - "对比学习"
---

# Semi-supervised Graph Anomaly Detection via Robust Homophily Learning

**会议**: NeurIPS 2025  
**arXiv**: [2506.15448](https://arxiv.org/abs/2506.15448)  
**代码**: [GitHub](https://github.com/mala-lab/RHO)  
**领域**: 其他  
**关键词**: 图异常检测, 同质性学习, 自适应频率滤波器, 半监督, 对比学习

## 一句话总结
提出RHO (Robust Homophily Learning)方法，通过自适应频率响应滤波器(AdaFreq)和图正常性对齐(GNA)模块，解决半监督图异常检测中正常节点同质性多样性的问题，在8个真实数据集上超越现有方法。

## 研究背景与动机
- 半监督图异常检测(GAD)利用少量标注正常节点从大量未标注节点中识别异常
- 现有方法假设：(1)正常节点具有相似的同质性水平；(2)标注正常节点能代表整体同质性模式
- **实际问题**：正常节点的同质性差异很大——在Amazon和Elliptic数据集上，部分正常节点具有很低的同质性
- 传统GCN滤波器（低频假设）和BWGNN滤波器（预定义频率响应）都无法适应不同同质性分布的正常节点
- 这导致低同质性正常节点被错误分类为异常

## 方法详解

### 整体框架
RHO包含三个核心组件：
1. **AdaFreq**：自适应频率响应滤波器，同时在跨通道和通道内视图中学习
2. **GNA**：图正常性对齐，强制两个视图的正常性表示一致
3. **单类分类损失**：将正常节点投射到超球面中心附近

### 关键设计

#### AdaFreq: 自适应频率响应滤波器
核心滤波函数：$g(\lambda) = 1 - k\lambda$，其中$k$为可学习参数
- $k > 0$：抑制高频，保留低频（高同质性节点）
- $k < 0$：强调高频（低同质性节点）
- $k = 0$：全通滤波器
- 堆叠$K$层后：$g(\lambda) = \prod_{i=1}^{K}(1 - k_i\lambda)$，可产生复杂频率响应

**跨通道视图**：所有通道共享单一$k$参数
$$H_{ccr}^{(t)} = \sigma((I - k\hat{L})H_{ccr}^{(t-1)}W_{ccr}^{(t)})$$

**通道内视图**：每个通道一个$k_j$，使用Hadamard积实现
$$H_{cwr}^{(t)} = \sigma((I - \hat{L})(H_{cwr}^{(t-1)} \odot K)W_{cwr}^{(t)})$$

#### GNA: 图正常性对齐
- 构建正对：同一节点在两个视图下的表征为正对
- 使用对比学习目标最大化正对相似性，最小化负对相似性
- 双锚点策略：分别以跨通道和通道内视图为锚点计算对比损失

### 损失函数 / 训练策略
总损失：$\mathcal{L}_{total} = \frac{1}{2}(\mathcal{L}_{ccr} + \mathcal{L}_{cwr}) + \alpha \mathcal{L}_{GNA}$

- 单类损失$\mathcal{L}_{ccr/cwr}$：最小化正常节点表征到超球面中心的距离
- 对齐损失$\mathcal{L}_{GNA}$：跨视图对比学习
- 推理时异常分数：节点到两个视图中心的平均距离

## 实验关键数据

### 主实验（AUROC，15%标注正常节点）

| 方法 | Reddit | Tolokers | Photo | Amazon | Elliptic | Question | T-Finance | DGraph |
|-----|--------|----------|-------|--------|----------|----------|-----------|--------|
| GGAD | 0.6354 | 0.5340 | 0.6476 | **0.9443** | 0.7290 | 0.5122 | 0.8228 | 0.5943 |
| BWGNN | 0.5580 | 0.5821 | 0.6861 | 0.8312 | 0.7241 | 0.5740 | 0.7683 | 0.4958 |
| CONSISGAD | 0.5347 | 0.5974 | 0.5859 | 0.8715 | 0.7354 | 0.5737 | 0.8277 | 0.5735 |
| **RHO** | 0.6207 | **0.6255** | **0.7129** | 0.9302 | **0.8509** | **0.5833** | **0.8623** | **0.6033** |

### 消融实验
- 跨通道视图单独使用：在Amazon上部分异常节点被误判为正常（位于中心附近）
- 通道内视图单独使用：部分正常节点偏离中心（产生假阳性）
- **两个视图联合使用(RHO)**：正常节点更好聚集，伪装异常被成功检测

### 关键发现
- RHO在8个数据集中的6个上超越最佳竞争方法GGAD，最大AUROC提升12.19%，AUPRC提升30.68%
- 在Elliptic数据集上效果最突出：AUROC从0.7354(CONSISGAD)提升至0.8509
- AdaFreq在三种不同同质性分布下均表现稳健，而GCN和BWGNN滤波器在某些分布下失败
- 生成异常的方法(GGAD)在Amazon上表现最好，但泛化性不如RHO

## 亮点与洞察
1. **揭示被忽视的问题**：首次系统性揭示半监督GAD中正常节点的同质性多样性问题
2. **理论保证**：定理1证明自适应滤波器可自动放大正常节点频谱一致的分量、抑制不一致的分量
3. **互补双视图设计**：跨通道和通道内视图捕获互补的正常性模式，联合使用显著提升检测能力
4. **无需异常标签**：不依赖任何标注异常数据或生成异常

## 局限与展望
- 超参数α需要按数据集调整（大数据集1.0，小数据集0.1）
- 计算复杂度与边数线性相关，大规模图可能存在效率问题
- 需要至少5%的标注正常节点
- 未探讨动态图或时序图上的扩展
- GNA对比学习中的温度参数τ的选择对性能的影响未详细讨论
- 仅在节点级异常检测中验证，未扩展到边级或子图级异常检测

## 相关工作与启发
- 谱GAD方法(AMNet, BWGNN, GHRN)使用预定义频率响应，缺乏自适应性
- 图同质性建模方法主要通过邻居选择（加/删边）处理同质性差异
- GGAD是首个专门为半监督GAD设计的方法（通过生成异常），但泛化性受限
- RHO首次从频率域角度自适应学习异质正常模式

## 评分
- 新颖性：⭐⭐⭐⭐ （自适应频率滤波+双视图对齐的组合设计新颖）
- 技术深度：⭐⭐⭐⭐⭐ （理论分析+方法设计+广泛实验）
- 实验充分性：⭐⭐⭐⭐⭐ （8个数据集，多种baseline，详细消融）
- 写作质量：⭐⭐⭐⭐ （逻辑清晰，图表丰富）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](../../CVPR2025/object_detection/simltd_simple_supervised_and_semi-supervised_long-tailed_object_detection.md)
- [\[ICML 2025\] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](../../ICML2025/object_detection/causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)
- [\[NeurIPS 2025\] Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)
- [\[CVPR 2025\] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](../../CVPR2025/object_detection/distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)

</div>

<!-- RELATED:END -->
