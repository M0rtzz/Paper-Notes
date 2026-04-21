---
title: >-
  [论文解读] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection
description: >-
  [ICML 2025][自监督学习][子集选择] 本文系统研究了基础模型（FM）替代传统信息提取器（IE）用于子集选择的优劣，发现 FM 在细粒度数据集上显著优于传统 IE，并提出 RAM-APL 方法，利用多个 FM（DINOv2 + CLIP）从类内和类间两个维度联合衡量样本重要性，在三个细粒度数据集上达到 SOTA。
tags:
  - ICML 2025
  - 自监督学习
  - 子集选择
  - 基础模型
  - 细粒度分类
  - 多模型融合
  - 数据高效训练
---

# Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection

**会议**: ICML 2025  
**arXiv**: [2506.14473](https://arxiv.org/abs/2506.14473)  
**代码**: [GitHub](https://github.com/ZhijingWan/RAM-APL)  
**领域**: 自监督学习  
**关键词**: 子集选择, 基础模型, 细粒度分类, 多模型融合, 数据高效训练

## 一句话总结

本文系统研究了基础模型（FM）替代传统信息提取器（IE）用于子集选择的优劣，发现 FM 在细粒度数据集上显著优于传统 IE，并提出 RAM-APL 方法，利用多个 FM（DINOv2 + CLIP）从类内和类间两个维度联合衡量样本重要性，在三个细粒度数据集上达到 SOTA。

## 研究背景与动机

子集选择（coreset selection）旨在从大规模训练数据中选出一个小的代表性子集，在不显著损失模型性能的前提下降低训练开销。传统的 one-shot 子集选择方法依赖预训练模型作为**信息提取器（IE）**，来获取特征/梯度/不确定性分数等信息。然而传统 IE 需要在目标数据集上预训练，存在**数据集依赖性**问题，限制了在大规模新数据集上的应用。

基础模型（Foundation Models, FMs）——如 DINOv2、CLIP、SigLIP、EVA-CLIP——凭借强大的泛化能力，有望替代传统 IE，构建**数据集无关**的子集选择流程。但先前工作（Xie et al., 2023）发现直接使用 FM 并不总是优于传统 IE，这引发了两个核心问题：

1. FM 在什么条件下能超越传统 IE？
2. 不同 FM 的表现是否一致？

本文通过大量实验回答了这两个问题，并基于发现设计了新方法。

## 方法详解

### 整体框架

本文提出的方法包含三个层面：

**第一层：Single-Model Study（单模型研究）**  
在 5 个数据集（CIFAR-10、CIFAR-10N、CIFAR-10I、Oxford-IIIT Pet、Pet-N）上，分别用三类 IE（目标数据集预训练模型 model-TD、TinyImageNet 预训练模型 model-TIN、单个 FM）配合四种经典选择算法（MIN、K-Center Greedy、Graph Cut、Moderate\_DS）在 10%/30%/50% 采样率下进行实验。得出三个关键观察：

- **Observation 1**：FM 在含噪声的粗粒度数据集上优势有限
- **Observation 2**：FM 在细粒度数据集（无论是否含噪声）上显著且一致地优于传统 IE
- **Observation 3**：不同 FM 的子集选择效果不同，且 FM 在下游任务上的准确率更高并不意味着子集选择更好

**第二层：Multi-Model Pipeline（多模型流程）**  
基于 Observation 2 和 3，提出使用多个 FM 联合作为 IE 的新流程，避免了单模型流程中需要额外步骤选择最优 FM 的问题。

**第三层：RAM-APL 方法**  
在多模型流程下设计同时考虑**类内分布**和**类间分布**的样本重要性度量方法。

### 关键设计

#### RAnking Mean（RAM）—— 类内分布度量

RAM 解决的核心问题是：不同 FM 提取的特征维度不同、空间不对齐，如何融合？

**核心思路**：将特征从不对齐的特征空间映射到统一的**距离排名空间**。

具体步骤：

1. 对每个 FM $M_F^i$ 提取特征集 $\mathcal{F}^i$
2. 计算每个类别 $c$ 的中心特征 $\tilde{F}_c^i$（类内所有样本特征的均值）
3. 计算每个样本到其所属类中心的欧氏距离 $d(F_j^i, \tilde{F}_c^i)$
4. 在每个类内按距离排序，得到排名值 $r_j^i$
5. 对所有 FM 的排名取平均并归一化：$\bar{r}_j = \frac{1}{m \times |S|} \sum_{i=1}^{m} r_j^i$

排名均值越小 → 样本越接近类原型 → 类内代表性越强。这种排名空间映射巧妙地绕过了特征维度不对齐的问题。

#### Accuracy of Pseudo-class Labels（APL）—— 类间分布度量

APL 衡量样本在不同 FM 特征空间中**是否容易被错分到其他类**。

具体步骤：

1. 对每个 FM $M_F^i$，计算所有 $C$ 个类的中心特征
2. 对每个样本计算到所有类中心的距离，将最近类中心的类别作为**伪类标签** $\tilde{y}_j^i$
3. 若伪类标签 = 真实标签，得分 $\varphi_j^i = 1$；否则 $\varphi_j^i = 0$
4. 对所有 FM 取平均：$\bar{\varphi}_j = \frac{1}{m} \sum_{i=1}^{m} \varphi_j^i$

$\bar{\varphi}_j$ 越低 → 样本越容易被误分 → 与其他类更相似 → 是类间决策边界上的困难样本。

#### 最终评分与选择

将两个维度线性组合为最终评分：

$$Score = W_1 \times \bar{\mathcal{R}} + W_2 \times (1 - \bar{\varphi})$$

选择得分最小的样本进入子集。

### 损失函数 / 训练策略

**自适应权重机制**：$W_1$ 和 $W_2$ 随采样率 $p$ 动态调整：

$$W_1 = \alpha + (1-\alpha) \times \frac{1}{1 + e^{\beta(p - 0.5)}}, \quad W_2 = 1 - W_1$$

设计思路：

- **低采样率**（如 1%、10%）：$W_1$ 较大，优先选择类内代表性强的"简单"样本，有益于模型优化
- **高采样率**（如 50%、70%）：$W_2$ 逐渐增大，引入更多类间困难样本，增强模型对细粒度差异的区分能力
- 关键约束：始终保持 $W_1 > W_2$，即类内评估始终占主导

默认超参数：$\alpha = 0.2$，$\beta = 1$。

**目标模型训练**：SGD 优化器，batch size 128，初始学习率 0.1，Cosine 衰减，momentum 0.9，weight decay $5 \times 10^{-4}$，训练 200 epochs，随机裁剪至 224×224 + 随机水平翻转。

## 实验关键数据

### 主实验

在三个细粒度数据集上，与 12 种基线方法对比，报告各采样率下平均准确率增益（相比 Random 方法）：

| 数据集 | 指标 | RAM-APL | 次优方法 (GC) | 提升 |
|---|---|---|---|---|
| Oxford-IIIT Pet | 平均增益 | **+3.74%** | +1.52% | +2.22% |
| Food-101 | 平均增益 | **+4.44%** | +3.04% | +1.40% |
| CUB-200-2011 | 平均增益 | **+6.40%** | +2.78% | +3.62% |

RAM-APL 在所有数据集的所有采样率下均优于所有基线方法。

### 消融实验

| 配置 | 1% | 50% | 70% | 说明 |
|---|---|---|---|---|
| MIN (Model-TD) | 5.6±0.7 | 40.3±2.6 | 55.2±2.7 | 传统 IE 基线 |
| MIN (CLIP) | 5.6±0.2 | 45.9±1.8 | 56.3±0.7 | 单 FM 作为 IE |
| MIN (DINOv2) | 6.2±0.1 | 46.8±2.0 | 60.5±2.9 | 单 FM，DINOv2 更优 |
| RAM (CLIP+DINOv2) | 5.9±0.3 | 47.1±1.4 | 56.5±2.7 | 仅用 RAM 融合，不劣于任一单模型 |
| **RAM-APL** | **6.5±0.4** | **47.5±1.9** | 58.7±2.2 | 完整方法，1%和50%均最优 |

特征融合策略对比（Pet 数据集）：

| 融合策略 | 1% | 30% | 50% | 70% |
|---|---|---|---|---|
| Concatenate（拼接） | 5.9±0.4 | 31.7±1.3 | 47.7±3.0 | 57.8±1.2 |
| **RAM-APL（排名融合）** | **6.5±0.4** | **32.4±2.9** | 47.5±1.9 | **58.7±2.2** |

### 关键发现

1. **FM 在细粒度数据集上具有决定性优势**：在 Pet 和 Pet-N 上，FM 在 12 种设置中有 9 种是最优 IE；但在 CIFAR-10N 上仅 4/12 最优
2. **FM 下游性能 ≠ 子集选择性能**：EVA-CLIP 在 Pet 上零样本分类最强，但无一种选择方法使其成为最优 IE
3. **多模型优于单模型**：DINOv2 + CLIP 组合在效率和精度间取得最佳平衡
4. **排名空间融合优于特征拼接**：在高采样率下尤为明显
5. **增加 FM 数量并非总是更好**：4 个 FM 全用时性能并不优于 DINOv2 + CLIP 双模型
6. **自适应权重优于等权融合**：$\alpha=0.2, \beta=1$ 的 sigmoid 权重策略优于 $W_1=W_2=1$

## 亮点与洞察

- **排名空间映射是核心创新**：通过将不对齐的高维特征映射到整数排名再取均值，RAM 优雅地解决了多模型特征融合的对齐问题，无需额外的对齐网络或投影层
- **类内 + 类间的双视角度量**：现有方法要么只看类内分布（如几何方法），要么只关注决策边界（如 margin 方法）。RAM-APL 同时考虑两个维度，更全面
- **采样率自适应权重的直觉很好**：低采样率时先选"简单样本"帮助模型收敛，高采样率时逐步引入"困难样本"增强泛化，这与课程学习的思路一致
- **实验设计扎实**：单模型研究覆盖 5 个数据集 × 4 种算法 × 3 种 IE × 3 种采样率 = 180 组实验，结论有充分的实验支撑
- **方法简洁高效**：无需额外训练、无需微调 FM、无需梯度计算，只需特征提取 + 距离计算 + 排名

## 局限与展望

1. **仅验证了图像分类**：方法能否推广到检测、分割等下游任务有待验证
2. **仅限细粒度数据集**：在粗粒度和含噪声的数据集上 FM 优势不明显，RAM-APL 也主要面向细粒度场景
3. **伪标签依赖真实标签**：APL 中判断伪类标签是否正确需要 ground-truth 标签，在无监督/弱监督场景下不适用
4. **FM 组合靠手动选择**：当前使用 DINOv2 + CLIP 作为默认组合，缺乏自动化的 FM 选择/加权机制
5. **权重函数形式固定**：sigmoid 权重函数的 $\alpha, \beta$ 是手动调参，未探索更自适应的策略
6. **类别不平衡处理有限**：方法在类平衡采样下评估，对严重类别不平衡的鲁棒性未充分讨论

## 相关工作与启发

- **Moderate\_DS (Xia et al., 2023)**：选择距离类中心"中等"的样本，是本文 RAM 组件的灵感来源
- **TDDS (Zhang et al., 2024)**：需要 90 epoch 提取训练动态，凸显了传统 IE 高成本的问题
- **Swayamdipta et al., 2020**：数据地图概念——"简单样本有助于优化"，启发了本文自适应权重中先选简单后选困难的设计
- **DeepCore (Guo et al., 2022)**：子集选择的统一实验框架，本文基于其实现基线方法
- **DINOv2 + CLIP 互补性**：DINOv2 擅长视觉特征表示，CLIP 具备语义对齐能力，两者特征空间互补

## 评分

| 维度 | 分数 | 评价 |
|---|---|---|
| 新颖性 | 7/10 | 多 FM 融合做子集选择是新方向，但排名均值和伪标签准确率本身不算复杂 |
| 技术深度 | 7/10 | 方法简洁有效，数学推导清晰；但理论分析偏少，缺乏为何 FM 在细粒度数据上更优的深层解释 |
| 实验充分度 | 9/10 | 单模型研究 + 多模型实验 + 消融 + 参数分析 + 融合策略对比，非常系统 |
| 写作质量 | 8/10 | 结构清晰，观察→动机→方法的故事线连贯 |
| 实用价值 | 7/10 | 方法简单易实现，但适用场景限于细粒度分类 |
| **综合** | **7.5/10** | 扎实的实证工作，方法简洁有效，实验充分 |

<!-- RELATED:START -->

## 相关论文

- [A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints](a_bayesian_model_selection_criterion_for_selecting_pretraining_checkpoints.md)
- [Griffin: Towards a Graph-Centric Relational Database Foundation Model](griffin_towards_a_graph-centric_relational_database_foundation_model.md)
- [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/self_supervised/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications](../../CVPR2026/self_supervised/momo_mars_orbital_model_foundation_model_for_mars_orbital_applications.md)

<!-- RELATED:END -->
