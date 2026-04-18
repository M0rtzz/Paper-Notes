---
title: >-
  [论文解读] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval
description: >-
  [ICCV 2025][Hyperbolic Space] 首次提出在双曲空间中编码用户定义的多层视觉层次结构的学习范式，通过基于角度的 entailment 对比损失在无需显式层次标签的情况下学习 scene→object→part 层次，并引入基于最优传输的层次检索评估指标。
tags:
  - ICCV 2025
  - Hyperbolic Space
  - Visual Hierarchy
  - Entailment Learning
  - Image Retrieval
  - Contrastive Loss
---

# Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval

**会议**: ICCV 2025  
**arXiv**: [2411.17490](https://arxiv.org/abs/2411.17490)  
**代码**: 无  
**领域**: 图像检索 / 表征学习  
**关键词**: Hyperbolic Space, Visual Hierarchy, Entailment Learning, Image Retrieval, Contrastive Loss

## 一句话总结

首次提出在双曲空间中编码用户定义的多层视觉层次结构的学习范式，通过基于角度的 entailment 对比损失在无需显式层次标签的情况下学习 scene→object→part 层次，并引入基于最优传输的层次检索评估指标。

## 研究背景与动机

人类以层次结构组织世界知识，但主流图像理解模型（SimCLR、MoCo、CLIP 等）仅关注**视觉相似性**，无法捕捉层次语义关系。例如：一个城市街景图像包含建筑→摩天大楼→窗户的层次结构，但相似性度量无法区分这种包含关系。

现有层次学习方法的局限：
- 大多需要预定义的**显式层次标签**（如 ImageNet 标签树），成本高且不灵活
- 仅处理单类图像的简单层次（如 HCL），无法建模多物体场景的复杂 part-based 层次
- 对称距离（内积、余弦相似度）无法表达**非对称的包含关系**

双曲空间的优势：其体积随半径**指数增长**，天然适合嵌入树状层次结构。

## 方法详解

### 整体框架

1. 利用 bounding box 标注定义 part-based 图像层次（scene→object→part）
2. 将层次分解为成对 entailment 关系（A→B 表示 A 包含 B）
3. 使用基于角度的 entailment 对比损失在双曲空间中强制执行这些关系
4. 训练时仅使用 pairwise entailment，不使用层次树

### 关键设计

1. **Part-Based Image Hierarchy 定义**:

    - 全场景图像处于层次最高层，包含的 bounding box 构成下层
    - 大 bounding box 若显著包围小 bounding box（≥80% 面积重叠），则构成 entailment 关系
    - **图像内层次**：递归应用包含规则构建树（road scene → cyclist → bicycle → wheels）
    - **跨图像层次**：对每个 bounding box，从其他图像采样 K 个同类 bounding box，建立全场景到跨图同类物体的 entailment
    - 设计动机：仅需 bounding box + 类别标注（无需层次标注），即可自动构建复杂多层层次

2. **Angle-Based Entailment Loss（双曲角度 entailment 损失）**:

    - 给定 entailment 对 (x→y)，在双曲空间中最大化外角 $\beta_1$ 和 $\alpha_2$：
        - $\beta_1(\mathbf{x}, \mathbf{y}) = \pi - \text{ext}(\mathbf{x}, \mathbf{y})$
        - $\alpha_2(\mathbf{y}, \mathbf{x}) = \text{ext}(\mathbf{y}, \mathbf{x})$
    - 外角通过 Lorentz 模型的内积计算：$\text{ext}(\mathbf{x}, \mathbf{y}) = \cos^{-1}\left(\frac{y_{\text{time}} + x_{\text{time}} c\langle\mathbf{x}, \mathbf{y}\rangle_{\mathbb{H}}}{\|\mathbf{x}_{\text{space}}\|\sqrt{(c\langle\mathbf{x}, \mathbf{y}\rangle_{\mathbb{H}})^2 - 1}}\right)$
    - 使用 multi-positive InfoNCE 损失处理一个 parent 对应多个 child 的情况
    - 双向损失：$L_{\text{angle}} = L^{p\to c}(\mathcal{D}, \beta_1) + L^{c\to p}(\mathcal{D}, \alpha_2)$
    - 设计动机：角度度量提供径向轴上的额外自由度，允许嵌入沿树结构自然分布；双向约束比单向更稳定

3. **Hierarchical Retrieval Evaluation（最优传输评估指标）**:

    - 对每个查询图像 $I$，预计算其层次树 $\mathcal{H}_I$ 的标签分布 $\mathbf{h}_I$
    - 计算检索标签分布 $\mathbf{r}_I$ 与 $\mathbf{h}_I$ 的 1-D Wasserstein 距离：$\text{OT}(\mathbf{h}_I, \mathbf{r}_I) = \text{Wasserstein}(\bar{\mathbf{h}}_I, \bar{\mathbf{r}}_I)$
    - 较小距离表示更好的层次对齐
    - 设计动机：标准 Recall@k 忽略类别分布不平衡问题，OT 距离衡量检索结果与真实层次分布的匹配度

### 损失函数 / 训练策略

- 使用 AdamW 优化器，lr=2e-5，(β₁,β₂)=(0.9, 0.999)
- CLIP ViT 有效 batch size=640，MoCo-v2 batch size=1984
- 双曲模型使用可学习曲率参数，嵌入维度 128
- 在 HierOpenImages（基于 OpenImages 构建的层次数据集）上微调
- 温度参数 τ 可学习，初始化 0.07

## 实验关键数据

### 主实验（HierOpenImages 检索）

**Child-to-Parent 同类检索 (CLIP ViT)**:

| 模型 | 指标 | Top-5 | Top-10 | Top-50 | Top-100 |
|---|---|---|---|---|---|
| CLIP | Cos Sim. | 53.04 | 51.69 | 47.87 | 45.79 |
| HCL | Cos Sim. | 55.48 | 54.81 | 52.23 | 50.67 |
| CLIP-euc† | Euc Ang.* | 75.63 | 74.65 | 72.25 | 70.52 |
| **CLIP-hyp†** | **Hyp Ang.*** | **77.28** | **75.91** | **72.85** | **70.94** |

**层次检索评估 (Parent-to-Child, CLIP ViT)**:

| 模型 | Recall@150k↑ | Recall@250k↑ | OT@150k↓ | OT@250k↓ |
|---|---|---|---|---|
| CLIP | 66.63 | 86.57 | 21.31 | 23.79 |
| CLIP-euc† | 76.46 | 91.38 | 15.65 | 21.09 |
| **CLIP-hyp†** | **77.00** | **91.89** | **14.96** | **20.76** |

### 消融实验（跨图 entailment 的效果）

从 Precision-Recall 曲线（图 4）可以看到：
- 仅用图像内 entailment 训练已有显著提升
- 加入跨图 entailment 在中高 recall 区域进一步提升性能，代价是 top-rank 精度略微下降（因跨图关系引入了视觉多样性）

**域外泛化（LVIS 数据集 Child-to-Parent）**:

| 模型 | Top-5 | Top-10 | Top-50 |
|---|---|---|---|
| CLIP | 15.00 | 14.22 | 11.91 |
| CLIP-euc† | 28.45 | 26.76 | 22.65 |
| **CLIP-hyp†** | **28.84** | **27.02** | **22.87** |

**零样本目标检测（VOC/COCO）**:

| 数据集 | CLIP | CLIP-euc† | CLIP-hyp† |
|---|---|---|---|
| VOC | 87.3 | 93.6 | **94.2** |
| COCO | 63.6 | 73.4 | **73.9** |

### 关键发现

- 双曲空间始终优于欧几里得空间，无论 CLIP 还是 MoCo 骨干
- HCL 微调后 Hyp Dist 指标甚至下降，说明其训练方法不适合复杂视觉层次
- 双曲嵌入的径向结构自然形成：高层级概念靠近原点，细粒度/模糊裁剪被推向边界
- 在未见过的 LVIS 和 VOC/COCO 数据集上也有显著提升，展现了学习到的层次表征的泛化能力
- 即使仅使用角度 entailment 损失（不约束距离），模型也能形成良好的径向分层结构

## 亮点与洞察

- **首次在无显式层次标签的情况下编码复杂多层视觉层次**，只需 bounding box + 类别信息
- 成对 entailment 关系足以学习全局树结构——这一发现理论价值高
- OT 评估指标解决了层次检索中标准 Recall 指标的局限性
- 域外泛化实验令人信服，说明学到的层次表征具有通用性
- 可视化展示了嵌入空间的优美结构：harbor → boat parts 的层次清晰可见

## 局限与展望

- 层次定义依赖 bounding box 标注，未标注数据无法使用
- 80% 面积重叠阈值为手动设定，不同数据集可能需要调整
- 仅用图像编码器，未利用 CLIP 的文本编码器（HierOpenImages 不适合文本评估）
- 标签分布高度不平衡（图 8），可能导致对稀有类别的层次学习不充分
- 未探索 3D 场景理解等更复杂的层次结构

## 相关工作与启发

- HCL 是最直接的对比方法，但仅处理 scene-object 两层简单层次
- ACCEPT 提出了基于角度的 entailment 损失，本文将其扩展到 multi-positive 场景
- Poincaré Embeddings 是双曲表征学习的先驱工作
- 本文的核心启发：**层次是非对称关系，需要非对称度量**——角度度量在双曲空间中自然提供这种非对称性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次在图像域实现无标签的多层复杂视觉层次学习
- **实验充分度**: ⭐⭐⭐⭐ — in-domain + out-of-domain 全面评测；更多骨干/数据集可增强说服力
- **写作质量**: ⭐⭐⭐⭐ — 理论与方法铺垫充分，但公式较多可能影响可读性
- **价值**: ⭐⭐⭐⭐ — 开辟了视觉层次学习的新方向，检索评估指标有参考价值
