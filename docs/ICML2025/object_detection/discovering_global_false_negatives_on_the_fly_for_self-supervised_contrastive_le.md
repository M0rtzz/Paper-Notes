---
title: >-
  [论文解读] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning
description: >-
  [ICML2025][目标检测][对比学习] 提出 GloFND，通过为每个锚点样本学习动态阈值，在训练过程中实时发现并过滤全局假阴性（false negatives），以低额外开销提升对比学习表示质量。
tags:
  - ICML2025
  - 目标检测
  - 对比学习
  - 假阴性发现
  - 自监督表示学习
  - 全局阈值优化
  - SogCLR
---

# Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning

**会议**: ICML2025  
**arXiv**: [2502.20612](https://arxiv.org/abs/2502.20612)  
**代码**: [vibalcam/GloFND](https://github.com/vibalcam/GloFND)  
**领域**: 对比学习 / 自监督学习  
**关键词**: 对比学习, 假阴性发现, 自监督表示学习, 全局阈值优化, SogCLR

## 一句话总结

提出 GloFND，通过为每个锚点样本学习动态阈值，在训练过程中实时发现并过滤全局假阴性（false negatives），以低额外开销提升对比学习表示质量。

## 研究背景与动机

自监督对比学习（SimCLR、MoCo、SogCLR 等）中，正样本对由同一图像的不同增强视角构成，负样本对则从数据集中随机采样（排除锚点）。然而，这种随机采样会将语义相似的样本错误地标记为负样本，即**假阴性（false negatives, FN）**。

**假阴性的危害**：以 ImageNet100 上 SogCLR 预训练为例，约 1% 的负对是假阴性——batch size=1024 时每批约 20,000 个、batch size=128 时约 325 个。这些假阴性会迫使编码器丢弃关键语义信息，在半监督设置下线性分类器准确率最多**下降 10%**。

**已有方法的局限**：

- **局部方法**（WCL, FNC）：仅在 mini-batch 内寻找假阴性，小 batch 时 top-k 相似样本不可靠
- **全局方法**（IFND）：在特定 epoch 对全数据集做 k-means 聚类，大规模数据集上计算代价过高

→ **核心动机**：需要一种全局（dataset-wise）、动态（on-the-fly）、且与 batch 大小无关的假阴性发现方法。

## 方法详解

### 核心思想

GloFND 为每个锚点样本 $\mathbf{x}_i$ 学习一个**动态阈值** $\lambda_i$，用于筛选出与该锚点相似度最高的 top-$\alpha$% 负样本作为假阴性并过滤掉。

### 阈值学习——凸优化问题

将寻找 $(1-\alpha)$-分位数的问题建模为如下凸优化：

$$\lambda_i = \arg\min_{\nu \in [-1,1]} \nu\alpha + \frac{1}{|R_i|}\sum_{r \in R_i}(r - \nu)_+$$

其中 $R_i = \{\text{sim}(E_\mathbf{w}(\mathbf{x}_i), E_\mathbf{w}(\mathbf{x})) \mid \mathbf{x} \in \mathcal{S}_i^-\}$ 是锚点与所有负样本的余弦相似度集合。

**Lemma 3.1**：解 $\lambda_i$ 恰好是 $R_i$ 中第 $k = \lceil \alpha|\mathcal{S}_i^-| \rceil$ 大的值（或处于第 $k$ 与第 $k+1$ 大之间），从而精确地选出 top-$\alpha$% 相似负样本。

### 阈值 SGD 更新

每个迭代中，对 mini-batch 内的锚点 $\mathbf{x}_i$ 计算随机次梯度：

$$\hat{\nabla}_{\lambda_i} = \alpha - \frac{1}{|\mathcal{B}_i^-|}\sum_{\mathbf{x} \in \mathcal{B}_i^-} \mathbb{I}(\text{sim}(\mathbf{z}_i, E_\mathbf{w}(\mathbf{x})) > \lambda_i)$$

然后投影更新：

$$\lambda_i \leftarrow \Pi_{[-1,1]}[\lambda_i - \theta \hat{\nabla}_{\lambda_i}]$$

其中 $\theta$ 为阈值学习率。非 batch 内的 $\lambda_j$ 保持不变。

### 修改的对比损失

去除假阴性后的全局对比损失：

$$\ell(\mathbf{w}, \lambda_i; \mathbf{x}_i) = -\text{sim}(\mathbf{z}_i, \mathbf{z}_i') + \tau \log(|\tilde{\mathcal{S}}_i^-| \cdot g(\mathbf{w}, \lambda_i; \mathbf{x}_i, \tilde{\mathcal{S}}_i^-))$$

其中 $\tilde{\mathcal{S}}_i^- = \{\mathbf{x} \mid \mathbf{x} \in \mathcal{S}_i^-, \text{sim}(\mathbf{z}_i, E_\mathbf{w}(\mathbf{x})) \leq \lambda_i\}$ 是过滤后的"干净"负样本集。

### 移动平均估计器（继承 SogCLR）

为避免大 batch 需求，沿用 SogCLR 的移动平均策略估计 $g$：

$$u_i \leftarrow (1-\gamma)u_i + \gamma \hat{g}(\mathbf{w}, \lambda_i; \mathbf{x}_i, \tilde{\mathcal{B}}_i^-)$$

编码器梯度估计：

$$\hat{\nabla}_\mathbf{w} = \frac{1}{|\mathcal{B}|}\sum_{\mathbf{x}_i \in \mathcal{B}} -\nabla_\mathbf{w}\text{sim}(\mathbf{z}_i, \mathbf{z}_i') + \frac{\tau \nabla_\mathbf{w}\hat{g}(\mathbf{w}, \lambda_i; \mathbf{x}_i, \tilde{\mathcal{B}}_i^-)}{u_i}$$

### 双模态扩展

GloFND 可直接扩展到 CLIP 式的图像-文本对比学习：为每个实例维护两个阈值 $\lambda_{I,i}$（图像锚点）和 $\lambda_{T,i}$（文本锚点），分别对两个模态的负样本做假阴性过滤。

### 算法流程（SogCLR + GloFND）

1. 初始化编码器 $\mathbf{w}$、移动平均 $\mathbf{u}$、阈值 $\boldsymbol{\lambda}$
2. 每次迭代：采样 batch $\mathcal{B}$，对每个锚点 $\mathbf{x}_i \in \mathcal{B}$
    - SGD 更新 $\lambda_i$（阈值步）
    - 用 $\lambda_i$ 过滤假阴性，得到 $\tilde{\mathcal{B}}_i^-$
    - 更新移动平均 $u_i$
3. 计算梯度 $\hat{\nabla}_\mathbf{w}$，用 Adam/SGD 更新编码器
4. 额外开销仅 $O(B^2)$，远小于前向/反向传播

## 实验关键数据

### 单模态对比学习

| 方法 | 数据集 | 设置 | 关键结果 |
|------|--------|------|----------|
| SogCLR + GloFND | ImageNet100 | 半监督 1% 标注 | 准确率提升可达 **~10%** vs. 不处理 FN |
| SogCLR + GloFND | ImageNet100 | 线性评估 | 一致优于 SogCLR baseline |
| SimCLR + GloFND | 多数据集 | 线性评估 | 兼容多种 CL 方法，均有提升 |

### 双模态对比学习（CLIP 式）

- 在图像-文本数据上集成 GloFND，检索和分类性能均有改善

### 关键观察

- t-SNE 可视化显示：GloFND 过滤假阴性后，不同类别的表示簇更紧凑、分离度更高
- α 是关键超参：控制假阴性比例，可适配粗粒度（车 vs 动物）和细粒度（犬种分类）任务
- 计算开销极低：GloFND 仅需 $O(B^2)$ 额外计算，相对前向/反向传播可忽略

## 亮点与洞察

1. **优化视角解决分位数估计**：将 top-α% 筛选转化为凸优化问题（Ogryczak & Tamir, 2003），避免了全数据集排序/聚类的高开销
2. **全局发现、局部计算**：阈值 $\lambda_i$ 隐式地反映了锚点在全数据集中的相似度分布，但每次更新只需 mini-batch 内的样本
3. **即插即用**：GloFND 与具体的对比损失解耦——可集成到 SogCLR、SimCLR、CLIP 等多种框架，仅需添加阈值更新步骤
4. **灵活应对假阴性**：虽然本文仅探索了"过滤"策略，但 GloFND 也支持将假阴性作为额外正对（FN attraction），为后续研究留下空间
5. **理论保障**：Lemma 3.1 证明优化问题的解精确对应 top-k 选择

## 局限与展望

1. **假阴性利用方式单一**：本文仅验证了"过滤"假阴性的策略，没有深入探索将假阴性转为正对（attraction）是否更优
2. **超参 α 需要调节**：α 依赖先验知识或下游任务粒度，缺乏自适应机制
3. **需要 warm-up 阶段**：GloFND 依赖"足够好"的初始表示来启动假阴性发现，cold start 下阈值可能不准
4. **忽略了超梯度**：将 $\lambda$ 关于 $\mathbf{w}$ 的超梯度直接置零（类似 MAML 一阶近似），理论上非最优
5. **缺少大规模实验**：主要在 ImageNet100 等中等规模数据集上验证，尚未展示在 ImageNet-1K/21K 等大规模设置下的表现

## 相关工作与启发

- **SogCLR**（Yuan et al., 2022）：随机优化的全局对比学习，GloFND 的基础框架
- **FNC**（Huynh et al., 2022）：batch 内 top-k 假阴性发现，局部方法的代表
- **IFND**（Chen et al., 2022）：全数据集聚类的全局方法，但计算量大
- **iSogCLR**（Qiu et al., 2023）：学习个体化温度应对假阴性，互补视角
- **SupCon**（Khosla et al., 2021）：用真实标签避免假阴性，上界参考
- **CVaR / 分位数优化**（Ogryczak & Tamir, 2003）：GloFND 阈值学习的数学工具来源

## 评分

- 新颖性: ⭐⭐⭐⭐ — 优化视角的全局假阴性发现是新颖且优雅的思路
- 实验充分度: ⭐⭐⭐ — 覆盖单模态/双模态/半监督，但缺少大规模数据集和消融
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，算法描述规范
- 价值: ⭐⭐⭐⭐ — 即插即用的假阴性过滤模块，实用性强

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment](../../CVPR2026/object_detection/falcon_false-negative_aware_learning_of_contrastive_negatives_in_vision-language.md)
- [\[ICML 2025\] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics](global_context-aware_representation_learning_for_spatially_resolved_transcriptom.md)
- [\[NeurIPS 2025\] The Complexity of Finding Local Optima in Contrastive Learning](../../NeurIPS2025/object_detection/the_complexity_of_finding_local_optima_in_contrastive_learning.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/object_detection/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[ICML 2025\] Self-Organizing Visual Prototypes for Non-Parametric Representation Learning](self-organizing_visual_prototypes_for_non-parametric_representation_learning.md)

<!-- RELATED:END -->
