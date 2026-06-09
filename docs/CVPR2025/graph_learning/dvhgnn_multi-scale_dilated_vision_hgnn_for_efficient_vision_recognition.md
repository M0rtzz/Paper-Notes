---
title: >-
  [论文解读] DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition
description: >-
  [CVPR 2025][图学习][超图神经网络] 提出 DVHGNN，一种利用多尺度膨胀超图捕获图像 patch 间高阶相关性的视觉骨干网络，通过聚类+膨胀超图构造 (DHGC) 获取多尺度超边、动态超图卷积实现自适应特征交换…
tags:
  - "CVPR 2025"
  - "图学习"
  - "超图神经网络"
  - "多尺度膨胀超图"
  - "视觉骨干"
  - "动态超图卷积"
  - "高阶相关性"
---

# DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition

**会议**: CVPR 2025  
**arXiv**: [2503.14867](https://arxiv.org/abs/2503.14867)  
**代码**: 无（论文未提供）  
**领域**: 图学习 / 视觉骨干网络  
**关键词**: 超图神经网络, 多尺度膨胀超图, 视觉骨干, 动态超图卷积, 高阶相关性

## 一句话总结
提出 DVHGNN，一种利用多尺度膨胀超图捕获图像 patch 间高阶相关性的视觉骨干网络，通过聚类+膨胀超图构造 (DHGC) 获取多尺度超边、动态超图卷积实现自适应特征交换，在 ImageNet-1K 上以 30.2M 参数达到 83.1% top-1 准确率，超越 ViG-S 1.0% 和 ViHGNN-S 0.6%。

## 研究背景与动机

**领域现状**：Vision GNN (ViG) 开创性地将图像视为图结构并用 GNN 处理，但面临两大问题。同时，ViHGNN 尝试用超图捕获高阶关系但改进有限。

**现有痛点**：(1) ViG 使用 KNN 图构造，计算复杂度为二次方且不可学习，可能丢失重要信息；(2) 普通图只能建模成对关系，无法捕获多个节点间的高阶相关性；(3) ViHGNN 用模糊 C 均值聚类构建超图，缺乏多尺度信息且不能动态适应学习过程。

**核心矛盾**：需要捕获高阶关系（超图）但现有超图构造方法要么忽略多尺度要么计算复杂度过高。

**本文目标** 如何高效构建能捕获多尺度高阶关系的超图视觉表示。

**切入角度**：结合聚类（捕获全局语义分组）和膨胀超图构造（捕获多尺度局部空间关系）的双路超图表示。

**核心 idea**：用聚类+不同膨胀率的局部超边构建多尺度超图，通过余弦相似度和稀疏感知权重实现动态超图卷积。

## 方法详解

### 整体框架
图像分为 N 个 patch 作为顶点。每个 block 包含：多尺度超图构造（聚类超边 + 膨胀超边）→ 两阶段动态超图卷积（顶点卷积聚合到超边 → 超边卷积分发回顶点）→ ConvFFN 增强特征变换。采用类似 ViG 的层级 isotropic 结构。

### 关键设计

1. **聚类 + 膨胀超图构造 (DHGC)**:

    - 功能：自适应获取多尺度超边集合
    - 核心思路：双路设计。聚类路径：将 patch 特征映射到相似性空间，通过余弦相似度分配到 C 个聚类中心形成语义级超边 $\mathcal{E}_c$。膨胀路径：对每个 $w \times w$ 窗口的中心顶点 $v_c$，以膨胀率 $r=1,2,3$ 构建局部超边（对应 3×3, 5×5, 7×7 感受野），每个膨胀超边有可学习的稀疏感知权重 $w_r$。引入区域分割（类似 Swin Transformer 的窗口）将复杂度从 $O(NCD)$ 降低到 $O(NCD/m)$
    - 设计动机：聚类超边捕获全局语义相似性但忽略空间局部性，膨胀超边捕获多尺度空间关系但范围有限，两者互补

2. **动态超图卷积 (DHConv)**:

    - 功能：自适应特征交换和融合
    - 核心思路：两阶段消息传递。顶点卷积阶段：对聚类超边，用余弦相似度的 sigmoid 加权聚合顶点特征到超边（$h_e = \frac{1}{C}(h_c + \sum \text{sig}(\alpha s_i + \beta) x_i)$）；对膨胀超边，用可学习稀疏权重 $w_r$ 加权聚合。超边卷积阶段：将超边特征通过余弦相似度或稀疏权重分发回顶点，用 GIN 风格的更新 $x'_i = FC(\sigma(\text{Conv}((1+\varepsilon)x_i + z_i)))$
    - 设计动机：不同类型超边需要不同的聚合策略——语义超边适合基于相似度的软分配，空间超边适合基于权重的固定分配

3. **ConvFFN + 多头机制**:

    - 功能：增强特征变换能力并缓解过平滑
    - 核心思路：在超图卷积后接 ConvFFN（类似 ViG 的前馈网络但带卷积），增加局部感受野。多头机制将特征分组独立做超图卷积然后拼接，增加表达多样性
    - 设计动机：纯超图卷积可能导致节点表征趋于一致（过平滑），ConvFFN 引入非线性和局部信息缓解此问题

### 损失函数 / 训练策略
标准 ImageNet 分类训练（交叉熵损失 + 标签平滑 + mixup 等常规增强）。

## 实验关键数据

### 主实验

| 模型 | 类型 | 参数 (M) | FLOPs (G) | Top-1 Acc |
|------|------|---------|----------|----------|
| DeiT-S | ViT | 22.1 | 4.6 | 79.8% |
| Swin-S | ViT | 50.0 | 8.7 | 83.0% |
| ViG-S | GNN | 27.3 | 4.6 | 82.1% |
| ViHGNN-S | HGNN | 28.5 | 6.3 | 82.5% |
| **DVHGNN-S** | **HGNN** | **30.2** | **5.2** | **83.1%** |
| **DVHGNN-B** | **HGNN** | **92.8** | **16.8** | **84.2%** |

下游任务：COCO 目标检测/分割、ADE20K 语义分割也有一致提升。

### 消融实验

| 配置 | Top-1 Acc |
|------|----------|
| 仅聚类超边 | 82.3% |
| 仅膨胀超边 | 82.5% |
| 聚类+膨胀（完整） | **83.1%** |
| 无动态权重（固定均匀聚合） | 82.7% |
| 固定窗口分割 vs 无分割 | 83.1% vs OOM |

### 关键发现
- 双路超图（聚类+膨胀）比任一单路都好（83.1% vs 82.3%/82.5%），证明语义和空间信息互补
- 动态超图卷积（基于余弦相似度）比固定聚合高 0.4%，说明自适应权重有效
- DVHGNN-S 相比 ViG-S 多 3M 参数但 FLOPs 更高效（5.2G vs 4.6G），精度高 1.0%
- 区域分割策略在保持性能的同时显著降低内存消耗

## 亮点与洞察
- **膨胀超图类比膨胀卷积**：将膨胀卷积的多尺度思想引入超图构造是一个直觉清晰且有效的设计，不同膨胀率对应不同尺度的空间关系
- **双路互补设计**：聚类超边（全局语义）+ 膨胀超边（局部空间）的组合类似于 SlowFast 网络的双路思想，但在超图框架下实现更自然
- **超图 GNN 作为通用视觉骨干的可行性**：在 ImageNet 上超越 Swin-S 同等精度区间，证明超图范式有潜力

## 局限与展望
- 参数量和 FLOPs 仍高于 ViG-S，效率有优化空间
- 聚类中心数 C 和膨胀率 R 为人工设定的超参数
- 未与最新的 Mamba 等线性复杂度架构对比
- 超图构造在每层独立进行，缺乏跨层的超图结构传递

## 相关工作与启发
- **vs ViG**: ViG 用 KNN 构建普通图（二次复杂度、成对关系），DVHGNN 用聚类+膨胀构建超图（线性复杂度、高阶关系）
- **vs ViHGNN**: ViHGNN 用模糊 C 均值（全局、非自适应），DVHGNN 用余弦相似度聚类+膨胀超图（多尺度、自适应）
- **vs Swin Transformer**: Swin 在窗口内做自注意力，DVHGNN 在窗口内做超图卷积，后者能捕获高阶关系

## 评分
- 新颖性: ⭐⭐⭐⭐ 膨胀超图构造+双路设计有创新性
- 实验充分度: ⭐⭐⭐⭐ ImageNet + COCO + ADE20K 全覆盖
- 写作质量: ⭐⭐⭐ 内容详实但结构稍显冗长
- 价值: ⭐⭐⭐⭐ 推动了超图视觉架构的发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Hypergraph Vision Transformers: Images are More than Nodes, More than Edges](hypergraph_vision_transformers_images_are_more_than_nodes_more_than_edges.md)
- [\[NeurIPS 2025\] The Underappreciated Power of Vision Models for Graph Structural Understanding](../../NeurIPS2025/graph_learning/the_underappreciated_power_of_vision_models_for_graph_structural_understanding.md)
- [\[ICML 2025\] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](../../ICML2025/graph_learning/open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)
- [\[ACL 2025\] M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations](../../ACL2025/graph_learning/m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)
- [\[ECCV 2024\] GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition](../../ECCV2024/graph_learning/gkgnet_group_k-nearest_neighbor_based_graph_convolutional_network_for_multi-labe.md)

</div>

<!-- RELATED:END -->
