---
title: >-
  [论文解读] GGBall: Graph Generative Model on Poincaré Ball
description: >-
  [ICLR 2026][图像生成][双曲空间] 提出 GGBall，首个完全基于 Poincaré 球模型的图生成框架，通过双曲向量量化自编码器（HVQVAE）和黎曼流匹配先验，在层次图和分子图生成上达到 SOTA，在层次图数据集上平均生成误差降低 18%。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "双曲空间"
  - "图生成"
  - "Poincaré球模型"
  - "向量量化"
  - "Flow Matching"
---

# GGBall: Graph Generative Model on Poincaré Ball

**会议**: ICLR 2026  
**arXiv**: [2506.07198](https://arxiv.org/abs/2506.07198)  
**代码**: [GitHub](https://github.com/AI4Science-WestlakeU/GGBall)  
**领域**: 图生成 / 双曲几何  
**关键词**: 双曲空间, 图生成, Poincaré球模型, 向量量化, Flow Matching

## 一句话总结
提出 GGBall，首个完全基于 Poincaré 球模型的图生成框架，通过双曲向量量化自编码器（HVQVAE）和黎曼流匹配先验，在层次图和分子图生成上达到 SOTA，在层次图数据集上平均生成误差降低 18%。

## 研究背景与动机
- **领域现状**: 图生成是分子设计、材料发现等领域的核心任务，现有方法（如 DiGress、GDSS）主要在欧氏空间或离散图空间操作
- **现有痛点**: 欧氏隐空间天生不适合捕获图数据中的层次结构和幂律度分布，导致社区结构、父子关系等被扭曲
- **核心矛盾**: 图的组合性、层次性本质与欧氏空间的线性增长体积之间存在几何不匹配
- **切入角度**: 双曲空间的指数体积增长天然适合表示层次结构（Gromov 定理）
- **核心 idea**: 将标准欧氏隐空间生成管线完全转换为双曲空间，统一用节点级隐变量表示图拓扑

## 方法详解

### 整体框架
两阶段生成：(1) 在 Poincaré 球上编码图结构为离散隐空间令牌（HVQVAE）；(2) 用黎曼流匹配建模隐空间先验，生成时从先验采样→量化→解码。GNN 编码局部结构，Transformer 传播全局依赖。

### 关键设计

1. **Poincaré 图神经网络 (Poincaré GNN)**:

    - 功能：在双曲空间中进行消息传递，将边和节点信息编码到节点表示中
    - 核心思路：切空间聚合 + 距离调制消息函数。使用 $\log_0^c(\cdot)$ / $\exp_0^c(\cdot)$ 在切空间聚合后映射回流形
    - 消息调制：$\text{M}(\mathbf{m}_{ij}) = \gamma_{ij} \cdot \mathbf{m}_{ij} + \beta_{ij}$，其中 $\gamma_{ij}, \beta_{ij}$ 是双曲距离 $d_c(\mathbf{h}_i, \mathbf{h}_j)$ 的函数
    - 设计动机：曲率感知的距离调制使模型能直接编码层次关系的强度

2. **Poincaré Diffusion Transformer**:

    - 功能：建模全局图结构，替代点积注意力为测地线距离注意力
    - 测地线注意力：$\alpha_{ij} \propto \exp(-\tau d_c(\mathbf{q}_i, \mathbf{k}_j))$
    - 值聚合用 Möbius gyromidpoint 保持几何一致性
    - 时间调制：注入时间步嵌入用于流匹配先验

3. **双曲向量量化自编码器 (HVQVAE)**:

    - 功能：将连续双曲嵌入离散化到可学习的 Poincaré 码本 $\mathcal{C}$
    - 核心思路：用测地线距离最近邻量化 $\mathbf{z}_q = \arg\min_{\mathbf{c}_j} d_c(\mathbf{z}, \mathbf{c}_j)$
    - 码本初始化用双曲 k-means 聚类，黎曼优化器更新
    - 稳定机制：过期阈值替换不活跃码本条目，加权 Einstein 中点更新

### 损失函数 / 训练策略
- 自编码器损失：重建损失 + 度-边一致性损失 + L2 正则
- HVQVAE 损失：$\mathcal{L}_{\text{HVQVAE}} = \lambda_1 \mathcal{L}_{\text{AE}} + \lambda_2 \mathbb{E}[d_c^2(\text{sg}(\mathbf{z}_q), \mathbf{z})] + \lambda_3 \mathbb{E}[d_c^2(\mathbf{z}_q, \text{sg}(\mathbf{z}))]$
- 流匹配先验：黎曼条件流匹配目标，测地线插值路径

## 实验关键数据

### 主实验：抽象图生成

| 方法 | Community-small Avg↓ | Ego-small Avg↓ | 空间 |
|------|---------------------|----------------|------|
| GraphVAE | 0.6233 | 0.1167 | 欧氏 |
| GDSS | 0.0460 | 0.0173 | 图空间 |
| DiGress | 0.0380 | - | 图空间 |
| HGDM | 0.0240 | 0.0137 | 混合 |
| **GGBall** | **0.0197** | **0.0117** | 双曲 |

### 分子图生成 (QM9)

| 方法 | Validity↑ | Uniqueness↑ | Novelty↑ | V.U.N↑ |
|------|-----------|-------------|----------|--------|
| DiGress | 99.00 | 96.34 | 32.51 | 31.04 |
| CatFlow | 98.47 | 97.58 | 65.62 | 63.02 |
| **GGBall** | **98.33** | **96.38** | **93.77** | **88.45** |

### 关键发现
- HVQVAE 相比 HAE 显著提升化学有效性（95.18→99.14%）和边精度
- 双曲编码器在度分布保持上比欧氏基线低 4× MMD 误差
- 新颖性 93.77% 远超 DiGress 的 32.51%，表明双曲空间的表达力更强

## 亮点与洞察
- 首个完全在 Poincaré 球上的图生成框架，统一了编码-量化-先验-解码全流程
- 用节点级隐变量统一表示图拓扑，将边连接视为隐空间几何的涌现属性
- 在 QM9 上实现最高 V.U.N 分数（新颖性+唯一性+有效性的综合度量）

## 局限与展望
- 自回归先验在初步实验中不如 FM，值得深入探索
- 分子图上有效性略低于 DiGress（98.33 vs 99.00），精细化学约束的融合有待加强
- 计算开销：双曲空间操作比欧氏操作更复杂

## 相关工作与启发
- **vs HGDM**: HGDM 仅在 Poincaré 嵌入上做边去噪，仍在离散图空间操作；GGBall 完全在双曲隐空间
- **vs DiGress**: DiGress 在图空间迭代去噪，受限于欧氏几何；GGBall 利用双曲变量解耦层次结构

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个全双曲图生成框架，从 GNN 到 Transformer 到 VQ 到 FM 全部双曲化
- 实验充分度: ⭐⭐⭐⭐ 覆盖抽象图和分子图，消融充分，但缺少大规模图数据
- 写作质量: ⭐⭐⭐⭐ 数学严谨，流程清晰
- 价值: ⭐⭐⭐⭐ 为非欧几何在生成模型中的应用开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)
- [\[ICLR 2026\] Verification of the Implicit World Model in a Generative Model via Adversarial Sequences](verification_of_the_implicit_world_model_in_a_generative_model_via_adversarial_s.md)
- [\[ICLR 2026\] HOG-Diff: Higher-Order Guided Diffusion for Graph Generation](hog-diff_higher-order_guided_diffusion_for_graph_generation.md)
- [\[ICLR 2026\] GenCP: Towards Generative Modeling Paradigm of Coupled Physics](gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)
- [\[ICLR 2026\] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training](generate_any_scene_scene_graph_driven_data_synthesis_for_visual_generation_train.md)

</div>

<!-- RELATED:END -->
