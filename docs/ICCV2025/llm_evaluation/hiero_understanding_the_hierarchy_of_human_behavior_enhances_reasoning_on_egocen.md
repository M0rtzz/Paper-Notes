---
title: >-
  [论文解读] HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos
description: >-
  [ICCV 2025][LLM评测][第一人称视频理解] 提出 HiERO，一种弱监督的层次化图架构，通过对齐视频片段与叙述文本来学习功能性活动线索的层次结构，使视频片段特征编码多尺度的行为依赖关系，在程序学习任务的零样本评估中大幅超越全监督方法（EgoProceL 上 F1 提升 +12.5%）…
tags:
  - "ICCV 2025"
  - "LLM评测"
  - "第一人称视频理解"
  - "层次化行为建模"
  - "图神经网络"
  - "程序学习"
  - "零样本推理"
---

# HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos

**会议**: ICCV 2025  
**arXiv**: [2505.12911](https://arxiv.org/abs/2505.12911)  
**代码**: [github.com/sapeirone/HiERO](https://github.com/sapeirone/HiERO)  
**领域**: LLM评测  
**关键词**: 第一人称视频理解, 层次化行为建模, 图神经网络, 程序学习, 零样本推理

## 一句话总结

提出 HiERO，一种弱监督的层次化图架构，通过对齐视频片段与叙述文本来学习功能性活动线索的层次结构，使视频片段特征编码多尺度的行为依赖关系，在程序学习任务的零样本评估中大幅超越全监督方法（EgoProceL 上 F1 提升 +12.5%），在视频-文本对齐基准上也达到了 SOTA。

## 研究背景与动机

### 问题定义

人类活动具有天然的层次结构：低层的单个动作（如切洋葱、削胡萝卜）可以聚合为中层的活动线索（如准备蔬菜），而多个活动线索又构成高层的日常程序（如准备晚餐）。现有的第一人称视频理解方法大多聚焦于动作级别的理解，忽略了这种内在的层次性。

### 已有方法的不足

**动作级视频-语言对齐**：EgoVLP 等方法对齐短视频片段与单个叙述文本，但忽略了动作间的时间上下文和功能依赖关系

**程序学习方法的局限**：
   - 有监督方法需要大量逐帧的步骤标注，标注代价极高
   - 自监督方法依赖同一任务的多个视频示例进行跨视频对齐，假设过强，不适用于非脚本化视频
   - 弱监督方法依赖 wikiHow 等外部知识图谱提供显式的程序监督

**单层聚合**：现有程序学习方法只考虑一个层级的聚合（动作→关键步骤），未捕捉多层次抽象

### 核心动机

**关键洞察**：
1. 功能性的行为线索可以从非脚本化视频中自然浮现，无需显式监督——经常共同出现的动作可以通过特征聚类自然聚合为高层活动
2. 不同的特征提取器决定了聚类能捕捉的抽象层级：视觉相似性 → 语义相似性 → 功能相似性的递进
3. 用层次化图表示编码多尺度的功能依赖关系，可以为多种视频理解任务提供强归纳偏置

## 方法详解

### 整体框架

HiERO 采用受 Graph U-Net 启发的编码器-解码器架构：
1. **时间编码器**：在局部时间邻域内逐层聚合信息，构建多尺度时间表示
2. **功能感知解码器**：通过谱图聚类发现功能相似的节点区域，在每个分区内独立进行时间推理
3. **训练目标**：视频-叙述对齐损失 + 功能线索损失

### 关键设计

#### 1. **视频图表示与时间编码器**

- **功能**：将视频表示为图，在局部时间邻域内逐层聚合信息
- **核心思路**：输入视频 $\mathcal{V}$ 被编码为图 $\mathcal{G} = (\mathbf{X}, \mathcal{E}, \mathbf{p})$，其中：
    - $\mathbf{X} \in \mathbb{R}^{N \times D}$：节点嵌入矩阵，每个节点对应固定长度的视频片段
    - $e_{ij} \in \mathcal{E}$：当两个节点时间距离小于阈值 $\tau$ 时连边
    - $\mathbf{p} \in \mathbb{R}^N$：节点的时间位置（时间戳，单位秒）

  编码器由 $N_l$ 个基于 TDGC 层的模块组成，实现时间感知的图卷积：

  $$\mathbf{x}_i^{l+1} = \mathbf{W}_r^T \mathbf{x}_i^l + \text{mean}_{j \in \mathcal{N}(i)} (s_{ij} (\mathbf{w}_{ij} \odot \mathbf{x}_j')) + \mathbf{b}_r$$

  其中 $s_{ij} = \text{sign}(p_i^l - p_j^l)$ 编码时间方向，$\mathbf{w}_{ij} = \text{MLP}(|p_i^l - p_j^l|)$ 根据时间距离调整权重。每层后对节点做时间子采样，使分辨率减半。

- **设计动机**：时间编码器逐渐扩展每个节点的时间上下文，无论相关动作是否在时间上连续。这为后续的功能线索发现提供了多尺度的时间特征。

#### 2. **功能感知解码器与 Cut & Match 模块**

- **功能**：发现并利用功能相似但可能时间上远隔的节点之间的关系
- **核心思路**：
  1. 接收编码器对应层的特征（通过跳连接）和上一层解码器的输出
  2. 使用 **Cut & Match 模块** 进行谱图聚类：
     - 构建基于节点嵌入余弦相似度的相似度矩阵：$S_{ij} = \exp(\mathbf{x}_i^T \mathbf{x}_j / (\kappa \|\mathbf{x}_i\|_2 \|\mathbf{x}_j\|_2))$
     - 计算归一化拉普拉斯矩阵：$\tilde{\mathbf{L}}_S = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{S} \mathbf{D}^{-1/2}$
     - 对 $\tilde{\mathbf{L}}_S$ 做特征分解，取前 $K$ 个最小特征值对应的特征向量
     - 在特征向量子空间中用 K-Means 聚类，将节点分为 $K$ 个簇
  3. 在每个簇内独立执行 TDGC 时间推理，然后映射回原始图

- **设计动机**：谱图聚类不需要数据分布假设，直接通过图的连通性分组。在每个功能分区内做时间推理，使得时间上远隔但功能相关的动作（如交叉进行的活动）能够交换信息。

#### 3. **训练目标：视频-叙述对齐与功能线索损失**

- **功能**：学习一个特征空间，使共同出现的动作相近，不同活动线索的动作远离

**视频-叙述对齐损失 $\mathcal{L}_{vna}$**：
- 与 EgoVLP 不同，不是对齐单个片段与单个叙述，而是将每个节点与其时间窗口内所有叙述对齐（正样本），同时远离窗口外和其他视频的叙述（负样本）
- 正样本集：$\mathcal{P}_j = \{(n,t) \in \mathcal{T}_i : |p - t| \leq 2^\alpha\}$
- 负样本集：$\mathcal{N}_j = \{(n,t) \in \mathcal{T}_i : 2^\alpha < |p - t| \leq 2^\beta\} \cup \mathcal{T}_{k, k \neq i}$

$$\mathcal{L}_{v2t} = \frac{1}{B} \sum_{\mathbf{v}_j} \frac{\sum_{n \in \mathcal{P}_j} \exp(h_v(\mathbf{v}_j)^T h_t(\mathcal{F}(n)) / \tau)}{\sum_{n \in \mathcal{P}_j \cup \mathcal{N}_j} \exp(h_v(\mathbf{v}_j)^T h_t(\mathcal{F}(n)) / \tau)}$$

**功能线索损失 $\mathcal{L}_{ft}$**：
- 利用解码器中 Cut & Match 模块的聚类分配，推动同一簇中的节点在特征空间中更近

$$\mathcal{L}_{ft} = \sum_{k=1}^{K} \sum_{c_i=k} \sum_{c_j=c_i} \frac{\exp(h_v(\mathbf{v}_i)^T h_v(\mathbf{v}_j) / \tau)}{\sum_{j'} \exp(h_v(\mathbf{v}_i)^T h_v(\mathbf{v}_{j'}) / \tau)}$$

总损失：$\mathcal{L} = \mathcal{L}_{vna} + \mathcal{L}_{ft}$

### 损失函数 / 训练策略

- 在 EgoClip（3.8M 片段-文本对）上训练 15 个 epochs
- 批次大小 8，学习率 $1 \times 10^{-5}$，前 5 个 epochs 线性预热 + 余弦退火
- 训练不到 20 GPU 小时
- 支持多种骨干网络：Omnivore、EgoVLP、LaViLa
- 零样本推理：通过在解码器输出上做聚类直接检测程序步骤

## 实验关键数据

### 主实验

**EgoProceL 程序学习基准（零样本 vs 全监督）**：

| 方法 | 监督类型 | 平均 F1 | 平均 IoU |
|------|---------|---------|---------|
| CnC | 自监督（跨视频对齐） | 22.0 | 10.7 |
| GPL | 自监督 | 25.6 | 13.9 |
| OPEL | 自监督 | 32.0 | 16.3 |
| EgoVLP 特征聚类 | 零样本 | 40.0 | 21.9 |
| **HiERO (EgoVLP)** | **零样本** | **44.5** | **25.3** |

**EgoMCQ 视频-文本对齐**：

| 方法 | Inter Acc(%) | Intra Acc(%) |
|------|-------------|-------------|
| EgoVLP | 90.6 | 57.2 |
| LaViLa | 94.5 | 63.1 |
| **HiERO (LaViLa)** | **94.6** | **64.4** |

**Ego4D Goal-Step 步骤定位（mAP 平均）**：

| 方法 | 监督类型 | Avg mAP |
|------|---------|---------|
| EgoOnly | 有监督 | 13.6 |
| EgoVLP | 零样本 | 8.3 |
| **HiERO (EgoVLP)** | **零样本** | **8.7** |

### 消融实验

**HiERO 各组件贡献（EgoVLP 骨干）**：

| 对齐损失 | 功能线索聚类 | 功能线索损失 | EgoMCQ Intra | EgoProceL F1 | Step-Grounding R@1 |
|---------|-----------|-----------|-------------|-------------|-------------------|
| ✗ | ✗ | ✗ | 57.2 | 40.0 | 10.73 |
| ✓ | ✗ | ✗ | 59.5 | 43.8 | 11.27 |
| ✓ | ✓ | ✗ | 59.6 | 43.3 | 11.44 |
| ✓ | ✓ | ✓ | **59.6** | **44.5** | **11.57** |

### 关键发现

1. **零样本大幅超越全监督**：在 EgoProceL 上，零样本的 HiERO 以 44.5% F1 大幅超越最佳自监督方法 OPEL 的 32.0%（+12.5%），后者还需要多个同一任务的视频示例
2. **功能模式自然浮现**：程序步骤可以通过无监督聚类从特征空间中自然浮现，无需步骤标注
3. **窗口级对齐优于单叙述对齐**：$\mathcal{L}_{vna}$ 的窗口对齐策略使特征更具上下文感知能力，显著提升了程序学习 F1（+3.8%）和 EgoMCQ Intra（+2.3%）
4. **功能线索损失引导更好的聚类**：$\mathcal{L}_{ft}$ 鼓励同簇节点特征相近，使聚类更精准地捕捉高层活动模式
5. **跨骨干网络一致提升**：在 Omnivore、EgoVLP、LaViLa 三种骨干上均取得提升，证明 HiERO 的通用性

## 亮点与洞察

1. **层次结构无需显式监督即可浮现**：这是本文最核心的洞察——通过图聚类在不同尺度上发现功能线索，无需任何程序级标注
2. **从短期时间对齐到长程功能推理**：编码器负责局部时间推理，解码器通过 Cut & Match 连接时间远隔但功能相关的片段，实现了真正的长程推理
3. **一个模型多种任务**：在视频-文本对齐（EgoMCQ/EgoNLQ）、程序学习（EgoProceL）、步骤定位（Goal-Step）等多个不同任务上均取得优异性能
4. **谱图聚类的优雅应用**：将视频理解建模为图上的分区问题，利用谱图理论的数学基础提供了强归纳偏置
5. **训练极其轻量**：仅需不到 20 GPU 小时即可训练完成

## 局限与展望

1. **聚类数 K 需要预设**：当前需要指定每层的聚类数，自适应确定 K 值可进一步提升实用性
2. **子采样策略固定**：编码器的时间子采样率固定为 2 倍，对不同时间尺度的活动可能不是最优
3. **依赖预提取特征**：HiERO 在预提取的视频特征上操作，未进行端到端训练，可能限制了其潜力
4. **步骤粒度歧义**：定性分析表明，部分失败案例源于步骤标注的粒度歧义（步骤 vs 子步骤的混淆）
5. **仅限第一人称视频**：仅在 Ego4D 数据集上训练和评估，对第三人称视频的迁移性未知

## 相关工作与启发

- 与 HierVL 的区别：HierVL 使用视频级摘要做层次对齐，需要额外高层标注；HiERO 的层次结构完全从数据中浮现
- 与 Paprika 的区别：Paprika 依赖 wikiHow 的外部程序知识图谱；HiERO 不需要任何外部知识源
- 与 TW-FINCH 的关系：TW-FINCH 证明了聚类是动作分割的强基线，HiERO 进一步证明层次化聚类可以捕捉更复杂的功能模式
- 从图论视角理解视频可以启发更多时序推理任务的方法设计

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 层次化功能线索从非脚本化视频中自然浮现的思路极具创意
- **实验充分度**: ⭐⭐⭐⭐ — 多个基准、多种骨干、多个任务的全面评估，消融清晰
- **写作质量**: ⭐⭐⭐⭐⭐ — 动机叙述生动（厨房例子），技术描述严谨，图示丰富
- **价值**: ⭐⭐⭐⭐⭐ — 零样本超越全监督的结果非常惊人，为程序学习和视频理解开辟了新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](../../NeurIPS2025/llm_evaluation/bayesian_evaluation_of_large_language_model_behavior.md)
- [\[NeurIPS 2025\] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities](../../NeurIPS2025/llm_evaluation/ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)
- [\[ICML 2025\] ResearchTown: Simulator of Human Research Community](../../ICML2025/llm_evaluation/researchtown_simulator_of_human_research_community.md)
- [\[ICLR 2026\] MOSIV: Multi-Object System Identification from Videos](../../ICLR2026/llm_evaluation/mosiv_multi-object_system_identification_from_videos.md)
- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)

</div>

<!-- RELATED:END -->
