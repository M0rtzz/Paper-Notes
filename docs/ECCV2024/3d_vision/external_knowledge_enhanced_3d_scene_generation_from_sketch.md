---
title: >-
  [论文解读] External Knowledge Enhanced 3D Scene Generation from Sketch
description: >-
  [ECCV 2024][3D视觉][3D场景生成] 提出SEK框架，结合手绘草图和外部物体关系知识库作为扩散模型的条件，通过知识增强图推理和频谱滤波器，端到端地同时生成3D室内场景的布局和物体几何形状。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D场景生成
  - 草图引导
  - 外部知识库
  - 扩散模型
  - 知识图谱推理
---

# External Knowledge Enhanced 3D Scene Generation from Sketch

**会议**: ECCV 2024  
**arXiv**: [2403.14121](https://arxiv.org/abs/2403.14121)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 3D场景生成, 草图引导, 外部知识库, 扩散模型, 知识图谱推理

## 一句话总结

提出SEK框架，结合手绘草图和外部物体关系知识库作为扩散模型的条件，通过知识增强图推理和频谱滤波器，端到端地同时生成3D室内场景的布局和物体几何形状。

## 研究背景与动机

**领域现状**: 3D场景生成在游戏开发、电影制作、AR/VR和室内设计等领域需求日益增长。现有方法主要依赖图像、文本、场景图或房间布局等条件来引导生成过程。

**现有痛点**: (1) 基于图像的方法受限于2D-3D一致性约束，多样性不足；(2) 基于草图的方法主要聚焦于单个3D物体生成，无法处理复杂的场景级生成；(3) 现有场景生成方法要么仅生成布局+从数据库检索物体（缺乏多样性），要么依赖手工定义的简单物体关系（如GRAINS的手工层次图）。

**核心矛盾**: 手绘草图天然稀疏且模糊，无法提供足够的信息来确定所有物体的精确形状和空间关系；同时用户可能期望场景中包含草图中未画出的物体（不可见物体），现有方法难以处理这种隐含需求。

**本文目标**: 如何从稀疏的手绘草图出发，生成包含合理布局、详细物体几何和可信物体关系的完整3D室内场景。

**切入角度**: 引入外部知识库提供物体间关系的先验信息，弥补草图的模糊性，同时利用扩散模型同时生成布局和物体形状。

**核心 idea**: 构建物体关系知识库，通过知识增强的图推理与草图特征融合，为条件扩散模型提供丰富的生成引导。

## 方法详解

### 整体框架

SEK的pipeline分为四部分：(1) **场景矩阵表示**：将3D场景编码为矩阵 $\mathcal{O} \in \mathbb{R}^{D \times M}$，每行对应一个物体的位置+形状潜码；(2) **外部知识库构建**：从训练数据中提取五种物体关系并计算概率；(3) **知识增强草图引导**：通过ViT提取草图特征，结合知识图推理特征形成扩散条件；(4) **3D场景去噪Transformer**：在频谱滤波器增强下，迭代去噪生成场景矩阵。

### 关键设计

1. **场景矩阵表示 (Matrix Conversion)**:

    - **功能**: 将3D场景统一编码为固定维度的矩阵，使扩散过程可以在矩阵空间中操作。
    - **核心思路**: 每个物体 $\mathbf{o}_i = [\mathcal{G}_i, \mathcal{F}_i] \in \mathbb{R}^{D \times 1}$，其中 $\mathcal{G}_i = [\alpha_i, \mathbf{s}_i, \mathbf{t}_i]$ 包含旋转角（2D正余弦参数化）、3D包围盒尺寸和平移，$\mathcal{F}_i$ 是通过DeepSDF训练得到的形状潜码。场景中物体数不同时用零向量填充到固定数量 $M$。
    - **设计动机**: 矩阵表示使得场景的布局和形状可以在统一的空间中被同时生成，避免了先生成布局再检索物体的两阶段方案。

2. **外部知识库 (Knowledge Base)**:

    - **功能**: 存储物体间丰富的关系先验 $KB = (\mathcal{V}, \mathcal{R}, p)$，为场景生成提供合理性约束。
    - **核心思路**: 使用DBSCAN对训练场景中的物体进行聚类，提取5种关系：(a) **Attachment** (相邻物体最小距离<体素长度)；(b) **Alignment** (包围盒平面共面)；(c) **Dependent** (同组但非attachment/alignment)；(d) **Parallel Collinearity** (不同组物体包围盒水平轴平行)；(e) **Co-occurrence** (不同组物体在同一场景中共现)。关系概率通过sigmoid归一化：$p_{ij} = 1/(1 + e^{-10 \cdot n_{ij}^{\mathcal{R}} / max(n)^{\mathcal{R}}})$。
    - **设计动机**: 草图本身信息稀疏且模糊，知识库可以：(1) 增强可见物体的描述（双向验证）；(2) 推断不可见物体（如看到sofa就可能需要table）。

3. **知识增强图推理 (KeGR)**:

    - **功能**: 将外部知识与目标物体实体结合，通过图卷积推理得到丰富的条件特征。
    - **核心思路**: 用GloVe初始化物体节点特征 $h_i \in \mathbb{R}^{1 \times D_\omega}$，构建全连接子图 $G_i^E = (h_i^E, \mathcal{E}_i^E, \mathcal{P}_i^E)$，通过多步图卷积推理：$H_i^{E(j)} = \delta(A_i^E H_i^{E(j-1)} W^{E(j)})$。对所有关系类型的特征用 $1 \times 1$ 卷积融合得到图特征 $H^G$。最终条件特征为草图ViT特征 $H^S$ 与图特征 $H^G$ 的拼接：$c = [H^S, H^G]$。
    - **设计动机**: 图卷积可以有效传播物体间的关系信息，使每个物体节点感知到其与其他物体的关系上下文。

4. **频谱滤波器 (Spectrum-Filter)**:

    - **功能**: 在去噪Transformer中抑制零填充值对有效物体特征的干扰。
    - **核心思路**: 观察到填充零相比有效物体表示具有低频方差分布，使用高通滤波器在频谱域中抑制低频填充成分：$\text{EF}(\mathcal{O}_I, B) = \mathcal{O}_I + e^{-t} \Theta_{IFFT}(\text{Conv}(\sigma(\mathcal{O}_I, B) \circledast \Theta_{FFT}(\mathcal{O}_I)))$，其中 $e^{-t}$ 使滤波强度随时间步递减（因为噪声增加时填充的低频特性逐渐消失）。
    - **设计动机**: 推理时没有填充mask可用来过滤无意义的生成值，频谱滤波器提供了一种自适应的方式来增强有效物体特征。

### 损失函数 / 训练策略

场景扩散损失：$\mathcal{L}_{sce} = \mathbb{E}_{c,t,\epsilon,\mathcal{O}_0}[\|\epsilon - \epsilon_\theta(c, t, \mathcal{O}_t)\|^2]$，其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$。采用标准DDPM训练策略，去噪过程通过逆马尔科夫链迭代生成场景矩阵。场景补全任务使用DDIM inversion。

## 实验关键数据

### 主实验

在3D-FRONT数据集上的场景生成：

| 方法 | Bedroom FID↓ | Dining FID↓ | Living FID↓ | Dining CKL↓ |
|------|-------------|-------------|-------------|-------------|
| ATISS | 18.60 | 38.66 | 40.83 | 0.64 |
| DiffuScene | 18.29 | 32.60 | 36.18 | 0.22 |
| Graph-to-3D | 61.24 | 54.11 | 41.13 | 1.68 |
| **SEK (Ours)** | **15.21** | **25.46** | **31.24** | **0.16** |

相比最强对手DiffuScene，在Dining room上FID提升17.41%，CKL提升37.18%。

场景补全任务：

| 方法 | Bedroom FID↓ | Dining FID↓ | Dining KID↓ |
|------|-------------|-------------|-------------|
| DiffuScene | 27.32 | 40.99 | 6.31 |
| **SEK (Ours)** | **21.84** | **33.03** | **5.18** |

平均FID提升19.12%，KID提升20.06%。

### 消融实验

| 配置 (Dining room) | FID↓ | SCA% | CKL↓ |
|-----|------|------|------|
| ×Knowledge / ViT Sketch + SF | 33.29 | 56.81 | 0.85 |
| ResNet50 Sketch + Knowledge + SF | 24.68 | 52.70 | 0.18 |
| ViT Sketch + Knowledge, w/o SF | 25.83 | 54.19 | 0.37 |
| **ViT Sketch + Knowledge + SF (Full)** | **23.97** | **51.97** | **0.16** |

### 关键发现

- **草图和知识互补性极强**：仅用草图FID为33.29，仅用知识FID为32.26，两者结合后FID降至23.97，说明草图提供空间布局信息，知识提供关系约束，缺一不可。
- **知识库可跨数据集迁移**：从3D-FRONT构建的知识库直接用于ScanNet场景生成，性能仅有微小下降（FID: 33.81 vs 33.47），说明知识库捕获的是通用的物体关系模式。
- 频谱滤波器（SF）对FID有约2点的提升，验证了填充零值干扰问题的存在及解决方案的有效性。
- ViT vs ResNet50作为草图编码器差异不大（有知识时），表明知识库才是性能的主要驱动力。

## 亮点与洞察

- **知识库设计思路务实**：不是手工定义关系，而是从数据中统计提取5种关系类型并概率化存储，兼顾了灵活性和可扩展性。
- **"不可见物体"生成**是独特的卖点：通过知识库可以推断出用户草图中未画但场景中应存在的物体（如画了sofa就推断出需要table），提高了场景的完整性和合理性。
- **同时生成布局和形状**：不同于大多数方法先布局再检索，SEK端到端生成所有属性，保证了更强的一致性。
- 频谱滤波器对零填充问题的处理很巧妙，利用了填充值在频域的低频特性。

## 局限与展望

- 草图获取方式是对渲染图像做Canny边缘检测后手动去除墙壁，与真实手绘草图存在domain gap，实际应用中可能需要额外的sketch-photo domain adaptation。
- 知识库是静态构建的，无法随用户交互动态更新或个性化定制。
- 场景中物体数量需要填充到固定上限 $M$，限制了可生成场景的复杂度。
- 仅在3D-FRONT数据集上训练和评估，室内场景类型有限（卧室、餐厅、客厅），对更复杂场景（如办公室、实验室）的泛化能力未知。
- 没有用户研究来验证生成场景是否真正符合用户意图。

## 相关工作与启发

- **vs DiffuScene**: DiffuScene是无条件扩散场景生成，SEK通过引入草图+知识条件实现了更好的可控性和质量。
- **vs Graph-to-3D**: Graph-to-3D依赖完整的场景图描述，用户指定成本高且不直观；SEK只需手绘草图+物体列表，交互更自然。
- **vs ATISS**: ATISS基于给定布局的自回归方法，SEK通过扩散模型可以同时考虑所有物体的全局一致性。
- **vs Sketch2Scene**: Sketch2Scene需要额外的3D模型库进行检索和放置，SEK是端到端生成，无需外部库。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将外部知识库与草图条件扩散模型结合用于3D场景生成是首次，频谱滤波器的设计也很新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖了生成、补全、知识迁移三个任务，消融全面，但缺少用户研究
- 写作质量: ⭐⭐⭐⭐ 整体结构清晰，知识库构建和推理过程描述详尽，图示丰富
- 价值: ⭐⭐⭐⭐ 知识增强的条件生成框架思路可推广到其他3D生成任务，知识库的可迁移性是重要的实践价值

<!-- RELATED:START -->

## 相关论文

- [DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting](dreamscene360_unconstrained_text-to-3d_scene_generation_with_panoramic_gaussian_.md)
- [WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution](../../AAAI2026/3d_vision/ie-srgs_an_internal-external_knowledge_fusion_framework_for_high-fidelity_3d_gau.md)
- [GVGEN: Text-to-3D Generation with Volumetric Representation](gvgen_text-to-3d_generation_with_volumetric_representation.md)
- [TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)

<!-- RELATED:END -->
