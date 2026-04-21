---
title: >-
  [论文解读] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation
description: >-
  [NeurIPS 2025][图像分割][开放词汇部件分割] 提出LangHOPS，首个基于多模态大语言模型（MLLM）的开放词汇物体-部件实例分割框架，在语言空间中建立object-part层次关系，利用MLLM的知识和推理能力链接多粒度概念，在PartImageNet上以56.9% AP超越SOTA 5.5%，跨数据集设置超4.8%。
tags:
  - NeurIPS 2025
  - 图像分割
  - 开放词汇部件分割
  - 物体-部件层次
  - MLLM
  - 语言空间层次建模
  - 实例分割
---

# LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2510.25263](https://arxiv.org/abs/2510.25263)  
**代码**: 有（待发布）  
**领域**: 分割  
**关键词**: 开放词汇部件分割, 物体-部件层次, MLLM, 语言空间层次建模, 实例分割

## 一句话总结

提出LangHOPS，首个基于多模态大语言模型（MLLM）的开放词汇物体-部件实例分割框架，在语言空间中建立object-part层次关系，利用MLLM的知识和推理能力链接多粒度概念，在PartImageNet上以56.9% AP超越SOTA 5.5%，跨数据集设置超4.8%。

## 研究背景与动机

**开放词汇分割的粒度局限**：当前OVS方法主要关注物体级分割，物体-部件（partonomic）分割仍是开放问题。将物体分解为语义部件（如"汽车→车轮、车门、引擎盖"）对机器人操作、细粒度识别等下游任务至关重要。

**现有部件分割方法的不足**：
   - VLPart、PartGLEE等方法依赖**启发式或可学习的视觉分组**来建模object-part关系
   - 视觉空间的分组缺乏语义先验知识（如"鸟有翅膀和喙"），对未见类别泛化能力差
   - 缺少object-part之间的层次上下文，导致部件解析不精确

**核心动机**：将object-part层次关系从视觉空间迁移到**语言空间**，利用MLLM内化的世界知识（"这个物体应该有哪些部件？"）来初始化和细化部件查询（part queries），实现更好的跨类别泛化。

## 方法详解

### 整体框架

LangHOPS采用两阶段架构：
1. **物体分割阶段**：检测并分割图像中的物体实例，生成物体查询 $\mathbf{O}^L$
2. **MLLM驱动的部件解析阶段**：将物体查询与语言空间中的层次化部件查询一起输入MLLM，利用MLLM的推理能力细化部件查询 $\mathbf{P}$，再送入部件解码器生成最终分割

输入为一张图像和候选物体-部件类别列表，输出为层次化的物体和部件实例分割结果。

### 关键设计

**1. 语言空间层次建模（Language-Grounded Hierarchies）**

区别于传统方法使用可学习的随机初始化查询，LangHOPS在语言空间中构建部件查询的初始表征：

- 给定候选部件类别名称，利用object-part之间的语义层次关系构建初始part queries
- 例如对于"dog"物体，其关联部件"head"、"leg"、"tail"的文本表征被用于初始化对应的part queries
- 这种初始化方式天然编码了物体-部件的归属关系，比随机初始化的可学习查询具有更好的语义先验

消融验证：用 $N$ 个可学习查询替代语言层次初始化（"w/o hierarchy"），在PartImageNet上AP从26.7%降至22.5%（-4.2%），证明语言层次的重要性。

**2. MLLM驱动的物体-部件解析（MLLM-based Object-Part Parsing）**

这是LangHOPS的核心创新模块：

- 物体分割阶段输出的物体查询 $\mathbf{O}^L$ 携带了物体的视觉特征
- 将 $\mathbf{O}^L$ 与语言层次初始化的部件查询 $\mathbf{P}^0$ 一起输入MLLM
- MLLM基于对物体视觉特征的理解和内化的世界知识，**推理出该物体应包含哪些部件**，并据此细化部件查询
- 输出refined part queries $\mathbf{P}$，送入部件解码器生成最终部件分割掩码

关键的梯度流设计：部件分割损失的梯度通过MLLM反传到物体查询 $\mathbf{O}^L$，实现**物体-部件协同训练**（Object-Part Synergy）——部件分割的优化同时改善物体分割的质量。

消融验证：
- 用Q-Former替代MLLM（"w/o MLLM"），PartImageNet AP从26.7%降至23.2%（-3.5%）
- 切断梯度流（"Detached Obj-Part Seg"），物体和部件的注意力分数均下降

**3. 物体-部件协同效应**

LangHOPS展示了联合训练物体+部件分割比单独训练物体分割更优的协同效应：

- "Obj Seg"（仅物体分割）：PartImageNet物体mAP 67.9%
- "Obj-Part Seg"（联合训练）：PartImageNet物体mAP **68.3%**（+0.4%），部件mAP **14.9%**
- 联合训练不仅提供部件分割能力，还**反过来提升物体分割**，证明MLLM解析模块的梯度回传有效改善了物体查询的质量

注意力分数分析：协同训练下物体注意力分数从0.76提升到0.82，部件从0.58提升到0.67。

### 损失函数 / 训练策略

- **两阶段训练**：先在物体分割上训练（Stage 1），再联合训练物体+部件分割（Stage 2）。两阶段策略在跨数据集泛化上优于一阶段直接训练
- 消融对比：两阶段 vs 一阶段在跨数据集设置下AP 26.7 vs 25.4（+1.3），但一阶段在in-domain上略优（58.6 vs 56.9）
- 训练数据可扩展性：支持渐进加入更多数据集（PPS-116 → +INS → +INS+PART），LangHOPS在加入部件级标注时获得最大增益

## 实验关键数据

### 跨数据集实验：PPS-116训练 → PartImageNet评测

| 方法 | PPS-116 obj | PPS-116 part | PPS-116 AP | +INS+PART AP |
|------|-------------|-------------|------------|--------------|
| PSALM† | 31.6 | 8.27 | 13.4 | 21.9 |
| PartGLEE | 38.4 | 9.20 | 15.6 | 21.0 |
| **LangHOPS** | **44.5** | **8.86** | **16.7** | **26.7** |

LangHOPS在加入部件级数据后获得+10.0 AP增益，远超PartGLEE的+5.4和PSALM的+8.5。

### In-domain实验：PartImageNet训练+评测

| 方法 | obj mAP | part mAP | AP |
|------|---------|----------|-----|
| PSALM† | 79.2 | 40.1 | 48.7 |
| PartGLEE | 81.4 | 41.5 | 50.4 |
| **LangHOPS** | **83.9** | **49.2** | **56.9** |

LangHOPS在in-domain上超PartGLEE 6.5% AP，部件mAP提升7.7%。

### 零样本语义分割

| 方法 | PPS-116 hIoU | PartImageNet hIoU | ADE20K hIoU |
|------|-------------|-------------------|-------------|
| PartCLIPSeg | 38.8 | 53.9 | 38.6 |
| PartGLEE | 37.1 | — | 41.8 |
| PartCATSeg | 50.4 | 72.7 | **50.0** |
| **LangHOPS** | **52.1** | **72.8** | 49.5 |

LangHOPS在PPS-116和PartImageNet上取得最佳hIoU，在ADE20K上与专门为语义分割设计的PartCATSeg持平。

### 消融实验

| 消融设置 | PartImageNet AP | PPS-116 AP |
|---------|-----------------|------------|
| w/o MLLM (Q-Former) | 23.2 | 18.4 |
| w/o hierarchy | 22.5 | 19.1 |
| **LangHOPS** | **26.7** | **19.8** |

### 关键发现

1. **数据扩展性**：LangHOPS在加入部件级标注时增益最大（+10.0 AP），而PartGLEE反而出现部件mAP回退（+5.9→+5.4），说明缺乏层次上下文时更多数据不一定有益
2. **跨数据集泛化优势**：PartImageNet→PPS-116（更多新类和更细部件）方向对所有方法更难，但LangHOPS仍保持优势
3. **MLLM的推理能力**：MLLM贡献3.5% AP（vs Q-Former），不仅是特征提取，更是语义推理
4. **协同训练的双向增益**：部件分割的优化反过来改善物体分割（0.4% obj mAP提升）

## 亮点与洞察

1. **语言空间层次建模**：首次将object-part层次关系从视觉空间移到语言空间，天然利用语义先验
2. **MLLM驱动的部件解析**：利用MLLM的世界知识推理物体应有的部件，而非纯视觉分组
3. **协同训练机制**：通过梯度回传实现物体-部件双向增益，是一个优雅的多任务设计
4. **强跨数据集泛化**：在多个cross-dataset设置中稳定领先，证明语言先验对未见类别的泛化作用

## 局限与展望

1. **计算开销大**：MLLM集成显著增加推理成本，不利于实时或端侧部署
2. **训练数据限制**：主要使用常见物体/部件类别的数据集，特殊场景（如工业零件、医学影像）可能需要额外微调
3. **2D到3D的扩展**：当前仅限2D图像分割，结合2D-to-3D lifting用于机器人等3D应用是重要方向
4. 可探索用更轻量的语言模型替代MLLM以降低开销
5. 可结合SAM的prompt-based分割能力进一步增强部件掩码质量

## 相关工作与启发

- 与VLPart和PartGLEE形成对比：前者用文本引导的检测头，后者用统一架构但缺乏语义层次，LangHOPS补充了语义推理维度
- MLLM在分割任务中的应用正在兴起（PSALM、LISA等），LangHOPS开辟了**部件级**这一更精细的方向
- 语言空间层次建模思路可推广到其他层次化视觉任务（如场景图解析、关系推理）

## 评分

- 新颖性: ⭐⭐⭐⭐ 语言空间层次建模+MLLM部件解析，方向新颖
- 实验充分度: ⭐⭐⭐⭐⭐ in-domain/cross-dataset/zero-shot三设置+充分消融
- 写作质量: ⭐⭐⭐⭐ 框架清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 开辟MLLM在部件级分割的新方向

<!-- RELATED:START -->

## 相关论文

- [PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding](partnext_a_next-generation_dataset_for_fine-grained_and_hierarchical_3d_part_und.md)
- [COS3D: Collaborative Open-Vocabulary 3D Segmentation](cos3d_collaborative_open-vocabulary_3d_segmentation.md)
- [Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation](../../CVPR2025/segmentation/fine-grained_image-text_correspondence_with_cost_aggregation_for_open-vocabulary.md)
- [Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](../../CVPR2026/segmentation/geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)

<!-- RELATED:END -->
