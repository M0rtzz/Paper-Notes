---
title: >-
  [论文解读] Denoising Functional Maps: Diffusion Models for Shape Correspondence
description: >-
  [CVPR 2025][图像生成][功能图] 本文提出 DenoisFM，首次用去噪扩散模型直接预测形状间的功能图（functional map），通过模板匹配降低学习复杂度，并提出无监督方法解决拉普拉斯特征向量符号歧义问题，在人体和动物形状匹配上取得有竞争力的性能。
tags:
  - CVPR 2025
  - 图像生成
  - 功能图
  - 去噪扩散模型
  - 形状对应
  - 拉普拉斯特征向量符号歧义
  - 模板匹配
---

# Denoising Functional Maps: Diffusion Models for Shape Correspondence

**会议**: CVPR 2025  
**arXiv**: [2503.01845](https://arxiv.org/abs/2503.01845)  
**代码**: [https://github.com/alekseizhuravlev/denoising-functional-maps/](https://github.com/alekseizhuravlev/denoising-functional-maps/)  
**领域**: 3D形状匹配 / 形状对应  
**关键词**: 功能图、去噪扩散模型、形状对应、拉普拉斯特征向量符号歧义、模板匹配

## 一句话总结
本文提出 DenoisFM，首次用去噪扩散模型直接预测形状间的功能图（functional map），通过模板匹配降低学习复杂度，并提出无监督方法解决拉普拉斯特征向量符号歧义问题，在人体和动物形状匹配上取得有竞争力的性能。

## 研究背景与动机

**领域现状**：形状对应（找两个3D形状间的点对点映射）常通过功能图框架来解决——将高维的逐点映射压缩为小矩阵表示。现有方法通常先学习描述子函数再求解功能图，但受限于小数据集（不超过百个形状），缺乏跨类别泛化能力。

**现有痛点**：(1) 基于描述子的方法需要类别特定训练数据；(2) 大规模变形方法需要大量数据但直接学习变形；(3) 扩散模型在形状匹配中的应用仅限于从预训练模型中提取特征，未直接用于预测功能图。

**核心矛盾**：功能图是小尺寸矩阵（$n \times n$），适合扩散模型处理；但拉普拉斯特征向量的符号歧义导致同一对应关系可能对应 $2^n$ 种不同的功能图，大幅增加学习难度。

**本文目标**：用扩散模型直接预测功能图，利用大规模合成数据训练，并解决符号歧义问题。

**切入角度**：(1) 使用模板匹配（所有形状与同一模板比较）而非对匹配，将复杂度从 $O(N^2)$ 降到 $O(N)$；(2) 提出无监督符号校正方法减少歧义。

**核心 idea**：用 DDPM 学习"从噪声到功能图"的去噪过程，以形状几何特征为条件，利用 SURREAL 的 23万人体形状训练，并通过无监督符号校正将 $2^n$ 种可能功能图统一为一种。

## 方法详解

### 整体框架
训练时：从 SURREAL 大规模人体数据集提取23万个形状与模板的功能图。对每个形状计算拉普拉斯特征向量并用无监督方法校正符号。用标准 U-Net 扩散模型以几何特征为条件学习预测功能图。推理时：对新形状计算条件特征，多次采样功能图并选择 Dirichlet 能量最低的作为最终结果。

### 关键设计

1. **无监督特征向量符号校正**:

    - 功能：将每个形状的 $2^n$ 种可能基底统一为一种确定性选择
    - 核心思路：对每个形状提取无监督表面特征（如 HKS），计算每个特征向量 $\phi_i$ 与特征 $\varsigma_i$ 的带权内积 $\langle \phi_i, \varsigma_i \rangle_A$，如果为负则翻转符号。这利用了同类形状的特征向量与表面特征之间的统计一致性。
    - 设计动机：不解决符号歧义，扩散模型需要学习所有 $2^n$ 种映射，复杂度指数爆炸。

2. **模板匹配策略**:

    - 功能：降低训练和推理的复杂度
    - 核心思路：所有功能图都是"形状→模板"而非"形状A→形状B"。推理时如需A→B的对应，先得到A→模板和B→模板的功能图，再复合得到A→B。
    - 设计动机：对匹配需要 $O(N^2)$ 个功能图，模板匹配只需 $O(N)$。

3. **多样本选择策略**:

    - 功能：利用扩散模型的随机性提升结果质量
    - 核心思路：生成多个候选功能图，用 Dirichlet 能量（衡量对应的平滑性）选择最优。
    - 设计动机：扩散模型的概率性质在这里成为优势——多样本+选择策略比单次确定性预测更鲁棒。

### 损失函数 / 训练策略
标准 DDPM 去噪损失。训练数据23万个由 SMPL 参数模型生成的人体形状。

## 实验关键数据

### 主实验

| 数据集 | 类型 | 本文 | 描述子方法 | 大规模变形方法 |
|--------|------|------|-----------|-------------|
| FAUST (人体) | 同分布 | 有竞争力 | 较好 | 有竞争力 |
| SHREC (人体) | 异构连接 | 有竞争力 | - | - |
| DT4D (非等距人形) | 跨类别 | 泛化能力强 | 需重训 | - |
| SMAL (动物) | 跨类别 | 可行 | 类别特定 | - |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无符号校正 | 性能显著下降 | 歧义增加学习难度 |
| 对匹配 vs 模板匹配 | 模板匹配更好 | 复杂度降低+一致性 |
| 单样本 vs 多样本选择 | 多样本更好 | 利用概率性质 |

### 关键发现
- 扩散模型可以有效学习功能图空间，不需要手工描述子
- 符号校正是关键——不校正时性能大幅下降
- 在非等距人形和动物上零样本泛化良好，证明模型学到了通用几何先验

## 亮点与洞察
- **全新范式**：用生成模型直接预测形状对应的低维表示，跳过了传统的描述子→优化流水线。
- **符号歧义的无监督解决**：用表面特征统计量来确定特征向量符号，简单有效。
- **扩散模型多样本策略**：将扩散的随机性从"缺点"变为"优点"。

## 局限与展望
- 模板匹配假设限制了无法与模板建立对应的形状
- 训练数据仅人体，对完全不同拓扑的形状泛化有限
- 功能图的分辨率受特征向量数 $n$ 限制

## 相关工作与启发
- **vs 描述子方法（DiffusionNet等）**: 不需要手工描述子流水线，直接端到端
- **vs 3D-CODED**: 都用大规模合成数据，但本文用扩散模型而非编码器-解码器

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将扩散模型用于功能图预测，符号校正方法新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证+跨类别泛化
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法动机充分
- 价值: ⭐⭐⭐⭐ 为形状匹配开辟了新的技术路线

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[CVPR 2025\] Optimizing for the Shortest Path in Denoising Diffusion Model](optimizing_for_the_shortest_path_in_denoising_diffusion_model.md)
- [\[CVPR 2025\] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)
- [\[CVPR 2025\] AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer](animer_animal_pose_and_shape_estimation_using_family_aware_transformer.md)
- [\[CVPR 2025\] SGMatch: Semantic-Guided Non-Rigid Shape Matching with Flow Regularization](sgmatch_semantic-guided_non-rigid_shape_matching_with_flow_regularization.md)

<!-- RELATED:END -->
