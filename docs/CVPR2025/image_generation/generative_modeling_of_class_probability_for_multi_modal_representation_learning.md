---
title: >-
  [论文解读] Generative Modeling of Class Probability for Multi-Modal Representation Learning
description: >-
  [CVPR 2025][图像生成][多模态对齐] CALM 提出了一种基于类锚点对齐的生成式多模态表示学习方法，通过引入独立数据集的类标签作为锚点来弥合视频和文本之间的模态鸠沟，利用跨模态概率变分自编码器建模不确定性，在四个 benchmark 上特别是在域外评估中显著超越现有方法。 1. 领域现状：多模态理解（特别是视频-…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "多模态对齐"
  - "类锚点概率分布"
  - "变分自编码器"
  - "视频文本检索"
  - "跨模态生成建模"
---

# Generative Modeling of Class Probability for Multi-Modal Representation Learning

**会议**: CVPR 2025  
**arXiv**: [2503.17417](https://arxiv.org/abs/2503.17417)  
**代码**: 无  
**领域**: 多模态VLM / 视频文本检索  
**关键词**: 多模态对齐, 类锚点概率分布, 变分自编码器, 视频文本检索, 跨模态生成建模

## 一句话总结
CALM 提出了一种基于类锚点对齐的生成式多模态表示学习方法，通过引入独立数据集的类标签作为锚点来弥合视频和文本之间的模态鸠沟，利用跨模态概率变分自编码器建模不确定性，在四个 benchmark 上特别是在域外评估中显著超越现有方法。

## 研究背景与动机
1. **领域现状**：多模态理解（特别是视频-文本对齐）通常依赖对比学习，将不同模态的特征投影到共享嵌入空间。CLIP4Clip、TS2-NET、X-pool 等方法在视频文本检索上取得了不错效果。
2. **现有痛点**：对比学习依赖正负样本对的严格定义，但视频和文本之间存在信息不平衡——同一段文字可能对应多个不同视频，每个视频包含文本无法完全描述的丰富视觉信息。这种模态差异导致对比学习在建模底层数据分布时常常失败。
3. **核心矛盾**：模态间直接配对比较忽略了部分匹配和不确定性。判别式方法（对比学习）天然难以建模数据分布的变异性和跨模态的部分信息重叠。
4. **本文要解决什么？** (a) 如何在不依赖严格正负样本对的情况下实现跨模态对齐 (b) 如何建模模态间对齐的不确定性 (c) 如何提升跨域泛化能力
5. **切入角度**：同模态内部关系比跨模态关系简单（统计特性相似），因此可以通过一组与输入无关的"类锚点"建立桥梁——将跨模态对齐转化为对锚点的概率分布对齐。
6. **核心idea一句话**：用独立分类数据集的类标签构建共享锚点空间，让视频和文本分别与锚点计算概率分布，再通过 VAE 从视频-锚点分布生成文本-锚点分布来实现对齐。

## 方法详解

### 整体框架
CALM 基于预训练 CLIP 编码器。输入视频经过 CLIP 视觉编码器 + 时序融合得到视频特征 $\mathbf{V}$，文本经过 CLIP 文本编码器得到句子特征 $\mathbf{S}$。同时，从独立分类数据集（Charades，157 类）提取类标签，转化为 prompt（"The content of [label]"），通过 CLIP 文本编码器得到类锚点特征 $\mathbf{P}$。分别计算视频/文本与锚点的余弦相似度 + softmax 得到概率分布 $\mathbf{V}_p$ 和 $\mathbf{S}_p$。最后用跨模态概率 VAE 从 $\mathbf{V}_p$ 重建 $\mathbf{S}_p$，实现对齐。

### 关键设计

1. **类锚点概率分布 (Class-Prompt Probability Distribution)**:
    - 功能：为每种模态生成相对于共享语义空间的概率表示
    - 核心思路：选取 K=157 个类标签（来自 Charades 数据集），加 prompt 模板 "The content of [label_k]"，送入 CLIP 文本编码器 + 可学习位置嵌入得到锚点 $\mathbf{p}_k$。模态特征与锚点计算余弦相似度后经温度缩放的 softmax 得到概率分布：$\mathbf{V}_p = \text{softmax}(\tau \cdot \cos(\bar{\mathbf{h}}^v, \mathbf{P}))$，$\mathbf{S}_p = \text{softmax}(\tau \cdot \cos(\mathbf{h}_{CLS}^s, \mathbf{P}))$。
    - 设计动机：类锚点独立于输入数据，提供补充语义信息。文本-锚点分布是同模态内部关系（简单），视频-锚点分布是跨模态关系（复杂）。通过对齐两者，将困难的跨模态问题转化为从跨模态分布到同模态分布的生成问题。

2. **跨模态概率变分自编码器 (Cross-Modal Probabilistic VAE)**:
    - 功能：从视频-锚点分布生成文本-锚点分布，同时建模对齐中的不确定性
    - 核心思路：编码器将 $\mathbf{V}_p$ 编码为高斯潜变量 $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$（重参数化技巧）。解码器从 $\mathbf{z}$ 重建 $\hat{\mathbf{S}}_p$。ELBO 分解为重建项 + KL 正则项。重建损失采用交叉熵（因为输出是概率分布）：$\mathcal{L}_{rec} = -\sum_k \mathbf{S}_p^{(k)} \log \hat{\mathbf{S}}_p^{(k)}$。KL 散度正则化后验分布接近标准高斯先验。
    - 设计动机：确定性映射无法捕捉模态间的固有不确定性（同一视频可对应不同描述）。VAE 的潜变量 $\mathbf{z}$ 自然建模这种一对多关系。通过条件分布 $p(\mathbf{S}_p | \mathbf{V}_p)$，模型隐式学习视频和文本的联合表示。

3. **温度缩放与位置嵌入增强**:
    - 功能：控制概率分布的锐度并区分不同类锚点
    - 核心思路：温度参数 $\tau$ 控制 softmax 输出的集中程度——较高温度产生更平滑的分布，较低温度产生更尖锐的分布。每个锚点还增加可学习的位置嵌入 $\mathbf{e}_k^{pos}$，使模型能区分不同锚点的语义角色。
    - 设计动机：纯余弦相似度可能产生过于集中或过于分散的概率，温度参数提供灵活控制。位置嵌入让锚点不仅仅是类标签的语义表示，还包含了可学习的结构信息。

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{rec} + \alpha \mathcal{L}_{KL} + \mathcal{L}_{task}$，其中 $\alpha=0.1$ 平衡重建和 KL 散度，$\mathcal{L}_{task}$ 为下游任务损失（如对比损失/caption 损失）。使用 ViT-B/32 CLIP，潜空间维度 d=256。AdamW 优化器 lr=1e-5，检索任务训练 5 epoch，caption 任务 20 epoch。视频均匀采样 12 帧，分辨率 224×224。

## 实验关键数据

### 主实验

**视频检索（域内）:**

| 数据集 | R@1 | CALM | 之前SOTA | 提升 |
|--------|-----|------|----------|------|
| MSR-VTT | R@1 | **50.8** | 49.0 (DiffusionRet) | +1.8 |
| DiDeMo | R@1 | **51.1** | 47.8 (EMCL) | +3.3 |
| LSMDC | R@1 | **27.5** | 26.0 (T-MASS) | +1.5 |

**视频检索（域外）——关键亮点:**

| 训练集→测试集 | R@1 | CALM | 之前SOTA | 提升 |
|---------------|-----|------|----------|------|
| MSR-VTT→DiDeMo | R@1 | **41.2** | 37.3 (T-MASS) | +3.9 |
| MSR-VTT→LSMDC | R@1 | **21.4** | 19.6 (T-MASS) | +1.8 |
| DiDeMo→LSMDC | R@1 | **22.1** | 20.4 (T-MASS) | +1.7 |
| DiDeMo→MSR-VTT | R@1 | **41.7** | 39.7 (T-MASS) | +2.0 |

### 消融实验

| 配置 | MSR-VTT R@1 | 说明 |
|------|-------------|------|
| CALM (full) | 50.8 | 完整模型 |
| w/o class anchors | ~47-48 (推测) | 退化为直接对比学习 |
| w/o VAE (直接分布对齐) | 较低 | 无法建模不确定性 |
| w/o positional embedding | 较低 | 锚点无法区分 |

### 关键发现
- CALM 在域外评估中优势更明显（平均 R@1 提升 2.3%），说明类锚点提供的补充语义信息有效提升了泛化能力
- 在 MSR-VTT→DiDeMo 的域外评估中，CALM 的 MnR 从 T-MASS 的 26.3 降到 16.1（提升巨大），表明排序质量大幅提升
- 使用来自独立数据集（Charades，157 类动作类别）的类标签作为锚点，验证了锚点不需要与任务数据重叠也能有效
- CALM 的域内到域外性能下降（平均 ~11%）与 CLIP4Clip 相当，但绝对性能高得多

## 亮点与洞察
- **类锚点桥接思路新颖**：不直接对比视频和文本，而是通过独立的语义锚点空间间接对齐。这个思路可以迁移到任何存在模态鸿沟的跨模态任务（如音频-文本、触觉-视觉）。
- **从判别到生成的范式转换**：将跨模态对齐从"判断是否匹配"转变为"从一个模态概率分布生成另一个"，天然适合处理一对多映射和部分匹配。
- **VAE 建模不确定性**：不同于确定性映射，潜变量 z 的方差显式捕捉了模态对齐的不确定程度，为下游应用提供了置信度信息。
- **域外泛化强**：类锚点作为与输入无关的语义基底，减少了对特定数据集分布的过拟合。

## 局限性 / 可改进方向
- 类锚点数量 K=157 是从 Charades 固定选取，未探索更大或更多样的锚点集合（如 ImageNet-1K 的 1000 类）对性能的影响
- 仅使用 ViT-B/32 CLIP，未验证在更大模型（ViT-L/14）上的效果
- VAE 的隐空间维度 d=256 的选择缺乏充分消融
- 概率分布是通过全局平均帧特征计算的，丢失了时序信息的细节；可以尝试帧级别的锚点分布
- 类锚点的选择对结果的影响值得深入探究——不同类别体系是否产生不同效果

## 相关工作与启发
- **vs CLIP4Clip**: CLIP4Clip 直接对比视频和文本特征，依赖正负样本对。CALM 引入锚点作为中介，避免了对比学习的刚性配对要求
- **vs DiffusionRet**: DiffusionRet 用扩散模型建立共享潜空间，但仍依赖直接匹配。CALM 的锚点方法更轻量且泛化更好
- **vs T-MASS**: T-MASS 将文本嵌入视为随机质量建模不确定性，CALM 通过 VAE 在概率分布层面更系统地建模不确定性
- **vs UATVR**: UATVR 直接在特征空间做分布匹配，在域外评估中性能下降明显；CALM 的锚点机制提供了更稳定的跨域表示

## 评分
- 新颖性: ⭐⭐⭐⭐ 类锚点概率分布对齐是新颖的视角，但核心组件（VAE、CLIP）都是成熟技术
- 实验充分度: ⭐⭐⭐⭐ 四个数据集 + 域内域外评估全面，但消融不够详细
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 域外泛化提升显著，对多模态对齐研究有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](../../NeurIPS2025/image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[CVPR 2025\] Symbolic Representation for Any-to-Any Generative Tasks](symbolic_representation_for_any-to-any_generative_tasks.md)
- [\[CVPR 2025\] Multi-Group Proportional Representation for Text-to-Image Models](multi-group_proportional_representations_for_text-to-image_models.md)
- [\[CVPR 2025\] SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction](syncvp_joint_diffusion_for_synchronous_multi-modal_video_prediction.md)

</div>

<!-- RELATED:END -->
