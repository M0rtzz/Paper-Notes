---
title: >-
  [论文解读] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models
description: >-
  [CVPR 2025][图像生成][3D头发重建] 本文通过自动化构建迄今最大的3D合成头发数据集（40K风格），训练一个基于扩散Transformer的头皮纹理生成模型，首次以图像条件方式直接预测单根发丝（而非引导发丝）的潜码纹理图，实现从单张图像重建包括爆炸头和秃顶在内的多样化3D发型。
tags:
  - CVPR 2025
  - 图像生成
  - 3D头发重建
  - 扩散模型
  - 合成数据
  - 头皮纹理
  - 发丝级别重建
---

# DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2505.06166](https://arxiv.org/abs/2505.06166)  
**代码**: [项目页](https://radualexandru.github.io/difflocks/)  
**领域**: 3D视觉 / 数字人  
**关键词**: 3D头发重建, 扩散模型, 合成数据, 头皮纹理, 发丝级别重建

## 一句话总结
本文通过自动化构建迄今最大的3D合成头发数据集（40K风格），训练一个基于扩散Transformer的头皮纹理生成模型，首次以图像条件方式直接预测单根发丝（而非引导发丝）的潜码纹理图，实现从单张图像重建包括爆炸头和秃顶在内的多样化3D发型。

## 研究背景与动机

**领域现状**：3D头发重建对数字人至关重要，但现有公开数据集极小（USC-HairSalon仅343个、CT2Hair仅10个），导致方法依赖引导发丝+上采样+后处理等低维中间表示来应对数据不足。

**现有痛点**：（1）引导发丝+上采样流程无法建模发丝间的空间关系，限制了复杂发型（尤其是卷发）的重建精度；（2）低维scalp嵌入（如HAAR的32×32）丢失细节；（3）现有方法无法处理非裔爆炸头和秃顶模式。

**核心矛盾**：数据量不足→必须使用低维表示→无法建模高维空间结构→无法表达复杂发型。如果数据充足，许多简化假设都不再需要。

**本文目标**：构建足够大的训练集，使模型能直接在高维空间（256×256 scalp texture，每texel一条发丝潜码）上学习，绕过引导发丝和后处理。

**切入角度**：用Blender geometry nodes自动化大规模3D头发数据生成，以数据规模驱动高维建模。

**核心 idea**：40K合成数据 + DINOv2条件 + Hourglass Diffusion Transformer 在256×256 scalp texture上直接生成单根发丝潜码。

## 方法详解

### 整体框架
输入单张RGB图，DINOv2提取局部和全局特征作为条件。扩散模型在256×256的scalp纹理+密度图上去噪，每个texel包含64维发丝潜码。采样后按密度图概率采样texel，用预训练的strand decoder并行解码约100K条发丝。

### 关键设计

1. **大规模自动化3D头发数据集**:

    - 功能：提供40K多样化3D发型训练数据
    - 核心思路：从75个手动创建的基础引导发丝（每个约50条，几分钟制作）出发，通过Blender的58个geometry nodes流水线（349个辅助节点）自动插值、聚拢、卷曲、加噪声和物理模拟。110个随机参数控制几何和材质，255个HDRI提供光照多样性。最终每个样本约100K发丝+768×768 RGB渲染
    - 设计动机：关键洞察——有足够数据后，引导发丝、低维嵌入、上采样和后处理都变得不必要

2. **单根发丝级别的Scalp纹理表示**:

    - 功能：在2D纹理空间编码完整3D发型
    - 核心思路：用Strand VAE将每条发丝压缩为64维潜码 $z$（编码器=1D卷积，解码器=modulated SIREN），根据发丝根部位置分配到256×256 UV纹理的对应texel。用push-pull算法填充稀疏区域。额外引入密度图 $D \in [0,1]^{256\times256}$ 表示每个位置生成发丝的概率
    - 设计动机：相比HAAR的32×32引导发丝纹理，256×256单根发丝纹理能直接建模发丝间的空间关系，Transformer的self-attention天然适合捕获这种关系

3. **条件Scalp扩散模型**:

    - 功能：从单张图像生成scalp纹理和密度图
    - 核心思路：使用Hourglass Diffusion Transformer (HDiT) 作为骨干，以DINOv2图像特征作为条件信号。扩散在拼接的纹理+密度图 $[T, D]$ 上进行像素级扩散（EDM框架）。DINOv2的局部patch特征和全局CLS特征分别提供空间和语义条件
    - 设计动机：DINOv2特征比传统方向图（orientation map）更丰富，包含外观和语义信息。预训练视觉backbone使合成数据训练的模型能泛化到真实图像

### 损失函数 / 训练策略
Strand VAE：位置L1 + 方向L1 + 曲率L1 + KL正则。扩散模型：EDM标准去噪损失。曲率损失对卷发尤其重要——局部形状比全局位置更影响视觉感知。

## 实验关键数据

### 主实验

| 方法 | Recall↑ | L2距离↓ | 支持发型类型 |
|------|---------|--------|------------|
| Neural Strands | 中等 | 中等 | 直发为主 |
| HAAR | 较好 | 较好 | 多种，不含爆炸头 |
| **DiffLocks** | **最优** | **最优** | **全类型含爆炸头和秃顶** |

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| 引导发丝(32×32) vs 单根发丝(256×256) | 256×256分辨率显著提升卷发重建质量 |
| 无曲率损失 | 高度卷曲发丝的形状恶化 |
| DINOv2 vs orientation map条件 | DINOv2泛化到真实图像更好 |
| 无密度图 | 无法建模秃顶、发际线等不均匀密度模式 |

### 关键发现
- 首次成功从单张图像重建爆炸头风格和秃顶发型
- 仅在合成数据训练就能泛化到真实照片，归功于DINOv2的预训练特征
- 生成的发丝可直接用于Unreal Engine等实时游戏引擎，无需额外稠密化处理
- 40K数据集是推动性能的核心因素

## 亮点与洞察
- "数据规模解锁建模维度"的策略——通过扩充数据移除所有之前需要的简化假设（引导发丝、低维嵌入、后处理）
- 密度图+潜码纹理的双通道设计优雅地处理了不均匀密度问题（秃顶、发际线）
- 曲率损失的引入反映了对卷发视觉感知的深刻理解

## 局限与展望
- 合成与真实域之间仍有gap，极端真实场景可能失败
- 不处理头发颜色/材质，仅重建几何
- 依赖SMPL-X的固定头皮拓扑
- 可考虑引入text conditioning实现文本驱动的3D发型生成

## 相关工作与启发
- **vs HAAR**: HAAR用VAE在32×32的引导发丝scalp空间生成，DiffLocks用扩散在256×256单发丝级别生成，精度和多样性大幅提升
- **vs Perm**: Perm解耦全局形状和局部细节，但数据集有限；DiffLocks用大规模数据直接端到端学习
- **vs Multi-view方法**: DiffLocks仅需单张图像，虽精度略低但实用性强得多

## 评分
- 新颖性: ⭐⭐⭐⭐ 大规模数据+高维建模的策略新颖且有效
- 实验充分度: ⭐⭐⭐⭐ 定量对比+真实图像定性评估，数据集将公开
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数据生成流水线详实
- 价值: ⭐⭐⭐⭐ 对数字人产业有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[CVPR 2025\] StdGEN: Semantic-Decomposed 3D Character Generation from Single Images](stdgen_semantic-decomposed_3d_character_generation_from_single_images.md)
- [\[CVPR 2025\] CustAny: Customizing Anything from A Single Example](custany_customizing_anything_from_a_single_example.md)

<!-- RELATED:END -->
