---
title: >-
  [论文解读] PartCraft: Crafting Creative Objects by Parts
description: >-
  [ECCV 2024][部件级控制] 提出 PartCraft，首次实现了基于部件选择的文本到图像生成控制——用户可以从不同物体中"挑选"各部件（如鸟的头、翅膀、身体），模型将它们自然地组合为一个全新且结构合理的创意物体。 当前生成式 AI（如 Stable Diffusion）的创意控制手段主要依赖文本描述或草图…
tags:
  - "ECCV 2024"
  - "部件级控制"
  - "文本到图像生成"
  - "文本反演"
  - "注意力损失"
  - "创意生成"
---

# PartCraft: Crafting Creative Objects by Parts

**会议**: ECCV 2024  
**arXiv**: [2407.04604](https://arxiv.org/abs/2407.04604)  
**代码**: [https://github.com/kamwoh/partcraft](https://github.com/kamwoh/partcraft)  
**领域**: 其他（生成式AI / 可控生成）  
**关键词**: 部件级控制, 文本到图像生成, 文本反演, 注意力损失, 创意生成

## 一句话总结

提出 PartCraft，首次实现了基于部件选择的文本到图像生成控制——用户可以从不同物体中"挑选"各部件（如鸟的头、翅膀、身体），模型将它们自然地组合为一个全新且结构合理的创意物体。

## 研究背景与动机

当前生成式 AI（如 Stable Diffusion）的创意控制手段主要依赖文本描述或草图，但：
- **文本控制不精确**：复杂视觉细节难以用语言精确描述，生成结果偏离预期
- **草图门槛高**：不是每个用户都具备细致绘画能力
- **现有个性化方法局限**：DreamBooth、Textual Inversion 等方法以"整个物体"为学习单位，无法实现部件级别的组合控制
- **额外控制信号复杂**：基于 bounding box 或 segmentation mask 的方法需要大量额外输入

**核心动机**：人类创造力往往是将已有概念的不同部分重组——比如想要一只具有蓝鸟头部、红雀翅膀、麻雀身体的"理想鸟"。PartCraft 让用户通过简单"选择"部件即可实现这种创意组合。

## 方法详解

### 整体框架

PartCraft 建立在 Stable Diffusion v1.5 之上，采用文本反演（Textual Inversion）策略。整体流程：
1. 无监督部件发现：利用 DINOv2 特征进行三层层级聚类，将物体分解为语义部件
2. 部件编码：将每个部件映射到文本 token 空间
3. 基于注意力损失的训练：确保各部件在图像中正确放置且互不重叠
4. Bottleneck 编码器：加速收敛并提升生成保真度

### 关键设计

1. **无监督部件发现（三层层级聚类）**：

    - 利用 DINOv2 提取所有训练图像的特征图
    - **顶层**：k-means (k=2) 分离前景与背景
    - **中层**：对前景 patch 聚类为 M 个语义部件（如鸟的头部、翅膀等）
    - **底层**：将每个中层聚类进一步细分为 K 个子类（对应不同物种的同一部件）
    - 每张图的每个区域获得聚类标签 $p = (0, k_0), (1, k_1), ..., (M, k_M)$，作为训练时的文本描述
    - 选择 DINOv2 而非 VLPart 等模型是为了更高的灵活性和域泛化能力

2. **Part Token Bottleneck 编码器**：

    - 传统文本反演直接学习词嵌入 $e(p)$，各 token 之间无信息交互，学习效率低
    - 引入两层 MLP + ReLU 的瓶颈网络 $f(\cdot)$：$y_p = f(e(p))$
    - **核心思路**：先将 token 投影到共享的"部件类别空间"（如"头部"的通用表示），再微调适应具体细节
    - 实验显示收敛速度显著加快（传统方法是 $f$ 为恒等函数的特例）

3. **基于熵的归一化注意力损失**：

    - 仅用扩散损失 $\mathcal{L}_{ldm}$ 训练会导致部件纠缠（因同一物种的头和身体总是成对出现）
    - 设计交叉熵形式的注意力正则化：
    $\mathcal{L}_{attn} = \mathbb{E}_{z,t,m}[-(S_m \log \hat{A}_m + (1-S_m)\log(1-\hat{A}_m))]$
    - 其中 $\hat{A}_m$ 是跨所有部件归一化后的注意力图，$S_m$ 是第 m 个部件的分割掩码
    - **归一化关键**：确保每个图像位置上所有部件的注意力之和为 1，即每个位置最多被一个部件占据
    - 相比 Break-a-Scene 的 MSE 注意力损失，熵损失天然适合"只有一个部件出现"的约束

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{total} = \mathcal{L}_{ldm} + \lambda_{attn} \mathcal{L}_{attn}$，其中 $\lambda_{attn} = 0.01$

- 使用 LoRA 微调交叉注意力块（而非全模型），降低训练开销
- 注意力图聚焦于 16×16 分辨率层（语义信息丰富）
- 鸟类设 M=5 个部件（头、胸腹、翅膀、腿、尾巴），狗设 M=7 个部件
- K=256 确保覆盖所有细粒度类别

## 实验关键数据

### 主实验

在 CUB-200-2011（鸟类）和 Stanford Dogs 上进行部件重建评估：

| 方法 | FID↓ | CLIP↑ | DINO↑ | EMR↑ | CoSim↑ |
|------|------|-------|-------|------|--------|
| Textual Inversion | 10.10 | 0.784 | 0.607 | 0.305 | 0.842 |
| DreamBooth | 12.94 | 0.775 | 0.594 | 0.355 | 0.856 |
| Custom Diffusion | 37.61 | 0.694 | 0.504 | 0.338 | 0.833 |
| Break-a-Scene | 20.05 | 0.742 | 0.549 | 0.390 | 0.854 |
| **PartCraft** | **12.86** | **0.783** | **0.618** | **0.460** | **0.882** |

PartCraft 在 EMR（精确匹配率）上比最强基线 Break-a-Scene 高 7%，CoSim 高 2.8%。

### 消融实验

| 配置 | FID↓ | EMR↑ | CoSim↑ | 说明 |
|------|------|------|--------|------|
| Full PartCraft | 12.86 | 0.460 | 0.882 | 完整模型 |
| w/o Bottleneck | 16.36 | ~0.460 | ~0.882 | FID退化3.5，生成质量下降 |
| MSE attn loss (BaS) | - | 显著下降 | 显著下降 | EMR/CoSim大幅退化 |
| w/o both | - | 最差 | 最差 | 双重退化 |

### 关键发现

- **部件组合数越多越难**：随着混合的物种数从 1 增加到 4，EMR 和 CoSim 均下降
- **PartCraft 在 4 物种组合下仍显著优于其他方法**，Break-a-Scene 虽也用了注意力损失但效果不如归一化熵损失
- **词嵌入空间可视化（tSNE）** 显示 PartCraft 的部件 token 按语义自然聚类（头部聚在一起、翅膀聚在一起），而其他方法的嵌入混乱
- **跨域迁移**：学到的狗部件可以迁移到猫/狮子上（如"一只猫长着比格犬的耳朵"），也可用于创意生成（鸟形机器人）

## 亮点与洞察

- **选择即创作**：将创意过程简化为"点选部件"，无需文本描述或绘画能力，优雅且实用
- **熵归一化注意力损失**是核心贡献——解决了多部件学习中的纠缠问题，设计动机清晰（每个位置只属于一个部件）
- Bottleneck 编码器的设计巧妙地利用了"共享部件知识"，让相同语义的不同实例（如不同鸟的头部）共享表示空间
- 无监督部件发现方案灵活可扩展，无需任何部件标注

## 局限与展望

- 部件发现依赖 DINOv2 的自监督特征，对精度有天然上限；可引入更强的编码器（如 VLPart 的改进版）
- **小部件（如尾巴、腿）的组合效果较差**，因为这些部件在图像中占据面积小，聚类和注意力监督均不够精确
- 跨域部件组合（如将动物部件与汽车部件组合）尚处初探阶段
- 仅在 Stable Diffusion v1.5 上验证，升级到更新模型可能性能更好

## 相关工作与启发

- 与 Break-a-Scene 最接近，但后者的 MSE 注意力损失不如本文的熵损失有效
- 可以启发 3D 生成领域：如果能在 3D 模型上做类似的部件级组合控制，将极具应用价值
- 部件发现模块可以独立使用，为其他细粒度控制任务提供语义分割

## 评分

- 新颖性: ⭐⭐⭐⭐ （部件选择 → 创意生成的思路新颖，但核心技术基于已有框架）
- 实验充分度: ⭐⭐⭐⭐ （两个数据集 + 定量定性 + 消融 + 可视化 + 迁移实验丰富）
- 写作质量: ⭐⭐⭐⭐⭐ （motivating examples 直观，方法描述清晰）
- 价值: ⭐⭐⭐⭐ （对创意设计领域有实际应用潜力）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] PISR: Polarimetric Neural Implicit Surface Reconstruction for Textureless and Specular Objects](pisr_polarimetric_neural_implicit_surface_reconstruction_for_textureless_and_spe.md)
- [\[ICLR 2026\] Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects](../../ICLR2026/others/articulation_in_motion_prior-free_part_mobility_analysis_for_articulated_objects.md)
- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ECCV 2024\] Event-based Mosaicing Bundle Adjustment](event-based_mosaicing_bundle_adjustment.md)
- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)

</div>

<!-- RELATED:END -->
