---
title: >-
  [论文解读] Effective SAM Combination for Open-Vocabulary Semantic Segmentation
description: >-
  [CVPR 2025][图像分割][开放词汇语义分割] 提出 ESC-Net，一种单阶段开放词汇语义分割模型，通过从 CLIP 图像-文本相关性图中生成伪提示（pseudo prompts）并将其嵌入预训练 SAM 解码器 block 中，高效利用 SAM 的类无关分割能力来增强空间聚合，配合 Vision-Language Fusion (VLF) 模块实现精确的掩码预测，在 ADE20K、PASCAL-VOC、PASCAL-Context 上均取得 SOTA 性能。
tags:
  - CVPR 2025
  - 图像分割
  - 开放词汇语义分割
  - SAM
  - CLIP
  - 伪提示
  - 视觉语言融合
---

# Effective SAM Combination for Open-Vocabulary Semantic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2411.14723](https://arxiv.org/abs/2411.14723)  
**代码**: 无  
**领域**: 语义分割 / 开放词汇分割  
**关键词**: 开放词汇语义分割, SAM, CLIP, 伪提示, 视觉语言融合

## 一句话总结
提出 ESC-Net，一种单阶段开放词汇语义分割模型，通过从 CLIP 图像-文本相关性图中生成伪提示（pseudo prompts）并将其嵌入预训练 SAM 解码器 block 中，高效利用 SAM 的类无关分割能力来增强空间聚合，配合 Vision-Language Fusion (VLF) 模块实现精确的掩码预测，在 ADE20K、PASCAL-VOC、PASCAL-Context 上均取得 SOTA 性能。

## 研究背景与动机

**领域现状**：开放词汇语义分割旨在对任意类别进行像素级标注。主流方法分两大类：(1) 两阶段方法先用强大的掩码生成器（如 SAM）产生类无关的 proposal mask，再用 CLIP 分类；(2) 单阶段方法直接建模图像-文本相关性来预测分割掩码（如 CAT-Seg）。

**现有痛点**：两阶段方法（如 OVSeg、SAN）需要完整运行 SAM 图像编码器（ViT-H），计算和内存开销极大，且裁剪区域送入 CLIP 时存在域差距问题。单阶段方法（如 CAT-Seg）效率更高，但 CLIP 天生关注全局语义对齐而非局部空间信息，导致相关性图分辨率低、掩码边界不精确。

**核心矛盾**：SAM 有强大的空间聚合和精细分割能力，但两阶段使用方式效率太低；单阶段方法效率高但缺乏精细的空间分割能力。能否在单阶段框架中"借用" SAM 的空间聚合能力而不增加太多计算开销？

**本文目标** 设计一种高效方式将 SAM 的分割能力融入单阶段开放词汇分割框架，兼得精度和效率。

**切入角度**：作者注意到 SAM 的核心能力在于其可提示的分割框架——即使提示模糊或指向多个物体，SAM 也能生成有效的分割掩码。因此可以跳过 SAM 的图像编码器（最重的部分），只使用 SAM 的解码器 block，用从 CLIP 相关性图生成的伪提示来驱动它。

**核心 idea**：用 CLIP 的图像-文本相关性图生成伪提示，输入预训练 SAM 解码器 block 增强 CLIP 特征的空间聚合。

## 方法详解

### 整体框架
ESC-Net 的输入是图像和候选类别文本描述。先用 CLIP 视觉和语言编码器提取特征 $F_v$ 和 $F_l$，计算余弦相似性得到初始相关性图 $C_{v\&l}$。然后通过 $N=4$ 个 ESC Block 迭代精化：每个 block 先用 PPG（Pseudo Prompt Generator）从相关性图生成伪提示，再用预训练 SAM block 处理 CLIP 图像特征增强空间聚合，最后用 VLF 模块精化相关性图。最终通过 U-Net 风格解码器生成分割掩码。

### 关键设计

1. **伪提示生成器 (PPG)**:

    - 功能：从 CLIP 的图像-文本相关性图中为每个候选类别生成 SAM 可接受的提示（点坐标 + 掩码）
    - 核心思路：对每个类别 $n$ 的相关性图 $C_{v\&l}^n$ 先做 softmax 得到概率图，用阈值 $\alpha$ 二值化得到近似物体区域，然后用 k-means 按像素位置聚类将区域分为 $N_o = 5$ 个物体区域（处理同一类别多个实例的情况）。将概率图与聚类区域相乘过滤出各物体的概率分布，取概率最高点作为伪点提示、区域掩码作为伪掩码提示。总共生成 $N_o$ 个点和 $N_o$ 个掩码，通过 SAM 的 prompt encoder 编码为稀疏和密集提示特征。
    - 设计动机：传统 SAM 使用精确的用户提示，但在开放词汇设置中无法获得精确提示。利用 CLIP 相关性图作为"近似定位信号"来生成伪提示是自然的选择。SAM 对模糊提示有鲁棒性，因此即使提示不精确也能产生有效分割。

2. **SAM Block 集成**:

    - 功能：利用预训练 SAM 解码器的空间聚合能力增强 CLIP 图像特征
    - 核心思路：取预训练 SAM 掩码解码器中的 Transformer block（包含 prompt self-attention 和 bidirectional cross-attention），将 CLIP 图像特征 $F_v$ 和每个类别的伪提示同时输入。SAM block 对每个类别并行处理（batch化），输出按类别增强的图像特征 $(F_v^n)'$，再通过 $1 \times 1$ 卷积将所有类别的增强特征融合回统一的 $F_v'$。关键是只使用 SAM 的解码器 block（不用图像编码器），因此计算开销远小于两阶段方法。
    - 设计动机：SAM 的解码器在预训练时已学会如何根据提示进行区域级空间聚合，直接复用这些预训练参数可以将"如何分割"的知识注入 CLIP 特征，而无需重新训练一个分割网络。

3. **视觉-语言融合模块 (VLF)**:

    - 功能：利用增强后的图像特征和文本特征来精化相关性图
    - 核心思路：分两步。(a) 图像引导：将相关性图 $C_{v\&l}^n$ 经 $1 \times 1$ 卷积嵌入后与增强图像特征 $F_v'$ 拼接，送入 Swin Transformer block 生成视觉相关性图 $C_v^n$。(b) 文本引导：用线性 Transformer（无位置嵌入以保持类别数量无关性）对 $C_v^n$ 和文本特征 $F_l$ 做交叉注意力，建模不同类别间关系，得到精化后的相关性图 $C_{v\&l}'$。两步均在所有类别上并行计算。
    - 设计动机：SAM block 只增强空间信息但不直接建模图像-文本关系，VLF 补充了跨模态精化的功能，使相关性图逐步变得更准确。

### 损失函数 / 训练策略
使用标准的语义分割交叉熵损失。CLIP 编码器和 SAM block 用较小学习率（$2 \times 10^{-6}$）微调注意力层，其余部分用较大学习率（$2 \times 10^{-4}$）。只在 COCO-Stuff 上训练，在 ADE20K、PASCAL-VOC、PASCAL-Context 上零样本评测。

## 实验关键数据

### 主实验（CLIP ViT-L/14）

| 方法 | 类型 | A-847 | PC-459 | A-150 | PC-59 | PAS-20 |
|---|---|---|---|---|---|---|
| CAT-Seg | 单阶段 | 16.0 | 23.8 | 37.9 | 63.3 | 97.0 |
| MAFT+ | 单阶段 | 15.1 | 21.6 | 36.1 | 59.4 | 96.5 |
| EBSeg | 两阶段(SAM) | 13.7 | 21.0 | 32.8 | 60.2 | 97.2 |
| **ESC-Net** | **单阶段** | **18.1** | **27.0** | **41.8** | **65.6** | **98.3** |

ESC-Net 在所有基准上均取得 SOTA，A-847 上比 CAT-Seg 提升 2.1，PC-459 上提升 3.2，A-150 上提升 3.9 mIoU。

### 消融实验

| 配置 | A-847 | PC-459 | A-150 | PC-59 | PAS-20 |
|---|---|---|---|---|---|
| 无 SAM block | 4.8 | 11.7 | 24.2 | 50.4 | 89.4 |
| SAM block (随机初始化) | 5.9 | 15.8 | 28.4 | 55.9 | 91.5 |
| SAM block (预训练) | **18.1** | **27.0** | **41.8** | **65.6** | **98.3** |

### 关键发现
- 预训练 SAM block 至关重要：随机初始化的 SAM block 只带来微弱提升（+1.1 A-847），而预训练参数带来 +13.3 A-847 的巨大提升，证明 SAM 的预训练分割知识被有效迁移
- 伪提示中"点+掩码"组合最优，bounding box 提示效果较差（因为伪 bbox 精度低且可能重叠）
- 类别数极多的数据集（A-847: 847类）上提升最为显著（+2.1 mIoU），说明 SAM 的空间聚合能力帮助模型更好地区分细粒度类别
- 可视化显示，随着 ESC Block 层数增加，相关性图对目标物体的定位越来越精确和密集

## 亮点与洞察
- **"只用 SAM 解码器不用编码器"的设计取舍**：这是整篇论文最聪明的地方。SAM 图像编码器是计算瓶颈，但分割知识主要在解码器中。通过伪提示桥接 CLIP 特征和 SAM 解码器，以极小的额外开销获得 SAM 的分割能力。
- **伪提示作为两个基础模型的桥梁**：CLIP 擅长语义理解但空间定位弱，SAM 擅长空间分割但不懂语义。用 CLIP 的相关性图生成伪提示驱动 SAM block，让两者各取所长。
- **批量并行的实现**：$N_c$ 个类别的 SAM block 计算通过 batch 并行化，保持了高效推理。

## 局限与展望
- PPG 中的 k-means 聚类数固定为 $N_o = 5$，对于类别实例数差异大的场景（如人群 vs 单个车辆）可能不够灵活
- 使用 SAM ViT-B 的解码器 block，未验证更大 SAM（ViT-L/ViT-H）的效果
- 所有类候选的 SAM block 并行处理意味着内存占用与类别数线性增长，在极多类别（如 A-847）时可能存在内存压力
- 训练只在 COCO-Stuff（171类）上进行，更大规模训练数据是否能进一步提升未验证

## 相关工作与启发
- **vs CAT-Seg**: 同为单阶段方法，CAT-Seg 直接用 CLIP 相关性建模，ESC-Net 在此基础上加入 SAM 解码器增强空间信息，A-847 上高 2.1
- **vs EBSeg/USE**: 这些方法也用 SAM 但需要完整的 SAM 图像编码器（两阶段），ESC-Net 只用解码器 block 就超过了它们
- **vs SAN**: SAN 用 side adapter 辅助 CLIP 分割，ESC-Net 改用预训练 SAM block 作为更强大的空间增强组件
- **vs MAFT+**: MAFT+ 用 content-dependent transfer 适配文本嵌入，ESC-Net 从图像空间增强入手，思路正交

## 评分
- 新颖性: ⭐⭐⭐⭐ 伪提示+SAM解码器的组合思路巧妙，但仍是已有模块的新组合
- 实验充分度: ⭐⭐⭐⭐⭐ 六个基准数据集、两种 VLM 规模、详尽消融
- 写作质量: ⭐⭐⭐⭐ 图示清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 为如何高效利用基础模型组合提供了实用参考

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)

<!-- RELATED:END -->
