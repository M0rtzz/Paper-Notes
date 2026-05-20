---
title: >-
  [论文解读] Rethinking Query-Based Transformer for Continual Image Segmentation
description: >-
  [CVPR 2025][语义分割][持续图像分割] 本文深入分析了 query-based Transformer 中 built-in objectness 的产生与消亡机制，提出 SimCIS 方法通过懒惰查询预对齐（QPA）、一致选择损失（CSL）和虚拟查询（VQ）三个模块…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "持续图像分割"
  - "Transformer"
  - "内建物体性"
  - "虚拟查询"
  - "灾难性遗忘"
---

# Rethinking Query-Based Transformer for Continual Image Segmentation

**会议**: CVPR 2025  
**arXiv**: [2507.07831](https://arxiv.org/abs/2507.07831)  
**代码**: [https://github.com/SooLab/SimCIS](https://github.com/SooLab/SimCIS)  
**领域**: 图像分割 / 持续学习  
**关键词**: 持续图像分割, query-based Transformer, 内建物体性, 虚拟查询, 灾难性遗忘

## 一句话总结

本文深入分析了 query-based Transformer 中 built-in objectness 的产生与消亡机制，提出 SimCIS 方法通过懒惰查询预对齐（QPA）、一致选择损失（CSL）和虚拟查询（VQ）三个模块，在保持 objectness 的同时提升可塑性，在 ADE20K 上的持续全景分割和持续语义分割任务中显著超越 SOTA。

## 研究背景与动机

**领域现状**：持续图像分割（CIS）需要模型在多阶段学习中逐步适应新类别，同时保留旧类别知识。近年来，基于 query 的 Transformer（如 Mask2Former）被引入 CIS 领域，其 built-in objectness 被认为可以缓解 mask 生成中的灾难性遗忘。现有方法（如 ECLIPSE、CoMasTRe）通常冻结 mask 生成相关的参数，将 mask 分割与持续学习过程解耦。

**现有痛点**：作者发现解耦框架存在两个关键问题：（1）**可塑性丧失**——objectness 的优势在短任务序列中减弱甚至产生负面影响，在最短的两任务设置中性能甚至低于 baseline；（2）**严重依赖输入数据顺序**——在十次随机试验中，最差情况比默认设置显著下降，缺乏鲁棒性。

**核心矛盾**：built-in objectness 虽然存在于特征图中（像素特征包含充分的语义先验），但随着训练阶段推移会逐渐消亡。根本原因是：由于背景语义偏移，不同阶段的语义先验不同，导致可学习 query 与旧类别的像素特征逐渐失去对齐。

**本文目标**：理解 built-in objectness 的本质，在不同任务长度和数据输入顺序下实现一致的性能提升，尤其是提升可塑性。

**切入角度**：作者发现高度聚合的图像特征为 query 提供了一条"捷径"——通过 decoder 简单地与特征图中的语义先验对齐就能生成 mask。基于此，直接从特征图选取像素特征作为 query 的初始化，可以实现"完美对齐"来保持 objectness。

**核心 idea**：用从特征图中选取的像素特征替代可学习 query（懒惰预对齐），结合跨阶段一致选择约束和虚拟查询回放机制，同时保持 objectness 和提升可塑性。

## 方法详解

### 整体框架

SimCIS 基于 Mask2Former 架构，包含三个核心模块：（1）懒惰查询预对齐（QPA）从像素特征中选取语义最显著的位置初始化 query，保证每个阶段的 objectness；（2）一致选择损失（CSL）确保跨阶段的选择稳定性；（3）虚拟查询（VQ）存储并回放旧类别的 query 特征以避免类别预测的灾难性遗忘。输入图像经过 backbone 和 pixel decoder 提取多尺度像素特征后，QPA 从中选取 N 个最显著的特征点作为 object query，送入 Transformer decoder 生成 mask 和类别预测。

### 关键设计

1. **懒惰查询预对齐（QPA）**:

    - 功能：从像素特征图中选取语义最显著的位置作为 query 的初始特征，确保 query 与语义先验"完美"预对齐
    - 核心思路：为每个类别维护一组可训练的 prototype $p^i \in \mathbb{R}^D$，计算每个像素特征与所有 prototype 的相似度，选取相似度最高的 N 个特征点作为 query。关键是对 query 做 stop gradient，防止训练过程破坏特征图中的信息。每个新阶段的 prototype 集合通过拼接旧阶段和新阶段的 prototype 得到：$\mathcal{P}^t = \text{concat}(\mathcal{P}^{t-1}, \{p^i | i \in C^t\})$
    - 设计动机：传统可学习 query 在多阶段训练中会与特征图失对齐，导致 objectness 消亡。直接从特征图选取可以保证每个阶段都与当前语义先验对齐，同时 stop gradient 保持特征稳定性

2. **一致选择损失（CSL）**:

    - 功能：确保同一图像在不同阶段选择的语义最显著位置保持一致
    - 核心思路：在当前阶段 t 训练时，用上一阶段 t-1 的选择索引 $\mathcal{I}^{t-1}$ 从当前特征图中取出特征点，计算它们与旧 prototype 的相似度分布，通过 KL 散度损失约束这个分布与上一阶段保持一致。公式为 $L_{csl} = \frac{1}{|\mathcal{I}^{t-1}|} \sum KL(\text{旧分布} \| \text{新分布})$
    - 设计动机：避免了传统蒸馏方法在保留旧先验时重新引入背景标注错误的问题。得益于 QPA 的设计，可以自然地保留旧类别 query 位置，同时允许新 query 选择新类别

3. **虚拟查询（VQ）**:

    - 功能：通过存储和回放旧类别的 query 特征来避免类别预测的灾难性遗忘
    - 核心思路：分三步实现。首先，通过二分匹配结果从 decoder 输出中选取匹配的 query 存入类别队列形成 VQ bank。其次，利用伪分布统计分析当前阶段各旧类别的出现频率，对稀有类别加权采样。最后，采样的虚拟查询拼接到正常 query 中送入 decoder，但设计了 skip attention 策略——VQ 跳过 cross-attention 和 self-attention，只参与 FFN 层的计算，避免影响正常 query 的注意力过程
    - 设计动机：相比传统图像回放方法，VQ 将存储需求降低约 10 倍，独立于输入数据顺序，且保护数据隐私。虚拟查询天然包含类别语义信息，可以模拟特定语义而无需实际包含对应类别的图像

### 损失函数 / 训练策略

整体损失包含 Mask2Former 原有的分类损失和 mask 损失，加上 CSL 损失用于跨阶段一致性约束。VQ 仅计算分类损失 $L_{\text{class}}$ 来处理类别遗忘问题。采用预训练的 ResNet-50（全景分割）和 ResNet-101（语义分割）作为 backbone，输入分辨率分别为 640×640 和 512×512。虚拟查询数量设为 80。

## 实验关键数据

### 主实验

| 数据集/设置 | 指标 | SimCIS | ECLIPSE | BalConpas | 提升(vs BalConpas) |
|------------|------|--------|---------|-----------|-------------------|
| ADE20K CPS 100-5 | PQ(all) | **35.4** | 32.9 | 30.8 | +4.6 |
| ADE20K CPS 100-10 | PQ(all) | **38.1** | 33.9 | 34.7 | +3.4 |
| ADE20K CPS 100-50 | PQ(all) | **40.0** | 35.6 | 37.1 | +2.9 |
| ADE20K CPS 50-10 | PQ(all) | **36.3** | 26.8 | 31.4 | +4.9 |
| ADE20K CSS 100-5 | mIoU(all) | **38.7** | 34.2 | 33.8 | +4.9 |
| ADE20K CSS 100-10 | mIoU(all) | **42.3** | 34.6 | 38.6 | +3.7 |
| ADE20K CSS 100-50 | mIoU(all) | **48.6** | 37.1 | 43.3 | +5.3 |

### 消融实验

| 配置 | CPS base | CPS all | CSS base | CSS all | 说明 |
|------|----------|---------|----------|---------|------|
| Baseline (Pseudo Label) | 31.6 | 28.2 | 15.6 | 13.2 | Mask2Former + 伪标签 |
| + QPA | 30.7 | 27.9 | 37.4 | 30.5 | base mIoU +21.8 |
| + QPA + CSL | 35.7 | 31.8 | 43.2 | 34.5 | CSL 对 base 提升显著 |
| + QPA + VQ | 35.1 | 31.2 | 42.5 | 34.8 | VQ 对 new 有帮助 |
| Full (QPA+CSL+VQ) | **42.1** | **35.4** | **46.7** | **38.7** | 三者互补 |

### 关键发现

- **QPA 贡献最大**：在 CSS 任务中将 base mIoU 从 15.6% 提升到 37.4%（+21.8%），证明直接从特征图选取 query 是保持 objectness 的核心
- **VQ 存储效率远高于图像回放**：使用 80 个 VQ 样本（5.9MB）比使用 300 张回放图像（11.8MB）性能更好，PQ 提升 +1.4% 且仅用 27% 存储
- **对数据输入顺序的鲁棒性**：在 10 次随机试验中，SimCIS 的性能方差远小于 ECLIPSE 和 BalConpas，证明 objectness 的有效利用提升了鲁棒性
- **在 100-50 任务中接近 joint 上限**：CPS 100-50 任务 PQ 达到 40.0 vs joint 的 40.4，base 类别甚至超过 joint

## 亮点与洞察

- **对 built-in objectness 的深入分析**：通过可视化揭示了 objectness 来源于 query 与特征图语义先验的对齐、消亡于多阶段训练中的对齐漂移，这一分析框架具有广泛参考价值
- **VQ 的 skip attention 设计**：虚拟查询跳过 attention 只参与 FFN，既避免了对真实 query 的干扰，又能利用 FFN 层传播类别信息，这种设计可以迁移到其他需要混合真实/虚拟 token 的场景
- **用特征选取代替特征学习**：放弃学习 query 转而直接选取像素特征的"懒惰"思路很巧妙，本质上是利用了预训练特征已有的语义聚类性质

## 局限与展望

- 方法依赖于 prototype 能准确表征类别语义，在类别数量极大时可能面临 prototype 管理的可扩展性问题
- VQ bank 的队列长度和采样策略对性能有影响，作者选择了 80 个样本作为最优，但不同数据集可能需要不同设置
- 仅在 ADE20K 上评估，缺乏在 COCO、Cityscapes 等其他更具挑战性数据集上的验证
- 可以探索将 QPA 机制与其他持续学习方法（如 prompt tuning）结合

## 相关工作与启发

- **vs ECLIPSE**: ECLIPSE 通过冻结大部分参数并提供可训练 query 来微调，但冻结导致可塑性丧失。SimCIS 不冻结参数而是保证每次选取正确位置作为 query，同时保持灵活性
- **vs BalConpas**: BalConpas 使用特征蒸馏和图像回放，但在长序列任务中性能下降明显。SimCIS 的 VQ 机制在存储效率和性能上都优于图像回放
- **vs CoMFormer**: CoMFormer 是第一个在持续全景分割中使用 query-based 方法的工作，但依赖蒸馏和伪标签。SimCIS 从更本质的角度解决 objectness 保持问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 对 objectness 的分析有深度，QPA 和 VQ 设计简单有效但并非颠覆性创新
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖 CPS/CSS 两个任务、多种设置和随机序列鲁棒性测试，消融实验完整
- 写作质量: ⭐⭐⭐⭐ 分析链条清晰，图表丰富，但部分公式符号略显复杂
- 价值: ⭐⭐⭐⭐ 在持续分割领域提供了一个简单有力的 baseline，分析框架有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[CVPR 2025\] MambaVision: A Hybrid Mamba-Transformer Vision Backbone](mambavision_a_hybrid_mamba-transformer_vision_backbone.md)
- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](../../ICCV2025/segmentation/hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[CVPR 2025\] Your ViT is Secretly an Image Segmentation Model](your_vit_is_secretly_an_image_segmentation_model.md)
- [\[ACL 2026\] AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation](../../ACL2026/segmentation/anchorseg_language_grounded_query_banks_for_reasoning_segmentation.md)

</div>

<!-- RELATED:END -->
