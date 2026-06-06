---
title: >-
  [论文解读] DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers
description: >-
  [CVPR 2025][语义分割][视觉提示微调] DA-VPT 提出了一种分布感知的视觉提示微调框架，通过在 ViT 深层利用度量学习构建 prompt 与视觉 token/CLS token 之间的语义度量空间，引导 prompt 作为"语义桥梁"传递图像 patch 的类特异性信息到 CLS token…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "视觉提示微调"
  - "参数高效微调"
  - "度量学习"
  - "Transformer"
---

# DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers

**会议**: CVPR 2025  
**arXiv**: [2505.23694](https://arxiv.org/abs/2505.23694)  
**代码**: [https://github.com/Noahsark/DA-VPT](https://github.com/Noahsark/DA-VPT)  
**领域**: 图像分割  
**关键词**: 视觉提示微调, 参数高效微调, 度量学习, Vision Transformer, 语义分割

## 一句话总结
DA-VPT 提出了一种分布感知的视觉提示微调框架，通过在 ViT 深层利用度量学习构建 prompt 与视觉 token/CLS token 之间的语义度量空间，引导 prompt 作为"语义桥梁"传递图像 patch 的类特异性信息到 CLS token，在 24 个识别任务和 2 个分割任务上以极少参数显著超越标准 VPT。

## 研究背景与动机
预训练 Vision Transformer（ViT）在各类计算机视觉任务中表现出色，但全量微调面临计算开销大、过拟合和灾难性遗忘等问题。参数高效微调（PEFT）方法应运而生，其中视觉提示微调（VPT）通过在每层 ViT 中插入可学习的 prompt token 来适配下游任务，是最有前景的方向之一。

然而，现有 VPT 方法（包括 VPT-Deep、E2VPT、GateVPT 等）主要关注 prompt 的连接结构和动态门控机制，却忽略了一个根本问题：**prompt 与数据表征之间的内在关系**。当前 VPT 随机初始化 prompt，仅通过下游任务目标优化，导致 prompt 的分布无约束 —— prompt 可能从任意类别的特征中吸引信息，反而妨碍了 CLS token 聚合类特异性信息。

核心问题是：**能否引导 prompt 促进图像 token 和 CLS token 之间的信息流动，从而增强表征学习？**

DA-VPT 的核心 idea：在 ViT 深层构建 prompt 与图像 token 之间的语义度量空间，利用代理锚（Proxy-Anchor）度量学习，使每个 prompt 选择性地从相关类别的视觉 token 中捕获信息，并将其传递给 CLS token，形成"图像 patch → prompt → CLS token"的语义信息桥梁。

## 方法详解

### 整体框架
DA-VPT 建立在 VPT-Deep 基础之上。在 ViT 的每层中插入 M 个可学习 prompt token，与图像 patch token 和 CLS token 一起通过 Transformer block 处理。关键改进在于：对深层的 prompt 施加度量学习约束，使 prompt 与同类视觉 token 靠近、与异类视觉 token 远离。同时构建 CLS token 与 prompt 之间的类似度量。整体损失为交叉熵 + 两项度量学习损失。

### 关键设计
1. **Prompt-Token 度量学习 $\mathcal{L}_{ML}(\mathbf{X}, \mathbf{P})$**:
    - 在 ViT 深层为每个 prompt 分配一个类标签（通过动态映射）
    - 使用 Proxy-Anchor 损失函数构建度量空间，使 prompt 与同类视觉 token 的余弦相似度尽可能大、与异类 token 的相似度尽可能小
    - 直觉：余弦相似度与注意力权重的 Query-Key 匹配天然对齐，因此在球面空间中更近的 (prompt, token) 对在注意力图中也会有更高的匹配概率
    - 选择 Proxy-Anchor 而非 Proxy-NCA 或三元组损失的原因：prompt 数量远少于数据 token（M ≪ N），需要考虑这种不对称性
    - 实际中使用 Query 投影向量 $\mathbf{Q} = \mathbf{P}^l \mathbf{W}_Q^l$ 进行比较效果更好

2. **CLS-Prompt 度量学习 $\mathcal{L}_{ML}(\mathbf{P}, \mathbf{x}_{cls})$**:
    - 拉近 CLS token 与相应类别 prompt 的距离，推远与不同类别 prompt 的距离
    - 确保 CLS token 能通过注意力机制从正确的 prompt 中高效聚合信息
    - 与 prompt-token 度量共同作用，形成完整的信息传递链路

3. **动态类-Prompt 映射 (Dynamic Mapping)**:
    - 由于类别数 C >> prompt 数 M，需要将 C 个类映射到 M 个 prompt
    - 训练前先用预训练 ViT 跑一轮获取各类别的 CLS 表征均值
    - 使用 k-means 聚类将类别分成 M 个簇，每个簇对应一个 prompt
    - 每个 epoch 结束后用更新的类表征重新聚类，保持映射准确性
    - 后续 epoch 的 k-means 用上一轮质心初始化，计算开销随训练递减

4. **显著性 Patch 选择 (Saliency Patch Selection)**:
    - 使用注意力层输出后的表征 $\mathbf{X}^l = \text{MHSA}(\mathbf{X}^l)$ 作为显著性聚合
    - 避免直接从注意力图提取显著 patch 的计算开销（不兼容 Flash Attention）

5. **高效偏置微调 (Efficient Bias Tuning)**:
    - DA-VPT+ 额外释放 ViT 主干的 Key 和 Value 线性投影的偏置项
    - 偏置参数极少但提供额外灵活性，在度量学习引导下效果显著

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{CE} + \beta \mathcal{L}_{ML}(\mathbf{X}, \mathbf{P}) + \lambda \mathcal{L}_{ML}(\mathbf{P}, \mathbf{x}_{cls})$
- 度量学习参数：margin $\delta=32$, temperature $\tau=10$
- 最优 prompt 数量约为 20
- 度量学习损失仅在最后一层应用时效果最佳

## 实验关键数据

### 主实验

| 数据集 | 指标 | DA-VPT+ | VPT-Deep | E2VPT | 提升 (vs VPT-Deep) |
|--------|------|------|----------|------|------|
| FGVC (5 datasets) | Mean Acc | 91.94 | 89.11 | 89.22 | +2.83 |
| VTAB-1K (19 datasets) | Mean Acc | 76.14 | 71.96 | 73.94 | +4.18 |
| ADE20K 分割 | mIoU-SS | 46.47 | 44.08 | - | +2.39 |
| PASCAL Context 分割 | mIoU-SS | 50.40 | 49.51 | - | +0.89 |

| 预训练方式 | 方法 | 参数量 (M) | FGVC Mean | VTAB Mean |
|-----------|------|-----------|-----------|-----------|
| MAE (自监督) | VPT-Deep | 0.20 | 72.02 | 41.73 |
| MAE (自监督) | DA-VPT+ | 0.22 | 83.20 | 69.61 |
| MoCo-V3 (自监督) | VPT-Deep | 0.20 | 83.12 | 65.90 |
| MoCo-V3 (自监督) | DA-VPT+ | 0.24 | 86.16 | 73.53 |

### 消融实验

| 配置 | VTAB Natural | CUB-200 | 说明 |
|------|---------|------|------|
| VPT-Deep 基线 | 79.45 | 88.64 | 无度量学习 |
| + $\mathcal{L}_{ML}(\mathbf{X}, \mathbf{P})$ + $\mathcal{L}_{ML}(\mathbf{P}, \mathbf{x}_{cls})$ | 80.53 (+1.08) | 89.86 (+1.22) | 度量学习核心增益 |
| + Efficient Bias | 81.98 (+2.53) | 90.89 (+2.25) | 完整 DA-VPT+ |

### 关键发现
- 度量学习损失在最深层（第 12 层）应用时效果最佳，因为深层有更高级的语义特征
- 使用数据均值初始化 prompt 反而会降低 DA-VPT 的效果，因为同质化初始化增加了引导 prompt 捕获判别信息的难度
- 在自监督预训练模型（MAE, MoCo）上提升尤为显著，DA-VPT+ 在 MAE 上将 VTAB 性能从 41.73 提升到 69.61（+27.88 pp）
- DA-VPT+ 在所有预训练设置下以更少参数超越全量微调
- 可视化显示正向 prompt 成功识别了被 CLS token 随后选择的信息性 patch，验证了"桥梁"假说

## 亮点与洞察
1. **prompt 作为语义桥梁的新视角**：不仅仅把 prompt 看作额外容量，而是赋予其明确的语义角色 —— 连接图像 patch 与 CLS token 的信息传递中介
2. **度量学习与注意力机制的理论联系**：通过定理证明，余弦相似度的变化直接影响注意力权重，建立了度量学习引导注意力的数学基础
3. **自监督模型上的巨大提升**：在 MAE 模型上，DA-VPT+ 将 VTAB 从 41.73 提升到 69.61，说明在 feature 分布不太结构化的自监督模型上，语义引导的 prompt 学习价值更大
4. **少即是多**：使用约 20 个 prompt（远少于类别数）配合动态映射即可获得最佳效果

## 局限性 / 可改进方向
- 动态类-prompt 映射需要额外的预热 epoch 和每 epoch 的 k-means 聚类
- 目前仅在分类和分割任务上验证，检测等其他密集预测任务未涉及
- 分割实验仅在 ViT-L 上进行，尚需在 ViT-B 等更小规模上验证参数效率的完整泛化性
- 注意力伪影（artifact）问题未完全解决，仅在微调阶段引入 prompt 可能有固有限制
- 与 LoRA、Adapter 等其他 PEFT 方法的系统组合对比不够充分

## 相关工作与启发
- 与 Proxy-Anchor 度量学习方法的巧妙结合：将 prompt 类比为度量学习中的 proxy
- E2VPT 和 GateVPT 关注 prompt 的连接结构，DA-VPT 则关注 prompt 的语义分布
- 启发：在其他 PEFT 方法（如 LoRA）中是否也可以引入度量学习来引导适配参数的分布？
- 对分割任务的启发：通过更好地引导 patch-level 的特征聚合，可以用极少参数实现接近全量微调的分割性能

## 评分
- 新颖性: ⭐⭐⭐⭐ 将度量学习引入 prompt 优化的视角新颖，理论分析（注意力-相似度关系）增加了可解释性
- 实验充分度: ⭐⭐⭐⭐⭐ 24 识别任务 + 2 分割任务 + 3 种预训练模型 + 详尽消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论推导完整，可视化分析充分
- 价值: ⭐⭐⭐⭐ 为 prompt 学习提供了新的优化范式，在自监督模型上的巨大提升具有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](../../NeurIPS2025/segmentation/vision_transformers_with_self-distilled_registers.md)
- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](../../ICCV2025/segmentation/legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)

</div>

<!-- RELATED:END -->
