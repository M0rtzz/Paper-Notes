---
title: >-
  [论文解读] Refer to Any Segmentation Mask Group With Vision-Language Prompts
description: >-
  [ICCV 2025][图像分割][全模态指代分割] 提出全模态指代表达分割（ORES）任务及 RAS 框架，通过掩码级 LMM 和非自回归解码机制，根据视觉-语言混合提示从候选掩码中选择目标掩码组，在新 ORES 数据集及经典 RES/GRES 基准上取得 SOTA。
tags:
  - ICCV 2025
  - 图像分割
  - 全模态指代分割
  - 掩码分组
  - 视觉-语言提示
  - 大型多模态模型
  - 非自回归解码
---

# Refer to Any Segmentation Mask Group With Vision-Language Prompts

**会议**: ICCV 2025  
**arXiv**: [2506.05342](https://arxiv.org/abs/2506.05342)  
**代码**: [Ref2Any](https://Ref2Any.github.io)  
**领域**: segmentation  
**关键词**: 全模态指代分割, 掩码分组, 视觉-语言提示, 大型多模态模型, 非自回归解码

## 一句话总结

提出全模态指代表达分割（ORES）任务及 RAS 框架，通过掩码级 LMM 和非自回归解码机制，根据视觉-语言混合提示从候选掩码中选择目标掩码组，在新 ORES 数据集及经典 RES/GRES 基准上取得 SOTA。

## 研究背景与动机

指代表达分割（RES）将文本描述与分割掩码关联，实现语言驱动的目标定位。然而在自动驾驶、机器人操作、AR 和图像编辑等实际应用中，用户常需表达涉及**参考视觉实体**的复杂关系（如"所有与这个物体颜色相同的东西"），仅依赖文本描述往往难以准确定位参考实体。

现有方法存在三大局限：

**交互式分割模型**（如 SEEM）支持视觉提示，但视觉提示只能直接指向目标实体，无法表达"与参考实体相关的其他目标"

**Grounding LMM**（如 Groundhog）支持区域描述任务，但不能根据掩码提示执行分割

**大多数方法每次只输出单个目标**，无法处理多目标场景

核心动机：定义一个新任务 ORES，允许用户通过**文本+参考掩码**混合提示，一次性返回满足条件的**一组掩码**，实现更灵活、更实用的分割交互。

## 方法详解

### 整体框架

RAS（Refer to Any Segmentation Mask Group）基于 LLaVA-1.5（Vicuna-13B）扩展，包含四个核心模块：

1. **分割基础模型**（SAM/Co-DETR）：生成候选掩码池
2. **视觉编码器集成**：CLIP + SigLIP + ConvNeXt-CLIP + DINOv2 + 2D 位置编码
3. **掩码投影器**：将掩码级特征映射到语言空间
4. **二分类选择器**：对每个候选掩码做是否入组的二分类

### 关键设计一：掩码 Token 化

对每个候选掩码，将其下采样到各视觉编码器特征图的空间尺寸，在掩码区域内做平均池化得到掩码级特征。拼接来自所有编码器的特征后，通过掩码投影器映射到语言特征空间，形成**掩码 Token**。

- 候选掩码 Token 前置 `<mask-pool-pre>` 特殊标记
- 参考掩码 Token 前置 `<mask-ref-pre>` 特殊标记
- 两者共享掩码 Token 化流程，通过不同特殊标记区分角色

### 关键设计二：非自回归掩码组解码

传统自回归方式逐个预测掩码嵌入存在两大问题：(a) LLM 本质建模离散 Token 分布，预测连续嵌入不自然；(b) 无序集合预测需不稳定的二分匹配。

RAS 将掩码组预测重新建模为**逐掩码二分类问题**：

1. 先输入所有上下文 Token（全局视觉 + 文本 + 参考掩码）
2. 再次输入候选掩码 Token，捕获 LLM 输出隐状态
3. 在隐状态之上用二分类器判断每个候选掩码是否应纳入目标组

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} w_i \cdot \text{BCE}(\hat{y}_i, y_i)$$

其中正样本权重 $w_i$ 更大以应对正负样本不平衡。该策略支持**单次前向**完成所有候选掩码的分类，推理延迟仅 0.56s（vs 自回归 2.13s）。

### 关键设计三：多阶段训练

**阶段 1 — 掩码投影器预训练**：冻结所有模块，仅训练掩码投影器。只用掩码 Token（不加全局视觉 Token）让 LLM 预测图像描述，对齐掩码表示和语言空间。

**阶段 2 — 视觉指令微调**：解冻除视觉编码器外的所有模块，在掩码分组任务上训练。可进一步在 RES/GRES 数据上微调以适配下游任务。

### 数据集构建

- **MaskGroups-2M**：从 MS-COCO/LVIS/VG/RES/GRES 数据集中自动构建 200 万掩码分组样本，覆盖类别/属性/位置/自由描述四类准则
- **MaskGroups-HQ**：人工标注 100,299 个高质量掩码组（96,697 训练 + 3,599 评估），28% 包含参考掩码

## 实验

### 主实验：ORES 任务（MaskGroups-HQ）

| 模型 | 文本提示 gIoU | 文本提示 cIoU | 混合提示 gIoU | 混合提示 cIoU | 总体 cIoU |
|------|:---:|:---:|:---:|:---:|:---:|
| ReLA | 34.93 | 43.22 | - | - | - |
| GSVA-13B | 41.98 | 49.55 | - | - | - |
| **RAS-13B (SAM)** | 55.82 | 60.12 | 35.91 | 37.77 | 53.93 |
| **RAS-13B (ORES-FT)** | **66.71** | **74.59** | **58.72** | **68.77** | **73.13** |

RAS 是唯一能处理视觉参考提示的方法。经 ORES 微调后，cIoU 从 53.93 跃升至 73.13。

### RES/GRES 基准

| 模型 | RefCOCO val | RefCOCO+ val | RefCOCOg val | 平均 |
|------|:---:|:---:|:---:|:---:|
| PSALM-1.3B | 83.6 | 72.9 | 73.8 | 77.1 |
| RAS-13B (RES-FT) | **81.0** | **75.1** | **76.0** | **77.8** |

在 GRES（gRefCOCO）上同样取得最优：总体 cIoU 71.79。

### 消融实验

| 解码范式 | cIoU | 推理延迟(s) |
|---------|:---:|:---:|
| 自回归 | 45.34 | 2.13 |
| **非自回归** | **53.75** | **0.56** |

| 视觉编码器 | 总体 cIoU |
|-----------|:---:|
| 仅 CLIP | 52.44 |
| 仅 DINOv2 | 47.71 |
| **四编码器集成** | **53.75** |

### 关键发现

1. 候选掩码质量分析：SAM/Co-DETR 的 Oracle cIoU 达 86-87，远超现有方法 77 的最终性能，说明候选掩码池质量极高
2. 非自回归解码比自回归提升 +8.4 cIoU 且推理速度快 3.8x
3. 四编码器集成比单编码器一致性地提升性能，ConvCLIP 在视觉参考任务上贡献最大

## 亮点与洞察

- **任务定义创新**：ORES 首次统一文本和视觉参考提示，输出掩码组而非单个掩码，贴近实际应用
- **解耦分割与理解**：利用分割基础模型提供高质量候选，用 LMM 做语义理解和选择，各取所长
- **非自回归解码设计精巧**：将集合预测转化为逐元素二分类，规避无序集合匹配难题

## 局限性

- 候选掩码质量依赖分割基础模型，若候选池未覆盖目标则无法恢复
- 基于 Vicuna-13B 的 LLM 参数量大，部署成本较高
- 不支持文本生成能力（如解释预测结果），限制了交互性

## 相关工作

- RES/GRES 方向：ReLA、GSVA、PSALM 等扩展多目标/零目标查询
- Grounding LMM：LISA、Groundhog、GLaMM 等实现像素级 grounding
- 分割基础模型：SAM 提供类别无关的高质量掩码提案

## 评分

- **新颖性**: ⭐⭐⭐⭐ — ORES 任务定义和非自回归掩码组解码均具开创价值
- **技术深度**: ⭐⭐⭐⭐ — 掩码 Token 化、多编码器集成和训练策略设计完整
- **实验**: ⭐⭐⭐⭐ — ORES/RES/GRES 多任务评估充分，消融到位
- **写作**: ⭐⭐⭐⭐ — 行文清晰，任务动机论证有力

<!-- RELATED:START -->

## 相关论文

- [HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation?](how_do_optical_flow_and_textual_prompts_collaborate_to_assist_in_audio-visual_se.md)
- [O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views](o-mama_learning_object_mask_matching_between_egocentric_and_exocentric_views.md)
- [ActionVOS: Actions as Prompts for Video Object Segmentation](../../ECCV2024/segmentation/actionvos_actions_as_prompts_for_video_object_segmentation.md)
- [LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation](lawdis_language-window-based_controllable_dichotomous_image_segmentati.md)

<!-- RELATED:END -->
