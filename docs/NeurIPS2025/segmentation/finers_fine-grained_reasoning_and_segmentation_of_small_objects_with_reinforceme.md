---
title: >-
  [论文解读] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning
description: >-
  [NeurIPS 2025][语义分割][图像分割] 提出 FineRS 两阶段 MLLM 强化学习框架（全局语义探索 GSE → 局部感知精化 LPR），通过 locate-informed retrospective reward 耦合两阶段…
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "图像分割"
  - "MLLM"
  - "reinforcement-learning"
  - "GRPO"
  - "coarse-to-fine"
  - "high-resolution"
  - "UAV"
---

# FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.21311](https://arxiv.org/abs/2510.21311)  
**代码**: [https://iiau-zhanglu.github.io/FINERS/](https://iiau-zhanglu.github.io/FINERS/)  
**领域**: 图像分割  
**关键词**: small object segmentation, MLLM, reinforcement-learning, GRPO, coarse-to-fine, high-resolution, UAV

## 一句话总结

提出 FineRS 两阶段 MLLM 强化学习框架（全局语义探索 GSE → 局部感知精化 LPR），通过 locate-informed retrospective reward 耦合两阶段，在自建 FineRS-4k UAV 高分辨率数据集上实现超小目标的推理与分割，gIoU 达 55.1%（超 Seg-Zero† 8.5%），同时支持 VQA（MVQA 83.3%）。

## 研究背景与动机

**MLLM 分割的现状**：LISA 等方法将 MLLM 与 SAM 结合实现了推理分割，但它们针对标准分辨率和大尺度目标设计，对高分辨率图像中的超小目标（面积占比 <0.1%）严重失效——LISA 7B 在 FineRS-4k 上 gIoU 仅 9.0%。

**高分辨率理解的局限**：现有高分辨率方法（SEAL、DC2、MLLMs-Know）通过分块或注意力操作来感知细节，但多数为 training-free 流程，缺乏精确定位能力，无法输出 pixel-level mask。

**Vision RFT 的不足**：Seg-Zero 将 GRPO 引入分割，但受限于分辨率、不支持多任务（VQA + 分割统一）、奖励设计不够灵活。

**核心动机**：需要一个能处理 4K 图像中超小目标的统一框架，同时支持指令引导分割、开放式 VQA 和选择题 VQA，并通过强化学习实现数据高效训练。

## 方法详解

### 整体框架：两阶段 Coarse-to-Fine

FineRS 基于 Qwen2.5-VL-7B 构建，采用两阶段流水线：

**阶段一：Global Semantic Exploration (GSE)**

- 输入：1920×1080 高分辨率图像 + 用户指令
- 输出：文本回答 $A^{pre}$ + 粗定位区域 $B_r^{pre}$（固定 256×256 大小，只优化中心偏移）
- 作用：在全局视野中理解指令语义，输出目标的大致位置

**阶段二：Localized Perceptual Refinement (LPR)**

- 输入：根据 GSE 输出裁剪的 512×512 局部图像 + 指令
- 输出：精确 bounding box $B^{pre}$ + 两个点 $P_1^{pre}, P_2^{pre}$
- 最终将 box 和 points 送入冻结的 SAM2 生成分割 mask

### 训练流程：GRPO 强化学习

训练采用 GRPO 算法分两步进行，不依赖大规模有监督数据：

1. **先训练 LPR**：在随机裁剪的局部图像上学习精确定位，使用 Box IoU、Box L1、Point L1、JSON 格式、Think 格式、QA 准确率共 6 种奖励
2. **再训练 GSE**：利用已训练好的 LPR 来选择最优粗区域作为伪 GT，结合 Region IoU、Region L1、区域大小、Box-in-region、格式和 QA 准确率共 7 种奖励

### 核心创新：Locate-informed Retrospective Reward

- **问题**：GSE 输出的粗区域没有显式 GT 标注
- **解决**：对每个样本生成 $n$ 个随机偏移的候选粗区域（均覆盖 GT box），然后用已训练的 LPR 对每个区域做预测，选择 LPR 预测 IoU 最高的区域作为 GSE 的伪 GT $B_r^{gt}$
- **效果**：LPR 的定位能力反向指导 GSE 的探索行为，形成闭环优化

### 多任务统一的 Response Reward

通过 $R_{response}$ 统一三种任务：

- **指令引导分割**：回答包含 "is detected/found" 即得分 1
- **选择题 VQA**：精确匹配选项
- **开放式 VQA**：模糊匹配相似度 >0.8

## FineRS-4k 数据集

- **来源**：YouTube 和自采无人机视频，3840×2060 分辨率
- **规模**：4,563 张高分辨率图像，8,411 个小目标实体，12,132 个 text-mask 标注对
- **划分**：训练 8,956 / 验证 749 / 测试 2,427
- **目标尺度**：Small (>0.055%)、Extra Small (0.017%–0.055%)、Extra-Extra Small (<0.017%)
- **任务类型**：指令引导分割 39%、选择题 VQA 30.5%、开放式 VQA 30.5%
- **属性维度**：颜色、形状、位置、其他
- **标注流程**：14 名志愿者两两交叉检查 + 4 名高级审核员最终质检

与现有数据集对比：V* 仅 191 样本且无 mask，HR-Bench 仅 200 样本无 mask；refCOCOg 和 ReasonSeg 为标准分辨率；FineRS-4k 是首个同时提供 QA + mask 的高分辨率小目标数据集。

## 实验结果

### 主实验：FineRS-4k 测试集

| 方法 | gIoU | cIoU | MVQA | OVQA |
|------|------|------|------|------|
| LISA 7B（training-free） | 9.0 | 2.4 | 0.0 | 5.5 |
| LISA++ 7B | 12.3 | 5.2 | 6.7 | 8.2 |
| MLLMs-Know 13B + LISA 13B | 17.9 | 12.8 | 52.6 | 48.8 |
| Seg-Zero 7B（training-free） | 32.1 | 6.6 | – | – |
| LISA† 7B（重训练） | 12.1 | 9.6 | 23.9 | 22.0 |
| Seg-Zero† 7B（重训练） | 46.6 | 38.6 | – | – |
| **FineRS 7B** | **55.1** | **46.5** | **83.3** | **56.7** |

关键观察：

- FineRS 在 gIoU 上超过重训练的 Seg-Zero† 8.5 个点，cIoU 超 7.9 个点
- FineRS 是唯一同时在分割和 VQA 上都有强表现的方法
- 在 XXS（超超小）目标上优势最大：gIoU 47.2 vs Seg-Zero† 31.7（+15.5）

### 跨数据集泛化（无额外微调）

| 数据集 | FineRS | 次优方法 |
|--------|--------|----------|
| V* Overall | **77.5** | SEAL 75.4 |
| HR-Bench 4K Avg | **63.8** | DC2 50.0 |
| HR-Bench 8K Avg | **58.1** | DC2 40.8 |

在非 UAV 场景下同样大幅领先，证明方法的泛化能力。

### 消融实验

| 设置 | gIoU | cIoU | MVQA | OVQA |
|------|------|------|------|------|
| 完整 FineRS | 55.1 | 46.5 | 83.3 | 56.7 |
| 去掉 Retrospective Reward | 54.0 | 44.0 | 82.3 | 53.0 |
| 去掉 LPR 随机区域增强 | 53.9 | 46.7 | 83.7 | 61.9 |
| 去掉 QA Acc. Reward | 52.8 | 45.7 | – | – |
| 去掉 Box Size Reward | 51.0 | 43.4 | 56.5 | 40.8 |
| 去掉 Box-in-region Reward | 50.1 | 42.0 | 56.5 | 40.8 |

- Retrospective reward 对 cIoU 影响最大（+2.5），验证了跨阶段闭环优化的有效性
- Box Size 和 Box-in-region reward 对 VQA 影响巨大（去掉后 MVQA 从 83.3→56.5），说明区域约束对多任务统一至关重要

## 优缺点分析

**优点**：

1. 两阶段 coarse-to-fine 设计合理绕过了 MLLM 分辨率瓶颈，让 7B 模型也能处理 4K 小目标
2. Retrospective reward 设计巧妙，无需额外标注即可为粗区域提供有效监督
3. 统一推理与分割，一个框架同时输出文本回答和 mask
4. 数据高效——仅需约 9K 训练样本即可超越大规模 SFT 方法

**不足**：

1. 两阶段串行流水线增加推理延迟，论文未报告推理速度
2. GSE 粗区域大小固定（256×256），难以适应极端尺度变化
3. 数据集仅关注 UAV 俯拍场景，其他类型高分辨率图像（如医学影像）的适用性未验证

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将两阶段 coarse-to-fine 与 GRPO 强化学习结合，retrospective reward 的跨阶段闭环设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 自建数据集 + 多个公开基准 + 全面消融，但缺少推理效率分析
- 写作质量: ⭐⭐⭐⭐ — 方法描述清晰，图表得当
- 价值: ⭐⭐⭐⭐⭐ — 填补了 MLLM 在高分辨率小目标分割领域的空白，数据集和方法均有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning](sam-r1_leveraging_sam_for_reward_feedback_in_multimodal_segmentation_via_reinfor.md)
- [\[NeurIPS 2025\] GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset](gtpbd_a_fine-grained_global_terraced_parcel_and_boundary_dataset.md)
- [\[NeurIPS 2025\] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding](partnext_a_next-generation_dataset_for_fine-grained_and_hierarchical_3d_part_und.md)
- [\[CVPR 2025\] Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation](../../CVPR2025/segmentation/fine-grained_image-text_correspondence_with_cost_aggregation_for_open-vocabulary.md)
- [\[CVPR 2026\] Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](../../CVPR2026/segmentation/boundary_segment_action_segmentation.md)

</div>

<!-- RELATED:END -->
