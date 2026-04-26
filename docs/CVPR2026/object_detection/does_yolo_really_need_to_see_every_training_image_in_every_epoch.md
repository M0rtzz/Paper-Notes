---
title: >-
  [论文解读] Does YOLO Really Need to See Every Training Image in Every Epoch?
description: >-
  [CVPR 2026][目标检测][YOLO] 提出 Anti-Forgetting Sampling Strategy (AFSS)，根据每张训练图像的学习充分度（min(Precision, Recall)）动态决定哪些图像参与训练、哪些可以跳过，实现 YOLO 系列检测器 1.43× 以上的训练加速同时保持甚至提升检测精度。
tags:
  - CVPR 2026
  - 目标检测
  - YOLO
  - 训练加速
  - 自适应采样
  - 抗遗忘
  - 数据高效学习
---

# Does YOLO Really Need to See Every Training Image in Every Epoch?

**会议**: CVPR 2026  
**arXiv**: [2603.17684](https://arxiv.org/abs/2603.17684)  
**代码**: 无  
**领域**: 目标检测  
**关键词**: YOLO, 训练加速, 自适应采样, 抗遗忘, 数据高效学习

## 一句话总结

提出 Anti-Forgetting Sampling Strategy (AFSS)，根据每张训练图像的学习充分度（min(Precision, Recall)）动态决定哪些图像参与训练、哪些可以跳过，实现 YOLO 系列检测器 1.43× 以上的训练加速同时保持甚至提升检测精度。

## 研究背景与动机

YOLO 系列以极快推理速度著称（YOLO11s 达 200 FPS），但训练却出奇耗时：
- YOLO11s 在 COCO 上训练需 43.9 小时（双 RTX 4090），而同硬件下 Faster R-CNN 仅需 6.5 小时
- 原因在于 YOLO 采用**全覆盖训练范式**：每个 epoch 遍历全部训练集，每张图像处理数百次
- 当模型已充分学习某些图像后，继续以相同频率处理它们产生递减收益

**核心问题**：YOLO 真的需要在每个 epoch 看到每张训练图像吗？如果不需要，能否通过动态选择"看什么"和"何时看"来加速训练？

现有替代方案的局限：
- **课程学习**：固定由易到难顺序，困难样本学习不充分
- **数据集剪枝**：不可逆删除导致遗忘和偏差
- **数据集蒸馏**：合成数据缺乏真实世界多样性

## 方法详解

### 整体框架

AFSS 在每个 epoch 开始前，根据上一轮记录的状态字典评估每张训练图像的学习充分度，将其分为 Easy / Moderate / Hard 三个级别，分别采用不同的采样策略：
- **Easy**：仅 2% 参与，周期性复习防遗忘
- **Moderate**：40% 参与，保证短期全覆盖
- **Hard**：100% 参与，确保充分学习

### 关键设计

#### 1. 学习充分度度量（Learning Sufficiency Metric）

定义为检测精度和召回率的最小值：

$$\text{Learning Sufficiency for } \mathbf{I}_i = \min(P_i, R_i)$$

这一设计强调短板维度——无论分类还是定位不可靠，都视为学习不足。基于此分层：

| 级别 | 条件 | 含义 |
|------|------|------|
| Easy | $\min(P_i, R_i) > 0.85$ | 模型已自信处理 |
| Moderate | $0.55 \leq \min(P_i, R_i) \leq 0.85$ | 部分稳定，需继续优化 |
| Hard | $\min(P_i, R_i) < 0.55$ | 仍具挑战（遮挡、小目标等） |

#### 2. 连续复习（Continuous Review）—— 针对 Easy 图像

每个 epoch 仅 2% 的 Easy 图像参与训练，通过两阶段采样：

**强制复习集 $\mathcal{A}_f'$**：从超过 10 个 epoch 未使用的 Easy 图像中采样 $E_1$ 张，防止长期未见导致的遗忘

**随机多样性集 $\mathcal{A}_r$**：从剩余 Easy 图像中随机采样 $E_2$ 张，提供轻量变化

约束：$E_1 + E_2 = 0.02 \times |\mathcal{D}_{t-1}^1|$，$E_1 \leq 0.5 \times (E_1 + E_2)$

#### 3. 短期覆盖（Short-Term Coverage）—— 针对 Moderate 图像

每 epoch 使用 40% 的 Moderate 图像，确保每张图像在 3 个 epoch 内至少出现一次：

**强制覆盖集 $\mathcal{B}_f$**：连续 3 个 epoch 未使用的图像优先纳入

$$\mathcal{B}_f = \{(\mathbf{I}_i, P_i, R_i, ep_i) \in \mathcal{D}_{t-1}^2 \mid t - 1 - ep_i \geq 3\}$$

**随机补充集 $\mathcal{B}_r$**：从剩余 Moderate 图像中随机采样至 40% 总量

#### 4. 状态更新（State Update）

训练后更新状态字典 $\mathcal{D}_t$，记录每张图像的精度、召回率和最后使用 epoch：
- 精度/召回率每 5 个 epoch 刷新一次（减少冗余评估）
- 使用记录实时更新

最终每 epoch 使用的训练集：$\Omega = (\mathcal{A}_f' \cup \mathcal{A}_r) \cup (\mathcal{B}_f \cup \mathcal{B}_r) \cup \mathcal{D}_{t-1}^3$，$|\Omega| < K$（总图像数）

### 损失函数 / 训练策略

AFSS 是**架构无关**的纯采样策略，不修改 YOLO 的损失函数或模型结构：
- 直接应用于 Ultralytics YOLO 框架
- 600 epochs（COCO）/ 300 epochs（VOC/DOTA/DIOR-R）
- 默认 batch size 64，分辨率 640×640（COCO/VOC）

## 实验关键数据

### 主实验

**表1：MS COCO 2017 + PASCAL VOC 2007 训练加速**

| 模型 | COCO AP | 加速比 | VOC mAP | 加速比 |
|------|---------|--------|---------|--------|
| YOLO11s | 47.0 | — | 81.7 | — |
| YOLO11s + AFSS | **47.2** | **1.54×** | **81.8** | **1.64×** |
| YOLO12x | 55.2 | — | 86.2 | — |
| YOLO12x + AFSS | **55.4** | **1.68×** | **86.4** | **1.69×** |

**表2：遥感检测 DOTA-v1.0 + DIOR-R**

| 模型 | DOTA mAP | 加速比 | DIOR-R mAP | 加速比 |
|------|----------|--------|-----------|--------|
| YOLO11x-OBB | 81.3 | — | 83.6 | — |
| YOLO11x-OBB + AFSS | **81.4** | **1.69×** | **83.7** | **1.70×** |

跨 YOLOv8/v10/11/12 全部规模（n/s/m/l/x）一致有效。

### 消融实验

**表3：与其他训练策略对比（YOLO11s on COCO）**

| 方法 | AP | 加速比 |
|------|-----|--------|
| 课程学习 | 43.7 | 1.35× |
| 自步学习 | 44.5 | 1.30× |
| 数据剪枝 | 40.5 | 1.38× |
| 数据蒸馏 | 35.6 | 1.50× |
| **AFSS** | **47.2** | **1.54×** |

**表4：模块贡献消融**

| LSM | CR | STC | SU | AP | 加速比 |
|-----|----|----|-----|------|--------|
| ✓ | | | | 44.8 | 1.45× |
| ✓ | ✓ | | | 45.5 | 1.34× |
| ✓ | | ✓ | | 46.6 | 1.31× |
| ✓ | ✓ | ✓ | | 47.2 | 1.26× |
| ✓ | ✓ | ✓ | ✓ | **47.2** | **1.54×** |

State Update 是加速的关键——无 SU 时虽精度达标但加速仅 1.26×。

**表5：学习充分度度量对比**

| 度量 | AP | 加速比 |
|------|-----|--------|
| Loss-based | 46.0 | 1.52× |
| Gradient-based | 46.9 | 1.45× |
| F1 score | 46.6 | 1.51× |
| **min(Prec, Rec)** | **47.2** | **1.54×** |

### 关键发现

1. **模型越大加速越显著**：从 n（1.43×）到 x（1.68×），大模型学习能力更强，更多图像更快变"Easy"
2. **连续复习间隔 10 epoch 最优**：过短（5）浪费计算，过长（20）导致遗忘（AP 降至 44.8）
3. **短期覆盖间隔 3 epoch 最优**：间隔 5 时 AP 降至 44.2
4. **状态更新间隔 5 epoch 最优**：每 epoch 更新计算开销大（加速仅 1.26×），15 epoch 更新则信息过时
5. 训练过程中 Hard 图像数量持续减少，Easy/Moderate 增加，体现模型学习进度

## 亮点与洞察

1. **洞察深刻**：揭示了 YOLO "只看一次"推理哲学与"反复看"训练范式的矛盾
2. **设计简洁务实**：不修改模型架构和损失函数，仅改变数据采样策略，即插即用
3. **抗遗忘机制设计精巧**：三级分层 + 强制复习 + 短期覆盖，在加速与防遗忘间取得出色平衡
4. **实验规模空前**：覆盖 4 个 YOLO 版本（v8/v10/11/12）× 5 个规模（n/s/m/l/x）× 4 个数据集

## 局限与展望

1. 学习充分度评估需要额外推理计算（虽每 5 epoch 一次），对极大数据集可能是瓶颈
2. Easy/Moderate/Hard 的阈值（0.85/0.55）和采样比例（2%/40%/100%）为手动设定，未做自适应
3. 仅在 YOLO 系列上验证，对 DETR 等基于 Transformer 的检测器适用性未知
4. 未考虑图像间的相关性和互补性（如场景多样性），仅基于单图像的学习充分度

## 相关工作与启发

- **课程学习/自步学习**：固定易到难顺序，困难样本学习不足；AFSS 始终保留困难样本
- **数据集剪枝**（Deep Learning on a Data Diet）：静态不可逆删除导致遗忘；AFSS 动态可逆
- **数据集蒸馏**（Fetch and Forge）：合成数据多样性不足；AFSS 使用真实数据子集
- AFSS 的思想可推广到其他需要长时间训练的任务（如分割、实例分割、姿态估计）

## 评分

- **新颖性**: ★★★★☆ — 问题洞察新颖，方法设计简洁有效
- **技术深度**: ★★★☆☆ — 方法本身偏工程化，理论深度有限
- **实验充分性**: ★★★★★ — 覆盖面极广，消融详尽，对比全面
- **写作清晰度**: ★★★★★ — 标题引人入胜，叙事流畅，图表直观

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Does FLUX Already Know How to Perform Physically Plausible Image Composition?](../../ICLR2026/object_detection/does_flux_already_know_how_to_perform_physically_plausible_image_composition.md)
- [\[ICCV 2025\] YOLO-Count: Differentiable Object Counting for Text-to-Image Generation](../../ICCV2025/object_detection/yolo-count_differentiable_object_counting_for_text-to-image_generation.md)
- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](../../AAAI2026/object_detection/yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[CVPR 2026\] Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods](training-free_detection_of_generated_videos_via_spatial-temporal_likelihoods.md)
- [\[CVPR 2026\] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training](pet-dino_unifying_visual_cues_into_grounding_dino_with_prompt-enriched_training.md)

<!-- RELATED:END -->
