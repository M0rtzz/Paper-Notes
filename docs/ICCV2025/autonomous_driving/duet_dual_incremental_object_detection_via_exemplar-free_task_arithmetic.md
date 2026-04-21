---
title: >-
  [论文解读] DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic
description: >-
  [ICCV 2025][自动驾驶][增量目标检测] 提出 DuET 框架，首次以无样本（exemplar-free）的任务算术（Task Arithmetic）模型合并方式，同时解决目标检测中的类别增量和域增量问题（Dual Incremental Object Detection, DuIOD），并引入方向一致性损失（Directional Consistency Loss）缓解符号冲突，在 Pascal Series 和 Diverse Weather Series 上大幅超越现有方法。
tags:
  - ICCV 2025
  - 自动驾驶
  - 增量目标检测
  - 类别增量
  - 域增量
  - 任务算术
  - 模型合并
  - 灾难性遗忘
  - YOLO11
  - RT-DETR
---

# DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic

**会议**: ICCV 2025  
**arXiv**: [2506.21260](https://arxiv.org/abs/2506.21260)  
**代码**: 待确认  
**领域**: autonomous_driving / incremental_learning  
**关键词**: 增量目标检测, 类别增量, 域增量, 任务算术, 模型合并, 灾难性遗忘, YOLO11, RT-DETR

## 一句话总结

提出 DuET 框架，首次以无样本（exemplar-free）的任务算术（Task Arithmetic）模型合并方式，同时解决目标检测中的类别增量和域增量问题（Dual Incremental Object Detection, DuIOD），并引入方向一致性损失（Directional Consistency Loss）缓解符号冲突，在 Pascal Series 和 Diverse Weather Series 上大幅超越现有方法。

## 研究背景与动机

### 现实场景的双重挑战

真实目标检测系统（如自动驾驶、监控）需要持续学习新类别，同时适应环境变化（光照、天气、风格等域漂移）。现有方法只能处理其中一个维度：

- **类别增量目标检测（CIOD）**：逐步学习新类别，但假设域固定不变。在未见域上性能急剧下降。
- **域增量目标检测（DIOD）**：适应新域，但假设类别集合不变。无法检测新类别。

两类方法在同时面对类别和域漂移时都会失败，而这恰恰是真实场景最常遇到的情况。

### DuIOD 问题定义

作者提出 **Dual Incremental Object Detection（DuIOD）** 这一更实际的设定：模型需要在增量任务序列中，每个任务引入新类别 $\mathcal{C}_t$ 和新域 $\mathcal{D}_t$，且不保留任何历史训练数据（exemplar-free）。这带来了三个核心难题：

**灾难性遗忘**：学新忘旧

**域泛化**：旧类别在新域上的性能保持

**背景漂移**：旧类别在新任务中无标注，被当作背景训练

### 现有方法的不足

- CL-DETR（CIOD SOTA）：依赖 DETR 框架和 exemplar replay，在严重域漂移下表现差
- LDB（DIOD SOTA）：冻结基础模型学习域偏置，无法处理类别变化
- LwF / ERD：通用增量学习方法，在多阶段 DuIOD 中大幅退化

## 方法详解

### 整体框架

DuET 框架由两大核心组件构成：

1. **DuET Module**：通过逐层动态融合新旧任务向量来更新共享参数（backbone + neck），平衡知识保留与新知识吸收
2. **Incremental Head**：拼接新旧任务的检测头参数，扩展模型的类别检测能力

整体流程如下：

1. **基础任务 $\mathcal{T}_1$**：用预训练检测器在首个任务上微调，得到参数 $\theta_1$
2. **参数分解**：将模型参数分为共享参数 $\theta_s$（backbone + neck）和任务特定参数 $\theta_\tau$（检测头）
3. **增量任务 $\mathcal{T}_t, t \geq 2$**：
    - 顺序微调：以 $\theta_{t-1}$ 初始化，用总损失训练得到 $\theta_t$
    - 计算新旧任务向量：$\tau_{\text{old}} = \theta_{s_{t-1}} - \theta_{s_0}$, $\tau_{\text{curr}} = \theta_{s_t} - \theta_{s_0}$
    - DuET Module 合并共享参数
    - Incremental Head 拼接检测头参数
4. **推理**：使用合并后的增量权重进行检测

### 关键设计一：DuET Module — 逐层动态任务向量融合

DuET Module 是框架的核心，通过逐层计算保留因子 $\alpha_l$ 和适应因子 $\beta_l$ 来融合新旧任务向量。

对每一层 $l$，先计算 p-factor 衡量新旧更新的相对重要性：

$$p_l = \frac{\|\tau_{\text{old}}^l\| - \|\tau_{\text{curr}}^l\|}{\|\tau_{\text{old}}^l + \tau_{\text{curr}}^l\| + \epsilon}$$

经 $\tanh$ 映射和 clamp 后得到动态系数：

$$\alpha_l = \alpha_{\text{base}} + \text{clamp}(\gamma \cdot \tanh(p_l), -\gamma, \gamma), \quad \beta_l = 1 - \alpha_l$$

最终每层的共享参数更新为：

$$(\theta_{s_t}^l)_{\text{incre}} = \theta_{s_0}^l + \alpha_l \cdot \tau_{\text{old}}^l + \beta_l \cdot \tau_{\text{curr}}^l$$

**设计直觉**：当某层旧任务向量范数更大时，$\alpha_l$ 更大，优先保留旧知识（稳定性）；反之则更多吸收新知识（可塑性）。这避免了 Fisher Merging 等二阶方法的高计算开销。

### 关键设计二：Incremental Head — 任务特定参数拼接

检测头参数不做合并，而是直接拼接当前和历史的任务特定参数：

$$(\theta_{\tau_t})_{\text{incre}} = [\theta_{\tau_t}; (\theta_{\tau_{t-1}})_{\text{incre}}]$$

这使模型能同时输出所有已学类别的检测结果，是一种简洁高效的增量扩展策略。

### 关键设计三：检测器无关性

DuET 的参数分解策略具有通用性：
- **YOLO11**：backbone + neck 作为 $\theta_s$，检测头作为 $\theta_\tau$
- **RT-DETR**：同理适用
- **Deformable DETR**：同理适用

这使得 YOLO11、RT-DETR 等实时检测器首次能作为增量检测器使用。

### 损失函数

基础任务（$t=1$）仅使用标准检测损失 $\mathcal{L}_{\text{Detector}}$。

增量任务（$t \geq 2$）使用总损失：

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Detector}} + \lambda_{\text{Distill}} \mathcal{L}_{\text{Distill}}^* + \lambda_{\text{DC}} \mathcal{L}_{\text{DC}}$$

**方向一致性损失（Directional Consistency Loss）$\mathcal{L}_{\text{DC}}$**：

$$\mathcal{L}_{\text{DC}} = \sum_{i \in \theta_s} \text{ReLU}\left[-\left((\tau_{s_t}^{(i)} - \tau_{s_{t-1}}^{(i)}) \cdot (\tau_{s_{t-1}}^{(i)} - \tau_{s_{t-2}}^{(i)})\right)\right]$$

通过点积衡量连续增量更新的方向一致性：若当前更新方向与前一步相反（符号冲突），ReLU 会产生惩罚。这在模型合并阶段有效减少了约 34% 的符号冲突。

**修改的蒸馏损失 $\mathcal{L}_{\text{Distill}}^*$**：保持旧任务知识的标准蒸馏策略。

### 评估指标：Retention-Adaptability Index（RAI）

作者提出新评估指标，综合衡量保留能力和适应能力：

$$\text{RAI} = \frac{\text{Avg RI} + \text{Avg GI}}{2}$$

- **Avg RI（平均保留指数）**：最终模型在旧类别旧域上的 mAP 与初始学习时的比值，衡量遗忘程度
- **Avg GI（平均泛化指数）**：模型在未见类别上的 mAP 与参考模型的比值，衡量跨域泛化能力

## 实验关键数据

### 数据集

| 数据集系列 | 包含域 | 类别数 | 来源 |
|---|---|---|---|
| Pascal Series | VOC, Clipart, Watercolor, Comic | 3~20 | Pascal VOC, Cross-Domain Det |
| Diverse Weather Series | Daytime Sunny, Night Sunny, Daytime Foggy | 7 | BDD-100k, FoggyCityscapes, Adverse-Weather |

### 主实验：两阶段和多阶段结果（Table 2）

| 方法 | 基础检测器 | Pascal 2阶段 RAI | Pascal 4阶段 RAI | Weather 2阶段 RAI | Weather 3阶段 RAI |
|---|---|---|---|---|---|
| Sequential FT | YOLO11n | 6.81% | 5.53% | 22.94% | 15.26% |
| LwF | YOLO11n | 53.19% | 34.84% | 38.88% | 25.86% |
| ERD | YOLO11n | 56.17% | 47.95% | 59.92% | 42.00% |
| LDB | ViTDet | 42.83% | 52.83% | 11.76% | 27.96% |
| CL-DETR | Def. DETR | 54.51% | 54.18% | 57.09% | 53.86% |
| **DuET** | **YOLO11n** | **65.99%** | **65.95%** | **72.51%** | **65.25%** |

**关键发现**：
- DuET 在所有实验中均大幅领先，Pascal 多阶段 +13.12% RAI，Weather 多阶段 +11.39% RAI
- DuET 的 Avg RI 保持在 87~89%，遗忘极少
- DuET 参数量仅 2.58M，远小于 CL-DETR（39.85M）和 LDB（110.52M）

### 跨检测器泛化（Table 3）— Weather 两阶段

| 基础检测器 | 参数量 | GFLOPs | Avg RI | Avg GI | RAI |
|---|---|---|---|---|---|
| ViTDet | 110.52M | 1829.6 | 27.55% | 28.22% | 27.89% |
| Deformable DETR | 39.85M | 11.8 | 84.45% | 33.45% | 58.95% |
| RT-DETR-l | 32.00M | 103.4 | 47.73% | 21.00% | 34.37% |
| RT-DETR-x | 65.49M | 222.5 | 56.39% | 24.15% | 40.27% |
| **YOLO11n** | **2.58M** | **6.3** | **88.06%** | **56.95%** | **72.51%** |
| YOLO11x | 56.84M | 194.4 | 96.88% | 42.41% | 69.18% |

**关键发现**：YOLO11n 以最小参数量和计算量取得最佳 RAI，说明 DuET 的任务算术策略与轻量级检测器高度兼容。

### 消融实验（Table 4）— Pascal 两阶段，YOLO11n

| 配置 | Avg RI | Avg GI | RAI |
|---|---|---|---|
| 无增量（静态） | 0.5% | 9.13% | 4.82% |
| + 顺序微调 | 0.75% | 12.86% | 6.81% |
| + Incremental Head | 24.75% | 33.36% | 29.06% |
| + DuET Module | 75.00% | 37.26% | 56.13% |
| + $\mathcal{L}_{\text{Distill}}^*$ | 87.06% | 37.75% | 62.41% |
| + $\mathcal{L}_{\text{DC}}$（完整DuET） | **87.44%** | **44.54%** | **65.99%** |

**关键发现**：
- Incremental Head 贡献最大的 RAI 跳跃（+22.25%）
- DuET Module 进一步大幅提升 Avg RI（24.75% → 75.00%）
- $\mathcal{L}_{\text{DC}}$ 特别提升 Avg GI（+6.79%），有效改善泛化
- 每个组件都不可或缺

## 亮点与洞察

1. **问题定义有价值**：DuIOD 比单纯的 CIOD 或 DIOD 更贴合真实场景，是一个重要的新研究方向
2. **任务算术在检测中的创新应用**：首次将 Task Arithmetic 引入增量目标检测，且验证了其检测器无关性
3. **方向一致性损失巧妙**：通过点积约束连续更新方向的一致性来缓解符号冲突，简单有效，平均减少 34% 符号冲突
4. **轻量高效**：YOLO11n 仅 2.58M 参数 + 6.3 GFLOPs 即可作为实时增量检测器，实用性极强
5. **评估指标设计合理**：RAI 同时衡量保留和泛化，比现有纯遗忘指标更全面
6. **参数高效**：不需要 exemplar buffer，不需要生成式回放，只需保存任务向量和共享参数基准

## 局限性

1. **类别-域绑定假设**：每个增量任务的新类别和新域是绑定出现的，现实中可能有更复杂的组合情况（同一域新类别、同一类别跨多域等）
2. **需要保存基准权重 $\theta_{s_0}$**：任务向量的计算依赖初始预训练权重，存储开销随层数线性增长
3. **ViTDet 上效果差**：RAI 仅 27.89%，说明 DuET 的逐层融合策略可能不适合所有架构
4. **Avg GI 整体偏低**：即使是 DuET，Avg GI 最高也只有 56.95%，跨域泛化仍有很大提升空间
5. **仅验证 2~4 阶段**：更长的增量序列（10+ 任务）中的表现未知
6. **$\mathcal{L}_{\text{DC}}$ 需要至少 3 个任务**：方向一致性需要对比连续三步的更新，在两阶段实验中作用有限

## 相关工作与启发

- **Task Arithmetic [Ilharco et al., 2023]**：通过对任务向量做算术运算修改预训练模型——DuET 将此范式扩展到增量检测
- **TIES-Merging [Yadav et al., 2023]**：通过正交约束解决符号冲突——启发了 DuET 的 $\mathcal{L}_{\text{DC}}$
- **MagMax [Marczak et al.]**：通过重要参数选择缓解遗忘——DuET 的逐层 p-factor 是更动态的替代方案
- **CL-DETR [Liu et al., 2023]**：CIOD SOTA，使用知识蒸馏 + exemplar replay——DuET 在无 exemplar 条件下超越它
- **LDB [Chen et al., 2024]**：DIOD SOTA，学习域偏置——无法处理 DuIOD 中的类别变化

**启发**：任务算术的"向量空间"视角为增量学习提供了新思路——将新旧任务的模型权重差异视为方向向量，通过简单的线性组合就能有效平衡稳定性和可塑性，避免了复杂的正则化或回放策略。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|---|---|---|
| 创新性 | ⭐⭐⭐⭐ | 首次定义 DuIOD 问题 + 首次将 Task Arithmetic 引入增量检测 |
| 技术深度 | ⭐⭐⭐⭐ | 逐层动态融合 + DC Loss + 完整的评估体系 |
| 实验充分度 | ⭐⭐⭐⭐ | 7 个实验 + 6 种检测器 + 详尽消融 |
| 实用价值 | ⭐⭐⭐⭐⭐ | 检测器无关 + 轻量 + 无需 exemplar，实用性极强 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，图示信息量大 |
| 综合评分 | ⭐⭐⭐⭐ | 问题定义好、方法简洁有效、实验扎实 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception](maestro_task-relevant_optimization_via_adaptive_feature_enhancement_and_suppress.md)
- [IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](igl-nav_incremental_3d_gaussian_localization_for_image-goal_navigation.md)
- [EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)

<!-- RELATED:END -->
