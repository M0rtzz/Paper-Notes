---
title: >-
  [论文解读] UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions
description: >-
  [ICCV 2025][视频理解][visual object tracking] UMDATrack 提出了首个统一多域自适应跟踪框架，利用文本引导扩散模型合成少量（<2% 帧）多天气条件无标注视频，通过域定制适配器（DCA）高效迁移目标表征到不同天气域，并引入基于最优传输的目标感知置信度对齐（TCA）增强跨域定位一致性，在夜间/雾天/雨天等场景中大幅超越现有 SOTA 跟踪器。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "visual object tracking"
  - "域适应"
  - "adverse weather"
  - "domain-customized adapter"
  - "optimal transport"
  - "扩散模型"
  - "teacher-student"
---

# UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions

**会议**: ICCV 2025  
**arXiv**: [2507.00648](https://arxiv.org/abs/2507.00648)  
**代码**: [https://github.com/Z-Z188/UMDATrack](https://github.com/Z-Z188/UMDATrack)  
**领域**: 目标跟踪 / 域自适应 / 恶劣天气  
**关键词**: visual object tracking, multi-domain adaptation, adverse weather, domain-customized adapter, optimal transport, text-to-image diffusion, teacher-student

## 一句话总结
UMDATrack 提出了首个统一多域自适应跟踪框架，利用文本引导扩散模型合成少量（<2% 帧）多天气条件无标注视频，通过域定制适配器（DCA）高效迁移目标表征到不同天气域，并引入基于最优传输的目标感知置信度对齐（TCA）增强跨域定位一致性，在夜间/雾天/雨天等场景中大幅超越现有 SOTA 跟踪器。

## 研究背景与动机

视觉目标跟踪（VOT）在良好光照日间场景已取得卓越进展，但在恶劣天气条件下（夜间、雾天、雨天等）面临严重性能退化：

**巨大的域偏移**：现有主流跟踪器（OSTrack、ARTrackV2 等）在 LaSOT/TrackingNet 等日间数据集上训练，迁移到恶劣天气时因外观分布差异显著下降。

**单域方法的局限**：现有跨域跟踪器（如 UDAT）仅针对单一天气条件设计。例如 UDAT 专为夜间优化，进入雾天时性能反而急剧下降，缺乏多条件泛化能力。

**数据合成成本高**：现有方法需要生成大量目标域样本进行知识迁移，过程耗时且各域独立处理，忽略了目标对象在不同域间的内在关联。

**冗余参数问题**：对不同天气条件分别引入独立的特征对齐参数，无法高效进行跨域交互。

核心动机：能否设计一个统一框架，用极少量合成数据覆盖多种天气条件，通过轻量级适配器实现高效多域迁移？

## 方法详解

### 整体架构概览
UMDATrack 由三个核心组件构成：可控场景生成器（CSG）、带域定制适配器（DCA）的编码器网络、带目标感知置信度对齐（TCA）的定位头，整体遵循 teacher-student 训练范式。

### 组件一：可控场景生成器 (CSG)

- 利用 Stable Diffusion-Turbo 将日间域视频帧翻译为不同天气条件。
- 输入：源域视频帧 x + 文本提示 c_X（如 "Car in the night/haze/rain/snow"）。
- 输出：目标域帧 y = G_SDT(x, c_X, ε)，通过 skip connections 和 Zero-Convs 保留结构细节。
- **高效合成**：仅需 1-4 步迭代即可快速翻译，简单改变文本提示即可切换天气条件。
- **数据量极小**：仅从 GOT-10k 合成，目标域帧数不到源域的 2%。

### 组件二：域定制适配器 (DCA)

DCA 用于将目标对象表征从源域高效迁移到多个目标域，无需重复训练整个 backbone：

1. **冻结 backbone**：预训练 ViT-Base 保持不变，仅训练 DCA 模块。
2. **结构设计**：
    - 用轻量 ResNet 块将目标域搜索图像 X^T 转换为查询 Q ∈ R^{K×C}。
    - 初始化高斯随机变量作为可学习 token bank B ∈ R^{L'×C}。
    - Token bank 通过两个 FC 层投影为 Key 和 Value。
    - 计算结构化 token S = Softmax(QK^T/√d_k)V，编码目标域的潜在图像内容表示。
3. **注入方式**：结构化 token S 被送入冻结的 ViT，与源域模板-搜索 token 拼接，使模型快速找到各天气条件下的最优收敛点。
4. **训练效率**：每个天气条件只需额外训练 DCA 50 epochs，无需重复 backbone 训练阶段。所有天气条件总训练时间仅需一天半。

### 组件三：目标感知置信度对齐 (TCA)

基于最优传输 (OT) 理论对齐源域和目标域的定位置信度分布：

1. **伪标签传播**：Teacher 网络为目标域生成伪标签，Student 网络据此更新。
2. **问题**：伪标签可能有噪声，错误标签会误导目标状态预测。
3. **OT 代价矩阵设计**：同时考虑空间和置信度差异：
    - 置信度代价 C^Conf：衡量学生和教师在最高置信位置的响应值差异。
    - 位置代价 C^Pos：衡量学生和教师最高置信位置的空间偏移。
    - 总代价 C = C^Conf + C^Pos。
4. **位置敏感最优传输损失 (PSOT)**：用 Sinkhorn 距离算法求解 OT 问题的对偶形式，最小化将源域置信度分布传输到目标域的代价。
5. **联合损失**：L = L_t + λL_p，其中 L_t = L_cls + βL_1 + γL_GIoU（分类 + L1 回归 + GIoU）。

### 训练策略
- **两阶段训练**：
    - 阶段一（Backbone 训练，250 epochs）：不引入 DCA，用目标监督损失 + PSOT 损失进行域自适应。4 个源域数据集 + 3 个合成数据集（比例 1:1:1:1:4:4:4）。
    - 阶段二（DCA 训练，50 epochs）：冻结 backbone，仅训练 DCA 模块。
- 推理时模板特征初始化后缓存，后续帧直接复用。

## 实验关键数据

### 合成数据集实验

| 跟踪器 | GOT-10k-Foggy AO | DTB70-Foggy AUC/P | GOT-10k-Dark AO | DTB70-Dark AUC/P | GOT-10k-Rainy AO | DTB70-Rainy AUC/P |
|---------|---:|---:|---:|---:|---:|---:|
| **UMDATrack** | **66.6** | **66.21/86.05** | **65.4** | **66.07/85.72** | **68.5** | **66.75/87.60** |
| DCPT | 61.6 | 58.31/75.33 | 62.4 | 61.87/80.11 | 62.3 | 61.68/82.56 |
| UDAT-CAR | 51.5 | 50.21/69.41 | 56.8 | 57.20/75.80 | 59.5 | 56.42/75.36 |
| ARTrackV2 | 64.8 | 62.25/80.15 | 63.1 | 62.87/80.56 | 66.2 | 63.84/83.32 |

在所有三种天气条件的合成数据集上全面领先。DTB70-Dark 上超第二名 3.06% AUC / 4.15% Precision。

### 真实世界数据集

| 跟踪器 | NAT2021 AUC/P | UAVDark70 AUC/P | AVisT AUC/P |
|---------|---:|---:|---:|
| **UMDATrack** | **54.58/70.78** | **60.05/73.35** | **60.50/59.01** |
| DCPT | 52.55/69.01 | 56.86/70.16 | 55.66/52.41 |
| UDAT-CAR | 48.75/65.96 | 51.25/70.22 | 38.91/33.65 |

在真实世界夜间和混合恶劣天气数据集上同样取得 SOTA。

### 效率对比
- 推理速度：UMDATrack 达到最高推理速度，且参数量和计算量保持较低。
- 训练时间：所有天气条件总计仅需一天半（单 backbone 训练 + 各域 DCA 分别 50 epochs）。

## 亮点与洞察

- **首个统一多域自适应跟踪器**：一个框架覆盖夜间/雾天/雨天/雪天等多种恶劣天气，无需为每种条件单独设计模型。
- **极少量合成数据的高效迁移**：仅需源域不到 2% 的合成帧，大幅降低数据收集和标注成本。
- **DCA 的精巧设计**：冻结 backbone + 可学习 token bank + 结构化注意力，在几乎不增加参数的前提下实现灵活的多域适应。简单改变 text prompt 即可扩展到新天气条件。
- **OT 理论的创新应用**：将最优传输用于跟踪任务中的跨域定位对齐，同时考虑空间和置信度两个维度，比简单的 KL 散度约束更全面。
- **Teacher-Student 范式与 DCA 的结合**：DCA 生成的结构化 token 注入冻结 backbone，EMA 更新教师模型，形成渐进式的域知识传递闭环。

## 局限与展望

- CSG 依赖 Stable Diffusion-Turbo 的质量，对某些天气条件（如暴雨+强风的复合场景）合成效果可能不够真实。
- 文本提示模板较简单（"Car in the night"），可能无法覆盖复杂的真实天气变化。
- DCA 每种天气条件仍需独立训练 50 epochs，若天气类型极多时仍有一定训练开销。
- 实验仅覆盖了雾/夜/雨/雪四种条件，对极端场景（如沙尘暴、积水反射等）未验证。
- 基于 ViT-Base 的 backbone 选择可能限制了表征能力，未探索更大模型。
- PSOT 损失中 λ 等超参数的敏感性分析不够充分。

## 相关工作与启发

- **vs. UDAT**：UDAT 仅针对夜间，用 Transformer bridging layer 进行单域知识迁移。UMDATrack 用统一 DCA + CSG 覆盖多域，且数据需求更少。
- **vs. DCPT**：DCPT 也做跨域跟踪但各域独立处理。UMDATrack 通过共享 backbone + 域特定 DCA 实现参数高效的多域适应。
- **vs. SAM-DA**：SAM-DA 依赖增强来统一表征，本质上是 "Track-by-Enhancement" 范式，增强质量会瓶颈跟踪性能。UMDATrack 直接在特征空间做域适应，更加端到端。
- **ControlNet 的启发**：CSG 中使用 Zero-Convs 保留结构细节的思想源自 ControlNet，证明了可控生成在跟踪数据增广中的价值。
- 多目标域适应（MTDA）在检测和分类中已有探索，本文首次将其引入跟踪任务，启发未来更多 MTDA 在细粒度视觉任务中的应用。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个统一多域自适应跟踪框架，DCA 设计简洁有效，OT 用于跟踪定位对齐是亮点
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖合成和真实数据集共 8 个 benchmark，三种天气条件全面评估，对比 14+ 个基线
- 写作质量: ⭐⭐⭐⭐ 架构图清晰，问题动机论述充分，三种跟踪范式对比图直观
- 价值: ⭐⭐⭐⭐ 解决了恶劣天气跟踪领域的实际痛点，统一框架设计对后续工作有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Lifelong Domain Adaptive 3D Human Pose Estimation](../../AAAI2026/video_understanding/lifelong_domain_adaptive_3d_human_pose_estimation.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](aim_adaptive_inference_of_multi-modal_llms_via_token_merging_and_pruning.md)
- [\[CVPR 2026\] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](../../CVPR2026/video_understanding/egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)
- [\[ICCV 2025\] What You Have is What You Track: Adaptive and Robust Multimodal Tracking](what_you_have_is_what_you_track_adaptive_and_robust_multimodal_tracking.md)
- [\[AAAI 2026\] PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking](../../AAAI2026/video_understanding/plugtrack_multi-perceptive_motion_analysis_for_adaptive_fusion_in_multi-object_t.md)

</div>

<!-- RELATED:END -->
