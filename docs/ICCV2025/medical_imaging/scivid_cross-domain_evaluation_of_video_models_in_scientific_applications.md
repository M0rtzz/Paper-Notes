---
title: >-
  [论文解读] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications
description: >-
  [ICCV 2025][医学图像][视频基础模型] 提出 SciVid 基准，包含动物行为分类、组织追踪、天气预测等 5 个跨学科科学视频任务，系统评估 6 类视频基础模型（ViFM），发现用简单可训练 readout 适配冻结的 ViFM backbone 即可在多个科学应用中达到 SOTA…
tags:
  - "ICCV 2025"
  - "医学图像"
  - "视频基础模型"
  - "跨域评估"
  - "科学应用"
  - "benchmark"
  - "时空建模"
---

# SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications

**会议**: ICCV 2025  
**arXiv**: [2507.03578](https://arxiv.org/abs/2507.03578)  
**代码**: [github.com/google-deepmind/scivid](https://github.com/google-deepmind/scivid)  
**领域**: 医学图像  
**关键词**: 视频基础模型, 跨域评估, 科学应用, benchmark, 时空建模

## 一句话总结

提出 SciVid 基准，包含动物行为分类、组织追踪、天气预测等 5 个跨学科科学视频任务，系统评估 6 类视频基础模型（ViFM），发现用简单可训练 readout 适配冻结的 ViFM backbone 即可在多个科学应用中达到 SOTA，首次证明通用 ViFM 在科学领域的可迁移性。

## 研究背景与动机

### 问题定义

视频基础模型（ViFM）在自然视频理解上取得了显著进展（动作识别、视频问答等），但在科学领域的应用仍然有限。每个科学领域（医学、动物行为、气象）通常各自开发专用模型，且仅在该领域内评估。

核心问题：**通用 ViFM 能否跨不同科学领域有效迁移？单个预训练 ViFM 能否与领域专用 baseline 竞争？**

### 已有方法的不足

**领域专用模型**：Endo-FM（内窥镜）、EchoCLIP（超声心动图）等仅面向特定领域，跨领域泛化性未知

**缺乏统一评估**：各领域使用不同的度量、数据格式和评估协议，无法横向比较不同 ViFM 的科学适用性

**ViFM 上下文有限**：现有 ViFM 基准（如 SSv2、Kinetics）主要评估自然视频理解，不涉及医学、气象等科学应用，无法判断 ViFM 在大幅域迁移下的表现

### 核心动机

**关键洞察**：许多科学任务可以表示为视频建模问题——医学组织追踪是点追踪、天气预测是时空预测、动物行为分析是视频分类。如果通用 ViFM 能有效迁移到这些差异巨大的领域，将极大降低科学应用中专用模型开发的门槛。SciVid 旨在提供统一的评估框架回答这个问题。

## 方法详解

### 整体框架

SciVid 的评估范式遵循统一的表示学习流程：
1. 使用预训练 ViFM 作为冻结 backbone 提取视频特征
2. 在特征上训练轻量级任务特定 readout 模块
3. 可选地微调 backbone

这确保了不同 ViFM 间的公平比较——唯一变量是 backbone 的特征质量。

### 关键设计

#### 1. **任务设计原则与五个基准任务**

- **功能**：构建覆盖三大科学领域、多种输出类型的 5 个视频任务。
- **核心思路**：

  任务选择遵循四个原则：(1) 广泛覆盖科学应用挑战；(2) 包含多样域和分布偏移；(3) 强调时序理解需求；(4) 混合成熟和待开发的任务。

  **动物行为分类**：
  - FlyVsFly：果蝇社交行为分类（7 类），灰度视频，144×144，16 帧输入
  - CalMS21：小鼠社交行为分类（4 类），灰度视频，285×512，16 帧输入

  **医学组织追踪（STIR）**：
  - 手术中组织表面运动追踪，RGB 视频，1024×1280，7~19419 帧
  - 任务：给定首帧查询点，追踪至末帧

  **天气预报（WeatherBench 2）**：
  - 中期天气预测，输入 8 天（16 帧），输出 8 天
  - 预测 Z500（位势高度）、T850（温度）、Q700（比湿）

  **气旋压力预测（Digital Typhoon）**：
  - 红外卫星图像，输入 12 帧，预测未来 12 小时中心气压
  - 时间序列回归任务

- **设计动机**：五个任务在输入模态（灰度/RGB/红外/气象变量）、输出形式（分类/点追踪/稠密预测/标量回归）和数据规模（60 到 1M 样本）上差异巨大，能全面测试 ViFM 的通用性。

#### 2. **Backbone 选择与评估**

- **功能**：系统评估 6 类 ViFM backbone 的特征质量。
- **核心思路**：

  评估的 backbone 包括：
  - **图像模型**：DINOv2 (ViT-L/g)——自蒸馏训练的纯图像模型，加可学习时序位置编码
  - **视频模型**：
    - VideoPrism (B/g)：两阶段——视频文本对比学习 → 掩码自编码，使用了语言监督
    - VideoMAE / VideoMAEv2 (B/L/H/g)：像素空间掩码自编码
    - V-JEPA (L/H)：潜空间掩码预测（JEPA 范式）
    - 4DS (L/e)：像素空间掩码自编码，参数量 300M~4B
  - **Resize baseline**：将输入视频 resize 到低分辨率作为朴素特征（验证 backbone 确实提取了有意义的信息）

  关键设计：所有 backbone 接收标准 3 通道时空片段，确保评估协议一致。

- **设计动机**：覆盖了当前主流的三大 ViFM 训练范式——对比学习（VideoPrism）、像素级掩码重建（VideoMAE/4DS）、潜空间预测（V-JEPA），以及纯图像基线（DINOv2），可以系统性地分析不同预训练策略对科学应用的影响。

#### 3. **任务 Readout 设计**

- **功能**：将 ViFM 的通用特征适配到具体任务输出。
- **核心思路**：

  **分类/压力预测**：Cross-attention readout——单个可学习 query 通过交叉注意力聚合 backbone 特征，输出类别 logits 或压力预测。损失为 sigmoid cross-entropy（分类）或 L2（回归）。

  **组织追踪（STIR）**：Cross-attention readout——query 由查询点位置编码提供，key/value 由 backbone 特征提供。预测所有目标点位置、可见性和不确定性。损失为 Huber loss + BCE。

  **天气预报**：DPT readout——系列可训练卷积和重组层，将特征上采样为逐像素预测。损失为面积加权 L1。

  所有 readout 从头训练，backbone 冻结。整套实验在单 H100 GPU 上不到一天即可完成。

- **设计动机**：readout 设计尽可能简单，将性能差异归因于 backbone 的表示质量而非任务适配的复杂度。cross-attention 比简单线性投影显著更好（验证了特征的位置信息是有用的）。

### 损失函数 / 训练策略

- **分类**：Sigmoid cross-entropy loss
- **追踪**：Huber loss（位置）+ BCE（可见性/不确定性）
- **天气预报**：面积加权 L1 loss（channel-weighted）
- **压力预测**：L2 loss on pressure offsets

所有任务统一使用 40k 训练步（部分需 400k 达最优），冻结 backbone。

## 实验关键数据

### 主实验

**SOTA 对比**（冻结 backbone 的 readout 训练）：

| 任务 | 领域专用 SOTA | 最佳 ViFM (冻结) | ViFM 是否达 SOTA |
|------|-------------|-----------------|----------------|
| CalMS21 | VideoPrism-g 91.5 mAP | V-JEPA-H **92.4 mAP** | ✅ 超越 |
| FlyVsFly | VideoPrism-g 92.0 mAP | VideoPrism-g **92.5 mAP** | ✅ 超越 |
| STIR | MFT 68.5%/77.6% acc | 4DS-e 51.3%/57.8% (冻结) → 61.2%/69.2% (微调) | ❌ 差距明显 |
| Digital Typhoon | Kitamoto 11.71 RMSE | 4DS-L **3.88 RMSE** (val) | ✅ 大幅超越 |
| WeatherBench 2 | GenCast ~最优 | 4DS-e/VideoMAEv2-g 中等 | ❌ 差距明显 |

### 消融实验

**不同 backbone 在 5 个任务上的冻结特征性能**：

| Backbone | 参数(M) | CalMS21 mAP↑ | FlyVsFly mAP↑ | STIR Acc↑ | DT RMSE↓ | WB2 Z500↓ |
|----------|--------|-------------|--------------|----------|---------|----------|
| 4DS-e | 3811 | 0.817 | 0.894 | **0.513** | 4.23 | **601** |
| DINOv2-g | 1135 | **0.866** | 0.866 | 0.215 | 6.33 | 627 |
| VideoMAEv2-g | 1013 | 0.862 | 0.887 | 0.344 | 4.53 | 594 |
| V-JEPA-H | 635 | 0.828 | **0.901** | 0.443 | **4.16** | 619 |
| VideoPrism-g | 1113 | 0.855 | 0.839 | 0.351 | 5.01 | 635 |
| Resize | 0 | 0.122 | 0.095 | 0.280 | 10.0 | 642 |

**Readout 架构消融**：

| 任务 | Linear readout | Cross-attention readout |
|------|---------------|----------------------|
| FlyVsFly mAP↑ | 0.568 | **0.894** |
| CalMS21 mAP↑ | 0.525 | **0.817** |
| Digital Typhoon RMSE↓ | 7.45 | **4.23** |

### 关键发现

1. **没有单一最优 backbone**：4DS-e 在追踪和天气预报上最好，V-JEPA-H 在 FlyVsFly 上最好，DINOv2 在 CalMS21 上最好。任务特性决定了最优模型
2. **纯视频模型整体优于图像模型**：DINOv2 在需要强时序建模的任务（STIR、WB2）上远弱于视频模型（STIR 0.215 vs 4DS-e 0.513）
3. **像素级掩码自编码模型在时空预测上占优**：VideoMAE、4DS 系列在 WeatherBench 2 上一致优于其他范式
4. **ViFM 在 3/5 任务上达到 SOTA**：动物行为分类和气旋预测上超越领域专用方法，但在组织追踪和天气预报上仍有显著差距
5. **时序建模确实重要**：frame shuffling 实验显示追踪任务性能大幅下降，分类任务影响较小
6. **模型规模并非总是更好**：4DS-L(300M) 在 Digital Typhoon 上超过 4DS-e(4B)，VideoMAE-B 在 STIR 上接近 VideoMAE-L

## 亮点与洞察

1. **首个跨学科科学 ViFM 基准**：将医学、动物行为、气象三个完全不同的领域统一到一个评估框架中，填补了重要空白
2. **实验设计的公平性**：统一的 readout 架构和训练协议确保了 backbone 间的可比性，单 H100 不到一天完成全部实验
3. **积极发现**：通用 ViFM 在多个科学任务上可以超越领域专用方法，证明了预训练知识的跨域可迁移性
4. **对实践的指导意义**：科学家不需要从头训练专用模型，pick 一个好的 ViFM backbone + 简单 readout 即可获得有竞争力的结果

## 局限与展望

1. **任务覆盖有限**：仅 5 个任务、3 个领域，未涵盖显微成像、卫星时间序列、水下视频等
2. **短片段评估为主**：除 STIR 外均为短片段（8-16 帧），未涉及长视频理解
3. **未探索数据高效适配**：仅初步研究了低数据场景（附录），未深入探索 few-shot 适配策略
4. **天气预报差距大**：ViFM 在 WeatherBench 2 上与 GraphCast/GenCast 差距显著，可能需要更好的预训练或适配方法
5. **STIR 追踪较弱**：简单 readout 缺乏追踪任务的关键组件（特征金字塔、相关体积、迭代细化）

## 相关工作与启发

- 与 VideoEval 的关系：VideoEval 评估 ViFM 在挑战性任务上的表现，但 SciVid 覆盖更多科学领域，且与领域专用 SOTA 做仔细对比
- 与 ClimaX 的关系：ClimaX 是面向天气/气候的 foundation model，但仅限于单一领域。SciVid 研究通用 ViFM 的跨领域表现
- 启发：未来 ViFM 预训练可以有意识地加入科学领域数据（医学视频、气象数据等），以提升跨域迁移能力

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 跨学科基准的设计理念新颖，但核心技术（冻结 backbone + readout）相对标准
- **实验充分度**: ⭐⭐⭐⭐⭐ — 6 类 backbone × 5 个任务的全矩阵评估，时序消融、readout 消融、规模消融均覆盖
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构清晰，表格和图示丰富，关键结论一目了然
- **价值**: ⭐⭐⭐⭐ — 为科学领域使用 ViFM 提供了重要参考，但实际贡献是基准而非方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](../../NeurIPS2025/medical_imaging/endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)
- [\[ICLR 2026\] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](../../ICLR2026/medical_imaging/adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[ICCV 2025\] ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users](progait_a_multi-purpose_video_dataset_and_benchmark_for_transfemoral_prosthesis_.md)
- [\[CVPR 2025\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](../../CVPR2025/medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)

</div>

<!-- RELATED:END -->
