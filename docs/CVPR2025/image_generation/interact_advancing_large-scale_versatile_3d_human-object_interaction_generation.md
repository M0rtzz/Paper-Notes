---
title: >-
  [论文解读] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation
description: >-
  [CVPR 2025][图像生成][人物-物体交互] 本文提出 InterAct 基准，整合并标准化了 21.81 小时的 3D 人物-物体交互数据（扩展到 30.70 小时），通过统一优化框架校正运动捕捉伪影并增强数据，定义六项生成任务和统一建模方法，在多个 HOI 生成任务上取得 SOTA 表现。
tags:
  - CVPR 2025
  - 图像生成
  - 人物-物体交互
  - 动作生成
  - 数据集
  - 交互校正增强
  - 多任务学习
---

# InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation

**会议**: CVPR 2025  
**arXiv**: [2509.09555](https://arxiv.org/abs/2509.09555)  
**代码**: https://github.com/wzyabcas/InterAct  
**领域**: 3D视觉 / 人物交互生成  
**关键词**: 人物-物体交互, 动作生成, 数据集, 交互校正增强, 多任务学习

## 一句话总结
本文提出 InterAct 基准，整合并标准化了 21.81 小时的 3D 人物-物体交互数据（扩展到 30.70 小时），通过统一优化框架校正运动捕捉伪影并增强数据，定义六项生成任务和统一建模方法，在多个 HOI 生成任务上取得 SOTA 表现。

## 研究背景与动机

1. **领域现状**：大规模人体运动捕捉数据集推动了人体动作生成的发展，但动态 3D 人物-物体交互（HOI）的建模和生成仍面临挑战，主要受限于数据集质量和规模。
2. **现有痛点**：现有 HOI 数据集存在三个关键问题：(a) 数据集有限且不一致——不同数据集使用不同的人体表示、坐标系和标注格式，难以整合；(b) 标注粗糙不完整——缺乏对交互细节、涉及身体部位的详细文本描述；(c) 普遍存在伪影——接触穿透、浮动接触、不准确的手部姿态和运动抖动等问题。
3. **核心矛盾**：高质量 HOI 数据的稀缺限制了生成模型学习真实交互动力学的能力，而直接整合现有数据集面临表示不一致和质量低下的双重障碍。
4. **本文目标** (a) 如何统一整合异构 HOI 数据集？(b) 如何自动修复 MoCap 伪影？(c) 如何在不采集新数据的情况下扩展数据量？(d) 如何设计统一的多任务 HOI 生成框架？
5. **切入角度**：利用"交互不变性"原则——即使人体运动发生微小变化，人物-物体的接触关系应保持一致——来增强数据。
6. **核心 idea**：构建首个大规模统一 HOI 基准，通过数据校正和基于接触不变性的增强扩展数据，并用统一多任务学习框架实现六项 HOI 生成任务的 SOTA。

## 方法详解

### 整体框架
InterAct 分为数据构建和方法设计两部分。数据构建整合 7 个源数据集共 21.81 小时的交互数据，统一人体表示为 marker-based 表示，通过两阶段标注（人工+GPT-4）生成文本描述，然后通过三步优化管线（全身校正→手部校正→交互增强）提升质量并扩展至 30.70 小时。方法设计定义了六项任务，并使用基于 transformer 的扩散模型进行多任务联合建模。

### 关键设计

1. **Marker-based 统一人体表示**:

    - 功能：统一不同人体模型（SMPL-H, SMPL-X）的表示，便于交互建模
    - 核心思路：选择人体表面的特定顶点作为 marker，而非使用关节点或旋转参数。通过在 SMPL-H 表面索引 marker 点，再在 SMPL-X 上找到最近对应点来建立不同模型间的 marker 对应关系。marker 位于人体表面，直接参与交互，误差控制在 1cm 以内
    - 设计动机：关节点位于身体内部，不直接参与接触交互；SMPL 旋转表示不如笛卡尔坐标直观；marker 表示既统一了不同模型，又天然适合接触建模

2. **分阶段交互校正与增强优化框架**:

    - 功能：修复 MoCap 数据中的伪影并生成合成数据扩充数据集
    - 核心思路：分三步执行梯度优化——(1) **全身校正**：优化全身人体和物体姿态，使用重建损失保持接近原始数据，加上接触损失和穿透损失减少伪影；(2) **手部校正**：单独优化手部姿态，使用接触促进损失 $E_{\text{cont}} = \sum_i c_i \sum_j d_j[i]$ 引导手部贴合物体，加上指关节活动范围约束保持自然性；(3) **交互增强**：对物体轨迹施加随机位移，然后优化人体运动保持接触一致性，使用加权距离损失 $E_{\text{align}} = \sum_{i,j,k} \frac{1}{(\hat{D}_{jk}+\epsilon)^2} |\hat{D}_{jk} - D_{jk}|^2$，最后过滤低质量结果
    - 设计动机：手部校正与全身校正分离，因为手部在 SMPL 表示中参数多但在整体损失中贡献小，分开处理可更好平衡；增强基于"交互不变性"——握着箱子走路时，步态微变不影响手-箱接触

3. **统一多任务 HOI 生成建模**:

    - 功能：用一个模型统一处理文本/动作/物体/人体条件生成和交互预测五项运动学生成任务
    - 核心思路：为每项任务统一 HOI 序列表示 $\langle h, o \rangle$（包含 marker 坐标、速度、到物体的签名距离向量、脚地接触标签和物体运动+BPS 几何编码），引入额外的人物-物体关系特征 $\eta$（人体 marker 到物体最近点的向量），作为多任务学习的附加输出。使用 transformer 扩散模型，训练时联合回归运动和接触特征
    - 设计动机：统一表示避免了为每个任务设计独立模型；联合建模接触关系 $\eta$ 迫使模型学习交互的空间关系，一致性提升

### 损失函数 / 训练策略
数据校正使用梯度下降优化，包含重建损失、接触损失、穿透损失、平滑损失和先验损失。生成模型使用扩散去噪损失，并可选地加入 classifier guidance 利用接触预测。文本编码使用 Sentence-BERT + InfoNCE 对比学习训练的交互感知编码器。

## 实验关键数据

### 主实验

**文本条件交互生成：**

| 配置 | R-Prec Top1 ↑ | FID ↓ | MM Dist ↓ | Diversity |
|------|--------------|-------|-----------|-----------|
| Ground Truth | 0.852 | 0.000 | 2.810 | 11.489 |
| 基线 (无HOI感知) | 0.733 | 3.192 | 4.950 | 11.192 |
| +Contact 建模 | 0.730 | 1.997 | 4.752 | 11.501 |
| +HOI感知物体编码 | 0.737 | 1.837 | 4.631 | 11.369 |
| +HOI感知文本编码 | 0.784 | 1.570 | 4.414 | 11.409 |
| **+Guidance (Full)** | **0.784** | **1.567** | **4.412** | 11.518 |

**动作条件交互生成：**

| 方法 | FID ↓ | Multimodality ↑ | Diversity |
|------|-------|----------------|-----------|
| HOI-Diff | 3.566 | 5.321 | 10.989 |
| **Ours** | **2.161** | **5.792** | **11.291** |

### 消融实验

| 数据版本 | Penetration ↓ | Contact Ratio | 用户偏好 |
|---------|--------------|---------------|---------|
| BEHAVE 原始 | 0.017 | 0.048 | 22.3% |
| BEHAVE 校正 | 0.016 | 0.071 | 39.7% |
| BEHAVE 校正+增强 | 0.016 | 0.069 | 38.0% |
| OMOMO 原始 | 0.009 | 0.071 | 23.9% |
| OMOMO 校正 | 0.007 | 0.131 | 39.4% |

| 人体表示 | Penetration ↓ | Contact Ratio |
|---------|--------------|---------------|
| SMPL | 0.030 | 0.025 |
| Joint | 0.027 | 0.032 |
| **Marker** | **0.025** | **0.028** |

### 关键发现
- 交互校正显著提升数据质量：接触比率提升 47%-85%，用户偏好从 22-24% 提升至 39-40%
- 增强数据质量接近校正数据，证明了交互不变性原则的有效性
- HOI 感知文本编码器贡献最大：R-Precision 从 0.737 提升到 0.784，FID 从 1.837 降到 1.570
- Marker 表示在穿透和接触指标上均优于 SMPL 和关节表示
- 交互预测任务中，更大的模型和更多的数据持续提升性能（scaling law 成立）

## 亮点与洞察
- **"交互不变性"数据增强**是一个漂亮的洞察：通过位移物体并优化人体保持接触一致性来生成新数据，既增加了多样性又保证了交互的物理合理性。这个思路可迁移到手-物交互等需要接触一致性的场景。
- **Marker-based 表示**兼顾了表示统一性和交互建模需求：位于表面的 marker 点天然适合计算接触距离和穿透检测，比内部关节点更适合 HOI 任务。
- **分离手部和全身校正**的策略值得借鉴：当高维参数空间中不同分量的尺度差异大时，分阶段优化能获得更好的平衡。

## 局限与展望
- 数据增强中的物体位移是均匀随机的，更智能的位移策略（如基于场景语义）可能生成更有意义的变体
- 当前仅处理单人-单物体交互，多人或多物体场景未覆盖
- 物理模拟（交互模仿任务）与运动学生成仍然分离，端到端的物理一致性生成仍有空间
- 文本描述主要由 GPT-4 改写，可能引入偏差

## 相关工作与启发
- **vs InterDiff**: InterDiff 首次引入多样化动态物体交互，但数据规模受限。InterAct 通过大规模数据和多任务学习全面超越
- **vs OMOMO**: OMOMO 使用两阶段生成（先手再全身），适合手部主导的交互但不适合全身交互。InterAct 的单阶段多任务学习更通用
- **vs PhysHOI**: 物理模拟方法能保证物理合理性，但交互模式单一。InterAct 证明使用校正数据可以将模拟成功率从 84.4% 提升到 90.7%

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据构建流水线和交互不变性增强有创新，但多任务学习框架相对常规
- 实验充分度: ⭐⭐⭐⭐⭐ 六项任务全面评估，包含定量指标、用户研究、消融和物理模拟验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但内容量大导致部分细节需看补充材料
- 价值: ⭐⭐⭐⭐⭐ 作为目前最大规模的 3D HOI 基准，对该领域有基础性推动作用

<!-- RELATED:START -->

## 相关论文

- [HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](an_image-like_diffusion_method_for_human-object_interaction_detection.md)
- [RORem: Training a Robust Object Remover with Human-in-the-Loop](rorem_training_a_robust_object_remover_with_human-in-the-loop.md)
- [ViHOI: Human-Object Interaction Synthesis with Visual Priors](../../CVPR2026/image_generation/vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](../../ICCV2025/image_generation/scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model](re-hold_video_hand_object_interaction_reenactment_via_adaptive_layout-instructed.md)

<!-- RELATED:END -->
