---
title: >-
  [论文解读] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment
description: >-
  [NeurIPS 2025][医学图像][帕金森病] 发布 Care-PD——目前最大的面向帕金森病步态分析的多站点匿名 3D 网格数据集（9 个队列、8 个临床中心、362 名受试者、8477 段步行），并在 UPDRS 步态评分和运动预训练任务上提供系统性 benchmark，证明在 Care-PD 上微调可将 MPJPE 从 60.8mm 降至 7.5mm，F1 提升 17 个百分点。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "帕金森病"
  - "步态分析"
  - "SMPL 3D 网格"
  - "多站点数据集"
  - "运动表征学习"
---

# Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment

**会议**: NeurIPS 2025  
**arXiv**: [2510.04312](https://arxiv.org/abs/2510.04312)  
**代码**: [Website](https://neurips2025.care-pd.ca)  
**领域**: 医学图像  
**关键词**: 帕金森病, 步态分析, SMPL 3D 网格, 多站点数据集, 运动表征学习

## 一句话总结
发布 Care-PD——目前最大的面向帕金森病步态分析的多站点匿名 3D 网格数据集（9 个队列、8 个临床中心、362 名受试者、8477 段步行），并在 UPDRS 步态评分和运动预训练任务上提供系统性 benchmark，证明在 Care-PD 上微调可将 MPJPE 从 60.8mm 降至 7.5mm，F1 提升 17 个百分点。

## 研究背景与动机

**领域现状**：帕金森病（PD）的步态评估目前主要依赖临床医生主观评分（UPDRS），自动化运动分析虽能提供客观指标，但受限于小规模、单站点数据集。

**现有痛点**：现有 PD 步态数据集规模小、模态单一（仅有 IMU 或视频）、缺乏标准化表征和临床标注，模型在跨站点泛化时性能急剧下降。通用运动数据集（如 Human3.6M、AMASS）只含健康人动作，缺少病理步态模式。

**核心矛盾**：缺乏大规模、多站点、标准化且公开可用的 PD 步态运动数据集是制约泛化模型发展的根本瓶颈。

**本文目标**：构建统一的 3D 表征格式（SMPL mesh），聚合多个来源的 RGB 视频和光学动态捕捉数据，提供包含临床标注的公开 benchmark。

**切入角度**：用 SMPL 作为统一的隐私保护 3D 网格表征，将异构采集模态（MoCap 和 RGB）归一化到同一空间。

**核心 idea**：统一数据格式 + 多站点聚合 + 分层评估协议（站内/跨站/LODO/MIDA）+ 运动预训练验证。

## 方法详解

### 整体框架
Care-PD 并非一个新的模型，而是一个数据集 + benchmark 工作。核心流程为：(1) 从 9 个队列收集 RGB 视频和 MoCap 数据；(2) 统一预处理为 30Hz SMPL 网格序列并匿名化；(3) 设计两类 benchmark 任务——临床评分预测和运动预训练任务；(4) 在 4 种泛化协议下评估 7 个运动编码器和手工特征基线。

### 关键设计

1. **数据统一化管线**:

    - 功能：将异构输入（RGB 视频 / 光学 MoCap）转为统一的 SMPL 3D 网格序列
    - 核心思路：MoCap 数据经质量控制→关节标准化→SparseFusion 优化拟合 SMPL→降采样至 30Hz；RGB 视频经 WHAM 单目 mesh 恢复→视觉验证→非步行行为筛除→Kabsch 算法校正天花板相机倾斜
    - 设计动机：SMPL 提供了与采集设备无关的统一解剖学框架，同时通过只发布无纹理的 mesh 参数（排除形状参数）实现身份匿名化

2. **临床评分预测任务**:

    - 功能：从步行序列 $\mathbf{M}^{1:T}$ 预测 UPDRS 步态评分 $S \in \{0,1,2,3\}$
    - 核心思路：对变长输入按编码器要求切分为固定长度片段 $\{\mathbf{p}_1^{1:N}, ...\}$，冻结预训练运动编码器 $\mathcal{E}$ 提取特征，用轻量分类头 $\Phi$ 预测片段级别评分，最后通过多数投票得到序列级预测
    - 评估了 7 个 SOTA 运动编码器（POTR、MixSTE、PoseFormerV2、MotionBERT、MotionAGFormer、MotionCLIP、MoMask）+ Random Forest 手工特征基线

3. **四种泛化评估协议**:

    - Within-dataset：Leave-One-Subject-Out 交叉验证
    - Cross-dataset：在一个数据集上训练，在其余数据集上测试
    - LODO (Leave One Dataset Out)：在 D-1 个数据集上训练，在剩余一个上测试
    - MIDA (Multi-dataset In-domain Adaptation)：在 LODO 基础上用目标域少量数据微调分类头
    - 设计动机：模拟从受控实验到真实部署的递进场景，暴露域迁移瓶颈

4. **运动预训练任务**:

    - 功能：评估 Care-PD 在 2D→3D 关键点提升和 3D 重建两个预训练任务上的价值
    - 核心思路：比较四种训练方案——零样本、在 Care-PD 上微调、在健康步态上微调、从零训练
    - 设计动机：验证临床数据的不可替代性——健康步态微调带来的增益远小于病理步态数据

### 数据规模
总计 362 名受试者，8477 段步行，18.66 小时 3D 网格序列；覆盖 6 个国家的 8 个临床中心，含 RGB 和 MoCap 两种模态。

## 实验关键数据

### 主实验（运动预训练对下游影响）

| 模型 | 训练数据 | 微调数据 | MPJPE ↓ (mm) | PA-MPJPE ↓ | F1 ↑ |
|------|---------|---------|-------------|------------|------|
| MotionAGFormer | H3.6M | - | 60.7 | 21.4 | 48.1 |
| MotionAGFormer | H3.6M | Healthy Gait | 29.8 | 7.3 | 50.1 |
| MotionAGFormer | H3.6M | **Care-PD** | **7.5** | **2.6** | **65.1** |
| MotionAGFormer | Care-PD | - | 9.0 | 3.2 | 62.3 |
| MoMask | HumanML3D | - | 22.5 | 17.8 | 41.4 |
| MoMask | HumanML3D | Healthy Gait | 22.3 | 13.7 | 40.6 |
| MoMask | HumanML3D | **Care-PD** | **8.7** | **6.3** | **62.7** |
| MoMask | Care-PD | - | 9.6 | 7.3 | 59.8 |

在 Care-PD 上微调后 MotionAGFormer 的 MPJPE 从 60.7mm 降至 7.5mm（降 87.6%），F1 从 48.1 升至 65.1（+17 pp）。仅用健康步态微调 F1 只提升 2 pp，证明性能增益来自病理运动学特征。

### 评估协议对比

| 协议 | BMClab F1₀₋₂ | PD-GaM F1₀₋₂ | 说明 |
|------|-------------|--------------|------|
| Within-dataset (LOSO) | ~0.68 | ~0.73 | 站内最佳 |
| Cross-dataset | 0.20-0.40 | 0.40+ | 跨站下降 0.2-0.4 |
| LODO | ~0.50 | ~0.50 | 多数据集训练有帮助但仍有域差距 |
| MIDA | 0.74-0.78 | 0.63-0.70 | 少量域内数据大幅提升 |

### 关键发现
- 跨站点评估时 F1 通常下降 0.2-0.4，暴露严重的域迁移问题
- MoMask（VQ-VAE 架构）在跨站点场景中最鲁棒，平均跨站 F1 > 0.40
- MIDA 比 LODO 大幅提升（BMClab 上从 ~0.50 到 ~0.78），说明少量域内监督至关重要
- 亚组分析表明模型能区分药物开/关状态（Cliff's Δ=0.42）、冻结/非冻结步态（Δ=0.25）和 PD/健康人（Δ=0.50）

## 亮点与洞察
- **SMPL 统一表征**：用 SMPL 3D mesh 统一 MoCap 和视频两种模态，同时解决跨模态对齐和隐私保护两个问题——这一策略可迁移到其他需要多模态运动数据融合的临床场景
- **四层评估协议**：从 within→cross→LODO→MIDA 递进式暴露泛化瓶颈，比单纯报一个准确率有信息量得多
- **病理数据不可替代性**：在健康步态上微调几乎无增益，只有病理数据才能让模型学到临床相关特征——这是一个重要的实证结论

## 局限与展望
- UPDRS 步态评分 3（严重损伤）极度稀缺，仅在 PD-GaM 和 3DGait 两个子集中出现，严重影响分类器性能
- 所有运动编码器都是冻结评估（frozen probe），未探索端到端微调的上限
- SMPL 拟合本身存在误差，尤其是 RGB→SMPL 管线（WHAM）可能在严重步态异常下失效
- 未考虑纵向追踪——同一患者在不同时间的病情演变
- 可探索将病理步态数据作为课程学习/数据增强策略融入通用运动编码器训练

## 相关工作与启发
- **vs Human3.6M/AMASS**：通用运动数据集规模大但只含健康人活动，缺乏病理模式；Care-PD 小但针对性强，微调后增益远超通用数据
- **vs DaphNet/mPower 等 IMU 数据集**：IMU 数据集便携但缺少解剖学上下文；Care-PD 的 3D mesh 提供完整身体运动学信息
- **vs CellProfiler 范式**：类似于显微镜图像分析中手工特征 vs 深度特征的争论，本文表明在步态分析中学习到的表征在泛化性上优于手工步态特征

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个多站点 3D mesh PD 步态数据集，统一化管线设计周到
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个编码器 × 4 种评估协议 × 亚组分析 + 预训练任务，非常全面
- 写作质量: ⭐⭐⭐⭐ 数据集描述详尽，实验组织清晰，但篇幅较长
- 价值: ⭐⭐⭐⭐⭐ 填补了 PD 步态分析领域缺乏大规模公开标准化数据集的空白，对社区有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](../../ICLR2026/medical_imaging/care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)
- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[NeurIPS 2025\] DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders](dermacon-in_a_multi-concept_annotated_dermatological_image_dataset_of_indian_ski.md)
- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[CVPR 2026\] Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](../../CVPR2026/medical_imaging/unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)

</div>

<!-- RELATED:END -->
