---
title: >-
  [论文解读] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model
description: >-
  [CVPR 2026][医学图像][域适应] 提出 Tell2Adapt 统一框架，利用视觉基础模型（BiomedParse）的泛化知识，通过上下文感知提示正则化（CAPR）生成高质量伪标签，再经视觉合理性精炼（VPR）去除解剖学不合理预测，实现跨 10 个域迁移方向、22 个解剖目标的统一无源域自适应医学图像分割。
tags:
  - CVPR 2026
  - 医学图像
  - 域适应
  - vision foundation model
  - 图像分割
  - pseudo label
  - 提示学习
---

# Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model

**会议**: CVPR 2026  
**arXiv**: [2603.05012](https://arxiv.org/abs/2603.05012)  
**作者**: Yulong Shi, Shijie Li, Ziyi Li, Lin Qi
**代码**: [derekshiii/Tell2Adapt](https://github.com/derekshiii/Tell2Adapt)  
**领域**: medical_imaging  
**关键词**: source-free domain adaptation, vision foundation model, medical image segmentation, pseudo label, prompt regularization

## 一句话总结

提出 Tell2Adapt 统一框架，利用视觉基础模型（BiomedParse）的泛化知识，通过上下文感知提示正则化（CAPR）生成高质量伪标签，再经视觉合理性精炼（VPR）去除解剖学不合理预测，实现跨 10 个域迁移方向、22 个解剖目标的统一无源域自适应医学图像分割。

## 研究背景与动机

Source-Free Unsupervised Domain Adaptation (SFUDA) 在医学影像部署中至关重要——源域数据因隐私限制无法共享，模型需仅凭目标域未标注数据完成自适应。

现有 SFUDA 方法存在关键局限：
- **场景特定性强**：大多数方法针对低域差距的特定迁移任务设计（如 MRI→MRI），无法扩展为统一的多模态、多目标框架
- **泛化能力弱**：当面临大域差距（如 CT→MRI）或多个解剖目标时，现有方法性能显著下降
- **伪标签质量差**：基于源模型直接生成的伪标签在目标域噪声大，严重限制自适应效果
- **临床可靠性不足**：缺乏对预测结果的解剖学合理性验证，可能产生临床不可接受的假阳性

核心观察：Vision Foundation Model（VFM）如 BiomedParse 在大规模生物医学数据上预训练，具备跨模态、跨解剖结构的广泛知识。如何有效地将这些知识迁移到轻量级部署模型，同时保证临床可靠性，是本文要解决的关键问题。

## 方法详解

### 整体框架

Tell2Adapt 由三个核心阶段组成：**提示正则化 → 伪标签生成与知识蒸馏 → 视觉合理性精炼**。整体流程为：先用 CAPR 将多样化的文本提示标准化，再用 BiomedParse 生成高质量伪标签，通过知识蒸馏训练轻量级 nnUNet 学生模型，最后用 VPR 移除解剖学不合理的预测组件。

### 阶段1：上下文感知提示正则化（CAPR）

VFM（BiomedParse）是 prompt-driven 的分割模型，其性能高度依赖输入提示的质量。然而在实际部署中，不同用户、不同临床场景下的文本提示表述差异巨大。

CAPR 的核心思路：
1. 设计 meta-prompt，利用 LLM（如 GPT-4）将各种非标准的文本提示翻译为 BiomedParse 能理解的规范化指令
2. 确保不同表述的同一解剖概念（如 "left ventricle"、"LV"、"左心室"）映射到一致的 canonical prompt
3. 正则化后的提示直接输入 BiomedParse 进行推理，生成高质量的分割伪标签

### 阶段2：伪标签生成与知识蒸馏

1. 使用 CAPR 正则化后的提示，BiomedParse 对目标域图像进行推理，生成伪标签
2. 将伪标签格式转换为 nnUNet 兼容格式（BiomedParse → nnUNet 格式转换）
3. 以伪标签为监督信号，通过知识蒸馏训练轻量级 nnUNet 学生模型
4. nnUNet 作为最终部署模型，推理速度快、资源需求低，适合临床部署

知识蒸馏的优势：
- BiomedParse 作为教师模型提供泛化知识，nnUNet 学生模型继承其分割能力
- 学生模型无需访问源域数据，完全在目标域伪标签上训练
- 轻量化设计适合实际临床环境的计算资源限制

### 阶段3：视觉合理性精炼（VPR）

VPR 利用 BiomedParse 预计算的解剖学先验，对学生模型的预测进行后处理验证和精炼。

核心算法：
1. 从 Anatomical_Priors.json 加载目标模态和解剖结构的统计先验（Beta 分布参数）
2. 对每个预测的连通分量 $p_i$，计算 log 空间合理性分数：
$$\log S(p_i) = \sum_{k=1}^{4} \left[ (\alpha_k-1)\log f_{i,k} + (\beta_k-1)\log(1-f_{i,k}) - \log B(\alpha_k, \beta_k) \right]$$
其中 $f_{i,k}$ 为连通分量在目标图像低层视觉特征空间中的第 $k$ 维特征值
3. 丢弃分数低于 $\mu_S - 2\sigma_S$ 的连通分量（即解剖学上不合理的假阳性）
4. 小于最小尺寸阈值的组件也被直接移除

支持的模态覆盖广泛：CT 腹部/胸部/肝脏、MRI 腹部/心脏/脑部、X 光胸片、超声心脏、内镜息肉、眼底/皮肤镜/OCT 等 14 种临床场景。

## 实验关键数据

### 实验设置
- **评估规模**：10 个域迁移方向、22 个解剖目标，号称迄今最大规模的 SFUDA 评估之一
- **解剖区域**：脑部（BraTS 脑肿瘤分割）、心脏（M&Ms 心脏 MRI）、息肉（内镜息肉分割）、腹部（AMOS/CHAOS 腹部器官）
- **VFM 教师**：BiomedParse
- **学生模型**：nnUNet
- **评估指标**：Dice Similarity Coefficient (DSC)、Hausdorff Distance 95% (HD95)

### Table 1: 跨模态腹部器官分割 Dice (%) 对比

| 方法 | 类型 | Liver | R.Kidney | L.Kidney | Spleen | Avg |
|---|---|---|---|---|---|---|
| Source Only | — | 58.2 | 47.6 | 46.1 | 42.3 | 48.6 |
| TENT | Test-time | 63.4 | 52.1 | 51.8 | 48.7 | 54.0 |
| AdaptSeg | UDA | 71.5 | 63.2 | 62.4 | 59.8 | 64.2 |
| DPL | SFUDA | 69.3 | 58.7 | 57.2 | 55.1 | 60.1 |
| ProSFDA | SFUDA | 74.8 | 65.3 | 64.1 | 62.4 | 66.7 |
| **Tell2Adapt** | **SFUDA** | **82.6** | **76.8** | **75.4** | **73.1** | **77.0** |

Tell2Adapt 在平均 Dice 上领先第二名 ProSFDA 约 10%，在高域差距迁移中优势更为显著。

### Table 2: 心脏 MRI 分割与消融结果

| 配置 | LV | RV | Myo | Avg DSC | Avg HD95 |
|---|---|---|---|---|---|
| Source Only | 72.3 | 65.1 | 58.4 | 65.3 | 14.2 |
| ProSFDA | 81.2 | 74.6 | 69.3 | 75.0 | 8.7 |
| Tell2Adapt (w/o CAPR) | 83.1 | 77.2 | 71.8 | 77.4 | 7.4 |
| Tell2Adapt (w/o VPR) | 85.4 | 79.8 | 74.1 | 79.8 | 6.8 |
| **Tell2Adapt (Full)** | **87.6** | **82.3** | **76.5** | **82.1** | **5.6** |

消融分析表明：
- CAPR 提升约 4.7% Dice（提示正则化对 VFM 推理质量影响显著）
- VPR 进一步提升约 2.3% Dice 并大幅降低 HD95（有效去除假阳性和噪声）
- 两个模块互补，完整框架达到最优性能

## 亮点与洞察

- **统一框架设计**：首次构建覆盖 10 个域迁移方向、22 个解剖目标的统一 SFUDA 框架，打破了现有方法"一任务一模型"的局限
- **VFM 知识迁移路径**：提出 VFM→伪标签→知识蒸馏→轻量模型 的完整迁移路径，既利用了大模型的泛化能力，又满足临床部署的效率需求
- **提示工程的正则化**：CAPR 解决了 prompt-driven VFM 中提示表述不一致的实际问题，通过 LLM 实现 canonical mapping
- **解剖学先验驱动的后处理**：VPR 不依赖额外训练，仅通过统计先验和低层视觉特征即可有效去除假阳性，增强临床可靠性
- **广泛的模态覆盖**：解剖先验库支持 CT、MRI、X 光、超声、内镜、眼底、皮肤镜、OCT、病理等 14 种临床场景

## 局限性

- **依赖 VFM 质量**：框架性能上界受 BiomedParse 能力限制；对 VFM 覆盖不好的罕见解剖结构或模态，伪标签质量可能不足
- **CAPR 依赖 LLM API**：提示正则化需要调用 LLM（如 GPT-4），在无网络或受限环境中部署存在障碍
- **解剖先验需预设**：VPR 的先验参数需从 BiomedParse 预计算，新增解剖目标需要更新先验库
- **端到端性不足**：三阶段流水线设计使整体流程较复杂，各阶段独立优化可能存在信息传递损耗
- **计算成本**：BiomedParse 推理 + nnUNet 蒸馏训练的整体计算成本仍较高，尽管最终部署模型轻量

## 相关工作

- **SFUDA 方法**：TENT（测试时自适应）、DPL（域伪标签）、ProSFDA（渐进式无源域自适应）→ 多针对特定域差距设计，缺乏统一框架
- **视觉基础模型**：SAM（通用分割基础模型）、BiomedParse（生物医学分割基础模型）→ 本文基于 BiomedParse 作为知识来源
- **知识蒸馏**：将大模型知识迁移到轻量模型的经典范式 → 本文用于从 VFM 伪标签训练 nnUNet
- **医学图像分割 DA**：AdaptSeg（对抗训练）、UAMT（不确定性感知）→ 多需源域数据或仅支持特定模态
- **Tell2Adapt 定位**：将 VFM 泛化知识通过提示正则化和解剖先验精炼，高效迁移到轻量级部署模型，实现首个统一多模态多目标 SFUDA 框架

## 评分

- 新颖性: ⭐⭐⭐⭐ — VFM + CAPR + VPR 的组合思路清晰有新意，将基础模型知识迁移到 SFUDA 场景的路径设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ — 10 个域迁移方向、22 个解剖目标，规模在 SFUDA 领域非常突出
- 写作质量: ⭐⭐⭐⭐ — 框架结构清晰，但三阶段设计描述略显分散
- 价值: ⭐⭐⭐⭐ — 统一 SFUDA 框架对医学影像临床部署有直接实用价值，代码已开源

<!-- RELATED:START -->

## 相关论文

- [Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy](unsupervised_domain_adaptation_with_target-only_margin_disparity_discrepancy.md)
- [Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning](reclaiming_lost_text_layers_for_source-free_cross-domain_few-shot_learning.md)
- [GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)
- [SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)

<!-- RELATED:END -->
