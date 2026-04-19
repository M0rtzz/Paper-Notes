---
title: >-
  [论文解读] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-Supervised Medical Image Segmentation
description: >-
  [CVPR 2025][医学图像分割] 提出增强SAM框架，利用CLIP和VQA无监督生成标注高效的提示，结合DPO偏好优化和虚拟标注器实现半监督医学图像分割，在肺/乳腺肿瘤/器官分割上达到SOTA
tags:
  - CVPR 2025
  - 医学图像分割
  - SAM
  - 偏好优化
  - 半监督学习
  - DPO
---

# Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-Supervised Medical Image Segmentation

**会议**: CVPR 2025  
**arXiv**: 见CVF  
**代码**: 待确认  
**领域**: 医学图像 / 对齐RLHF  
**关键词**: SAM, 半监督分割, DPO偏好优化, 无监督提示生成, 虚拟标注器

## 一句话总结

提出一种增强 SAM 的半监督医学图像分割框架：通过 CLIP 和 VQA 无监督生成包含语义、位置和形状信息的高效提示（无需专家标注），再用 DPO 偏好优化技术配合虚拟标注器（代替人类标注者提供排名/评分）训练最优分割策略，在肺分割、乳腺肿瘤分割、器官分割等多模态任务上达到 SOTA。

## 研究背景与动机

**领域现状**：Segment Anything Model（SAM）作为视觉基础模型在医学图像分割领域展现出强大潜力，理论上可以泛化到多种下游分割任务。然而 SAM 本质上是一个监督模型，其强大表现依赖于大规模标注数据或领域专家提供的高质量提示（如点/框/掩码提示）。

**现有痛点**：医学图像标注成本极高——需要放射科医生等专业人员逐像素标注，且不同模态（X-ray/超声/CT）的标注协议不同，难以规模化。传统的缓解策略如主动学习（active learning）虽然减少了标注量，但仍然需要持续的人工参与来精炼标签或建立奖励的 ground truth，范围有限且流程复杂。更关键的是，SAM 的提示机制（prompt）本身就需要专家介入来提供位置先验，这形成了"使用基础模型仍需专家"的悖论。

**核心矛盾**：SAM 的强大泛化能力需要精确的提示输入来激活，但医学场景下获取精确提示的成本与获取标注几乎一样高。如何在无监督或半监督设置下自动生成有效的 SAM 提示，同时保证分割质量，是核心挑战。

**本文目标** (1) 如何不依赖专家标注就能为 SAM 生成有效的分割提示？(2) 如何在低标注数据场景下优化 SAM 的分割策略，使其产出高保真分割结果？

**切入角度**：将 CLIP 的视觉-语言对齐能力和 VQA 的问答能力用于自动提取医学图像的语义/位置/形状信息作为 SAM 提示，再借鉴 LLM 领域的 DPO 偏好优化思想，用虚拟标注器提供的简单排名/评分来优化分割策略，完全绕过了传统的像素级标注需求。

**核心 idea**：用 CLIP+VQA 自动生成 SAM 提示，用 DPO+虚拟标注器做偏好优化，实现无需专家标注的高质量半监督医学分割。

## 方法详解

### 整体框架

输入医学图像 → CLIP 提供语义级特征（识别目标区域的语义信息）+ VQA 模块通过问答获取位置和形状先验 → 融合生成 SAM 的提示（点/框提示或密集提示）→ SAM 生成多个候选分割 → 虚拟标注器对候选分割进行排名/评分 → DPO 偏好优化训练最优分割策略。全流程无需人工标注介入。

### 关键设计

1. **无监督提示生成模块（CLIP + VQA）**:

    - 功能：在无人工标注的情况下，自动为 SAM 生成包含语义、位置和形状信息的高效提示
    - 核心思路：利用 CLIP 的对比学习预训练能力做语义级别的区域定位——将医学术语（如"lung"、"tumor"）与图像区域对齐，找到最可能包含目标的位置。同时利用 VQA 模型回答关于目标形状、大小、边界等问题，补充空间先验。两者融合后生成 SAM 可以接受的提示格式（如边界框或点标注）
    - 设计动机：传统 SAM 提示需要专家手动提供，成本高且不可扩展。CLIP 和 VQA 都是预训练模型，不需要针对具体任务的标注数据，天然适合无监督场景

2. **DPO 偏好优化策略**:

    - 功能：通过偏好学习而非传统监督学习来优化 SAM 的分割输出质量
    - 核心思路：借鉴 LLM 领域的 DPO（Direct Preference Optimization），将分割任务重构为偏好学习问题。给定同一图像的两个候选分割结果，由虚拟标注器判断哪个更好（而非提供像素级标签），模型学习一个最优策略使其输出的分割结果符合偏好排序。DPO 的核心优势是不需要显式的奖励模型，直接从偏好对中学习
    - 设计动机：医学分割的 ground truth 标注成本高，但"哪个分割更好"的相对判断远比"像素级精确标注"简单得多。DPO 将标注需求从绝对判断降级为相对排名

3. **虚拟标注器（Virtual Annotator）**:

    - 功能：模拟人类标注过程，为候选分割结果提供评分或排名
    - 核心思路：设计一个自动化的评估器，基于分割结果的连通性、边界平滑度、与 CLIP 语义特征的一致性等指标，自动对多个候选分割进行排序。这个虚拟标注器替代了传统 RLHF 中的人类标注环节，使整个流程完全自动化
    - 设计动机：DPO 需要偏好数据（win/lose对），人类标注者成本高且不一致。虚拟标注器通过可计算的质量指标提供一致的、可扩展的偏好信号

### 损失函数 / 训练策略

采用 DPO 损失函数训练分割策略：给定偏好对 $(y_w, y_l)$（win 分割和 lose 分割），DPO 损失鼓励模型输出更接近 $y_w$ 的分割结果。训练分为两阶段：先用无监督提示生成初始候选分割并收集偏好数据，再用 DPO 优化模型策略。

## 实验关键数据

### 主实验

| 任务 | 模态 | 指标 | 本文框架表现 |
|------|------|------|------------|
| 肺分割 | X-ray | Dice / IoU | SOTA |
| 乳腺肿瘤分割 | 超声 | Dice / IoU | SOTA |
| 器官分割 | 腹部CT | Dice / IoU | SOTA |

方法在三种不同模态的医学分割任务上均达到了最优表现，证明了框架的跨模态泛化能力。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Full model（CLIP + VQA + DPO） | 最优 | 完整框架 |
| w/o DPO（仅用无监督提示） | 下降 | 偏好优化对提升分割质量至关重要 |
| w/o VQA（仅用 CLIP 提示） | 下降 | VQA 提供的形状/位置先验有贡献 |
| w/o 虚拟标注器（用随机偏好） | 显著下降 | 虚拟标注器的质量判断是 DPO 有效性的关键 |

### 关键发现

- 无监督提示生成的质量出乎意料地好——CLIP 的跨模态语义对齐能力在医学图像上也有效
- DPO 偏好优化相比传统半监督方法的优势在于：不需要精确的伪标签，只需要"哪个更好"的相对判断
- 虚拟标注器的设计是方法有效性的关键环节——随机偏好对训练几乎没有帮助
- 跨三种模态的一致提升说明框架的泛化性好，不依赖于特定模态的先验知识

## 亮点与洞察

- **将 DPO 引入医学分割是新颖的跨领域迁移**——DPO 原本用于 LLM 对齐，本文巧妙地将其应用到分割任务中，将标注需求从"像素级标签"降级为"排名/评分"，大幅降低了标注成本
- **CLIP + VQA 做无监督提示生成**的思路可以迁移到其他需要 SAM 但缺乏标注的领域（如遥感、工业检测）
- **虚拟标注器替代人类标注**的设计将 RLHF 的成本进一步降低到零人工，是自动化 alignment 的有价值探索

## 局限与展望

- 虚拟标注器的质量上界受限于其设计的启发式指标——如果指标不能准确反映分割质量，DPO 优化方向可能偏差
- CLIP 的医学图像理解能力有限——通用CLIP可能对专业医学术语和罕见病变的定位不够精确，可考虑使用 BiomedCLIP 等医学专用预训练模型
- 半监督设置下与全监督方法的差距未明确量化——需要了解在不同标注比例下性能曲线
- 未探索 3D 医学分割场景（如 3D CT/MRI volume）——当前验证集中在 2D 切片

## 相关工作与启发

- **vs MedSAM**: MedSAM 通过大规模医学数据微调 SAM，仍依赖大量标注。本文方法不需要微调 SAM 本身，而是优化提示和策略
- **vs SAMed**: SAMed 用 LoRA 适配 SAM 到医学领域，是监督方法。本文创新点在于半监督 + DPO
- **vs 传统半监督分割（如 Mean Teacher, FixMatch）**: 传统方法依赖伪标签的一致性正则化，本文用偏好优化替代，避免了伪标签噪声的累积问题
- 该框架的 DPO + 虚拟标注器范式可能启发其他视觉基础模型（如 DINO、GroundingDINO）在低标注场景的应用

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 DPO 偏好优化引入医学分割是巧妙的跨领域迁移
- 实验充分度: ⭐⭐⭐⭐ 三种模态三种任务覆盖面广，但缺乏具体数值表格
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法流程直观
- 价值: ⭐⭐⭐⭐ 对低标注医学分割有实际价值，DPO+虚拟标注器范式可迁移

<!-- RELATED:START -->

## 相关论文

- [InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](inpo_inversion_preference_optimization_with_reparametrized_ddim_for_efficient_di.md)
- [Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](../../ACL2025/llm_alignment/fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](../../ACL2025/llm_alignment/kpo_protein_safety.md)
- [DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](../../NeurIPS2025/llm_alignment/dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)

<!-- RELATED:END -->
