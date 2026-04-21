---
title: >-
  [论文解读] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events
description: >-
  [NeurIPS 2025][遥感][remote sensing] 构建了RSCC——首个大规模灾害感知遥感变化描述数据集（62,351对灾前/灾后图像+详细变化描述），覆盖地震/洪水/野火等31个全球事件，利用QvQ-Max视觉推理模型生成高质量标注，并建立了全面的基准评测体系。
tags:
  - NeurIPS 2025
  - 遥感
  - remote sensing
  - change captioning
  - disaster monitoring
  - bi-temporal
  - 视觉语言
---

# RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events

**会议**: NeurIPS 2025  
**arXiv**: [2509.01907](https://arxiv.org/abs/2509.01907)  
**代码**: https://github.com/Bili-Sakura/RSCC (有)  
**领域**: 遥感 / 视觉语言  
**关键词**: remote sensing, change captioning, disaster monitoring, bi-temporal, vision-language model

## 一句话总结
构建了RSCC——首个大规模灾害感知遥感变化描述数据集（62,351对灾前/灾后图像+详细变化描述），覆盖地震/洪水/野火等31个全球事件，利用QvQ-Max视觉推理模型生成高质量标注，并建立了全面的基准评测体系。

## 研究背景与动机

**领域现状**：时序遥感图像对于灾害监测至关重要，需要对灾前灾后的变化进行详细的文本描述和分析。多模态大语言模型(MLLM)在自然图像理解方面取得了很大进展，但在时序遥感图像理解方面仍未被充分探索。**现有痛点**：现有遥感图像-文本数据集存在三大缺陷——要么是单时间快照缺乏时序信息(UCM-Captions, RSICD)，要么有时序但无文本标注(fMoW, SpaceNet 7)，要么规模小且描述简短缺乏灾害上下文(LEVIR-CC: 20K对/40词，Dubai-CCD: 1K对/35词)。**核心矛盾**：缺少一个同时具备"大规模+灾害特定+双时相+高质量长文本描述"的数据集来训练和评估视觉语言模型。**切入角度**：利用xBD和EBD两个建筑损伤评估数据集的图像和标注信息，结合QvQ-Max视觉推理模型自动生成详细的变化描述文本。

## 方法详解

### 整体框架
RSCC构建流程分为四步：(1) 从xBD(MAXAR OpenData)和EBD两个建筑损伤评估数据集获取灾前/灾后图像对和建筑损伤标注，xBD图像从1024×1024裁剪为512×512无重叠，EBD保持原始512×512分辨率；(2) 将建筑损伤标签(基于Joint Damage Scale的no damage/minor/major/destroyed四级)转化为结构化辅助信息，用不同颜色边界框标注在灾后图像上作为visual prompt；(3) 构建<任务指令><灾害描述><建筑损伤详情><输出格式>的结构化文本prompt，调用QvQ-Max(qvq-max-2025-03-25) API生成变化描述文本；(4) 通过Qwen2.5-Max自动后处理修正元数据不一致+10%随机抽样的3人专家验证确保标注质量。最终数据集分为61,363对训练集(31个事件)和988对测试集(19个事件)。

### 关键设计

1. **基于视觉推理的标注生成**:

    - 功能：自动为62,351对图像生成高质量变化描述
    - 核心思路：使用QvQ-Max（阿里巴巴的视觉推理模型），输入包括原始灾前图像+带建筑损伤边界框标注的灾后图像+结构化文本指令。关键创新是将建筑损伤评估标签(Joint Damage Scale)作为in-context辅助信息，用不同颜色的边界框标注损伤等级作为visual prompt
    - 设计动机：QvQ-Max的结构化推理能力能推断时空关系，while传统MLLM偏向识别型输出；visual prompt engineering受Shtedritski等人的marking-based方法启发

2. **两阶段后处理与质量控制**:

    - 功能：确保生成标注的可靠性和一致性
    - 核心思路：第一阶段用Qwen2.5-Max自动修正元数据不一致（灾害类型、损伤描述），将灾害类型一致性从93.2%提升到100%；第二阶段随机抽取10%样本由3位专家按4个维度(灾害类型准确性/损伤细节完整性/事实一致性/清晰度)进行二值评分，通过率100%
    - 设计动机：LLM生成的内容可能与元数据不一致，自动+人工双重保障是必要的

3. **专用变化描述模型训练**:

    - 功能：验证RSCC数据集对VLM训练的有效性
    - 核心思路：在Qwen2.5-VL 7B基础上全参数微调，使用61,363对训练集，batch_size=1，在2×H800 GPU上训练2个epoch(共40 GPU小时)。LLM backbone学习率1e-6，视觉编码器1e-5，cosine衰减
    - 设计动机：证明RSCC dataset能有效提升通用MLLM在遥感时序理解上的能力

### 损失函数 / 训练策略
标准的自回归语言建模损失。保持RSCC原始分辨率512×512作为输入。

## 实验关键数据

### 主实验

| 模型 | ROUGE(%)↑ | METEOR(%)↑ | ST5-SCS(%)↑ | Avg_L |
|------|-----------|------------|-------------|-------|
| **Ours (7B)** | **14.99** | **16.05** | **58.52** | 44 |
| InternVL 3 (8B) | 12.76 | 15.77 | 51.84 | 64 |
| TEOChat (7B) | 7.86 | 5.77 | 52.64 | 15 |
| Kimi-VL (3B) | 12.47 | 16.95 | 51.35 | 87 |
| CCExpert (7B) | 7.61 | 4.32 | 40.81 | 12 |
| Qwen2-VL (7B) | 11.02 | 9.95 | 45.55 | 42 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 零样本 vs 文本prompt vs 视觉prompt | 视觉prompt显著提升 | 建筑损伤信息辅助增强效果 |
| 不同模型规模(3B→72B) | 随规模增长但非线性 | Kimi-VL 3B超预期(51.35% ST5-SCS) |
| 缩放校正解码(VCD/DoLa/DeCo) | 无明显提升 | 遥感变化描述需要复杂视觉推理而非简单去幻觉 |
| 人类偏好研究 | QvQ-Max胜率80.7-99.0% | 对所有baseline均显著占优 |

### 关键发现
- 在RSCC上微调的7B模型(ST5-SCS 58.52%)大幅超越通用模型(Qwen2-VL: 45.55%)和专用模型(CCExpert: 40.81%)
- 专用遥感模型CCExpert和TEOChat在长文本变化描述任务上表现不佳——输出过短(Avg_L仅12和15词)，说明现有专用模型的时空推理和长文生成能力不足
- BLIP-3和LLaVA-OneVision存在严重的重复生成问题（Avg_L分别达456和221词），指向解码策略的缺陷
- 训练免费的缩放校正解码策略(VCD/DoLa/DeCo)在遥感变化描述上效果有限，作者推测原因是遥感变化描述需要复杂的视觉推理而非简单的物体级去幻觉
- 评估使用Sentence T5-XXL Embedding + Sharpened Cosine Similarity(q=0, p=3)作为语义相似度指标，比传统n-gram指标更适合长文本评估
- 模型规模不是唯一决定因素：Kimi-VL(3B)在ST5-SCS(51.35%)上超越了Qwen2-VL(7B, 45.55%)和LLaVA-OneVision(8B, 46.15%)

## 亮点与洞察
- 填补了灾害遥感+双时相+长文本描述数据集的空白，规模(62K对)远超现有同类数据集(LEVIR-CC 20K, Dubai-CCD 1K)
- Visual prompt engineering思路新颖：用彩色边界框编码损伤等级(Joint Damage Scale)作为VLM的视觉提示，显著提升描述质量
- 数据集构建成本可控（约$5/千对），具有可扩展性，为大规模遥感标注提供了可行范式
- 人类偏好评估结果(80-99%胜率)为QvQ-Max生成标注的质量提供了强有力证据
- 首次系统性研究了缩放校正解码策略(VCD/DoLa/DeCo)在遥感场景的效果，发现训练免费的correction decoding在需要复杂视觉推理的任务上效果有限
- 数据集涵盖31个全球灾害事件、6大灾害类型(地震/洪水/飓风/龙卷风/火山/野火)，地理多样性高

## 局限与展望
- 生成的描述可能包含模糊描述，即使是领域专家也难以为某些复杂变化场景确认细节
- 评估指标限于文本相似度(ROUGE/METEOR/ST5-SCS)，缺少专门的多图像描述评估指标，现有image captioning指标(FLEUR/SPARC/G-VEval)仅支持单图
- VLM在变化检测和多标签分类等视觉为中心的任务上仍远不如专用CV模型
- 灾害类型分布不均（以飓风和洪水为主），可能影响模型在少数灾害类型上的泛化
- EBD数据集缺少人工建筑损伤标注，只能使用简化的naive prompt生成描述

## 相关工作与启发
- **LEVIR-CC**(20K对/40词)/**Dubai-CCD**(1K对/35词)：此前最大的遥感变化描述数据集，规模和描述质量远不及RSCC(62K对/72词)
- **CCExpert**：基于LLaVA-OneVision的专用模型，引入差异聚焦集成组件但长文本描述能力不足（ST5-SCS仅40.81%）
- **TEOChat**：用共享视觉编码器增强LLaVA-1.5的时序理解能力，但输出过短（Avg_L=15）
- **WHU-CDC**(14.8K对)：也提供变化描述标注但无灾害特定上下文
- **Diffusion-RSCC**：概率扩散模型做RSICC，关注像素级差异
- 启发：利用强大商用VLM(QvQ-Max成本约$5/K对)自动生成高质量标注的范式具有通用性，可推广到其他遥感理解任务；visual prompt engineering（彩色边界框编码语义信息）是提升VLM遥感理解的有效手段
- xBD和EBD数据集来自MAXAR OpenData Program，确保了数据的可获取性和可复现性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个大规模灾害感知遥感变化描述数据集，visual prompt设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 覆盖10+基线模型，含人类偏好评估和多种增强策略分析
- 写作质量: ⭐⭐⭐⭐ 数据构建流程描述清晰，数据集统计详实
- 价值: ⭐⭐⭐⭐⭐ 数据集贡献大，代码和数据开源，对遥感vision-language社区有重要推动

<!-- RELATED:START -->

## 相关论文

- [CityNav: A Large-Scale Dataset for Real-World Aerial Navigation](../../ICCV2025/remote_sensing/citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)
- [GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](../../CVPR2026/remote_sensing/olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)
- [GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction](greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)
- [Resampling Augmentation for Time Series Contrastive Learning: Application to Remote Sensing](../../ICML2025/remote_sensing/resampling_augmentation_for_time_series_contrastive_learning_application_to_remo.md)

<!-- RELATED:END -->
