---
title: >-
  [论文解读] CheX: Interactive Localization and Region Description in Chest X-rays
description: >-
  [ECCV 2024][医学图像][胸部X光] 提出ChEX——一个同时支持文本提示和边界框查询的交互式胸部X光解释模型，通过DETR风格的prompt检测器和多任务联合训练，在9个胸部X光任务上与SOTA竞争，同时提供独特的定位可解释性和用户交互能力。
tags:
  - ECCV 2024
  - 医学图像
  - 胸部X光
  - 报告生成
  - 视觉定位
  - 交互式诊断
  - 多任务学习
---

# CheX: Interactive Localization and Region Description in Chest X-rays

**会议**: ECCV 2024  
**arXiv**: [2404.15770](https://arxiv.org/abs/2404.15770)  
**代码**: https://github.com/philip-mueller/chex (有)  
**领域**: 医学图像  
**关键词**: 胸部X光, 报告生成, 视觉定位, 交互式诊断, 多任务学习

## 一句话总结

提出ChEX——一个同时支持文本提示和边界框查询的交互式胸部X光解释模型，通过DETR风格的prompt检测器和多任务联合训练，在9个胸部X光任务上与SOTA竞争，同时提供独特的定位可解释性和用户交互能力。

## 研究背景与动机

胸部X光的自动报告生成模型虽然进展迅速，但在临床应用上面临两个核心障碍：**缺乏可解释性**（模型决策过程不透明，医生无法验证预测依据）和**缺乏交互性**（输出固定，无法根据用户关注点调整）。

现有工作各有局限：RGRG [Tanida et al.] 通过解剖区域的边界框提供了一定可解释性，但不支持文本查询且仅聚焦解剖结构。RaDialog、Med-PaLM M等支持文本交互但无法预测边界框进行视觉定位。OmniFM-DR虽然可以为文本提示预测边界框，但不会描述框内内容。

**核心矛盾**：没有任何现有模型能同时具备"文本/边界框双向交互"和"视觉定位可解释性"。

**本文切入角度**：设计统一的多任务架构，将文本提示和边界框融合为查询机制，在同一框架内支持定位、分类、区域描述、句子定位和全文报告生成等多种任务，通过多数据集联合训练获得零样本泛化能力。

## 方法详解

### 整体框架

ChEX的pipeline分为四个阶段：(1) **图像编码器**提取胸部X光的patch特征；(2) **Prompt编码器**（冻结的CLIP文本编码器）将文本查询编码为prompt token；(3) **Prompt检测器**（DETR风格解码器）基于prompt token和patch token预测边界框和ROI特征；(4) **句子生成器**（GPT2-medium）基于ROI token独立生成每个区域的文本描述。

当用户提供边界框查询时，直接通过Gaussian ROI Pooling计算ROI特征，跳过检测器的文本处理部分。

### 关键设计

1. **Prompt检测器（DETR-style Decoder）**: 对每个prompt token预测 $M=3$ 个边界框。具体做法是将每个prompt token与 $M$ 个可学习token相加，形成 $Q \times M$ 个decoder query tokens。经过6层DETR decoder处理后，用MLP预测框坐标和置信度。然后通过**Gaussian ROI Pooling**在patch token上计算框特征，最终对 $M$ 个框特征按置信度加权平均得到每个prompt的ROI token。设计动机：支持单个查询对应多个定位区域（如双侧胸腔积液），同时通过随机skip connection保证梯度流。

2. **多类型Prompt Token**: 训练时使用三类prompt：(a) **病理token**——预定义的病理名称（如"pleural effusion"）；(b) **解剖token**——解剖区域名称（如"right lung"）；(c) **句子token**——报告中的每个句子。不同样本根据可用标注选择性使用对应类型的token。这种设计使模型在不同类型的监督信号下学会统一的定位-描述能力。

3. **边界框查询模式**: 除文本查询外，随机选择部分batch使用目标边界框直接通过Gaussian ROI Pooling计算ROI特征，跳过检测和编码。这使得模型在推理时可以同时支持文本和边界框两种查询方式，且两者可以组合使用以获得更精确的预测。

4. **句子生成器**: 使用GPT2-medium（PubMed预训练），通过P-tuning v2将每个ROI token作为条件独立生成描述。额外引入3层post decoder（cross-attention到patch features）以注入全局上下文。

### 损失函数 / 训练策略

- **边界框损失**：修改版DETR匈牙利匹配，使用L1 + gIoU损失，省略了交叉熵而改用Focal Loss训练框置信度
- **病理分类损失**：InfoNCE对比损失，将ROI token与病理prompt配对（正例如"pleural effusion"，负例如"no pleural effusion"或其他不存在的病理）
- **文本生成损失**：自回归语言建模 + ROI token与对应句子的对比学习 + 全局CLIP损失

训练数据：MIMIC-CXR（227K图像，含Chest ImaGenome的29个解剖区域框+53类标签）+ VinDr-CXR（15K图像，22类病理框）。对VinDr-CXR进行过采样平衡数据量。

## 实验关键数据

### 主实验

ChEX在9个任务的综合评估中，在8/9个任务上与最佳baseline竞争（1-std内）或更优。

| 任务/数据集 | 指标 | ChEX | 最佳Baseline | 说明 |
|-------------|------|------|-------------|------|
| SG/MS-CXR | mAP | **44.47** | 44.05 (SupVG) | 与专门训练的TransVG齐平 |
| OD/NIH8 | mAP | **11.14** | 6.69 (SupOD) | 几乎2倍于最佳监督检测器 |
| OD/MS-CXR | mAP | **16.60** | 15.83 (SupOD) | 优于监督和弱监督检测器 |
| RC/MS-CXR | AUROC | **82.33** | 76.13 (SupOD) | 提升8% |
| RC/CIG | wAUROC | **70.46** | 66.96 (Contrastive) | 提升5% |
| RE/CIG | METEOR | **10.18** | 7.88 (RGRG) | 提升29% |
| RE/CIG | Mac-F1-14 | **29.13** | 20.88 (RGRG) | 提升40% |
| RG/MIMIC-CXR | Ex-F1-14 | **58.76** | 47.6 (Med-PaLM M) | 提升23%，新SOTA |
| RG/MIMIC-CXR | Mic-F1-14 | 52.32 | 55.7 (MAIRA-1) | MAIRA-1是7倍大的模型 |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| 仅病理token训练 | OD性能好但RC/RE大幅下降 | 解剖token对区域任务至关重要 |
| 仅解剖token训练 | RC/RE最优但OD/SG下降 | 病理token对定位任务不可缺 |
| 去掉句子token | 定位任务略有提升，文本生成下降 | 句子token主要服务于生成质量 |
| 去掉框监督 | 所有任务全面下降 | 边界框监督是最重要的 |
| 去掉对比学习 | OD和部分RC/RE下降 | 对比学习增强了病理区域的理解 |
| 报告级生成（非区域级） | Mic-F1-14下降约3-5% | 区域级逐句生成是ChEX强性能的关键 |

### 关键发现

- **交互能力**：添加粗粒度区域提示（如"左肺"）可显著改善定位质量；精细提示（如"左上肺"）进一步小幅提升
- **方向引导**：当查询指向无病理的对侧肺时，模型会正确转向该区域（而非始终指向病理），表明模型理解用户意图
- **文本+框组合查询**：同时提供文本和框可获得最佳预测准确率
- **Prompt集可定制**：使用不同的prompt集可平衡精度和召回率，Mic-F1-14范围50.08-52.37

## 亮点与洞察

- **独特定位**：唯一同时支持"文本提示→框+描述"和"框查询→描述"的医学影像模型
- **小模型大能力**：ChEX参数量仅为Med-PaLM M的1/10、MAIRA-1的1/7，却在多数任务上竞争甚至超越
- **多数据集融合训练**：一套架构无缝整合框标注、分类标签、报告文本三种异构监督
- 输出自带边界框的报告可直接辅助放射科医生快速核查

## 局限与展望

- 文本查询仅限于区域提示或病理名称，不支持复杂推理问题（如"比较左右肺"）
- 答案基于报告句子，可能出现与前片对比的幻觉（仅使用单张图像）
- 未来可结合instruction tuning或LLM提升对话能力
- 缺少放射科医生的系统化用户体验评估

## 相关工作与启发

- 与RGRG相比，ChEX扩展了token类型（病理+句子）和对比学习，带来全面的性能提升
- 从DETR到医学应用的成功迁移表明，以区域为中心的目标检测范式对医学影像理解有天然优势
- 多任务训练的溢出效应（如报告生成从定位任务中受益）值得在更多医学场景中验证

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在医学影像中实现双向交互+视觉定位的统一模型
- 实验充分度: ⭐⭐⭐⭐⭐ 9个任务、多个数据集、充分的消融和交互分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，交互案例直观
- 价值: ⭐⭐⭐⭐ 为医学影像交互式诊断提供了实用且可扩展的基础架构

<!-- RELATED:START -->

## 相关论文

- [CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](../../NeurIPS2025/medical_imaging/cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](../../CVPR2026/medical_imaging/instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence](../../AAAI2026/medical_imaging/human-in-the-loop_interactive_report_generation_for_chronic_disease_adherence.md)
- [Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](../../CVPR2026/medical_imaging/adaptation_of_weakly_supervised_localization_in_hi.md)

<!-- RELATED:END -->
