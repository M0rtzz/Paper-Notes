---
title: >-
  [论文解读] Effective Training Data Synthesis for Improving MLLM Chart Understanding
description: >-
  [ICCV 2025][多模态VLM][图表理解] 提出模块化的五步图表数据合成流水线，生成包含10k+图表图像和300k+ QA对的高质量训练集ECD（Effective Chart Dataset），在多种开源MLLM上一致提升图表理解能力。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "图表理解"
  - "数据合成"
  - "多模态大语言模型"
  - "训练数据"
  - "数据质量"
---

# Effective Training Data Synthesis for Improving MLLM Chart Understanding

**会议**: ICCV 2025  
**arXiv**: [2508.06492](https://arxiv.org/abs/2508.06492)  
**代码**: [https://github.com/yuweiyang-anu/ECD](https://github.com/yuweiyang-anu/ECD)  
**领域**: 多模态VLM  
**关键词**: 图表理解, 数据合成, 多模态大语言模型, 训练数据, 数据质量

## 一句话总结

提出模块化的五步图表数据合成流水线，生成包含10k+图表图像和300k+ QA对的高质量训练集ECD（Effective Chart Dataset），在多种开源MLLM上一致提升图表理解能力。

## 研究背景与动机

图表理解是构建科学AI agent的核心能力之一，但现有开源MLLM在挑战性benchmark上的成功率仅30%-50%。虽然图表可以程序化精确合成（相比自然图像有天然优势），但已有合成图表训练集存在显著问题：

1. 早期数据集（PlotQA、OpenCQA）使用固定代码模板，图表类型少、样式单一
2. ChartBench虽分离数据生成，但可视化代码多样性不足
3. ReachQA让LLM同时生成代码和数据，反而限制了数据复杂度

这些限制导致合成数据与真实科学论文中的图表差异大，微调效果不理想。本文的核心思路是：**通过模块化和多样化来缩小合成图表与真实图表之间的差距**。

## 方法详解

### 整体框架

五步数据合成流水线：单图生成 → 组合子图生成 → 视觉多样化 → 质量过滤 → QA对生成与过滤。

### 关键设计

1. **模块化单图生成**: 将图表函数与数据生成解耦。人工预定义29种图表函数（每种包含参数化的Python绘图代码），给GPT-4o三个输入：(1)图表主题、(2)图表函数、(3)参数描述+few-shot示例，让GPT专注生成数据表和文本元素。分步生成确保数据分布更丰富、数据值与文本元素语义关联。共生成10,875个单图。

2. **条件式子图组合**: 生成多子图时采用迭代条件生成——生成第3个子图时参考前2个子图的数据，确保主题一致性。模拟科学论文中多子图展示互补数据视角的习惯。共生成6,006个多子图图表，平均4个子图/图。

3. **视觉多样化**: 用GPT-4o修改Python绘图代码，随机添加注释、箭头、区域阴影、缩放框、副标题等视觉元素，修改字体颜色/样式/大小，使用Seaborn等额外库提升美观度。同时进行后处理调整figsize/dpi等参数。

4. **双指标质量过滤**: 使用GPT-4o评估两个维度——视觉清晰度 $r_{vis}(\mathbf{x}, c_{layout})$ 和语义一致性 $r_{sem}(\mathbf{x}, c_{theme})$，保留高于均分的图表。从16,829张过滤至10,535张（过滤率37.4%）。

5. **QA对生成与过滤**: GPT-4o基于图表图像+代码+数据生成描述性和推理性QA对，要求模型给出1-5信心分，仅保留满分5分的QA对。从348,862过滤至321,544（过滤率7.8%）。

### 损失函数 / 训练策略

- 对4种开源MLLM进行微调：LLaVA-Next-Llama3-8B（LoRA）、MiniCPM-V2.6（LoRA）、Phi-3-Vision（全参数）、Qwen2.5-VL-7B（LoRA）
- 冻结vision tower，仅微调其余部分
- 训练1个epoch，学习率1e-4（LoRA）或5e-6（全参数）
- 评估指标：GPT-Acc（使用GPT-4o提取答案并评估正确性）

## 实验关键数据

### 主实验 (ECD微调效果)

| 模型 | CharXiv Avg | ChartQA | ChartX | ECDBench Avg |
|------|------------|---------|--------|-------------|
| LLaVA-Next-8B | 35.06 | 64.56 | 27.69 | 10.95 |
| + ECD | **51.60** (+16.54) | **68.64** (+4.08) | **46.61** (+18.92) | **31.58** (+20.63) |
| Phi-3-Vision | 54.72 | 81.92 | 67.53 | 31.41 |
| + ECD | **61.08** (+6.36) | **84.88** (+2.96) | **71.44** (+3.91) | **44.40** (+12.99) |
| Qwen2.5-VL-7B | 61.36 | 83.04 | 67.80 | 38.19 |
| + ECD | **67.40** (+6.04) | **85.32** (+2.28) | **70.83** (+3.03) | **50.86** (+12.67) |

ECD在4个MLLM上6个测试集上整体一致提升。

### 消融实验 (设计选择影响)

**与其他训练集对比 (LLaVA-Next基线)**:

| 训练集 | CharXiv | ChartQA | ReachQA | ChartX | ECDBench |
|--------|---------|---------|---------|--------|----------|
| 无微调 | 35.06 | 64.56 | 15.65 | 27.69 | 10.95 |
| ChartQA | 35.16 | **68.92** | 15.00 | 31.51 | 13.11 |
| ChartBench | 32.86↓ | 61.56↓ | 18.35 | 37.33 | 10.99 |
| ReachQA | 30.68↓ | 64.50 | **24.35** | 39.24 | 13.48 |
| **ECD** | **51.60** | 68.64 | 25.10 | **46.61** | **31.58** |

其他训练集往往只提升自身分布相似的测试集，甚至导致其他测试集下降。ECD是唯一在所有6个测试集上都一致提升的训练数据。

**数据规模效应**: 2k→40k图像逐步提升性能，ReachQA持续提升（18.25→24.75），CharXiv在20k后饱和。

**视觉多样化效果**: FID降低19.64（80.38→60.74），平均熵提升0.57（1.67→2.24），证实多样化显著缩小了与真实图表的分布差距。

### 关键发现

- **模块化是关键**: 分离函数和数据生成比端到端生成更有效，让LLM专注数据复杂度
- **图表类型多样性持续有利**: 从5种→29种图表类型在CharXiv上持续提升
- **推理QA的增益大于描述QA**: 单独使用推理QA提升6.40%，描述QA提升4.16%，两者结合最优（47.02%）
- **ECD的FID最低**: 与CharXiv（真实图表）的FID为60.74，远低于其他合成数据集，说明ECD与真实科学图表更相似
- **QA过滤有正收益** 但幅度不大（0.38%提升），图像质量过滤更重要
- 少量指标出现下降（如ChartBench的Binary问题），因其分布与ECD差异较大

## 亮点与洞察

- **理念简洁有效**: 不靠更大的模型或更多数据，而是通过更好的数据合成流程提升质量
- **数据分析充分**: FID、平均熵、消融实验全面验证了每个流水线步骤的必要性
- **通用性强**: 在4种不同架构的MLLM上都有效，且不需要改模型架构
- **ECDBench构建严谨**: 两阶段人工审核确保测试集质量

## 局限与展望

- 10k图像规模相对较小，扩大数据量可能进一步提升（特别是预训练阶段）
- 对已有强大图表预训练的模型（如Qwen2.5-VL）部分指标提升有限甚至略降
- 描述/推理QA最优比例尚未充分探索（实验显示1:1或2:3较优）
- 依赖GPT-4o生成数据和评估质量，成本较高
- 29种图表函数为人工定义，覆盖面可能仍不足（如3D图表、网络图等）

## 相关工作与启发

- CharXiv benchmark展示了真实图表理解的挑战性（开源模型仅30-50%）
- 程序化图像（图表、SVG）与自然图像的合成逻辑完全不同，可以做到无损精确合成
- 条件式子图生成确保多图一致性，是一个值得借鉴的设计模式

## 评分

- 新颖性: ⭐⭐⭐⭐ 五步流水线设计系统性强，模块化思想有启发
- 实验充分度: ⭐⭐⭐⭐⭐ 4个模型、6个测试集、详细消融和对比分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表可视化quality高
- 价值: ⭐⭐⭐⭐ 数据合成的系统方法论，对其他可编程图像领域也有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[CVPR 2025\] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](../../CVPR2025/multimodal_vlm/img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
