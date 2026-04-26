---
title: >-
  [论文解读] PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow
description: >-
  [CVPR 2026][图像生成][自动图形设计] 本文提出PSDesigner，一个模拟人类设计师创意工作流的自动图形设计系统，通过AssetCollector（资源收集）、GraphicPlanner（规划工具调用）和ToolExecutor（执行PSD操作）三个模块协作，利用首个PSD格式设计数据集CreativePSD训练模型学习专业设计流程，能直接生成可编辑的PSD设计文件。
tags:
  - CVPR 2026
  - 图像生成
  - 自动图形设计
  - PSD文件操作
  - 工具调用
  - 强化学习
  - 创意工作流
---

# PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow

**会议**: CVPR 2026  
**arXiv**: [2603.25738](https://arxiv.org/abs/2603.25738)  
**代码**: https://henghuiding.com/PSDesigner/  
**领域**: 图像生成 / 自动设计  
**关键词**: 自动图形设计, PSD文件操作, 工具调用, 强化学习, 创意工作流

## 一句话总结

本文提出PSDesigner，一个模拟人类设计师创意工作流的自动图形设计系统，通过AssetCollector（资源收集）、GraphicPlanner（规划工具调用）和ToolExecutor（执行PSD操作）三个模块协作，利用首个PSD格式设计数据集CreativePSD训练模型学习专业设计流程，能直接生成可编辑的PSD设计文件。

## 研究背景与动机

1. **领域现状**：图形设计在电商和广告中至关重要。现有自动化方法主要分两类：(a) 文生图模型（FLUX、Glyph-Byt5等）生成设计图片；(b) MLLM驱动的方法（LaDeCo、COLE等）直接生成JSON格式的可编辑设计文件。

2. **现有痛点**：文生图方法生成的图片不可编辑且文字渲染不准确（尤其中文）；MLLM方法将图层按预定义类别（underlay/text）分组一次性预测所有属性，设计过程不直观且灵活性有限。

3. **核心矛盾**：现有方法大幅简化了专业设计流程——(a) 按类别分组不如按视觉概念分组直观；(b) 一次性预测所有图层属性缺乏渐进式设计的灵活性；(c) 只能处理简单的图层层次和有限的属性类型，距离产品级设计差距很大。

4. **本文目标** 构建一个模拟人类设计师工作流的自动设计系统，能处理复杂的PSD图层层次结构（平均48.35层），支持丰富的图层类型和60+种属性，生成可编辑的专业级PSD文件。

5. **切入角度**：观察人类设计师的工作流——先收集主题资源，然后按视觉概念分组迭代集成资源，每步集成后检修缺陷——将这一过程形式化为VLM的工具调用预测问题。

6. **核心 idea**：将图形设计建模为VLM的工具调用序列预测，通过SFT+GRPO训练使模型学会迭代式的资源集成和缺陷修复操作。

## 方法详解

### 整体框架

给定用户指令，PSDesigner分三步工作：(1) AssetCollector利用LLM识别视觉概念并为每个概念收集相关资源（图片/文字）；(2) 对嵌套式图层层次进行自底向上遍历，每个迭代中GraphicPlanner先预测资源集成的工具调用（$\mathcal{X}_{gen}$模式），再预测缺陷修复的工具调用（$\mathcal{X}_{edt}$模式）；(3) ToolExecutor在Adobe Photoshop中执行这些工具调用，通过UXP API实现70+种PSD操作。

### 关键设计

1. **CreativePSD数据集（首个PSD格式设计数据集）**:

    - 功能：为GraphicPlanner提供专业设计操作的监督训练数据
    - 核心思路：三阶段构建——Stage I: 从互联网和付费来源收集高质量PSD文件，由专业标注人员按视觉概念分组图层；Stage II: 解析PSD文件提取原始资源、元数据和中间渲染结果；Stage III: 从提取信息中构建$\mathcal{X}_{gen}$（资源集成）和$\mathcal{X}_{edt}$（缺陷修复）两种模式的训练数据。每个训练样本为 $(a, \mathcal{C}, x)$ 三元组：资源+观察+工具调用序列
    - 设计动机：现有数据集（CGL/Crello/Design39K）平均仅4-5层、2种图层类型和有限属性；CreativePSD有10,454样本，平均48.35层，5种图层类型和60+种属性，能学习真实设计流程

2. **GraphicPlanner（双模式VLM工具调用预测器）**:

    - 功能：基于当前设计状态预测下一步的PSD操作工具调用
    - 核心思路：基于Qwen2.5-VL-7B构建，注入模式特定的LoRA模块。$\mathcal{X}_{gen}$模式接收资源$a$和观察$(M, R)$（图层元数据+当前渲染），预测集成工具调用；$\mathcal{X}_{edt}$模式接收观察$(M, R, G)$（含扰动元数据+渲染+组前渲染），预测修复工具调用。两阶段训练：先SFT学习基本的工具-参数映射，再用GRPO强化学习精炼参数值的精确度
    - 设计动机：资源集成和缺陷修复是两种本质不同的操作，使用模式特定的LoRA避免任务干扰；GRPO的奖励函数直接比较工具名和参数值的正确性，提升了工具调用的精确度

3. **ToolExecutor（基于UXP的PSD操作执行器）**:

    - 功能：将GraphicPlanner预测的工具调用转化为实际的PSD文件操作
    - 核心思路：基于Adobe UXP框架用JavaScript API实现了70+种Photoshop操作，包括插入图片/文字/调整图层、应用效果（内发光、投影等）、设置混合模式、裁切蒙版等
    - 设计动机：直接操作PSD文件而非JSON格式，能支持产品级设计中的复杂图层属性和效果配置

### 损失函数 / 训练策略

SFT阶段使用标准的自回归交叉熵损失。GRPO强化学习阶段设计了专用奖励函数$r$，比较生成的工具调用与ground truth的工具名、参数名和参数值。SFT训练15,000步（$\mathcal{X}_{gen}$）/ 12,000步（$\mathcal{X}_{edt}$），batch=64, lr=2e-4, LoRA rank=32。GRPO训练6,000步，group size=8，使用4,000个PSD文件。

## 实验关键数据

### 主实验

用户意图→设计（VLM评分，满分10分）：

| 方法 | 质量 | 布局 | 相关性 | 和谐性 | 创新性 | 可编辑 |
|------|------|------|--------|--------|--------|--------|
| **PSDesigner** | 7.62 | **8.68** | 7.78 | 8.02 | **8.45** | ✓ PSD |
| CanvaGPT | **8.52** | 8.15 | 4.72 | 7.21 | 7.52 | ✓ |
| FLUX | 8.18 | 6.88 | 6.92 | 6.82 | 6.95 | ✗ |
| PosterCraft | 7.95 | 8.35 | **8.42** | **8.05** | 5.87 | ✗ |
| OpenCOLE | 5.12 | 3.66 | 5.25 | 6.68 | 6.08 | 部分 |

Crello-v5上的设计组合（VLM评分）：

| 方法 | 质量 | 布局 | 和谐性 | 创新性 |
|------|------|------|--------|--------|
| **Ours** | **7.85** | **7.43** | 6.77 | **6.94** |
| LaDeCo | 5.95 | 6.03 | **7.22** | 5.75 |
| Ground Truth | 8.13 | 9.18 | 8.90 | 7.12 |

### 消融实验

| 配置 | 质量 | 布局 | 和谐性 | 创新性 |
|------|------|------|--------|--------|
| Full model (Crello) | **7.85** | **7.43** | **6.77** | **6.94** |
| w/o $\mathcal{X}_{edt}$ | 6.05 | 5.88 | 5.90 | 6.75 |
| w/o 层信息M | 6.25 | 6.10 | 6.18 | 6.02 |
| w/o RL (GRPO) | 6.38 | 6.00 | 6.35 | 6.20 |
| Full model (PSD) | **6.28** | **6.15** | **7.02** | **6.88** |
| w/o $\mathcal{X}_{edt}$ (PSD) | 5.32 | 5.15 | 6.22 | 6.05 |

### 关键发现

- 去掉$\mathcal{X}_{edt}$模式（缺陷修复）对质量和布局影响最大（Crello上分别下降1.80和1.55分），说明迭代修复是设计质量的关键
- 去掉层信息M导致模型无法感知同组其他元素，布局和和谐性显著下降
- GRPO强化学习对精确预测工具调用的参数值至关重要，去掉后布局下降1.43分
- PSDesigner是唯一能同时生成可编辑PSD文件且准确渲染中文文字的系统

## 亮点与洞察

- **工作流拟人化设计**：将设计过程分解为"收集→集成→修复"的迭代流程，与人类设计师的思维方式高度一致，这种问题分解思路可迁移到其他需要多步骤创作的任务（如PPT制作、视频剪辑）
- **CreativePSD数据集**：首个PSD格式的设计训练数据集，平均48层、60+属性，极大地扩展了自动设计系统能处理的复杂度层级。数据构建的三阶段pipeline（收集→解析→构建训练数据）也是可复用的方法论
- **GRPO用于工具调用精炼**：将强化学习应用于提升工具调用参数的精确度是一个巧妙的设计，因为SFT只能学习到近似的参数值分布，而GRPO的奖励信号能直接优化输出与GT的匹配度

## 局限与展望

- 评估主要依赖VLM打分和用户研究，缺乏更客观的定量指标（如图层属性预测精度）
- AssetCollector依赖外部图片搜索/生成模型的质量，资源收集的失败会级联影响设计质量
- 仅支持静态设计，未涉及动态/交互式设计（如网页、动画）
- 数据集规模（10,454样本）相比LLM训练数据仍较小，可能限制泛化性
- 与Photoshop的深度耦合（UXP API）限制了向其他工具的扩展

## 相关工作与启发

- **vs LaDeCo**: LaDeCo按预定义类别分组(image/text)，一次性预测同类所有层属性，处理不了复杂层次。PSDesigner按视觉概念分组、迭代集成，在Crello上质量高出1.9分
- **vs COLE/OpenCOLE**: COLE构建多个任务特定模型管线，但输出的可编辑性有限（仅单图层+文字层）。PSDesigner统一用VLM驱动工具调用，支持完整PSD层次
- **vs T2I (FLUX/PosterCraft)**: 这些方法生成视觉质量高但不可编辑的光栅图片，且中文/复杂文字渲染经常出错

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将VLM工具调用范式应用于PSD级别的自动设计，CreativePSD数据集也是首创
- 实验充分度: ⭐⭐⭐⭐ 对比了多种方法，消融实验充分，但缺乏工具调用精度等客观指标
- 写作质量: ⭐⭐⭐⭐ 人类设计师工作流与PSDesigner的对比图非常直观，问题motivate清晰
- 价值: ⭐⭐⭐⭐ 对自动设计领域有重要推动，展示了VLM+工具调用在创意任务中的潜力

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement](showtable_unlocking_creative_table_visualization_with_collaborative_reflection_a.md)
- [\[CVPR 2025\] From Elements to Design: A Layered Approach for Automatic Graphic Design Composition](../../CVPR2025/image_generation/from_elements_to_design_a_layered_approach_for_automatic_graphic_design_composit.md)
- [\[ICCV 2025\] Rethinking Layered Graphic Design Generation with a Top-Down Approach](../../ICCV2025/image_generation/rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)
- [\[CVPR 2026\] GIST: Towards Design Compositing](gist_towards_design_compositing.md)
- [\[CVPR 2026\] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation](ar2can_an_architect_and_an_artist_leveraging_a_canvas_for_multi-human_generation.md)

<!-- RELATED:END -->
