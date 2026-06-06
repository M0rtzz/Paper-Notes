---
title: >-
  [论文解读] CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation
description: >-
  [CVPR 2025][LLM对齐][参数化CAD生成] 本文提出 CAD-Llama 框架，通过层次化标注管线将 3D CAD 模型转化为富含语义描述的 Python 风格代码（SPCC），再用自适应预训练和指令微调将 LLaMA3-8B 转化为参数化 CAD 模型生成器…
tags:
  - "CVPR 2025"
  - "LLM对齐"
  - "参数化CAD生成"
  - "大语言模型"
  - "结构化CAD代码"
  - "层次化标注"
  - "指令微调"
---

# CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation

**会议**: CVPR 2025  
**arXiv**: [2505.04481](https://arxiv.org/abs/2505.04481)  
**代码**: 无  
**领域**: 对齐RLHF / CAD生成  
**关键词**: 参数化CAD生成, 大语言模型, 结构化CAD代码, 层次化标注, 指令微调

## 一句话总结

本文提出 CAD-Llama 框架，通过层次化标注管线将 3D CAD 模型转化为富含语义描述的 Python 风格代码（SPCC），再用自适应预训练和指令微调将 LLaMA3-8B 转化为参数化 CAD 模型生成器，在 text-to-CAD 任务上精度超出先前方法约 14%，并支持补全、添加、删除等多种 CAD 编辑任务。

## 研究背景与动机

**领域现状**：计算机辅助设计（CAD）生成建模是研究和工业界的热门方向。近年来大语言模型（LLM）在文本生成、代码生成等任务上展现了强大的泛化能力，自然引发了将 LLM 用于参数化 CAD 序列生成的探索兴趣。

**现有痛点**：现有方法面临多重挑战。首先，参数化 CAD 序列与 LLM 预训练中的自然语言/代码存在巨大鸿沟——LLM 在预训练阶段从未接触过 CAD 参数序列，也缺乏对 3D 结构的直接感知。其次，之前尝试用 LLM 生成 CAD 的工作（如 CAD-LLM、LLM4CAD）大多只能处理简单模型，泛化能力弱，无法根据复杂文本指令生成精确的 CAD 模型。第三，基于 encoder-decoder 的方法（如 Text2CAD、CAD-Translator）受限于有限的模型容量，在分布外样本上泛化性不佳。

**核心矛盾**：LLM 强大的生成先验与 CAD 数据的专业性之间存在语义鸿沟——CAD 参数序列本身缺乏设计意图和几何形状的文本描述，LLM 无法理解那些纯数字化的操作序列的含义。

**本文目标** 1）如何让 LLM 理解和生成参数化 CAD 操作序列？2）如何构建连接自然语言与 CAD 数据的桥梁？3）如何支持从文本生成到编辑的多种 CAD 下游任务？

**切入角度**：作者洞察到 LLM 特别擅长代码生成，而代码之所以被 LLM 理解得好，是因为代码通常伴随丰富的注释和功能描述。因此，关键在于为 CAD 数据添加结构化的语义标注，将其转化为"带注释的代码"形式。通过 VLM（如 GPT-4o）对 CAD 模型的 3D 渲染图和 2D 草图进行层次化描述，生成详尽的文本注释。

**核心 idea**：将 CAD 参数序列转化为附带层次化语义描述的 Python 风格代码（SPCC），然后通过自适应预训练和指令微调让 LLM 掌握 CAD 生成和编辑能力。

## 方法详解

### 整体框架

CAD-Llama 框架分为两大部分：（1）SPCC 数据合成——通过层次化标注管线将 CAD 序列转化为 SPCC 表示；（2）模型训练——先在 SPCC 语料上进行自适应预训练，再在多任务指令数据上进行指令微调。最终得到的模型 CAD-Llama-INS 支持 text-to-CAD、补全、caption、添加、删除等多种任务。

### 关键设计

1. **层次化标注管线（Hierarchical Annotation Pipeline）**:

    - 功能：为 CAD 模型生成多层次、结构化的文本描述
    - 核心思路：分两阶段进行。**第一阶段——组件描述**：对 CAD 模型的每个组件分别渲染 3D 投影图和 2D 草图，输入 VLM（GPT-4o）生成该组件的详细几何描述（形状、尺寸、拉伸方向等）。**第二阶段——全局描述**：渲染各组件的轮廓图（目标组件高亮、其他组件半透明），连同第一阶段的组件描述一起输入 GPT-4o，生成两部分全局描述：（a）抽象概述 $\mathcal{A}$（一句话说明模型是什么），（b）详细描述 $\mathcal{T}$（组件间的空间关系和装配过程），以及每个组件的简短命名 $\mathcal{S}$（链接全局和局部描述）。为提升输出稳定性，按 CAD 复杂度分 5 级，每级提供 50 个高质量示例，使用 two-shot 提示
    - 设计动机：单次 VLM 调用无法同时捕获细粒度几何属性和组件间的组合关系。分层标注使每层聚焦不同粒度的信息，确保描述的全面性和准确性

2. **结构化参数 CAD 代码（SPCC）**:

    - 功能：创建 LLM 友好的 CAD 数据表示
    - 核心思路：受 LLM 代码生成能力的启发，将 CAD 参数序列转为 Python 风格代码。每个草图表示为循环列表（如 `sketch_i.append(loop1)`），循环中调用 `Line`、`Arc`、`Circle` 等方法绘制。拉伸操作引用对应草图完成。坐标使用 8 位量化参数（起点重定到 (0,0)），角度使用 0-360 的离散值。然后将层次化描述嵌入代码中：组件描述作为代码注释附在对应组件代码前，全局描述作为整体前缀。最终的 SPCC 语料包含详细描述版（$\tilde{\mathcal{D}}$）和仅含抽象描述版（$\dot{\mathcal{D}}$），使 LLM 能处理不同详细程度的输入
    - 设计动机：代码格式利用了 LLM 在代码数据上的已有能力；层次化描述弥补了 CAD 序列缺乏语义标注的核心缺陷

3. **自适应预训练和指令微调**:

    - 功能：将通用 LLM 转化为 CAD 领域专家
    - 核心思路：**预训练阶段**：在 SPCC 语料上用标准自回归语言建模目标进行全参数微调。创新点在于将相似 CAD 模型分组到同一上下文中——使用 CLIP 编码 CAD 渲染图、按余弦相似度构建文档图、通过遍历图构建训练上下文。这使 LLM 能在上下文中对比相似模型的差异，更高效地学习。上下文窗口 2048，得到 CAD-Llama。**指令微调阶段**：构建包含 6 种 CAD 任务的指令数据集（text-to-CAD、caption、补全、添加、删除及其 SPCC 增强版），每个任务 12K 条数据，共 48K 条。使用 LoRA（rank=256, α=128）进行参数高效微调，上下文窗口 4096，得到 CAD-Llama-INS
    - 设计动机：相似模型分组受启发于课程学习，让模型先看到相似的样本来学习细粒度差异。指令微调的多任务设计使模型能统一处理 CAD 相关的多种下游任务

### 损失函数 / 训练策略

预训练阶段使用标准的 next-token prediction 损失：$\mathcal{L}(\mathcal{X}) = \sum_{i=1}^{n} \log P(x_i | x_{i-1}, ..., x_0; \Phi)$，对所有 token 计算。指令微调阶段同样使用 next-token prediction 但只对输出部分计算损失：$\mathcal{L}(D) = \sum_{i=1}^{N} \log P(Y_i | X_i; \Theta)$。基座模型为 LLaMA3-8B-HF，优化器为 AdamW（lr=2e-5），使用 DeepSpeed 和 FlashAttention 加速训练。

## 实验关键数据

### 无条件生成

| 方法 | COV ↑ | MMD ↓ | JSD ↓ | SR ↑ | Novel ↑ |
|------|-------|-------|-------|------|---------|
| DeepCAD | 66.68 | 1.19 | 2.59 | 61.84 | 91.7 |
| SkexGen | 77.42 | 1.07 | 0.93 | 72.26 | 99.1 |
| HNC-CAD | 80.46 | 0.98 | 0.74 | 79.11 | 93.9 |
| **CAD-Llama** | **79.93** | **0.96** | **0.66** | **99.90** | **97.1** |

### Text-to-CAD

| 方法 | ACC_T ↑ | MCD ↓ | MMD ↓ | JSD ↓ |
|------|---------|-------|-------|-------|
| GPT-4 | 20.03 | 25.62 | 3.33 | 18.09 |
| LLaMA3 | 17.26 | 17.33 | 4.10 | 12.36 |
| Text2CAD | 69.91 | 20.64 | 3.02 | 9.98 |
| CAD-Translator | 70.36 | 21.29 | 2.94 | 10.92 |
| **CAD-Llama-INS** | **84.72** | **10.53** | **1.54** | **3.59** |

### 消融实验（CAD 表示方式）

| 表示方式 | ACC_cmd ↑ | ACC_param ↑ | SR ↑ |
|----------|-----------|-------------|------|
| SDCS (单描述+序列) | 39.17 | 25.56 | 18.14 |
| SDCC (单描述+代码) | 42.62 | 27.13 | 21.46 |
| SPCS (层次描述+序列) | 73.13 | 47.32 | 98.71 |
| **SPCC (层次描述+代码)** | **80.41** | **59.09** | **99.30** |

### 关键发现

- CAD-Llama-INS 在 text-to-CAD 上精度（ACC_T=84.72）超出先前最佳 CAD-Translator 约 14 个百分点
- 层次化描述是性能提升的最大贡献者——SPCC vs SDCC 的 ACC_cmd 差距达 38 个百分点
- 代码格式也有贡献，但次于层次描述——SPCC vs SPCS 的 ACC_cmd 差距约 7 个百分点
- 无条件生成的 SR 达到 99.90%，远高于所有基线，说明生成稳定性极高
- 在 CAD 编辑任务中，使用 SPCC 格式（deletion* / addition*）显著优于纯 CAD 代码格式，GPT-4 在 deletion* 上的 EM 也从 66.20 提升到 90.41，证明 SPCC 的结构化信息对所有 LLM 都有帮助

## 亮点与洞察

- **将 CAD 类比为"带注释的代码"是核心洞察**：利用 LLM 已有的代码生成能力，通过添加语义标注来桥接 CAD 与自然语言的鸿沟
- **层次化标注管线设计精巧**：从组件到整体的两阶段标注、复杂度分级的 few-shot 策略，确保了 VLM 输出的质量和一致性
- **99.90% 的生成成功率令人印象深刻**：说明 SPCC 格式确实让 LLM "理解"了 CAD 模型的结构约束
- **多任务指令微调的实用性**：一个模型同时支持生成、补全、编辑、描述等多种操作，符合实际 CAD 工作流程

## 局限与展望

- 基于 DeepCAD 数据集（178K 模型），规模和复杂度有限——主要是机械零件，缺乏建筑、有机形状等更复杂的 CAD 类型
- 标注管线依赖 GPT-4o，成本较高且存在幻觉风险
- 8 位量化的坐标精度（256 级）可能不满足工业级 CAD 的精度要求
- 仅使用 2D 渲染图进行标注，3D 信息的保留可能不够完整
- 与多模态输入（点云、图像 → CAD）的方法未做对比
- 未做人类评估，模型生成的 CAD 在实际设计可用性方面缺乏定性验证

## 相关工作与启发

- **DeepCAD (Wu et al., 2021)**：参数化 CAD 生成的基础工作，使用 Transformer 自回归生成 CAD 序列
- **Text2CAD / CAD-Translator**：基于 encoder-decoder 的文本到 CAD 方法，但泛化能力受限
- **OpenECAD**：使用 VLM 配合 PythonOCC CAD 内核进行 CAD 生成
- **CAD-GPT / CAD-MLLM**：使用多模态 LLM 生成 CAD 序列，支持图像/点云等多种输入
- 本文的 SPCC 思路可推广到其他专业领域——为专业数据添加层次化语义描述，将其转化为 LLM 友好的格式，是释放 LLM 领域能力的通用策略

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Instruction Tuning of Large Language Models for Tabular Data Generation—in One Day](../../ICML2025/llm_alignment/instruction_tuning_of_large_language_models_for_tabular_data_generation-in_one_d.md)
- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)
- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](../../ICML2025/llm_alignment/on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)

</div>

<!-- RELATED:END -->
