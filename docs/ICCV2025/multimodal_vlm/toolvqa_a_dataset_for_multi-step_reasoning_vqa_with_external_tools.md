---
title: >-
  [论文解读] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools
description: >-
  [ICCV 2025][多模态][视觉问答] 提出 ToolVQA——一个包含 23K 样本的大规模多模态工具增强 VQA 数据集，通过 ToolEngine 管道（结合图像引导的 DFS 和 LCS 示例匹配）自动生成真实场景下的多步推理数据，在其上微调的 LLaVA-7B 在 5 个 OOD 基准上超越 GPT-3.5-Turbo。
tags:
  - ICCV 2025
  - 多模态
  - 视觉问答
  - 工具使用
  - 多步推理
  - 数据集
  - 大模型
  - 工具代理
---

# ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools

**会议**: ICCV 2025  
**arXiv**: [2508.03284](https://arxiv.org/abs/2508.03284)  
**代码**: [GitHub](https://github.com/Fugtemypt123/ToolVQA-release)  
**领域**: multimodal_vlm  
**关键词**: 视觉问答, 工具使用, 多步推理, 数据集, 大模型, 工具代理

## 一句话总结

提出 ToolVQA——一个包含 23K 样本的大规模多模态工具增强 VQA 数据集，通过 ToolEngine 管道（结合图像引导的 DFS 和 LCS 示例匹配）自动生成真实场景下的多步推理数据，在其上微调的 LLaVA-7B 在 5 个 OOD 基准上超越 GPT-3.5-Turbo。

## 研究背景与动机

将外部工具集成到大型基础模型（LFM）中是构建通用 AI 助理的关键方向。但现有工具增强 VQA 数据集存在三大差距：

**场景不真实**：使用合成图像或过度简化的 PDF 文件，与真实世界场景复杂度不匹配

**查询过于简单**：仅需单步推理或显式提供工具使用提示（如"使用 Cheap YouTube API 工具"），缺乏隐式多步推理

**标注成本高**：依赖人工标注的数据集（如 GAIA 仅 500 样本）难以扩展

**本文核心目标**：构建同时满足真实场景（Real-world Scenarios）和真实查询（Real-world Queries）的大规模可扩展数据集，弥合合成数据与真实工具使用之间的差距。

与现有数据集的对比优势（Table 1）：ToolVQA 是唯一同时满足多模态输入、真实场景/查询、可评估答案、高推理复杂度（2.38）的大规模数据集。

## 方法详解

### ToolEngine 数据构建管道

包含三个核心组件（Fig. 3）：

**1. 真实世界示例构建**

邀请 10 位来自不同学科（数学、计算机、经济学、中文、艺术）的用户，每人记录 15 个常见工具使用场景。合并功能相似的工具后选出 10 个最常用工具，并将 150 个初始场景精炼为 34 个代表性示例。

**2. 图像引导的 DFS 搜索**

在工具图上执行深度优先搜索，构建多步工具使用轨迹：

$$t_i = \mathcal{M}(choices=\mathcal{T}, image=\mathcal{I}, examples=\text{Ret}(\mathcal{E}, \mathcal{P}_{i-1}))$$
$$a_i = \mathcal{M}(tool=t_i, image=\mathcal{I}, examples=\text{Ret}(\mathcal{E}, \mathcal{P}_i))$$

其中控制器 $\mathcal{M}$ 为 ChatGPT-4o-latest，每步选择工具并生成参数，涉及真实工具调用以提取图像信息。

**3. LCS 示例匹配**

基于最长公共子序列（LCS）算法进行动态示例匹配。在第 $i$ 步，计算当前轨迹 $\mathcal{P}_i$ 与示例集 $\mathcal{P}^e$ 中每个元素的 LCS 匹配度，检索 Top-k 最匹配示例：

$$\text{Ret}(\mathcal{E}, \mathcal{P}_i) = \text{TopK}_{\mathcal{P}^e \in \mathcal{E}}\{\text{LCS}(\mathcal{P}^e, \mathcal{P}_i)\}$$

关键优势：不同于固定示例匹配，LCS 允许在 DFS 过程中动态切换示例，融合不同类型知识，显著提升推理复杂度和数据质量。

### 工具集设计

10 个工具覆盖 4 大类：
- **感知**：ImageCaption, OCR, ObjectDetection, RegionDescription
- **操作**：DrawBox, GoogleSearch
- **逻辑**：Calculator, Plot, ItemCount
- **创造**：TextToImage

### 训练目标

使用交叉熵损失微调 LLaVA-7B：

$$\mathcal{L} = \mathbb{E}_{\mathcal{E} \sim \mathcal{D}}\left[\frac{1}{n}\sum_{i=1}^{n} -\log p(t_i, a_i, r_i \mid \mathcal{I}, \mathcal{T}, \mathcal{Q}, \mathcal{P}_{i-1})\right]$$

训练配置：batch_size=2, lr=2e-4, LoRA 微调, 4000 epochs, 4×GTX3090。

## 实验关键数据

### 主实验：ToolVQA 测试集（Table 4）

| 模型 | 设置 | End-to-End Acc.↑ | Inst.↑ | Tool.↑ | Arg.↑ | Summ.↑ |
|------|------|-----------------|--------|--------|-------|--------|
| ChatGPT-4o-latest | VLM | 38.29 | - | - | - | - |
| ChatGPT-4o-latest | VLM+tool | 34.96 | 36.5 | 14.68 | 8.92 | 56.1 |
| GPT-3.5-Turbo | LLM+tool | 18.37 | 73.24 | 30.46 | 20.08 | 58.18 |
| LLaVA-7B（原始） | VLM+tool | 1.17 | 16.39 | 9.43 | 0 | 0.01 |
| **Tuned LLaVA-7B** | **VLM+tool** | **18.80** | **86.62** | **61.61** | **39.34** | 30.91 |

关键发现：
- 微调后 7B 模型的端到端准确率接近大规模闭源 GPT-3.5-Turbo
- 指令格式化和工具选择显著提升（Inst. 86.62%, Tool. 61.61%），但参数预测和答案总结仍是瓶颈
- GPT-4o 的 VLM+tool 反而不如纯 VLM（34.96 < 38.29），工具引入的噪声超过了收益

### OOD 泛化实验（Table 5）

| 模型 | TextVQA | TallyQA | InfoSeek | GTA | TEMPLAMA |
|------|---------|---------|----------|-----|----------|
| GPT-3.5-Turbo | 36.3 | 61 | 11.3 | 23.62 | 33.67 |
| LLaVA-7B | 41.2 | 60.1 | 5.2 | 12.12 | 3.06 |
| **Tuned LLaVA-7B** | **47** | **64.3** | **13.8** | **33.29** | 21.43 |

微调模型在 5 个 OOD 基准中的 4 个上超越 GPT-3.5-Turbo，展示了强泛化能力。

### ToolEngine 消融实验（Table 3）

| 方法 | Acc.↑ | Cor.↑ | Nec.↑ | R.C.↑ |
|------|-------|-------|-------|-------|
| **ToolEngine（完整）** | **90.8** | **85.2** | **87.51** | **2.38** |
| w/o Example + LCS | 27.3 | 77.6 | 21.04 | 1.1 |
| w/o LCS | 41.6 | 81.4 | 54.26 | 1.61 |

LCS 匹配对数据质量至关重要：移除后准确率从 90.8% 降至 41.6%，推理复杂度从 2.38 降至 1.61。

### Few-shot ICL 实验（Table 6）

| 模型 | 0-shot | 1-shot | 5-shot | 10-shot |
|------|--------|--------|--------|---------|
| GPT-4o | 34.96 | 37.20 | 38.41 | 38.63 |
| Tuned LLaVA-7B | 18.80 | 19.41 | 21.13 | 20.69 |

微调模型仍能从 ICL 中受益（18.80→21.13），说明微调和 ICL 是互补的。

### 数据集统计

- 总样本数：23,655
- 工具调用次数：65,785
- 平均推理轨迹长度：2.78 步
- 平均问题长度：15.74 tokens
- 平均答案长度：2.69 tokens（简洁可评估）

## 亮点与洞察

1. **LCS 动态匹配的精妙设计**：通过逐步匹配不同示例，让 DFS 过程能够融合多种知识，产生更复杂的推理链。固定示例匹配无法适配不同场景，这是数据质量提升的关键
2. **生成容易、回答难**：用于生成问题的 GPT-4o 在回答自己生成的问题时准确率不到 40%，证明了 ToolEngine 管道成功解耦了单步推理与多步端到端推理
3. **工具使用的双刃剑效应**：多个模型的 VLM+tool 表现反而低于纯 VLM，说明工具引入的噪声可能超过收益。微调能有效抑制工具噪声
4. **性能瓶颈在参数预测和答案总结**：这些任务需要理解工具返回的新信息并从中提取有意义的响应，微调对此改进有限

## 局限性

1. 工具集规模较小（10 个），虽然每个工具泛化能力强，但无法覆盖所有真实场景
2. 数据生成依赖 GPT-4o，成本较高且可能引入偏差
3. 90.8% 的自动生成准确率意味着约 10% 的训练数据存在噪声
4. 仅在 7B 规模模型上进行微调实验，更大模型的表现未知

## 相关工作与启发

- 与 MM-Traj（Gao et al., 2025）相比，ToolVQA 使用真实场景图像且答案经过验证
- LCS 匹配思路可推广到其他需要动态示例选择的数据合成场景
- 工具使用能力的瓶颈（参数预测、答案总结）指向了多轮对话中动态信息处理的研究方向
- 将高泛化工具（如 GoogleSearch）与特定任务工具结合的设计理念值得借鉴

## 评分 ⭐⭐⭐⭐

创新性 ★★★★☆：ToolEngine 管道和 LCS 示例匹配设计新颖
实验 ★★★★☆：覆盖多种模型和 OOD 基准，分析全面
写作 ★★★★☆：数据质量评估透彻，错误分析有深度
实用性 ★★★★★：公开代码和数据集，可直接用于工具代理训练和评估

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Acknowledging Focus Ambiguity in Visual Questions](acknowledging_focus_ambiguity_in_visual_questions.md)
- [\[ICCV 2025\] MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning](mmat1m_a_large_reasoning_dataset_for_multimodal_agent_tuning.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](llava-cot_let_vision_language_models_reason_step-by-step.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] VQ-FocusAmbiguity: Acknowledging Focus Ambiguity in Visual Questions](vq_focusambiguity_acknowledging_focus_ambiguity_visual_questions.md)

<!-- RELATED:END -->
