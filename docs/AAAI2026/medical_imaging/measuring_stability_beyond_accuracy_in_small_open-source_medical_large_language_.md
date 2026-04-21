---
title: >-
  [论文解读] Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology
description: >-
  [AAAI 2026][医学图像][医学LLM] 系统评估了6个小型开源医学LLM（<10B参数）在儿科内分泌领域的表现，揭示仅靠准确率不足以衡量模型可靠性：语义无关的提示微调导致模型输出显著变化（Stuart-Maxwell p<10⁻⁴），高一致性不等于正确，甚至CUDA版本差异也能引发统计显著的输出偏移。
tags:
  - AAAI 2026
  - 医学图像
  - 医学LLM
  - 小模型评估
  - 稳定性
  - 可复现性
  - 提示敏感性
  - 儿科内分泌
  - 自我评估偏差
---

# Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology

**会议**: AAAI 2026  
**arXiv**: [2601.11567](https://arxiv.org/abs/2601.11567)  
**代码**: [GitHub](https://github.com/vanessadamario/SmallMedLLMs-PedEndo)  
**领域**: 医学AI / LLM评估  
**关键词**: 医学LLM, 小模型评估, 稳定性, 可复现性, 提示敏感性, 儿科内分泌, 自我评估偏差

## 一句话总结
系统评估了6个小型开源医学LLM（<10B参数）在儿科内分泌领域的表现，揭示仅靠准确率不足以衡量模型可靠性：语义无关的提示微调导致模型输出显著变化（Stuart-Maxwell p<10⁻⁴），高一致性不等于正确，甚至CUDA版本差异也能引发统计显著的输出偏移。

## 研究背景与动机

**领域现状**：大型医学LLM（如GPT-4级别、70B+参数）在多种临床任务上展现了强大能力，但这些模型往往是闭源的、计算成本高，在低资源或低收入医疗场景中不可用。小型开源医学LLM（<10B参数）提供了一种轻量、可本地部署、透明的替代方案。

**现有痛点**：
   - 当前对医学LLM的评估主要依赖MCQ准确率，忽视了**一致性**、**鲁棒性**和**推理质量**
   - LLM对提示变化的敏感性在非医学领域已有报道，但在医学LLM中缺乏系统研究
   - 小模型是否编码了足够的临床知识以支持实际使用，尚无定论
   - 可复现性问题——硬件/软件栈差异可能导致临床决策不同的输出

**为什么选儿科内分泌？** 该领域正面临患者量增长、等待时间延长、专科医生短缺等困境，是AI辅助决策的潜在应用场景，但目前缺少该领域的LLM评估。

**核心研究问题**：小型开源医学LLM在准确率之外的稳定性和可复现性如何？仅凭准确率能否反映模型的实际可靠性？

## 方法详解

### 整体框架
多维度评估框架，包含三个核心实验 + 数值稳定性分析：

### 评估数据
- **Pediatric ESAP 2021-2022**：儿科内分泌自评程序，100个临床案例/知识问题（MCQ格式，5选1）
- 排除依赖图像/图表的9题，最终91题
- 每题含临床描述、问题、5个选项、正确答案、金标准解释

### 被评估模型（6个小型开源医学LLM）

| 模型 | 基座 | 参数量 |
|------|------|--------|
| HuatuoGPT-o1-8B | LLaMA 3.1 8B | ~8B |
| Diabetica-7B | Qwen2-7B | ~7B |
| Diabetica-o1 | Diabetica-7B自蒸馏 | ~7B |
| Meditron3-8B | LLaMA 3.1-8B | ~8B |
| MedFound-7B | BLOOM-7B | ~7B |
| ClinicalGPT-base-zh | BLOOM-7B | ~7B |

### 实验设计

**实验1：准确率与提示稳定性（确定性设置，T=0）**
- 三种提示策略测试：原始提示A、语法变体提示B、去除选项字母的提示A
- 评估维度：准确率、McNemar检验（准确率变化）、Stuart-Maxwell检验（输出分布变化）、Cohen's κ（一致性）、匹配率

**实验2：非确定性设置下的稳定性**
- 对实验1中表现最好的4个模型，在 T=0.3/0.6/1.0 下各运行10次
- 评估一致性（多数投票频率）与正确性的关系

**实验3：自我评估偏差与金标准推理判别**
- 给模型展示两个候选解释（模型自身的 vs 专家金标准），让其选择更好的
- 测试位置偏差：交换两个解释的顺序
- 儿科内分泌专家对 HuatuoGPT-o1 答错的案例进行人工评审

**数值稳定性分析**
- 在两台不同GPU/CUDA版本的机器上运行相同推理（CUDA 11.8 vs CUDA 12.8）
- 检验硬件-软件栈差异对输出的影响

## 实验

### 实验1：准确率与提示稳定性

| 模型 | 提示A准确率 | 提示B准确率 | 去字母准确率 | 可用率 |
|------|------------|------------|-------------|--------|
| HuatuoGPT-o1-8B | **0.35** | **0.35** | **0.33** | 100% |
| Diabetica-o1 | 0.33 | 0.34 | 0.27 | 100% |
| Meditron3-8B | 0.33 | 无输出 | 0.34 | 97.8% |
| Diabetica-7B | 0.30 | 0.32 | 0.27 | 98.9% |
| ClinicalGPT-base-zh | 0.20 | 0.20 | 0.20 | 79.1% |
| MedFound-7B | 0.04 | 0.12 | 0.04 | 18.6% |

**关键发现**：
- McNemar检验显示所有模型准确率变化不显著（p>0.4）
- 但 Stuart-Maxwell 检验显示**输出分布显著变化**（p<10⁻⁴），即准确率稳定≠行为稳定
- Cohen's κ 在去除选项字母时 ≤0.4，说明模型严重依赖字母标记而非语义理解
- Meditron3-8B 在提示B下完全无法产生输出

### 实验2：非确定性设置下的一致性 vs 正确性

| 模型 | T=0.3一致且正确 | T=0.3一致但错误 | T=1.0一致且正确 | T=1.0一致但错误 |
|------|----------------|----------------|----------------|----------------|
| HuatuoGPT-o1 | 14 | 11 | 10 | 4 |
| Diabetica-7B | 13 | 16 | 3 | 7 |
| Diabetica-o1 | 5 | 4 | 5 | 3 |
| Meditron3-8B | 2 | 0 | 0 | 0 |

**关键发现**：
- **高一致性不等于正确性**：Diabetica-7B 在 T=0.3 时一致但错误的案例（16）比一致且正确的（13）还多
- HuatuoGPT-o1 是唯一在高一致性时正确率超过错误率的模型
- 温度升高时，Meditron3-8B 的多数投票结果变为"幻觉/无输出"（71/91）

### 实验3：自我评估偏差

- HuatuoGPT-o1 在59个错误案例中：27个选了金标准（两个位置都选），19个仅在一个位置选，8个从未选
- Diabetica-o1 表现更差：19个两位置都选，19个从未选
- 存在**位置偏差**：解释的呈现顺序影响选择结果
- 专家评审12个HuatuoGPT-o1错误案例：Likert评分2-4分，仅2例有明确事实错误，多数是推理不完整

### 数值稳定性

| 模型 | Δ准确率 | Stuart-Maxwell p值 | Cohen's κ | 匹配率 |
|------|---------|-------------------|-----------|--------|
| HuatuoGPT-o1 | +0.044 | <10⁻⁴ | 0.51 | 0.60 |
| Diabetica-o1 | -0.022 | <10⁻⁴ | 0.53 | 0.63 |
| Diabetica-7B | +0.011 | <10⁻⁴ | 0.68 | 0.75 |
| Meditron3-8B | +0.011 | <10⁻⁴ | 0.31 | 0.45 |

- **CUDA版本不同即导致统计显著的输出分布偏移**，尽管准确率变化在置信区间内
- Meditron3-8B 跨系统一致性最差（κ=0.31），意味着45%的案例给出不同答案

## 亮点与洞察

1. **"准确率稳定≠行为稳定"的深刻发现**：McNemar检验通过但Stuart-Maxwell检验失败，揭示传统评估指标的盲区
2. **"一致性≠正确性"**：模型可能对错误答案高度一致，在临床场景中可能导致系统性地推荐不当治疗
3. **字母依赖性强于语义理解**：去除选项字母后一致性大幅下降（κ≤0.4），暴露模型依赖表面线索
4. **硬件-软件栈影响临床决策**：仅CUDA版本差异即可导致25-55%的案例给出不同临床建议，对可复现性构成严重挑战
5. **评估方法论贡献**：提出了多维度诊断框架（提示敏感性+随机稳定性+推理一致性+数值稳定性），可推广到其他场景
6. **人工评审与自动评估互补**：放弃GPT-4自动评审（因其自身不一致），采用非专家+专家混合评审

## 局限性

1. **绝对性能较低**：最好的模型准确率仅35%（随机猜测18.2%），缺乏人类基线对比
2. **仅测试了MCQ格式**：未涉及开放式问答或场景式评估，MCQ可能掩盖推理不一致
3. **超参数探索有限**：仅变动了提示和温度，其他推理超参数（top-p、repetition penalty等）保持固定
4. **数值稳定性分析范围有限**：仅对比了两台机器，未系统测试更多硬件配置
5. **非专家评审可能遗漏细微医学错误**：虽用专家补充，但全面专家评审资源不足
6. **数据集较小**：仅91个案例，统计检验力有限

## 相关工作

- **医学LLM评估**：MedQA、MedMCQA、PubMedQA 基准；HealthBench (OpenAI)
- **LLM稳定性研究**：提示敏感性（li2024can, khatun2024study）、自我评估偏差（xu2024pride）
- **小型开源医学LLM**：HuatuoGPT-o1, Meditron3, MedFound, ClinicalGPT
- **可复现性问题**：CUDA非确定性（he2025nondeterminism）、推理管道差异

## 评分 ⭐⭐⭐⭐

首个系统性地从稳定性和可复现性角度评估小型医学LLM的工作，方法论贡献显著，尤其是"准确率稳定≠行为稳定"和"CUDA版本影响临床输出"两个发现具有重要的警示意义。但数据集小、绝对性能低限制了结论的普适性。

<!-- RELATED:START -->

## 相关论文

- [Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models](coarse-to-fine_open-set_graph_node_classification_with_large_language_models.md)
- [Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)

<!-- RELATED:END -->
