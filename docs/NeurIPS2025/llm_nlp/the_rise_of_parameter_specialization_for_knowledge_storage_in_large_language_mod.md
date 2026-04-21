---
title: >-
  [论文解读] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models
description: >-
  [NeurIPS 2025][LLM/NLP][参数特化] 系统分析 20 个开源 LLM，发现更强的模型在 MLP 参数向量中展现出更高的知识特化程度（Parameter Specialization），即相似知识倾向于集中编码到少数参数向量中，并通过因果实验验证该特化程度与模型知识任务性能之间存在因果关系。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - 参数特化
  - 知识存储
  - MLP可解释性
  - 知识叠加
  - 大语言模型
---

# The Rise of Parameter Specialization for Knowledge Storage in Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.17260](https://arxiv.org/abs/2505.17260)  
**代码**: 未公开  
**领域**: llm_nlp  
**关键词**: 参数特化, 知识存储, MLP可解释性, 知识叠加, 大语言模型  

## 一句话总结

系统分析 20 个开源 LLM，发现更强的模型在 MLP 参数向量中展现出更高的知识特化程度（Parameter Specialization），即相似知识倾向于集中编码到少数参数向量中，并通过因果实验验证该特化程度与模型知识任务性能之间存在因果关系。

## 研究背景与动机

大语言模型凭借庞大的参数空间能编码海量知识，但从微观视角来看，**如何在有限参数中更好地存储和压缩知识**仍是未充分研究的问题。

已有工作将 Transformer 的 MLP 层解释为键值记忆（Geva et al., 2021），其中 up-projection 矩阵为键（key），down-projection 矩阵为值（value），每个值向量可作为知识存储的基本单元。然而，由于数据特征远多于模型参数，每个参数不可避免地与多种知识重叠（Knowledge Superposition）。

本文的核心问题是：**不同能力的 LLM 在知识存储模式上有何区别？参数特化程度与模型性能之间是否存在因果关系？**

## 方法详解

### 整体框架

研究分为两个阶段：
1. **观察性分析**：在 20 个 LLM 上系统衡量参数特化程度与模型性能的关联
2. **因果验证**：通过控制实验确认参数特化与性能提升的因果关系

### 关键设计

**MLP 输出的键值分解**：Transformer 第 l 层 MLP 的输出可表示为值向量的线性组合，其中每个系数分数控制对应值向量的贡献权重。

**知识向量遮蔽方法（Knowledge Vectors Masking）**：为识别与特定概念相关的参数向量，对每个概念运行相关和无关问题，计算各层激活系数的差异分数，选择 top-k 差异最大的值向量进行遮蔽（系数设为零）。

**参数特化分数（Parameter Specialization Score, PSS）**：

PSS = |General Score after surgery - Concept Specific Score after surgery| / General Score before surgery

PSS 越高表示参数向量对特定知识的特化程度越高；PSS 越低表示知识叠加（superposition）越严重。

### 数据集构建

构建 **SpecWiki** 数据集：
- 从 Wikipedia 选取 525 个实体概念（人物、事件、地点等）
- 按页面浏览量分为高频、中频、低频三档
- 为每个概念设计 10 道多选题 + 开放生成题
- 使用 GPT-4o 生成问题和选项

### 训练策略（因果验证实验）

设计四种微调策略在 LLaMA2-7B 和 Qwen2-7B 上验证因果关系：
- **FT-FV**（Full Vectors）：微调 MLP 所有参数向量
- **FT-PV**（Partial Vectors）：仅微调 top-k/8 高激活参数向量
- **FT-CV**（Complementary Vectors）：仅微调互补集的参数向量
- **FT-RV**（Random Vectors）：随机选择等量参数向量微调

## 实验关键数据

### 主实验

对 20 个开源 LLM 的分析结果：

- PSS 与 SpecWiki 性能的 **Pearson 相关系数 = 0.92，Spearman 相关系数 = 0.93**
- 更新近发布的模型和 MMLU 性能更强的模型倾向于有更高的 PSS

**模型规模影响**：

| 模型 | Accuracy_MCQ | PSS |
|------|-------------|-----|
| Qwen1.5-0.5B | 0.61 | 0.019 |
| Qwen1.5-1.8B | 0.61 | 0.044 |
| Qwen1.5-4B | 0.73 | 0.075 |
| Qwen1.5-7B | 0.75 | 0.121 |
| Qwen1.5-14B | 0.82 | 0.184 |
| Gemma2-2B | 0.72 | 0.057 |
| Gemma2-9B | 0.86 | 0.138 |

### 消融实验（因果验证）

**微调策略对比**（LLaMA2-7B 上 10 个高频概念）：

| 方法 | Accuracy_MCQ | Accuracy_OEG | PSS | Semantic Entropy↓ | LID↓ |
|------|-------------|-------------|-----|-------------------|------|
| Original | 0.60 | 0.51 | 0.67 | 0.67 | 11.23 |
| FT-FV | 0.63 | 0.54 | 0.65 | 0.62 | 11.12 |
| **FT-PV** | **0.67** | **0.59** | **0.72** | **0.50** | **7.89** |
| FT-CV | 0.62 | 0.51 | 0.63 | 0.62 | 11.12 |
| FT-RV | 0.58 | 0.49 | 0.65 | 0.65 | 11.07 |

FT-PV 在所有指标上全面领先：知识准确率最高、特化程度最高、幻觉最低。

**预训练过程中的特化演化**（OLMo-2-7B，10 个检查点）：
- 早期训练（10K-210K步）：PSS 和准确率接近零
- 中期（310K-510K步）：准确率开始提升但 PSS < 0.1
- 后期（610K-910K步）：参数特化显著出现，伴随准确率大幅提升

### 关键发现

1. **同族模型中更先进的模型有更高的 PSS 峰值**，LLaMA/Qwen/Mistral/Gemma 四个家族均一致
2. **概念频率影响特化程度**：高频概念对应更高的 PSS，低频概念不仅准确率低，特化程度也低
3. **意外发现**：遮蔽少量概念相关向量（5%-10%）后，模型在无关问题上的表现甚至有所提升

## 亮点与洞察

1. **首次量化和跨模型比较参数特化程度**，填补了 LLM 知识存储微观机制的重要空白
2. **因果实验设计精妙**：FT-PV vs FT-FV vs FT-CV vs FT-RV 的四组对照精确地分离了变量
3. **预训练动态观察**：特化是训练后期涌现的现象，类似于其他涌现能力
4. **实际启发**：针对高激活向量微调不仅效果最好，还能有效减少幻觉

## 局限与展望

1. **仅分析 MLP 层**：知识可能也存储在注意力模块中，但未涉及
2. **模型规模限制**：受 GPU 限制，最大仅测试 14B 参数
3. **分析粒度较粗**：以参数向量为单位，未深入到单个神经元层面
4. **SpecWiki 数据集局限**：基于 Wikipedia 百科知识，不涵盖程序性、推理性知识
5. **未探讨与 MoE 架构的联系**：Mixture of Experts 天然通过专家参数特化提升性能

## 相关工作与启发

- **Knowledge Neurons**（Dai et al., 2022）和 **Knowledge Editing**（Meng et al., 2022）提供了知识定位基础
- **Knowledge Superposition**（Elhage et al., 2022）揭示了参数复用的普遍现象
- 与 **Sparse Autoencoders** 解纠缠特征的思路相关但视角不同
- 对 **知识蒸馏** 和 **模型压缩** 有直接启发

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次系统性量化LLM参数特化程度并建立因果关系，视角新颖
- **技术深度**: ⭐⭐⭐⭐ — 20个模型的大规模分析 + 精心设计的因果验证实验
- **实验质量**: ⭐⭐⭐⭐ — 多模型族、多规模、预训练动态、频率分析、因果实验，全面覆盖
- **实用性**: ⭐⭐⭐ — 发现有启发意义（如FT-PV策略），但直接可落地的方法有限
- **总体评分**: ⭐⭐⭐⭐ (7.5/10)

<!-- RELATED:START -->

## 相关论文

- [Knowledge Boundary of Large Language Models: A Survey](../../ACL2025/llm_nlp/knowledge_boundary_survey.md)
- [GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models](geocad_local_geometry-controllable_cad_generation_with_large_language_models.md)
- [Taming Knowledge Conflicts in Language Models](../../ICML2025/llm_nlp/taming_knowledge_conflicts_in_language_models.md)
- [Solving Inequality Proofs with Large Language Models](solving_inequality_proofs_with_large_language_models.md)
- [Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)

<!-- RELATED:END -->
