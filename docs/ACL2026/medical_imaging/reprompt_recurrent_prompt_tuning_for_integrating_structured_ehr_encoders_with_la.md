---
title: >-
  [论文解读] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models
description: >-
  [ACL 2026][医学图像][电子健康记录] 本文提出 RePrompT，一种时间感知的 LLM 框架，通过循环提示调优（将前一次就诊的隐状态作为下一次就诊的软提示）和结构化编码提示调优（注入群体级 EHR 编码器的嵌入）两种互补机制，在 MIMIC-III/IV 上的再入院和死亡率预测任务上一致超越 EHR 基线和 LLM 基线。
tags:
  - ACL 2026
  - 医学图像
  - 电子健康记录
  - 提示调优
  - 循环状态传播
  - 结构化编码器
  - 临床预测
---

# RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.17725](https://arxiv.org/abs/2604.17725)  
**代码**: https://github.com/KU-AI4H/RePrompT  
**领域**: 医学图像  
**关键词**: 电子健康记录, 提示调优, 循环状态传播, 结构化编码器, 临床预测

## 一句话总结
本文提出 RePrompT，一种时间感知的 LLM 框架，通过循环提示调优（将前一次就诊的隐状态作为下一次就诊的软提示）和结构化编码提示调优（注入群体级 EHR 编码器的嵌入）两种互补机制，在 MIMIC-III/IV 上的再入院和死亡率预测任务上一致超越 EHR 基线和 LLM 基线。

## 研究背景与动机

**领域现状**：电子健康记录（EHR）包含患者多次就诊的诊断、用药、手术等纵向信息，是临床决策支持的重要数据源。大语言模型在 EHR 挖掘任务（如死亡率预测、再入院预测）上展现了潜力，但在处理结构化 EHR 信号时仍面临两个核心挑战。

**现有痛点**：第一个挑战是时间结构丢失。将纵向 EHR 数据线性化为纯文本（"Visit 1: ... Visit 2: ..."）时，时间依赖和临床代码的离散身份会被模糊化。虽然可以用分隔符标记就诊边界，但 LLM 本质上将输入视为单一文档，缺乏对就诊间时间依赖的显式建模机制。第二个挑战是缺乏群体级信息。传统的 EHR 预测模型（如 RETAIN、GRAM）在患者群体上联合训练，学习一个共享的、任务对齐的表示空间，能发现疾病共现、纵向进展等群体级模式。而 LLM 在每个患者上独立推理（case-isolated），缺乏利用其他相似患者信息来辅助预测的机制。

**核心矛盾**：LLM 擅长从文本中提取丰富的上下文信息，但缺乏对纵向时间结构的建模能力和群体级知识的利用能力；结构化 EHR 编码器擅长时间建模和群体级模式发现，但表示能力有限。如何结合两者的互补优势是核心问题。

**本文目标**：设计一种轻量化的框架，在不修改 LLM 架构的前提下，将结构化 EHR 编码器的时间和群体级信息注入 LLM。

**切入角度**：提示调优（prompt tuning）允许在不改变 LLM 参数的情况下注入外部信息——只需将外部信息编码为可训练的软提示向量即可。

**核心 idea**：用两种互补的软提示机制增强 LLM：（1）循环提示将前一次就诊的 LLM 隐状态传递到当前就诊，显式建模纵向依赖；（2）结构化编码提示将群体训练的 EHR 编码器（RETAIN）的嵌入注入 LLM，引入群体级模式。

## 方法详解

### 整体框架
RePrompT 包含三个模块：（1）临床记录合成：用 DeepSeek-V3 将出院记录和结构化医疗代码合成为简洁的患者摘要；（2）循环状态提示调优：LLM 逐次就诊处理，将前一次就诊的隐状态通过线性变换作为当前就诊的软提示；（3）结构化编码提示调优：用 RETAIN 编码器将结构化 EHR 序列编码为稠密表示，作为另一组软提示。最终 LLM 的输入是三部分的拼接：循环提示 $G_{i,t}$、结构化提示 $S_{i,t}$ 和文本嵌入。

### 关键设计

1. **循环状态提示调优（State-Recurrent Prompt Tuning）**:

    - 功能：显式建模纵向就诊间的时间依赖
    - 核心思路：不将所有就诊拼接为单一文档，而是让 LLM 每次只处理一次就诊。处理完后提取最后一层的 token 级隐状态 $H_{i,t}$，通过平均池化得到就诊级隐状态 $\hat{H}_{i,t}$，再经线性变换 $G_{i,t+1} = w_t \hat{H}_{i,t} + b_t$ 生成下一次就诊的软提示。这样 LLM 在处理每次就诊时都能"记住"之前的所有就诊信息
    - 设计动机：标准 LLM 将所有就诊文本拼接后作为单一文档处理，无法区分不同就诊的时间边界。循环机制使 LLM 显式地将信息从一次就诊传递到下一次，类似 RNN 的隐状态传递

2. **结构化编码提示调优（Struct-Encoded Prompt Tuning）**:

    - 功能：将群体训练的 EHR 编码器的知识注入 LLM
    - 核心思路：使用 RETAIN 模型（具有双层注意力和循环建模能力）在患者群体上训练，编码就诊级别的医疗代码序列 $\{V_{i,j}\}_{j=1}^t$ 为稠密表示 $S_{i,t} \in \mathbb{R}^{P \times D}$，作为软提示注入 LLM
    - 设计动机：LLM 在每个患者上独立推理，无法利用其他相似患者的信息。RETAIN 在群体上训练后学到了疾病共现、用药模式等共享模式，将其嵌入注入 LLM 相当于为每个患者提供了"群体参考"

3. **临床记录合成**:

    - 功能：将冗长嘈杂的出院记录和结构化代码合成为简洁的患者摘要
    - 核心思路：用 DeepSeek-V3 对每次就诊的出院记录和相应的医疗代码进行摘要，去除模板化段落和冗余信息，产生统一的患者表示 $\hat{C}_{i,t}$
    - 设计动机：原始出院记录冗长且噪声大，直接输入 LLM 会浪费上下文窗口且引入噪声；通过 LLM 预处理可以提炼关键临床信息

### 损失函数 / 训练策略
使用 Binary Cross-Entropy 损失进行二分类/多标签分类。可训练参数包括：RETAIN 编码器、循环状态的线性变换层、输出分类头。LLaMA 模型在训练和推理过程中保持冻结。使用 LLM2Vec 框架的 Llama 3.1 1B 作为嵌入提取器。

## 实验关键数据

### 主实验（与 EHR 基线对比）

| 模型 | MIMIC-IV 再入院 AUROC | MIMIC-IV 死亡率 AUROC | MIMIC-III 再入院 AUROC | MIMIC-III 死亡率 AUROC |
|------|---------------------|---------------------|---------------------|---------------------|
| RETAIN | 0.670 | 0.601 | 0.660 | 0.608 |
| StageNet | 0.656 | 0.664 | 0.676 | 0.633 |
| ARCI | 0.663 | 0.611 | 0.652 | 0.618 |
| **RePrompT** | **0.706** | **0.673** | **0.688** | **0.646** |

### 消融实验

| 配置 | MIMIC-IV 再入院 AUROC | MIMIC-IV 死亡率 AUROC | 说明 |
|------|---------------------|---------------------|------|
| RePrompT 完整 | 0.706 | 0.673 | 完整模型 |
| 去掉两个模块 | 0.673 | 0.635 | 基线水平 |
| 去掉循环提示 | 0.693 | 0.642 | 循环提示贡献更大 |
| 去掉结构化编码 | 0.698 | 0.665 | 结构化编码也有贡献 |
| 去掉 DeepSeek 摘要 | 0.685 | 0.640 | 摘要预处理也有帮助 |

### 关键发现
- RePrompT 在所有数据集和任务上一致优于所有 EHR 基线和 LLM 基线，AUROC 提升约 3-7 个百分点
- 循环状态提示的贡献大于结构化编码提示——去掉循环提示后 AUROC 降低更多，说明纵向时间依赖建模是最关键的
- 与 LLM 基线的比较尤为惊人：Zero-shot LLM（GPT-5）AUROC 仅 0.512（再入院），RePrompT 达到 0.706，差距巨大
- 在不同 EHR 编码器的对比中，RETAIN 表现最好，Transformer 编码器反而最差——因为 Transformer 在建模时序就诊依赖时不如 RNN 有效
- DeepSeek 摘要预处理虽然有帮助，但去掉后 RePrompT 仍优于 RETAIN，证明性能提升主要来自框架设计而非摘要质量

## 亮点与洞察
- **循环提示调优**的设计非常优雅——它将 RNN 的序贯处理思想引入 LLM 的提示空间，使 LLM 能像 RNN 一样逐步处理时序数据。这个思路可以迁移到任何需要处理时序文档的 LLM 应用
- **将 EHR 编码器作为软提示注入 LLM** 提供了一种通用的"专家知识注入"范式——任何领域特定的编码器都可以通过这种方式与 LLM 融合，无需修改 LLM 本身
- Transformer 在 EHR 时序建模上不如 RNN 这一发现很有启发性——位置编码不足以替代 RNN 的显式时序传递

## 局限与展望
- 目前仅验证了 Llama 3.1 1B 作为基础模型，更大规模的 LLM 可能有不同表现
- 使用 DeepSeek-V3 做摘要预处理引入了额外的依赖和成本
- 软提示数 $P=10$ 是超参，论文未充分讨论不同值的影响
- 可以考虑将循环提示扩展为多尺度——不仅传递最近一次就诊的信息，还传递更长时间跨度的聚合信息
- 未来可以探索将方法扩展到多标签药物推荐等更复杂的临床预测任务

## 相关工作与启发
- **vs RETAIN**: RETAIN 仅用结构化代码序列，缺乏文本理解能力；RePrompT 通过 LLM 引入了对出院记录的深层语义理解，同时保留了 RETAIN 的群体级模式
- **vs COCONUT**: COCONUT 也使用软 token 进行推理，但缺乏显式的时间结构建模，在 EHR 时序任务上表现不如 RePrompT
- **vs Zero-shot GPT-5**: 即使是最强的通用 LLM 在 EHR 预测上也远逊于领域特化的方法，说明结构化医学知识的注入不可或缺

## 评分
- 新颖性: ⭐⭐⭐⭐ 循环提示调优和结构化编码提示的组合设计有新意，但各单独组件（提示调优、RNN 状态传递）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、两个任务、8 个 EHR 基线 + 3 个 LLM 基线、多层消融和编码器对比，非常充分
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，动机阐述充分，消融分析详尽
- 价值: ⭐⭐⭐⭐ 提供了一种轻量化且有效的 LLM-EHR 融合范式，对临床 AI 有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)

<!-- RELATED:END -->
