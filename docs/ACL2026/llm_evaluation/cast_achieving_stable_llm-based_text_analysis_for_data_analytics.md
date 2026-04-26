---
title: >-
  [论文解读] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics
description: >-
  [ACL 2026][输出稳定性] 提出CAST框架，通过算法提示（Algorithmic Prompting）和先思考后输出（Thinking-before-Speaking）两种机制约束LLM的潜在推理路径，显著提升文本摘要和标注任务的运行间稳定性，同时不损失输出质量。
tags:
  - ACL 2026
  - 输出稳定性
  - 文本分析
  - 表格数据
  - 算法提示
  - 中间状态承诺
---

# CAST: Achieving Stable LLM-based Text Analysis for Data Analytics

**会议**: ACL 2026  
**arXiv**: [2602.15861](https://arxiv.org/abs/2602.15861)  
**代码**: https://github.com/jxtse/CAST-text-analysis  
**领域**: LLM评估  
**关键词**: 输出稳定性, 文本分析, 表格数据, 算法提示, 中间状态承诺

## 一句话总结

提出CAST框架，通过算法提示（Algorithmic Prompting）和先思考后输出（Thinking-before-Speaking）两种机制约束LLM的潜在推理路径，显著提升文本摘要和标注任务的运行间稳定性，同时不损失输出质量。

## 研究背景与动机

**领域现状**：Text Analysis for Data Analysis (TADA) 是将表格中的自由文本列转化为结构化表示（如摘要主题、行级标签）的范式。LLM是执行TADA的天然候选工具，一个模型通过自然语言查询即可完成多种文本分析任务。

**现有痛点**：LLM生成的概率性本质与数据分析的确定性需求之间存在根本冲突。同一输入在不同运行中可能产生语义漂移的输出（如同一条评论被标记为"Customer Service"或"Support Team"），导致下游的过滤、分组、聚合结果不一致，破坏可复现性和信任度。

**核心矛盾**：不稳定性的根源在于LLM内部存在无约束的潜在推理轨迹。从概率视角看，提示LLM会在可能的推理路径上诱导一个分布；当该分布熵较高（模型对下一步推理不确定）时，微小的随机波动就会导致输出漂移。现有方法（如Self-Consistency）通过多次采样投票来提升正确性，但不针对稳定性设计。

**本文目标**：在不依赖重复采样的前提下，通过约束生成过程中的推理路径来实现输出稳定性。

**切入角度**：作者观察到，让模型在生成最终输出前先产生相关的中间推理状态，即使不指定具体内容，也能显著降低输出长度和内容的方差。

**核心 idea**：通过算法提示提供程序化脚手架约束推理转移，通过先思考后输出机制将关键中间状态显式固定，两者协同使推理路径集中在少数高概率轨迹上。

## 方法详解

### 整体框架

CAST以单次结构化LLM调用实现：输入为表格文本数据和分析查询，输出为稳定的结构化结果（摘要或标签）。同一模板通过切换任务特定的schema和约束来适配不同任务，内部流程包含中间承诺写入和自验证。

### 关键设计

1. **算法提示 (Algorithmic Prompting, AP)**:

    - 功能：为任务提供程序化脚手架，约束有效的推理状态转移
    - 核心思路：将经典的确定性工作流和专家启发式编码为结构化提示序列。对于摘要任务，指导LLM先解释查询、分解约束，再按算法流程执行。形式上，AP在每一步引入门控函数 $g_t(z_t, z_{<t}, x)$，通过硬掩码或软加权使概率质量集中在更少的合理下一状态上，降低局部不确定性 $H(Z_t|Z_{<t}, x, \mathcal{C}_{AP})$
    - 设计动机：无约束的推理转移是不稳定性的根源，AP通过提供确定性的分析工作流作为强先验来"修剪"无效推理路径

2. **先思考后输出 (Thinking-before-Speaking, TbS)**:

    - 功能：通过强制模型显式承诺关键中间状态来减少路径分歧
    - 核心思路：不让模型隐式遍历推理轨迹只暴露最终输出，而是要求模型依次生成中间状态（如领域判断、主题schema、聚类结果），每个后续生成都以先前承诺为条件。基于条件化降低熵的信息论原理：$H(Z_{>t}|X=x, Z_{\leq t}) \leq H(Z_{>t}|X=x)$
    - 设计动机：一旦schema、主题集或领域决策被固定，后续生成被迫与之保持一致，使推理路径对微小随机波动不敏感

3. **稳定性评估指标 (CAST-S/CAST-T)**:

    - 功能：专门量化运行间稳定性的评估指标
    - 核心思路：CAST-S用于摘要，结合语义分数 $S_{sem}$（内容重叠）和位置分数 $S_{pos}$（基于Kendall's Tau的排序一致性），$S_{CAST-S}(\alpha) = \alpha \cdot S_{sem} + (1-\alpha) \cdot S_{pos}$，$\alpha=0.9$时与人类判断相关性最高（$r=0.813$）。CAST-T用于标注，先由LLM按语义等价聚类多次运行的标签，再计算主导聚类的比例
    - 设计动机：现有的ROUGE-L、余弦相似度等指标不敏感于分析场景中重要的语义漂移和排序变化

### 损失函数 / 训练策略

CAST是纯推理时方法，不涉及训练。通过精心设计的结构化提示在单次API调用中实现约束推理，无需多次采样或投票。

## 实验关键数据

### 主实验

| 模型 | 方法 | 摘要稳定性(CAST-S)↑ | 独立标注准确率↑ | 联合标注稳定性(CAST-T)↑ |
|------|------|---------------------|-----------------|------------------------|
| GPT-5.2 | Baseline | 9.24 | 95.0% | 9.40 |
| GPT-5.2 | Self-Consistency | 7.40 | 96.2% | 9.16 |
| GPT-5.2 | CAST | **9.39** | **98.2%** | **9.60** |
| DeepSeek-V3.2 | Baseline | 8.15 | 92.7% | 8.78 |
| DeepSeek-V3.2 | CAST | **9.47** | 95.6% | **9.14** |
| Gemini-3-Flash | Baseline | 9.80 | 96.0% | 8.18 |
| Gemini-3-Flash | CAST | **9.93** | **96.8%** | 8.26 |

### 消融实验

| 配置 | 摘要稳定性 (DeepSeek) | 说明 |
|------|----------------------|------|
| Full CAST (AP+TbS) | 9.47 | 完整模型 |
| AP Only | 8.97 | 仅算法提示 |
| TbS Only | 9.46 | 仅先思考后输出 |
| Few-shot | 8.96 | 少样本提示 |
| Self-Consistency | 7.06 | 多次采样投票反而更差 |

### 关键发现
- Self-Consistency在稳定性上反而最差，因为其扩散采样不适合可靠的事后聚合，且计算开销是CAST的3倍以上
- AP和TbS有协同效应，完整CAST通常优于单独使用任一组件
- CAST在提升稳定性的同时还略微提升了摘要质量（recall从0.854提升至0.879）

## 亮点与洞察
- 从信息论角度形式化了LLM输出不稳定性的机制——推理路径的高熵，并给出了约束推理降低熵的理论框架，这比经验性调参更有说服力
- 发现即使不指定中间状态的具体内容，仅要求模型产生相关中间推理就能降低输出方差，这个观察极其实用
- CAST-S/CAST-T评估指标填补了稳定性量化的空白，适用于任何需要LLM输出一致性的场景

## 局限与展望
- 算法脚手架目前需要人工设计，扩展到全新任务领域可能受限
- 实验主要覆盖摘要和标注，未验证在更复杂的TADA组合工作流中的效果
- 过度约束可能抑制某些分析场景中有价值的细微变化

## 相关工作与启发
- **vs Self-Consistency**: SC通过多次采样投票提升正确性，但不保证稳定性，且计算成本高。CAST用单次调用通过约束推理路径实现稳定性
- **vs Algorithm-of-Thoughts**: AoT目标是提升正确性，CAST目标是提升稳定性，是对"约束推理"思路的不同应用方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统化地研究LLM在数据分析场景中的输出稳定性问题
- 实验充分度: ⭐⭐⭐⭐ 3个模型、多个基线和消融、人类评估验证指标
- 写作质量: ⭐⭐⭐⭐⭐ 理论框架和实证观察结合紧密，叙述清晰
- 价值: ⭐⭐⭐⭐ 对LLM在生产环境中的可靠部署有重要参考价值

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)
- [\[ICLR 2026\] When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](../../ICLR2026/llm_evaluation/when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)
- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)

<!-- RELATED:END -->
