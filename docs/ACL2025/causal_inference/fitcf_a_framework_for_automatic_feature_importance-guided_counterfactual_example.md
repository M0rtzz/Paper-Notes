---
title: >-
  [论文解读] FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation
description: >-
  [ACL 2025][反事实生成] 提出 FitCF 框架，利用 BERT 特征归因方法（LIME/IG/SHAP等）提取重要词来引导 LLM 在 zero-shot 下生成反事实样本（ZeroCF），再经标签翻转验证筛选后作为 few-shot 示例，在新闻分类和情感分析任务上一致性超越 Polyjuice、BAE、FIZLE 三种基线。
tags:
  - ACL 2025
  - 反事实生成
  - 特征归因
  - few-shot提示
  - 标签翻转验证
  - 可解释性
---

# FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation

**会议**: ACL 2025  
**arXiv**: [2501.00777](https://arxiv.org/abs/2501.00777)  
**代码**: [https://github.com/qiaw99/FitCF](https://github.com/qiaw99/FitCF)  
**作者**: Qianli Wang, Nils Feldhus, Simon Ostermann, Luis Felipe Villa-Arenas, Sebastian Möller, Vera Schmitt
**机构**: TU Berlin, DFKI, Deutsche Telekom, Saarland Informatics Campus
**领域**: 因果推理 / 可解释AI  
**关键词**: 反事实生成, 特征归因, few-shot提示, 标签翻转验证, 可解释性

## 一句话总结

提出 FitCF 框架，利用 BERT 特征归因方法（LIME/IG/SHAP等）提取重要词来引导 LLM 在 zero-shot 下生成反事实样本（ZeroCF），再经标签翻转验证筛选后作为 few-shot 示例，在新闻分类和情感分析任务上一致性超越 Polyjuice、BAE、FIZLE 三种基线。

## 研究背景与动机

**领域现状**：反事实样本（counterfactual examples）在 NLP 中广泛用于数据增强和模型可解释性。现有自动生成方法包括 Polyjuice（微调 GPT-2）、BAE（BERT 掩码替换）、FIZLE（LLM zero-shot）等。

**现有方法的不足**：
   - 众包获取反事实成本高、不可扩展
   - FIZLE 依赖 LLM 自身提取重要词，但 LLM 存在幻觉问题——实验表明 Llama3-8B 在 AG News 上有 **64.5%** 的实例中，LLM 提取的重要词根本不存在于原始输入中
   - Zero-shot 设置下生成的反事实质量不稳定，标签翻转率低

**核心动机**：将特征归因方法（post-hoc 可解释性）与反事实生成（另一种可解释性方法）互补结合，利用经过验证的结构化特征重要性引导 LLM 生成更高质量的反事实样本

## 方法详解

### 3.1 ZeroCF（Zero-shot 反事实生成）

ZeroCF 的核心流程分为三步：

1. **预测**：用在目标数据集上微调的 BERT 模型对输入 x 进行分类，得到预测标签
2. **特征归因打分**：部署解释器（Ferret 框架），使用四种特征归因方法（Gradient、Integrated Gradients、LIME、SHAP）计算输入中每个词的重要性得分
3. **反事实生成**：提取 top 重要词 w，将任务指令、重要词、预测标签和原始输入组合为 prompt，以 zero-shot 方式提交给 LLM 生成反事实

**与 FIZLE 的关键区别**：FIZLE 让 LLM 自己识别重要词（容易幻觉），ZeroCF 使用 BERT 模型的特征归因分数提取更忠实的重要词。

### 3.2 FitCF（Few-shot 反事实生成框架）

FitCF 在 ZeroCF 基础上引入三个核心组件：

1. **Top-k 示例采样**：用 SBERT 将所有样本编码为句子嵌入，进行 k-means 聚类，从每个聚类中选取最接近质心的 c 个样本，确保示例多样性
2. **标签翻转验证**：对 ZeroCF 生成的反事实用 BERT 模型再次预测，仅保留预测标签确实发生翻转的反事实，排除无效样本
3. **Few-shot 反事实生成**：将验证通过的 input-counterfactual 对作为 demonstrations，结合当前输入的重要词，以 few-shot 方式提示 LLM 生成反事实

**设计选择**：

- 示例数 l = 2k（AG News: 10个，SST2: 8个）
- BERT 同时作为特征归因模型和标签翻转验证器，因其效率-性能平衡好
- 也可替换为其他分类模型（encoder-decoder 或 decoder-only 可用 Inseq 提取归因分数）

## 实验关键数据

### 实验设置

- **数据集**：AG News（4类新闻分类）、SST2（二元情感分类）
- **LLM**：Llama3-8B、Qwen2.5-32B、Qwen2.5-72B
- **特征归因方法**：Gradient、Integrated Gradients (IG)、LIME、SHAP
- **评估指标**：Soft Label Flip Rate (SLFR↑)、Perplexity (PPL↓)、Textual Similarity (TS↓)

### 主实验结果（Table 1，AG News + SST2）

| 方法 | 模型 | 归因方法 | SLFR(AG)↑ | PPL(AG)↓ | TS(AG)↓ | SLFR(SST2)↑ | PPL(SST2)↓ | TS(SST2)↓ |
|------|------|----------|-----------|----------|---------|-------------|------------|-----------|
| Polyjuice | GPT2 | - | 18.60% | 121.76 | 0.50 | 29.00% | 258.32 | 0.71 |
| BAE | BERT | - | 19.50% | 168.44 | 0.12 | 47.00% | 367.06 | 0.09 |
| FIZLE | Llama3-8B | - | 93.50% | 123.67 | 0.61 | 95.50% | 202.22 | 0.52 |
| ZeroCF | Llama3-8B | SHAP | **98.00%** | **99.08** | 0.27 | 94.00% | **204.76** | 0.46 |
| ZeroCF | Llama3-8B | IG | 95.50% | 109.09 | **0.27** | **99.50%** | 222.51 | **0.42** |
| FitCF | Llama3-8B | LIME | 95.50% | **75.15** | **0.19** | **100.00%** | **151.22** | 0.48 |
| FitCF | Llama3-8B | IG | 96.00% | 87.67 | 0.23 | **100.00%** | 161.88 | 0.48 |
| FIZLE | Qwen2.5-72B | - | 21.50% | 84.09 | 0.22 | 92.00% | 257.91 | 0.43 |
| FitCF | Qwen2.5-72B | Gradient | **77.00%** | 62.13 | 0.99 | **96.00%** | 595.71 | 0.38 |
| FitCF | Qwen2.5-72B | LIME | 45.00% | **61.54** | **0.35** | **96.50%** | 240.94 | 0.41 |

**关键发现**：FitCF 在所有 LLM + 数据集组合上一致超越三种基线；Llama3-8B（最小模型）反而取得最好整体性能。

### 消融实验汇总（Qwen2.5-72B，Table 2-4）

| 消融组件 | AG News SLFR 变化 | SST2 SLFR 变化 | 影响程度 |
|----------|-------------------|----------------|----------|
| 去掉重要词 | ↓1.96%~35.50% | ↓0%~2.50% | 中等 |
| 减半示例数（l=k） | ↓22.00%~63.50% | ↓1.50%~7.00% | **最大** |
| 去掉标签翻转验证 | ↓1.50%~43.00% | ↓0.50%~2.00% | 中等 |

**结论**：示例数量是 FitCF 性能最关键的因素；LIME 和 SHAP 组合最稳健。

### 忠实度与反事实质量相关性（Table 5）

| 归因方法 | AG News comp.↑ | AG News suff.↓ | SST2 comp.↑ | SST2 tau(loo)↑ |
|----------|----------------|----------------|-------------|----------------|
| Gradient | 0.12~0.20 | 0.12~0.13 | 0.20~0.21 | -0.03 |
| IG | 0.32~0.38 | 0.03 | 0.50~0.52 | 0.21~0.22 |
| LIME | 0.53~0.61 | -0.02~-0.01 | 0.67~0.68 | 0.29 |
| SHAP | 0.53~0.62 | -0.02~-0.01 | 0.59~0.60 | 0.25 |

LIME 和 SHAP 的特征归因忠实度显著优于 Gradient 和 IG，且与反事实质量呈强正相关。

## 亮点与创新

1. **两种可解释性方法的互补结合**：首次系统性地将特征归因（feature importance）与反事实生成（counterfactual generation）结合，解决了 LLM 直接提取重要词的幻觉问题
2. **全自动流水线**：从示例采样到 ZeroCF 生成再到标签翻转验证再到 few-shot 生成，无需人工标注的反事实数据
3. **深入的消融分析**：三个核心组件的消融实验清晰展示各组件贡献，尤其揭示了示例数量 > 重要词 > 标签翻转验证的影响排序
4. **忠实度-质量相关性发现**：发现特征归因分数的忠实度与反事实质量强正相关，为未来研究指明方向

## 局限性

1. **仅英文实验**：实验仅在英文数据集上进行，其他语言效果未验证
2. **依赖 BERT 微调模型**：特征归因和标签翻转验证都依赖于目标数据集上微调的 BERT，未探索其他模型架构
3. **任务范围有限**：仅验证了文本分类任务（4分类和2分类），未涉及问答、生成等更复杂任务
4. **忠实度评估不完整**：仅评估了忠实度（faithfulness），未评估合理性（plausibility），后者需要人类标注
5. **编辑距离权衡**：部分设置下 FitCF 的编辑距离较大（如 Qwen2.5-32B 在 AG News 上），说明反事实与原文偏差较大

## 相关工作

- **反事实生成**：MICE (Ross et al., 2021)、Polyjuice (Wu et al., 2021)、DISCO (Chen et al., 2023)、Tigtec (Bhan et al., 2023)、FIZLE (Bhattacharjee et al., 2024)
- **可解释性方法组合**：CREST (Treviso et al., 2023)、Krishna et al. (2023)、Gressel et al. (2023)
- **特征归因工具**：Ferret (Attanasio et al., 2023)、Inseq (Sarti et al., 2023)
- **Auto-CoT**：Zhang et al. (2023)，FitCF 的自动构建示例思路受其启发

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 两种可解释性方法的互补结合思路新颖，标签翻转验证设计巧妙 |
| 实验充分性 | 4 | 3种LLM x 4种归因方法 x 2个数据集 + 3组消融 + 相关性分析，覆盖全面 |
| 写作质量 | 4 | 结构清晰，图表信息量大，方法描述规范 |
| 实用性 | 3 | 框架依赖多个组件（BERT微调+特征归因+LLM），部署链较长 |
| 影响力 | 3 | 方法扎实但应用场景（自动反事实生成）偏niche |

**总分**: 3.6/5 — 方法设计扎实、实验全面，将特征归因引入反事实生成是有意义的贡献。主要遗憾在于任务范围局限于简单文本分类，缺乏更复杂场景验证。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation](../../ACL2026/causal_inference/parallel_universes_parallel_languages_a_comprehensive_study_on_llm-based_multili.md)
- [\[ACL 2025\] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)
- [\[ACL 2025\] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](iris_an_iterative_and_integrated_framework.md)
- [\[ICCV 2025\] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](../../ICCV2025/causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)
- [\[NeurIPS 2025\] Demystifying Spectral Feature Learning for Instrumental Variable Regression](../../NeurIPS2025/causal_inference/demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)

<!-- RELATED:END -->
