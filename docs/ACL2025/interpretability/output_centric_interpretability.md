---
title: >-
  [论文解读] Enhancing Automated Interpretability with Output-Centric Feature Descriptions
description: >-
   提出基于输出的特征描述方法（VocabProj和TokenChange），弥补了现有自动化可解释性管线仅依赖输入激活样本的局限，结合输入-输出双视角的集成方法在两类评估中均取得最优表现。
tags:

---

# Enhancing Automated Interpretability with Output-Centric Feature Descriptions

## 论文信息

- **会议**: ACL 2025
- **arXiv**: [2501.08319](https://arxiv.org/abs/2501.08319)
- **代码**: [https://github.com/yoavgur/Feature-Descriptions](https://github.com/yoavgur/Feature-Descriptions)
- **领域**: 可解释性 / 自动化解释管线
- **关键词**: 自动化可解释性, 特征描述, 输出导向, SAE特征, 模型转向

## 一句话总结

提出基于输出的特征描述方法（VocabProj和TokenChange），弥补了现有自动化可解释性管线仅依赖输入激活样本的局限，结合输入-输出双视角的集成方法在两类评估中均取得最优表现。

## 研究背景与动机

- **领域现状**: 自动化可解释性管线（如Bills et al., 2023）通过LLM描述模型特征（神经元、SAE方向等）所编码的概念，广泛采用MaxAct方法——收集最大激活输入样本让LLM生成描述。
- **MaxAct的三大缺陷**: (1) 计算成本高：需要在大规模语料上收集激活数据；(2) 因果不完整：仅描述"什么输入激活特征"而忽略"特征激活如何影响输出"；(3) 数据集依赖：不同数据集可能导致不一致的描述，甚至将有意义的特征误判为"死特征"。
- **关键洞察**: 特征的机制性角色由因果关系的两个方向决定——输入如何激活特征（input→feature）和特征激活如何影响输出（feature→output）。特征描述在模型转向(steering)等下游应用中应当是输出导向的。
- **核心提案**: 提出两种高效的输出导向方法，分别基于词表投影和token概率变化，并与MaxAct互补结合。

## 方法详解

### 整体框架

提出输入-输出双面评估框架：输入侧评估描述对激活触发条件的刻画准确性，输出侧评估描述对特征因果效应的捕获能力。在此框架下比较三种方法及其集成。

### 关键设计

1. **VocabProj（词表投影）**: 将特征向量 $\mathbf{v}_f$ 通过unembedding矩阵投影到词表空间 $\mathbf{w} = W_U \cdot \text{LayerNorm}(\mathbf{v}_f)$，取得分最高/最低的token作为该特征"促进/抑制"的概念，仅需一次矩阵乘法
2. **TokenChange（Token变化）**: 在k个随机prompt上分别运行原始模型和激活特征后的模型，计算各token logit的平均变化，取变化最大的token作为特征影响的概念描述，需≤2次推理
3. **双面评估框架**: 输入侧让LLM根据描述生成应激活/不激活样本，比较平均激活值；输出侧通过模型转向生成三组文本（目标特征 vs 两个随机特征），让judge LLM判断哪组匹配描述

### 集成策略

- **Ensemble Raw**: 将多种方法的原始数据（激活样本、top tokens等）拼接后送入explainer LLM生成统一描述
- **Ensemble Concat**: 简单拼接各方法生成的描述文本

## 实验

### 主实验：不同模型/特征类型上的输入-输出评估 (%, 越高越好)

| 方法 | Gemma-2 Res. SAE (Input/Output) | Gemma-2 MLP SAE (Input/Output) | Llama-3.1 Inst. MLP (Input/Output) |
|------|-------------------------------|------------------------------|----------------------------------|
| MaxAct | 56.6 / 49.2 | 50.4 / 35.1 | 85.6 / 36.9 |
| VocabProj | 50.1 / 56.5 | 20.9 / 37.2 | 71.2 / **45.8** |
| TokenChange | 44.7 / 54.9 | 22.3 / 40.3 | 74.0 / 43.8 |
| EnsembleR (All) | **66.6** / 64.9 | **55.7** / 48.7 | 86.2 / 41.8 |
| EnsembleC (All) | 57.7 / **66.9** | 31.6 / **49.9** | 84.9 / **44.6** |

### 消融分析：特征描述关系的人工分类 (100个Gemma Scope SAE特征)

| 描述关系类型 | 比例 | 说明 |
|------------|------|------|
| Similar (相似) | 41% | 输入输出描述高度一致 |
| Composition (组合) | 23% | 描述不同方面，组合后更全面 |
| Abstraction (抽象) | 23% | 输出描述是输入描述的更高层抽象 |
| Different (不同) | 13% | 描述不同方面，无明显关联 |

### 关键发现

1. **输入与输出视角互补**: MaxAct在输入评估上占优（+6-15%），VocabProj/TokenChange在输出评估上占优（+7-15%），说明两类方法捕获了不同的特征信息
2. **集成一致性最优**: Ensemble Raw在输入评估最优，Ensemble Concat在输出评估最优；三方法集成在所有模型/特征类型上一致优于任何单一方法
3. **死特征可复活**: 对Gemma-2中1850个"死特征"，通过VocabProj和TokenChange描述生成的探测输入成功激活了9.1%的MLP特征和62%的残差特征
4. **层位效应**: VocabProj在早期层表现较差但逐层提升，与"logit lens"的已有观察一致
5. **MLP vs 残差**: 输出评估在MLP特征上显著低于残差特征（45-50 vs ~66），可能因MLP层对残差流的影响是渐进式的

## 亮点

- 首次系统性提出特征描述的"输入-输出双面性"框架，填补了仅关注输入侧的空白
- VocabProj仅需一次矩阵乘法，计算效率远超MaxAct（需大规模语料扫描）
- 死特征复活实验（62%残差特征被成功激活）直接证明了输出导向方法的独特价值
- 方法通用：支持SAE特征、MLP神经元、残差流等多种特征类型

## 局限性

- 输出侧评估噪声较大，虽通过大量采样缓解但仍有改进空间
- 输出导向方法依赖模型词表，无法描述非词表概念（如位置特征）
- 未区分特征对概念的"促进"与"抑制"方向
- 未涉及补丁式(patching-based)等其他可解释性方法的对比
- 集成方法对prompt设计敏感

## 相关工作

- **自动化可解释性**: Bills et al. (2023) GPT-4解释GPT-2神经元；Bricken et al. (2023) SAE特征解释
- **特征描述改进**: Paulo et al. (2024) 优化explainer prompt；Choi et al. (2024) Transluce描述选择
- **模型内部表示**: Geva et al. (2021, 2022) MLP作为key-value记忆；logit lens方法
- **模型转向**: Templeton et al. (2024) 特征clamp转向；Rimsky et al. (2024) 行为控制
- **SAE训练**: Gemma Scope (Lieberum et al., 2024)、Llama Scope (He et al., 2024)

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总评 | 8/10 |

> **总结**: 本文的核心贡献在于将特征描述从单一的输入视角扩展为输入-输出双面问题，提出的VocabProj方法计算成本极低且效果可观。死特征复活实验尤为亮眼，直接证明了输出导向方法的不可替代性。对可解释性研究社区具有重要的方法论启示。

<!-- RELATED:START -->

## 相关论文

- [Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [Mechanistic Interpretability of Emotion Inference in Large Language Models](mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](normalized_aopc_faithfulness_metrics.md)
- [VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set](../../NeurIPS2025/interpretability/vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)
- [MIB: A Mechanistic Interpretability Benchmark](../../ICML2025/interpretability/mib_a_mechanistic_interpretability_benchmark.md)

<!-- RELATED:END -->
