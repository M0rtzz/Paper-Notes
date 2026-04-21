---
title: >-
  [论文解读] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language
description: >-
  [ICLR 2026][mechanistic interpretability] 提出 semantic regexes——一种用于自动描述 LLM 特征的结构化语言，通过 symbol/lexeme/field 三种原语及 context/composition/quantification 修饰符，在保持与自然语言同等准确度的同时，实现了更简洁、更一致的特征描述，并可量化特征复杂度随层的变化趋势。
tags:
  - ICLR 2026
  - mechanistic interpretability
  - feature description
  - structured language
  - sparse autoencoder
  - automated interpretability
---

# Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language

**会议**: ICLR 2026  
**arXiv**: [2510.06378](https://arxiv.org/abs/2510.06378)  
**代码**: https://github.com/apple/ml-semantic-regex  
**领域**: LLM/NLP  
**关键词**: mechanistic interpretability, feature description, structured language, sparse autoencoder, automated interpretability

## 一句话总结
提出 semantic regexes——一种用于自动描述 LLM 特征的结构化语言，通过 symbol/lexeme/field 三种原语及 context/composition/quantification 修饰符，在保持与自然语言同等准确度的同时，实现了更简洁、更一致的特征描述，并可量化特征复杂度随层的变化趋势。

## 研究背景与动机
自动可解释性（automated interpretability）旨在将 LLM 的内部特征翻译为人类可理解的描述。现有方法（如 Bills et al. 2023; Paulo et al. 2024）使用自然语言来描述特征，但自然语言存在三大核心痛点：

**模糊性**：自然语言描述往往过于冗长或含糊，不同人对同一特征的描述差异很大

**不一致性**：功能相同的特征可能得到完全不同的自然语言描述，不利于冗余特征检测和电路分析

**缺乏结构**：自然语言无法直接编码特征复杂度，难以进行模型级别的系统性分析

核心矛盾在于：自然语言的灵活性虽然足以描述单个特征，但其不确定性阻碍了大规模、系统性的特征分析。受正则表达式和编程语言的启发，作者切入的角度是设计一种兼具**精确性**和**表达力**的结构化语言，既能精确描述特征行为，又能为模型级分析提供结构化接口。

## 方法详解

### 整体框架
Semantic regex 语言嵌入标准的自动可解释性流水线中：给定一个 subject model 的特征及其激活数据，由 explainer model（GPT-4o-mini）生成 semantic regex 描述，再由 evaluator model 对描述质量进行评分。关键设计是将 **描述格式** 与 **生成流程** 解耦——只需修改 prompt 中的语言规范即可替换描述语言。

### 关键设计
1. **三级原语（Primitives）**：对应特征的三种抽象层次

    - `[:symbol X:]`：匹配精确字符串，如 `[:symbol color:]` 仅匹配 "color"
    - `[:lexeme X:]`：匹配词的语法变体，如 `[:lexeme color:]` 匹配 "color/colors/coloring"
    - `[:field X:]`：匹配语义相关词，如 `[:field color:]` 匹配 "red/blue/green"
   
   三级原语从精确到抽象，反映了 LLM 特征从底层 token 检测到高层语义概念的渐变。

2. **三类修饰符（Modifiers）**：扩展原语表达力

    - **Context**：`@{:context X:}(semantic regex)` 限定语义上下文，如 `@{:context politics:}([:symbol color:])` 仅在政治语境中匹配 "color"
    - **Composition**：支持序列组合和交替 `|`，如 `[:field color:]([:symbol and:]|[:symbol or:])[:field color:]`
    - **Quantification**：使用正则量词 `?` 表示可选，如 `[:symbol a:][:field color:]?[:field flower:]`

3. **语言设计方法论**：采用 grounded-theory 方法，通过在 Neuronpedia 上手动调研数千个特征，迭代性地引入新原语/修饰符，直到达到饱和——能描述所有观察到的特征模式。

### 训练策略
- Subject model：GPT-2-Small + Gemma-2-2B，使用 SAE 提取的 residual layer 特征
- Explainer/Evaluator：GPT-4o-mini
- Semantic regex 的生成 prompt 基于 max-acts 方法改造：更新指令为 semantic regex 语法 + 添加语法定义 + 修改 few-shot 示例
- 展示 top-10 激活例子，要求模型先输出简短解释再输出 semantic regex

## 实验关键数据

### 主实验（准确度对比）
在 GPT-2-RES-25k、Gemma-2-2B-RES-16k、Gemma-2-2B-RES-65k 上，每层评估 100 个特征：

| 方法 | Clarity (Gen.) | Detection (Disc.) | Fuzzing (Disc.) | Responsiveness (Disc.) | Faithfulness |
|------|:-:|:-:|:-:|:-:|:-:|
| token-act-pair | 基线 | 基线 | 基线 | 基线 | 基线 |
| max-acts | 中 | 中 | 中 | 中 | 中 |
| **semantic-regex** | **≥ max-acts** | **≥ token-act-pair** | **≥ token-act-pair** | **≥ max-acts** | 持平 |

- Semantic regex 在 clarity 上跨所有模型显著优于 token-act-pair (p<0.05)
- 在 detection/fuzzing/responsiveness 上对 GPT-2 和 Gemma-65k 显著优于 token-act-pair
- 非劣效性检验确认 semantic regex 与自然语言在准确度上无显著差距

### 消融实验（简洁性与一致性）

| 指标 | semantic-regex | max-acts | token-act-pair |
|------|:-:|:-:|:-:|
| 描述长度中位数（字符） | **41** (IQR: 19-59) | 139 (IQR: 119-166) | 55 (IQR: 46-66) |
| 一致性（相同描述比例） | **33.6%** | 0.0% | 12.2% |

- Semantic regex 比 max-acts 短 **3.4 倍**，比 token-act-pair 短 **1.3 倍**
- 一致性方面，semantic regex 是 token-act-pair 的 **2.8 倍**

### 关键发现
1. **特征复杂度随层增加**：早期层以简单 symbol 为主，后期层需要更多组合和 field 原语，平均组件数随层递增。symbol 占比随层下降，field 占比随层上升
2. **用户研究（24人）**：semantic regex 在 12 个特征中的 9 个上帮助用户构建了更准确的心智模型（正样本激活 vs 反例激活差异更大）
3. 用户仅需极少指导即可理解 semantic regex，收到的自然语言描述澄清问题反而更多
4. 自然语言描述中的额外细节常误导用户，而 semantic regex 的简洁性反而降低了认知负荷

## 亮点与洞察
- **结构化不等于降低表达力**：约束语言反而减少了噪声，提升了可用性
- **从个体特征到模型级分析**：原语类型分布可作为层复杂度的 proxy，无需额外探针或测试
- **与正则表达式的精妙类比**：正则表达式描述字符模式，semantic regex 描述语义模式，自然地桥接了符号系统与神经网络表示
- **工程上的解耦设计**：只需修改 prompt 即可集成到现有流水线，兼容未来方法

## 局限与展望
- 过于简洁的描述可能导致歧义（如 `[:field musician:]` 匹配"吉他手"还是仅限"知名音乐人"）
- 非唯一映射：同一特征可有多个合法 semantic regex，缺乏规范化的"风格指南"
- 某些组件未完全定义（如大小写敏感性），可能导致模型行为不一致
- 对多义特征（polysemanticity）支持较弱，高度纠缠的概念仍产生不连贯描述
- 模型需从极少示例学习新语言，偶尔出现"语法错误"

## 相关工作与启发
- 与 SAE/transcoder 等特征提取方法正交互补：semantic regex 负责描述，SAE 负责发现
- 与 Neuronpedia 平台深度集成，可直接用于交互式特征探索
- 启发：是否可以设计**领域特定**的结构化语言（如用于安全特征、注意力头、多模态特征）
- 类比编程语言发展史：不同解释性任务可能需要不同的"解释性编程语言"
- 对电路分析（circuit tracing）的直接支持：一致性描述使冗余特征识别变得可自动化
- 对 Gur-Arieh et al. (2025) 的 output-centric 方法互补：semantic regex 关注激活模式，后者关注输出影响

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将结构化语言引入 automated interpretability，概念新颖且方法论扎实
- 实验充分度: ⭐⭐⭐⭐ 多模型、多指标、含用户研究，但仅限 GPT-2 和 Gemma-2-2B
- 写作质量: ⭐⭐⭐⭐⭐ 可视化精美，论证逻辑清晰，类比恰当
- 价值: ⭐⭐⭐⭐ 为机械可解释性提供了新工具和新分析维度，但实际落地还需生态支持

<!-- RELATED:START -->

## 相关论文

- [Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context](implicit_statistical_inference_in_transformers_approximating_likelihood-ratio_te.md)
- [Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)
- [Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)
- [Internal Planning in Language Models: Characterizing Horizon and Branch Awareness](internal_planning_in_language_models_characterizing_horizon_and_branch_awareness.md)

<!-- RELATED:END -->
