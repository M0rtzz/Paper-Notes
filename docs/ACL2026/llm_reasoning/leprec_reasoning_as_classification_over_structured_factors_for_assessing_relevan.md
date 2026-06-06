---
title: >-
  [论文解读] LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues
description: >-
  [ACL 2026][LLM推理][法律问题相关性评估] 本文提出 LePREC，一种受法律专业人士启发的神经-符号框架，通过 LLM 生成推理问答对将非结构化法律文本转化为结构化特征，再利用稀疏线性模型进行相关性分类…
tags:
  - "ACL 2026"
  - "LLM推理"
  - "法律问题相关性评估"
  - "神经符号推理"
  - "特征选择"
  - "法律AI"
  - "结构化因子分类"
---

# LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues

**会议**: ACL 2026  
**arXiv**: [2604.19464](https://arxiv.org/abs/2604.19464)  
**代码**: 无  
**领域**: 法律NLP / 可解释性  
**关键词**: 法律问题相关性评估, 神经符号推理, 特征选择, 法律AI, 结构化因子分类

## 一句话总结

本文提出 LePREC，一种受法律专业人士启发的神经-符号框架，通过 LLM 生成推理问答对将非结构化法律文本转化为结构化特征，再利用稀疏线性模型进行相关性分类，在 769 个马来西亚合同法案例构建的 LIC 数据集上相比 GPT-4o 等 LLM 基线提升 30–40%。

## 研究背景与动机

**领域现状**：全球超过一半人口难以满足其民事司法需求。在 IRAC（Issue-Rule-Application-Conclusion）框架中，法律问题识别是关键的第一步，包括生成候选法律问题和评估其相关性。LLM 虽展现了强大的语言能力，但在真实法律场景中的精度仍然不足。

**现有痛点**：现有法律 AI 基准多限于简化或合成场景（如教科书案例），缺少基于真实法院案例的专家标注数据集。直接使用 GPT-4o 进行法律问题相关性评估仅达到 62% 的精度，因为 LLM 无法区分"与事实相关"和"真正涉及案件核心争议"的问题。

**核心矛盾**：法律专业人士评估相关性时需要考虑管辖权约束、程序性背景和案件特定因素等多层次上下文，而 LLM 倾向于进行表面事实匹配，缺乏深层法律推理能力。端到端的"黑箱"方法无法提供这种细粒度的判断。

**本文目标**：(1) 构建首个基于真实法院案例的法律问题相关性评估数据集 LIC；(2) 提出一种数据高效、可解释的神经-符号框架 LePREC，将法律推理转化为结构化因子上的统计分类。

**切入角度**：观察到法律专业人士的分析遵循两阶段过程——先识别关键分析因子（brainstorming），再权衡这些因子做出判断。这种分解天然对应神经-符号范式：神经部分提取因子，符号部分进行权衡推理。

**核心 idea**：将法律问题相关性评估从"事实-问题关系评估"重构为"因子-问题相关性分类"，通过 LLM 生成二值推理问题作为结构化特征，再用稀疏线性模型学习显式代数权重，实现可解释且数据高效的相关性判断。

## 方法详解

### 整体框架

LePREC 由两个阶段组成：(1) 神经组件——利用 LLM 从事实-问题对中生成二值推理问题并计算回答概率，将非结构化法律文本转化为结构化特征向量；(2) 符号组件——在离散特征上应用稀疏线性模型，学习显式权重进行相关性分类。输入为 (事实集, 候选法律问题) 对，输出为二值相关性标签（Relevant/Irrelevant）。

### 关键设计

1. **LIC 数据集构建与增量问题生成**:

    - 功能：提供首个真实法院案例法律问题相关性评估基准
    - 核心思路：从 769 个马来西亚合同法案例中使用 GPT-4o 提取事实和问题。为增加候选问题多样性，采用增量生成策略：给定事实列表 $\mathbf{X}=\{\mathbf{x}_1,\ldots,\mathbf{x}_m\}$，逐步加入事实生成问题 $\hat{\mathcal{Y}}=\bigcup_{i=1}^{m}\hat{\mathcal{Y}}_i$，而非一次性输入所有事实。由资深法律专家标注相关性，Fleiss' $\kappa$ = 0.659
    - 设计动机：通过变化上下文"深度"，鼓励 LLM 关注不同事实组合，发现单次生成可能遗漏的细微候选问题。增量方法在 FBD、EMBD 等质量指标和 Self-BLEU、Distinct-N 等多样性指标上均优于基线

2. **神经组件：推理问题生成与回答**:

    - 功能：将非结构化法律文本转化为结构化符号特征
    - 核心思路：对 LICU 中的事实-问题对，使用 LLM 生成二值推理问题，累积形成共享问题池 $\mathcal{Q}$（共 2,486 个问题）。对每个问题 $q_t \in \mathcal{Q}$，使用生成式验证器计算回答概率 $G_{q_t}(\mathbf{X}, \hat{Y}_j) \in (0,1)$，收集为特征向量 $\mathbf{f} = G_{\mathcal{Q}}(\mathbf{X}, \hat{Y}_j) \in \mathbb{R}^h$
    - 设计动机：采用概率分数而非直接二值回答，因为初步实验表明直接回答不可靠。连续概率信息被证明对分类至关重要，均优于二值标签变体

3. **符号组件：相关性感知的线性预测**:

    - 功能：通过显式代数运算实现可解释的相关性分类
    - 核心思路：预测 $\hat{y}_j = \text{sign}(\mathbf{w}^\top \mathbf{f})$。线性模型通过学习的系数实现相关性感知的特征加权：自动降低噪声/冗余特征的权重（解决语义相似问题产生冲突结果的挑战），对领域特定问题进行自适应加权而非全局删除（解决窄域问题在不相关案例中引入噪声的挑战）
    - 设计动机：线性模型兼具符号可解释性（显式权重系数和透明代数组合）和实用优势（数据效率高、参数量与训练数据量可比），同时支持对推理问题贡献的统计分析

### 损失函数 / 训练策略

神经组件使用 GPT-4o 生成问题，生成过程与模型无关（后续稀疏特征选择自动保留最具预测性的因子）。符号组件使用标准线性分类器（SVC、LR、Ridge 等），在 LICL 上进行 5 折分层交叉验证训练。L1 正则化变体用于特征选择实验。

## 实验关键数据

### 主实验

**RQ1: SOTA LLM 基线（直接判断）**

| 方法 | F1 | Accuracy | Precision | Recall |
|------|------|------|------|------|
| Claude | 54.55 | 70.91 | 66.00 | 56.19 |
| GPT-4o | 57.80 | 70.91 | 64.46 | 58.07 |
| GenQwen | 63.70 | 68.59 | 63.84 | 63.92 |
| LegalBERT | 52.31 | 41.28 | 52.10 | 50.79 |

**RQ2: LePREC 框架（神经+符号）**

| 方法 | F1 | Accuracy | Precision | Recall |
|------|------|------|------|------|
| SVCPhi | **80.19** | 82.66 | 79.67 | **81.01** |
| LRPhi | 79.70 | 82.49 | 79.58 | 80.05 |
| RidgePhi | 80.10 | 82.91 | 80.06 | 80.28 |
| L1RegPhi | 80.01 | **83.34** | **81.13** | 79.32 |
| LDAPhi | 79.56 | 83.50 | 81.77 | 78.39 |

### 消融实验

| 配置 | F1 | 说明 |
|------|------|------|
| 线性模型 (SVC/LR/Ridge) | 79.70–80.19% | 最佳，一致且稳定 |
| 树/距离模型 (RF/KNN) | 74–75% | 略低但有竞争力 |
| 深度学习 (Transformer/FFN) | 75.44/75.65% | 非线性未带来额外增益 |
| LLM-Select 特征选择 | 45–58% | 失败，LLM 无法识别有预测力的问题 |
| L1 SVC 特征选择 | 77.60% | 仅下降 2.5 个百分点 |

### 关键发现

- LePREC 相比最佳 LLM 基线（GenQwen 63.70%）实现了约 16.5 个百分点的 F1 提升，达到 80.19%
- 线性模型（SVC、LR、Ridge）在所有分类器中表现最一致（79.70–80.19% F1），证明简单线性加权足以捕捉法律推理模式
- 稳定性分析揭示不存在普遍"黄金问题集"：L1 LR 仅 0.04–0.53% 的特征在所有折中被一致选择，L1 LR 和 L1 SVC 之间仅 38% 特征重叠
- 法律从业者访谈证实：律师不依赖固定清单推理，而是从广泛的、上下文敏感的分析因子中进行判断

## 亮点与洞察

- 将法律推理重构为结构化因子上的统计分类，巧妙地将神经-符号范式应用于法律 AI，实现了可解释性和高性能的统一
- "不存在普遍核心问题集"的发现既有定量（特征选择不稳定性）又有定性（法律从业者访谈）支撑，揭示了法律推理的根本特征
- 问题生成过程与模型无关——稀疏特征选择自动过滤模型特定噪声，这使得框架具有良好的泛化性

## 局限与展望

- 数据集仅聚焦马来西亚合同法（英联邦法系），尚未在大陆法系等其他法律体系上验证
- 依赖 LLM 生成推理问题，替代问题获取方法可能提供新洞察
- 线性模型假设线性组合能捕捉相关性模式，从详细权重分布中提取高层洞察需要仔细分析
- 部署到实际法律实践中需要额外验证以避免偏见

## 相关工作与启发

- **vs 直接 LLM 判断 (GPT-4o/Claude)**: LLM 直接判断仅达 55–58% F1，LePREC 通过分解推理过程实现 80% F1，证明结构化方法远优于端到端黑箱
- **vs LegalBERT**: 法律预训练模型因训练数据不足表现出高方差（F1 = 52.31±13.4），LePREC 通过数据高效的线性模型解决了这一问题
- **vs GCI（因果推理方法）**: GCI 的严格因果发现过度限制特征空间，LePREC 的相关性方法保留了更广泛的信号

## 评分

- 新颖性: ⭐⭐⭐⭐ 将法律推理重构为结构化因子分类的思路新颖，神经-符号分解契合法律实践
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 RQ 系统回答，14 种分类器对比，稳定性分析+从业者访谈，极为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，逻辑严密，实验设计层层递进
- 价值: ⭐⭐⭐⭐ 为法律 AI 领域提供了可解释且数据高效的新范式，LIC 数据集填补了重要空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Accurate Legal Reasoning at Scale: Neuro-Symbolic Offloading and Structural Auditability for Robust Legal Adjudication](accurate_legal_reasoning_at_scale_neuro-symbolic_offloading_and_structural_audit.md)
- [\[ACL 2026\] LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models](legaldrill_diagnosis-driven_synthesis_for_legal_reasoning_in_small_language_mode.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[ACL 2025\] Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](../../ACL2025/llm_reasoning/unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)

</div>

<!-- RELATED:END -->
