---
title: >-
  [论文解读] Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents
description: >-
  [ICML 2025][uncertainty quantification] 本文是一篇 Position Paper，通过梳理文献中 aleatoric 和 epistemic 不确定性的多种相互矛盾的定义，论证传统二分法在 LLM 交互场景中根本性失效，并提出 underspecification uncertainty（任务/上下文欠规范）、interactive learning（通过追问减少不确定性）和 output uncertainty（用自然语言而非标量表达不确定性）三个新研究方向。
tags:
  - ICML 2025
  - uncertainty quantification
  - aleatoric
  - epistemic
  - 对话系统
  - position paper
---

# Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents

**会议**: ICML 2025  
**作者**: Michael Kirchhof, Gjergji Kasneci, Enkelejda Kasneci  
**arXiv**: [2505.22655](https://arxiv.org/abs/2505.22655)  
**代码**: 无  
**领域**: 对话系统  
**关键词**: uncertainty quantification, aleatoric, epistemic, LLM agents, position paper  

## 一句话总结

本文是一篇 Position Paper，通过梳理文献中 aleatoric 和 epistemic 不确定性的多种相互矛盾的定义，论证传统二分法在 LLM 交互场景中根本性失效，并提出 underspecification uncertainty（任务/上下文欠规范）、interactive learning（通过追问减少不确定性）和 output uncertainty（用自然语言而非标量表达不确定性）三个新研究方向。

## 研究背景与动机

**领域现状**：LLM 和 chatbot 不可避免地产生幻觉（hallucination），且近期研究证明这一问题从理论上无法完全消除。不确定性量化 (UQ) 因此至关重要——要么输出一个总不确定性分数，要么分别输出 aleatoric（不可约）和 epistemic（可约）两个数值。

**现有痛点**：传统 UQ 框架面对 LLM agent 的多轮对话交互时暴露出根本性缺陷：(1) aleatoric 和 epistemic 的定义在社区内存在严重冲突；(2) 估计出的两种不确定性在实践中高度相关（相关系数 0.8-0.999），无法真正解耦；(3) 在多轮交互中，同一不确定性可以在 aleatoric 和 epistemic 之间反复转换——用户提供更多信息时"不可约"变"可约"，代理决定停止追问时"可约"又变回"不可约"。

**核心矛盾**：传统 UQ 假设输入格式固定（一张图、一段特征向量），但 LLM agent 在开放环境中工作——任务本身可能不明确，输入信息可能不完整，输出不必是概率标量而可以是解释性文本。

**切入角度**：不纠缠于哪个 aleatoric/epistemic 定义是"正确的"，而是直接论证这种二分法对 LLM agent 不适用，转而提出更贴合交互场景的不确定性分类和处理方式。

**核心 idea**：LLM agent 的不确定性不应被压缩为 aleatoric/epistemic 两个标量，而应围绕"欠规范检测—交互式消减—富文本表达"三阶段来处理。

## 方法详解

### 整体框架

论文不是提出具体算法，而是构建了一个三层论证：
1. **破**（Sec 2）：系统梳理 aleatoric/epistemic 定义冲突，证明在 LLM agent 中不适用
2. **立**（Sec 3）：提出三个新研究方向
3. **反**（Sec 4）：公平讨论反方观点——传统 UQ 仍有价值的场景

### 关键设计

1. **Epistemic 不确定性的定义冲突**:

    - 核心论证：考虑一个 Bernoulli 分类问题，学习器认为只有 $\theta=0$ 或 $\theta=1$ 两种可能。(a) **分歧学派**（Houlsby et al., Gal et al.）用互信息 $I(y;\theta)$ 定义——两种信念最大分歧，epistemic 不确定性**最大**。(b) **可能模型数学派**（Wimmer et al.）用公理化定义——可能模型只剩两个，epistemic 不确定性**接近最小**。(c) **密度学派**（Mukhoti et al.）用训练数据密度定义——取决于 $x$ 离训练数据的距离，**答案不确定**。三种同样有理论基础的定义给出三种矛盾结论。
    - 论证意义：表明 epistemic uncertainty 并非一个普遍认同的概念，不同理论框架导致完全相反的结论。

2. **Aleatoric 不确定性的"可约的不可约性"**:

    - 核心论证：当模型类是线性的但数据生成过程是非线性时，最优线性模型仍有残余风险（模型偏差）。**贝叶斯最优学派**（Schweighofer et al.）认为这是不可约的→aleatoric。**数据不确定性学派**（Lahlou et al.）认为换更强模型类可以消除→不是 aleatoric。更关键的是，如果 epistemic = 总体 - aleatoric，那么 aleatoric 的边界直接决定 epistemic 的值——而这个边界是主观选择。
    - 论证意义：所谓"不可约"取决于你考虑的模型类边界在哪里。

3. **Underspecification Uncertainty（欠规范不确定性）**:

    - 功能：处理 LLM agent 独有的"任务/上下文不明确"问题
    - 核心思路：分为**任务欠规范**（用户意图不明，$P(y|x) = \int_{t \in \mathcal{T}} P(y|t) P(t|x) dt$，未知任务 $t$ 引入额外不确定性）和**上下文欠规范**（缺少关键信息，如"哈利波特电影什么时候上映"缺少国家信息——Natural Questions 中 56% 的问题存在此类歧义）。
    - 设计动机：这类不确定性既非传统意义的 aleatoric（可以通过追问消除），也非传统 epistemic（不是因为模型训练数据不够），而是推理时由用户输入不完整造成的——即使训练数据无限大、模型完美，仍然无法消除。

4. **Interactive Learning（交互式学习）**:

    - 功能：通过追问减少欠规范不确定性
    - 核心思路：LLM agent 可以主动提问来获取缺失信息，类似 active learning 但有两个关键区别：(a) 目的是解决当前问题而非改进全局模型；(b) 信息来源是用户而非数据库，涉及人机交互研究。需要在"问太多导致用户不耐烦"和"不问导致回答含糊"之间找到平衡。
    - 当前差距：即使 GPT-3.5-Turbo-16k，检测歧义问题的准确率仅 57%（50% 为随机），人类评估者认为只有 53% 的追问有帮助。

5. **Output Uncertainty（输出不确定性）**:

    - 功能：超越标量概率，用富文本表达不确定性
    - 核心思路：LLM 不应只输出"置信度 0.7"，而应列出可能的答案、解释不确定原因、说明哪些信息能消除不确定性。类似于将 conformal prediction 从"预测集合"扩展为"自然语言解释的可能性空间"。可用语言学手段（"most likely"、"perhaps"）甚至语音特征（语调犹豫）来传达不确定性。
    - 设计动机：用户面对数值概率容易盲目信任高置信度的错误输出（"blind trust" behavior），而文本化解释能提供更丰富的决策依据。

### 损失函数 / 训练策略

本文为 Position Paper，不涉及具体训练方法。公式 (1) 给出信息论分解 $\mathbb{H}(y) = \mathbb{E}_\theta[\mathbb{H}(y|\theta)] + \mathbb{I}(y;\theta)$ 作为文献综述的形式化框架，但论文质疑这一分解在实践中的可用性（Mucsányi et al. 2024 发现两个分量的秩相关高达 0.8-0.999）。

## 实验关键数据

### 文献综述定量证据

| 发现 | 来源 | 数据 |
|------|------|------|
| Aleatoric/epistemic 估计值高度相关 | Mucsányi et al. 2024 | 深度集成在 ImageNet-1k 上：秩相关 0.8-0.999 |
| Aleatoric 估计器可用于 OOD 检测（传统认为是 epistemic 任务） | Mucsányi et al. 2024 | 表现与 epistemic 估计器相当 |
| LLM 检测歧义问题能力极弱 | Zhang et al. 2024c | GPT-3.5-Turbo-16k 准确率仅 57%（随机 50%） |
| 追问质量差 | Zhang et al. 2024c | 人类评估者认为仅 53% 的追问有助于消歧 |
| "Aleatoric" 和 "epistemic" 在 arXiv 的使用量 | 论文统计 | 2024 年每天约 1 篇含这些词的预印本 |

### 反方证据总结

| 反方论点 | 作者回应 |
|---------|---------|
| Aleatoric/epistemic 仍有价值 | 同意——在训练阶段、active learning 中仍有用，但应明确定义 |
| Interactive learning = 标准 next-token prediction | 部分同意——在标准化交互中可行，但仍需验证追问是否反映真实内部知识 |
| 不确定性应为数值 | 当 LLM 与自动系统通信时确实需要数值，但人机交互中文本化表达更好 |

### 关键发现

- 传统 UQ 的 aleatoric/epistemic 二分法在社区内至少有 6 种相互矛盾的定义（Table 1），即使在最简单的 Bernoulli 例子中也给出矛盾结论
- 在 LLM agent 的多轮交互中，不确定性的"可约/不可约"性质随交互动态变化——Der Kiureghian & Ditlevsen (2009) 的结论是这种标签最终是建模者的主观选择
- 现有 LLM 对不确定性的内省能力极弱，但这恰恰说明这是亟需研究的方向

## 亮点与洞察

- 用一个极简的 Bernoulli 例子同时击穿 epistemic 和 aleatoric 的定义矛盾——这个论证简洁有力、难以反驳
- "可约的不可约性"概念精准捕捉了模型类选择对不确定性分类的影响——切中实践中常被忽略的要害
- 三个新方向（underspecification → interactive → output）构成一个完整的推理时不确定性处理流水线，而非零散建议

## 局限性

- 作为 Position Paper，不提供算法实现和实验验证——三个研究方向目前仅为概念性提案
- 论文主要从理论和哲学角度论证，对具体如何实现 interactive learning 或 output uncertainty 的技术细节涉及不深
- 反方讨论（Sec 4）较为简短，部分反方论点的回应不够深入
- 未讨论 chain-of-thought 推理中的不确定性传播问题
- 论文发表后已有 SelfReflect (Kirchhof et al., 2025) 等后续工作，论文本身对具体方向推进有限

## 相关工作与启发

- **vs Baan et al. (2023)**：他们在 NLP 中也提出 aleatoric/epistemic 二分法不够用，本文将这一观察扩展到更宏观的 LLM agent 交互场景
- **vs Der Kiureghian & Ditlevsen (2009)**：工程领域对 aleatoric/epistemic 主观性的经典讨论，本文将其洞察引入 ML 社区
- **vs Mucsányi et al. (2024, NeurIPS)**：提供了关键的实证证据——传统分解方法的两个分量高度相关，本文以此为核心论据之一
- **启示**：对于开发 LLM agent，不应追求分离 aleatoric/epistemic，而应关注"这个不确定性我现在能做什么"——能追问就追问，不能追问就解释清楚

## 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 三个新方向的提出及 LLM agent UQ 的统一框架有洞察力 |
| 技术深度 | ⭐⭐⭐ | 综述全面但缺乏形式化的新理论贡献 |
| 实验充分度 | ⭐⭐ | Position Paper 不要求实验，但引用的证据主要来自他人工作 |
| 写作质量 | ⭐⭐⭐⭐⭐ | 论证逻辑清晰、例子精炼、正反双方公平讨论 |
| 实用性 | ⭐⭐⭐⭐ | 为 LLM agent UQ 研究提供了有价值的路线图 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](../../ACL2025/dialogue/uniconv_retrieval_response_gen.md)
- [\[ACL 2025\] Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](../../ACL2025/dialogue/beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[ACL 2025\] Sparse Rewards Can Self-Train Dialogue Agents](../../ACL2025/dialogue/sparse_rewards_can_self-train_dialogue_agents.md)
- [\[ECCV 2024\] A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](../../ECCV2024/dialogue/a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)

</div>

<!-- RELATED:END -->
