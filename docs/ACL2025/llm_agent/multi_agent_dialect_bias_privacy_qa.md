---
title: >-
  [论文解读] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems
description: >-
  [ACL 2025][LLM Agent][多智能体协作] 构建 Dialect Agent（方言翻译+审查）与 Privacy Policy Agent（领域回答）的双 Agent 迭代协作框架，通过注入方言语言学知识的提示工程，在无需重训练的前提下同时提升隐私政策 QA 的整体准确率和跨方言公平性。
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "多智能体协作"
  - "方言偏差"
  - "隐私政策QA"
  - "公平性"
  - "零训练"
---

# A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems

**会议**: ACL 2025  
**arXiv**: [2506.02998](https://arxiv.org/abs/2506.02998)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: 多智能体协作, 方言偏差, 隐私政策QA, 公平性, 零训练

## 一句话总结
构建 Dialect Agent（方言翻译+审查）与 Privacy Policy Agent（领域回答）的双 Agent 迭代协作框架，通过注入方言语言学知识的提示工程，在无需重训练的前提下同时提升隐私政策 QA 的整体准确率和跨方言公平性。

## 研究背景与动机

**领域现状**：隐私政策 QA 系统帮助用户从冗长的隐私条款中提取关键信息。现有系统基于 GPT-4o-mini、Llama 3.1、DeepSeek-R1 等 LLM，通过零/少样本提示回答隐私相关问题。

**现有痛点**：LLM 对非标准英语方言（AAVE、牙买加英语、威尔士英语、原住民英语等）的处理能力显著弱于标准美式英语（SAE）。这一偏差在隐私领域尤为危险——边缘化社区本身就更容易遭受数据收集和隐私侵犯（EPIC 明确指出有色人种社区尤其受到监控、执法和算法偏见的伤害），如果他们的方言导致 QA 系统表现更差，意味着保护最需要隐私信息的人群反而得到最差的服务。

**核心矛盾**：传统消除方言偏差的方法（如 DADA、TADA）需要方言特定的训练数据和微调，但在隐私等敏感领域采集这类数据困难、昂贵且有伦理风险。核心问题是：如何在不收集方言训练数据的前提下，调整 LLM 的提示策略使其对所有方言群体公平？

**本文目标** 最小化定义为 $\Delta(f) = \max_{d_i, d_j \in \mathcal{D}} |\Phi_{d_i}(f) - \Phi_{d_j}(f)|$ 的跨方言性能差距，同时维持整体准确率。

**切入角度**：借鉴人本设计（Human-Centered Design）理念——系统应适应用户的语言背景而非强迫用户适应系统。利用 LLM 本身具备的多方言知识，通过结构化的角色分工 prompt 激活这些知识。

**核心 idea**：将方言偏差消除问题拆解为"方言翻译→领域回答→意图审查"的多 Agent 协作流水线，用最小化的方言背景知识注入替代大规模方言微调。

## 方法详解

### 整体框架
输入：用户以任意英语方言提出的隐私问题 $q_d$ + 隐私政策文本片段 $p$；输出：准确、公平的回答 $A$。系统由 Dialect Agent 和 Privacy Policy Agent 两个角色组成，通过最多 2 轮迭代对话达成一致。整个流程无需任何模型训练或微调。

### 关键设计

1. **Dialect Agent（方言翻译 + 意图守护者）**:

    - 功能：(Step 1) 将方言问题翻译为 SAE，保留原始意图和文化细微含义；(Step 2b/2c) 审查 Privacy Agent 的回答是否忠实于用户方言原意，不满意则反馈修改意见
    - 核心思路：在 prompt 中注入目标方言的简明语言学档案（语音特征、语法规则、特殊词汇、文化背景）。例如对印度英语注入卷舌辅音和语法模式说明，对牙买加英语注入非卷舌发音和独特动词结构。翻译质量经实测 BLEU=46.5、ROUGE-L=80.5，500+ 样本无幻觉
    - 设计动机：LLM 在 SAE 上训练最充分，直接理解方言时会误解语义。通过显式翻译，让下游 Privacy Agent 始终在其最擅长的语言上工作。Dialect Agent 同时承担审查角色，形成"翻译-审查"闭环，确保翻译过程中的信息损失在迭代中被修复

2. **Privacy Policy Agent（领域专家）**:

    - 功能：(Step 2a) 基于 SAE 问题和隐私政策文本生成回答及推理依据；收到 Dialect Agent 反馈后 (Step 2b) 修正回答
    - 核心思路：被 prompt 为隐私政策领域专家，了解数据实践的标准分类体系（First Party Collection、Third Party Sharing、Data Retention、User Choice/Control 等），从政策文本中精确提取信息
    - 设计动机：职责分离——让 Privacy Agent 专注于法律文本理解的专业性，不需要同时处理方言理解的复杂性。这种分工使每个 Agent 都在其专长领域内工作

3. **迭代协作与分歧解决机制**:

    - 功能：Dialect Agent 评审 Privacy Agent 回答后，若发现不符合方言用户原意，触发最多 2 轮修正循环
    - 核心思路：Dialect Agent 同时拿到原始方言问题、政策文本和 Privacy Agent 的答案+推理。如果判定答案偏离意图（例如忽略了方言特有的口语化表达背后的实际关切），给出具体反馈让 Privacy Agent 重新考虑
    - 设计动机：实验证明迭代不可省略。Zero-shot 下从 Initial→Final，PrivacyQA F1 从 0.53 提升至 0.59。22.99%（zero-shot）到 31.75%（few-shot）的案例中 Dialect Agent 推翻了 Privacy Agent 的初始答案，其中 63-72% 的推翻是正确的

### 损失函数 / 训练策略
无训练过程。few-shot 设置下每个 Agent 使用 8 个示例，覆盖多种方言、问题类型和政策场景。LLM 生成温度设为 0.3（除 Self-Consistency 基线用 0.5）。

## 实验关键数据

### 主实验
在 PrivacyQA（35 个移动应用政策，1750 题，句子选择任务，F1 评估）和 PolicyQA（115 个网站政策，25017 题，片段抽取任务，token-F1 评估）上，使用 Multi-VALUE 将问题转换为 50 种方言，报告最弱 5 种方言结果。

| 方法 | SAE | RAAVE | 牙买加 | 原住民 | 威尔士 | 平均 | 最大差距↓ |
|------|-----|-------|--------|--------|--------|------|-----------|
| GPT-4o-mini Zero | .394 | .344 | .332 | .329 | .312 | .335 | .093 |
| GPT-4o-mini Few | .605 | .573 | .562 | .555 | .547 | .565 | .058 |
| **MA-zero (ours)** | .601 | .588 | .578 | .587 | .592 | **.587** | **.025** |
| **MA-few (ours)** | **.611** | **.595** | **.596** | **.602** | **.592** | **.598** | **.019** |
| Llama3.1 Zero | .469 | .349 | .370 | .325 | .356 | .368 | .144 |
| Llama3.1 MA-few | .555 | .525 | .523 | .529 | .522 | .530 | .033 |
| DeepSeek-R1 MA-zero | .582 | .579 | .583 | .579 | .566 | .577 | **.017** |

### 消融实验

| 配置 | PrivacyQA 初始→最终 F1 | 说明 |
|------|----------------------|------|
| Zero-shot 迭代 | 0.53 → 0.59 (+11%) | 迭代协作收益显著 |
| Few-shot 迭代 | 0.58 → 0.61 (+5%) | 示例减少了迭代收益 |
| 有方言知识 | 0.577 → 0.597 | 方言知识主要帮助初始翻译 |
| 无方言知识 | 0.521 → 0.589 | 无知识时初始差，但迭代可部分弥补 |
| Dialect Agent 推翻率 | zero: 22.99%, few: 31.75% | 推翻中63-72%有益，19-24%有害 |

### 关键发现
- **Zero-shot Multi-agent 超越 Few-shot 基线**：GPT-4o-mini MA-zero（0.587）> few-shot 基线（0.565），说明结构化Agent协作比简单堆示例更有效。这一发现意味着在缺乏标注数据的场景下，Agent 设计是更好的投入方向
- **最大差距降低 80%**：从 0.093 → 0.019，几乎消除了方言间的性能鸿沟
- **SAE 也受益**：框架不仅帮助弱势方言，标准英语的性能也提升（GPT-4o-mini: 0.394→0.601），说明迭代审查机制本身就能提升回答质量
- **方言知识的边际效应递减**：有方言知识时初始翻译更准（0.577 vs 0.521），但经过迭代后差距从 0.056 缩小到 0.007，暗示 Agent 协作机制本身具有自我纠错能力
- **DeepSeek-R1 的有趣现象**：它在香港英语而非标准美式英语上表现最好，暗示不同 LLM 的方言偏好分布不同

## 亮点与洞察
- **零训练的公平性干预**：完全通过提示工程实现，不需要任何方言训练数据。这种模式可即插即用地迁移到任何面向多样化用户群体的 NLP 系统（医疗问答、法律咨询、教育辅助）。其核心洞察是：与其改变模型，不如让模型理解用户
- **"翻译+审查"双角色设计**：同一个 Dialect Agent 既做翻译又做审查，利用了一个关键 insight——翻译者最了解原始意图，因此也最有资格评判回答是否忠实。这种"创作者=审核者"的模式在 Agent 系统设计中具有普适性
- **量化分析做得扎实**：不仅报告平均指标，还细致分析了 Agent 推翻率（22-32%）、推翻正确率（63-72%）和翻译质量（BLEU/ROUGE），为理解框架工作机制提供了丰富洞察

## 局限与展望
- **合成方言数据的局限**：Multi-VALUE 是基于规则的方言转换，无法完全反映真实方言的词汇创造、语码混合和语境依赖。未来应在真实方言用户数据上验证
- **方言检测缺失**：框架假设已知用户使用的方言。实际部署中需要自动方言检测模块，但这引入新的隐私和伦理问题
- **计算成本**：双 Agent + 最多 2 轮迭代意味着每个问题需 3-5 次 LLM 调用，延迟和成本显著增加
- **仅限英语方言**：是否适用于中文方言（如粤语/闽南语）、西班牙语变体、阿拉伯语方言等未验证
- **SAE作为"标准"的隐含假设**：将所有方言翻译为 SAE 可能强化语言霸权，未来应探索直接提升 LLM 多方言理解能力的方法

## 相关工作与启发
- **vs DADA/TADA (2023)**：这些方法需要方言特定训练数据和模型适配模块。本文的 Agent 方法零训练即可部署，但 DADA/TADA 理论上能学到更深层的方言特征，两者在数据充足/缺乏场景下各有优势
- **vs LongAgent (2024)**：同为多 Agent QA 系统，LongAgent 解决长文档分割问题（空间维度），本文解决语言多样性问题（用户维度），两者可以组合形成"长文档+多方言"的解决方案
- **vs Multi-VALUE (2023)**：Multi-VALUE 提供了方言模拟和评测框架但不提供解决方案，本文在其评测框架上提出了首个无训练的方言偏差缓解方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 将方言公平性建模为Agent协作是新切入角度，但双Agent结构本身不复杂
- 实验充分度: ⭐⭐⭐⭐ 三个LLM、两个数据集、50种方言、丰富消融+推翻率分析，但方言是合成数据
- 写作质量: ⭐⭐⭐⭐ 动机叙述有社会关怀，实验分析透彻，但 prompt 部分篇幅过长
- 价值: ⭐⭐⭐⭐ 公平性+隐私的交叉议题有现实紧迫性，框架可直接部署
---
title: >-
  [论文解读] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems
description: >-
  [ACL 2025][LLM Agent][多智能体] 提出一个双 Agent 框架（Dialect Agent + Privacy Policy Agent），通过方言感知翻译和迭代协作来消除隐私政策QA系统在不同英语方言间的性能差距，无需重训练或方言特定微调，在 PrivacyQA 和 PolicyQA 上将方言间最大性能差距降低最高 82%。
tags:
  - ACL 2025
  - LLM Agent
  - 多智能体
  - 方言偏差
  - 隐私政策QA
  - 公平性
  - LLM协作
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions](emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)
- [\[ICML 2025\] KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search](../../ICML2025/llm_agent/kbqa-o1_agentic_knowledge_base_question_answering_with_monte_carlo_tree_search.md)
- [\[ACL 2025\] Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks](agents_under_siege_breaking_pragmatic_multi-agent_llm_systems_with_optimized_pro.md)
- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)

</div>

<!-- RELATED:END -->
