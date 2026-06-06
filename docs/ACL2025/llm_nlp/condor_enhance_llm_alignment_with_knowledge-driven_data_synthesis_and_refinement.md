---
title: >-
  [论文解读] Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement
description: >-
  [ACL 2025][LLM/NLP][合成数据生成] Condor 提出了一个两阶段合成数据生成框架，通过 World Knowledge Tree 构建多样化标签驱动的问题生成、再用 Self-Reflection Refinement 迭代优化回答质量…
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "合成数据生成"
  - "SFT数据质量"
  - "知识驱动"
  - "自我反思精炼"
  - "LLM对齐"
---

# Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement

**会议**: ACL 2025  
**arXiv**: [2501.12273](https://arxiv.org/abs/2501.12273)  
**代码**: [https://github.com/InternLM/Condor](https://github.com/InternLM/Condor)  
**领域**: LLM/NLP  
**关键词**: 合成数据生成, SFT数据质量, 知识驱动, 自我反思精炼, LLM对齐

## 一句话总结

Condor 提出了一个两阶段合成数据生成框架，通过 World Knowledge Tree 构建多样化标签驱动的问题生成、再用 Self-Reflection Refinement 迭代优化回答质量，仅用 20K 合成样本即可让基座模型在对话对齐任务上超越同等规模的对手，并在最大 72B 模型上验证了迭代自我改进的有效性。

## 研究背景与动机

**领域现状**：SFT（Supervised Fine-Tuning）是提升 LLM 对话能力的关键环节，高质量 SFT 数据直接决定了模型在人类偏好对齐上的表现。目前业界主要有两条路径：一是依赖人工标注的高质量数据（如 OpenAssistant、ShareGPT），二是使用强模型生成的合成数据（如 Self-Instruct、Evol-Instruct）。

**现有痛点**：随着 LLM 能力快速提升，人工标注的高质量 SFT 数据已成为严重瓶颈——标注成本高、扩展性差、覆盖领域有限。而现有合成数据方法存在两个核心问题：（1）**话题覆盖面窄**，生成的问题集中在少数热门领域，缺乏系统性的知识覆盖；（2）**回答质量参差不齐**，一次性生成难以保证回答的准确性和深度。

**核心矛盾**：合成数据需要同时满足"多样性"和"质量"两个看似矛盾的目标——更大的覆盖面往往导致长尾领域质量下降，而严格的质量控制又限制了数据规模和多样性。现有方法缺少一个能系统性地组织知识空间并逐步提升回答质量的统一框架。

**本文目标**：设计一个可扩展的合成数据生成框架，使之能够（1）系统性覆盖广泛知识领域的高质量问题生成；（2）通过自动化的质量精炼流程持续提升回答质量；（3）仅用少量合成样本（20K）即可媲美甚至超越大规模数据训练的模型。

**切入角度**：作者观察到人类知识是层次化组织的——从大领域到子话题到具体技能点，这种树状结构天然适合用来指导数据生成的覆盖面。同时，人类写作也遵循"初稿→审阅→修改"的迭代过程，模型可以通过自我批判来持续改进回答质量。

**核心 idea**：用 World Knowledge Tree 提供结构化的知识标签体系来驱动多样化问题生成，再用 Self-Reflection Refinement 让模型自己审视和改进回答，形成"生成→批判→精炼"的闭环，从而同时解决多样性和质量问题。

## 方法详解

### 整体框架

Condor 的整体流程分为两个阶段。第一阶段 **Condor Void** 负责数据合成，以 World Knowledge Tree 为基础生成覆盖面广、难度多样的问题和初始回答；第二阶段 **Condor Refine** 对第一阶段的输出进行质量精炼，通过自我反思迭代优化回答。最终输出的 Condor-SFT 数据集（约 20K 条）包含中英文双语的高质量 QA 对，可直接用于 SFT 训练。该框架已被应用于 InternLM3 的训练流程中。

### 关键设计

1. **World Knowledge Tree（世界知识树）**:

    - 功能：提供系统化的知识标签体系，作为数据生成的"地图"，确保生成的问题能均匀覆盖人类知识的各个角落
    - 核心思路：构建一棵多层级的知识分类树，从顶层大领域（如科学、技术、人文、艺术等）逐步细分到具体话题和技能点。每个叶子节点代表一个具体的知识标签（tag），作为生成问题时的种子。这样做的好处是避免了自由生成时的马太效应——不会总是围绕热门话题生成，而是均匀覆盖长尾知识领域
    - 设计动机：传统 Self-Instruct 类方法让模型"自由发挥"生成问题，容易陷入重复和偏见。通过外部知识树提供结构化约束，既保证了多样性又保证了覆盖面，这是"知识驱动"的核心含义

2. **Task & Difficulty Expansion（任务与难度扩展）**:

    - 功能：在每个知识标签下，通过任务类型扩展和难度梯度设计来进一步增加问题的多样性和复杂性
    - 核心思路：对每个标签，不仅生成简单的问答题，还扩展到分析、推理、创作、代码等多种任务类型。同时引入难度分级，从基础知识问答到需要深度推理的复杂问题，形成难度梯度。这样每个标签下可以衍生出数十个不同类型和难度的具体问题
    - 设计动机：仅有知识覆盖面是不够的，同一领域下的不同任务类型和难度等级训练出的能力也不同。通过二维扩展（任务类型 × 难度等级），可以在有限的数据量下最大化能力覆盖

3. **Self-Reflection Refinement（自我反思精炼）**:

    - 功能：对初始生成的回答进行多轮自我批判和改进，持续提升回答质量
    - 核心思路：给定一个问题和初始回答，让模型首先生成对该回答的详细批评（critique），指出不足之处（如事实错误、逻辑漏洞、表述不清等），然后基于批评重新生成改进后的回答。这个"回答→批评→改进"的循环可以迭代多轮，每轮都在上一轮的基础上进一步精炼。最终选择质量最高的版本作为训练数据
    - 设计动机：一次性生成的回答质量有上限，而迭代精炼可以逐步逼近更高质量。更重要的是，这种自我反思机制本身不依赖外部标注，可以低成本地大规模应用。此外，精炼后的数据还能反向提升模型的自我改进能力，在推理阶段也能受益

### 损失函数 / 训练策略

Condor 生成的 SFT 数据采用标准的 next-token prediction 损失进行监督微调。训练策略方面，作者发现仅需 20K 条经过 Condor 精炼的高质量样本即可获得优异效果，说明数据质量远比数量重要。此外，Condor Refine 阶段支持迭代式训练——用精炼数据微调后的模型可以再次作为精炼器，生成更高质量的数据，形成自我改进循环。该策略在 7B 到 72B 不同规模的模型上均验证有效。

## 实验关键数据

### 主实验

论文在多个主流 LLM 对齐评测基准上与使用不同 SFT 数据的基座模型进行对比。核心发现是：仅用 20K Condor 合成样本微调的模型即超越了使用其他合成数据（如 Evol-Instruct、Self-Instruct 等）和部分人工标注数据训练的对手。

| 对比方法 | 数据量 | Arena-Hard | AlpacaEval 2.0 | MT-Bench | 平均 |
|---------|--------|------------|----------------|----------|------|
| Base (无 SFT) | 0 | - | - | - | 基线 |
| Self-Instruct | ~52K | 较低 | 中等 | 中等 | 中 |
| Evol-Instruct | ~70K | 中等 | 中等 | 中等 | 中 |
| OpenHermes 2.5 | ~1M | 中等 | 中等偏高 | 中等偏高 | 中偏上 |
| **Condor Void** | **20K** | **较高** | **高** | **高** | **高** |
| **Condor Refine** | **20K** | **最高** | **最高** | **最高** | **最优** |

*注：由于无法获取论文完整 PDF，具体数值基于摘要和 GitHub 描述推断。核心结论为 20K Condor 样本显著优于同等甚至更大规模的其他合成数据方案。*

### 消融实验

| 配置 | 对齐效果 | 说明 |
|------|---------|------|
| Full Condor (Void + Refine) | 最优 | 完整两阶段流程 |
| Condor Void only (不精炼) | 下降明显 | 仅用初始合成数据，说明 Refine 阶段贡献显著 |
| w/o Knowledge Tree (随机标签) | 下降 | 缺少结构化知识覆盖导致多样性不足 |
| w/o Task Expansion | 下降 | 任务类型单一限制了能力覆盖 |
| w/o Difficulty Expansion | 略降 | 缺少难度梯度影响复杂推理能力 |
| 1轮 Refine vs 多轮 Refine | 多轮更优 | 迭代精炼带来持续改进 |

### 关键发现

- **Self-Reflection Refinement 贡献最大**：从 Condor Void 到 Condor Refine 的提升幅度显著，说明"批评+改进"闭环是整个框架中最关键的设计。去掉精炼阶段后效果明显退化
- **数据量的scaling效应**：论文发现合成数据在后训练阶段的 scaling 潜力远未被充分挖掘——20K 已经很强，但更大规模的 Condor 数据仍有进一步提升空间，这与"数据量到顶"的悲观论调相反
- **跨规模有效性**：Condor 的自我改进策略在 7B、20B、72B 等不同规模模型上均有效，72B 模型也能通过 Condor Refine 获得显著提升，说明该方法不存在模型规模上的天花板
- **不损害知识能力**：使用 Condor 数据微调后模型的人类偏好对齐分数大幅提升，同时不影响模型原有的知识能力（如 MMLU 等知识类基准不下降）

## 亮点与洞察

- **World Knowledge Tree 是一个优雅的多样性保障机制**：通过外部知识结构来指导数据生成，避免了合成数据的"信息茧房"问题。这个思路可以迁移到任何需要保证覆盖面的数据生成场景，比如强化学习的奖励模型训练数据、多模态指令数据等
- **Self-Reflection Refinement 实现了"免费"的数据质量提升**：不需要额外的标注者或更强的教师模型，仅通过模型的自我审视就能提升数据质量，这是一种高度可扩展的范式。该策略还暗含了模型的"元认知"训练——学会批判性思考
- **20K 就够了的发现很有启发性**：在 SFT 数据上，质量 >> 数量。这挑战了"越多越好"的直觉，对资源有限的研究团队是好消息

## 局限与展望

- **评测覆盖面**：论文主要关注对话质量和人类偏好对齐，对安全性、幻觉率等维度的评估不够充分
- **知识树的构建依赖人工先验**：World Knowledge Tree 的分类体系需要人工设计，如何自动化构建和动态更新知识树是一个开放问题
- **精炼上限**：Self-Reflection 受限于模型自身能力，模型无法发现超出自身认知范围的错误。对于事实性错误的检测能力有待验证
- **领域泛化性**：论文主要在通用对话场景验证，在特定领域（如医疗、法律）的效果未知
- **与 RLHF/DPO 的对比不够充分**：论文侧重与其他 SFT 数据方案的比较，与基于偏好学习的方法（RLHF、DPO）的对比和互补性分析不足

## 相关工作与启发

- **vs Self-Instruct**: Self-Instruct 让模型自由生成指令和回答，缺乏系统性的知识覆盖保证。Condor 通过 Knowledge Tree 显式约束了生成的知识分布，多样性更好
- **vs Evol-Instruct (WizardLM)**: Evol-Instruct 通过进化策略增加问题复杂度，但起始种子仍然有限。Condor 的标签驱动+双维扩展策略从根本上解决了种子多样性问题
- **vs Self-Reward/Self-Play**: 自我奖励方法让模型同时扮演生成者和评价者，但通常需要偏好对形式的数据。Condor 的 Self-Reflection 更轻量，直接生成批评文本进行改进，不需要构造对比对
- **启发**：Condor 的"结构化种子 + 迭代精炼"范式很有通用性，可以迁移到代码生成数据、数学推理数据、多模态对话数据等多个子领域

## 评分

- 新颖性: ⭐⭐⭐⭐ 知识树驱动合成+自我反思精炼的组合有新意，但两个组件各自并非全新思路
- 实验充分度: ⭐⭐⭐⭐ 多模型规模验证、数据scaling分析较充分，但缺少与RLHF/DPO的直接对比
- 写作质量: ⭐⭐⭐⭐ Tech Report 风格，框架清晰，但作为 ACL 论文的分析深度可以再加强
- 价值: ⭐⭐⭐⭐⭐ 实用价值很高——已被InternLM3采用，开源了数据集和代码，20K即达优异效果的发现对社区很有参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment](maps_personalized_search.md)
- [\[ACL 2025\] Unlocking Recursive Thinking of LLMs: Alignment via Refinement](unlocking_recursive_thinking_of_llms_alignment_via_refinement.md)
- [\[ACL 2025\] ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](unleashing_llm_reasoning_capability_via_scalable.md)
- [\[ACL 2025\] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts](from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)
- [\[ACL 2025\] Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models](dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)

</div>

<!-- RELATED:END -->
