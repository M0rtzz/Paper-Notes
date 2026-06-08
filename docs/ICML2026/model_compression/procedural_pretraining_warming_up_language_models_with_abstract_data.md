---
title: >-
  [论文解读] Procedural Pretraining: Warming Up Language Models with Abstract Data
description: >-
  [ICML 2026][模型压缩][程序化数据] 在标准语言/代码/数学预训练之前插入一段极轻量的"程序化数据"（形式语言、栈、元胞自动机等）"预热"，仅 0.1–0.3% 额外 token 就能稳定提升下游性能…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "程序化数据"
  - "预热预训练"
  - "算法技能"
  - "形式语言"
  - "数据效率"
---

# Procedural Pretraining: Warming Up Language Models with Abstract Data

**会议**: ICML 2026  
**arXiv**: [2601.21725](https://arxiv.org/abs/2601.21725)  
**代码**: 有  
**领域**: LLM 预训练 / 数据中心 AI  
**关键词**: 程序化数据, 预热预训练, 算法技能, 形式语言, 数据效率

## 一句话总结
在标准语言/代码/数学预训练之前插入一段极轻量的"程序化数据"（形式语言、栈、元胞自动机等）"预热"，仅 0.1–0.3% 额外 token 就能稳定提升下游性能，并让模型用 55–86% 的原始数据复现同等 loss——是一种把"推理脚手架"和"知识"解耦的预训练策略。

## 研究背景与动机

**领域现状**：当前 LLM 训练的事实标准是"一锅端"——直接在 web 规模语料上做下一个 token 预测，让模型同时学会语义知识和操作这些知识的技能。

**现有痛点**：知识和推理被高度纠缠地学进同一组权重，模型因而容易依赖表层启发式而非系统化的推理过程；多篇工作（Han 2025、Kumar 2025、Nikankin 2025）已指出这是当前模型的核心缺陷。

**核心矛盾**：用"语义数据"既学知识又学算法技能效率很低——语义里掺杂大量近似/含糊/捷径，难以稳定锤炼出精确的符号操作能力。理想情况应该像人类一样"先学逻辑和数学、再学高阶推理"，但目前没有一条工程化的路径把这两者拆开。

**本文目标**：把"先学结构、再学语义"的认知发展思路落地为可复制的预训练流水线，验证它在多种规模、多个语义域下的有效性，并解释机制层面到底是哪些组件在受益。

**切入角度**：作者从"程序化数据"出发——由形式语言、简单算法、元胞自动机等显式规则生成的非语义序列。与"由 LLM 生成的合成数据"不同，程序化数据完全没有语义内容，但具有严格的结构（嵌套、递归、长程依赖），相当于纯粹的"算法脚手架"。如果先在这些脚手架上预热，再喂给标准语料，模型也许能把"操作能力"和"知识"分阶段学好。

**核心 idea**：把 0.1–0.3% 的程序化数据**前置**到标准预训练之前，作为一个"预-预训练"阶段；这一阶段不教任何语义，只教模型怎么压栈、平衡括号、做序列变换。

## 方法详解

### 整体框架
方法把标准预训练拆成两阶段：阶段一从头训练一个 GPT-2 风格的 decoder-only transformer，喂的是 $T_1$ 个完全没有语义的"程序化 token"（括号串、栈操作、元胞自动机演化等）；阶段二紧接着用 $T_2$ 个语义 token（C4、CodeParrot 或 DeepMind-Math）继续训练同一组权重，全程标准 next-token 损失、不冻结任何层。Baseline 就是 $T_1=0$ 的纯标准预训练。围绕这条流水线作者设两种核心量法：**Additive** 固定 $T_2$、额外加一段 $T_1$，量"白送的 token 能涨多少"；**Substitutive** 缩小 $T_2$、补一小段 $T_1$，量"能省掉多少语义数据还达到同样 loss"。

### 关键设计

**1. 程序化数据生成器：用规则造出零语义、强结构的"算法脚手架"。**

整锅端式预训练的痛点是知识和算法技能纠缠在一起，模型学到的常是表层捷径而非精确的符号操作。作者的对策是先喂一批 $\le 128$ token 的短序列，每条都由显式规则生成、不含任何语义，只考验精确操作、组合性和长程依赖。这个数据家族有四类：**序列变换**（Set/Reverse/Identity/Union/Sort/Delete，给原序列让模型预测变换后的结果）、**记忆操作**（Stack，模拟 push/pop 后预测栈内容）、**形式语言**（$k$-Dyck 平衡括号及其非嵌套的 Shuffle 版本，$k$ 控制嵌套深度）、**元胞自动机**（ECA rule 110，二元序列按确定性 Markov 动力学演化、预测下一状态）。之所以坚持"程序化"而不用 LLM 合成数据，是因为它有可证的生成过程和清晰结构，能精确解耦"哪种结构教会了哪种技能"；而 $\le 128$ 的短序列保证算法信号不会被长上下文淹没。

**2. 诊断→迁移两段实验：先在小模型钉死因果，再到大模型验证迁移。**

要回答"程序化数据到底有没有用、为什么有用"，直接在大模型上消融既贵又容易被规模噪声盖住。作者于是先用 2 层 4 头的小 transformer 配 10 个 seed 当"诊断仪"：对每一对（程序化数据类型，算法任务）跑 additive 设置，任务覆盖 Haystack（长上下文检索）、Addition / Reversed Addition / Multiplication（算术）和 Sorting（排序），从而把"哪种结构教会哪种技能"的因果关系钉死。同一阶段还做 shuffled 控制——把程序化序列内部 token 顺序打乱、保持 token 分布不变，性能立刻塌回 baseline，证明真正起作用的是"结构"而非"token 统计"。诊断结论确立后再把规模拉到 1.3B 参数、10.5B token，在 C4 / CodeParrot / DeepMind-Math 上检验 additive 与 substitutive 两种设置的 loss 和下游表现，专门避开"小尺度有效、放大就失效"的陷阱。

**3. 选择性层迁移：用 attention/MLP 分别迁移当机制探针，定位知识存在哪。**

光知道程序化预热有用还不够，作者想知道收益到底存进了哪类参数。做法是在阶段二开始时只保留某一类层的预训练权重（attention-only 或 MLP-only），把另一类层重置为随机初始化，再跑标准训练，与 full-model 迁移对比。如果某类层单独迁移反而比全模型更强，就说明它是"有用结构"的真正载体、另一类层带来的是负迁移。结果呈现清晰的分工：自然语言任务（C4）上 MLP-only 更好，结构化代码（CodeParrot）上 attention-only 更好，语言与结构混合的 DeepMind-Math 上两者都重要。这套模块化探针既验证了"MLP 存知识、attention 存模式"的猜想，又给出实用工程指南——下游域偏哪种类型，就保留对应的层，避免无关层拖后腿。

### 损失函数 / 训练策略
两个阶段都用标准 next-token prediction 损失；对涉及输入/输出对的程序化数据，损失只在输出 token 上计算。阶段二在 Section 5、6 中把 token embedding 重置为随机值（因为程序化和语义词表无对应），在 Section 4（程序化→算法）里则把 embedding 设为均值向量（无语义偏移）。作者还做了两类反事实控制：(a) 显式加 attention sharpening 正则，无法复现程序化预训练的收益，排除"只是把注意力变锐"的简单解释；(b) 把预训练权重按层 shuffle、保留幅值分布破坏结构，性能立刻塌下来，排除"只是调整了初始化尺度"的解释。

## 实验关键数据

### 主实验

| 设置 | 数据 | 收益 |
|------|------|------|
| Haystack（context recall） | $k$-Dyck 预热 vs baseline | 准确率 10% → 98% |
| Additive，1.3B 模型 | +0.1–0.3% 程序化 token | C4 / CodeParrot / DeepMind-Math 一致改进 |
| Substitutive，达到同等 loss | C4 | 只需 55% 原始数据 |
| Substitutive，达到同等 loss | CodeParrot | 只需 67% 原始数据 |
| Substitutive，达到同等 loss | DeepMind-Math | 只需 86% 原始数据 |
| 下游任务 | 语言、代码生成、常识推理 | 程序化预热的收益持续存在 |

### 消融实验

| 配置 | 含义 | 结论 |
|------|------|------|
| Full（结构化程序化数据） | 完整方法 | 基线 |
| Shuffled 程序化数据 | 同 token 分布、破坏结构 | 收益塌回 baseline，证明结构是关键 |
| Attention sharpening 正则 | 显式锐化注意力替代预热 | 复现不出预热收益，否定"注意力变锐"的简单解释 |
| 权重幅值 shuffle | 保留每层权重幅值分布，打乱位置 | 性能急剧下降，否定"调整初始化尺度"的解释 |
| Attention-only 迁移 | 只保留 attention 层预训练权重 | Identity/Haystack 任务上比 full-model 多约 80 个百分点；结构化代码域整体更优 |
| MLP-only 迁移 | 只保留 MLP 层预训练权重 | Reversed Addition、自然语言（C4）任务上更优 |
| 多种程序化数据混合 / 权重融合 | 综合不同脚手架的好处 | 出现进一步收益，给后续工作打开方向 |

### 关键发现
- 不同程序化数据各自专精不同技能：$k$-Dyck 强化长上下文检索和排序，ECA rule 110 强化反向加法，Union / Delete 强化乘法——说明"算法结构"和"算法技能"之间存在可追踪的对应关系。
- 程序化预热的信息高度局部化：attention 层主要承载结构化能力（栈、括号、检索模式），MLP 层在自然语言中价值更大，恰好印证了 MLP 是知识容器的主流假说，但它能从"非语义"程序化数据中获益是反直觉的发现。
- 反向加法是少见的 MLP-only / full-model 优于 attention-only 的任务，说明该任务的瓶颈在数值处理而非模式匹配，给"算法 → 模块"映射提供了反例校验。

## 亮点与洞察
- "把 0.1% 的脚手架放到训练最前面"是一种近乎免费的改进——既能加性提升性能、又能替代性节省数据，对工业级预训练直接可用，且不需要改架构、不需要更多算力。
- "结构对应技能、技能对应组件（MLP/attention）"形成了一条清晰的因果链，这种从"诊断 → 迁移 → 机制 → 组合"的递进研究结构非常值得借鉴；后续凡是研究数据混入策略的工作都可以模仿这套方法论。
- 程序化数据可以脱离语言、跨模态——并发工作 Shinnick 2026 已经在视觉上看到类似收益，提示存在"模态无关的算法机制"，这是非常有想象力的方向。

## 局限与展望
- 模型尺度做到 1.3B，相对当下旗舰 LLM 仍偏小，10B+ 甚至 100B+ 规模下程序化预热是否仍然有效需要更多验证；学界已经反复看到"小规模有效、大规模失效"的反例。
- 程序化数据类型仍有限（栈、Dyck、ECA、序列变换），混合 / 融合策略只是初步探索；如何系统地"选数据 + 配比例 + 决定顺序"还没有方法论。
- 下游评估以语言建模 loss、代码生成、常识推理为主，没有覆盖更结构化的下游任务（如多步推理 benchmark、agent 行为），算法技能是否真的稳定迁移到复杂任务仍待检验。
- 现象上看到 attention 学到了"模式"、MLP 学到了"某种结构"，但没有给出机理级的解释（如电路分析、激活 patching），属于黑箱诊断而非白箱理解。

## 相关工作与启发
- **vs Hu et al. (2025) "pre-pretraining on formal languages"**: Hu 等人把形式语言看作"每 token 价值更高的预训练数据"，强调"替代"；本文把视角扩大到一组"程序化数据家族"，强调"补充"以及"模块化机制"。
- **vs Wu et al. (2022) / Zhang et al. (2024)**: 他们用简单算法或元胞自动机数据替换标准预训练，本文证明哪怕只是前置 0.1–0.3% 也足够受益、并系统比较不同算法的技能特化。
- **vs 代码预训练范式**: 代码长期被用作"提升推理能力"的隐性脚手架，本文相当于把这种直觉从"代码"推广到"任何带结构的非语义数据"，并解释了为什么它有效。
- **vs LLM 合成数据**: 合成数据靠 teacher LLM 生成，本质仍带语义；程序化数据从规则生成，零语义、强结构，更适合用作"技能脚手架"，两条路线互补而非替代。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"先学结构、再学语义"做成可复制的工程方案，并配合精细的诊断 + 机制实验，框架感强。
- 实验充分度: ⭐⭐⭐⭐ 跨模型规模（小诊断 + 1.3B）、跨数据域（语言/代码/数学）、跨设置（additive/substitutive）+ 强反事实控制，证据链完整。
- 写作质量: ⭐⭐⭐⭐⭐ "诊断 → 迁移 → 机制 → 组合"四段式叙事极其清晰，take-away 提炼到位。
- 价值: ⭐⭐⭐⭐ 工业级预训练几乎零成本即可采用，且为"知识-推理解耦"指出了具体可操作的方向。

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Alignment Tuning for Large Language Models: A Data-Centric Lens on Alignment Data Pipelines](../../ACL2026/model_compression/alignment_tuning_for_large_language_models_a_data-centric_lens_on_alignment_data.md)
- [\[ICML 2026\] IDLM: Inverse-distilled Diffusion Language Models](idlm_inverse-distilled_diffusion_language_models.md)
- [\[ICML 2026\] An Algebraic View of the Expressivity of Recurrent Language Models](an_algebraic_view_of_the_expressivity_of_recurrent_language_models.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ACL 2025\] Language Models Resist Alignment: Evidence From Data Compression](../../ACL2025/model_compression/language_models_resist_alignment.md)

</div>

<!-- RELATED:END -->
