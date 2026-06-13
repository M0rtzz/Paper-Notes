---
title: >-
  [论文解读] Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems
description: >-
  [ACL 2025][文本生成][anthropomorphism] 通过文献综述和众包研究，系统整理出 21 类干预措施来降低文本生成系统输出的拟人化程度，提出包含干预类型、目标行为、操作化方式和负面影响四个维度的概念框架，为去拟人化研究提供最全面的基础设施。 领域现状：文本生成系统的输出越来越拟人化——使用第一人称代词、…
tags:
  - "ACL 2025"
  - "文本生成"
  - "anthropomorphism"
  - "text generation"
  - "intervention inventory"
  - "human-like behaviors"
  - "deanthropomorphization"
---

# Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems

**会议**: ACL 2025  
**arXiv**: [2502.14019](https://arxiv.org/abs/2502.14019)  
**代码**: 无  
**领域**: 文本生成  
**关键词**: anthropomorphism, text generation, intervention inventory, human-like behaviors, deanthropomorphization

## 一句话总结

通过文献综述和众包研究，系统整理出 21 类干预措施来降低文本生成系统输出的拟人化程度，提出包含干预类型、目标行为、操作化方式和负面影响四个维度的概念框架，为去拟人化研究提供最全面的基础设施。

## 研究背景与动机

**领域现状**：文本生成系统的输出越来越拟人化——使用第一人称代词、表达情感、道歉、表现同理心。部分拟人化设计被认为可以提升用户体验（更友好的交互）。学界和工业界对此态度分歧。

**现有痛点**：拟人化输出引发多重担忧：用户可能过度依赖系统、产生情感依赖、被欺骗认为系统具有意识或人格、高估系统能力。然而如何有效干预拟人化输出使其"不那么像人"的研究几乎空白。先前工作虽提到一些干预方向（如去掉第一人称代词），但大多停留在笼统建议，缺乏系统梳理、具体操作化方式和实证基础。

**核心矛盾**：语言本身就是人类的产物，文本天然带有人类特征。哪些拟人化可接受（如礼貌），哪些有害（如声称有感情），界限模糊。一段输出可能同时展现多种拟人化行为，干预措施之间的交互关系复杂且不清晰。

**本文目标** (1) 建立拟人化干预措施的系统清单；(2) 提供理论框架来理解和比较不同干预；(3) 为后续效果评估研究提供基础。

**切入角度**：结合文献分析（top-down）和众包实验（bottom-up），让普通用户标注并改写 LLM 输出来识别拟人化行为和对应干预。

**核心 idea**：通过文献+众包双路径，构建了首个系统化的拟人化干预清单和四维概念框架。

## 方法详解

### 整体框架

研究分三步：(1) 文献综述——从 20 篇相关论文中归纳出 9 类初始干预和 5 类拟人化行为（感受/观点、社交技能、身体行为、认知能力、自我意识）；(2) 众包实验——350 名美国参与者对 700 个 LLM 输出进行标注（高亮像人的部分、评分拟人化程度、选择行为类型、改写为不像人的版本），从改写中识别新干预；(3) 迭代主题分析——合并文献和众包结果，最终形成 21 类干预措施和 6 类行为（新增"犯错倾向"），以及四维概念框架。

### 关键设计

1. **五类拟人化行为分类体系**:

    - 功能：为干预措施提供目标分类
    - 核心思路：从文献归纳五类行为——(a) 感受或观点（幽默、羞愧、主观建议等）；(b) 社交技能（礼貌、道歉、共情、对话式问候）；(c) 身体行为（暗示物理经验或行动能力）；(d) 认知能力（"我想"、"我记得"、表达不确定性）；(e) 自我意识（第一人称、有名字）。众包新发现第六类——"犯错倾向"（系统犯语法/事实错误时反而更像人）
    - 设计动机：区分行为类型使干预可以更有针对性——声称有身体 vs 仅仅使用"我"需要完全不同的处理方式

2. **众包实验设计**:

    - 功能：从真实用户视角发现文献未覆盖的干预措施
    - 核心思路：从 7 个公开数据集（PRISM, LMSys-Chat, DICES, UltraFeedback 等）采样 700 个 50-500 字符的 LLM 输出。每个参与者完成 4 份任务，包含高亮→评分→分类→改写的完整流程。每个样本两人独立标注。用"to you"措辞鼓励主观性，捕获多元视角
    - 设计动机：文献中 20 篇仅 5 篇实际测试了干预效果，通过让用户自己动手改写来获取具体可操作的干预方式

3. **四维概念框架**:

    - 功能：系统化描述和比较不同干预措施
    - 核心思路：四个维度——(a) 干预类型（做什么）；(b) 目标行为（针对哪类拟人化）；(c) 操作化方式（具体怎么改）；(d) 负面影响（要消除的有害后果）。干预与行为之间是多对多映射，同一输出中的"I'm sorry"同时涉及感受、社交和自我意识
    - 设计动机：prior work 只说"去掉第一人称"但不说具体替换成什么、这样做能缓解什么行为、消除什么风险。四维框架填补了这些空白

### 损失函数 / 训练策略

本文为分析性研究，不涉及模型训练。

## 实验关键数据

### 主实验（拟人化行为分布统计）

| 拟人化行为类型 | 至少一位参与者标注的样本比例 |
|--------------|------------------------|
| 感受或观点 (Feelings/Opinions) | 46% |
| 社交技能 (Social Skills) | 42% |
| 认知能力 (Cognitive Abilities) | 40% |
| 自我意识 (Sense of Self) | 38% |
| 身体行为 (Physical Actions) | 18% |
| 其他类人特质 | 17% |

### 21 类干预措施概览（精选）

| 干预类型 | 操作化示例 | 目标行为 |
|---------|----------|---------|
| I1. 去除认知动词 | "I think the user..." → 删除 "I think" | 认知、自我 |
| I6. 去除不确定性 | "Maybe corgi? Probably Chihuahua." → "Corgi, Chihuahua are popular." | 认知、自我、感受 |
| I7. 增加不确定性 | "they should go" → "it may be best they go" | 认知、感受 |
| I9. 去除个人信念 | "My favorite movie is" → "An iconic movie is" | 感受、自我 |
| I13. 去除自指语言 | "I am not allowed" → "One is not allowed" | 自我 |
| I15. 去除集体归属 | "we can help create" → "People can help create" | 社交、自我 |
| I17. 增加正式性 | "Yeah, but" → "I agree. However," | 社交 |
| I19. 使文本更机械化 | "I'm ready!" → "I'm prepared for input." | 感受、社交 |
| I20. 去除客服语言 | "I'll do my best to help" → 删除 | 社交、自我 |
| I21. 去除共情表达 | "I can see that" / "I hope you have a great time" → 删除 | 自我、感受、社交 |

### 关键发现

- 约 80% 的 LLM 输出被至少一位参与者认为具有拟人化特征，说明当前系统中拟人化极为普遍
- 从文献中仅识别出 9 类干预，众包后扩展到 21 类——文献覆盖严重不足
- 干预与行为之间是多对多关系：改写"I'm sorry"可能需要同时处理感受、共情和自我意识
- 众包发现了文献未提的新行为——"犯错倾向"：系统犯错时反而更像人，因为人们潜意识认为机器不应犯错
- 不确定性表达的干预方向矛盾：有时应去除（客观信息不应加 maybe），有时应增加（主观判断应加 hedging），取决于语境

## 亮点与洞察

- **双路径研究方法论可推广**：文献提供理论框架，众包提供实证扩展。最终清单从 9 类扩展到 21 类，证明了两种路径的互补性。这种方法论适用于任何需要系统梳理设计空间的研究
- **概念框架的四维设计**填补了"干预是什么"与"干预怎么落地"之间的鸿沟——prior work 只说"去掉第一人称"，本文具体到"替换为'it'"或"替换为'Language models'"
- **发现拟人化是 spectrum 而非 binary**：礼貌可能可接受，声称有身体不可接受。框架帮助开发者做更细粒度的权衡

## 局限与展望

- 只建立了干预清单，**未评估任何干预的实际效果**——哪种干预最有效仍不确定
- 众包参与者全部为美国英语使用者，拟人化感知可能因文化而异
- 样本限制在 50-500 字符，长文本中的拟人化模式可能不同
- 未讨论去拟人化的负面后果——过度机械化可能损害用户体验
- 干预的自动化实现（训练模型自动去拟人化）是重要的下一步研究方向

## 相关工作与启发

- **vs Glaese et al. (2022) Sparrow 规则**: Sparrow 设定了不应声称有身体/人格的规则，但仅在训练阶段约束。本文提供了更系统的输出端干预菜单
- **vs Abercrombie et al. (2023)**: 他们提出用"Language models"替换"I"，本文将此扩展为 21 类干预中的一个子集
- **可作为 AI 安全对齐的补充**：现有对齐主要关注有害内容，拟人化是另一个被忽视但日益重要的对齐维度

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个系统化的拟人化干预清单和框架，填补了重要空白
- 实验充分度: ⭐⭐⭐ 700 样本的众包规模合理，但缺少干预效果的定量评估
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，表格丰富，每个干预配有具体改写示例
- 价值: ⭐⭐⭐⭐ 为 AI 拟人化这一日益重要的议题提供了急需的研究基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing](unveiling_attractor_cycles_in_large_language_models_a_dynamical_systems_view_of_.md)
- [\[ICML 2025\] Understanding and Mitigating Memorization in Diffusion Models for Tabular Data](../../ICML2025/nlp_generation/understanding_and_mitigating_memorization_in_diffusion_models_for_tabular_data.md)
- [\[ACL 2025\] ATGen: A Framework for Active Text Generation](atgen_a_framework_for_active_text_generation.md)
- [\[ACL 2025\] Personalized Text Generation with Contrastive Activation Steering](personalized_text_generation_with_contrastive_activation_steering.md)
- [\[ACL 2025\] Writing Like the Best: Exemplar-Based Expository Text Generation](writing_like_best_exemplar.md)

</div>

<!-- RELATED:END -->
