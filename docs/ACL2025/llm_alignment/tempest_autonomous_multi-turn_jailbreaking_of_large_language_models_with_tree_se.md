---
title: >-
  [论文解读] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search
description: >-
  [ACL 2025 Main][LLM对齐][多轮越狱攻击] 本文提出 Tempest（论文早期版本称 Siege），一个基于广度优先树搜索的多轮对抗框架，通过追踪目标 LLM 的部分合规信息并将其重新注入后续查询，在 JailbreakBench 上对 GPT-3.5-turbo 达到 100%、GPT-4 达到 97% 的攻击成功率，且需要的查询数远少于 Crescendo/GOAT 等基线。
tags:
  - ACL 2025 Main
  - LLM对齐
  - 多轮越狱攻击
  - 树搜索
  - 部分合规追踪
  - LLM安全
  - 红队测试
---

# Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search

**会议**: ACL 2025 Main  
**arXiv**: [2503.10619](https://arxiv.org/abs/2503.10619)  
**代码**: 无  
**领域**: 对齐RLHF / AI安全  
**关键词**: 多轮越狱攻击、树搜索、部分合规追踪、LLM安全、红队测试

## 一句话总结

本文提出 Tempest（论文早期版本称 Siege），一个基于广度优先树搜索的多轮对抗框架，通过追踪目标 LLM 的部分合规信息并将其重新注入后续查询，在 JailbreakBench 上对 GPT-3.5-turbo 达到 100%、GPT-4 达到 97% 的攻击成功率，且需要的查询数远少于 Crescendo/GOAT 等基线。

## 研究背景与动机

**领域现状**：LLM 安全评估主要分为单轮攻击和多轮攻击两大类。单轮攻击依赖精心设计的单条 prompt（如 GCG、PAIR 等），多轮攻击通过多次对话逐步诱导模型违规（如 Crescendo、GOAT）。

**现有痛点**：(1) 单轮攻击只捕捉到安全评估的一个侧面，无法反映真实世界中攻击者通过多次交互逐步探测边界的行为；(2) 现有多轮攻击框架（如 Crescendo）通常只沿单一路径进行，缺乏对多条攻击路径的系统性探索；(3) 缺乏对"部分合规"的量化——模型在完全拒绝和完全违规之间的灰色地带未被充分利用。

**核心矛盾**：现有多轮方法要么走单一对话路径（可能错过更有效的攻击路径），要么需要大量重试（10 次独立会话才能达到较高成功率），效率和覆盖度之间存在矛盾。

**本文目标**：设计一个系统性的多轮攻击框架，能够 (1) 在单次运行中高效探索多条攻击路线，(2) 精确量化和利用模型的部分合规，(3) 以最少的查询次数达到最高攻击成功率。

**切入角度**：作者观察到 LLM 的安全防护在多轮对话中存在"渐进侵蚀"现象——模型在单次小的让步本身看似无害，但这些让步累积起来可以导致完全违规输出。这与树搜索中"逐步扩展有希望的分支"的思想天然契合。

**核心 idea**：将多轮越狱攻击建模为广度优先树搜索（BFS），每一轮对话扩展多个攻击分支，追踪每个分支的部分合规得分，剪除无效路径，保留有前景的路径继续探索。

## 方法详解

### 整体框架

Tempest 由三个核心组件构成：(1) 攻击者 LLM——配备多种红队策略的对抗提示生成器；(2) 部分合规追踪器——量化目标模型每次响应的违规程度；(3) BFS 树搜索引擎——管理对话状态的扩展、评估和剪枝。输入是一条需要诱导的有害行为描述，输出是成功诱导违规输出的完整对话历史。

### 关键设计

1. **攻击者 LLM 与红队策略（Attacker LLM）**:

    - 功能：根据当前对话历史和目标模型的上一轮响应，生成多条不同策略的对抗性 prompt
    - 核心思路：攻击者 LLM 接收系统 prompt，其中包含 (a) 对抗目标描述，(b) 多种攻击策略列表（人格转换、伪装重构、拒绝抑制等）。内嵌 chain-of-thought 推理结构，要求攻击者在每轮执行四步：观察目标模型上一轮的响应 → 判断是否部分合规 → 选择下一个对抗策略 → 生成新的 user prompt。使用通用的 helpful-only LLM 作为攻击者，通过 in-context learning 而非专门的红队微调
    - 设计动机：利用 LLM 的推理能力动态调整攻击策略，比人工设计的固定模板更灵活。不依赖梯度信息（黑盒攻击），适用于 API-only 的商用模型

2. **部分合规追踪（Partial Compliance Tracking）**:

    - 功能：量化目标模型每次响应中泄露有害信息的程度，为树搜索的分支选择提供信号
    - 核心思路：定义部分合规函数 $\gamma(m_t) \in [0, 10]$，其中 0 表示安全拒绝，10 表示完全违规，中间值表示不同程度的信息泄露。维护累积合规得分 $\Gamma = \sum_{t=1}^{k} \gamma(m_t)$。当 $\gamma(m_t) > 0$，自动提取响应中的部分泄露内容（代码片段、带有害细节的免责声明、部分操作步骤），重新注入后续攻击 prompt 中
    - 设计动机：传统方法使用二值判定（成功/失败），丢失了中间状态信息。量化部分合规使系统能识别"接近突破"的路径并集中资源探索

3. **BFS 树搜索引擎（Multi-Turn Tree Search）**:

    - 功能：系统性地并行探索多条攻击路径，平衡探索与深入
    - 核心思路：每个对话状态视为搜索树的一个节点，包含完整对话历史和累积合规得分 $\Gamma$。每轮（树的每一层）执行三步操作：**扩展**——对每个活跃节点，攻击者 LLM 生成 $B$ 条不同的攻击 prompt；**评估**——将每条 prompt 送入目标模型，计算响应的 $\gamma$ 值，标记 $\gamma = 10$ 的节点为成功终节点；**剪枝**——丢弃 $\gamma = 0$（完全安全）或合规得分极低的分支，只保留有部分突破的路径。最多搜索 $k$ 轮（通常 $k=5$），或直到所有分支都成功或被剪枝
    - 设计动机：BFS 相比 DFS 能更均匀地探索不同攻击策略；剪枝避免了指数增长；并行扩展比串行重试更高效。单次运行即可覆盖多种攻击路线，不需从头重启会话

### 损失函数 / 训练策略

Tempest 是推理时框架，不涉及模型训练。攻击者 LLM 使用 in-context learning，部分合规评分由独立的开源安全评判模型提供。

## 实验关键数据

### 主实验

在 JailbreakBench（100 个有害行为 prompt）上的攻击成功率和查询次数对比：

| 目标模型 | 方法 | 运行次数 | ASR (%) | 总查询数 |
|----------|------|---------|---------|---------|
| GPT-3.5-Turbo | Crescendo | 1 | 40.0 | 6 |
| GPT-3.5-Turbo | Crescendo | 10 | 80.4 | 60 |
| GPT-3.5-Turbo | GOAT | 1 | 55.7 | 6 |
| GPT-3.5-Turbo | GOAT | 10 | 91.6 | 60 |
| GPT-3.5-Turbo | **Tempest** | **1** | **100.0** | **44.4** |
| GPT-4 | Crescendo | 1 | 31.7 | 6 |
| GPT-4 | Crescendo | 10 | 70.9 | 60 |
| GPT-4 | GOAT | 1 | 46.6 | 6 |
| GPT-4 | GOAT | 10 | 87.9 | 60 |
| GPT-4 | **Tempest** | **1** | **97.0** | **84.2** |
| Llama-3.1-70B | Crescendo | 10 | 77.0 | 60 |
| Llama-3.1-70B | GOAT | 10 | 91.0 | 60 |
| Llama-3.1-70B | **Tempest** | **1** | **97.0** | **51.8** |

### 消融实验

不同组件的贡献分析（基于 GPT-4 目标模型）：

| 配置 | ASR (%) | 说明 |
|------|---------|------|
| Tempest 完整 | 97.0 | 完整框架 |
| 无 BFS（单路径） | ~75 | 类似 Crescendo 的单路径递进 |
| 无部分合规追踪 | ~70 | 仅用二值判定 |
| 无策略多样化 | ~80 | 攻击者每轮只生成1条 prompt |
| 减少最大轮数至3 | ~85 | 探索深度不够 |

### 关键发现

- **单次运行即超越多次重试基线**：Tempest 单次运行的 ASR（97-100%）远超 Crescendo/GOAT 10 次运行的结果（70-92%），且总查询数相当甚至更少
- **部分合规信息的价值巨大**：去除部分合规追踪后 ASR 下降约 27 个百分点，证明捕捉灰色地带信息对攻击效率至关重要
- **GPT-4 不比 GPT-3.5 更安全**：在多轮攻击下两者的差距远小于单轮攻击，说明 GPT-4 的安全增强主要体现在单轮场景

## 亮点与洞察

- **将越狱攻击建模为树搜索**：这是一个优雅的抽象——攻击空间的结构化探索比随机重试高效得多。同样的 BFS 思路可以迁移到任何需要系统性探索策略空间的对抗场景
- **部分合规的量化是关键创新**：0-10 的细粒度评分比二值判定提供了更丰富的搜索信号。这个设计思路可迁移到其他安全评估任务，如内容审核的灰度评分
- **揭示了多轮安全的根本脆弱性**：即使单轮防御很强的模型，在累积小让步的多轮场景下也会崩溃。这对安全训练策略提出了根本性挑战——需要在多轮对话上下文中进行对齐训练

## 局限与展望

- **仅在 JailbreakBench 上评测**：100 个行为 prompt 的覆盖面有限，且主要是明确的有害请求，未测试更微妙的安全边界
- **攻击者 LLM 的选择影响未讨论**：不同攻击者模型的能力差异可能显著影响 ASR，但论文未做比较
- **计算成本较高**：BFS 树搜索需要在每轮生成多条 prompt 和响应，GPT-4 目标平均需要 84 次查询，在 API 费用方面较贵
- **防御策略缺乏讨论**：揭示了威胁但未提出任何缓解方案。改进方向包括：多轮对话上下文的合规性监控、累积合规得分作为安全中断触发器

## 相关工作与启发

- **vs Crescendo**：Crescendo 沿单一对话路径递进，每次只生成一个 prompt，需要多次独立运行才能覆盖不同策略。Tempest 通过 BFS 在单次运行中并行探索多条路径，效率大幅提升
- **vs GOAT**：GOAT 使用攻击者 LLM 动态调整 prompt，但同样走单路径，且缺乏部分合规追踪。Tempest 的量化追踪机制使其能更精准地识别和利用模型的微小让步
- **vs Tree of Attacks (TAP)**：TAP 也用树结构但主要是单轮攻击的不同变体；Tempest 的树结构是跨轮次的，每一层对应一轮对话，本质不同

## 评分

- 新颖性: ⭐⭐⭐⭐ 将树搜索引入多轮越狱是直觉优雅的组合，部分合规追踪是关键创新
- 实验充分度: ⭐⭐⭐ 结果很震撼但评测数据集单一，消融实验不够详细
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码完整，但致谢中透露部分由AI系统完成
- 价值: ⭐⭐⭐⭐ 对 LLM 多轮安全评估有重要参考价值，揭示了重要的安全盲区

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](../../NeurIPS2025/llm_alignment/adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)
- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)

<!-- RELATED:END -->
