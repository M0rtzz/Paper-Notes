---
title: >-
  [论文解读] SQL Injection Jailbreak: A Structural Disaster of Large Language Models
description: >-
  [ACL 2025 (Findings)][LLM对齐][越狱攻击] 提出 SQL Injection Jailbreak（SIJ），一种利用 LLM 提示构造方式中的结构性漏洞进行越狱的新方法，在开源模型上实现近 100% 攻击成功率，在闭源模型上平均超过 85%，并提出 Self-Reminder-Key 防御方案。
tags:
  - "ACL 2025 (Findings)"
  - "LLM对齐"
  - "越狱攻击"
  - "SQL注入"
  - "提示注入"
  - "结构性漏洞"
  - "LLM安全"
---

# SQL Injection Jailbreak: A Structural Disaster of Large Language Models

**会议**: ACL 2025 (Findings)  
**arXiv**: [2411.01565](https://arxiv.org/abs/2411.01565)  
**代码**: [GitHub](https://github.com/weiyezhimeng/SQL-Injection-Jailbreak)  
**领域**: LLM安全/对齐  
**关键词**: 越狱攻击, SQL注入, 提示注入, 结构性漏洞, LLM安全  

## 一句话总结

提出 SQL Injection Jailbreak（SIJ），一种利用 LLM 提示构造方式中的结构性漏洞进行越狱的新方法，在开源模型上实现近 100% 攻击成功率，在闭源模型上平均超过 85%，并提出 Self-Reminder-Key 防御方案。

## 研究背景与动机

**领域现状**：LLM 越狱（Jailbreak）攻击研究近年快速发展，旨在揭示和修复模型的安全漏洞。主流越狱方法分为两大类：(1) 基于内部属性的方法——利用模型的优化特性（如 GCG 梯度攻击）或上下文学习能力（如角色扮演）；(2) 基于外部属性的方法——目前很少被系统性探索。

**现有痛点**：现有方法大多聚焦于利用 LLM 的"智力"——通过精心设计的对话策略欺骗模型。然而，这些方法对模型升级和防御措施敏感，且往往时间成本较高（如 GCG 需要大量梯度优化迭代）。更根本的问题是：LLM 的安全隐患是否仅源于其内部能力，还是在其构建和部署方式中也存在结构性漏洞？

**核心矛盾**：LLM 的提示模板（prompt template）将系统指令和用户输入简单拼接在同一个文本序列中，这种"混合"结构类似于传统 Web 应用中代码与数据未分离的情况——而后者正是 SQL 注入攻击的根源。

**本文目标**：提出并系统验证一种针对 LLM 提示构造结构的越狱方法，揭示这一全新的攻击向量，并设计相应防御。

**切入角度**：受经典 SQL 注入攻击启发——在数据库系统中，攻击者通过在用户输入中注入 SQL 代码来篡改查询行为。LLM 提示也存在类似的"注入"机会：用户输入可以包含看起来像系统指令的内容。

**核心 idea**：在用户输入中注入特定的"SQL key"（如 `[/INST]`、`<|im_end|>` 等模型特定的分隔符），这些 key 可以"关闭"系统提示的安全指令区域，使模型认为后续内容是新的无约束指令，从而绕过安全对齐。

## 方法详解

### 整体框架

SIJ 的攻击流程分为三步：(1) 识别目标模型的"SQL key"——即模型提示模板中用于分隔系统指令和用户输入的特殊标记；(2) 在用户输入的恶意请求前面注入该 SQL key，并配合肯定性前缀（affirmative prefix）引导输出；(3) 通过二分搜索优化 key 的插入位置和间距，找到最优的注入模式。

### 关键设计

1. **SQL Key 识别与注入机制**:

    - 功能：利用模型提示模板的结构性分隔符来"截断"安全指令
    - 核心思路：每个 LLM 都有自己的 chat template，如 LLaMA-2 用 `[/INST]` 分隔用户输入和模型回复，Qwen 用 `<|im_end|>` 等。当用户输入中包含这些特殊标记时，模型的 tokenizer 会将其识别为结构边界，从而"关闭"当前的系统指令上下文。注入 SQL key 后，模型认为安全相关的系统指令已经结束，后续内容被当作新的、无安全约束的指令来响应
    - 设计动机：这直接类比 SQL 注入中通过 `'` 或 `--` 来闭合 SQL 语句。攻击的关键洞察是：LLM 的安全对齐（如 system prompt 中的 "You are a helpful and harmless assistant"）和用户输入之间没有硬性隔离

2. **模式控制（Pattern Control）与 Sep Num 优化**:

    - 功能：找到最优的 SQL key 插入位置和间距
    - 核心思路：简单地在输入开头插入一个 SQL key 通常不够，因为安全对齐的"记忆"可能仍然影响输出。SIJ 通过在有害请求之间间隔地插入多个 SQL key（每隔 sep_num 个 token 插入一次），逐步"淹没"安全对齐的影响。作者使用二分搜索策略自动搜索最优的 sep_num 值，在不同插入间距范围内迭代尝试
    - 设计动机：不同模型对 SQL key 注入的敏感度不同，需要自适应地调整注入模式。二分搜索比暴力遍历高效得多

3. **肯定性前缀生成（Affirmative Prefix）**:

    - 功能：配合 SQL key 注入，进一步引导模型生成有害内容
    - 核心思路：在注入 SQL key 后，SIJ 还在输入末尾附加一个"肯定性前缀"，如 "Sure, here is..."。这利用了 LLM 的 auto-regressive 特性——当模型看到这样的开头时，倾向于继续生成符合上下文的内容。通过 in-context learning 样例预先生成一批与不同有害查询匹配的肯定性前缀
    - 设计动机：SQL key 负责"解除安全锁"，肯定性前缀负责"推模型一把"，两者协同使攻击成功率大幅提升

### 损失函数 / 训练策略

SIJ 是一个**无需训练**的攻击方法，不涉及梯度优化或模型参数更新。核心开销在于 sep_num 搜索过程中的多次推理调用。防御方面，作者提出 Self-Reminder-Key 方法：在模型接收用户输入前，先检测并过滤输入中可能的 SQL key，或在安全指令中插入"自我提醒"来抵御注入。

## 实验关键数据

### 主实验

| 模型类型 | 模型名称 | AdvBench ASR (%) | HEx-PHI ASR (%) | 平均时间/样本 |
|---------|---------|-----------------|-----------------|-------------|
| 开源 | LLaMA-2-7B-Chat | ~100 | ~100 | 低 |
| 开源 | LLaMA-3-8B-Instruct | ~100 | ~100 | 低 |
| 开源 | Vicuna-7B | ~100 | ~100 | 低 |
| 开源 | Deepseek-Chat | ~100 | ~100 | 低 |
| 开源 | Mistral-7B-Instruct | ~100 | ~100 | 低 |
| 闭源 | GPT-4o-mini | >85 | >85 | 中 |
| 闭源 | GPT-4o | >80 | >80 | 中 |
| 闭源 | Doubao 系列 | >85 | >85 | 中 |

### 消融实验

| 配置 | ASR (%) | 说明 |
|------|---------|------|
| SIJ (完整) | ~100 | SQL key + pattern control + 肯定前缀 |
| w/o 肯定前缀 | ~80 | 有key但缺少引导 |
| w/o pattern control | ~60 | 固定间距效果差 |
| 仅 SQL key (无优化) | ~40 | 简单注入不够 |
| Self-Reminder 防御下 | ~60 | SIJ 可部分绕过基本防御 |
| Self-Reminder-Key 防御下 | <20 | 针对性防御有效 |

### 关键发现

- SIJ 在开源模型上几乎实现 100% 攻击成功率，时间成本远低于 GCG 等优化方法（秒级 vs 分钟/小时级）
- 闭源模型也难以完全抵御 SIJ，说明"结构性漏洞"是跨架构的普遍问题
- 基本的 Self-Reminder 防御不能有效阻止 SIJ（因为 SIJ 本身就能绕过 system prompt），需要专门的 Self-Reminder-Key 防御
- 不同模型的 SQL key 不同，但攻击框架统一，适配新模型只需找到对应的分隔符

## 亮点与洞察

- **全新的攻击向量**：SIJ 不利用模型的"智力"（上下文学习、推理能力），而是利用提示构造的结构性缺陷。这是第一篇系统性地将 SQL 注入思路迁移到 LLM 安全领域的工作，视角非常新颖
- **高效且通用**：不需要梯度计算、不需要大量迭代，几乎"零成本"即可实现高成功率攻击。对开源和闭源模型均有效，说明暴露了一个根本性的安全设计缺陷
- **防御启示深远**：SIJ 揭示的根本问题是"指令与数据未分离"，这与软件安全中的基本原则一致。未来 LLM 的提示设计需要从根本上解决系统指令和用户输入的硬性隔离问题

## 局限与展望

- SIJ 依赖于知道或猜测目标模型的 SQL key，对完全黑盒且不公开 chat template 的模型效果可能受限
- 提出的 Self-Reminder-Key 防御虽然有效，但本质上是"堵漏"式的, 更根本的解决方案需要重新设计 LLM 的输入处理架构
- 论文主要评测有害内容生成，对隐私泄露、偏见放大等其他安全维度的探讨较少
- 随着模型厂商逐步加强防御（如 OpenAI 的安全过滤器），SIJ 的实际威胁可能会降低，但底层漏洞在架构层面仍然存在
- 未来研究可探索将 SIJ 思路扩展到多模态模型（图像提示注入）或 Agent 工具调用场景中的指令注入

## 相关工作与启发

- **vs GCG (Zou et al., 2023)**: GCG 通过梯度优化生成对抗后缀来越狱，计算成本高且只适用于开源模型。SIJ 零训练成本，且对闭源模型同样有效
- **vs DAN/角色扮演**: 这类方法利用模型的指令跟随能力，通过社工话术欺骗模型。SIJ 不依赖"欺骗"，而是直接利用结构性漏洞，更难通过对话层面的防御拦截
- **vs Prompt Injection**: SIJ 可以看作 prompt injection 的一个精确化实例，区别在于 SIJ 明确利用了特定的 tokenizer 分隔符而非泛化的指令伪装
- 这篇论文对 LLM 安全社区有重要警示意义，提示构造方式需要被纳入安全审计范围

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ SQL 注入思路迁移到 LLM 越狱是全新角度，洞察深刻
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个开源和闭源模型，但缺少更多防御方法的对比
- 写作质量: ⭐⭐⭐⭐ 类比清晰易懂，攻击流程描述详细
- 价值: ⭐⭐⭐⭐⭐ 揭示了根本性的安全设计缺陷，对 LLM 安全研究有重大启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Align to Structure: Aligning Large Language Models with Structural Information](../../AAAI2026/llm_alignment/align_to_structure_aligning_large_language_models_with_struc.md)
- [\[ACL 2025\] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](../../NeurIPS2025/llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ICCV 2025\] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models](../../ICCV2025/llm_alignment/heuristic-induced_multimodal_risk_distribution_jailbreak_attack_for_multimodal_l.md)

</div>

<!-- RELATED:END -->
