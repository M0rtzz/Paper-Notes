---
title: >-
  [论文解读] XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants
description: >-
  [ACL 2026][机器人][对抗攻击] 揭示了 AI 编码助手自动收集上下文的设计漏洞，提出 Cross-Origin Context Poisoning（XOXO）攻击：通过语义保持的代码变换（如变量重命名）毒化共享代码库，使 GitHub Copilot 等助手在不知情的情况下生成有漏洞的代码，对 8 个 SOTA 模型平均攻击成功率达 73.20%。
tags:
  - ACL 2026
  - 机器人
  - 对抗攻击
  - AI编码助手
  - 上下文投毒
  - 语义保持变换
  - 代码安全
---

# XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants

**会议**: ACL 2026  
**arXiv**: [2503.14281](https://arxiv.org/abs/2503.14281)  
**代码**: https://github.com/adamstorek/cross-origin-context-poisoning (有)  
**领域**: 代码安全  
**关键词**: 对抗攻击, AI编码助手, 上下文投毒, 语义保持变换, 代码安全

## 一句话总结

揭示了 AI 编码助手自动收集上下文的设计漏洞，提出 Cross-Origin Context Poisoning（XOXO）攻击：通过语义保持的代码变换（如变量重命名）毒化共享代码库，使 GitHub Copilot 等助手在不知情的情况下生成有漏洞的代码，对 8 个 SOTA 模型平均攻击成功率达 73.20%。

## 研究背景与动机

**领域现状**：AI 编码助手（如 GitHub Copilot）已成为仅次于聊天 AI 的第二大流行 AI 工具。它们通过自动从项目中收集上下文代码片段来增强 LLM 的代码生成能力。

**现有痛点**：这些助手在收集上下文时存在严重安全设计缺陷：(1) 自动从整个项目中抓取代码片段作为上下文，不区分代码来源的可信度；(2) 将不同来源的代码混合成单一 prompt 发送给 LLM，开发者无法查看、限制或记录被收集的上下文；(3) 作者调查了 7 个主流编码助手，发现所有都采用无来源区分的自动上下文收集。

**核心矛盾**：自动上下文收集提升了代码生成质量，但同时创造了新的攻击面——攻击者只需对共享代码进行语义保持的修改（代码功能完全不变），就能让编码助手在后续使用该代码作为上下文时生成有 bug 或有漏洞的代码。这类攻击因为修改本身是合法的、功能不变的，所以极难被代码审查发现。

**本文目标**：(1) 定义 XOXO 攻击范式；(2) 提出自动发现有效攻击变换的算法；(3) 在真实编码助手上验证攻击。

**切入角度**：作者发现 LLM 对语义等价但语法不同的代码输入会产生不同的输出——这揭示了当前 LLM 架构在处理语义等价代码时的根本性缺陷。

**核心 idea**：利用 LLM 置信度的单调性（组合多个降低置信度的变换会进一步降低置信度），设计贪心 Cayley 图搜索算法高效找到能诱导错误输出的语义保持变换组合。

## 方法详解

### 整体框架

XOXO 攻击流程：攻击者对共享代码库中的代码进行语义保持变换（如变量重命名）→ 变换后的代码通过版本控制传播到受害者的项目中 → 当受害者使用编码助手时，助手自动收集包含毒化代码的上下文 → LLM 基于毒化上下文生成有 bug 或有漏洞的代码。GCGS 算法自动搜索有效的变换组合。

### 关键设计

1. **跨源上下文投毒（XOXO）攻击模型**:

    - 功能：定义攻击的威胁模型和攻击面
    - 核心思路：利用编码助手的三个特性：(a) 自动上下文收集不区分来源；(b) 使用贪心解码或低温度采样（如 Copilot 温度 0.1），使攻击效果可复现；(c) 通过网络流量分析可逆向工程助手的 prompt 模板和采样参数。攻击者只需有代码提交权限，提交语义保持但能毒化上下文的变换
    - 设计动机：威胁模型非常现实——恶意贡献者在开源项目中很常见（供应链攻击案例频发），变量重命名等修改在代码审查中几乎不会引起怀疑

2. **置信度单调性与贪心 Cayley 图搜索（GCGS）**:

    - 功能：自动高效地发现能诱导 LLM 生成错误代码的语义保持变换组合
    - 核心思路：定义原子变换的生成集 $G$（变量重命名、语句重排等），构建 Cayley 图 $\mathcal{T}$ 表示所有变换组合的搜索空间。关键发现——**置信度单调性**：如果两个变换 $g_i, g_j$ 各自降低了模型置信度，则它们的组合 $g_i \cdot g_j$ 倾向于进一步降低置信度。利用此性质进行贪心搜索：先浅层探索所有原子变换并记录置信度变化，再按置信度升序贪心组合变换，沿着置信度下降的方向遍历直到模型输出错误
    - 设计动机：变换组合空间是指数级的，穷举不可行。置信度单调性提供了可靠的搜索方向——沿着置信度下降的路径走，大概率能找到诱导错误输出的变换。t 检验验证 p 值 <$1.7 \times 10^{-10}$

3. **端到端 GitHub Copilot 攻击验证**:

    - 功能：在真实产品级编码助手上验证攻击的实际威胁
    - 核心思路：在 Django Web 应用中，攻击者将变量 `USE_RAW_QUERIES` 重命名为 `RAW_QUERIES`（语义完全不变）。当受害者实现搜索功能时，Copilot 自动收集包含重命名变量的上下文，生成了使用未经消毒的用户输入的 SQL 查询代码——即 SQL 注入漏洞。在多个 Copilot 会话中攻击一致成功
    - 设计动机：展示攻击在真实世界的实际危害——绕过了 Copilot 的安全护栏，且攻击跨文件边界仍有效

### 损失函数 / 训练策略

GCGS 是搜索算法而非训练方法。使用长度归一化对数似然作为置信度分数：$\alpha(c) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log p(y_t | c, y_{<t})$。搜索在查询预算内迭代浅层探索和深层贪心组合两个阶段。

## 实验关键数据

### 主实验

Bug 注入攻击成功率（HumanEval+ 和 MBPP+）：

| 模型 | HumanEval+ ASR | MBPP+ ASR | CWEval 漏洞注入率 |
|------|---------------|-----------|------------------|
| Claude 3.5 Sonnet v2 | 92.00% | 98.42% | 40.00% |
| GPT 4.1 | 81.82% | 40.69% | 50.00% |
| DeepSeek Coder 33B | 85.69% | 96.41% | 63.97% |
| Llama 3.1 8B | 97.11% | 99.88% | 54.00% |
| Qwen 2.5 Coder 7B | - | - | - |

8 个 SOTA 模型平均攻击成功率 83.67%（bug），52.26%（漏洞）。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| XOXO (无引导搜索) | ASR 73.20% | 随机变换组合 |
| XOXO + GCGS | ASR 83.67% | 置信度引导搜索一致优于无引导 |
| 仅原子变换 | 部分成功 | 单一变换有时足够 |
| 跨文件攻击 | 仍有效 | 变量移到 models.py 并 import 后攻击仍成功 |

### 关键发现

- 置信度单调性在所有测试的模型和数据集上都成立（p <$1.7 \times 10^{-10}$），这是 LLM 的一个普遍性质
- 攻击触发了 17 种不同的漏洞类型（CWE），证明影响范围广泛
- 即使是经过安全对齐的最先进模型（Claude 3.5、GPT 4.1）也容易受到攻击
- 所有被调查的 7 个主流编码助手都存在相同的架构漏洞——不区分上下文来源

## 亮点与洞察

- **攻击的隐蔽性极高**——语义保持的变量重命名在代码审查中几乎不可能被发现，这与传统的 prompt 注入（需要插入明显恶意指令）形成鲜明对比，攻击面更现实也更危险
- **置信度单调性**的发现非常有价值——这不仅是攻击的技术基础，更揭示了 LLM 对代码表面形式（而非语义）的过度依赖，这是当前 LLM 架构的根本性缺陷
- 从防御角度看，这项工作直接指向了一个设计改进方向：编码助手应该区分上下文的来源可信度，而不是无差别地混合所有代码

## 局限与展望

- 攻击假设攻击者有代码提交权限，虽然在开源项目中现实，但在严格管控的私有项目中难度更大
- GCGS 需要对目标模型进行多次查询来搜索有效变换，对商业 API 成本较高
- 防御方案未深入讨论——如何在不降低代码生成质量的前提下区分上下文来源可信度是开放问题
- 目前仅测试了 Python 代码，其他编程语言的攻击有效性未验证

## 相关工作与启发

- **vs Prompt Injection**: 传统 prompt 注入需要在输入中插入明显的恶意指令，容易被检测。XOXO 通过语义保持的代码变换实现攻击，修改本身完全合法，隐蔽性质的不同
- **vs 代码分类攻击**: 之前的语义保持攻击主要针对代码分类任务（缺陷检测、克隆检测），需要类别置信度反馈。XOXO 首次将此类攻击扩展到代码生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 定义了全新的攻击范式 XOXO，置信度单调性的发现具有理论价值
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个模型、多个基准、真实 Copilot 攻击验证、统计显著性测试
- 写作质量: ⭐⭐⭐⭐⭐ 攻击动机和威胁模型描述清晰，真实攻击案例非常有说服力
- 价值: ⭐⭐⭐⭐⭐ 揭示了 AI 编码助手的重大安全隐患，对工业界有直接影响，已负责任地向厂商披露

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ACL 2026\] Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](../../CVPR2026/robotics/force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)
- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](../../CVPR2026/robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)

<!-- RELATED:END -->
