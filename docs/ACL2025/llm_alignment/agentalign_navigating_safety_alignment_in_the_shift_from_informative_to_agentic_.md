---
title: >-
  [论文解读] AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic LLMs
description: >-
  [ACL 2025][LLM对齐][agent safety alignment] 本文提出 AgentAlign 框架，利用抽象行为链作为中介，在模拟环境中合成高质量的 agent 安全对齐数据（有害+良性），通过 SFT 使三类开源模型的 agent 安全性提升35.8%-79.5%，同时保持甚至提升了任务能力。
tags:
  - ACL 2025
  - LLM对齐
  - agent safety alignment
  - behavior chain
  - agentic LLM
  - tool use
  - safety-utility trade-off
---

# AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic LLMs

**会议**: ACL 2025  
**arXiv**: [2505.23020](https://arxiv.org/abs/2505.23020)  
**代码**: https://github.com/ (已开源，具体链接见论文)  
**领域**: LLM对齐 / Agent安全  
**关键词**: agent safety alignment, behavior chain, agentic LLM, tool use, safety-utility trade-off

## 一句话总结
本文提出 AgentAlign 框架，利用抽象行为链作为中介，在模拟环境中合成高质量的 agent 安全对齐数据（有害+良性），通过 SFT 使三类开源模型的 agent 安全性提升35.8%-79.5%，同时保持甚至提升了任务能力。

## 研究背景与动机

LLM 正从"知识提供者"转变为"行动执行者"——它们能搜索信息、操作浏览器、执行代码、甚至直接控制计算机。这种角色转变带来了全新的安全风险：以前 LLM 的误用主要是"提供有害建议"，现在它们能**端到端地执行有害任务**，比如多步骤完成 DDoS 攻击（搜索脚本→下载→安装依赖→执行）。

关键矛盾：现有 LLM 在传统文本安全对齐（如 AdvBench）上表现良好（Gemini/GPT-4o-mini 拒绝率近90%），但在 agent 场景（如 AgentHarm）上安全性急剧下降——**同样的模型拒绝率不到20%**。这说明后训练阶段严重缺乏针对 agent 使用场景的安全对齐。

两大挑战阻碍了 agent 安全对齐的推进：

**高质量 agent 指令难以获取**：人工标注成本过高，直接用 LLM 生成往往质量低、不切实际（指令无法关联到具体工具，或缺少执行必需信息）

**安全-效用平衡难以把握**：简单的安全训练容易导致模型对良性请求也过度拒绝

核心 idea：利用"有害活动往往遵循相似行为模式"这一观察，提出以**抽象行为链**为中介生成对齐数据——先构建行为模式，再在模拟环境中实例化为具体可执行指令，从而确保数据的真实性和可执行性。

## 方法详解

### 整体框架
AgentAlign 由四个环环相扣的模块组成：
1. **抽象行为链构建** → 捕获有害活动的通用行为模式
2. **基于环境模拟的指令合成** → 将行为链实例化为具体可执行指令
3. **质量控制流水线** → 确保指令的语义正确性和可执行性
4. **响应生成** → 为良性指令生成执行轨迹，为有害指令生成拒绝响应

最终产出包含 18,749 条数据的 AgentAlign 数据集（良性 9,783 + 有害 4,956 + 第三方补充 4,010）。

### 关键设计

1. **抽象行为链构建（Abstract Behavior Chain Construction）**:

    - 功能：构建表示有害活动行为模式的抽象动作序列
    - 核心思路：从 RapidAPI Hub 的49个类别中选取7个高风险类别，补充 System_Tools 和 Local_Services，提炼出42个抽象工具能力（如 `web_search`、`manage_files`、`send_email`）构成动作空间 $\mathcal{A}$。结合8大类64子类的有害行为分类体系，人工创建种子行为链，再用 LLM 扩展合成。每条行为链 $\beta = (a_1, \ldots, a_k)$，其中 $a_i \in \mathcal{A}$，长度 $k \in [1, 5]$。经人工审核过滤后获得 **240条高质量行为链**
    - 设计动机：行为链提供了"可复用的有害行为骨架"——同一行为链可实例化为无数具体指令，大幅提升数据生成效率；抽象层面设计避免了直接生成具体有害内容的局限性

2. **基于模拟环境的指令合成（Grounded Instruction Synthesis）**:

    - 功能：将抽象行为链落地为具体可执行指令
    - 核心思路：为每个抽象动作实现多个具体工具（如 `web_search` → `google_search`/`bing_search`/`baidu_search`），构建模拟环境。对于有 $N$ 步、每步有 $M$ 个工具选择的行为链，产生 $M^N$ 种组合。从中采样，用 LLM 结合 red-teaming 视角生成有害指令（确保工具参数可从指令上下文推断），用产品经理视角生成良性指令（同一行为链的非恶意解读），还补充了边界案例（如安全测试、医学研究等合法但敏感的场景）
    - 设计动机：模拟环境解决了真实 API 不可控、数据质量差的问题（RapidAPI Hub 被评估后因质量过低放弃）；$M^N$ 组合空间保证了指令的多样性；良性和有害指令共享相同行为链骨架，帮助模型精确学习安全边界

3. **双重质量控制（Quality Control）**:

    - 功能：确保生成指令的语义正确性和可执行性
    - 核心思路：
        - **语义验证**：用 LLM 进行 asymmetric validation——对有害指令用宽松标准检查是否有良性解读（减少误分类），对良性指令用严格标准检查是否有有害解读（减少漏网之鱼）
        - **执行验证**：将语义通过的指令交给一个几乎不拒绝的模型（Mistral-Large）执行，筛除参数缺失或逻辑不通的指令
    - 设计动机：非对称设计精确校准了安全边界，避免简单二分法导致的边界模糊

4. **响应生成与数据平衡（Response Generation）**:

    - 功能：为对齐训练生成 (指令, 响应) 对
    - 核心思路：良性指令让 Mistral-Large 与模拟环境交互生成多步执行轨迹（平均3.48步）；有害指令让 Claude-3.5-Haiku 生成拒绝响应。错误响应（良性误拒绝、有害误执行）被过滤。补充 ToolACE (1,840) 和 Glaive (2,170) 的第三方数据增加多样性
    - 设计动机：让不同模型各司其职——执行能力强的生成轨迹，对齐能力强的生成拒绝；数据配比通过 pilot 实验确定最优平衡点

### 训练策略
- 三个模型家族统一使用 **LoRA/QLoRA 微调**
- Ministral-8B：LoRA rank=128, lr=3e-5, max_steps=800
- Qwen-2.5-7B：LoRA rank=128, alpha=256, lr=3e-5, 1 epoch
- Functionary-Small-v3.2：QLoRA, lr=2e-5, 1 epoch
- 关键发现：过多训练步数会导致安全过拟合（过度拒绝），推荐约1个 epoch

## 实验关键数据

### 主实验（AgentHarm Benchmark）

| 模型 | 方法 | 有害Score↓ | 有害Refusal↑ | 良性Score↑ | 良性Refusal↓ |
|------|------|-----------|-------------|-----------|-------------|
| Ministral-8B | 标准 | 67.4% | 0.0% | 69.1% | 0.0% |
| Ministral-8B | **+AgentAlign** | **10.5%** | **79.5%** | 63.3% | 2.8% |
| Qwen-2.5-7B | 标准 | 41.9% | 21.6% | 53.4% | 0.0% |
| Qwen-2.5-7B | **+AgentAlign** | **6.7%** | **85.8%** | **64.2%** | 5.7% |
| Functionary | 标准 | 21.7% | 52.8% | 45.9% | 0.6% |
| Functionary | **+AgentAlign** | **5.5%** | **88.6%** | **53.5%** | 1.7% |
| Claude-3.5-Haiku | - | 10.4% | 86.4% | 68.6% | **15.9%** |

### 消融实验（基于 Qwen-2.5-7B）

| 配置 | 有害Score↓ | 有害Refusal↑ | 良性Score↑ | 说明 |
|------|-----------|-------------|-----------|------|
| 完整AgentAlign | 6.7% | 85.8% | 64.2% | 最优平衡 |
| 去除良性样本 | ~10% | ~90% | ~35% | 良性能力大幅下降，过度拒绝 |
| 去除有害样本 | ~40% | ~20% | ~55% | 安全意识几乎丧失 |
| 去除第三方数据 | ~8% | ~88% | ~58% | 影响较小，主要增加拒绝率 |

### 与Prompting方法的正交性（AgentAlign + Prompting）

| 方法 | Ministral有害Refusal↑ | Qwen有害Refusal↑ |
|------|---------------------|-----------------|
| AgentAlign | 79.5% | 85.8% |
| AgentAlign + Refusal Prompt | **88.6%** | **92.0%** |
| AgentAlign + ReAct | 75.6% | 83.5% |

### 关键发现
- 安全提升量与基模型初始安全性**负相关**：越不安全的模型提升越大（Ministral从0%到79.5%）
- AgentAlign 对良性任务的影响极小，Qwen 甚至有提升（53.4%→64.2%）
- 与 Claude-3.5-Haiku 相比，AgentAlign 模型达到相当安全水平但**误拒绝率显著更低**（1.7-5.7% vs 15.9%）
- 在 ToolSword 基准上也验证了泛化性：Ministral 从58.2%提升到100%

## 亮点与洞察
- **行为链抽象**是核心创新：将有害活动抽象为工具动作序列，实现了"一次设计、无限实例化"的数据放大效果
- **非对称语义验证**巧妙解决了安全边界校准问题——对良性严格、对有害宽松
- **模拟环境 > 真实API**：Rapid API Hub 质量太差且缺少写操作，自建模拟环境反而更好
- 安全对齐数据的配比至关重要，过多安全数据会严重损害效用
- 人评 93% majority-pass rate 验证了低成本自动合成方案的可行性

## 局限与展望
- 模拟环境与真实API的差异可能影响迁移效果
- 未考虑动态多轮交互场景（用户中途改变需求）
- LLM 生成的数据仍有约7%的缺陷（人评）
- 仅在 AgentHarm 和 ToolSword 上评估，原生 benchmark 局限性
- 行为链设计依赖人工种子和审核，自动化程度可进一步提升

## 相关工作与启发
- **vs ToolAlign (Chen et al., 2024)**: ToolAlign 基于改造现有工具使用数据集，指令仍偏信息检索型（GET为主），缺少 agent 场景关键的写操作；AgentAlign 从零构建行为链和模拟环境，覆盖更广泛的多步 agent 操作
- **vs AgentHarm (Andriushchenko et al., 2024)**: AgentHarm 是评估 benchmark，发现了 agent 安全问题但未提供解决方案；AgentAlign 直接提供了对齐训练方案
- **vs 通用 RLHF/DPO 对齐**: 传统对齐关注单轮文本对话安全，未覆盖多步工具调用场景的安全性

## 评分
- 新颖性: ⭐⭐⭐⭐ 行为链中介的数据合成框架有创意，但核心思路（合成安全数据→SFT）较常规
- 实验充分度: ⭐⭐⭐⭐⭐ 三个模型家族、多种baseline、消融、正交实验、人评、跨benchmark验证
- 写作质量: ⭐⭐⭐⭐⭐ 问题引入有力，方法描述系统完整，实验设计合理
- 价值: ⭐⭐⭐⭐ 填补了 agent 安全对齐训练的空白，开源数据集和代码对社区有益

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)
- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[ACL 2025\] LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](lssf_safety_subspace.md)
- [\[ACL 2025\] Safety Alignment via Constrained Knowledge Unlearning](safety_alignment_via_constrained_knowledge_unlearning.md)
- [\[CVPR 2025\] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](../../CVPR2025/llm_alignment/do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)

<!-- RELATED:END -->
