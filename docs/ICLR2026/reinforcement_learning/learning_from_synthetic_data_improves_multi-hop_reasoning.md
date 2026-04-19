---
title: >-
  [论文解读] Learning from Synthetic Data Improves Multi-hop Reasoning
description: >-
  [ICLR 2026][合成数据] 发现在完全虚构的规则生成合成数据上做RLVR训练，能显著提升LLM在真实多跳推理任务上的表现（Qwen3-0.6B提升56%-131%），因为模型学到了知识组合这一通用推理技能而非记忆事实知识。
tags:
  - ICLR 2026
  - 合成数据
  - 多跳推理
  - RLVR
  - 知识组合
  - 虚拟世界
---

# Learning from Synthetic Data Improves Multi-hop Reasoning

**会议**: ICLR 2026  
**arXiv**: [2603.02091](https://arxiv.org/abs/2603.02091)  
**代码**: [GitHub](https://github.com/kilian-group/phantom-reasoning)  
**领域**: LLM推理/强化学习  
**关键词**: 合成数据, 多跳推理, RLVR, 知识组合, 虚拟世界

## 一句话总结
发现在完全虚构的规则生成合成数据上做RLVR训练，能显著提升LLM在真实多跳推理任务上的表现（Qwen3-0.6B提升56%-131%），因为模型学到了知识组合这一通用推理技能而非记忆事实知识。

## 研究背景与动机

**领域现状**：RLVR通过可验证奖励训练LLM推理能力，在数学、编程等领域取得显著进展。但RLVR依赖大量高质量可验证数据——人工标注昂贵、LLM生成的合成数据有幻觉且成本高。

**现有痛点**：(1) 高质量训练数据稀缺且昂贵；(2) LLM生成的合成数据继承验证困难和预训练知识污染；(3) 规则生成的合成数据语义简单、完全虚构，能否教会有用技能存疑。

**核心矛盾**：PhantomWiki的问题如"Who is the nephew of the friend of the person whose hobby is birdwatching?"与HotpotQA的"Aside from Yodobashi, what other towns were merged into..."差距巨大——虚构简单模板 vs 真实复杂语言。从前者到后者的迁移并非显而易见。

**切入角度**：假设多跳推理的核心是"知识组合"——将多步信息链接的能力，这是一种领域无关的技能。虚构世界中零知识重叠意味着模型无法靠记忆走捷径，必须学会组合操作本身。

**核心 idea**：规则生成的虚构合成数据通过RLVR教会LLM知识组合这一通用技能，可免费、无限扩展地迁移到真实多跳推理。

## 方法详解

### 整体框架
用4种规则生成的合成数据集（PhantomWiki/GSM-∞/RG-Family/RG-Knights）做RLVR训练，在5个真实多跳QA基准上评估迁移效果。用GRPO算法，4个LLM家族（Qwen3/Phi-4, 0.6B-4B）。

### 关键设计

1. **合成数据集选择**:

    - PhantomWiki：虚构人物的多跳问答，模板+上下文无关文法，1-9跳难度
    - GSM-∞：无限数学应用题，随机计算图→自然语言，2-20步
    - RG-Family：推断家族树中两人关系（逻辑推理）
    - RG-Knights：骑士与说谎者逻辑谜题
    - 设计动机：覆盖不同推理风格，全部基于虚构世界，零知识重叠

2. **In-context推理设置**:

    - 功能：所有相关上下文放入prompt，测试in-context推理能力
    - 核心思路：PhantomWiki放入全部25篇文章，GSM-∞放入题目描述，用<answer>标签提取答案
    - 设计动机：控制变量，确保衡量的是推理能力而非知识检索

3. **因果分析**:

    - 格式消融：仅用<answer>格式做RLVR→Qwen3/Phi无提升→证明迁移来自推理能力而非格式学习
    - SFT对比：SFT在合成任务上有效但无法迁移到真实任务→证明RL教的是技能而非模式
    - 中间答案分析：训练中正确中间答案出现频率递增→证明组合能力在增长

### 损失函数 / 训练策略
- GRPO (无KL惩罚)，Hugging Face TRL v0.21.0
- PhantomWiki用F1奖励（多答案），其他用精确匹配二值奖励
- 10K训练样本，多种难度混合

## 实验关键数据

### 主实验
Qwen3-0.6B + PhantomWiki训练 → 真实基准：

| 基准 | Base F1 | +PhantomWiki | 相对提升 |
|------|---------|-------------|---------|
| HotpotQA | 0.36 | 0.73 | +103% |
| 2WikiMQA | 0.37 | 0.86 | +132% |
| MuSiQue | 0.14 | 0.28 | +100% |
| CofCA | 低 | 显著↑ | +56-131% |
| SynthWorlds | 低 | 显著↑ | +大幅 |

### 关键消融
| 配置 | 合成任务 | 真实任务 | 说明 |
|------|---------|---------|------|
| RLVR on PhantomWiki | ✓提升 | ✓迁移 | 完整方案 |
| SFT on PhantomWiki | ✓提升 | ✗不迁移 | SFT过拟合模式 |
| 格式only RLVR | ✗无帮助 | ✗无帮助 | Qwen3/Phi已会格式 |
| 更多合成数据 | 持续↑ | 持续↑ | 无过拟合迹象 |

### 关键发现
- 所有4种合成数据都产生正迁移，PhantomWiki（与目标任务最对齐）效果最好
- SFT在合成任务上提升但不迁移→RL确实教了通用技能，SFT只教了模式
- 模型在虚构的held-out世界和OOD难度上也能泛化→证明学到了可迁移的组合技能
- 性能随合成样本数单调增长，无过拟合→合成数据是可扩展的资源
- 训练过程中模型生成的正确中间答案数量增加→emergent grounded reasoning

## 亮点与洞察
- **"免费午餐"的极致**：完全不需要真实数据、不需要LLM标注、不需要GPU生成——普通电脑上用模板就能生成无限训练数据，且能大幅提升真实推理能力。
- **SFT vs RL的分水岭**：同样的合成数据，SFT会过拟合到合成任务的表面模式，RL教会了深层的组合技能。这是RL的independent价值的有力证据。
- **知识组合作为独立技能**：论文清晰分离了"知道事实"和"会组合事实"两种能力，证明后者可以独立学习。这挑战了"RL只是激活预训练知识"的观点。

## 局限与展望
- 真实任务vs合成任务的差距仍存在——真实数据RLVR效果更好
- 仅验证了0.6B-4B模型，更大模型的迁移模式可能不同
- 多跳推理是相对简单的推理类型，更复杂推理（如数学证明）的迁移待验证
- 合成数据的最优难度分布和混合策略未深入探索

## 相关工作与启发
- **vs 真实数据RLVR**: 真实数据效果更好但昂贵，合成数据免费且无限可扩展，两者互补
- **vs LLM蒸馏合成**: LLM生成的数据贵且有污染风险，规则生成的数据干净且免费
- **vs STILL/ART等**: 之前工作用合成数据评估RL，本文首次系统证明合成→真实的迁移

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 虚构合成数据→真实推理的迁移发现具有重要意义
- 实验充分度: ⭐⭐⭐⭐⭐ 4模型×4合成数据×5真实基准×多种消融，极其充分
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链条清晰，因果分析严谨
- 价值: ⭐⭐⭐⭐⭐ 开辟了用免费合成数据提升LLM推理的新范式

<!-- RELATED:START -->

## 相关论文

- [MMhops-R1: Multimodal Multi-hop Reasoning](../../AAAI2026/reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)
- [Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy](../../AAAI2026/reinforcement_learning/mathsmith_towards_extremely_hard_mathematical_reasoning_by_forging_synthetic_pro.md)
- [Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination](../../AAAI2026/reinforcement_learning/reasoning_or_memorization_unreliable_results_of_reinforcement_learning_due_to_da.md)
- [$\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)

<!-- RELATED:END -->
