---
title: >-
  [论文解读] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks
description: >-
  [ICLR 2026][LLM对齐][多轮越狱攻击] 提出 SEMA 框架，通过预填充自调优和带意图漂移感知奖励的 RL 两阶段训练，在无需任何现有攻击策略或外部数据的条件下，训练出能自动生成多轮越狱攻击的 attacker，在 AdvBench 上跨三个受害模型平均 ASR@1 达 80.1%…
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "多轮越狱攻击"
  - "强化学习红队"
  - "意图漂移"
  - "开环攻击"
  - "LLM安全"
---

# SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks

**会议**: ICLR 2026  
**arXiv**: [2602.06854](https://arxiv.org/abs/2602.06854)  
**代码**: [https://github.com/fmmarkmq/SEMA](https://github.com/fmmarkmq/SEMA)  
**领域**: 对齐RLHF  
**关键词**: 多轮越狱攻击, 强化学习红队, 意图漂移, 开环攻击, LLM安全

## 一句话总结
提出 SEMA 框架，通过预填充自调优和带意图漂移感知奖励的 RL 两阶段训练，在无需任何现有攻击策略或外部数据的条件下，训练出能自动生成多轮越狱攻击的 attacker，在 AdvBench 上跨三个受害模型平均 ASR@1 达 80.1%，超越 SOTA 33.9%。

## 研究背景与动机
多轮越狱比单轮越狱更贴近真实威胁模型——真实世界中用户与 chatbot 的交互是持续对话，攻击者可以逐步引导模型放松防御。然而，现有多轮攻击方法面临严重挑战：

**挑战一：探索复杂度爆炸**。多轮设置下，动作空间随轮次指数增长，RL 智能体很难高效探索有效的攻击路径。

**挑战二：意图漂移（Intent Drift）**。多轮对话中，攻击者容易在逐步引导过程中偏离原始的有害目标——前几轮为了"铺垫"而引入的无害话题可能让后续对话永远回不到有害意图。

**现有方法的局限**：人工设计的多轮攻击策略（如 Crescendo、PAIR）依赖固定模板，缺乏适应性；基于 RL 的方法需要闭环与受害模型交互（closed-loop），训练成本高且易受反馈不稳定影响。

**核心idea**：采用开环（open-loop）攻击范式——生成完整的多轮攻击序列而不需要受害模型的中间反馈，统一单轮和多轮设置，大幅降低探索复杂度；同时设计意图漂移感知奖励来锚定有害目标。

## 方法详解

### 整体框架
SEMA 以 Llama-3.1-8B-Instruct 为底座，从零开始训练一个能自动写出多轮越狱对话的 attacker，全程不碰任何现成攻击模板或外部数据。训练分两阶段：先用预填充自调优（Prefilling Self-Tuning）让模型学会写出格式正确、不自我拒绝的多轮攻击序列，给后续 RL 一个能跑起来的起点；再用带意图漂移感知奖励的 GRPO 把攻击能力推到位。整个攻击采用开环（open-loop）范式，一次性生成完整多轮序列，不依赖受害模型的中间反馈。

### 关键设计

**1. 预填充自调优：解决 RL 的冷启动死局。** 直接拿未训练的 LLM 上 RL 是行不通的——它根本不会写多轮攻击序列，初始 rollout 全是质量极差或干脆拒绝的输出，奖励信号近乎全零，RL 永远收敛不了。SEMA 的破解办法是让模型自举：给它一个最小前缀（比如有害目标的开头若干 token），强制它沿着这个前缀续写出非拒绝（non-refusal）、结构完整的多轮对抗 prompt；把这些自生成的样本收集起来做一轮标准 SFT，模型就学会了"多轮攻击长什么样"。整个过程不需要任何人工模板或外部标注，前缀长度是这一阶段唯一的关键超参——太短则模型容易跑偏成拒绝，太长则等于直接喂答案、丧失多样性。消融显示去掉这一步后 RL 阶段无法收敛，它是后续训练成功的硬前提。

**2. 开环攻击范式：用一次性生成统一单轮与多轮。** 传统 RL 红队是闭环（closed-loop）的：发一轮、等受害模型回复、再据此生成下一轮，动作空间随轮次指数膨胀，探索复杂度爆炸，且训练严重依赖受害模型反馈的稳定性。SEMA 反其道而行——attacker 一口气把完整的多轮攻击序列写完再整体发给受害模型，中途不看任何反馈。这样做有三个直接收益：探索被压缩成"生成一段序列"的单步决策，复杂度大幅下降；单轮只是多轮序列长度为 1 的特例，两种设置被天然统一；由于训练时不绑定特定受害模型的反馈，学出的 attacker 可以零样本迁移到没见过的受害模型，包括闭源模型。

**3. 意图漂移感知奖励：把攻击锚回原始有害目标。** 多轮攻击的通病是意图漂移（intent drift）——为了铺垫而引入的无害话题会让对话越走越偏，最后再也回不到原始有害意图。SEMA 把这件事写进奖励函数：最终奖励是三个维度的加权组合——意图对齐（intent alignment，整段多轮对话是否仍咬住原始有害目标）、合规风险（compliance risk，受害模型是否真的给出了有害回答）、以及详细程度（level of detail，有害回答的具体充分程度）。三者各自独立打分后加权合并；其中意图对齐这一项是关键，它像锚一样把整条攻击轨迹拉回有害目标，消融中去掉它后攻击明显偏航、ASR 下降。

### 损失函数 / 训练策略
Stage 1 用标准 SFT 交叉熵损失在自生成数据上微调。Stage 2 用 GRPO（Group Relative Policy Optimization），以上述意图漂移感知奖励作为信号；GRPO 的组内相对比较恰好契合开环"一次生成整段序列"的设定。训练用 4-8×H100，Stage 2 的受害模型可与 attacker 不同，以此把"跨模型迁移"直接写进训练目标。

## 实验关键数据

### 主实验

| 受害模型 | 数据集 | SEMA ASR@1 | 之前SOTA | 提升 |
|---------|--------|-----------|---------|------|
| 3个模型平均 | AdvBench | 80.1% | 46.2% | +33.9% |
| 闭源模型 | AdvBench | 高 | 低 | 显著提升 |
| 开源模型 | AdvBench | 高 | 中等 | 显著提升 |

### 消融实验

| 配置 | ASR | 说明 |
|------|-----|------|
| SEMA (完整) | 最高 | 两阶段+意图漂移奖励 |
| SFT-only | 低 | 仅Stage 1，无RL |
| DPO变体 | 中等 | 偏好优化不如RL |
| 无意图漂移奖励 | 下降 | 攻击容易偏离目标 |
| 单轮设置 | 有效但低于多轮 | 验证统一性 |

### 关键发现
- SEMA 超越所有单轮基线、人工脚本多轮基线和模板驱动多轮基线
- 同时超越 SFT 和 DPO 变体，证明 RL 在多轮攻击学习中的优势
- 开环攻击可以直接迁移到未见过的受害模型，包括闭源模型
- 预填充自调优是 RL 阶段成功的关键前提——缺少它 RL 无法收敛
- 方法紧凑、可复现，代码将在 ICLR 会议前开源

## 亮点与洞察
- 分析了多轮越狱的真实威胁模型，指出单轮攻击只是特例，这重新定义了LLM安全评估的标准
- 开环攻击范式的设计非常巧妙——避免了闭环交互的复杂性，同时保证了迁移能力
- 意图漂移感知奖励是对多轮安全研究的重要贡献——精确定义了"多轮攻击成功"的含义
- 从零开始自举（bootstrap）攻击能力的思路是自动红队的重要进展
- Prefilling Self-Tuning的设计解决了RL冷启动的通用难题，可推广到其他序列生成RL任务
- 33.9%的ASR@1提升（vs SOTA）表明现有LLM的多轮防御能力远不够
- 方法的紧凑性（代码量小、训练成本适中）使其适合作为标准化的安全压力测试工具

## 局限与展望
- 代码仍在 Microsoft Research 审核中，尚未完全公开
- 开环攻击可能不如闭环攻击在特定受害模型上的效果好（无法利用中间反馈调整策略）
- 防御方可以简单地检测多轮对话中的模式来防御此类攻击
- 伦理考虑——该方法可能被滥用，但作者定位为暴露LLM安全漏洞的压力测试工具
- 尚未评估在最新防御方法（如Llama Guard 3等专用安全检测器）下的效果
- 训练数据的有害内容需要严格的访问控制和使用规范

## 相关工作与启发
- **vs Crescendo/PAIR**: 人工设计模板，SEMA完全从数据中学习攻击策略
- **vs AutoDAN-Turbo**: 依赖现有攻击策略库，SEMA完全自举
- **vs 单轮攻击(GCG/AutoDAN)**: SEMA统一了单轮和多轮，更贴近真实威胁

## 评分
- 新颖性: ⭐⭐⭐⭐ 开环多轮攻击+意图漂移感知奖励的组合很有创新性
- 实验充分度: ⭐⭐⭐⭐ 37页、13表、7图，多数据集多受害模型
- 写作质量: ⭐⭐⭐⭐ 结构清晰，两阶段设计动机阐述到位
- 价值: ⭐⭐⭐⭐ 对LLM安全红队研究有直接推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search](obscure_but_effective_classical_chinese_jailbreak_prompt_optimization_via_bio-in.md)
- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](../../ACL2025/llm_alignment/m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[ACL 2025\] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search](../../ACL2025/llm_alignment/tempest_autonomous_multi-turn_jailbreaking_of_large_language_models_with_tree_se.md)

</div>

<!-- RELATED:END -->
