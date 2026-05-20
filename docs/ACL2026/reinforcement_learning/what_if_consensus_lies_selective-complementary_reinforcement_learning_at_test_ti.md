---
title: >-
  [论文解读] SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time
description: >-
  [ACL 2026][强化学习][测试时强化学习] 本文提出 SCRL（Selective-Complementary Reinforcement Learning），一个鲁棒的测试时强化学习框架…
tags:
  - "ACL 2026"
  - "强化学习"
  - "测试时强化学习"
  - "伪标签噪声"
  - "负标签"
  - "共识可靠性"
  - "无监督推理"
---

# SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time

**会议**: ACL 2026  
**arXiv**: [2603.19880](https://arxiv.org/abs/2603.19880)  
**代码**: [https://github.com/Jasper-Yan/SCRL](https://github.com/Jasper-Yan/SCRL)  
**领域**: 强化学习 / LLM推理  
**关键词**: 测试时强化学习, 伪标签噪声, 负标签, 共识可靠性, 无监督推理

## 一句话总结

本文提出 SCRL（Selective-Complementary Reinforcement Learning），一个鲁棒的测试时强化学习框架，通过选择性正伪标签（严格共识标准过滤不可靠多数）和熵门控负伪标签（首次在 TTRL 中引入负监督信号来修剪错误轨迹）缓解标签噪声放大问题，在 AIME25 上比 TTRL 提升高达 10.1 个百分点。

## 研究背景与动机

**领域现状**：测试时强化学习（TTRL）让 LLM 在无标签测试流上通过多数投票共识派生伪奖励进行自我改进，已成为无监督推理的关键范式。

**现有痛点**：现有 TTRL 方法完全依赖正伪标签策略——多数投票选择最频繁答案作为正标签。但在困难问题上，答案分布高度分散，共识薄弱。GRPO 的组归一化会放大噪声：当正标签比例 $f$ 很小时，正样本的归一化优势 $\hat{A}^+ = \sqrt{(1-f)/f}$ 变得很大，少数错误的正伪标签就会不成比例地影响策略更新，导致过早收敛到虚假解。

**核心矛盾**：在困难问题上，确定正确答案很难，但识别错误答案相对容易。然而现有方法忽视了负标签的潜力——当无法可靠地识别正确答案时，可以通过修剪错误轨迹来缩小搜索空间。

**本文目标**：在 TTRL 中同时利用正信号和负信号，在共识不可靠时通过负标签修剪搜索空间而非强行选择正标签。

**切入角度**：区分"低频但可能正确"和"低频且确实错误"的答案——通过生成不确定性（token 级熵）来判断：高频低熵=可能正确，低频高熵=大概率错误。

**核心 idea**：当共识足够强时才提供正监督（选择性），当共识不够时通过负标签修剪确定错误的轨迹（互补性），两者配合动态奖励塑形实现鲁棒的测试时学习。

## 方法详解

### 整体框架

SCRL 由三个组件构成：(1) 选择性正伪标签——仅在答案分布呈尖锐集中且与次优有明确分离时才赋予正标签；(2) 熵门控负伪标签——对同时满足低频和高不确定性的答案赋予负标签；(3) 动态奖励塑形——基于共识强度校准奖励幅度，整合正负信号。基于 GRPO 算法。

### 关键设计

1. **选择性正伪标签**:

    - 功能：防止在弱共识时将错误答案作为正标签强化
    - 核心思路：给定 $N$ 个响应的答案分布 $\{p_j\}$，声明正伪标签 $y^+ = a_{j^*}$ 当且仅当最高比例 $p_{j^*} \geq \tau_{\text{pos}}$（足够多支持）且 $(p_{j^*} - p_{(2)}) > \tau_{\text{marg}}$（与次优有足够间距）。否则 $y^+ = \varnothing$，放弃正监督
    - 设计动机：当答案分布分散时，多数投票选出的答案可能只比其他答案略多，不可靠。严格的阈值+间距双重条件确保只在高置信度时提供正信号

2. **熵门控负伪标签**:

    - 功能：首次在 TTRL 中引入负监督，修剪确定错误的轨迹
    - 核心思路：计算每个答案 $a_j$ 对应轨迹的平均 token 级熵 $\bar{H}_j$。答案 $a_j$ 被标记为负伪标签当且仅当 $p_j < \tau_{\text{neg}}$（低频）且 $\bar{H}_j \geq \bar{H}$（高于全局平均不确定性）。关键约束 $\bar{H}_j \geq \bar{H}$ 确保不惩罚低频但低不确定性的答案（可能是罕见但正确的解）
    - 设计动机：低频可能是"罕见但正确"或"确实错误"。熵条件区分了这两种情况——高不确定性表明模型在该轨迹上"不自信"，很可能是错误推理

3. **动态奖励塑形**:

    - 功能：根据共识强度校准正负信号的奖励幅度
    - 核心思路：正奖励 = 答案比例 $p(a_i)$（共识越强奖励越大）；负奖励 = $(p(a_i) - \tau_{\text{neg}})$（越稀少惩罚越重）；熵惩罚 = $-\lambda_H(\bar{H}(a_i) - \bar{H})$（偏向低不确定性响应）。三项加权组合
    - 设计动机：固定奖励在共识强弱变化时会放大噪声。动态缩放让奖励信号的强度与共识可靠性成正比

### 损失函数 / 训练策略

使用 GRPO 作为基础 RL 算法。AdamW 优化器，余弦学习率调度（峰值 $5 \times 10^{-7}$）。rollout 生成 64（或 32）个候选响应用于标签估计，下采样 32（或 16）个用于训练更新。阈值 $\tau_{\text{pos}}=0.375, \tau_{\text{marg}}=0.125, \tau_{\text{neg}}=0.125$。8×A100 80GB GPU。

## 实验关键数据

### 主实验

**Qwen2.5-Math-7B 上的 pass@1 准确率（%）**

| 方法 | AIME25 | AMC | MATH-500 | Minerva | 平均 |
|------|--------|-----|---------|---------|------|
| 基线 | 4.6 | 34.0 | 46.5 | 10.1 | 23.6 |
| + TTRL | 16.8 | 65.7 | 85.7 | 14.5* | 41.6 |
| **+ SCRL** | **26.9** | **66.9** | **85.6** | **41.6** | **49.3** |
| Δ vs TTRL | +10.1 | +1.2 | -0.1 | +27.1 | +7.7 |

*注：TTRL 在 Minerva 上达到 14.5% 峰值后性能急剧退化*

**Llama-3.1-8B-Instruct 上的 pass@1（%）**

| 方法 | AIME24 | AMC | 平均 |
|------|--------|------|------|
| + TTRL | 10.0 | 32.3 | 21.2 |
| + RESTRAIN | 16.7 | 40.0 | 28.4 |
| **+ SCRL** | **21.9** | 36.1 | **29.0** |

### 消融实验

| 配置 | AIME25 | AMC | 说明 |
|------|--------|-----|------|
| Full SCRL | 26.9 | 66.9 | 完整模型 |
| w/o 负标签 | 19.4 | 65.3 | 负标签贡献 +7.5 (AIME25) |
| w/o 选择性正标签 | 21.5 | 66.1 | 选择性贡献 +5.4 |
| w/o 熵门控 | 23.1 | 65.8 | 熵条件贡献 +3.8 |
| w/o 动态奖励 | 24.2 | 66.0 | 动态奖励贡献 +2.7 |

### 关键发现

- 在最困难的任务（AIME25）上提升最大（+10.1），恰好是弱共识问题最严重的场景
- Minerva 上 TTRL 出现训练不稳定（性能先升后急剧下降），SCRL 保持稳定训练动态（41.6% vs 14.5%）
- 负标签在困难任务上贡献最大（AIME25 +7.5），验证了"不知道什么对就排除什么错"的策略价值
- SCRL 的效果在低 rollout 预算下更显著——预算受限时共识更不可靠，SCRL 的保护机制更重要
- 跨模型族（Qwen、Llama）和跨规模（1B-7B）一致有效，展现模型无关性

## 亮点与洞察

- "共识可能是错的"这一洞察直击 TTRL 的根本假设，负标签机制是自然且优雅的解决方案
- 熵门控是区分"罕见正确"和"确实错误"的关键——仅用频率或仅用不确定性都不够，两者的交叉条件才可靠
- 动态奖励塑形让正负信号的强度与共识可靠性自适应匹配，避免了固定奖励的噪声放大

## 局限与展望

- 阈值参数（$\tau_{\text{pos}}, \tau_{\text{marg}}, \tau_{\text{neg}}$）跨所有实验固定，可能对某些任务不够灵活
- 仅在数学和通用推理任务上验证，代码生成等任务未涉及
- 负标签在简单任务上贡献较小，可能引入不必要的复杂性
- 未来可探索自适应阈值机制和更细粒度的不确定性估计

## 相关工作与启发

- **vs TTRL**: TTRL 仅用正标签，在弱共识时放大噪声；SCRL 增加选择性和负标签两个保护机制
- **vs RESTRAIN**: RESTRAIN 惩罚过度自信和低一致性响应，但仍在正标签框架内；SCRL 引入了真正的负监督信号
- **vs SPINE**: SPINE 限制更新到高熵 forking token，SCRL 在答案级别操作，互补性强

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在 TTRL 中引入负监督信号，选择性+互补性的框架设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多规模、多任务、详尽消融和标签质量分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰，公式推导完整
- 价值: ⭐⭐⭐⭐⭐ 解决了 TTRL 的核心瓶颈，提升幅度显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning](beyond_majority_voting_towards_fine-grained_and_more_reliable_reward_signal_for_.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [\[ICML 2025\] Test-Time Adaptation with Binary Feedback](../../ICML2025/reinforcement_learning/test-time_adaptation_with_binary_feedback.md)
- [\[AAAI 2026\] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](../../AAAI2026/reinforcement_learning/aligning_machiavellian_agents_behavior_steering_via_test-tim.md)

</div>

<!-- RELATED:END -->
