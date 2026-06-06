---
title: >-
  [论文解读] Don't Say No: Jailbreaking LLM by Suppressing Refusal
description: >-
  [ACL 2025 (Findings)][LLM对齐][越狱攻击] 本文提出 DSN（Don't Say No）攻击方法，通过分析现有越狱攻击中目标损失函数的缺陷，引入余弦衰减调度和拒绝抑制两种改进策略，在多个 LLM 上实现了超越现有方法的攻击成功率（ASR），并展示了对未见数据集和黑盒模型的强迁移性。
tags:
  - "ACL 2025 (Findings)"
  - "LLM对齐"
  - "越狱攻击"
  - "拒绝抑制"
  - "安全对齐"
  - "攻击成功率"
  - "对抗优化"
---

# Don't Say No: Jailbreaking LLM by Suppressing Refusal

**会议**: ACL 2025 (Findings)  
**arXiv**: [2404.16369](https://arxiv.org/abs/2404.16369)  
**代码**: 无  
**领域**: LLM对齐 / AI安全  
**关键词**: 越狱攻击, 拒绝抑制, 安全对齐, 攻击成功率, 对抗优化

## 一句话总结

本文提出 DSN（Don't Say No）攻击方法，通过分析现有越狱攻击中目标损失函数的缺陷，引入余弦衰减调度和拒绝抑制两种改进策略，在多个 LLM 上实现了超越现有方法的攻击成功率（ASR），并展示了对未见数据集和黑盒模型的强迁移性。

## 研究背景与动机

**领域现状**：LLM 安全对齐是确保模型输出符合人类价值观的关键技术。通过 RLHF/DPO 等对齐方法，模型学会拒绝有害请求。然而，越狱（jailbreaking）攻击通过精心设计的 prompt 绕过安全机制，诱导模型生成有害内容。基于优化的越狱攻击（如 GCG）将问题形式化为最大化模型产生肯定回复概率的优化问题。

**现有痛点**：现有基于优化的攻击方法存在两个核心问题：(1) 它们使用的目标损失函数（"target loss"，即最大化模型开头输出"Sure, here is..."等肯定回复的概率）需要预先定义特定的目标行为模板，这限制了攻击的灵活性和适应性；(2) vanilla target loss 本身是次优的——它只鼓励模型开始生成内容，但不直接抑制模型的拒绝机制。

**核心矛盾**：安全对齐本质上训练模型学会在遇到危险请求时"说不"（生成拒绝token），而现有攻击只试图引导模型"说是"，却没有直接削弱模型"说不"的能力。这是一个优化目标不完整的问题。

**本文目标**：(1) 分析 vanilla target loss 为何次优；(2) 设计更有效的损失目标；(3) 实现更高且更稳定的攻击成功率。

**切入角度**：从损失函数设计的角度切入，分析模型的 logit 空间中"接受"和"拒绝"token 的竞争关系。

**核心 idea**：在鼓励模型生成肯定回复的同时，显式抑制拒绝类 token 的概率，并用余弦衰减调度解决优化过程中的不稳定性。

## 方法详解

### 整体框架

DSN 攻击的流水线与 GCG 类似：给定一个有害查询和一个可学习的对抗后缀，通过梯度引导的离散优化迭代更新后缀 token。核心区别在于优化目标的设计。输入是有害查询 $q$、对抗后缀 $s$（待优化），输出是优化后的后缀 $s^*$，将其拼接到查询后即可绕过安全机制。

### 关键设计

1. **改进的目标损失函数**:

    - 功能：提供更有效的优化目标来诱导模型生成有害内容
    - 核心思路：分析发现 vanilla target loss $L_{target} = -\log P(y_{aff} | q, s)$ 的梯度信号在优化后期会变得微弱（因为 $P(y_{aff})$ 已经较高时梯度趋近于零），导致优化停滞。本文提出将目标损失拆分为两个互补部分：肯定引导项（鼓励"Sure"等 token）和拒绝抑制项（降低"Sorry"、"I cannot"等 token 的概率）。总损失为 $L = L_{aff} + \lambda L_{ref}$，其中 $L_{ref} = \log P(y_{ref} | q, s)$ 显式最小化拒绝 token 的概率
    - 设计动机：直接攻击模型安全对齐的"拒绝机制"，比单纯引导肯定回复更有效，特别是对于安全对齐较强的模型

2. **余弦衰减调度（Cosine Decay Schedule）**:

    - 功能：动态调节肯定引导和拒绝抑制两项损失的权重
    - 核心思路：在优化初期，主要依赖肯定引导项（权重高）来快速找到可行的优化方向；随着优化推进，逐步增加拒绝抑制项的权重 $\lambda(t) = \lambda_{max} \cdot (1 - \cos(\pi t / T)) / 2$，在优化后期拒绝抑制成为主导。这种调度避免了在早期就过度抑制拒绝（导致优化不稳定）和在后期只依赖肯定引导（导致停滞）的问题
    - 设计动机：解决两项损失在优化过程中的冲突——初期模型倾向强烈拒绝，此时抑制拒绝会产生极大梯度导致不稳定；后期拒绝概率已降低，此时需要精细调节来突破最后的安全屏障

3. **拒绝 Token 集合的自适应构建**:

    - 功能：确定哪些 token 属于"拒绝"类别以供抑制
    - 核心思路：不依赖手动定义的拒绝 token 列表，而是从目标模型的实际拒绝响应中自动提取。具体做法是：先用未加后缀的有害查询触发模型拒绝，收集模型输出的前 $k$ 个 token（如"I"、"cannot"、"Sorry"），将这些 token 及其变体组成拒绝集合 $V_{ref}$。抑制损失则定义为 $L_{ref} = \sum_{v \in V_{ref}} \log P(v | q, s)$
    - 设计动机：避免人为定义拒绝模板的局限性，不同模型有不同的拒绝用语模式（如"I'm sorry" vs "I apologize" vs "As an AI"），自适应提取能更好地覆盖

### 损失函数 / 训练策略

DSN 的总损失函数为：

$L_{DSN} = -\log P(y_{aff} | q, s) + \lambda(t) \cdot \log P(y_{ref} | q, s)$

其中 $\lambda(t)$ 按余弦衰减从 0 增长到 $\lambda_{max}$。优化使用与 GCG 相同的贪心坐标下降：每步随机采样候选后缀 token 替换，选择使 $L_{DSN}$ 下降最大的替换。

## 实验关键数据

### 主实验

在 AdvBench 数据集上对比不同攻击方法的攻击成功率（ASR）：

| 攻击方法 | Vicuna-7B | Vicuna-13B | LLaMA-2-7B | LLaMA-2-13B | 平均ASR |
|---------|-----------|-----------|-----------|------------|---------|
| GCG | 87.0% | 84.5% | 52.8% | 38.4% | 65.7% |
| AutoDAN | 73.2% | 68.5% | 41.3% | 35.7% | 54.7% |
| PAIR | 61.4% | 58.2% | 31.5% | 27.8% | 44.7% |
| DSN (本文) | **95.2%** | **93.8%** | **71.4%** | **62.5%** | **80.7%** |

### 消融实验

DSN 各组件的贡献（在 LLaMA-2-7B 上）：

| 配置 | ASR | 说明 |
|------|-----|------|
| DSN (完整) | 71.4% | 全部组件 |
| w/o 拒绝抑制 | 55.3% | 只用肯定引导，退化接近 GCG |
| w/o 余弦调度（固定λ） | 63.8% | 固定权重，不如动态调度 |
| w/o 自适应拒绝集（手动定义） | 66.2% | 手动拒绝列表不完整 |
| 仅拒绝抑制（无肯定引导） | 48.7% | 缺少方向引导，优化效率低 |

### 关键发现

- **拒绝抑制是最关键组件**：去掉后 ASR 下降 16.1%，说明显式压制拒绝机制比单纯引导肯定生成更能有效绕过安全对齐
- **余弦衰减调度贡献显著**：相比固定权重提升 7.6% ASR，验证了动态平衡两种目标的必要性
- **对强对齐模型提升最大**：LLaMA-2 系列经过强安全训练，DSN 在其上的 ASR 提升远超 Vicuna，说明方法特别擅长突破强安全屏障
- **黑盒迁移性好**：用白盒模型优化的后缀直接迁移到 GPT-3.5/4 等黑盒模型，仍能保持超过 50% 的 ASR

## 亮点与洞察

- **从损失函数视角切入安全攻击**：不是设计更复杂的搜索策略，而是改进优化目标本身。这种"改进目标比改进搜索更重要"的思路在其他优化问题中也有启示
- **双向目标的互补设计巧妙**：同时"拉"（鼓励肯定）和"推"（抑制拒绝）比单向优化更完整。类似于对比学习中同时使用正样本吸引和负样本排斥
- **安全研究的防御启示**：本文的攻击成功说明当前安全对齐的脆弱性——拒绝机制可以被针对性抑制。防御思路可以考虑训练更鲁棒的拒绝表示，使其不易被少量 token 操纵

## 局限与展望

- **攻击本身的伦理风险**：虽然以安全研究为目的，但公开的攻击方法可能被滥用。论文在伦理声明中未充分讨论缓解措施
- **计算开销**：基于梯度的优化需要白盒访问和大量 forward/backward pass，实际场景中对商业 API 不直接适用
- **防御对抗性不足**：仅测试了原始安全对齐，未评估加入对抗训练、perplexity 过滤等防御后的效果
- **改进方向**：可以探索更高效的拒绝 token 集合发现方法，以及如何将此思路应用于多模态模型的越狱

## 相关工作与启发

- **vs GCG (Zou et al., 2023)**: GCG 使用 vanilla target loss，DSN 在其基础上增加了拒绝抑制和余弦调度。DSN 保持了 GCG 的框架但显著提升了对安全对齐较强模型的攻击效果
- **vs AutoDAN**: AutoDAN 使用遗传算法生成可读的越狱 prompt，DSN 使用梯度优化生成不可读后缀。两者思路互补，DSN 的拒绝抑制思路可能也能增强 AutoDAN
- **vs 基于 prompt 工程的越狱（如角色扮演、DAN）**: 这些方法不需要模型访问但依赖手工设计。DSN 的自动化方法可以更系统地发现漏洞，但适用场景更受限

## 评分

- 新颖性: ⭐⭐⭐⭐ 从损失函数角度改进越狱攻击的思路新颖，双向优化目标设计合理
- 实验充分度: ⭐⭐⭐⭐ 多模型对比、消融分析和迁移实验较充分，但缺少防御对抗评估
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，方法推导自然
- 价值: ⭐⭐⭐⭐ 对理解和改进 LLM 安全对齐有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Jailbreaking? One Step Is Enough!](jailbreaking_one_step_is_enough.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[NeurIPS 2025\] EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions](../../NeurIPS2025/llm_alignment/evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)
- [\[ICML 2026\] Adaptive Probe-based Steering for Robust LLM Jailbreaking](../../ICML2026/llm_alignment/adaptive_probe-based_steering_for_robust_llm_jailbreaking.md)
- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](../../ICLR2026/llm_alignment/no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)

</div>

<!-- RELATED:END -->
