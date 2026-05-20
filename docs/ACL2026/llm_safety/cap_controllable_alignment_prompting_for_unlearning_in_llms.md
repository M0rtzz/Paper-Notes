---
title: >-
  [论文解读] CAP: Controllable Alignment Prompting for Unlearning in LLMs
description: >-
  [ACL 2026][LLM安全][LLM遗忘] 提出 CAP 框架，通过训练轻量 SLM 生成可控的提示前缀来引导冻结的 LLM 选择性遗忘目标知识，无需修改模型参数，实现了可逆、可迁移的 LLM 知识遗忘。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "LLM遗忘"
  - "提示驱动"
  - "强化学习"
  - "可控对齐"
  - "知识消除"
---

# CAP: Controllable Alignment Prompting for Unlearning in LLMs

**会议**: ACL 2026  
**arXiv**: [2604.21251](https://arxiv.org/abs/2604.21251)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: LLM遗忘, 提示驱动, 强化学习, 可控对齐, 知识消除

## 一句话总结

提出 CAP 框架，通过训练轻量 SLM 生成可控的提示前缀来引导冻结的 LLM 选择性遗忘目标知识，无需修改模型参数，实现了可逆、可迁移的 LLM 知识遗忘。

## 研究背景与动机

**领域现状**：LLM 在无过滤语料上训练，不可避免地保留敏感信息。GDPR 等法规要求选择性知识遗忘（unlearning）。现有方法主要通过修改模型参数实现。

**现有痛点**：(1) 基于重训练和梯度的方法计算成本高；(2) 遗忘边界不可控，常导致整体性能退化；(3) 严格依赖模型权重访问，对闭源模型不可用；(4) 现有非侵入方法依赖经验设计提示，缺乏系统化的端到端训练框架。

**核心矛盾**：修改参数的方法虽然直接但代价高且不可逆，而不修改参数的方法（如提示工程）虽轻量但缺乏可控性和系统性优化。

**本文目标**：设计一个端到端的提示驱动遗忘框架，在不修改 LLM 参数的前提下实现精确、可控、可逆的知识遗忘。

**切入角度**：将遗忘问题转化为推理时控制问题——训练一个轻量 SLM 作为策略网络，生成输入条件化的控制前缀来引导冻结 LLM 的输出行为。

**核心 idea**：SLM 为每个输入查询生成两类提示前缀（遗忘提示和保留提示），通过变分信息瓶颈对比目标和 Beam PPO 强化学习优化，使 LLM 在抑制目标知识的同时保持一般能力。

## 方法详解

### 整体框架

CAP 包含两个阶段：(1) 提示生成器优化——使用 RL 训练 SLM 生成有效的遗忘/保留提示前缀；(2) 推理阶段——冻结的 SLM 生成提示前缀，配合 Self-Check 指令引导 LLM 最终输出。

### 关键设计

1. **双提示前缀机制（遗忘 + 保留）**:

    - 功能：分别引导 LLM 抑制目标知识和保持一般能力
    - 核心思路：SLM 为每个查询生成 $n$ 个遗忘提示候选 $\mathcal{P}_f^k$ 和 $n$ 个保留提示候选 $\mathcal{P}_r^k$，分别与查询拼接后输入冻结 LLM，得到遗忘答案集和保留答案集
    - 设计动机：双提示设计将遗忘和保留解耦为两个可独立优化的方向，避免单一提示中遗忘与保留的冲突

2. **变分信息瓶颈对比目标（VIB）**:

    - 功能：以信息论方式引导遗忘和保留的优化方向
    - 核心思路：对遗忘分支最小化 LLM 输出与标签间的互信息（变分上界 KL 散度），对保留分支最大化互信息（InfoNCE 下界）；两分支联合优化，$\beta$ 控制权衡
    - 设计动机：直接在信息论层面建模遗忘（压缩信息）和保留（保留信息），比基于启发式奖励的方法更有理论基础

3. **Beam PPO 强化学习优化**:

    - 功能：增强策略探索的稳定性和多样性
    - 核心思路：维护 $k$ 个锚策略的集束（beam），当前策略 $\pi_\theta$ 相对于所有锚策略的最小 KL 散度进行正则化，避免标准 PPO 的局部最优和策略崩溃
    - 设计动机：标准 PPO 在提示生成中缺乏稳定性，Beam PPO 通过多路径探索提供更大的参数空间覆盖

### 损失函数 / 训练策略

总奖励函数 $\mathcal{R} = \lambda_{VIB} \cdot \mathcal{R}_{VIB} + \lambda_{label} \cdot \mathcal{R}_{label} + \lambda_{len} \cdot \mathcal{R}_{len}$，其中 VIB 奖励引导信息压缩/保留，标签奖励评估遗忘/保留对齐度，长度正则化鼓励生成接近理想长度的简洁提示。B-PPO 目标函数在标准 PPO 的 clip 损失基础上加入多锚点 KL 正则化。

## 实验关键数据

### 主实验

| 模型 | 方法 | RWKU ASG↓ | WMDP Bio Acc↓ | MMLU Acc↑ |
|------|------|----------|--------------|----------|
| Zephyr-7B | Original | 63.0 | 63.7 | 54.1 |
| Zephyr-7B | NPO | 28.9 | 43.1 | 48.6 |
| Zephyr-7B | ICUL | 30.3 | 44.9 | 44.5 |
| Zephyr-7B | **CAP** | **6.2** | **24.8** | **51.5** |
| GPT-4.1 | ICUL | 36.7 | 38.6 | 81.5 |
| GPT-4.1 | **CAP** | **7.5** | **35.9** | **80.6** |
| Claude-Sonnet-4 | **CAP** | **7.4** | **30.1** | **84.2** |

### 消融实验

| 配置 | 遗忘 Acc↓ | 保留 Acc↑ | 说明 |
|------|----------|----------|------|
| 无 IB + 标准 PPO | 37.5 | 49.8 | 无结构化奖励 |
| + IB + B-PPO（完整 CAP） | 24.8 | 51.5 | 最佳平衡 |
| 仅遗忘 VIB | 25.6 | 44.7 | 保留性能受损 |
| 仅保留 VIB | 38.6 | 52.2 | 遗忘能力减弱 |
| 随机选择 vs Self-Check | 26.2/24.8 | 48.5/51.5 | Self-Check 为稳定性微调 |

### 关键发现
- CAP 在生成式任务中将 ASG 从 63.0 降至 6.2（Zephyr-7B），远超所有基线
- 在判别式任务中，CAP 显著降低 WMDP 准确率的同时保持了接近原始的 MMLU 性能
- CAP 无缝迁移到闭源模型（GPT-4.1、Claude-Sonnet-4、DeepSeek-V3 等），仅需离散提示
- Beam size $k=4$、候选数 $n=3$、最大提示长度 $L=16$ 为最优超参数配置
- 不同 SLM（Qwen3-0.6B、Qwen2.5-0.5B、Gemma3-1B）均可有效引导遗忘，方法具有模型无关性

## 亮点与洞察
- 将遗忘从参数空间转移到输出空间，通过离散提示实现可逆遗忘是核心创新——移除提示生成器即可恢复原始模型
- VIB 对比目标从信息论角度统一了遗忘（压缩）和保留（保留），比启发式奖励更优雅
- Beam PPO 对标准 PPO 的改进具有通用价值，不限于遗忘任务
- 隐藏状态可视化直觉地展示了提示如何将内部激活从知识区域重定向到安全/拒绝区域

## 局限与展望
- 两阶段推理（SLM 生成前缀 + LLM 生成输出）引入了边际延迟开销
- 生成的控制前缀占用 LLM 上下文窗口的一小部分
- SLM 固定为 Qwen3-0.6B（主实验），虽验证了其他 SLM 也有效，但最优 SLM 选择尚未充分探索
- 在对抗攻击下的鲁棒性虽优于基线，但仍非完美

## 相关工作与启发
- **vs LLMU/NPO**: 它们需修改 LLM 参数，不适用于闭源模型；CAP 完全不修改参数
- **vs ICUL**: ICUL 使用上下文学习驱动遗忘但缺乏负样本，对对抗分布适应性差；CAP 通过 RL 优化提示具有更强泛化性
- **vs SPUL**: SPUL 使用软提示调优但仍需梯度回传，CAP 使用离散提示无需访问 LLM 梯度
- **vs Pawelczyk et al.**: 他们提出基于分类器的非侵入方法但依赖分类器准确率；CAP 端到端优化更可靠

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 端到端提示驱动遗忘范式，VIB + Beam PPO 设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 7 个 LLM（含闭源）、多数据集、全面消融和敏感性分析
- 写作质量: ⭐⭐⭐⭐ 方法阐述清晰，理论推导完整
- 价值: ⭐⭐⭐⭐⭐ 对闭源 LLM 遗忘问题有重要实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](../../ICLR2026/llm_safety/inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)
- [\[ACL 2026\] Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment](can_persona-prompted_llms_emulate_subgroup_values_an_empirical_analysis_of_gener.md)
- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](../../CVPR2026/llm_safety/unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[ACL 2026\] FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness](faith_factuality_alignment_through_integrating_trustworthiness_and_honestness.md)

</div>

<!-- RELATED:END -->
