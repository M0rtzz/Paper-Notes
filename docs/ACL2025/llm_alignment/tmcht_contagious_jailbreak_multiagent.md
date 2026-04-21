---
title: >-
  [论文解读] A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns
description: >-
  [ACL 2025][LLM对齐][多智能体攻击] 提出TMCHT（大规模多智能体多拓扑文本攻击评估框架）和ARCJ（对抗性复制传染越狱）方法——通过优化检索后缀提高毒性样本被检索概率+优化复制后缀使毒性信息具有自我复制传染能力，解决了多智能体系统中单智能体攻击方法面临的"毒性消散"问题。
tags:
  - ACL 2025
  - LLM对齐
  - 多智能体攻击
  - 传染性越狱
  - 记忆攻击
  - 毒性消散
  - ARCJ
---

# A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns

**会议**: ACL 2025  
**arXiv**: [2410.16155](https://arxiv.org/abs/2410.16155)  
**代码**: 无  
**领域**: LLM安全 / 多智能体攻击  
**关键词**: 多智能体攻击, 传染性越狱, 记忆攻击, 毒性消散, ARCJ

## 一句话总结

提出TMCHT（大规模多智能体多拓扑文本攻击评估框架）和ARCJ（对抗性复制传染越狱）方法——通过优化检索后缀提高毒性样本被检索概率+优化复制后缀使毒性信息具有自我复制传染能力，解决了多智能体系统中单智能体攻击方法面临的"毒性消散"问题。

## 研究背景与动机

**领域现状**: LLM广泛用作agent，记忆是关键组件但易受越狱攻击。**现有痛点**: 现有攻击仅关注单智能体记忆或共享记忆，但真实场景中多智能体通常使用独立记忆。**核心矛盾**: 单智能体攻击方法在非完全图拓扑（line/star）和大规模（100+智能体）系统中效果急剧下降。**本文目标**: 在独立记忆的多智能体系统中实现有效的跨智能体越狱传播。**切入角度**: 发现"毒性消散"现象——有毒后缀在智能体间传播时逐渐消失，导致检索失败。**核心idea**: 让有毒信息具备"自我复制"能力，使其在传播过程中保持毒性。

## 方法详解

### 整体框架

TMCHT任务设定：一个攻击者智能体→在给定社交拓扑（graph/line/star）中→通过有限轮对话→误导整个社会的智能体。ARCJ方法两阶段：先优化检索后缀→再优化复制后缀。

### 关键设计

1. **检索后缀优化**:
    - 功能：使有毒样本在记忆检索中更容易被选中
    - 核心思路：最小化有毒样本嵌入与查询嵌入的距离：$\min_{\delta_r} \text{dist}(\text{Embed}(x + \delta_r), \text{Embed}(q))$，其中$\delta_r$是检索后缀
    - 设计动机：有毒样本必须首先被检索到才能发挥作用——如果正常内容的相似度更高，有毒样本永远不会被使用

2. **复制后缀优化（传染能力）**:
    - 功能：强制被攻击的LLM在回复中自动复制有毒信息
    - 核心思路：$\min_{\delta_c} -\log P(x + \delta_r + \delta_c | \text{context})$，最大化LLM回复中重现有毒文本的概率
    - 设计动机：解决"毒性消散"的核心——普通攻击在传播过程中后缀被改写/丢失，复制后缀让下游智能体的回复也包含完整有毒内容

3. **多拓扑评估框架（TMCHT）**:
    - 功能：定义graph/line/star三种社交拓扑下的多智能体攻击评估任务
    - 核心思路：攻击者仅与相邻智能体通信，需在有限轮数内影响整个网络；评估指标为ASR和影响范围
    - 设计动机：真实多智能体系统的通信拓扑不是完全图——line和star是最具挑战性的结构

### 损失函数 / 训练策略

检索后缀用梯度优化（GCG风格的离散token搜索），复制后缀用最大似然目标。两者串行优化。

## 实验关键数据

### 主实验

不同拓扑下的攻击成功率（%）：

| 方法 | Graph | Line | Star | 100智能体 |
|------|:---:|:---:|:---:|:---:|
| 单智能体基线 | ~40 | 20.69 | 19.19 | 32.25 |
| ARCJ (本文) | ~65 | **44.20** | **38.94** | **85.18** |
| 提升 | — | +23.51 | +18.95 | **+52.93** |

### 消融实验

各组件贡献：

| 配置 | Line ASR | Star ASR |
|------|:---:|:---:|
| 无后缀 | ~15 | ~12 |
| 仅检索后缀 | 20.69 | 19.19 |
| 检索+复制后缀 | **44.20** | **38.94** |

### 关键发现

1. **毒性消散是多智能体攻击的核心障碍**: 有毒后缀在传播中逐渐消失
2. **复制后缀贡献最大**: 从20.69%→44.20%（line），相比检索后缀贡献更大
3. **100智能体系统尤其脆弱**: 基线仅32.25%但ARCJ达85.18%——规模化反而更易攻击
4. **揭示多智能体架构的传染风险**: 独立记忆并不能保证安全

## 亮点与洞察

- **"毒性消散"现象的发现和命名**——精准描述了多智能体攻击失败的根本原因
- **"自我复制"思路独到**——让有毒信息像病毒一样在智能体间传播
- **100智能体实验规模**——首次在如此大规模下验证多智能体攻击
- **揭示独立记忆≠安全**的重要结论

## 局限与展望

- 仅在文本任务上验证，多模态多智能体系统未涉及
- 攻击者需要白盒访问来优化后缀
- 防御方法未充分探讨
- 社交拓扑结构较简单，真实网络拓扑更复杂

## 相关工作与启发

- **vs 单智能体记忆攻击（Chen et al. 2024）**: 仅攻击单体——本文扩展到多体传播
- **vs 共享记忆攻击（Ju et al. 2024）**: 共享记忆天然可传播——本文解决独立记忆的传播难题
- **vs GCG（Zou et al. 2023）**: 原始对抗后缀方法——本文加入复制能力
- **启发**: 多智能体系统的安全性需要从传播动力学角度思考

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 毒性消散发现+自我复制后缀设计非常创新
- 实验充分度: ⭐⭐⭐⭐ 多拓扑+多规模+消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，Town隐喻形象
- 价值: ⭐⭐⭐⭐ 揭示多智能体系统的传染风险

<!-- RELATED:START -->

## 相关论文

- [World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning](world_modeling_makes_a_better_planner_dual_preference_optimization_for_embodied_.md)
- [What Makes a Reward Model a Good Teacher? An Optimization Perspective](../../NeurIPS2025/llm_alignment/what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)
- [JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction](rewrite_to_jailbreak_discover_learnable_and_transferable_implicit_harmfulness_in.md)

<!-- RELATED:END -->
