---
title: >-
  [论文解读] GASP: Guided Asymmetric Self-Play For Coding LLMs
description: >-
  [ICLR 2026][LLM/NLP][非对称自博弈] 提出GASP框架，在非对称自博弈中引入"goalpost"（硬目标题）引导教师生成有针对性的训练问题，通过lemma（简化变体）→lift（加难变体）的课程结构逐步逼近困难目标…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "非对称自博弈"
  - "代码生成"
  - "课程学习"
  - "RLVR"
  - "目标引导"
---

# GASP: Guided Asymmetric Self-Play For Coding LLMs

**会议**: ICLR 2026  
**arXiv**: [2603.15957](https://arxiv.org/abs/2603.15957)  
**代码**: 无  
**领域**: LLM训练/代码推理  
**关键词**: 非对称自博弈, 代码生成, 课程学习, RLVR, 目标引导

## 一句话总结
提出GASP框架，在非对称自博弈中引入"goalpost"（硬目标题）引导教师生成有针对性的训练问题，通过lemma（简化变体）→lift（加难变体）的课程结构逐步逼近困难目标，在LiveCodeBench上超越无引导自博弈2.5%且解决了所有baseline无法解决的难题。

## 研究背景与动机

**领域现状**：非对称自博弈（如Absolute Zero/AZR）让LLM同时扮演教师（出题）和学生（解题），实现无人工数据的开放式训练。RLVR通过可验证奖励训练代码/数学推理能力。

**现有痛点**：现有自博弈是目标无关的——教师只关注学生的可学习性（题目不能太简单也不能太难），但不关注生成的题目是否"有趣"或"对下游任务有帮助"。结果是：很多在学习边界的难题对提升模型实际编程能力并不重要。

**核心矛盾**：自博弈需要探索困难问题来推进能力边界，但无引导的探索效率低——很多"难"题是人为构造的无意义难题，不代表真实编程挑战。

**本文目标**：(1) 能否用真实世界的难题引导自博弈？(2) 这种引导是否能提升下游编程能力？

**切入角度**：从训练集中筛选出RLVR训练后仍无法解决的硬题作为"goalpost"，教师被引导生成这些goalpost的简化版（lemma），再从lemma出发生成加难版（lift），形成逐步逼近的课程。

**核心 idea**：用真实难题做goalpost引导自博弈教师，通过lemma-lift踏脚石课程逐步突破能力边界。

## 方法详解

### 整体框架
GASP 想解决的是无引导自博弈"瞎探索"的问题：给自博弈的教师一个明确的攻坚目标（goalpost），让它生成的题目朝真实难题逼近，而不是凭空造一堆无意义的难题。整体是一个三阶段循环：先从训练集里筛出一批模型当前完全解不动的硬题当 goalpost；每一轮里，教师先围绕某个 goalpost 生成一道更简单的简化版（Lemma），再从这道简化版出发生成一道更难的加难版（Lift），用这一对踏脚石题去训练学生（Solver）。教师和学生共享同一套参数，只靠角色提示区分身份、同步更新。引导信号来自 146 道 goalpost 硬题——它们是从 601 道候选里筛出来、pass@100 仍为 0 的题。

### 关键设计

**1. Goalpost 筛选流水线：把"真正解不动的题"挑出来当攻坚靶子**

无引导自博弈的根本缺陷是没有方向，所以第一步要先定义"方向"——一批确实超出模型当前能力的硬题。GASP 用多阶段过滤来保证这一点：一道题要经过标准 RLVR 训练的所有检查点、AZR 检查点、以及额外的 RL 运行，全部跑下来 pass@100 仍然为 0，才会被选作 goalpost。这种三重过滤不是随便挑难题，而是要确保 goalpost 是 genuinely hard——既被反复验证确实超出当前边界，又来自真实训练集、和真实编程挑战相关，从而避免教师对着人为构造的无意义难题使劲。

**2. Lemma 生成：把遥不可及的硬题降到学生够得着的难度**

直接让学生啃 goalpost 是学不动的（pass@100=0 意味着完全没有正反馈），所以需要一块"踏脚石"。给定一个 goalpost $h$，教师生成它的简化变体 $\ell_0$，要求保留高层的算法主题，但把难度降到学生可学习的区间。控制难度落点的是奖励函数

$$r_{\text{lemma}} = [4p(1-p)]^5 \quad (0.3 \leq p \leq 0.7)$$

其中 $p$ 是学生在该题上的通过率，函数在 $p=0.5$ 处取峰值，把教师推向"中等难度"——既不会简单到没信息量，也不会难到学不动，同时仍与 goalpost 的主题相关。

**3. Lift 生成：从简化版往上加难，但故意不让教师看 goalpost**

只有 Lemma 还不够，课程得能往难处推，才能逼近 goalpost。于是教师从 Lemma $\ell_0$ 出发，生成一道更难的变体 $\ell_1$。这一步最关键的设计是教师**看不到原始 goalpost**，只能基于 $\ell_0$ 递增。奖励函数把难度落点压得更狠：

$$r_{\text{lift}} = 10p\left(\frac{1-p}{0.9}\right)^9 \quad (0.1 \leq p \leq 0.5)$$

峰值落在 $p=0.1$，鼓励教师生成更难的题。之所以刻意挡住 goalpost，是为了避免教师走捷径——如果让它直接对照目标，它会倾向于复制 goalpost 的表面特征；挡住之后，它只能从学生当前的能力边界一点点往上递增，让难度增长更贴合学习的渐进性。

**4. 难度轴：让加难沿一条可控的维度发生，而不是乱变**

为了让 Lemma 简化和 Lift 加难有章可循，GASP 把题目难度拆成两条正交的轴：I/O 轴改变输入输出的复杂度（比如把一个列表换成嵌套列表），f 轴改变算法本身的复杂度（比如增加约束或组合操作）。每生成一道 Lemma 时随机选定一条轴，随后的 Lift 就沿同一条轴继续加难——这样一对踏脚石题在"变难"时只动一个维度，难度变化是可解释、可累积的，而不是同时改一堆东西让课程失控。

### 损失函数 / 训练策略
训练用 Task-Relative REINFORCE++（沿用 AZR）。教师和学生共享参数、同步更新。任务分三类：Induction（最核心）、Deduction、Abduction，其中 Abduction 在 solver 阶段引入。

## 实验关键数据

### 主实验
LiveCodeBench v5 (Qwen2.5-Coder-7B)：

| 方法 | pass@1 | pass@20 | 说明 |
|------|--------|---------|------|
| Base model | 基线 | 基线 | 未训练 |
| RLVR (真实数据) | 好 | 好 | 上界参考 |
| AZR (无引导自博弈) | 中 | 中 | 无goalpost |
| GASP | **好** | **AZR+2.5%** | 有goalpost引导 |

### Goalpost进展

| 训练迭代 | 可解goalpost数 | 说明 |
|---------|-------------|------|
| 初始 | 0/146 | 全部无法解决 |
| RLVR | 0/146 | 标准RLVR仍无法解决 |
| AZR | 0/146 | 无引导自博弈仍无法解决 |
| GASP | **>0/146** | 部分goalpost被解决！ |

### 关键发现
- GASP在pass@20上超越AZR 2.5%，在大k时优势更大（说明课程增加了多样性）
- 最重要的是：GASP成功解决了所有baseline（RLVR/AZR）无法解决的部分goalpost题目
- 教师生成的lemma-lift课程质量随训练提升——后期lemma更接近goalpost难度
- 不给lift看goalpost很重要——直接给lift看goalpost导致教师复制表面特征而非递增式增难

## 亮点与洞察
- **目标引导自博弈**：在完全无监督的自博弈中引入外部"目标"信号，让教师的创造力有方向感。类似于RL中的goal-conditioned learning思想。
- **Lemma-Lift踏脚石**：不直接攻克难题，而是通过简化→逐步加难的课程逼近。这种curriculum设计思路可推广到其他领域的难题攻克。
- **"不给lift看goalpost"的巧妙设计**：强制教师从学生当前能力递增式增难，而非跳跃式复制目标，更符合学习的渐进性。

## 局限与展望
- 仅在代码领域验证，数学/通用推理领域的goalpost定义和效果未知
- Goalpost筛选依赖大量RL训练（多种子+多检查点），计算代价高
- Lemma-lift只有两级踏脚石，更长的课程链可能更有效
- 教师和学生共享参数限制了教师的出题能力，独立教师可能更好

## 相关工作与启发
- **vs AZR**: GASP在AZR基础上增加goalpost引导，证明了引导的价值。AZR是目标无关的，GASP有方向感。
- **vs SOAR**: SOAR用元学习rewarding教师，GASP更简单——不reward教师对goalpost的改善，解goalpost是课程学习的副产品
- **vs 标准RLVR**: RLVR用静态数据集，GASP自动生成新的训练数据且有方向引导

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 目标引导自博弈概念新颖，lemma-lift设计精巧
- 实验充分度: ⭐⭐⭐⭐ 与多个baseline比较，goalpost进展分析有说服力
- 写作质量: ⭐⭐⭐⭐ 动机清晰，算法描述详细
- 价值: ⭐⭐⭐⭐⭐ 对自博弈训练范式有重要推进，突破了无引导自博弈的天花板

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](../../NeurIPS2025/llm_nlp/acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)
- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](../../NeurIPS2025/llm_nlp/triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)
- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance-guided_token.md)
- [\[NeurIPS 2025\] SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models](../../NeurIPS2025/llm_nlp/space_noise_contrastive_estimation_stabilizes_self-play_fine-tuning_for_large_la.md)

</div>

<!-- RELATED:END -->
