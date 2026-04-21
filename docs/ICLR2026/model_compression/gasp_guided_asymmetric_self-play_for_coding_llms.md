---
title: >-
  [论文解读] GASP: Guided Asymmetric Self-Play For Coding LLMs
description: >-
  [ICLR 2026][模型压缩][非对称自博弈] 提出GASP框架，在非对称自博弈中引入"goalpost"（硬目标题）引导教师生成有针对性的训练问题，通过lemma（简化变体）→lift（加难变体）的课程结构逐步逼近困难目标，在LiveCodeBench上超越无引导自博弈2.5%且解决了所有baseline无法解决的难题。
tags:
  - ICLR 2026
  - 模型压缩
  - 非对称自博弈
  - 代码生成
  - 课程学习
  - RLVR
  - 目标引导
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
三阶段循环训练：Phase 1 (Lemma生成) → Phase 2 (Lift生成) → Phase 3 (Solver训练)。教师和学生共享参数，通过角色提示区分。146个goalpost硬题（从601题中筛选，pass@100=0）作为引导信号。

### 关键设计

1. **Goalpost筛选流水线**:

    - 功能：多阶段过滤找到真正的硬题——标准RLVR训练所有检查点、AZR检查点、额外RL运行后仍pass@100=0的题
    - 核心思路：三重过滤确保goalpost确实在模型当前能力范围之外
    - 设计动机：goalpost必须是"genuinely hard"——已知困难且与真实编程挑战相关

2. **Lemma生成（简化变体）**:

    - 功能：给定goalpost $h$，教师生成更简单的变体 $\ell_0$，保留高层算法主题
    - 核心思路：奖励函数 $r_{\text{lemma}} = [4p(1-p)]^5$（当 $0.3 \leq p \leq 0.7$ 时），峰值在 $p=0.5$，鼓励中等难度
    - 设计动机：lemma在学生可学习范围内但仍有挑战性，且与goalpost主题相关

3. **Lift生成（加难变体）**:

    - 功能：从lemma $\ell_0$ 出发生成更难的变体 $\ell_1$，但**不看原始goalpost**
    - 核心思路：奖励 $r_{\text{lift}} = 10p(\frac{1-p}{0.9})^9$（$0.1 \leq p \leq 0.5$），峰值在 $p=0.1$，鼓励更难的题
    - 设计动机：不让教师看到goalpost是刻意设计——避免简单复制表面特征，鼓励从学生当前边界递增式增加难度

4. **难度轴**:

    - I/O轴：改变输入输出的复杂度（如一个列表→嵌套列表）
    - f轴：改变算法复杂度（如增加约束或组合操作）
    - 每次lemma随机选一个轴，lift沿同一轴增加难度

### 损失函数 / 训练策略
- Task-Relative REINFORCE++ (来自AZR)
- 教师和学生共享参数，同步更新
- 三种任务类型：Induction(最核心)、Deduction、Abduction（solver阶段引入）

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

## 相关论文

- [ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](../../AAAI2026/model_compression/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [SeeDNorm: Self-Rescaled Dynamic Normalization](seednorm_self-rescaled_dynamic_normalization.md)
- [Reference-Guided Machine Unlearning](reference-guided_machine_unlearning.md)
- [Draft-based Approximate Inference for LLMs](draft-based_approximate_inference_for_llms.md)
- [Highly Efficient and Effective LLMs with Multi-Boolean Architectures](highly_efficient_and_effective_llms_with_multi-boolean_architectures.md)

<!-- RELATED:END -->
