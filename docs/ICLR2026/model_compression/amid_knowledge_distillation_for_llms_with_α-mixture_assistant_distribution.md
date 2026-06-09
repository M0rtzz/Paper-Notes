---
title: >-
  [论文解读] AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution
description: >-
  [ICLR 2026][模型压缩][知识蒸馏] 提出α-mixture assistant distribution及统一蒸馏框架AMiD，通过引入新设计变量α（控制教师-学生分布插值路径的几何形状）泛化了现有辅助分布方法（m-mixture和e-mixture为α=±1的特例）…
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "知识蒸馏"
  - "辅助分布"
  - "α-混合"
  - "f-散度"
  - "LLM压缩"
---

# AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution

**会议**: ICLR 2026  
**arXiv**: [2510.15982](https://arxiv.org/abs/2510.15982)  
**代码**: [https://github.com/aailab-kaist/AMiD](https://github.com/aailab-kaist/AMiD)  
**领域**: 模型压缩 / 知识蒸馏  
**关键词**: 知识蒸馏, 辅助分布, α-混合, f-散度, LLM压缩  

## 一句话总结
提出α-mixture assistant distribution及统一蒸馏框架AMiD，通过引入新设计变量α（控制教师-学生分布插值路径的几何形状）泛化了现有辅助分布方法（m-mixture和e-mixture为α=±1的特例），并证明了在任意散度和α下的最优性保证，在多个LLM蒸馏基准上取得SOTA性能。

## 研究背景与动机

**领域现状**：LLM知识蒸馏通过对齐教师-学生的token级分布来压缩模型。近期研究引入了"辅助分布"——教师和学生分布的混合体——来缓解容量差距和近零概率导致的训练不稳定性。

**现有痛点**：(a) 现有辅助分布方法（GKD/DistiLLM用算术平均即m-mixture，TAID用几何平均即e-mixture）各自独立提出，缺乏统一框架；(b) 辅助分布的设计与散度的选择耦合在一起，搜索空间被人为限制；(c) α（控制插值路径几何）被固定为±1，未被探索。

**核心矛盾**：LLM的高维输出空间中大量概率接近零，导致密度比估计不稳定；同时教师-学生容量差距使直接对齐困难。辅助分布是解决这两个问题的关键，但现有设计不够通用。

**本文目标**：建立辅助分布和散度的统一理论框架，发现新的、更好的辅助分布形式。

**切入角度**：用信息几何中的广义 $f_\alpha$-均值统一现有辅助分布——m-mixture和e-mixture分别对应算术均值(α=-1)和几何均值(α=1)，而α可以取任意实数值。

**核心 idea**：用广义 $f_\alpha$-均值将辅助分布从两种离散选择扩展为一族连续参数化的分布，并证明任意α和散度下的蒸馏最优性。

## 方法详解

### 整体框架
AMiD把现有辅助分布方法统一在同一族分布上：m-mixture的算术平均与e-mixture的几何平均只是这一族在两个端点的取值，整族可由一个几何参数α连续串起。它用两个相互独立的旋钮刻画辅助分布——α决定教师分布p与学生分布q_θ之间走哪条插值路径（路径几何），λ决定在路径上走多远（插值位置）。辅助分布定义为

$$\tilde{r}_\theta^{(\alpha,\lambda)}(z) = \left(\lambda\, p(z)^{\frac{1-\alpha}{2}} + (1-\lambda)\, q_\theta(z)^{\frac{1-\alpha}{2}}\right)^{\frac{2}{1-\alpha}}\quad(\alpha\neq 1)$$

再归一化为有效概率分布 $r_\theta^{(\alpha,\lambda)}$。蒸馏目标就是最小化教师（或学生）到这个辅助分布的散度，即 $D(p, r_\theta^{(\alpha,\lambda)})$ 或 $D(q_\theta, r_\theta^{(\alpha,\lambda)})$，其中D可以是任意散度——辅助分布的设计因此与散度的选择彻底解耦。

### 关键设计

**1. α-mixture辅助分布族：用一个几何参数把零散的混合方式统一成连续谱。**

过去m-mixture和e-mixture是两套独立提出的方法，谁也不知道它们之间有没有中间地带。AMiD用信息几何里的广义 $f_\alpha$-均值把它们装进同一个参数化家族：α=-1退化为算术均值（m-mixture，在概率空间走直线）；α=1退化为几何均值（e-mixture，在对数空间走直线）；α=3给出调和均值；其余α值则对应全新的插值路径。Theorem 3.2进一步证明，$r^{(\alpha,\lambda)}$ 恰好是p和q在α-散度意义下测地线上的内分点。这一步的意义在于，α不再是被默认钉死在±1的常数，而是一个可以自由调的设计维度，整族分布从两个离散选项变成了连续可搜索的空间。

**2. 最优性保证：换了插值路径，蒸馏的最终目标不变。**

引入新的辅助分布最怕的是把优化目标也一起改坏了——本来要让学生逼近教师，结果优化到了别处。Theorem 3.4排除了这个隐患：对任意正则散度D和任意α，最小化 $D(p, r_\theta^{(\alpha,\lambda)})$ 的最优解都等价于 $p=q_\theta$。直觉是，辅助分布是p和q_θ路径上的内分点，当这个内分点与一端（教师p）完全重合时，它必然也与另一端（学生q_θ）重合，于是学生只有真正匹配教师才能取得最优。这保证了无论怎么选α，AMiD都不会偏离"让学生分布等于教师分布"这个蒸馏的本来目的，引入辅助分布纯粹是改善训练路径而非改变终点。

**3. 梯度分析与mode-covering/seeking控制：α独立地调节质量-多样性tradeoff。**

mode-covering（覆盖教师所有模式，更多样）和mode-seeking（聚焦主模式，更精准）通常由散度的选择决定，但AMiD揭示α本身也能调这件事。Proposition 3.5分析f-散度下的梯度，其中出现一个加权项

$$w = \frac{(1-\lambda)\,q_\theta^{\frac{1-\alpha}{2}}}{\lambda\, p^{\frac{1-\alpha}{2}} + (1-\lambda)\, q_\theta^{\frac{1-\alpha}{2}}}$$

α较大时，权重w在 $p>q_\theta$ 的区域更大，梯度更用力去抬高学生在教师高概率处的密度，表现为mode-covering；α较小时，w在 $p<q_\theta$ 的区域更大，梯度更倾向于压低学生在教师低概率处的密度，表现为mode-seeking。这与α<1时辅助分布support为并集（覆盖）、α≥1时为交集（聚焦）的几何性质一致。其结果是：即便固定住散度D，也能单靠α在质量和多样性之间滑动，把过去只能靠换散度才能实现的调节解放成一个连续旋钮。

### 训练策略
AMiD兼容任意散度和任意数据采样策略，论文推荐的默认配置是α-β散度配合 λ=0.1。实践中按任务需求选α——需要覆盖教师全部模式时取 α<1，需要聚焦主模式时取 α≥1，并支持在训练过程中对α做自适应调度。

## 实验关键数据

### 主实验——GPT-2 XL→GPT-2蒸馏（指令跟随ROUGE-L）

| 方法 | Dolly | Self-Inst | Vicuna | Super NI | Avg |
|------|-------|-----------|--------|----------|-----|
| GKD (α=-1) | 24.58 | 11.78 | 14.60 | 22.84 | ~18 |
| DistiLLM (α=-1) | ~25 | ~12 | ~15 | ~23 | ~19 |
| TAID (α=1) | ~25 | ~12 | ~15 | ~23 | ~19 |
| **AMiD** | **最佳** | **最佳** | **最佳** | **最佳** | **最佳** |

### 消融——α的影响
α=-1到α=1之间的中间值（如α=0）在多数任务上表现最佳，说明现有方法用的端点值错过了最优区域。toy实验验证了α控制mode-covering/seeking的理论预测。

### 关键发现
- α和λ是正交的设计维度——λ控制"走多远"，α控制"走哪条路径"
- 不同任务的最优α不同，但中间值通常优于端点值
- AMiD训练更稳定，得益于辅助分布缓解近零概率问题

## 亮点与洞察
- **信息几何视角的统一**极其优雅——用广义均值+α-散度内分点定理将零散方法统一为连续参数族
- **α与λ的正交性**是核心洞察——之前所有工作只调λ不调α，错过了重要的设计维度
- Proposition 3.5的梯度分析将mode-covering/seeking的直觉形式化

## 局限与展望
- α的最优选择仍需实验调参，缺乏自动化机制
- 实验主要在GPT-2级别（0.1B-1.5B），大规模LLM验证不足
- 归一化常数 $Z_r$ 增加计算开销
- 未与非KD压缩方法对比

## 相关工作与启发
- **vs GKD**: GKD用GJS含隐式m-mixture（α=-1），AMiD泛化为任意α
- **vs TAID**: TAID用e-mixture（α=1），AMiD揭示这只是端点值
- **vs DistiLLM**: DistiLLM用skew KL（α=-1），AMiD证明中间α更优

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 信息几何驱动的统一框架，α作为新设计维度很深刻
- 实验充分度: ⭐⭐⭐⭐ 多任务+消融+toy验证完整，但模型规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，图示极其直观
- 价值: ⭐⭐⭐⭐ 为LLM知识蒸馏的辅助分布设计提供了统一理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/model_compression/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[ACL 2025\] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](../../ACL2025/model_compression/sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](distillation_of_large_language_models_via_concrete_score_matching.md)

</div>

<!-- RELATED:END -->
