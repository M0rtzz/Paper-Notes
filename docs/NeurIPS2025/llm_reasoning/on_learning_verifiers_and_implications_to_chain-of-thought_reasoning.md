---
title: >-
  [论文解读] On Learning Verifiers and Implications to Chain-of-Thought Reasoning
description: >-
  [NeurIPS 2025][LLM推理][验证器] 提出学习Chain-of-Thought验证器的形式化PAC框架，定义三种递进强度的验证目标（Simple → Trustable → γ-Trustable），证明当每个问题只有少量正确证明时样本复杂度为 $O(\log|H|)$…
tags:
  - "NeurIPS 2025"
  - "LLM推理"
  - "验证器"
  - "链式思维"
  - "PAC学习"
  - "样本复杂度"
  - "可信验证"
  - "计算复杂度差距"
---

# On Learning Verifiers and Implications to Chain-of-Thought Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2505.22650](https://arxiv.org/abs/2505.22650)  
**代码**: 无  
**领域**: LLM推理 / 计算学习理论  
**关键词**: 验证器, 链式思维, PAC学习, 样本复杂度, 可信验证, 计算复杂度差距

## 一句话总结

提出学习Chain-of-Thought验证器的形式化PAC框架，定义三种递进强度的验证目标（Simple → Trustable → γ-Trustable），证明当每个问题只有少量正确证明时样本复杂度为 $O(\log|H|)$，但当正确证明数量不受限时样本复杂度不可避免地跃升至 $\Theta(|H|)$，除非验证器类满足交集封闭性等额外结构假设；同时利用USAT问题证明验证与生成之间存在计算复杂度差距。

## 研究背景与动机

**领域现状**：CoT推理已成为LLM解决复杂数学和逻辑问题的主流范式，但推理链中常出现错误或无根据的推断步骤。形式化证明系统（如Lean）可以严格验证，但当前LLM难以直接生成形式化证明，甚至形式化一个非形式的问题陈述本身就很困难。

**现有痛点**：
- CoT推理存在"错误累积"问题——长推理链中细微错误难以检测
- 现有LLM验证器缺乏理论保证，不知道需要多少训练数据才能学到可靠的验证器
- 更关键的是：当验证器给出反馈后，推理模型会修改推理链重试，此时新推理链已经"分布外"（out-of-distribution），原验证器不再有保证

**核心矛盾**：最简单的验证目标（在同分布上正确）无法应对自适应使用场景——验证器的反馈本身导致分布偏移。需要一种更强的验证保证：对给定问题，拒绝**所有**错误推理链，而不仅仅是同分布的。

**本文目标** 为CoT验证器的学习建立严格的PAC学习框架，回答"学到可靠验证器需要多少数据"这一基本问题，并揭示不同验证强度目标下的样本复杂度本质差异。

**核心 idea**：验证器的可学习性取决于你要求它有多"可信"——从"同分布正确"到"拒绝一切错误证明"，所需样本量从对数级跳到线性级，但交集封闭结构可以打破这一壁垒。

## 方法详解

### 整体框架

论文定义了一个统一的验证框架：问题空间 $X$，推理步骤空间 $\Sigma$，验证器 $h: X \times \Sigma^* \to \{\text{YES}, \text{NO}\}$。给定问题 $x_0$ 和推理链 $\tau = (x_1, \ldots, x_T)$，验证器逐步检查每个前缀 $h(x_0, (x_1, \ldots, x_j))$，首次输出NO即定位第一个错误步骤。在此框架上定义三种递进强度的验证目标。

### 关键设计

1. **Simple Verification (SVPAC) — 简单验证**:

    - 功能：从随机标注的推理链样本中学习验证器，在同分布上以低错误率判断推理是否正确并定位第一个错误步骤
    - 核心思路：训练集由问题-推理链-标签三元组组成（标签含"正确"或"第一个错误位置"），学习器通过ERM选取与训练集一致的验证器 $h \in H$
    - 理论结果：有限 $H$ 时样本复杂度 $O(\frac{1}{\epsilon}(\log|H| + \log\frac{1}{\delta}))$；有限VC维时 $O(\frac{1}{\epsilon}(\text{VCdim}(H)\log T + \log\frac{1}{\delta}))$，与推理链长度 $T$ 仅对数依赖
    - 局限性：保证仅对同分布推理链有效。验证器给出反馈后推理模型修改推理链，新链已分布外——这正是引入Trustable验证的动机

2. **Trustable Verification (TVPAC) — 可信验证**:

    - 功能：对**大部分问题**，拒绝**任意**错误推理链（不限于同分布），同时接受所有gold standard正确证明
    - 核心思路：假设存在一个gold standard推理器 $g$，对每个问题 $x$ 提供 $\leq k$ 条正确推理链。利用 $g(x)$ 构建正例（正确链）和负例（正确链的单步偏离）训练验证器。关键是训练分布只在问题层面（$x \sim D$），对推理链保证是全局性的
    - 理论结果：样本复杂度与SVPAC相同量级——有限 $H$ 时 $O(\frac{1}{\epsilon}(\log|H| + \log\frac{1}{\delta}))$；有限VC维时 $O(\frac{1}{\epsilon}(\text{VCdim}(H)\log(kT|\Sigma|) + \log\frac{1}{\delta}))$
    - 额外发现：当 $k=1$ 时，stepwise TVPAC验证与CoT生成等价——好的验证器可以用来生成正确证明（逐步枚举 $\Sigma$ 中被接受的步骤）。但当 $|\Sigma|$ 很大时等价性失效

3. **γ-Trustable Verification (γ-TVPAC) — 松弛可信验证**:

    - 功能：当正确证明数量无上界时，从每个问题的随机正确证明（而非全部）中学习验证器，要求接受 $\geq \gamma$ 比例的正确证明且拒绝一切错误证明
    - 核心思路：Algorithm 1（交集一致验证器）——取所有与训练集一致的验证器的交集，只在所有一致验证器都说YES时才接受。这保证了零假阳率（永远不接受错误证明）
    - 理论结果：一般有限 $H$ 时样本复杂度 $O(\frac{|H|}{\eta\epsilon})$——从对数级跃升到线性级！下界证明这是不可避免的（Theorem 4.10, 4.11）
    - 突破口：若验证器类 $H$ 是交集封闭的，复杂度可回到 $O(\frac{\text{VCdim}(H)}{\eta\epsilon})$

### 损失函数 / 训练策略

- **SVPAC损失**：$\ell_h(x_0, \tau) = \mathbb{I}[h(x_0, \tau_j) \neq h^*(x_0, \tau_j) \text{ for some } j \leq \mathsf{f}(h^*, (x_0, \tau))]$ — 验证器必须在第一个错误步骤前与真实验证器完全一致
- **TVPAC**：ERM在由gold standard构建的正负样本上操作
- **γ-TVPAC**：取一致验证器的交集（Algorithm 1），不需要梯度优化，纯组合/集合操作

## 实验关键数据

本文为纯理论论文，无实验。以下汇总核心理论结果。

### 主要样本复杂度结果

| 验证目标 | 数据格式 | 有限 $H$ 复杂度 | 有限 VCdim 复杂度 | 学习算法 |
|---------|---------|----------------|------------------|---------|
| SVPAC (§3) | 随机(问题,推理链,首错位置) | $O(\frac{\log\|H\|}{\epsilon})$ | $O(\frac{\text{VCdim}(H)\log T}{\epsilon})$ | ERM |
| TVPAC (§4.1) | 随机问题 + $\leq k$ 条gold证明 | $O(\frac{\log\|H\|}{\epsilon})$ | $O(\frac{\text{VCdim}(H)\log(kT\|\Sigma\|)}{\epsilon})$ | ERM (含负例构造) |
| γ-TVPAC (§4.2) | 随机(问题,单条随机正确证明) | $O(\frac{\|H\|}{\eta\epsilon})$ | 交集封闭时 $O(\frac{\text{VCdim}(H)}{\eta\epsilon})$ | 一致验证器的交集 |
| Agnostic SVPAC (§C) | 同SVPAC | $O(\frac{\log\|H\|}{\epsilon^2})$ | $O(\frac{\text{VCdim}(H)\log T}{\epsilon^2})$ | ERM |

### 下界与不可能性结果

| 定理 | 条件 | 下界 | 含义 |
|------|------|------|------|
| Theorem 4.10 | proper learner, $\|\Sigma\| \geq 2$ | $\Omega(\|H\|)$ (仅要求soundness) | 即使只要求不接受错误证明，proper学习也需线性样本 |
| Theorem 4.11 | improper learner, $\frac{1}{2}$-complete | $\Omega(\|H\|)$ (要求soundness + completeness) | 即使允许输出 $H$ 外的验证器，加上completeness后线性下界仍成立 |

### 关键发现

- **复杂度跃迁**：从"少量正确证明"（TVPAC, $O(\log|H|)$）到"大量正确证明"（γ-TVPAC, $\Theta(|H|)$）存在指数级的样本复杂度跃迁，这是由"仅正例学习"的内在困难造成的
- **交集封闭性是关键结构**：若 $H$ 满足交集封闭性（如轴对齐超矩形、CNF公式等），γ-TVPAC的复杂度回到 $O(\text{VCdim}(H))$，避开线性下界
- **验证 vs 生成的计算差距**：利用USAT（唯一SAT）问题构造了一个具体例子——验证器可以多项式时间计算（检查赋值是否满足全部子句），但生成正确证明等价于求解USAT，在 $\text{RP} \neq \text{NP}$ 假设下无多项式时间算法
- **验证器可用于生成**：当 $k=1$ 且 $|\Sigma|$ 有限时，stepwise TVPAC验证器可以通过逐步枚举被接受步骤来生成正确证明——"好的验证器可以训练好的推理器"
- **Agnostic扩展**：将框架扩展到非realizable情况（$H$ 中无完美验证器），样本复杂度从 $O(1/\epsilon)$ 变为 $O(1/\epsilon^2)$，符合标准agnostic学习的规律

## 亮点与洞察

- **框架设计的层次感**：三种验证目标层层递进（同分布 → 全局soundness + 完全completeness → 全局soundness + 近似completeness），每层对应不同的实际场景需求和不同的理论难度
- **Theorem 4.9的证明技巧**：用 $2^{|H|}$ 个可能子集的union bound + 每个子集的独立样本概率衰减，建立了从一致验证器交集到样本复杂度的桥梁
- **USAT的计算差距例子**极其精妙：验证器只需检查赋值是否满足子句（P时间），但生成正确赋值需要求解USAT（NP-hard），完美对应了"验证比生成容易"的直觉
- **与CoT生成的等价性**（Remark 4.6）是意外发现——通常认为验证应比生成简单，但强验证器具有"引导"能力，可以用来生成证明。这为"用验证器训练推理器"提供了理论基础
- **交集封闭性**作为打破线性壁垒的结构条件，链接了经典PAC学习（closure algorithm）与新的验证框架

## 局限与展望

- **Realizability假设**：主要结果假设存在完美验证器 $h^* \in H$，agnostic扩展虽有但样本复杂度增加了 $O(1/\epsilon)$ 因子
- **问题分布假设**：Trustable验证器对推理链是分布无关的，但对问题仍依赖分布 $D$——换言之，在训练分布外的新类型问题上无保证
- **有限 $\Sigma$ 假设**：TVPAC要求推理步骤空间有限，限制了对连续/开放式自然语言推理的直接适用性
- **计算效率**：Algorithm 1（一致验证器交集）概念上简洁但计算上可能不可行——枚举 $H_S$ 中所有一致验证器并取交集对大 $|H|$ 不实际
- **交互式验证**：论文提出但未解决——是否存在交互式协议（验证器可向证明者提问）能绕过线性下界？
- **无限长推理链**：当推理链长度 $T$ 不受限时，多类分类中标签空间无限的可学习性是开放问题

## 相关工作与启发

- 与**JVB+ (2025)**的CoT生成理论关系密切：SVPAC的证明技术直接改编自他们的生成模型样本复杂度证明，但TVPAC验证器提供了更强的"分布外"保证，且agnostic setting在生成模型中是开放问题
- **closure algorithm**来自经典的one-sided error学习文献（Natarajan 1987, Kivinen 1995），被巧妙地重新包装为验证器学习的核心工具
- 与**Interactive Provers**（GRSY 2021, AGPR 2024）互补：他们研究如何让prover说服固定的verifier，本文研究如何从数据中学习好的verifier
- 对实践的启发：LightmanVerify-Step-by-Step (2024)等经验工作表明逐步验证优于整体验证，本文的stepwise vs overall soundness区分为此提供了理论基础
- "验证器可训练推理器"的发现可能启发新的训练范式——先学好验证器，再用验证器引导推理模型

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个系统性的CoT验证器PAC学习框架，三层验证目标的定义和 $O(\log|H|)$ 到 $\Theta(|H|)$ 的复杂度跃迁是全新的理论贡献
- 实验充分度: ⭐⭐ 纯理论论文无实验，但有丰富的具体例子（USAT、图路径、LLM验证器集合等）弥补了直观性
- 写作质量: ⭐⭐⭐⭐⭐ 定义清晰、定理陈述精确、证明简洁，从弱到强的递进结构读起来非常流畅
- 价值: ⭐⭐⭐⭐ 为CoT验证器的数据需求提供了基本理论指导，交集封闭性条件和验证-生成等价性有潜在的实践影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[NeurIPS 2025\] Value-Guided Search for Efficient Chain-of-Thought Reasoning](value-guided_search_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought](reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)
- [\[NeurIPS 2025\] Curriculum Abductive Learning](curriculum_abductive_learning.md)
- [\[NeurIPS 2025\] SafePath: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](safepath_preventing_harmful_reasoning_in_chain-of-thought_via_early_alignment.md)

</div>

<!-- RELATED:END -->
