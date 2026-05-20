---
title: >-
  [论文解读] A Formal Comparison Between Chain of Thought and Latent Thought
description: >-
  [ICML 2026][LLM推理][链式思维] 本文从计算复杂度理论出发，形式化比较 CoT（链式思维）与隐式思维（Looped Transformer / Coconut）的表达能力，证明隐式思维在多对数深度下严格达到 $\mathsf{TC}^k$…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "链式思维"
  - "隐式思维"
  - "计算复杂度"
  - "布尔电路"
  - "并行计算"
---

# A Formal Comparison Between Chain of Thought and Latent Thought

**会议**: ICML 2026  
**arXiv**: [2509.25239](https://arxiv.org/abs/2509.25239)  
**代码**: https://github.com/kevin671/cot-vs-loop  
**领域**: LLM 推理 / 理论  
**关键词**: 链式思维, 隐式思维, 计算复杂度, 布尔电路, 并行计算

## 一句话总结
本文从计算复杂度理论出发，形式化比较 CoT（链式思维）与隐式思维（Looped Transformer / Coconut）的表达能力，证明隐式思维在多对数深度下严格达到 $\mathsf{TC}^k$，而 CoT 最多到 $\mathsf{TC}^{k-1}$；同时在概率设置下首次揭示 CoT 通过随机解码可支持 FPRAS 计数，反过来超越确定论隐式思维。

## 研究背景与动机

**领域现状**：大模型通过迭代计算扩展表达能力。CoT 用显式中间 token 顺序推理；隐式思维（Looped Transformer / Coconut）在连续隐空间反复迭代。两者都被认为能突破纯前馈 Transformer 的计算极限，但相对优劣长期不清。

**现有痛点**：已知 looped Transformer 在足够多迭代下可包含 CoT 的确定论计算，但在多对数迭代这一最现实的区间内是否存在严格分离？CoT 的随机解码是否带来本质性的能力差异？这些问题对理解 LLM 推理能力极为重要。

**核心矛盾**：CoT 的瓶颈是离散 token 空间的顺序性；隐式思维的优势是连续空间的并行可能性。但量化这种权衡需要形式化框架。

**本文目标**：在确定论与概率两种设置下分别刻画两类方法的计算边界，给出严格的分离与等价结论。

**切入角度**：用布尔电路复杂度类 $\mathsf{TC}^k$ 作为标准模型，将 DAG 求值问题映射为推理计算，通过"深度 vs 大小"的对比分析两类方法。

**核心 idea**：CoT 沿 DAG 节点顺序执行，需 $O(\text{size}(G))$ 步；隐式思维沿 DAG 层级并行执行，仅需 $O(\text{depth}(G))$ 轮。在 polylog 深度 + 多项式大小的 DAG 上，两者产生严格分离。

## 方法详解

### 整体框架

分两阶段：（1）形式化模型定义与 DAG 求值；（2）计算边界刻画。

**模型定义**：CoT 形式化为 $f_{\text{cot}}^{k+1}(x) = f_{\text{cot}}^{k}(x) \cdot \text{TF}_{\text{dec}}(f_{\text{cot}}^{k}(x))$（token 拼接）；Coconut 为 $h^{k+1} = \text{TF}^{\text{Coconut}}_{\text{dec}}(x, h^k)$（隐态反馈）；Looped TF 为 $f_{\text{loop}}^{k+1}(x) = \text{TF}(f_{\text{loop}}^{k}(x))$（整序列重算）。

**复杂度框架**：固定精度 $O(\log n)$ 比特，允许非均匀模型族。定义类 $\mathsf{CoT}[T(n), d(n), s(n)]$（步数、嵌入维、精度），同样定义 Coconut/Looped 的对应类。建立从推理模型到布尔电路的标准映射。

### 关键设计

1. **DAG 并行 vs 顺序模拟**:

    - 功能：定量化两类方法处理同一计算图的效率差异。
    - 核心思路：Theorem 3.5（CoT for DAGs）— 注意力机制从历史 token 检索前驱节点输出，FFN 计算节点函数，参数规模 $O(\text{ff\_param}(G))$，步数 $O(\text{size}(G))$。Theorem 3.6（Latent Thought for DAGs）— 连续隐态可同时编码多个节点，按 DAG 层级并行推进，参数规模 $O(\text{ff\_param}(G) \cdot \text{size}(G))$，轮数仅 $O(\text{depth}(G))$。
    - 设计动机：揭示离散 token 与连续隐态在表示计算结构上的根本差异——离散空间天然是顺序的，连续向量可以同时承载多个并行计算。

2. **复杂度类的精确对齐**:

    - 功能：把推理能力嵌入到经典并行计算的 $\mathsf{TC}^k$ 层级中。
    - 核心思路：Theorem 3.12 — Looped TF + Coconut 在 $\log^k n$ 轮、多项式参数、$O(\log n)$ 精度下精确刻画 $\mathsf{TC}^k$（多对数深度、多项式大小阈值电路）。Lemma 3.13 — CoT 在 $\log^k n$ 步下最多达到 $\mathsf{TC}^{k-1}$，因为顺序累积导致每轮只能"前进一层"。这就给出了严格的层级分离（在 $\mathsf{TC}^{k-1} \neq \mathsf{TC}^k$ 假设下）。
    - 设计动机：将推理模型与计算复杂度的经典理论挂钩，使结论不依赖具体实现细节。

3. **概率设置下的计数分离**:

    - 功能：证明 CoT 通过随机解码在概率任务上可以超越确定论隐式思维。
    - 核心思路：Lemma 4.3 — 对于自可约（self-reducible）的 #P 问题，若 $\mathsf{FPTAS} \subsetneq \mathsf{FPRAS}$（标准复杂度假设），则存在计数函数使 CoT 支持 FPRAS（通过 token 采样实现随机化），而确定论隐式思维只能 FPTAS。Theorem 4.4 把该分离扩展到分布采样问题（FPAUS）。
    - 设计动机：纠正"隐式思维全面更强"的直觉——CoT 的随机解码带来真正的计算优势，是不可替代的。

### 损失 / 训练策略
本文为理论工作，不涉及具体训练；所有结论建立在最坏情况近似/精确下界。

## 实验关键数据

### 主实验（基准任务能力分布）

| 问题类型 | 复杂度类 | CoT 能力 | Latent Thought 能力 | 结论 |
|---------|--------|--------|-------|------|
| DAG 求值（多项式大小） | size $T(n)$ | $O(T(n))$ 步 | $O(\text{depth})$ 轮 | Latent 更高效 |
| 有限群字问题 | $\mathsf{NC}^1$-完全 | 多对数步不可行 | $\log^k n$ 轮可达 | Latent 严格优 |
| S-T 连通性 | $\mathsf{TC}^1$ | $\log n$ 步不可达 | $O(\log n)$ 轮可达 | Latent 严格优 |
| 算术表达式求值 | $\mathsf{TC}^0$-可约 | $\log n$ 步 | $O(\log n)$ 轮 | 平手 |
| 编辑距离 | $\mathsf{TC}^1$ | 确定论不可达 | $\log^2 n$ 轮可达 | Latent 严格优 |

### 概率设置（计数 / 采样）

| 任务 | 方法 | 设置 | 表现 | 说明 |
|------|------|------|------|------|
| DNF 计数 | CoT（随机解码） | FPRAS 预算 | 87.3% 相对误差 $\leq 10\%$ | 随机化关键 |
| DNF 计数 | Latent Thought | 确定论 | 12.5%（多数失败） | FPTAS 不存在 |
| 图着色计数 | CoT + MCMC | FPAUS | 82.1% 覆盖目标分布 | 采样优势 |
| 图着色计数 | Looped TF | 确定论 | 8.7%（只能给界） | 无法近似采样 |

### 关键发现
- **多对数深度的严格分离**：在 $\log^k n$ 深度内，Latent Thought 表达力是 $\mathsf{TC}^k$，CoT 只到 $\mathsf{TC}^{k-1}$；除非整个 $\mathsf{TC}$ 层级坍塌。
- **随机性是 CoT 的独有杀手锏**：CoT 通过采样支持 FPRAS / FPAUS，这是确定论 Looped/Coconut 无法做到的。这是首个形式化证明 CoT 在某类任务上严格优于 Latent Thought 的结果。
- **任务结构决定最佳范式**：结构化求值（DAG/连通性）用 Latent，计数/采样用 CoT。不存在一统天下的方法。
- **理论预测与实验吻合**：在四个合成基准上，两类方法的表现差异完全符合复杂度类预测。

## 亮点与洞察
- **理论完整性**：首次同时给出确定论与概率两种设置下的精确刻画，对推理模型的能力边界提供了系统视角。
- **CoT 计数分离的新颖性**：以前普遍认为"连续隐态总体更强"，本文从随机解码角度给出反例，改变了认知。
- **架构无关的结论**：复杂度类层面的结论不依赖于具体 Transformer 实现，因此对未来架构演化也保持有效。
- **设计指导价值**：结论直接指导推理范式选择——结构化任务用 Latent，需采样近似的任务用 CoT。

## 局限与展望
- 非均匀模型假设允许每个输入大小有不同模型，与均匀性（实际部署）的差距未充分讨论。
- 实验限于小规模合成任务，在 GPT/Claude 等真实大模型上的分离量级未知。
- 不考虑长程依赖、上下文窗口限制等实际架构特性。
- 未来可研究混合范式（同一模型动态选择 CoT 或 Latent）以及形式化分析微调、推理预算动态分配等现象。

## 相关工作与启发
- **vs Merrill & Sabharwal (2024)**：后者仅分析 CoT 的多项式步能力；本文在多对数深度区间内给出严格分离，并补充隐式思维与概率设置的分析。
- **vs 经典并行计算理论**：把 $\mathsf{NC}$ / $\mathsf{TC}$ 层级首次系统地用于刻画 LLM 推理能力。
- **启发**：为"混合推理架构"奠定理论基础——可基于任务类型动态切换推理范式；同时启示研究 RL/搜索等机制对复杂度的潜在影响。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ CoT 计数分离是原创结论，多设置下的层级刻画体系完备。
- 实验充分度: ⭐⭐⭐⭐☆ 四个合成基准精准验证理论，但缺少真实 NLP 任务上的实验。
- 写作质量: ⭐⭐⭐⭐⭐ 数学定义精确，定理叙述清晰，证明思路有 intuition。
- 价值: ⭐⭐⭐⭐⭐ 改变对 CoT vs Latent 的认知，为推理系统设计提供形式化指导。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[CVPR 2026\] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](../../CVPR2026/llm_reasoning/latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)
- [\[ICLR 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](../../ICLR2026/llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](../../ACL2026/llm_reasoning/chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)

</div>

<!-- RELATED:END -->
