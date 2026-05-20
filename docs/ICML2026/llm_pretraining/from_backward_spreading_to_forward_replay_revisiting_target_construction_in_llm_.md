---
title: >-
  [论文解读] From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing
description: >-
  [ICML 2026][预训练][参数编辑] 本文系统剖析了 locate-then-edit 编辑中 backward spreading 为什么能 work 又为什么 work 得不彻底，并提出 forward replay：把第一决定层作为优化变量、再通过标准前向传播得到后续各层 target…
tags:
  - "ICML 2026"
  - "预训练"
  - "参数编辑"
  - "MEMIT"
  - "多层协同"
  - "前向传播"
  - "hidden state"
---

# From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing

**会议**: ICML 2026  
**arXiv**: [2605.00358](https://arxiv.org/abs/2605.00358)  
**代码**: https://github.com/jugechengzi/FE (有)  
**领域**: LLM 知识编辑 / Locate-then-edit  
**关键词**: 参数编辑, MEMIT, 多层协同, 前向传播, hidden state

## 一句话总结
本文系统剖析了 locate-then-edit 编辑中 backward spreading 为什么能 work 又为什么 work 得不彻底，并提出 forward replay：把第一决定层作为优化变量、再通过标准前向传播得到后续各层 target，无需额外算力就能在 MEMIT/RECT/PRUNE/AlphaEdit 之上一致涨点。

## 研究背景与动机

**领域现状**：以 ROME / MEMIT 为代表的 locate-then-edit（LTE）范式已是 LLM 知识编辑主流：先用因果追踪定位关键 token 与几个 MLP 决定层，再求出末层的理想 hidden state $m_L$，最后求解闭式 rank-one 更新。为避免单层修改被正则项压制，主流做法把 $m_L$ 按 backward spreading 线性插值，分摊到更早的层（例如 $m_1 = h_1 + (m_3-h_3)/3$）。

**现有痛点**：backward spreading 的合理性长期没人系统检验。它隐含假设末层残差方向适用于所有更早的层，但前向动力学是非线性的，方向会随层旋转。BLUE 等近期方法尝试对首末两层独立优化，代价是算力翻倍且两层 target 在高维空间里不一定互相兼容。

**核心矛盾**：单层闭式解依赖准确的 $m_l$，多层协同则要求各层 $m_l$ 兼容；而 backward spreading 用一个方向去覆盖所有层，independent 优化又破坏兼容性。问题的根源在于 **target 的传播方向与模型真正的前向动力学不一致**。

**本文目标**：（1）形式化刻画 backward spreading 什么时候能用、什么时候失败；（2）提出一种与前向动力学天然一致、且不增加算力的多层 target 构造方式。

**切入角度**：既然把 $m_L$ 当可优化参数对末层做梯度下降时，反向计算图已经隐含了"前面各层为达到目标该走到哪"的信息，那么只要在**第一决定层**做优化、再做一次前向传播 replay，就能把这些信息显式恢复出来。

**核心 idea**：用 forward replay 替代 backward spreading——把 anchor point 从末层移到首层，其余各层 target 通过标准前向传播自然涌现，复杂度不变但跨层兼容性天然成立。

## 方法详解

### 整体框架
设要编辑的决定层为 $W_1,\dots,W_L$，关键 token 在 $W_l$ 前的 hidden state 为 $k_l$，目标 hidden state 为 $m_l$。LTE 的闭式解 $\Delta W^* = (M_I - W K_I) K_I^\top (K_I K_I^\top + \lambda K_J K_J^\top)^{-1}$ 依赖各层 $m_l$。MEMIT 只算 $m_L$，其余 $m_l$ 通过线性插值得到；本文方法 FE（Forward propagation Edit）只算 $m_1$，其余通过前向传播 replay 得到。两者整体 pipeline 相同，仅 target 构造方式不同。

### 关键设计

1. **Backward spreading 的理论刻画**:

    - 功能：解释 backward spreading 为什么能用、什么时候坏，为本文替代方案提供动机。
    - 核心思路：设 $\delta m_l = \Delta W_l k_l$ 是层 $l$ 的输出扰动，记 $\delta^l_{m_L} = J_{l\to L} \delta m_l$ 为它在末层引起的被动变化。理想要求 $\delta^l_{m_L} = \beta \delta m_l$，等价于 $\delta m_l$ 是 Jacobian $J_{l\to L}$ 的特征向量（Theorem 1，强条件，几乎不成立）。退一步只要求 $\cos(\delta^l_{m_L}, \delta m_l) > 0$，则等价于对称化 Jacobian $\frac{1}{2}(J_{l\to L} + J_{l\to L}^\top)$ 正定（Theorem 2，深层网络多数情况经验上成立）。在 Llama3-8B 上实测层 4 到层 8 的 cosine 从 $0.34$ 一路涨到 $1.00$，定量说明远层影响在末层方向越偏，是 backward spreading 难以无限叠层的根本原因。
    - 设计动机：先讲清旧方法为何"半好不坏"再设计替代方案，论证比直接换方法更扎实。

2. **Forward replay 构造多层 target**:

    - 功能：用一次前向传播得到所有决定层 mutually compatible 的 target hidden state。
    - 核心思路：MEMIT 把末层 $h_L$ 当可优化参数最小化交叉熵得到 $m_L$，再反推。FE 反过来——把**首层** $h_1$ 当可优化参数做同一目标的梯度下降得到 $m_1$；然后把 $m_1$ 接入模型继续标准前向传播，沿途记录每个决定层的 hidden state 作为该层的 $m_l$。这样得到的 $\{m_l\}$ 都来自同一条前向轨迹，天然兼容；同时由于反向梯度路径经过整张网络，$m_1$ 本身已经隐含了下游动力学的约束。
    - 设计动机：backward spreading 是用一个末层方向"硬塞"给前面各层；forward replay 是先让首层"知道末层想要什么"，再让网络自己把后续各层算出来。后者顺着模型的天然动力学走。

3. **即插即用的兼容性**:

    - 功能：让 FE 不止用于 MEMIT，还能给一大批 LTE 后续方法直接涨点。
    - 核心思路：FE 只替换 target 传播机制，不动闭式解、不动初始 $m$ 优化、不动正则项。因此 RECT（加 norm 正则）、PRUNE（裁剪奇异值）、AlphaEdit（null-space 投影）等都能直接拼上 "+FE" 后缀。
    - 设计动机：模型编辑社区分支极多，一个能与所有现有 pipeline 正交叠加的改进比另起炉灶更具传播力。

### 损失函数 / 训练策略
不引入新损失。优化 $m_1$ 时仍用交叉熵 $H(Y, Y_{\text{target}})$；得到 $m_1$ 后将其作为首决定层 hidden state，前向传播一次得到 $\{m_l\}_{l=2}^L$；最后用统一的 rank-one 闭式解逐层更新参数。计算复杂度与 MEMIT 完全相同。

## 实验关键数据

### 主实验
Llama3-8B-Instruct，按 EasyEdit 默认决定层 $[4,5,6,7,8]$，MCF 和 ZsRE 各编辑 2000 条知识，批量编辑。

| 数据集 | 方法 | Efficacy Success ↑ | Generalization Acc ↑ | DKL ↓ | Top-1 ↑ |
|--------|------|--------------------|------------------------|-------|---------|
| MCF | MEMIT | 97.4 | 48.0 | 0.41 | 77.5 |
| MCF | BLUE (2 层独立优化) | 98.4 | 60.1 | 0.35 | 81.0 |
| MCF | **MEMIT+FE** | **99.9** | **61.0** | **0.34** | **82.7** |
| ZsRE | MEMIT | 92.6 | 72.6 | 0.60 | 45.3 |
| ZsRE | BLUE | 94.6 | 78.2 | 0.18 | 66.7 |
| ZsRE | **MEMIT+FE** | **97.6** | **84.3** | **0.09** | **75.6** |

### 消融实验

| 配置 | MCF Efficacy Success | 说明 |
|------|-----------------------|------|
| OneLayer（只改末层） | 76.2 | 单层不够，多层协同必要 |
| MEMIT（backward, dividing） | 97.4 | 标准基线 |
| MEMIT（backward, no-dividing） | — | 末层残差 0.23 但首层反而远离目标 |
| MEMIT+FE | **99.9** | 各层 target 兼容，残差直接降到很低 |

Cosine 测量（Llama3-8B，layer 4→8）：$0.34, 0.41, 0.54, 0.72, 1.00$，定量说明远层方向漂移；这正是 backward spreading 在第 4 层贡献接近零的原因。

### 关键发现
- backward spreading 在浅层失效不是实现 bug 而是结构性问题：Jacobian 的旋转效应随深度差线性放大。
- FE 的提升不止来自 efficacy，还显著降低 DKL（MCF 0.41 → 0.34，ZsRE 0.60 → 0.09），说明编辑对无关知识的副作用更小——更准确的 target 让闭式解更贴目标方向，少误伤其他维度。
- 把 FE 套到 RECT/PRUNE/AlphaEdit 上同样一致涨点（AlphaEdit Efficacy 95.2 → 99.8），验证其作为通用插件的价值。

## 亮点与洞察
- 把"backward spreading 为何能用"这件大家默认的事，正式还原为"Jacobian 对称部分正定 + 步长足够小"两条假设；这种"先解释再替代"的写法比直接换方法更扎实。
- Forward replay 的核心洞察：在梯度反传时模型已经"算"过沿途所有层为达到目标该走到哪，只需做一次前向把它读出来——是非常优雅的"reuse 反传隐含信息"的思路。
- 不增加算力、不改其它组件的纯插件型改进，模型编辑社区任何 LTE 流水线几乎可以无痛接入。

## 局限与展望
- 仅在 MLP 决定层做实验，对注意力层是否同样适用未验证。
- 对 batch size 极大（同时编辑 $10^4$+ 条知识）的 scaling 表现没有专门评测；首层 $m_1$ 在更大批量下是否还稳定值得追踪。
- 当模型变得非常深（如 70B+）时，$m_1$ 到末层的前向轨迹本身可能逐渐失真，是否需要分段 anchor 是开放问题。
- 没有讨论与非 LTE 方法（hypernetwork、记忆库 WISE）的潜在结合方式。

## 相关工作与启发
- **vs MEMIT**：完全保留 MEMIT 闭式解和正则化，只换 target 来源；从 backward 改 forward，思路反转。
- **vs BLUE**：BLUE 独立优化首末两层但放弃中间层；FE 用一次优化得到所有层 target，cost 减半且 mutually compatible。
- **vs RECT / PRUNE / AlphaEdit**：这些方法改的是 $\Delta W$ 的正则或子空间，与 FE 改 target 来源完全正交，可叠加。
- **vs WISE / RLEdit**：那些方法走非 LTE 路线（额外记忆或 hypernetwork），本文不与之竞争，反而证明 LTE 路线还远未到上限。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 backward → forward 反转的思路简洁但确实是范式级改动。
- 实验充分度: ⭐⭐⭐⭐ 在两个数据集、三种模型、5+ 基线上做了全面对比，附录还有理论证明。
- 写作质量: ⭐⭐⭐⭐ "先剖析旧方法-再提出新方法"的结构清晰，Figure 1/2 对比直观。
- 价值: ⭐⭐⭐⭐⭐ 通用插件型改进，任何 LTE 方法都能直接受益，对模型编辑社区影响面广。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](../../ACL2026/llm_pretraining/forever_forgetting_curve-inspired_memory_replay_for_language_model_continual_lea.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)
- [\[ICML 2025\] Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning](../../ICML2025/llm_pretraining/revisiting_continuity_of_image_tokens_for_cross-domain_few-shot_learning.md)
- [\[ICML 2025\] Counting in Small Transformers: The Delicate Interplay between Attention and Feed-Forward Layers](../../ICML2025/llm_pretraining/counting_in_small_transformers_the_delicate_interplay_between_attention_and_feed.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)

</div>

<!-- RELATED:END -->
