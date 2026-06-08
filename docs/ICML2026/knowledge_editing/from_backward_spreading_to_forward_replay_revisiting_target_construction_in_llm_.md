---
title: >-
  [论文解读] From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing
description: >-
  [ICML 2026][知识编辑][参数编辑] 本文系统剖析了 locate-then-edit 编辑中 backward spreading 为什么能 work 又为什么 work 得不彻底，并提出 forward replay：把第一决定层作为优化变量、再通过标准前向传播得到后续各层 target…
tags:
  - "ICML 2026"
  - "知识编辑"
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
locate-then-edit 的难点不在闭式解本身，而在给每个决定层 $W_1,\dots,W_L$ 凑出一组互相兼容的目标 hidden state $m_l$。本文要做的就是把这组 target 的来源从"末层往前摊"换成"首层往后传"：先在第一决定层优化出 $m_1$，再让模型自己做一次前向传播把后续各层的 $m_l$ 算出来，闭式解部分完全不动。记关键 token 在 $W_l$ 前的 hidden state 为 $k_l$，各层共享同一个更新公式 $\Delta W^* = (M_I - W K_I) K_I^\top (K_I K_I^\top + \lambda K_J K_J^\top)^{-1}$，MEMIT 只算 $m_L$、其余靠线性插值，本文 FE（Forward propagation Edit）只算 $m_1$、其余靠 replay，整条 pipeline 唯一的区别就在这里。

### 关键设计

**1. 先把 backward spreading 的成立条件讲清楚：何时能用、何时崩**

替代一个被默认正确的旧做法之前，本文先正式刻画它为什么"半好不坏"。设 $\delta m_l = \Delta W_l k_l$ 是层 $l$ 的输出扰动，它经过 Jacobian 传到末层会变成 $\delta^l_{m_L} = J_{l\to L}\,\delta m_l$。backward spreading 暗含的理想假设是这个被动变化与原扰动同方向且仅差一个标量，即 $\delta^l_{m_L} = \beta\,\delta m_l$——Theorem 1 指出这等价于 $\delta m_l$ 恰是 $J_{l\to L}$ 的特征向量，是个几乎不可能成立的强条件。把要求放宽到只需方向不反，即 $\cos(\delta^l_{m_L}, \delta m_l) > 0$，Theorem 2 给出等价条件：对称化 Jacobian $\frac{1}{2}(J_{l\to L}+J_{l\to L}^\top)$ 正定，这在深层网络里经验上多数成立。问题出在"多数"二字——在 Llama3-8B 上实测层 4 到层 8 的 cosine 从 $0.34$ 一路涨到 $1.00$，说明层离末层越远、方向漂移越严重，浅层用末层方向去近似几乎贡献为零。这条线性放大的旋转效应，正是 backward spreading 无法无限叠层的结构性原因。

**2. Forward replay：让 target 顺着前向动力学自然涌现**

既然末层方向无法可靠地往浅层倒推，FE 干脆把 anchor point 从末层挪到首层。MEMIT 的做法是把末层 $h_L$ 当可优化参数、最小化交叉熵 $H(Y, Y_{\text{target}})$ 得到 $m_L$ 再反推；FE 反过来，把首层 $h_1$ 当可优化参数对同一目标做梯度下降得到 $m_1$，然后把 $m_1$ 接回模型继续标准前向传播，沿途把每个决定层路过的 hidden state 直接记下来当作该层的 $m_l$。这样得到的 $\{m_l\}$ 全部来自同一条前向轨迹，跨层兼容性天然成立、不需要额外约束；而且优化 $m_1$ 时反向梯度本就穿过整张网络，$m_1$ 已经隐含了下游各层"为达到目标该走到哪"的信息——forward replay 不过是把反传时算过一遍的东西用一次前向显式读出来，因此复杂度与 MEMIT 完全相同。一句话对比：backward spreading 是拿一个末层方向硬塞给前面各层，forward replay 是先让首层知道末层想要什么、再让网络自己把后续各层推出来。

**3. 只换 target 来源，保证对全套 LTE 流水线即插即用**

FE 刻意只替换 target 的传播机制，不碰闭式解、不碰初始 $m$ 的优化目标、也不碰正则项。这意味着任何在 $\Delta W$ 层面做文章的后续方法都能直接拼上 "+FE"：RECT 加的是 norm 正则、PRUNE 裁的是奇异值、AlphaEdit 投影到 null-space，三者改的都是更新量的形态，与 FE 改的 target 来源完全正交。模型编辑社区分支极多，一个能正交叠加在所有现有 pipeline 之上的改进，比另起炉灶的新方法传播力大得多。

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

- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)
- [\[ICML 2026\] CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing](crispedit_low-curvature_projections_for_scalable_non-destructive_llm_editing.md)
- [\[ACL 2025\] The Mirage of Model Editing: Revisiting Evaluation in the Wild](../../ACL2025/knowledge_editing/the_mirage_of_model_editing_revisiting_evaluation_in_the_wild.md)
- [\[ACL 2026\] CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](../../ACL2026/knowledge_editing/clare-ty_amid_chaos_quantifying_representational_entanglement_to_predict_ripple_.md)
- [\[ICML 2026\] Reverse-Engineering Model Editing on Language Models](reverse-engineering_model_editing_on_language_models.md)

</div>

<!-- RELATED:END -->
