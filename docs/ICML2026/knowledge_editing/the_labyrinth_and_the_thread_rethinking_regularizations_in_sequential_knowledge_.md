---
title: >-
  [论文解读] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models
description: >-
  [ICML 2026][知识编辑][序列知识编辑] 本文从优化角度证明：序列编辑（SE）之所以稳定，本质是"累积更新等价于一次性编辑（OTE）的解"，而 AlphaEdit 的零空间投影、PRUNE/RECT 的后处理正则等花哨机制并非关键——只要保证 OTE-SE 对齐…
tags:
  - "ICML 2026"
  - "知识编辑"
  - "序列知识编辑"
  - "AlphaEdit"
  - "零空间投影"
  - "OTE-SE 等价"
  - "正则化必要性"
---

# The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models

**会议**: ICML 2026  
**arXiv**: [2605.26670](https://arxiv.org/abs/2605.26670)  
**代码**: https://github.com/Wangzzzzzzzz/OTE-SE-Alignment (有)  
**领域**: 知识编辑 / LLM  
**关键词**: 序列知识编辑, AlphaEdit, 零空间投影, OTE-SE 等价, 正则化必要性

## 一句话总结
本文从优化角度证明：序列编辑（SE）之所以稳定，本质是"累积更新等价于一次性编辑（OTE）的解"，而 AlphaEdit 的零空间投影、PRUNE/RECT 的后处理正则等花哨机制并非关键——只要保证 OTE-SE 对齐，去掉这些正则也能在 4 个主流 LLM 上稳定完成 2000 步序列编辑。

## 研究背景与动机

**领域现状**：结构化知识编辑要在不重训的前提下，把 LLM 中以 $(s,r,o)$ 三元组形式存储的事实精准改掉。主流走 locate-and-edit 路线：先定位事实所在的 FFN 输出投影 $\mathbf{W}$，再对该层做受约束的最小二乘更新。为支持"持续/序列编辑"，近几年涌现了 AlphaEdit（零空间投影）、PRUNE（限制更新谱）、RECT（限制相对权重变化）、MEMIT（约束最优化）等一堆机制，越叠越复杂。

**现有痛点**：每个方法都自带一套正则/约束，并各自声称是稳定性的关键，但缺乏统一理论。尤其 AlphaEdit 把"上千步稳定编辑"几乎归功于零空间投影 $\mathbf{P}$（满足 $\mathbf{K}_0^\top \mathbf{P}=\mathbf{0}$），这个解释从未被严格检验；新方法只能继续往上堆约束，设计空间像迷宫一样越来越乱。

**核心矛盾**：现有解释把"稳定性"和"某个特定正则机制"绑死了，但没人回答更基本的问题——序列编辑稳定的真正充要条件是什么？如果零空间投影真是核心，为什么去掉对历史知识的约束（只保留当前一条）后，加上 $\mathbf{P}$ 反而模型立刻崩溃？

**本文目标**：(1) 证伪"零空间投影是 AlphaEdit 成功的核心"；(2) 给出一条统一的、独立于具体正则的稳定 SE 设计准则；(3) 评估各类正则在该准则下的真实必要性。

**切入角度**：把所有 locate-and-edit 都写成同一个普通最小二乘（OLS）问题，然后比较"分 $T$ 步累积"和"一次性批量"两个解。如果数学上恒等，那么"是否使用某种正则"就和"是否稳定"解耦了。

**核心 idea**：稳定性 = SE 累积更新严格重建 OTE 闭式解；零空间投影只是恰好满足这个等价性的一种实例，不是充分必要条件。

## 方法详解

### 整体框架
作者把所有 locate-and-edit 方法都改写成同一个带正则的最小二乘问题——同时拟合保留集 $(\mathbf{K}_0,\mathbf{V}_0)$ 和当前编辑集 $(\mathbf{K}_t,\mathbf{V}_t)$，在 $\mathbf{W}+\boldsymbol{\Delta}$ 上求闭式解。关键转换是：与其每步"拍脑袋加一套正则"，不如先写出一次性编辑（OTE）想要的目标，再把第 $t$ 步真正要执行的增量定义成相邻两步 OTE 闭式解之差 $\tilde{\boldsymbol{\Delta}}_t := \boldsymbol{\Delta}^*_{\text{total},t} - \boldsymbol{\Delta}^*_{\text{total},t-1}$。这一个差分定义把"稳定"从"用了哪种正则"中解耦出来，重新表述为"序列累积是否严格重建 OTE 解"。顺着这条线索，全文回答三个递进问题，正好对应下面三个关键设计：先证伪"零空间投影是 AlphaEdit 成功的核心"、确立 OTE-SE 等价这条统一判据（RQ1）；再给出从任意带正则 OTE 目标机械构造稳定 SE 算法的差分映射（RQ2）；最后为 PRUNE/RECT 这类后处理正则补上可解析的误差补偿（RQ3）。这是一篇纯优化理论分析，方法即一套闭式解与等价性证明，没有可画的多阶段 pipeline。

### 关键设计

**1. 零空间证伪 + OTE-SE 等价定理：先打掉"机制崇拜"，再立统一判据**

为了正面回答"AlphaEdit 凭什么稳"，作者设计了一个最小可证伪场景 Memorize-the-Latest：每步只要求记住当前这一条事实、丢掉对历史保留集的约束，但仍保留零空间投影 $\mathbf{P}$（满足 $\mathbf{K}_0^\top\mathbf{P}=\mathbf{0}$）。在 LLaMA-3 + CounterFact 上模型语言能力立刻归零（GLUE 全 0.000），直接证明 $\mathbf{P}$ 单独撑不起稳定性。深挖原因：原 AlphaEdit 法方程含 $\mathbf{K}_0\mathbf{K}_0^\top\mathbf{P}$ 和 $(\mathbf{V}_0-\mathbf{W}_{t-1}\mathbf{K}_0)\mathbf{K}_0^\top\mathbf{P}$ 两项，展开后累积偏差恰好等于 $-\sum_{\tau=1}^{t}\boldsymbol{\Delta}_\tau^*\mathbf{K}_0\mathbf{K}_0^\top\mathbf{P}$，随编辑步数线性放大，只有当历史约束被显式吸收时这项才抵消。在此基础上 Lemma 3.1 证明 AlphaEdit 严格满足 $\sum_{\tau=1}^t \boldsymbol{\Delta}_\tau^* = \boldsymbol{\Delta}^*_{\text{total},t}$，Proposition 3.2 进一步说明对任意 $\mathbf{P}$、$\lambda\geq 0$ 的广义形式都成立（MEMIT 是 $\mathbf{P}=\mathbf{I},\lambda=0$ 的特例）。于是"稳定性"被重新定义为一条独立于正则形式的物理判据——累积更新重建 OTE，而非依附某个特定机制。

**2. OTE→SE 构造性映射：把"加正则"换成"求差分"**

有了等价判据，作者给出从任意带凸正则 $\mathcal{R}$ 的 OTE 目标机械化构造稳定 SE 算法的流水线。核心仍是差分式更新 $\tilde{\boldsymbol{\Delta}}_t := \boldsymbol{\Delta}^*_{\text{total},t} - \boldsymbol{\Delta}^*_{\text{total},t-1}$：Proposition 3.4 证明，只要损失 $\mathcal{L}_t$ 满足"平移二次可表示性"（shifted quadratic representability，最小二乘损失天然满足），这个差分就是凸子问题 $\arg\min_{\boldsymbol{\Delta}} \ell_t(\boldsymbol{\Delta}) + \langle \nabla\mathcal{L}_t(\boldsymbol{\Delta}^*_{\text{total},t-1}),\boldsymbol{\Delta}\rangle + \mathcal{R}(\boldsymbol{\Delta}^*_{\text{total},t-1}+\boldsymbol{\Delta})$ 的唯一解。这样一来，SE 算法设计就从"拍脑袋叠约束"变成"先选 OTE 目标、再机械求差"，从根上杜绝累积漂移；任何现成 OTE 编辑器都能照此升级出对齐版本。

**3. 后处理正则的可解析误差补偿（Algorithm 1）：把每步偏差记账扣回**

PRUNE、RECT 这类算法的麻烦在于"先解出 $\boldsymbol{\Delta}_t$、再施加后处理正则 $\mathcal{R}_p(\boldsymbol{\Delta}_t)$"，每步引入的偏差会沿编辑步累积，正是它们长序列上掉点的真凶。作者的修法是显式维护累积误差 $\mathbf{E}_t \leftarrow (\mathcal{R}_p(\boldsymbol{\Delta}_t)-\boldsymbol{\Delta}_t)\mathbf{C}_t$，下一步把它从残差里扣回去 $\boldsymbol{\Delta}_t = (\mathbf{R}_t\mathbf{K}_t^\top - \mathbf{E}_{t-1})\mathbf{C}_t^{-1}$，其中 $\mathbf{C}_t = \mathbf{C}_{t-1} + \mathbf{K}_t\mathbf{K}_t^\top$。这一项纯解析、无需重训，就能保证 $\sum_\tau \mathcal{R}_p(\boldsymbol{\Delta}_\tau)$ 仍累积重建对应的 OTE 解，把 PRUNE/RECT 一次性拉回 OTE 等价。

### 损失函数 / 训练策略
所有方法的损失都是 Frobenius 范数最小二乘 $\|(\mathbf{W}+\boldsymbol{\Delta})[\mathbf{K}_0\mid\mathbf{K}_\cdot] - [\mathbf{V}_0\mid\mathbf{V}_\cdot]\|_F^2$，配合 $\lambda \mathbf{I}$ 的 ridge 正则；保留集 $(\mathbf{K}_0,\mathbf{V}_0)$ 从 Wikitext 抽 100,000 条三元组估计。处理冲突编辑时按 Proposition 3.5 在重叠键集 $\mathcal{K}_o$ 上引入 Resolve 函数解析合并 $\mathbf{V}$，对应闭式 $\boldsymbol{\Delta}_t^* = (\mathbf{R}_t\mathbf{K}_t^\top - (\mathbf{V}_{\mathcal{B}_o^{(t)}}-\mathbf{W}_{t-1}\mathbf{K}_{\mathcal{B}_o^{(t)}})\mathbf{K}_{\mathcal{B}_o^{(t)}}^\top)(\mathbf{K}_{\mathcal{P}_{t-1}}\mathbf{K}_{\mathcal{P}_{t-1}}^\top + \mathbf{K}_t\mathbf{K}_t^\top - \mathbf{K}_{\mathcal{B}_o^{(t)}}\mathbf{K}_{\mathcal{B}_o^{(t)}}^\top)^{-1}$。

## 实验关键数据

### 主实验
在 CounterFact 和 ZsRE 上跨 GPT-2 XL (1.5B) / GPT-J (6B) / LLaMA-3 (8B) / Qwen-2.5 (7B) 验证 OTE-SE 等价。设置：每步编辑 100 条、共 20 步 = 2000 条。

| 设置 | 方法 | Eff.↑ | Gen.↑ | Spe.↑ | 备注 |
|------|------|-------|-------|-------|------|
| Fully Aligned | PRUNE (aligned) | 99.87±0.03 | 94.91±0.22 | 79.90±0.20 | OTE-SE 对齐 + 误差补偿 |
| Fully Aligned | RECT (aligned) | 99.88±0.08 | 94.34±0.09 | 81.56±0.22 | 同上 |
| Not OTE Aligned | PRUNE (Naive) | 56.30±1.25 | 53.90±0.75 | 48.18±0.21 | 朴素重复 OTE |
| Not OTE Aligned | RECT (Naive) | 60.35±1.12 | 58.35±1.25 | 46.80±0.20 | 同上 |

GLUE 通用能力测试（LLaMA-3，证伪零空间投影）：

| 设置 | SST | MMLU | MRPC | CoLA | RTE | NLI |
|------|-----|------|------|------|-----|-----|
| Pre-edit | 0.831 | 0.562 | 0.658 | 0.761 | 0.284 | 0.666 |
| Full 法方程 | 0.846 | 0.548 | 0.643 | 0.779 | 0.292 | 0.668 |
| Null-Space 简化 | 0.000 | 0.014 | 0.000 | 0.000 | 0.000 | 0.000 |

### 消融实验

| 配置 | PRUNE (Eff./Gen./Spe.) | RECT (Eff./Gen./Spe.) | 说明 |
|------|------------------------|-----------------------|------|
| Fully Aligned | 99.87 / 94.91 / 79.90 | 99.88 / 94.34 / 81.56 | 对齐 + 补偿 |
| No Err. Correction | 99.82 / 95.22 / 80.19 | 96.98 / 83.60 / 84.86 | 仅对齐 |
| Not OTE Aligned | 56.30 / 53.90 / 48.18 | 60.35 / 58.35 / 46.80 | 朴素 |

### 关键发现
- 是否 OTE 对齐对编辑成功率影响 ≈40 个百分点，是真正的稳定性主因；零空间投影、谱裁剪等正则在长序列上是干扰项。
- 误差补偿对 RECT 比对 PRUNE 重要得多——RECT 砍掉相对权重比例的操作每步都掉信息，必须用 $\mathbf{E}_t$ 显式扣回；PRUNE 只丢极端特征值，本来就近似恒等。
- 隐空间 t-SNE 可视化显示，PRUNE (Naive)/RECT (Naive) 编辑后分布大幅漂移；做完 OTE 对齐后漂移几乎消失，说明 "post-editing distribution shift" 不是正则不够强、而是 SE 与 OTE 不一致引起的。

## 亮点与洞察
- 用一个反例（Memorize-the-Latest）就把 AlphaEdit 几年的"零空间叙事"打回原形：把对历史的约束去掉、只留 $\mathbf{P}$，模型立刻全 0 崩溃。这种"靶向证伪"比换数据集刷点优雅得多，值得借鉴到任何"机制核心论"的批评里。
- Proposition 3.4 等于给出一个机械流水线：把任意带凸正则的 OTE 目标自动升级出对齐 SE 版本。对未来想发明新编辑器的人来说，少走的弯路远比 SOTA 数字值钱。
- Algorithm 1 的误差补偿项 $\mathbf{E}_t$ 形式上和"伪逆里的 deflation"很像，本质是"把后处理引入的非凸偏差线性化吸进残差"。这条思路可以迁到任何"先解再投影"的算法上，比如稀疏化训练、剪枝重训、量化补偿。

## 局限与展望
- 理论假设 shifted quadratic representability，只覆盖最小二乘 / ridge 类目标；对非二次损失（如 KL、对比损失）能否套用尚未给出。
- 只在 FFN 输出投影一层做编辑，没考虑跨层联合编辑；对 ROME 这种 rank-one 单层方法成立的等价是否能扩展到多层耦合需要新工具。
- "保留集" $(\mathbf{K}_0,\mathbf{V}_0)$ 仍然是从 Wikitext 抽 10 万条估计的，量级不大时 OTE 解本身就不准；理论保证的是与 OTE 等价，不是与"真实分布"对齐，这一阶差距没量化。
- 改进思路：(1) 把 OTE-SE 等价推广到带 KL 罚的对话/对齐场景，统一 RLHF 增量更新；(2) 把误差补偿做成可学习项，处理量化/低秩压缩后的非线性偏差。

## 相关工作与启发
- **vs AlphaEdit (ICLR 2025)**: 都用零空间投影写闭式解，但 AlphaEdit 把稳定归因于 $\mathbf{P}$，本文证明真正起作用的是 OTE-SE 等价，$\mathbf{P}$ 是充分非必要；MEMIT（$\mathbf{P}=\mathbf{I}$）只要对齐 OTE 同样稳定。
- **vs PRUNE / RECT**: 它们靠后处理正则限制更新幅度，本文揭示这种做法在长序列上累积偏差并提供 Algorithm 1 的可解析补偿，把它们升级成 PRUNE (aligned)/RECT (aligned) 后性能追平 AlphaEdit。
- **vs SimIE / LyapLock / AnyEdit / SIR**: 这些走的是"换更聪明的约束"路线，本文反方向走"减约束"：先证明哪些约束多余、再给出最简稳定模板。两条路线互补——前者拓 scope，后者清家底。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 Memorize-the-Latest 证伪 + OTE-SE 等价证明，提供原创理论视角而非新模块。
- 实验充分度: ⭐⭐⭐⭐ 四个 LLM × 两个数据集 × GLUE 通用能力 + 消融，足以覆盖结论，但缺多层联合编辑与对话场景。
- 写作质量: ⭐⭐⭐⭐⭐ "迷宫与线团"隐喻贯穿全文，RQ1-3 结构清晰，符号统一好读。
- 价值: ⭐⭐⭐⭐⭐ 直接简化整个序列编辑设计空间，给出可机械化套用的稳定 SE 构造流程，对工业级长寿命模型更新影响明显。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)
- [\[ACL 2025\] Neuron-Level Sequential Editing for Large Language Models](../../ACL2025/knowledge_editing/neuron-level_sequential_editing_for_large_language_models.md)
- [\[ICML 2026\] Reverse-Engineering Model Editing on Language Models](reverse-engineering_model_editing_on_language_models.md)
- [\[AAAI 2026\] Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](../../AAAI2026/knowledge_editing/multiplicative_orthogonal_sequential_editing_for_language_models.md)

</div>

<!-- RELATED:END -->
