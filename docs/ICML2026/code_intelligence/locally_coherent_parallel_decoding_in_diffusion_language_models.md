---
title: >-
  [论文解读] Locally Coherent Parallel Decoding in Diffusion Language Models
description: >-
  [ICML 2026][代码智能][扩散语言模型] 本文提出 CoDiLA，在 masked 扩散语言模型（DLM）外挂一个轻量自回归（AR）小模型，用"软嵌入"接收 DLM 的边缘分布并在小块内做局部自回归解码，从而在保留 DLM 全局双向能力的同时消除并行采样产生的局部不连贯问题…
tags:
  - "ICML 2026"
  - "代码智能"
  - "扩散语言模型"
  - "并行解码"
  - "局部自回归"
  - "软条件"
  - "代码生成"
---

# Locally Coherent Parallel Decoding in Diffusion Language Models

**会议**: ICML 2026  
**arXiv**: [2603.20216](https://arxiv.org/abs/2603.20216)  
**代码**: https://github.com/IBM/coherent-diffusion-local-autoregression (有)  
**领域**: LLM效率 / 扩散语言模型 / 并行解码  
**关键词**: 扩散语言模型, 并行解码, 局部自回归, 软条件, 代码生成

## 一句话总结
本文提出 CoDiLA，在 masked 扩散语言模型（DLM）外挂一个轻量自回归（AR）小模型，用"软嵌入"接收 DLM 的边缘分布并在小块内做局部自回归解码，从而在保留 DLM 全局双向能力的同时消除并行采样产生的局部不连贯问题，在代码生成上以 ≥2× 吞吐建立新的 Pareto 前沿。

## 研究背景与动机

**领域现状**：当前 SoTA 离散扩散语言模型（Dream-Coder、LLaDA、DiffuCoder 等）通过 [MASK] 吸收态学反向去噪，理论上一次可并行预测多个 token，相比 AR 模型有望突破线性延迟；这种双向、可回填的特性对代码生成尤其有吸引力。

**现有痛点**：标准 DLM 在反向步用条件**边缘**分布独立采样每个 mask token —— 即"条件 token 独立性假设"（Conditional Token Independence）。对距离远的 token 这一假设无碍，但对同时解码的相邻 token（多字节单词、语法块）就会产生"单看 token 都对、连起来语法不通"的不连贯输出（Figure 1a，预测出 `merge_intervals problem):` 这种碎片）。实践中只能每步解出极少 token 才能保住精度，亚线性延迟优势被吃光。

**核心矛盾**：并行性 vs 局部相关性。要并行必须假设 token 间独立，但相邻 token 间的强相关恰恰是语法正确的关键。

**本文目标**：在不破坏 DLM 全局非因果能力（infilling、双向规划）的前提下，恢复同步解码的相邻 token 之间的联合分布建模。

**切入角度**：作者把"扩散"从 token 粒度提到**块（block）**粒度——块内联合、块间仍按 DLM 独立。理论上证明块独立性偏置严格降低 NELBO 下界（Theorem 3.2），但直接对 $|V|^B$ 维联合分布建模又组合爆炸。于是把块内联合建模"外包"给一个小 AR 模型，让它只在 $B$ 个 token 的小窗口里做因果解码，单次 AR 延迟受限于 $B$ 而非 $L$。

**核心 idea**：DLM 做全局草稿、AR 做局部清洁工——用 DLM 的边缘概率向量当**软嵌入**喂进 AR，AR 输出局部相干的块。

## 方法详解

### 整体框架
CoDiLA 要解决的是 DLM 并行采样时"相邻 token 各自都对、连起来语法不通"的局部不连贯问题。它的做法是让冻结的 DLM 主干继续做全局草稿、再外挂一个轻量 AR 小模型在每个小块内做局部清洁。具体来说，DLM（双向 Transformer，参数 $\psi$，如 Dream-Coder-Instruct-7B）对带 [MASK] 的序列 $x_t$ 一次前向，给出每个位置的边缘分布 $\pi^j_\psi(x_t)\in\Delta^{|V|-1}$；序列被切成长度 $B$ 的连续块，要解码的块把它 $B$ 个边缘分布转成"软嵌入"喂给 AR 小模型（参数 $\phi$，如 Qwen3-0.6B），AR 在块内自回归地解出真实 token。最终联合概率 $p_\theta(b^i_0\mid x_t)=p^{\text{AR}}_\phi(b^i_0\mid\pi_\psi(x_t))$，整体参数 $\theta=[\psi,\phi]$ 中只训练 AR。

### 关键设计

**1. 块级扩散与 NELBO 严格下降：把"为什么做块"立成硬下界**

痛点是 token 级独立性假设让相邻 token 的联合分布被强行拆开，而这正是语法正确的关键。CoDiLA 把假设松弛为"块独立、块内联合"：将 $x_0$ 切成 $L/B$ 个块 $b^i_0\in W=V^B$，沿块因子化反向过程 $p_\theta(x_0\mid x_t)=\prod_i p_\theta(b^i_0\mid x_t)$，而 NELBO 每步损失仍保留 token 级独立模型的 cross-entropy 形式 $L_t=\mathbb{E}_{q(x_t\mid x_0)}\big[\sum_i -\delta_{x^i_t,[\text{MASK}]}\frac{\alpha_{t-1}-\alpha_t}{1-\alpha_t}\log p_\theta(x^i_0\mid x_t)\big]$。

之所以这一步是整套方法的理论基底，是因为 Theorem 3.2 证明了 $B_1-B_B=\sum_{t,i}\big(\sum_j H[x^{(i-1)B+j}_{t-1}\mid x_t]-H[b^i_{t-1}\mid x_t]\big)\geq 0$——块内的"总相关性"恰好等于 token 级模型多付的那部分不可约误差，块越大下界越紧。以往工作（Huang 2022、Liu 2025a 等）只讨论 $B=1$ 的退化情形，这里首次量化了 $B>1$ 的改善方向，Figure 3 也实证 $B$ 越大训练 PPL 越低且尚无饱和。

**2. 软条件接口：让 AR 用"自己听得懂的话"接收 DLM 的完整边缘信号**

块内联合分布是 $|V|^B$ 维的，直接建模组合爆炸，于是把它外包给 AR；但 AR 怎么接收 DLM 的输出是关键。朴素做法（APD / FlashDLLM）是把边缘分布 top-1/top-$k$ 截断成离散 token 再喂 AR，看似省事却会把"最该被选出来的相干序列"挡在解空间外。CoDiLA 改用软嵌入：把 $\pi^j_t$ 当成对 AR 嵌入表 $E_\phi$ 的期望权重，算 $e^j_t=\sum_{v\in V}[\pi^j_t]_v\cdot E_\phi(v)$，再拼成 $[E_\phi(\langle\text{think}\rangle), e^1_t,\ldots,e^B_t, E_\phi(\langle\backslash\text{think}\rangle)]$ 喂给 AR。这样既把高维概率向量无损压进 AR 已有的语义嵌入空间、免去从零预训练，又用 `<think>` 边界 token 把这段输入"伪装成在思考"以激活 AR 的内省式解码路径。

理论上这种设计的充分性由 Theorem 3.3 兜底：条件在**完整**边缘上时存在 $\phi$ 使 $p^{\text{AR}}_\phi(\cdot\mid\pi)=q(\cdot)$ 精确恢复目标分布；而 top-$k$ 截断只能恢复受限于截断 Fréchet 类的分布，且存在 $q$ 让其全局众数 $b^*=\arg\max_b q(b)$ 被永久排除、引入不可约偏差。工程上这两个细节都不是可有可无的——消融显示去掉 `<think>` 边界 token，NELBO 就从 13.6 飙到 15.5（$B=4$）。

**3. 三档生成调度：在精度-吞吐 Pareto 上选位点且不丢双向能力**

训练好的 CoDiLA 需要可调的推理策略来覆盖不同场景，单一调度难以同时兼顾精度、吞吐与双向能力。围绕块熵 $h^i_t(k)=\frac{1}{k}\sum_j H[p_\theta(x^{(i-1)B+j}_0\mid x_t)]$ 定义三档：**静态并行**每步解 1 个完整块，选 $h^i_t(B)$ 最低者并用前后 10 块的局部窗口防止过早 EOS，简单但精度随 $B$ 下降；**动态并行**解满足 $h^i_t(k)\leq\tau$ 的最长 partial 块（$k\leq B$），$k=1$ 时退化用 DLM 自身的置信度+采样（单 token 不存在相干性问题），从而"难处少并行、易处多并行"地恢复精度；**AR 验证模式**让 AR 只比对自己的 top-1 与 DLM top-1，分歧处给该位置加置信度惩罚，零侵入地嵌进任何置信度调度并保留任意序解码，适合 ParallelBench / Graph Traversal 这类全局规划任务——AR 不当生成器只当判官，DLM 的非因果性完全保留。

### 损失函数 / 训练策略
端到端用方程 (2) 的 cross-entropy NELBO 训练，DLM 主干冻结、只微调 AR；前向加噪从 token 级改成**整块加噪**（一次 mask 连续 $B$ 个），让 AR 真正学到"基于 DLM 软嵌入预测整块"的能力。在 Ling-Coder-SFT 上每个 $B$ 训练 32k 步，单卡 A100 80GB / PyTorch 2.7 / bf16。

## 实验关键数据

### 主实验

主实验在 Dream-Coder-Instruct-7B 上以静态并行（K=B per step）对比 baseline 与 ADJUST（Bansal & Sanghavi 2025）。下表是 HumanEval 上的语法错误率（提取脚本无法抽出代码的占比，％），最能说明"局部相干性"问题：

| 模型 | K=B=2 | K=B=4 | K=B=8 |
|------|-------|-------|-------|
| Dream-Coder-Instruct-7B | 18 | 38 | 70 |
| **CoDiLA**（本文） | **4** | **13** | **16** |
| 改善 (pp) | −14 | −25 | **−54** |

在并行度最大（$B=8$）的设定下，语法错误率从 70% 降到 16%，54 个百分点。Figure 4 报告了 HumanEval/+, MBPP/+, BigCodeBench (full/hard) 全套基准上 Pass@1 vs 吞吐（tokens/sec, batch=1, A100）的 Pareto 曲线——CoDiLA 在所有 6 个基准上都占据最外沿。作者特别澄清：精度增益不是 AR 小模型独立完成的——Qwen3-0.6B 单独在 HumanEval 只有 35%，CoDiLA（$B \leq 4$）显著超过 0.6B 的能力上限。

HumanEval-Infilling（双向能力保留验证）：

| 模型 | Pass@1 (%) | Tokens/step |
|------|-----------|-------------|
| Deepseek-Coder-6.7B | 45.7 | 1 |
| Qwen2.5-Coder-7B | 58.7 | 1 |
| DreamOn (K=1, 顺序) | 62.5 | 1 |
| DreamOn (K=2, 并行) | 53.1 | 2 |
| DreamOn + CoDiLA ($\tau=0.2$) | **62.5** | 1.3 |
| DreamOn + CoDiLA ($\tau=0.5$) | 61.5 | **1.5** |

并行解码情况下精度不掉，并行度提升 1.3–1.5×。

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| Soft 条件 vs Top-K 条件 | Top-K 全面掉点 | 实证 Theorem 3.3：信息截断引入不可约偏差 |
| 去掉 `<think>/<\think>` 边界 | NELBO 13.6 → 15.5 ($B=4$) | 边界 token 是激活预训练 AR 推理路径的必要钥匙 |
| AR size: 0.6B → 1.7B → 4B | 无一致增益 | 不靠 AR 主干堆参数，0.6B 足矣 |
| 单 $B=8$ 块 vs 两个 $B=4$ 块 | 单块高 8 pp | 块内联合性是吃掉并行收益的关键 |
| 候选范围：局部 10 vs 50 块 | 吞吐差 < 15% | 局部窗口选择很鲁棒 |
| 生成顺序 Spearman 相关 | 随 $B$ 增大下降 | CoDiLA 没有把全局拉回左到右，DLM 的任意序优势保留 |
| Batch size = 8 | AR 开销几乎被完全摊销 | bs=1 的小额外延迟到 bs=8 就消失 |

### 关键发现
- **大块的训练损失更低且无饱和**（Figure 3，$B \in \{2,4,8,16,32\}$ 在 32-token 连续 mask 设定下单调下降），首次实证了 Theorem 3.2 中 $B_1 - B_B$ 与块内总相关性等号的方向性。
- **精度提升的根源是局部相干性而非 AR 能力**：AR 小模型独立在 HumanEval 只有 35%，但 CoDiLA 把 7B DLM 从语法错误率 70% 救到 16% —— 0.6B 不是新生成器，而是相干性裁判 + 局部清洁工。
- **动态并行能完全消除精度退化**：$B=4$ + 阈值 $\tau$ 调度，在 ≥2× 加速下追平 sequential（$K=1$）精度；优于小块（$B=2$）的静态采样。
- **双向能力没付学费**：infilling、ParallelBench 等价/规划等非因果任务上 CoDiLA 保留甚至改善了 DLM 原生能力，因为 AR 只对块内做因果，块间仍是 DLM 的双向注意力。

## 亮点与洞察
- **理论与工程罕见地完美闭环**：从 NELBO 严格下降的不等式（Theorem 3.2）到 Fréchet 类截断偏差（Theorem 3.3）再到训练损失曲线（Figure 3）再到下游精度，几条链路全打通。同行很多并行 DLM 工作是"先做后讲故事"，这里反过来。
- **"软嵌入 + `<think>` 包络"是真正巧妙的工程巧思**：把 DLM 边缘分布投影到预训练 AR 自己的嵌入空间，相当于让 AR 用"自己听得懂的话"理解 DLM；再用思考标签把这段输入"伪装成在思考"，激活 AR 的内省式解码路径——这两个小设计加起来贡献了相当大一部分增益（去掉边界 NELBO +14%）。
- **"DLM 做草稿、AR 做执行"的分工**对其它任务也有迁移价值：任何需要"全局规划 + 局部精细执行"的生成任务（结构化报告、SQL、HTML/CSS、JSON schema 生成）都可套用，把 AR 限定在局部短段、把 DLM 留给全局编辑/回填。
- 与 ADJUST、APD、TiDAR 等竞品对比，CoDiLA 是唯一**既快又保留双向能力**的方案。其它方案要么牺牲非因果性（拖回左到右）、要么训练成本极高（从零预训练辅助 DLM）。

## 局限与展望
- 作者承认的局限：当前块长 $B$ 固定，未来可探索**语义自适应块长**（引用 Zhang et al. 2026）；大 $B$ 虽然降低损失但 AR 串行延迟逐步吃掉并行收益，需要在 $B$ 和并行度间手调。
- 自己发现的局限：(i) AR 训练 + DLM 主干冻结的"双阶段"设定假设 DLM 已经训得很好，对中型/小型 DLM 是否一样有效未验证；(ii) 软嵌入要求 AR 与 DLM 的 tokenizer 严格一致（这里都用 Qwen tokenizer），跨家族（如 Dream + LLaMA AR）需另做对齐；(iii) 实验集中在代码生成，自然语言生成、数学推理等任务上联合分布的"局部相干性"是不是同样硬伤还是个开放问题；(iv) batch=1 评测有夸大延迟收益之嫌，虽然作者在 batch=8 的消融里说 AR 开销被摊销，但仍未给完整服务化场景下的吞吐数据。
- 改进思路：把 AR 的"信赖域"动态扩张（$B$ 自适应）；做 multi-AR 共存——不同尺寸 AR 对应不同熵区间；把验证模式与 speculative decoding 融合，让 AR 同时充当 verifier 和 proposer。

## 相关工作与启发
- **vs ADJUST (Bansal & Sanghavi 2025)**：同样思路是"外挂辅助小模型增相干性"，但 ADJUST 的辅助模型是单层 DLM、必须从零预训练、并对全序列重复跑全注意力；CoDiLA 用预训练 AR，作用范围严格限定在块内，省了预训练成本同时拿到更大的精度增益。Figure 4 对比清楚显示 ADJUST 在所有基准上都被 CoDiLA 占优。
- **vs APD (Israel 2025) / FlashDLLM (Hu 2026) / TiDAR (Liu 2025b)**：这些方法用 AR 对 DLM 输出做左到右验证 / self-speculative，相当于把 DLM 退化成 quasi-AR，扔掉了双向、infilling 能力。CoDiLA 在块内 AR、块间仍 DLM 双向，是真正保留 DLM 优势的解决方案。
- **vs Discrete Copula Diffusion (Liu 2025a)**：思路最接近——把 DLM 边缘与 AR 联合合成；但 Copula 需要多次全序列计算，开销大；CoDiLA 的软嵌入是单次注入，开销低。
- **vs Block Diffusion / Fast-dLLM / D2F**：这些半 AR 方法固定左到右块顺序换 KV-cache 复用，牺牲任意序生成；CoDiLA 在块内严格因果但块间自由，互补关系——作者明确指出可叠加。
- **启发**：对所有"快速生成但局部不连贯"的并行模型（图像 patch、video frame、语音码本），都可考虑"全局并行扩散 + 局部因果小模型"的二级分工范式。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "全局 DLM + 局部 AR 软嵌入"是该方向上第一个把理论、接口、训练、调度全部打通的端到端方案。
- 实验充分度: ⭐⭐⭐⭐ 6 个代码基准 + infilling + 规划任务 + 7 类消融覆盖广，但 batch=1 偏 latency 视角，缺服务化吞吐数据。
- 写作质量: ⭐⭐⭐⭐⭐ 两条理论定理给出硬下界与硬偏差刻画，问题陈述、方法、实验衔接清晰，Figure 1 单图把动机讲透。
- 价值: ⭐⭐⭐⭐⭐ 对追求亚线性延迟的 DLM 是必读，思路（块独立性 + 软嵌入 AR）可迁移到所有并行生成范式。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](../../ICLR2026/code_intelligence/training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)
- [\[ACL 2026\] Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding](../../ACL2026/code_intelligence/static_program_slicing_using_language_models_with_dataflow-aware_pretraining_and.md)
- [\[ICML 2026\] Entropy-informed Decoding: Adaptive Information-Driven Branching](entropy-informed_decoding_adaptive_information-driven_branching.md)
- [\[ICML 2026\] Poison with Style: A Practical Poisoning Attack on Code Large Language Models](poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)
- [\[AAAI 2026\] DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](../../AAAI2026/code_intelligence/diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)

</div>

<!-- RELATED:END -->
