---
title: >-
  [论文解读] Optimizing Token Choice for Code Watermarking: An RL Approach
description: >-
  [ICML 2026][LLM安全][代码水印] CodeTracer 在冻结的 code LLM 旁边挂一个小的 watermark policy 网络，用 GRPO + 双奖励（执行通过 + z-score）+ Gumbel-Top-k 直通估计联合学习"在哪个 token 位置加水印、选哪一组 gree…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "代码水印"
  - "GRPO"
  - "Gumbel-Top-k"
  - "直通估计"
  - "z-score"
---

# Optimizing Token Choice for Code Watermarking: An RL Approach

**会议**: ICML 2026  
**arXiv**: [2508.11925](https://arxiv.org/abs/2508.11925)  
**代码**: https://github.com/TimeLovercc/CodeTracer (有)  
**领域**: LLM安全 / 代码水印 / 强化学习  
**关键词**: 代码水印, GRPO, Gumbel-Top-k, 直通估计, z-score

## 一句话总结
CodeTracer 在冻结的 code LLM 旁边挂一个小的 watermark policy 网络，用 GRPO + 双奖励（执行通过 + z-score）+ Gumbel-Top-k 直通估计联合学习"在哪个 token 位置加水印、选哪一组 green token"，在几乎不掉 Pass@1 的前提下把代码水印的检测 AUROC 从 ~70% 抬到 ~78%。

## 研究背景与动机

**领域现状**：主流 LLM 水印（Kirchenbauer 2023 的 green-red 方案）在生成时把词表随机切成 green/red 两半，对 green token 加固定 logit bias $\delta$，检测时统计绿 token 频次做 z-test。在自然语言上效果不错，因为大多数位置允许多个语义等价的 token。

**现有痛点**：代码生成场景里 (1) 大量位置是语法强制的（`def`、括号、关键字），改动直接编译失败；(2) 不同位置对扰动的容忍度异质（变量名能改、API 名不能改）；(3) 低熵分布让无差别 bias 容易把代码改坏。早期 SWEET、CodeIP 等方法要么需要在检测时拿到原 LLM 的 logits 或 prompt 计算熵，要么需要手写每种语言的语法变换规则，实际部署门槛高。

**核心矛盾**：水印的"统计可检测性"与代码的"功能正确性"在低熵、强语法约束下是直接对抗的——加得弱检测不出来，加得强代码跑不通。

**本文目标**：(i) 自动判断哪些位置可以安全加水印，(ii) 在可加位置选一组保功能的 green token 集合 $G$，(iii) 检测端不依赖原 LLM。

**切入角度**：把"是否加水印 $w$"和"green 集合 $G$"建模成一个上下文相关的 policy $\pi_\phi(a\mid\mathbf{c})$，与冻结 LLM $\pi_\theta$ 组合成 $\pi_{\theta\oplus\phi}$，让强化学习自己去学语法/语义约束——因为代码同时具备两类天然 verifiable reward：单元测试通过与否、z-score 高低。

**核心 idea**：把代码水印改写成"训练一个小的 policy 网络去 bias LLM 的下一 token 分布"的 RL 问题，用 GRPO + STE + Gumbel-Top-k 把离散决策塞进端到端梯度里。

## 方法详解

### 整体框架
输入 prompt $\mathbf{x}$ 后，冻结的 LLM $\pi_\theta$ 算 logits $\mathbf{l}\in\mathbb{R}^{|\mathcal{V}|}$；同时可训练 watermark policy $\pi_\phi$ 看一个固定窗口的上下文 $\mathbf{c}$，输出 $(w, G)$：$w\in\{0,1\}$ 表示该位置是否加水印，$G\subset\mathcal{V}$ 是大小 $k=\lfloor\gamma|\mathcal{V}|\rfloor$ 的 green token 集合。组合后的 watermarked logits 是 $\tilde{l}_j = l_j + w\cdot\delta\cdot\mathbb{1}_{v_j\in G}$，再 softmax 采样得到 $\tilde y_t$。检测时只用 $\pi_\phi$ 重放每个位置的 $(w, G)$，对 $w=1$ 的子集统计落在 $G$ 内的频次，做单比例 z-test $z = (N_G - T\gamma)/\sqrt{T\gamma(1-\gamma)}$，不需要原 LLM 介入。

### 关键设计

1. **策略化水印（Policy-driven watermarking with frozen LLM）**:

    - 功能：把原本固定的 green-red 划分升级为上下文相关、可学习的策略，并保证训练好的 $\pi_\phi$ 是个可插拔模块。
    - 核心思路：训练时 LLM 参数 $\theta$ 全冻结，只学 watermark 模型 $\phi$（约 118M，相对 1.5B base LLM < 10% 参数）。$\pi_\phi$ 是一个小 Transformer，输出 $(|\mathcal{V}|+1)$ 维向量 $(w_\phi, \mathbf{l}_\phi)$，其中 $w_\phi$ 决定 $w$、$\mathbf{l}_\phi$ 决定 $G$ 的排序。检测端只持有 $\pi_\phi$，能复现 $(w, G)$。
    - 设计动机：冻结 LLM 避免 fine-tune 破坏代码能力（Xu et al. 2024 那种 fine-tune LLM 的方式会有不可预期 side effect）；同时 $\pi_\phi$ 可作为 plug-in 套到没见过的更大 LLM 上（论文用 1.5B 训练、8B 推理验证迁移性）。

2. **GRPO + 三段式奖励（execution + z-score + process token-level）**:

    - 功能：在没有任何预先标注的"水印代码"数据的情况下，让策略学到"在哪儿加 & 怎么选"。
    - 核心思路：用 DeepSeek-R1 的 GRPO 框架，奖励由三部分构成——$R_1$ 是执行 reward（全部 test case 通过 = 1，否则 0，硬约束保功能）；$R_2$ 是饱和 z-score reward（$z\geq 4$ 给 1，$0<z<4$ 线性，$z\leq 0$ 给 0，逼检测显著性）；$R_3$ 是 token 级 process reward（$w_t=1$ 且 $s_t\in G_t$ 给 $+1$，落 red 给 $-1$，不加水印给 0）。优势函数 $\hat A(s_t, a_t) = (A_1 + A_2)\cdot\mathbb{1}_{\text{is\_code}}(s_t)$ 把 outcome 级 $A_1$ 与 token 级 $A_2$ 相加后再屏蔽非代码 token。
    - 设计动机：纯 outcome 奖励对序列里每个 token 给同样的信号，对"哪些位置该加"的指导太粗；引入 process-level $R_3$ 后训练速度和最终性能都明显提升（消融里 $R_3$ 去掉 AUROC -7.84pp，TPR -16.05pp）。$\mathbb{1}_{\text{is\_code}}$ 屏蔽是为了避免在自然语言 chain-of-thought 段落上浪费水印预算。

3. **STE + Gumbel-Top-k 让 $(w, G)$ 可微**:

    - 功能：$w\in\{0,1\}$ 和"从 $|\mathcal{V}|$ 里 top-$k$ 选 $G$"都是离散操作，梯度过不去；这一设计把它们打通成端到端可训练。
    - 核心思路：对 $w$ 用 Straight-Through Estimator：$w = \mathbb{1}_{w_\phi>0} + \sigma(w_\phi) - \text{sg}(\sigma(w_\phi))$，前向走硬阈值、反向沿 $\sigma$ 的梯度。对 $G$ 用 Gumbel-Top-$k$：扰动 logits $\mathbf{g} = \mathbf{l}_\phi + (-\log(-\log \mathbf{u}))$，$\mathbf{u}\sim\text{Uniform}(0,1)^{|\mathcal{V}|}$，取 top-$k$ 得到 $G$；并用 $\mathbf{l}_G = \mathbb{1}_{v\in G} + \mathcal{S}(\mathbf{g}) - \text{sg}(\mathcal{S}(\mathbf{g}))$ 形式的 indicator，前向硬选、反向沿 Gumbel-Softmax 松弛。
    - 设计动机：传统做法（Categorical reparam）只能近似单个采样，没法直接处理"选 $k$ 个"。Gumbel-Top-$k$（Xie & Ermon 2019）正好对应大小固定的 green 集合，离散性保留了水印的统计可验证性，连续性又喂进 PPO/GRPO 的策略梯度。

### 损失函数 / 训练策略
最终目标是带 KL 正则的 GRPO clipped objective：

$\max_\phi \mathbb{E}_{s\sim\mathcal{D}}\left[\frac{1}{|s|}\sum_t \min\left(r_t(\phi)\hat A_t, \text{clip}(r_t(\phi), 1-\varepsilon, 1+\varepsilon)\hat A_t\right)\right] - \beta D_{\text{KL}}(\pi_{\theta\oplus\phi}\|\pi_{\text{ref}})$

其中 $r_t(\phi) = \pi_{\theta\oplus\phi}(s_t|s_{<t})/\pi_{\text{ref}}(s_t|s_{<t})$。参考策略 $\pi_{\text{ref}}$ 用 $\pi_{\theta\oplus\phi}$ 的旧拷贝（self-referential）。训练流程是先用 SFT 让 $\pi_\phi$ 学到代码 token 分布，再上 GRPO；整个训练在单卡 A100 上约 1 天完成。Base LLM 用 OpenCoder-1.5B-Instruct，$\gamma=0.5$、$\delta$ 走标准设置。

## 实验关键数据

### 主实验
HumanEval / MBPP 上对比 post-hoc 检测（logp、LogRank、DetectGPT、GPTZero）和 active 水印（WLLM、EXP-edit、SWEET）：

| 数据集 | 方法 | Pass@1 (%) | AUROC (%) | TPR@5%FPR (%) |
|--------|------|-----------|-----------|---------------|
| HumanEval | Base (无水印) | 65.42 | – | – |
| HumanEval | WLLM | 58.05 | 70.17 | 20.73 |
| HumanEval | EXP-edit | 59.29 | 66.50 | 25.61 |
| HumanEval | SWEET† | 60.46 | 76.24 | 27.44 |
| HumanEval | **CodeTracer** | **62.65** | **77.71** | **32.32** |
| MBPP | Base | 43.35 | – | – |
| MBPP | WLLM | 39.66 | 76.44 | 27.80 |
| MBPP | SWEET† | 39.64 | 77.24 | 24.80 |
| MBPP | **CodeTracer** | **42.10** | **78.42** | **31.60** |

Post-hoc 方法 AUROC 普遍 47–52%（基本随机），代码场景已经不能依赖被动检测。CodeTracer 在 Pass@1 上相对其它水印掉点最少（HumanEval -2.77pp vs WLLM -7.37pp），同时 TPR 比次优高 ~5pp。1.5B 上训好的 $\pi_\phi$ 直接套到 OpenCoder-8B 上 Pass@1 71.77%（vs Base 72.04%），AUROC 78.69%（vs WLLM 65.90%、EXP-edit 54.21%），plug-in 迁移成立。

### 消融实验
| 配置 | Pass@1 (%) | AUROC (%) | TPR (%) | 说明 |
|------|-----------|-----------|---------|------|
| CodeTracer (full) | 60.82 | 82.95 | 46.34 | 完整三奖励 |
| w/o $A_2$ (无 token 级 $R_3$) | 61.15 | 75.11 | 30.29 | 检测能力崩 |
| w/o $A_1$ (无 outcome 奖励) | 60.34 | 79.52 | 34.91 | 两端都掉 |
| CodeTracer-1 (纯 RL, 无 SFT 初始化) | 62.65 | 77.71 | 32.32 | 保功能档 |
| CodeTracer-2 (SFT + RL) | 60.82 | 82.95 | 46.34 | 重检测档 |

### 关键发现
- 三个奖励里 process-level $R_3$ 最关键：去掉它 AUROC 直接掉 7.84pp、TPR 掉 16pp，说明 token 级即时反馈对策略收敛的贡献远大于序列级的 z-score 反馈。
- SFT 初始化在 detectability 与 functionality 间提供一个明确旋钮（CodeTracer-1 vs -2），实际部署可按场景选档。
- 攻击鲁棒性：DIPPER 改写下 AUROC 58.42 vs WLLM 55.92；变量重命名下 AUROC 73.36 vs WLLM 70.91；攻击后仍保持领先但所有方法都明显退化。
- 推理 overhead 几乎可忽略：$\pi_\phi$ 与 LLM 并行跑，附加延迟 < 100μs（LLM 本身 500–800ms），显存增加 < 0.5GB。
- 跨语言：Java / C++（HumanEvalPack）上 CodeTracer 表现一致，说明 RL 学到的是较通用的"哪些位置可改"先验，不是 Python 专属。

## 亮点与洞察
- **把"找可水印位置"自动化**：之前的代码水印要么手写 AST 规则、要么依赖 LLM 自带 entropy。CodeTracer 直接用 RL 让 policy 自己学，省掉了一套和语言强耦合的工程。
- **Gumbel-Top-k 用在 watermark 上很贴切**：green set 本质就是"固定大小的子集采样"，比传统 categorical reparam 更天然，思路可以迁移到任何需要"固定基数离散选择 + 端到端梯度"的场景（如稀疏注意力、可学习 prompt-token mask）。
- **三奖励里 process reward 占主导，违反"end-to-end + outcome only 更高级"的常见直觉**：在 token 级标签便宜可得时，pure outcome RL 反而更慢、更差，提醒做 RLVR 时应该主动找 cheap dense signal。
- **检测端零 LLM 依赖**：把 watermarking 做成"只携带 $\pi_\phi$ 即可验证"的服务化形态，对 API 供应商更友好——可以把检测器交给第三方而不暴露 base LLM。

## 局限与展望
- 作者承认在 DIPPER 改写攻击下 AUROC 跌到 58.42%、TPR 仅 14.31%，对强语义改写仍脆弱；变量重命名鲁棒性也只是相对最好、绝对值掉得不少。
- 自己看到的局限：（1）训练成本仍需 RL rollout + 真实代码执行 sandbox，rollout 时的 test 执行延迟会显著拖慢训练，可扩展到的语言/库被沙箱覆盖度限制；（2）$\gamma$、$\delta$ 等水印超参仍是固定全局值，文中没探索按位置自适应；（3）只在 1.5B/8B OpenCoder 上验证，更大尺寸（70B+）的 plug-in 是否仍稳定未知；（4）安全场景下还需考虑 adversary 拿到 $\pi_\phi$ 后能否"反水印"——本文未做白盒攻击。
- 改进思路：把 $\delta$ 也做成 policy 输出（按位置自适应强度），用 sandbox-less 的执行近似（learned reward model）替代真实 test 以提速；或者把 $\pi_\phi$ 进一步蒸馏成 logit bias lookup table，让 detection 端连 transformer 都不用跑。

## 相关工作与启发
- **vs WLLM (Kirchenbauer 2023a)**：WLLM 用 PRF 固定切 green/red、固定 $\delta$，本文把这两个量都让 policy 学；在代码上 WLLM 掉 7+pp Pass@1，CodeTracer 只掉 ~3pp，说明可学习的位置/集合是低熵场景的关键。
- **vs SWEET (Lee 2023)**：SWEET 用 entropy 阈值选哪些位置加水印，检测时要原 LLM 重新算 entropy；本文把"该加哪"内化进 $\pi_\phi$，检测端零 LLM 依赖，部署更轻。
- **vs CodeIP (Guan 2024)**：CodeIP 靠 token type 预测器 + 语法规则注入，强工程；CodeTracer 用 RL 自动学语法约束，跨语言泛化更好。
- **vs Xu 2024 (RL for LM watermarking)**：Xu 2024 直接 fine-tune LLM 来学水印，可能破坏原能力；本文冻结 LLM 只学旁路 $\pi_\phi$，更可控、可迁移。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把代码水印重新表述成 policy 学习问题、用 Gumbel-Top-k + STE 解决离散梯度的 RL 公式很清晰；不过组件本身（GRPO、STE、Gumbel-Top-k）都是现成的。
- 实验充分度: ⭐⭐⭐⭐ HumanEval / MBPP / HumanEvalPack 三个 benchmark + 鲁棒性 + 迁移 + 消融齐全；缺更大模型（70B+）和针对 $\pi_\phi$ 的白盒攻击实验。
- 写作质量: ⭐⭐⭐⭐ 问题动机层层推进，公式与算法表述准确；图表略多但可读。
- 价值: ⭐⭐⭐⭐ 代码水印是 LLM 安全里被严重低估的方向，给出了一个"训练一次、可插拔到更大 LLM、检测端零依赖"的范式，工业可用性高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach](../../AAAI2026/llm_safety/uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[AAAI 2026\] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking](../../AAAI2026/llm_safety/watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)
- [\[ICML 2026\] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)

</div>

<!-- RELATED:END -->
