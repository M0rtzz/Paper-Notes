---
title: >-
  [论文解读] Predicting Large Model Test Losses with a Noisy Quadratic System
description: >-
  [ICML 2026][预训练][噪声二次系统] 本文提出 Noisy Quadratic System (NQS)——一个把 LLM 测试损失建模为 $L(N, B, K)$（模型大小 / 批大小 / 更新步数）的 mechanistic 损失模型…
tags:
  - "ICML 2026"
  - "预训练"
  - "噪声二次系统"
  - "Chinchilla"
  - "scaling law"
  - "batch size 建模"
  - "外推预测"
---

# Predicting Large Model Test Losses with a Noisy Quadratic System

**会议**: ICML 2026  
**arXiv**: [2605.09154](https://arxiv.org/abs/2605.09154)  
**代码**: 论文承诺 GitHub release  
**领域**: LLM 预训练 / scaling law / 训练动力学  
**关键词**: 噪声二次系统、Chinchilla、scaling law、batch size 建模、外推预测

## 一句话总结
本文提出 Noisy Quadratic System (NQS)——一个把 LLM 测试损失建模为 $L(N, B, K)$（模型大小 / 批大小 / 更新步数）的 mechanistic 损失模型，首次在 scaling law 中显式建模 batch size，并在 Pythia + OWT2 上把外推预测能力从 Chinchilla 的 ~20× 算力提升到 ~4000× 算力。

## 研究背景与动机
**领域现状**：Chinchilla 把 LLM 测试损失建模为 $L(N, D)$ 的简单 power law，用来在固定算力 $C \approx 6ND$ 下选最优 $N, D$ 比例。但模型规模上去后，研究者发现需要建模更多变量（batch size、学习率、weight decay），而 Chinchilla 的 functional form 难以扩展。

**现有痛点**：(1) Chinchilla 等纯 functional fitting 在 holdout 外推 50× 以上 compute 就明显失效；(2) loss-model-free 路线（如 Bergsma 等对最优 token budget 拟合 power law、$\mu$P 路线让最优 lr scale-invariant）依赖人类对模式的洞察，多条规则之间的交互不清晰，且离 loss prediction 太远难以严格评测；(3) 大家都不知道如何 principled 地把 batch size 加入 loss model。

**核心矛盾**：要在多个预训练变量（$N, B, K, D, lr, wd, \dots$）上做精确 loss prediction，纯 phenomenological power law 没有 mechanistic guidance，扩展时维度灾难；而严格的理论训练动力学（NQM、linear regression scaling）只有 asymptotic 表达且分多个 phase，无法直接当 prediction tool 用。

**本文目标**：构造一个 (1) 像 Chinchilla 一样轻量易用、(2) 可自然扩展到多个预训练变量、(3) 能在 train/holdout 严格分离下大幅外推预测的 loss model。

**切入角度**：把训练动力学领域三条经典理论线索——linear regression scaling（提供 $\mathcal{E}_{\mathrm{appx}}$ / $\mathcal{E}_{\mathrm{bias}}$ 项）、Noisy Quadratic Model（NQM，刻画 batch size 引起的方差项）、LayerNorm 等价于动态调整 lr——统一进一个 stochastic optimization 模型，但放弃 closed-form asymptotic，改用 numerical computation 直接评估全 trajectory。

**核心 idea**：用 "投影 SGD on quadratic + power-law 噪声 + LayerNorm 等价 lr 调度" 三件套构造 mechanistic 损失模型，把 LLM 测试损失表达为 7+1 个超参的封闭形式数值积分，把 Chinchilla 的 "$N, D$ 拟合" 升级为 "$N, B, K$ 全 trajectory 模拟"。

## 方法详解

### 整体框架
NQS 把 LLM 训练想象成一件很物理的事：在一个无穷维的二次损失面 $\mathcal{Q}^{\mathrm{NQS}}(w) = \mathcal{E}_{\mathrm{irr}} + \tfrac{1}{2}\langle w-w^*, H(w-w^*)\rangle$ 上跑带噪声的投影 SGD。把 Hessian $H$ 的特征向量按特征值从大到小排好，模型只在前 $N$ 个方向上更新（对应它有限的可训练参数），每步注入一个方差正比于 $1/B$ 的 mini-batch 噪声，跑 $K$ 步。整套动力学因此只由 $N, B, K$ 三个变量驱动，最终给出一个可数值计算的封闭损失 $L_\theta(N, B, K)$，其中 $\theta$ 含 7 个超参（三条谱的 power 指数 $p, q, r$、对应 scale 系数、学习率、不可约误差），再为 LayerNorm 补一个第 8 参数 $s = \mathbb{E}[\|w^{(0)}\|^2]$。它的精妙之处在于 "mechanistic but tractable"——保留二次优化的可解释结构，却把所有恼人的 asymptotic phase 隐式塞进数值参数里。

### 关键设计

**1. 三条 power-law 谱：让一个系统自动覆盖多个 asymptotic phase**

Chinchilla 那类纯 functional fitting 扩展不动，根子在于它没有 mechanistic 结构去描述训练里到底发生了什么。NQS 用三个相互独立的 power law 把整个系统参数化：Assumption 4.1 用 $\mathbb{E}[\lambda_n (\langle v_n, w^{(0)} - w^*\rangle)^2] = P/n^p$ 刻画初始偏差沿各特征方向的分布，Assumption 4.2 用 $\lambda_n = Q/n^q$ 刻画 Hessian 谱的衰减，Assumption 4.3 用 $\xi_n^{(k)} \sim \mathcal{N}(0, R/(n^r B))$ 刻画 mini-batch 噪声谱。这相当于把 Bordelon 等 linear-regression scaling 模型简化（固定 projection 替掉随机 $P$），同时把 NQM 的批噪声假设放宽到 $r \neq q$。关键在于：理论上 mini-batch 噪声会让训练经过好几个 functional form 各不相同的 asymptotic phase（Paquette 2025），NQS 不去逐 case 推公式，而是让 $p, q, r$ 三个指数自动 "插值" 到正确的 phase——既能复现 Chinchilla 的渐近形式 $L \sim N^{-(p-1)} + D^{-(p/q - 1/q)}$，又免去人工分段。

**2. 投影 SGD + Euler-Maclaurin 积分：把不可预测的理论变成秒级可算的预测器**

NQM 这类训练动力学模型通常只给 asymptotic bound，没法直接拿来预测。NQS 的破局点是把更新规则写成 $w^{(k)} = w^{(k-1)} - \gamma \mathrm{Proj}_{\mathbb{W}_N}(Hw^{(k-1)} - Hw^*) + \gamma \sum_{n=1}^N \xi_n^{(k)} v_n$，只在前 $N$ 个特征方向上更新并注入噪声——"投影到前 $N$ 维" 直接对应模型参数有限这一事实，剩下没训到的维度就是 latent 误差，正好给出 Chinchilla 里的 $\mathcal{E}_{\mathrm{appx}} \sim P/N^{p-1}$ 项。跑 $K$ 步后的期望 loss 有封闭表达：$K$ 上是几何级数可显式求和，$N$ 上的求和则用 Euler-Maclaurin 公式近似成积分，成本压到 $\mathcal{O}(1)$。结果是评估任意 $(N, B, K)$ 配置不到 1 秒、拟合整个 $\theta$ 也只要约 5 分钟，且不再被任何 asymptotic phase 的边界卡住。

**3. LayerNorm 等价学习率 $\gamma_k \propto 1/\|w^{(k)}\|^2$：补上小 batch 的最后一块拼图**

经验上 vanilla NQS 对大 batch 拟合很好，但小 batch 系统性偏差大——而要支持 compound resource allocation（在 time / memory 约束下选 $B$），恰恰需要模型能预测 "非临界 batch size" 区间的 loss。问题出在 normalization：受 van Laarhoven 启发，LayerNorm 等价于让有效学习率随 weight norm 反向变化 $\gamma_k \propto 1/\|w^{(k)}\|^2$，而这个效应在小 batch、噪声大时最显著。NQS 因此显式建模它，引入第 8 参数 $s = \mathbb{E}[\|w^{(0)}\|^2]$，并用 $\|w^{(k)}\|^2 \approx \mathbb{E}[\|w^{(k)}\|^2]$ 的近似把 $s$ 代入推导。$s$ 的常见取值是标准 init 下的 $s = N \times 0.02^2$，作者也建议在小 batch 数据子集上 grid search 确定——这一项正是 NQS 能把预测覆盖到小 batch 区间的关键。

### 损失函数 / 训练策略
推断 $\theta = (P, Q, R, p, q, r, \gamma, \mathcal{E}_{\mathrm{irr}})$ 分四步：先收集训练数据 $\{(N_i, B_i, K_i, l_i)\}$，再以对数空间 Huber/MSE 目标 $\mathcal{L}_\theta = \tfrac{1}{m}\sum_i (\log L_\theta(N_i, B_i, K_i) - \log l_i)^2$ 拟合，用 gradient-based optimizer 配多初始化并行下探 loss 表面；$s$ 因数值原因不和 $\theta$ 联合优化，而是先用大 batch 数据定好 $\theta$、再用小 batch 数据 grid search 选 $s$。

## 实验关键数据

### 主实验
Pythia + OpenWebText2 + LM1B 三组 LLM 数据上对 Chinchilla method 3 的外推预测能力：

| 数据 | 评测维度 | Compute Gap | Chinchilla Holdout Huber ×10⁻⁵ | NQS Holdout Huber ×10⁻⁵ |
|------|------|------|------|------|
| Pythia + OWT2 | IsoFLOPs | 1024× | 9.0 | **2.5** |
| Pythia + OWT2 | B-K Plane | 1024× | 9.8 | **5.6** |
| Pythia + OWT2 | IsoFLOPs | 64× | 5.6 | **2.6** |
| Llama + LM1B | IsoFLOPs | 6× | 3.7 | **2.9** |
| Llama + LM1B | B-K Plane | 6× | 8.7 | **8.2** |

NQS 在 IsoFLOPs（变 $N$）和 B-K Plane（变 $B, K$）两种 holdout 上都优于 Chinchilla，差距随外推距离扩大。

### 消融实验
论文做了 LayerNorm correction 必要性、复杂度公平性、外推 robustness 三个 ablation：

| 配置 | 关键效果 | 说明 |
|------|------|------|
| Vanilla NQS（无 LN correction） | 大 batch 拟合好 | 小 batch 训练偏差大 |
| NQS + LN correction（$\gamma \propto 1/\|w\|^2$） | 小 batch 显著改善 | 验证 inspiration 3.3 必要性 |
| Chinchilla on train | Huber ~1.0 | in-distribution 拟合佳 |
| Chinchilla on x20 holdout | 仍可接受 | extrapolation 边界 ~20× |
| Chinchilla on x100+ holdout | 急剧恶化 | functional form 不足以外推 |
| NQS on x4000 holdout | 仍稳定 | mechanistic form 外推力强 |

### 关键发现
- NQS 在 train loss 上比 Chinchilla 高（complexity 大），但 holdout loss 显著更低，说明 mechanistic structure 有效防止 overfit—— complexity 不来自参数数量，而来自 functional form 是否反映真实动力学。
- LayerNorm correction 在小 batch 训练下不可或缺，这给学界一个启发：写 scaling law 不能忽略 normalization layer 对有效学习率的影响。
- NQS 可直接用于 compound resource allocation：在 IsoFLOPs 平面上叠加 time / memory / data 约束，NQS 选出的 $(N, B, K)^*$ 几乎都接近 ground truth optimal——这把 scaling law 从 "做研究" 推到 "做产品" 的实际应用。
- 外推到 4000× compute gap 才开始崩溃，比 Chinchilla 的 ~20× 极限高两个数量级，这对实际预训练规划意义巨大——用 100 PetaFLOP 训练数据可以预测 400000 PetaFLOP 模型的损失。

## 亮点与洞察
- "loss prediction as a better alternative to heuristic-based laws" 这个 framing 很重要：作者把 scaling law 研究方法论本身重新定位为 "loss model fitting + holdout evaluation"，让该领域可以被严格量化评测，避免越来越复杂的 heuristic 堆砌。
- 用数值积分代替 asymptotic closed-form 是 mechanistic modeling 的 powerful trick：保留 theoretical 推导给出的结构，但放弃只在极限下成立的简化，让模型能在 finite 实际配置下精确预测——可迁移到 scaling 之外的其它 theoretical-empirical gap。
- Power law spectrum 的三参数化（$p, q, r$）让 NQS 隐式覆盖 Paquette 等理论里识别的多个 asymptotic phase，避免 case 分析——这种 "用参数空间覆盖 phase" 的思路对学界很有启发。
- 与 LayerNorm correction 类似的扩展机制可以处理 lr schedule、batch schedule，论文 discussion 暗示 NQS 可作为 "scaling law sandbox" 用于 task-specific optimizer 设计，潜力非常大。

## 局限与展望
- 当前 lr 参数 $\gamma_0$ 的影响在 NQS 中比真实 LLM 大，意味着系统对 lr 的建模还不够精确，目前还不能预测 lr × batch / lr × model size 的交互。
- $s$ 必须单独 grid search 而非和 $\theta$ joint optimize，作者承认是数值原因 hack；理想情况下应当统一优化。
- 用 NQS 推断的 $\theta$ 不能直接解释为物理意义上的 Hessian 谱或噪声强度，仍是 fitting parameter——mechanistic 与 interpretable 之间还有距离。
- 实验只覆盖 Pythia / Llama 两个家族 + 标准 Adam 优化器；对 SGD、AdamW、Adafactor 等其它优化器的鲁棒性未知。
- 7+1 自由度比 Chinchilla 的 5 多，虽然在 holdout 上不 overfit，但需要更多训练点才能稳定拟合，论文未给最小数据点数推荐。

## 相关工作与启发
- **vs Chinchilla Method 3（Hoffmann/Besiroglu）**：Chinchilla 是 $L(N, D)$ 的纯 phenomenological power law，外推 20× 后即崩溃；NQS 是 $L(N, B, K)$ 的 mechanistic 模型，外推 4000× 仍稳定，且首次显式建模 batch size。
- **vs Noisy Quadratic Model（Zhang 等 2019）**：NQM 只刻画 estimation error（bias + variance），naive 增 $N$ 反而让 loss 增；NQS 加入投影到前 $N$ 维 + $\mathcal{E}_{\mathrm{appx}}$ 类项，修正了这个 unphysical 行为。
- **vs Linear Regression Scaling（Bordelon 等）**：那一类只给 asymptotic 表达；NQS 通过数值积分扩展到 finite regime，并显式加入 mini-batch 噪声。
- **vs Bergsma 等的最优 batch 拟合**：那些是 loss-model-free 的 heuristic 规律；NQS 提供同时刻画 loss 和最优配置的统一框架，且能处理 compound resource constraints。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个把 batch size 纳入 mechanistic loss model 的 scaling law，并把外推能力推高两个数量级。
- 实验充分度: ⭐⭐⭐⭐ Pythia + OWT2 + Llama + LM1B + compound resource case 都做了，extrapolation curve 详尽。
- 写作质量: ⭐⭐⭐⭐ 从 Chinchilla 痛点 → 三条 theoretical inspiration → mechanistic 构造 → ablation 一气呵成。
- 价值: ⭐⭐⭐⭐⭐ 直接服务于工业级预训练规划，可大幅减少昂贵的 scaling sweep 成本。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Language Model Developers Should Report Train-Test Overlap](../../ICML2025/llm_pretraining/language_model_developers_should_report_train-test_overlap.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](../../ICLR2026/llm_pretraining/predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)
- [\[ICML 2026\] Scaling Depth Capacity via Zero/One-Layer Model Expansion](scaling_depth_capacity_via_zeroone-layer_model_expansion.md)
- [\[ICML 2026\] FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment](flexrank_nested_low-rank_knowledge_decomposition_for_adaptive_model_deployment.md)

</div>

<!-- RELATED:END -->
