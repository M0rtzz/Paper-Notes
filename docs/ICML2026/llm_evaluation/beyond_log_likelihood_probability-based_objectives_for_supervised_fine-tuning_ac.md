---
title: >-
  [论文解读] Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum
description: >-
  [ICML 2026][LLM评测][监督微调] 本文系统研究了 SFT 中概率类目标函数的行为规律，发现标准 NLL 并非普适最优：在模型先验强的任务上先验倾向（prior-leaning）目标如 $-p$ 显著优于 NLL（最高提升 16%），而在先验弱的任务上 NLL 仍然占优…
tags:
  - "ICML 2026"
  - "LLM评测"
  - "监督微调"
  - "损失函数"
  - "模型能力连续谱"
  - "概率目标"
  - "先验倾向"
---

# Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum

**会议**: ICML 2026  
**arXiv**: [2510.00526](https://arxiv.org/abs/2510.00526)  
**代码**: https://github.com/GaotangLi/Beyond-Log-Likelihood  
**领域**: LLM训练  
**关键词**: 监督微调, 损失函数, 模型能力连续谱, 概率目标, 先验倾向  

## 一句话总结
本文系统研究了 SFT 中概率类目标函数的行为规律，发现标准 NLL 并非普适最优：在模型先验强的任务上先验倾向（prior-leaning）目标如 $-p$ 显著优于 NLL（最高提升 16%），而在先验弱的任务上 NLL 仍然占优，揭示了由模型能力连续谱（model-capability continuum）主导的目标函数选择原则。

## 研究背景与动机

**领域现状**：监督微调（SFT）是 LLM 后训练的标准范式，默认训练目标为负对数似然（NLL，$-\log p$）。NLL 在从零训练的经典理论中被证明是最优的。

**现有痛点**：大量实践发现 SFT 泛化能力有限，但这一缺陷可能并非来自模仿学习范式本身，而是来自 NLL 目标函数。后训练与从零训练有本质区别：模型已经编码了大量任务相关先验，且监督序列常达数千 token 并可能含噪声。NLL 强制模型逐 token 复制参考答案，对低概率 token 施加过大的梯度信号，可能损害泛化。

**核心矛盾**：NLL 的最优性假设（独立同分布、从零学习）在后训练场景中被违反——模型已有先验，且先验强度因任务领域而异。将 NLL 推广到参数化目标族 $f^\alpha(p) = (1-p^\alpha)/\alpha$（$\alpha \to 0$ 时退化为 NLL）后，发现 $\alpha=1$（即 $-p$）和 $\alpha=10$ 在数学推理上显著超越 NLL。

**本文目标**：不是推销某个"万能"损失，而是系统刻画在什么条件下哪种目标最优，建立"模型能力连续谱"的分析框架。

**切入角度**：作者观察到目标函数的有效性与基础模型对任务的先验强度密切相关。先验强（如数学，预训练中 25% token 为数学相关）时，先验倾向目标更好；先验弱（如 figfont 字谜，预训练中完全未见）时，NLL 更好。

**核心 idea**：SFT 目标的选择应匹配模型能力——用模型能力连续谱替代"一刀切"的 NLL。

## 方法详解

### 整体框架
给定预训练模型 $p_\theta$ 和 SFT 数据集 $T$，标准 SFT 最小化 NLL $\mathcal{L}_{-\log p}$。本文把它推广成一个通用概率目标 $\mathcal{L}_{f(p)}(\theta) = \mathbb{E}_{(x,\tilde{y})\sim T}[\sum_t f(p_\theta(y_t|y_{<t},x))]$（$f:[0,1]\to\mathbb{R}$ 非增可微），然后用梯度权重的形状把不同 $f$ 分成「先验倾向」和「先验厌恶」两类，再沿「模型能力连续谱」找出每一类各自的甜区。整条线索是：先给一个能在 NLL 与 $-p$ 之间连续滑动的目标族，再用一个标量诊断量决定该滑到哪一端。

### 关键设计

**1. 统一概率目标族 $f^\alpha(p)$：把 NLL 和 $-p$ 装进一根可调旋钮。**

NLL 之所以被默认采用，是因为从零训练的经典理论说它最优；但后训练里模型已有先验、序列又长又带噪，这个最优性前提其实被违反了。与其在 NLL 之外另设一个损失，本文干脆定义一族目标 $f^\alpha(p) = (1-p^\alpha)/\alpha$，让 $\alpha$ 这一个标量参数连续地控制「有多尊重先验」。当 $\alpha \to 0$ 它退化为 $-\log p$（NLL），$\alpha=1$ 就是 $1-p$（等价于最大化平均预测准确率），$\alpha \geq 1$ 时函数为凹、$0 \leq \alpha \leq 1$ 时为凸。关键在于它对正确 logit 的梯度权重为 $W_f(p) = p^\alpha(1-p)$：$\alpha$ 越大，低概率 token 的梯度信号衰减得越快——也就是越不去硬逼模型复制那些它本来就觉得不该出现的 token。这样凸/凹性就成了「先验利用程度」的天然代理指标，且整段谱系可以连续插值，便于扫描。

**2. 模型能力连续谱：用一个预测概率决定该用哪种目标。**

同一个损失在数学推理上大胜、在字谜任务上惨败，背后的变量是「基础模型对这个任务的先验有多强」。本文用模型在训练集上的平均预测概率把任务排成一条连续谱，并切成三段诊断。**Model-Strong（MS）端**是预训练充分覆盖的领域（如数学，Qwen2.5-Math-7B 平均预测概率高达 $0.81$），此时先验倾向目标最优——因为低概率 token 多半是噪声，硬学反而有害。**Model-Weak（MW）端**是预训练几乎没见过的领域（如 figfont 字谜，平均概率约 $0.01$），此时 NLL 最优——先验帮不上忙，必须老老实实逐 token 学。**Model-Intermediate（MI）区**是部分覆盖的领域（如医学推理，约 $0.50$），两类目标打平。这条谱的实用价值在于：平均预测概率本身就是个可直接计算的诊断量，高于 $0.5$ 考虑先验倾向目标、低于 $0.1$ 老实用 NLL。

**3. 梯度权重与先验倾向分类：把「NLL 过度关注低概率 token」做成严格定理。**

前两点的直觉需要数学背书，否则只是经验观察。Lemma 3.1 给出任意目标 $f$ 对正确 token logit 的梯度权重

$$W_f(p) = -f'(p) \cdot p(1-p),$$

Proposition 3.2 进一步证明：凸函数的 $W_f$ 峰值落在 $[0, 0.5]$，意味着梯度主要砸向低概率 token，是「先验厌恶」；凹函数峰值落在 $[0.5, 1]$，梯度集中在已经高概率的 token 上，是「先验倾向」。据此 $-\log p$（凸）被归为先验厌恶，$-p$、$-p^{10}$（凹）归为先验倾向，正好对上经验现象。最后 Theorem 6.4 用梯度流理论收口：在 MS 端先验倾向目标的风险下降更快，在 MW 端 NLL 下降更快——把「什么时候该滑到哪一端」从经验规律升级成了可证明的结论。

## 实验关键数据

### 主实验：Model-Strong 端（数学推理）

| 模型 | 目标 | Math500 | Minerva | OlympiadBench | AIME24 | AMC23 | 均值 |
|------|------|---------|---------|---------------|--------|-------|------|
| Qwen2.5-Math-1.5B | Base | 30.71 | 8.81 | 14.88 | 2.49 | 17.97 | 14.97 |
| | $-\log p$ (NLL) | 42.52 | 12.71 | 12.09 | 0.62 | 17.03 | 17.00 |
| | $-\log p \cdot \mathbf{1}_{p \geq 0.2}$ | 63.95 | 24.79 | 26.08 | 7.09 | 38.28 | **32.04** |
| | $-p$ | **65.27** | **26.18** | **26.66** | 6.88 | 38.13 | **32.75** |
| Qwen2.5-Math-7B | Base | 40.38 | 13.66 | 16.36 | 6.04 | 24.69 | 20.23 |
| | $-\log p$ (NLL) | 51.90 | 18.88 | 17.37 | 2.70 | 22.50 | 22.67 |
| | $-\log p \cdot \mathbf{1}_{p \geq 0.2}$ | **67.85** | **32.47** | **33.90** | **8.76** | **47.81** | **38.16** |
| | $-p$ | 68.47 | 31.99 | 32.26 | 8.75 | 41.09 | 36.51 |

### Model-Weak 端与连续谱总结

| 维度 | Model-Strong (MS) | Model-Intermediate (MI) | Model-Weak (MW) |
|------|-------------------|------------------------|-----------------|
| 代表领域 | 数学推理 | 医学推理 | figfont 字谜 |
| 模型预测概率 | 0.76–0.81 | ~0.50 | ~0.01 |
| 最优目标 | $-p$ / 阈值化 NLL | 两者相近 | NLL ($-\log p$) |
| $-p$ vs NLL 差距 | $-p$ 胜出最高 +16% | 差距 <2% | NLL 胜出（$-p$ 几乎 0 分） |
| 根本原因 | NLL 过度关注低概率噪声 token | 先验不够强也不够弱 | $-p$ 强化错误高概率 token |

### 关键发现
- 在 MS 端，仅用训练集 top 10% 高概率 token 训练即可超越全 token NLL，证明低概率 token 在先验强时本质上是噪声
- $\alpha$ 从 0.1 增到 10.0 时，MS 端准确率单调上升，MW 端单调下降，形成完美的"X 形"交叉
- 通用指令微调（Qwen2.5-3B/7B/14B）中，随模型规模增大，$-p$ 的优势逐步显现，验证连续谱在同一数据分布内也成立
- 编码任务同样属于 MS 端（$-p$ 优于 NLL），低资源多语言属于 MW 端（NLL 优于 $-p$），证明连续谱跨领域泛化

## 亮点与洞察
- 核心洞察极其简洁：一个标量参数 $\alpha$ 就能控制先验利用程度，而最优 $\alpha$ 由模型对任务的先验强度决定。这比设计复杂的自适应损失优雅得多
- 阈值化 NLL（$-\log p \cdot \mathbf{1}_{p \geq 0.2}$）作为一种实用中间方案，不需要改变损失形状，只需过滤低概率 token 就能获得接近 $-p$ 的效果，实施成本几乎为零
- 训练集平均预测概率可作为目标选择的定量诊断指标——高于 0.5 考虑先验倾向目标，低于 0.1 用 NLL。这一规则可直接迁移到任何 SFT 场景

## 局限与展望
- 目标选择仍为静态的：训练过程中模型能力在变化，但本文全程使用固定目标，未探索自适应调度 $\alpha$
- 模型能力的度量依赖预测概率，在模型严重 miscalibrated 或产生自信幻觉时可能误导目标选择
- MI 区域（如医学推理）中两类目标均无明显优势，改进可能需要从损失函数以外的方向（如数据质量、领域知识注入）入手
- 理论分析基于简化假设（如 MW 端模型预测均匀分布），真实场景更复杂

## 相关工作与启发
- Wu et al. (2026) 提出的均匀梯度重加权本质上等价于 $-p$ 目标，本文将其纳入统一框架并给出了失效条件
- Focal loss 在本框架下属于"比 NLL 更先验厌恶"的目标，Huber 类概率损失属于先验倾向
- 启发：后训练不应盲目套用 NLL，评估基础模型的先验强度后选择匹配的目标可能是最简单有效的 SFT 提升方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning](agzo_activation-guided_zeroth-order_optimization_for_llm_fine-tuning.md)
- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[AAAI 2026\] Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](../../AAAI2026/llm_evaluation/low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](../../NeurIPS2025/llm_evaluation/hyperbolic_fine-tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
