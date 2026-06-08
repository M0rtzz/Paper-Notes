---
title: >-
  [论文解读] DLLM-JEPA: Joint Embedding Predictive Architectures for Masked Diffusion Language Models
description: >-
  [ICML 2026][可解释性][JEPA] 在掩码扩散语言模型 (masked diffusion LM) 的微调阶段加上 JEPA 表示对齐目标：把同一句话用不同掩码比例切成"低掩码上下文视图"和"高掩码目标视图"…
tags:
  - "ICML 2026"
  - "可解释性"
  - "JEPA"
  - "掩码扩散语言模型"
  - "表示学习"
  - "微调"
  - "EMA 目标编码器"
---

# DLLM-JEPA: Joint Embedding Predictive Architectures for Masked Diffusion Language Models

**会议**: ICML 2026  
**arXiv**: [2606.00091](https://arxiv.org/abs/2606.00091)  
**代码**: 待确认  
**领域**: LLM 预训练 / 表示学习 / 扩散语言模型  
**关键词**: JEPA, 掩码扩散语言模型, 表示学习, 微调, EMA 目标编码器

## 一句话总结
在掩码扩散语言模型 (masked diffusion LM) 的微调阶段加上 JEPA 表示对齐目标：把同一句话用不同掩码比例切成"低掩码上下文视图"和"高掩码目标视图"，仅对上下文视图做一次带梯度前向同时算扩散 loss 和 JEPA embedding、对目标视图用 EMA 副本无梯度前向，相比 LLM-JEPA 节省 33% 训练 FLOPs，并在 4 个任务 × 2 个 backbone 上稳定涨点（GSM8K 最高 +18.7 pp）。

## 研究背景与动机
**领域现状**：大语言模型的主流训练范式是 input-space 重建——自回归下一 token 预测（GPT 家族）或掩码 token 重建（BERT）。而视觉领域近年大量转向 Joint Embedding Predictive Architecture (JEPA)：在 latent space 中用一个视图的 embedding 预测另一个视图的 embedding，避免像素级重建带来的低层 bias，从而学到更抽象的表示（I-JEPA, V-JEPA）。

**现有痛点**：把 JEPA 搬到语言模型只有 LLM-JEPA 一次尝试，它把 (text, code) 对视为"同一知识的两个视图"加 JEPA loss。但它有两个深层缺陷：① **显式视图依赖**——必须有天然成对数据 (text↔code)，没法像视觉那样靠数据增强；作者自己承认这是关键限制。② **计算开销翻倍**——自回归模型要因果 mask + block-causal attention，两个视图都得带梯度前向，整个训练 step 的 FLOPs 是普通 SFT 的 2 倍。

**核心矛盾**：JEPA 的"两个视图 + latent 预测"范式天然假设可以并行地、双向地编码两个视图；但自回归 LM 的因果性强行打破了这一假设，导致既要造视图又要付出双倍算力。

**本文目标**：找到一种 LM 架构，让 JEPA 的两个视图可以从单个输入自然产生（无需配对数据），并且只需要一个带梯度前向就能同时拿到任务 logit 和 JEPA embedding。

**切入角度**：作者观察到掩码扩散语言模型（LLaDA, MDLM, SEDD）天然满足这两点——它用**双向 attention** + **随机掩码 denoising**，其训练过程在结构上就和 JEPA 的"view prediction"同构：不同掩码比例就是天然的两个视图。

**核心 idea**：用扩散噪声 schedule 当数据增强器（同一句子采两个掩码率 $t_L<t_H$ 生成两视图），上下文视图的单次带梯度前向同时输出扩散 logits 和 pooled embedding，目标视图用 EMA 副本无梯度前向，省掉 LLM-JEPA 一半的反传成本。

## 方法详解

### 整体框架
输入是一句干净文本 $x_0$。先按掩码扩散前向过程独立采样两个掩码率 $t_L=0.2$（context view）和 $t_H=0.7$（target view），分别把 $x_0$ 加噪成 $x_{t_L}$（20% 位置变 [MASK]）和 $x_{t_H}$（70% 位置变 [MASK]）。online backbone $f_\theta$ 对 $x_{t_L}$ 做**一次带梯度前向**，同时输出：(a) 每个掩码位置的 token 分布——用于标准扩散 loss $\mathcal{L}_\text{diff}$；(b) 对非掩码、非 padding token 做 mean pooling + LayerNorm 得到 JEPA 上下文嵌入 $z_{t_L}$。target 编码器 $f_{\theta'}$ 是 $f_\theta$ 的 EMA 副本（decay $\tau=0.996$），在 `no_grad` 下对 $x_{t_H}$ 前向得到 $z_{t_H}$。轻量 predictor $g_\phi$（$k$ 层 transformer decoder）把 $z_{t_L}$ 映射成 $\hat z_{t_H}=g_\phi(z_{t_L})$。总 loss 是扩散 + cosine 形式的 JEPA 对齐：$\mathcal{L}_\text{total}=\mathcal{L}_\text{diff}+\lambda(1-\cos(\text{sg}(z_{t_H}), \hat z_{t_H}))$。每步算力 $\approx 4F$（1 带梯度前向 + 1 无梯度前向 + 1 反传≈2F），比 LLM-JEPA 的 $6F$ 少 33%。

### 关键设计

**1. 用扩散噪声 schedule 当数据增强器，免费造出无配对的两视图。**

LLM-JEPA 最大的桎梏是必须有天然成对的 (text, code) 数据才能造出两个视图，这把它锁死在少数有配对资源的场景。本文的关键观察是：掩码扩散的前向过程 $q(x_t^i|x_0^i)$ 本身就是一个随机掩码增强器，不同掩码率天然对应不同抽象程度的视图——低掩码率的 $x_{t_L}$ 保留大部分 token，是"近完整上下文"；高掩码率的 $x_{t_H}$ 只剩稀疏 token，是"高度抽象的目标"。于是从同一句 $x_0$ 采两个掩码率 $t_L<t_H$ 就直接得到 JEPA 所需的 context view 和 target view（主实验固定 $t_L=0.2, t_H=0.7$，base-preservation 实验改用更宽的 Wide-tt 配置 $(0.1, 0.9)$），既彻底摆脱配对数据、适用于任意文本数据集，又因为视图生成与扩散训练目标共用同一套噪声 schedule 而零额外数据成本。本质上是把 vision JEPA "靠 augmentation 造视图"和扩散 LM "靠 random masking 造训练样本"这两件原本正交的事用一个 schedule 缝在一起，这是整个方法能成立的支点。

**2. 单次带梯度前向同时产出扩散 logits 和 JEPA 嵌入，目标分支只走 EMA 无梯度副本。**

LLM-JEPA 慢的根源不是 JEPA 本身贵，而是自回归的因果 mask 逼着两个视图都做带梯度前向，整个 step 翻倍到 $6F$。扩散 LM 的双向 attention 恰好解开这个结：$f_\theta(x_{t_L})$ 的同一份 hidden state，既能接 token classifier 拿扩散 logits 算 $\mathcal{L}_\text{diff}$，又能 pool 成向量 $z_{t_L}=\text{Pool}(f_\theta(x_{t_L}))$ 当 JEPA 输入，一次带梯度前向两用，根本不需要第二次前向。target 视图则交给 $f_\theta$ 的 EMA 副本 $f_{\theta'}$（decay $\tau=0.996$）在 `no_grad` 下前向得到 $z_{t_H}$，再由 $k\in\{1,...,5\}$ 层 decoder 预测器 $g_\phi$ 映射出 $\hat z_{t_H}=g_\phi(z_{t_L})$。这样目标分支没有反传、没有第二份梯度内存、没有第二份 optimizer state，每步算力从 LLM-JEPA 的 $6F$（+100%）压到 $4F$（1 带梯度前向 + 1 无梯度前向 + ≈2F 反传，+33%），把 JEPA 真正"白嫖"进了扩散 LM 的微调主循环。

**3. 用扩散主目标当额外 anchor，让 cosine-only 的 JEPA 不塌缩。**

cosine-only 的对齐目标在视觉里是被反复警告"容易 collapse"的配方，本文却不引入任何对比负样本、也不依赖 VICReg 那类方差/协方差正则。防塌缩靠四件事联手：EMA target 慢速演化提供 non-trivial 目标，stop-gradient（损失写成 $\mathcal{L}_\text{JEPA}=1-\cos(\text{sg}(z_{t_H}), \hat z_{t_H})$）阻断 target 分支的退化梯度通路，非对称的 predictor $g_\phi$ 引入非平凡不动点，而真正的关键 anchor 是同步优化的扩散 denoising loss——它强行约束 token 级输出分布，从根上不允许 backbone 退化成常数映射。前三件继承自 I-JEPA/BYOL，第四件是扩散主目标自带的任务监督，把"防 collapse"从一个架构 trick 变成有监督在场的天然性质。实测验证有效：fine-tune 后 pooled embedding 的 effective rank 仍是 42–44（base 模型 42–43）、per-dim std 0.73–0.95、cosine diversity 0.25–0.28，与 baseline 几乎一致，没有降秩也没有方差塌缩。

### 损失函数 / 训练策略
总目标 $\mathcal{L}_\text{total}=\mathcal{L}_\text{diff}+\lambda\,\mathcal{L}_\text{JEPA}$；其中 $\mathcal{L}_\text{diff}=\mathbb{E}_{t,x_t}[-\frac{1}{|\mathcal{M}_t|}\sum_{i\in\mathcal{M}_t}\log p_\theta(x_0^i|x_t)]$ 是标准掩码扩散交叉熵，$\mathcal{L}_\text{JEPA}=1-\cos(\text{sg}(z_{t_H}), g_\phi(z_{t_L}))$。训练用 AdamW + 8×A100-80G + gradient checkpointing，2 epoch 全参微调；主实验 lr=$1\times 10^{-5}$、$(t_L,t_H)=(0.2,0.7)$；base-preservation 实验用更温和的 Wide-tt 配置 lr=$1.4\times 10^{-6}$、$(0.1,0.9)$。超参网格 $\lambda\in\{0.5,1,2\}$、$k\in\{1..5\}$、EMA $\tau=0.996$。

## 实验关键数据

### 主实验
4 个任务 × 2 个 backbone (LLaDA-8B, Dream-7B)，统一 4-shot 评测协议，每格 (task, arch) 选最优 $(\lambda, k)$。

| 任务 | 指标 (4-shot) | LLaDA-8B BL→JEPA | Δ | Dream-7B BL→JEPA | Δ |
|------|---------------|-------------------|------|-------------------|------|
| GSM8K | accuracy | 42.61 → 61.33 | **+18.73** | 34.87 → 46.25 | +11.38 |
| NL-RX | func match | 47.50 → 58.20 | +10.70 | 42.00 → 46.80 | +4.80 |
| Spider | exec match | 35.40 → 39.36 | +3.97 | 20.89 → 25.15 | +4.26 |
| Django | ws-prefix match | 74.40 → 75.40 | +1.00 | 69.58 → 72.35 | +2.77 |

LLaDA-8B GSM8K Wide-tt 三 seed 均值：baseline 65.23±0.93 → DLLM-JEPA 67.07±0.41（+1.84 pp，方差减半）。

### Base preservation（Table 3, LLaDA-8B GSM8K, Wide-tt）

| 方法 | GSM8K 0-shot | Wikitext Δloss (vs base) |
|------|--------------|--------------------------|
| Base (无 fine-tune) | – | 0.0000 |
| Diffusion Baseline ($\lambda=0$) | 65.23 ± 0.93 | −0.0004 |
| L2-to-base anchor ($\lambda_{L2}=10^{-4}$) | 65.18 ± 0.87 | −0.0007 ± 0.0002 |
| **DLLM-JEPA (ours)** | **67.07 ± 0.41** | **−0.0017** |

DLLM-JEPA 是唯一**同时**做到任务涨点 + Wikitext loss 比 base 还低的方法；L2 anchor 能压住参数漂移但 0 任务收益，说明 base preservation 不能只靠参数距离正则换来。MMLU 500 题 sanity check：base 57.40 / BL 57.93±0.42 / JEPA 57.53±0.23，均无 catastrophic forgetting。

### 算力对比 (Table 1，每步 FLOPs)

| 方法 | Fwd (grad) | Fwd (no grad) | Backward | Total | Overhead |
|------|------------|---------------|----------|-------|----------|
| AR Baseline | 1F | – | ≈2F | 3F | – |
| LLM-JEPA | 2F | – | ≈4F | 6F | +100% |
| Diffusion Baseline | 1F | – | ≈2F | 3F | – |
| **DLLM-JEPA** | **1F** | **1F** | **≈2F** | **4F** | **+33%** |

### 关键发现
- **几何漂移 vs 功能遗忘 dissociation**：DLLM-JEPA 训练出来的模型相对预训练初始化的 hidden-state drift 反而**更大**（1.3–3.6× baseline，GSM8K 上集中在中间层 transformer），但 Wikitext functional forgetting **更小**（43–58%）。Dream-7B 上复现（1.28× drift），说明不是 LLaDA 特例。作者把它解释为"JEPA objective 不是最小化表示变化，而是**重定向**表示变化"。
- **方差收紧**：LLaDA-8B GSM8K 这种 baseline seed-to-seed spread 高达 ±8.9 pp 的高方差 cell 上，DLLM-JEPA 把方差压到 ±3.9 pp，best-seed 提升甚至到 +18.7 pp。MMLU 上 std 也减半。
- **未塌缩**：fine-tune 后 pooled embedding 的 effective rank 42–44、per-dim std 0.73–0.95、cosine diversity 0.25–0.28，与 baseline 完全一致，cosine-only JEPA 在扩散主目标加持下不塌缩。
- **不和 LLM-JEPA 直接 head-to-head**：作者明确把 LLM-JEPA 定位为"结构动机"而非直接对手，因为二者底层 attention substrate 不同（causal vs bidirectional），所以对照组始终是 diffusion-only fine-tuning on the same backbone。

## 亮点与洞察
- **"扩散噪声 = 天然数据增强"** 的视角转换非常漂亮：把视觉 JEPA 里需要靠 crop/augmentation 凑出来的两视图，直接借用扩散前向过程不同 $t$ 的随机掩码产物，免费完成视图生成，且与训练目标共用一套噪声 schedule。这种"借用已有随机性当 augmentation"的思路可迁移到任何带有随机损坏过程的生成模型（音频扩散、code 扩散、图扩散）。
- **算力账写得极其干净**（Table 1 的 $F/B$ 分解）：明确告诉读者 LLM-JEPA 之所以慢不是因为 JEPA 贵，而是因为 AR causal mask 强制两次带梯度前向；换到双向 attention substrate，整个 cost 就只多了一个 no-grad 前向。这对后续把 JEPA 搬到任何双向架构（BERT 系、encoder-decoder、扩散 LM）都是一份直接可复用的论证模板。
- **"redirect representation 而非 minimize representation change"** 是非常反直觉但很有想象空间的实证发现。传统 catastrophic forgetting 防治范式（EWC、L2-to-base）默认"少动 = 少忘"，本文用 latent-space 的 JEPA 对齐做出了"多动但少忘"的反例，给 fine-tuning dynamics 研究开了一个新窗口：什么样的"动"才是无害的？

## 局限与展望
- **唯一 head-to-head 对手是 diffusion-only baseline**：作者明确不和 LLM-JEPA 直接比，因为底层架构不同；但这也意味着读者看不到"如果给 AR LM 同样 33% 优化，谁更强"的直接证据。
- **只在 2 个 backbone (LLaDA-8B, Dream-7B) 上验证**，且只在 4 个相对小规模任务（Django, Spider, NL-RX, GSM8K）上跑 2 epoch SFT，没有真正的 pretraining-from-scratch 实验，"JEPA 改善 representation"这一更强主张其实没被 pretraining setting 验证。
- **关键超参 $(t_L, t_H)$ 写死在 (0.2, 0.7) 或 (0.1, 0.9)**，没系统扫这个 schedule 的敏感性；Wide-tt vs aggressive 之间的取舍很大程度上是经验调参，缺少理论指导。
- **drift–forgetting dissociation 是相关性而非因果**：作者反复强调"descriptive sense"，没给机制性解释（为什么 middle layer 漂移大反而 forgetting 小？）；如果有 controlled ablation（如手动放大/缩小 drift）会更有力。
- **Wikitext Δloss 在 $10^{-3}$ 量级**，绝对差距小，对"base preservation"的强主张需要谨慎；好在与 task gain 配着读才有意义。

## 相关工作与启发
- **vs LLM-JEPA (Huang et al., 2025)**：LLM-JEPA 把 (text, code) 当成两视图，需要配对数据 + 双带梯度前向 + 自定义 block-causal mask；DLLM-JEPA 在双向扩散 LM 上用扩散噪声造视图、单带梯度前向、标准 bidirectional attention，把 JEPA "贵 100%" 降到 "贵 33%"，且适用于任意文本数据。本质上是把 LLM-JEPA 的范式搬到了一个更合适的 substrate。
- **vs I-JEPA / V-JEPA (Assran et al., 2023; Bardes et al., 2024)**：完全继承 vision JEPA 的"EMA target + stop-gradient + cosine predictor"防 collapse 配方，只是把"图像 patch + augmentation"换成"token + 掩码率"。是该范式从 vision → language 的一次成功移植。
- **vs LLaDA / MDLM / SEDD (Nie et al., 2025; Sahoo et al., 2024; Shi et al., 2024)**：本文不是新的扩散 LM 架构，而是给这些已有扩散 LM 加一个表示层正则化目标，可即插即用到任何 masked diffusion LM 的微调流程。
- **vs EWC / L2-to-base (Kirkpatrick et al., 2017)**：传统 catastrophic forgetting 防治走"参数空间正则"路线（约束参数别动）；本文走"表示空间对齐"路线（不约束参数，但约束 latent 几何）。Table 3 显示后者能在不牺牲任务收益的前提下达到更强的 base preservation。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把扩散噪声 schedule 当 JEPA 视图生成器是非常聪明的观察，但每个零件（JEPA、扩散 LM、EMA target）都不是新发明，主要价值在于干净的组合 + 算力账。
- 实验充分度: ⭐⭐⭐ 2 个 backbone × 4 个任务 + Wide-tt preservation + MMLU sanity 已经讲清主张；但只跑 SFT 2 epoch、没 from-scratch pretraining，对"representation learning" 的强主张支撑略弱。
- 写作质量: ⭐⭐⭐⭐⭐ Scope of comparison / Reporting protocol 等小段写得极其负责，Table 1 的算力账、Table 3 的三方对比、drift–forgetting dissociation 的实证都讲得非常透彻。
- 价值: ⭐⭐⭐⭐ 给扩散 LM 微调圈贡献了一个低成本（+33% FLOPs）的 plug-in 表示正则方法，且揭示了"redirect drift ≠ amplify forgetting"的反直觉现象，对 fine-tuning dynamics 研究有启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](../../ACL2026/interpretability/towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[NeurIPS 2025\] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement](../../NeurIPS2025/interpretability/far_from_the_shallow_brain-predictive_reasoning_embedding_through_residual_disen.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](towards_atoms_of_large_language_models.md)

</div>

<!-- RELATED:END -->
