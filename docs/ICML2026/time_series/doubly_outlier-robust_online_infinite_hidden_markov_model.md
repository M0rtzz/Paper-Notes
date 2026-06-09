---
title: >-
  [论文解读] Doubly Outlier-Robust Online Infinite Hidden Markov Model
description: >-
  [ICML 2026][时间序列][无限隐马尔可夫模型] 本文提出 BR-iHMM：把"鲁棒观测更新（WoLF）"与"批量化状态推断（degenerate sticky HDP prior）"结合起来…
tags:
  - "ICML 2026"
  - "时间序列"
  - "无限隐马尔可夫模型"
  - "在线推断"
  - "鲁棒贝叶斯"
  - "异常值"
  - "Posterior Influence Function"
---

# Doubly Outlier-Robust Online Infinite Hidden Markov Model

**会议**: ICML 2026  
**arXiv**: [2604.14322](https://arxiv.org/abs/2604.14322)  
**代码**: 无  
**领域**: 时间序列 / 贝叶斯在线学习 / Regime Switching  
**关键词**: 无限隐马尔可夫模型, 在线推断, 鲁棒贝叶斯, 异常值, Posterior Influence Function

## 一句话总结
本文提出 BR-iHMM：把"鲁棒观测更新（WoLF）"与"批量化状态推断（degenerate sticky HDP prior）"结合起来，给在线无限隐马模型同时在观测空间和状态空间提供有界的 Posterior Influence Function（PIF），在金融订单簿、电力负荷、合成回归三类含异常值的流式数据上把一步预测 RMSE 最多降低 67%。

## 研究背景与动机

**领域现状**：处理非平稳流数据有两大派系。Bayesian changepoint detection（BOCD）与 Kalman 滤波每次检测到 changepoint 就"重置或遗忘"，无法复用历史 regime；在线 iHMM（HDP-iHMM）维护一个可复用的 regime 库，能在历史 regime 重现时快速回归，更适合金融、电力、连续学习这类"旧 regime 反复出现 + 偶发新 regime"的场景。

**现有痛点**：iHMM 的灵活性是双刃剑——一个异常点同时会（i）污染当前 regime 的参数后验，造成后续预测劣化；（ii）让模型误以为出现新 regime 而创建虚假状态，破坏可解释性和预测精度。已有鲁棒方法要么只关心观测空间（鲁棒 KF/WoLF），要么只关心状态空间的离线 iHMM 修剪，没有把两者在 online 设置下同时解决。

**核心矛盾**：在 HDP-iHMM 框架里，"观测鲁棒"和"状态鲁棒"是**互相独立**的两个 PIF 维度——作者证明 Theorem 4.1：即使观测端用 WoLF 保证 PIF$_{\theta_t}$ 有界，状态端的 PIF$_{s_t}$ 仍可被异常值推向无穷大（因为大残差让"新 regime"在 HDP 先验下变得最具吸引力）。

**本文目标**：（1）形式化定义在线 iHMM 的双重鲁棒性；（2）设计一个同时让 PIF$_{\theta_t}$ 和 PIF$_{s_t}$ 都有界的算法；（3）在不牺牲计算复杂度的前提下保持 online 实时性。

**切入角度**：观测端复用 generalised Bayes 框架下的 WoLF（用 IMQ 权重把异常点似然降权）；状态端则借鉴 batch inference 思路——**异常点单独不足以创建新 regime，必须有连续多个观测的一致证据**。

**核心 idea**：用"degenerate sticky HDP prior"把状态切换强制收缩到批边界（intra-batch 自转移概率 $\kappa_t=\infty$，inter-batch $\kappa_t=0$），从而要求新 regime 必须在长度 $B$ 的窗口内聚集到足够证据；这同时给出了一个可调的鲁棒-自适应权衡参数 $B$。

## 方法详解

### 整体框架
BR-iHMM 的目标是让在线 iHMM 在观测空间和状态空间**同时**抗异常值，整套推断用 Particle Learning（SMC）跑、以 $B$ 步为一个 batch。每个 batch 内先用各粒子的状态 $s_t^{(i)}$ 对未来 $B$ 步做预测 $\hat y_{t+1:t+B}$，再用 IMQ 权重 $w_{l,t}^{(i)} = W(y_{t+b}, \hat y_{l,t+b|t})$ 给观测降权，然后计算只允许在 batch 边界切换状态的 batched posterior $\nu(s_{1:t+B})$，ESS 过低就重采样，最后用 WoLF 更新活跃状态的高斯后验 $\Psi$、用 Antoniak 辅助变量更新 HDP 结构参数 $\Phi$。关键在于 batch 内部强制 self-transition，所以状态采样每 batch 只做一次，从根上避开了 batch 长度 $B$ 带来的路径指数爆炸。

### 关键设计

**1. WoLF 加权观测更新：给单个极端观测对参数后验的影响硬性封顶。**

标准贝叶斯更新在线性高斯模型下的 PIF 是无界的——残差任意大，后验就能被任意拉偏，一个极端观测就能污染当前 regime 的参数。WoLF 把似然换成加权似然 $P(y_t\mid\theta,x_t)^{W(y_t,\hat y_{s_t})^2}$，权重取 IMQ 形式 $W(y,\hat y)^2 = 1/(1 + c^{-2}\|y-\hat y\|_{R_t}^2)$。在线性高斯 emission 下它仍保共轭，闭式更新只把 Kalman 增益里的协方差 $S_{s_t}$ 换成 $S_{s_t} = f(x_t)\Sigma_{s_t}f(x_t)^\top + R_t/w_{s_t,t|t-1}^2$：残差越大 $w^2 \to 0$、$S_{s_t}\to\infty$、Kalman 增益趋近 0，后验直接被冻住。这样既把 PIF$_{\theta_t}$ 锁死，又没丢掉共轭性带来的在线效率。

**2. 批量推断 + Degenerate Sticky HDP：让"新 regime"必须有连续证据才允许诞生。**

只做观测鲁棒是不够的——Theorem 4.1 证明即便 PIF$_{\theta_t}$ 有界，状态端的 PIF$_{s_t}$ 仍可能被异常值推向无穷，因为大残差会让"开个新 regime"在 HDP 先验下变得最诱人，于是一个孤立异常点就能制造虚假状态。BR-iHMM 的对策是把状态决策搬到 batch 层面：定义 batched log posterior $\log\nu(s_{1:t+B}) = \sum_{b=1}^B w_{s_{t+b},t+b|t}^2\log P(y_{t+b}\mid\dots) + \log\sum_{s_{1:t}}P(s_{1:t}|D)P(s_{t+1}|s_t,\Phi_t)\prod_{b=2}^B\mathbb{1}(s_{t+b-1}=s_{t+b})$，再用 sticky HDP 把自转移偏置 $\kappa_t$ 取极限——batch 内部 $\kappa_t=\infty$（强制状态一致）、边界 $\kappa_t=0$。这等价于把 PIF 从"单点扰动"推广到"短序列扰动"（batched PIF）：只有 batch 内多个观测一致地支持"新 regime 更合理"，路径后验才会切换。参数 $B$ 因此成了一个可解释的鲁棒-自适应旋钮——$B$ 越大越抗持续噪声，但检测真实切换的延迟也越长。

**3. Antoniak 辅助变量 + State Pruning：在无限状态名义下，把流式 bookkeeping 控制成常数。**

iHMM 名义上允许无限状态，但 streaming 场景里如果不剪枝，计数矩阵 $\mathbf{N}_t \in \mathbb{N}^{t\times t}$ 会随时间爆炸。BR-iHMM 每个 batch 用 $\mathbf{M}_t \sim \text{Antoniak}(\mathbf{N}_t,\alpha,\beta)$ 采辅助变量更新 HDP 全局权重 $\hat\beta_t$，并对超过 MAX_STATES 的粒子，按使用频率和近期性启发式删掉老旧 regime（连同计数和全局权重一起删）。这样状态数被压在常数级，而 Proposition D.1/D.2 形式化保证 batched 机制的复杂度仍是每 batch O(1) 次状态采样——可扩展性和前两条的双重鲁棒互不打架。

### 损失函数 / 训练策略
- 不训练 NN，纯贝叶斯在线推断；用 JAX 实现，RTX 3090 单卡。
- 超参 $B$、IMQ 阈值 $c$、ESS 阈值 $\tau_{\text{ESS}}$、粒子数 $N$ 由 bayesian-optimization 在训练分区上调；不同任务的 $B$ 范围在附录给出。
- 浓度参数 $\hat\alpha_0,\hat\gamma_0\sim\text{Gam}(1,1)$ 用无信息先验，配合 Escobar–West conjugate update。

## 实验关键数据

### 主实验
一步预测 RMSE（100 次重复均值 ± stdev）：

| 模型 | Synthetic ($d=100$, 1% 异常) | Electricity | OFI |
|------|------|------|------|
| BOCD | 123.12 ± 0.014 | 0.80 ± 0.11 | 0.733 |
| iHMM | 101.7 ± 0.026 | 0.57 ± 0.03 | 0.620 ± 0.080 |
| WoLF-iHMM | 103.8 ± 0.012 | 0.63 ± 0.03 | 0.623 ± 0.089 |
| **BR-iHMM (ours)** | **46.1 ± 0.003** | **0.47 ± 0.04** | **0.616 ± 0.082** |
| offline-iHMM (oracle) | 2.9 | 0.32 | 0.552 |

Synthetic 任务上 BR-iHMM 相对 iHMM 降低 RMSE 约 55%、相对 BOCD 降低 63%；电力数据上 BR-iHMM 是唯一一个识别出 2020 年 3 月 COVID-19 引发 regime switch 的在线模型，iHMM 和 WoLF-iHMM 整段都困在一个 regime。

### 消融实验

| 配置 | Synthetic RMSE | 失败模式 |
|------|------|------|
| iHMM（基线） | 101.7 | 30+ 虚假 regime，每个异常点都触发新状态 |
| WoLF-iHMM（仅观测鲁棒） | 103.8 | 参数后验稳定但状态仍碎裂，**反而略劣于纯 iHMM** |
| BR-iHMM (B=1) | ≈100 | 等价于 WoLF-iHMM |
| **BR-iHMM (B>1)** | **46.1** | 短期校准后稳定，恢复真实 3 个 regime |

### 关键发现
- **单一鲁棒不够**：WoLF-iHMM 比 iHMM 反而略差，验证了 Theorem 4.1——只做观测鲁棒会让 PIF$_{s_t}$ 主导失败模式。
- **B 是关键权衡参数**：附录 Figures E.10 / E.12 显示 $B$ 越大对短异常越鲁棒但检测延迟变长，金融订单簿 OFI 上 $B$ 较小、电力上 $B$ 较大。
- **复杂度优势**：标准 iHMM 在 batch 内允许任意切换会导致路径数 exponentially in $B$；degenerate sticky 把它降到每 batch 一次状态采样，复杂度与 batch 大小无关。
- **预测和分割双赢**：Table 2（segmentation）显示 BR-iHMM 在 changepoint detection 指标上也优于 DSM-BOCD 和 iHMM（unknown-var）。

## 亮点与洞察
- **理论先行**：先把"鲁棒性"严格定义为 PIF 有界，再用 Theorem 4.1/4.2 证明双重鲁棒是双重必要条件，方法设计有据可依。
- **batch-PIF 概念**：把 PIF 从"单点扰动"推广到"短序列扰动"，自然给出 $B$ 这个可解释参数；这种"batched robustness"思路可迁移到其他在线贝叶斯模型（如 GP、流式 VI）。
- **Degenerate sticky HDP 的双用**：既是数学上的状态空间收缩（用 $\kappa_t\in\{0,\infty\}$ 极限），又是计算上的复杂度神器（消除路径指数爆炸），一举两得。
- **Theorem 4.1 的反直觉发现**：单纯加强观测鲁棒反而可能让状态推断更糟（因为残差被压低后，"新 regime"的相对似然反而上升），给后续相关工作敲了警钟。

## 局限与展望
- 只在 LG emission 下做了完整推导，作者声称框架可扩展到指数族但未实证。
- $B$ 是 fixed-a-priori 的超参，需要 BayesOpt 调参；自适应 $B$（如根据 SNR 动态调整）是自然延伸。
- prune 启发式（usage frequency + recency）相对粗糙，对长尾 regime 可能误删；理论上没有保证 pruning 不破坏 PIF 界。
- 实验最大维度 $d=100$，超高维（如 image features）场景未验证 IMQ 权重的有效性。
- 离线 oracle（offline-iHMM）仍显著优于 BR-iHMM（synthetic RMSE 2.9 vs. 46.1），说明 online–offline gap 仍很大，本质是 SMC 粒子数与 burn-in 限制。

## 相关工作与启发
- **vs. 标准 iHMM (Beal et al. 2001; Teh et al. 2006)**: 增加双重鲁棒性，几乎不增加计算开销。
- **vs. WoLF (Duran-Martin et al. 2024)**: WoLF 只做单状态 LG 模型鲁棒；本文把它嵌入 HDP-iHMM 的多状态框架并补足状态空间鲁棒。
- **vs. DSM-BOCD (Altamirano et al. 2023)**: BOCD 不支持 regime 复用；本文同时保留鲁棒和复用能力。
- **vs. offline iHMM (Van Gael et al. 2008)**: 后者用 beam sampling 做离线 MCMC，达到 oracle 性能但需要 1000 次迭代；BR-iHMM 在线一次扫过即可。

## 评分
- 新颖性: ⭐⭐⭐⭐ "双重鲁棒"形式化 + degenerate sticky HDP 的批量化构造组合很新
- 实验充分度: ⭐⭐⭐⭐ 合成 + 电力 + 订单簿三类数据 + 100 次重复，但都偏低维
- 写作质量: ⭐⭐⭐⭐ PIF 定义、Theorem 4.1/4.2 推导、算法伪代码组织得很清晰
- 价值: ⭐⭐⭐⭐ 给金融、传感器、连续学习等"既要复用历史 regime 又要扛异常"的场景提供了完整工具链

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal](distmatch_adaptive_binning_via_distribution_matching_for_robust_sequential_confo.md)
- [\[ICML 2026\] Divide and Contrast: Learning Robust Temporal Features Without Augmentation](divide_and_contrast_learning_robust_temporal_features_without_augmentation.md)
- [\[ICLR 2026\] Online Time Series Prediction Using Feature Adjustment](../../ICLR2026/time_series/online_time_series_prediction_using_feature_adjustment.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](../../ICLR2026/time_series/delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)

</div>

<!-- RELATED:END -->
