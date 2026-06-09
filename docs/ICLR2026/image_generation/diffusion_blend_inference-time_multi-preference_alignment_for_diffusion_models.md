---
title: >-
  [论文解读] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models
description: >-
  [ICLR 2026][图像生成][multi-preference alignment] 提出 Diffusion Blend，通过在推理时混合多个奖励微调模型的反向扩散过程来实现多偏好对齐：DB-MPA 支持任意奖励线性组合、DB-KLA 支持动态 KL 正则化控制、DB-MPA-LS 通过随机 LoRA…
tags:
  - "ICLR 2026"
  - "图像生成"
  - "multi-preference alignment"
  - "inference-time"
  - "backward SDE blending"
  - "KL regularization control"
  - "Pareto-optimal"
---

# Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models

**会议**: ICLR 2026  
**arXiv**: [2505.18547](https://arxiv.org/abs/2505.18547)  
**代码**: 有（GitHub）  
**领域**: 扩散模型 / 多目标对齐  
**关键词**: multi-preference alignment, inference-time, backward SDE blending, KL regularization control, Pareto-optimal  

## 一句话总结
提出 Diffusion Blend，通过在推理时混合多个奖励微调模型的反向扩散过程来实现多偏好对齐：DB-MPA 支持任意奖励线性组合、DB-KLA 支持动态 KL 正则化控制、DB-MPA-LS 通过随机 LoRA 采样消除推理开销，理论上证明了混合近似的误差界并在实验中接近 MORL oracle 上界。

## 研究背景与动机

**领域现状**：RL 微调扩散模型通常固定单一奖励函数和 KL 正则化权重 $\alpha$。微调完成后，模型锁定在特定的 $(r, \alpha)$ 配置上，无法适应不同用户偏好。

**现有痛点**：(a) 用户可能要求不同的美学/语义一致性/人类偏好权衡，需要为每个偏好组合都微调一个模型（开销巨大）；(b) KL 正则化太弱导致 reward hacking，太强导致对齐不足，最优值需要 grid search；(c) Rewarded Soup（权重空间线性组合）过于粗糙，guidance 方法需要可微奖励且计算量大。

**核心矛盾**：部署后偏好的灵活性 vs 微调的固定性。如何在不重新训练的情况下在推理时适应任意偏好组合？

**本文目标** 给定 $m$ 个基础奖励函数各自微调的模型，在推理时按用户指定权重 $w$ 生成 $r(w) = \sum w_i r_i$ 对齐的图像，且支持动态调整 KL 强度。

**切入角度**：从扩散模型的反向 SDE 角度出发，证明对齐模型的漂移项 $f^{(r,\alpha)}$ 可以表示为预训练漂移 + 控制项，通过 Jensen gap 近似将控制项线性化，从而实现反向 SDE 的线性混合。

**核心 idea**：奖励对齐的扩散模型的反向 SDE 漂移项可以线性组合近似任意奖励线性组合的对齐效果。

## 方法详解

### 整体框架

分两阶段：
- **微调阶段**：为每个基础奖励 $r_i$ 独立 RL 微调，获得 $m$ 个微调模型 $\theta_i^{\text{rl}}$
- **推理阶段**：用户指定权重 $w$，将 $m$ 个模型的反向 SDE 漂移按 $w$ 加权混合

### 关键设计

**1. 对齐模型的 SDE 分解（Proposition 1）：把"对齐"从漂移项里剥出来。**

要在推理时混合多个对齐模型，第一步得搞清楚"对齐"到底改了反向扩散的什么。论文证明，任意一个用奖励 $r$、KL 权重 $\alpha$ 微调出来的模型，其反向 SDE 漂移可以干净地拆成两块：预训练漂移加一个控制项，$f^{(r,\alpha)}(x_t, t) = f^{\text{pre}}(x_t, t) - \beta(t)\, u^{(r,\alpha)}(x_t, t)$。这里控制项 $u^{(r,\alpha)} = \nabla_{x_t} \log \mathbb{E}_{x_0 \sim p_{0|t}^{\text{pre}}}[\exp(r(x_0)/\alpha)]$ 就是奖励对采样轨迹施加的全部"拉力"，而 $f^{\text{pre}}$ 在所有对齐模型之间是共享的。这一步的意义在于：既然各模型唯一的差别都集中在 $u^{(r,\alpha)}$ 上，那么混合多个偏好的问题就被归约成了"怎么组合这些控制项"。

**2. Jensen gap 近似把控制项线性化（Lemma 2）：让奖励的线性组合对应漂移的线性组合。**

控制项里那个 $\log \mathbb{E}[\exp(\cdot)]$ 是非线性的，没法直接对奖励做线性叠加。论文交换 log-exp 与期望的顺序，用 $\bar{u}^{(r,\alpha)} = \nabla_x \mathbb{E}[r(x_0)/\alpha]$ 来近似 $u^{(r,\alpha)}$——这就是 Jensen gap 近似，DPS、RGG 等扩散引导方法里早已被广泛使用，且其误差随 $t \to 0$ 趋于 0。一旦控制项变成对奖励的线性算子，期望的线性性立刻生效：对线性奖励 $r(w) = \sum w_i r_i$，就有 $f^{(r(w),\alpha)} \approx \sum w_i f^{(r_i, \alpha)}$。也就是说，想要 $r(w)$ 对齐的生成效果，只需把各单奖励模型的漂移按 $w$ 加权相加——这正是后面三个算法共同的理论地基。

**3. DB-MPA：推理时按用户权重直接混合反向 SDE。**

有了上面的线性化，多偏好对齐就落地成一个极简的推理改动：用户给定权重 $w$，每一步去噪时不用单一模型，而是把 $m$ 个奖励微调模型的噪声预测按权重加权，$\hat{\epsilon}_t = \sum w_i\, \epsilon_{\theta_i^{\text{rl}}}(x_t, t)$，再照常更新。整个过程不重新训练、不需要奖励可微、也不需要推理时搜索，用户改一下 $w$ 就能实时滑动 aesthetics 与 alignment 之间的权衡。代价是每步要前向 $m$ 个模型，开销 $m\times$。

**4. DB-KLA：把 KL 正则化强度也变成推理时可调的旋钮。**

同一套分解还能用来调 KL 强度，而不只是调奖励配比。论文把目标 KL 权重从 $\alpha$ 缩放到 $\alpha/\lambda$，并近似为预训练模型和微调模型的凸组合 $f^{(r, \alpha/\lambda)} \approx (1-\lambda) f^{\text{pre}} + \lambda f^{(r,\alpha)}$。$\lambda$ 越大越偏向对齐（但可能 reward hacking），越小越保守保留预训练质量。这样原本需要 grid search 才能定下来的 KL 权重，现在变成一个推理时连续可调的标量，避免了为找最优 $\alpha$ 反复微调。

**5. DB-MPA-LS：用随机 LoRA 采样消掉 $m\times$ 开销（Proposition 2）。**

DB-MPA 每步评估全部 $m$ 个模型是它最大的实用瓶颈。论文的巧解是：每个去噪步不再把所有模型都算一遍取加权平均，而是按权重 $w$ 当成概率分布，随机采样一个 LoRA adapter（两个奖励用 Bernoulli、多个用 Categorical）来出这一步。Proposition 2 证明，这条逐步随机采样的 SDE 和原来的加权混合 SDE 拥有相同的边际分布——根源在于扩散过程本身的噪声注入，使得"逐步随机选模型"的统计效果等价于"每步加权平均"。于是推理开销从 $m\times$ 降回 $1\times$，几乎不损失质量。值得注意的是，这个等价性是扩散模型独有的：LLM 在离散 token 空间里没有这种连续的随机性可以利用。

### 损失函数 / 训练策略

- 使用 DPOK 算法对 SD v1.5 做 RL 微调
- 每个基础奖励独立微调
- 推理时无需训练，仅修改去噪步的噪声预测

## 实验关键数据

### 主实验（SD v1.5, ImageReward + VILA/PickScore）

DB-MPA 在 Pareto 前沿上全面优于 Rewarded Soup (RS)、CoDe、RGG，接近 MORL oracle 上界。

关键数值特征：DB-MPA 在 $w=0.5$ 时两个奖励都接近各自独立微调模型的 85-90% 性能水平，而 RS 只达到 60-70%。

### 消融实验

| 方法 | 推理开销 | 性能 (vs MORL) |
|------|---------|--------------|
| DB-MPA | $m \times$ | ~95% of MORL |
| DB-MPA-LS | $1 \times$ | ~90% of MORL |
| RS | $1 \times$ | ~70% of MORL |
| RGG | $1 \times$ (+ gradient) | ~60% of MORL |
| CoDe | $N \times$ (search) | ~65% of MORL |

DB-KLA 可以平滑控制 KL：$\lambda > 1$ 增强对齐但可能过拟合，$\lambda < 1$ 保守但保留预训练质量。

### 关键发现
- DB-MPA 在 Pareto 前沿上显著优于 RS（权重空间混合），说明**反向 SDE 混合优于参数空间混合**
- DB-MPA-LS 随机 LoRA 采样近似几乎不损失性能（~5% 差距），但推理开销降至 1×
- DB-KLA 提供了比重新微调更灵活的 KL 控制方式
- Jensen gap 近似在 JPEG compressibility（与 aesthetics 对抗的奖励）上也有效

## 亮点与洞察
- **SDE 混合 vs 参数混合** 的对比清晰有力——Rewarded Soup 在参数空间线性化，DB-MPA 在 SDE 漂移空间线性化，后者更有理论基础且性能更好。核心原因是 SDE 漂移的线性化近似误差有界（Lemma 1），而参数空间的线性化没有类似保证。
- **Proposition 2 (随机 LoRA 采样等价性)** 是一个优雅的理论结果——利用 SDE 的噪声注入使得逐步随机选择等价于加权平均。这在 LLM 中不可能做到（离散 token 空间），是扩散模型独有的优势。
- 推理时灵活性极高：用户可以用滑动条实时调整 aesthetics vs alignment 的 trade-off。

## 局限与展望
- Jensen gap 近似在 $\alpha$ 很小时误差增大（Lemma 1 的 $L_{t,2}$ 项），无法处理极端对齐需求
- 仅在 SD v1.5 上验证，更大模型（SDXL/Flux）的可行性未测试
- DB-MPA 的推理开销为 $m \times$，奖励函数多时不实用（但 DB-MPA-LS 可缓解）
- 线性奖励组合假设限制了表达能力，非线性偏好关系无法处理
- 没有与 DAV/DenseGRPO 等最新对齐方法的对比

## 相关工作与启发
- **vs Rewarded Soup**: 参数空间线性化 vs SDE 漂移线性化。DB-MPA 理论更严谨且性能更好。
- **vs Guidance (RGG/CoDe)**: 不需要可微奖励，不需要推理时搜索，且性能更优。
- **vs LLM DeRa**: 灵感来源相同（混合对齐和基础模型），但针对扩散模型做了 SDE 理论分析和随机 LoRA 采样创新。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SDE 混合的理论框架新颖，随机 LoRA 采样等价性是优雅的贡献
- 实验充分度: ⭐⭐⭐⭐ 多种奖励组合、Pareto 分析、KL 控制实验全面，但仅 SD v1.5
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，图表直观，动机-理论-算法-实验逻辑紧凑
- 价值: ⭐⭐⭐⭐⭐ 为扩散模型的多偏好部署提供了实用且理论扎实的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RNE: plug-and-play diffusion inference-time control and energy-based training](rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)
- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](glass_flows_reward_alignment_diffusion.md)
- [\[AAAI 2026\] Multi-Metric Preference Alignment for Generative Speech Restoration](../../AAAI2026/image_generation/multi-metric_preference_alignment_for_generative_speech_restoration.md)
- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)
- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](test-time_iterative_error_correction_for_efficient_diffusion_models.md)

</div>

<!-- RELATED:END -->
