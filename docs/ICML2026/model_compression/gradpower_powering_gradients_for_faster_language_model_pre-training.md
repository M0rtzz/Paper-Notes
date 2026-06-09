---
title: >-
  [论文解读] GradPower: Powering Gradients for Faster Language Model Pre-Training
description: >-
  [ICML 2026][模型压缩][梯度变换] GradPower 在喂给任意梯度优化器之前对原始梯度做一次逐元素的"符号保留幂次"变换 $\varphi_p(g_i)=\mathrm{sign}(g_i)\,|g_i|^p$，仅多一行代码、不动 AdamW/Muon 内部逻辑和超参…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "梯度变换"
  - "AdamW"
  - "Muon"
  - "MoE 预训练"
  - "wsd 调度"
---

# GradPower: Powering Gradients for Faster Language Model Pre-Training

**会议**: ICML 2026  
**arXiv**: [2505.24275](https://arxiv.org/abs/2505.24275)  
**代码**: 论文未明确给出仓库链接  
**领域**: LLM预训练 / 优化器 / 训练加速  
**关键词**: 梯度变换、AdamW、Muon、MoE 预训练、wsd 调度

## 一句话总结
GradPower 在喂给任意梯度优化器之前对原始梯度做一次逐元素的"符号保留幂次"变换 $\varphi_p(g_i)=\mathrm{sign}(g_i)\,|g_i|^p$，仅多一行代码、不动 AdamW/Muon 内部逻辑和超参，就能在 LLaMA / Qwen2MoE 从 66M 到 2B 的多个规模上一致拿到更低的终末 loss，尤其在 MoE + wsd 学习率调度下增益最显著。

## 研究背景与动机
**领域现状**：LLM 预训练算力极其昂贵，优化器是最直接的效率杠杆。AdamW 因为坐标自适应学习率成为事实标准，最近一批工作（Muon、Blockwise LR、Lion、SOAP、CAME 等）尝试加曲率信息、矩阵预条件、混合动量、cautious update 来再压一点终末 loss。

**现有痛点**：这些"侵入式"改造往往要重新设计动量、二阶矩、甚至整个 update rule，落到训练 pipeline 里意味着重新调一遍 lr / β1 / β2 / weight decay / clipping，工程成本极高，社区采纳缓慢。

**核心矛盾**：想给 AdamW "再加速"和想"不动现有 pipeline"是直接对立的——所有改 update rule 的方法都会破坏调好的超参组合。

**本文目标**：找到一个能即插即用、不动 AdamW 内部逻辑、不需要重调超参、对所有现代优化器都适用的加速插件。

**切入角度**：作者把优化器写成预条件形式 $\theta_{t+1}=\theta_t-\eta_t\,\mathcal{Q}(\varphi(g_1),\dots,\varphi(g_t))$，所有现有优化器之争其实是在争 $\mathcal{Q}$。那能不能反过来——固定 $\mathcal{Q}=\text{AdamW}$，只在最外层换 $\varphi$？已知 LLM 预训练长期处于"噪声主导"区，梯度幅值差异主要来自噪声，而最近一批 EoS / river-valley / bulk direction 工作显示 loss 下降的关键在于"沿平坦方向的慢动态"。所以 $\varphi$ 的目标应该是把"小但持续"的平坦方向相对放大。

**核心 idea**：对每个梯度分量做 $\varphi_p(g)=\mathrm{sign}(g)\,|g|^p$，$p>1$ 时拉开"大方向被打压、小方向被相对放大"的对比度，从而加快平坦方向上的累积进展；选择 $p=1.2$ 默认值，跨架构 / 跨规模 / 跨调度都鲁棒。

## 方法详解

### 整体框架
GradPower 不是一个新优化器，而是在梯度进入优化器之前插一层"逐元素幂次变换"：反向传播照常拿到 mini-batch 梯度 $g_t\in\mathbb{R}^d$，先过一行 $g_t\leftarrow \mathrm{sign}(g_t)\odot|g_t|^p$，再把这个变换后的梯度当作"梯度"喂进 AdamW / Muon / Blockwise LR / AdaGrad，优化器的更新规则、一阶矩、二阶矩、weight decay、所有超参一律不动。整个改动只多了一次按元素的取符号与取幂、不依赖任何状态，实测在 LLaMA-0.25B / OpenWebText 上每步只增加约 0.4% 墙钟开销（0.7565s vs 0.7534s），相对总训练时间几乎可忽略；梯度裁剪放在变换之前还是之后都不影响最终曲线，两种顺序都能保证更新有界。

### 关键设计

**1. 符号保留的幂次变换 $\varphi_p$：用一行非线性放大平坦方向的累积进展**

整篇方法的核心就是这一个算子 $\varphi_p(g)=\mathrm{sign}(g)\,|g|^p$：保留每个分量的方向，只对幅值做幂运算，$p>1$ 时拉大分量之间的相对差距、同时整体压低绝对幅值，$p<1$ 时反过来。它要回答的问题是"该往哪个方向加速"。作者用一维玩具例子 $g_t\sim\mathrm{Unif}(\mu-\sigma,\mu+\sigma)$ 算出 AdamW 的长期累积更新 $u_t=m_t/(\sqrt{v_t}+\epsilon)$，证明在高噪声区（$\sigma\gg\mu$，对应 LLM 预训练这种 batch 远小于全数据、梯度幅值差异主要来自噪声的体制）最优 $p^\star>1$，而在低噪声区（大 batch）最优 $p^\star<1$——因为大 batch 时需要的是抑制偶发噪声而不是放大它。

之所以"放大小分量"在高噪声区有效，是因为它直接踩在 EoS / river-valley 这套视角上：loss 的下降取决于沿平坦方向（river 方向）的稳态慢动态累积，而不是 sharp 方向上的振荡幅度。那些"信号小但持续稳定"的平坦方向恰恰幅值偏小，$p>1$ 把它们相对抬起来，等于用最小代价人为加快了 river 方向上的进展；而幅值大、多由噪声主导的 sharp 方向被相对打压，振荡也随之收敛。这解释了为什么 $p\to0$ 的极端（sign-SGD / Lion，丢弃全部幅值信息）在大 batch 下不灵——它落在了错误的噪声区。

**2. 基优化器完全不变：把 $\varphi$ 和 $\mathcal{Q}$ 解耦，消除重调超参的采纳门槛**

GradPower 能否被产业接受，瓶颈从来不是算法强弱，而是迁移成本——任何动 update rule 的方法都要重调一遍 lr / $\beta_1$ / $\beta_2$ / weight decay / clipping。作者的处理是把优化器写成预条件形式 $\theta_{t+1}=\theta_t-\eta_t\,\mathcal{Q}(\varphi(g_1),\dots,\varphi(g_t))$，故意让外层变换 $\varphi$ 和内层优化器 $\mathcal{Q}$ 彻底解耦：变换只发生在喂进优化器之前，AdamW 的 $\beta_1,\beta_2,\epsilon,\lambda$、Muon 的正交化、Blockwise LR 的分块系数全部保留原值，已有 recipe 里花大算力调好的超参一个都不用碰。

这样唯一新增的自由度就只剩 $p$，且只需在一个小规模上 grid search 一次。论文用 LLaMA-0.2B / C4 一次性定下 $p=1.2$，然后跨模型尺寸（66M→2B）、跨架构（dense LLaMA、MoE Qwen2MoE）、跨数据（C4、OpenWebText）、跨调度（cos、wsd）全部沿用同一个值。落地时任何在跑的 pipeline 加一行 `g = g.sign() * g.abs().pow(p)` 即可，"不重调超参"是它能立刻被采用的最关键卖点。

**3. 与现代优化器和调度器的正交叠加：定位成通用插件而非 AdamW 专属变种**

因为接口是 $\varphi$ 这一侧，GradPower 抓的自由度和"改 $\mathcal{Q}$ 内部"的那批方法天然正交，可以直接和它们相加：把 Muon 的正交化 update rule 当作 $\mathcal{Q}$、前面套一层 $\varphi_{1.2}$ 就得到 MuonPower，把 AdamW + Blockwise LR 当作 $\mathcal{Q}$ 套一层就得到 BlockwisePower。实验里 AdamWPower 的增益（0.015）和 Blockwise LR 的增益（0.030）联合后约等于 0.045，近似线性叠加，说明 GradPower 拿到的"沿平坦方向放大累积"这个红利，确实和"分块学习率 / 矩阵预条件"是不同的维度。

在 wsd 调度下这个优势还会随 stable 阶段时长稳步扩大，正好契合 DeepSeek-V3 等现代 pipeline"长 stable + 短 decay"的趋势。作者借此把 GradPower 定位成"通用插件"——只要能套进 $\varphi$ 接口，任何优化器都能吃到这份红利，从而能跟着社区后续的新优化器一起演化。

### 损失函数 / 训练策略
不引入额外 loss，沿用语言模型的 next-token 交叉熵；clipping 阈值 1.0、weight decay 0.1、$\beta_1=0.9,\beta_2=0.95$ 保持 LLaMA 原 recipe；lr_max 先按 AdamW 在 `{1e-4, 2e-4, 3e-4, 6e-4, 1e-3, 1.5e-3}` 上调到最优，AdamWPower 直接沿用这个 lr_max。$p=1.2$ 在所有主实验里固定。

## 实验关键数据

### 主实验
LLaMA-2B 在 C4 上预训练后做 zero-shot 评估，AdamWPower 6 个任务里赢 5 个：

| 数据集 | 指标 | AdamW | AdamWPower(p=1.2) | 提升 |
|--------|------|-------|--------------------|------|
| ARC-E | acc | 60.02 | 60.35 | +0.33 |
| HellaSwag | acc | 44.65 | 44.93 | +0.28 |
| OBQA | acc | 24.80 | 25.00 | +0.20 |
| WinoGrande | acc | 56.83 | 59.43 | +2.60 |
| PIQA | acc | 73.56 | 73.61 | +0.05 |
| 6 项平均 | acc | 47.72 | 48.26 | +0.54 |

预训练终末 loss 上，跨 66M / 0.2B / 0.4B / 1B / 2B、跨 C4 / OpenWebText、跨 cos / wsd 共多套组合 AdamWPower 全胜；MoE 增益更显著——Qwen2MoE-2B 的绝对 loss 提升 0.028 反而比 LLaMA-2B 的 0.022 还大（且 Qwen2MoE-2B 起点 loss 已降到 1.93，更难再降）。

### 消融实验
$p$ 选取与 batch size 的关系（ResNet-34 / CIFAR-10 验证 GradPower 不止适用于语言模型）：

| batch size | p=0.8 | p=0.9 | p=1.0 | p=1.1 | p=1.2 |
|------------|-------|-------|-------|-------|-------|
| 128 | 94.35 | 94.22 | 93.98 | 93.38 | 93.15 |
| 64 | 94.22 | 94.22 | 94.10 | 93.97 | 93.77 |
| 32 | 94.04 | 94.15 | 94.30 | 94.25 | 93.85 |

观察到清晰趋势：**batch 越大（噪声越低），最优 $p$ 越小**——大 batch 时最优 $p<1$，小 batch / 语言模型预训练时最优 $p>1$，与理论分析中"高噪声区放大平坦方向"完全吻合。

### 关键发现
- GradPower 增益在 **MoE + wsd** 组合下最大：1B 和 2B 的 Qwen2MoE 在 AdamW 下出现 loss spike，AdamWPower 几乎消除——作者推测幂变换抑制了 sharp 方向上的高频振荡，因此训练更稳定。
- $p=1.2$ 这个值在 LLaMA-0.2B 调出后，从 66M 到 2B、从 dense 到 MoE、从 cos 到 wsd 一致最优，**跨规模可迁移性极强**，避免了"换模型就重调"的代价。
- 与 Blockwise LR / Muon 的增益近似可加——意味着 GradPower 抓的是与"分块学习率""矩阵预条件"正交的另一个自由度（沿平坦方向的累积放大），能与未来新优化器持续叠加。

## 亮点与洞察
- 用一行代码 + 一个超参 $p$ 拿到 MoE/LLM 预训练上 0.02–0.03 量级的终末 loss 提升，从产业落地角度的"投入产出比"几乎无敌——这是 ICML 体系里少见的"零工程成本"加速器。
- 把"优化器之争"重新框成 $\varphi$ 与 $\mathcal{Q}$ 的分解，是一个非常干净的视角；以前所有人都在动 $\mathcal{Q}$，作者第一个认真挖 $\varphi$ 这一侧的空间，开了一个新的设计维度。
- 高噪声区 $p^\star>1$、低噪声区 $p^\star<1$ 的相变可以解释为什么过去类似想法（如 sign-SGD 对应 $p=0$）在大 batch 下不灵——它们恰好落在错误的噪声区。这个洞察可以反过来指导何时该用 Lion-类、何时该用 GradPower-类。
- 设计哲学"不动既有 pipeline 才会被采用"值得迁移到其他系统级工作——很多论文输给 AdamW 不是因为算法弱，而是因为重调超参的迁移成本太高。

## 局限与展望
- 论文承认对 MoE 上"幂变换为何抑制 loss spike"只给了直觉解释（抑制 sharp 方向振荡），缺少严谨证明；理论部分主要在 1D 玩具例子和一般非凸 AdaGrad 上展开，离真实 Transformer 优化几何仍有距离。
- 实验最大规模到 2B，没有验证在更大尺寸（10B+）下 $p=1.2$ 是否仍然最优——理论暗示噪声水平随 batch 变化，超大模型实际训练通常用更大 token batch，可能需要把 $p$ 重调到 $<1.2$。
- "river-valley / flat direction"图景虽然在最近文献里流行，但定义不够形式化（用小随机梯度方向近似 Hessian 小特征向量方向），这也直接影响 GradPower 解释的严密性。
- 改进方向：可以做 per-layer / per-block 的自适应 $p$（结合 Blockwise LR 思路），或者根据训练阶段动态调节 $p$（早期高 $p$ 探索平坦方向、后期降到 $p\approx 1$ 精调），论文里有一些迹象支持这个思路。

## 相关工作与启发
- **vs Muon (Jordan et al. 2024 / Liu et al. 2025a)**：Muon 改 $\mathcal{Q}$，引入矩阵正交化预条件；GradPower 改 $\varphi$，做逐元素非线性变换。实验显示二者可叠加（MuonPower），说明它们抓取的是优化器空间里正交的两个维度。
- **vs Blockwise LR (Wang et al. 2025)**：Blockwise LR 给不同 Transformer block 分配不同 lr，仍然是 $\mathcal{Q}$ 内部的细化；和 GradPower 联合时增益几乎线性相加（0.030 + 0.015 ≈ 0.045）。
- **vs sign-SGD / Lion (Chen et al. 2024b)**：本质上就是 $p\to 0$ 的极限版本，丢弃所有幅值信息。GradPower 显示在 LLM 预训练的高噪声体制下 $p$ 应当 $>1$ 而不是 $\to 0$，对 Lion 类"激进 sign 化"的方向提出反向证据。
- **vs Cautious update (Liang et al. 2024) / Variance reduction (Yuan et al. 2024)**：这些都在 $\mathcal{Q}$ 内部修改 update rule，要重调超参；GradPower 的卖点是"完全外挂"。
- 启发：可以在 RLHF / SFT / fine-tune 阶段也试试 GradPower——这些阶段也存在 small-batch / 高噪声 / 平坦方向慢动态的特征，理论上同样适用，是后续低成本扩展的自然方向。

## 评分
- 新颖性: ⭐⭐⭐⭐ "改 $\varphi$ 而不是改 $\mathcal{Q}$"的视角足够清晰，但单点幂变换形式简单，理论部分主要是 1D 玩具例子的扩展。
- 实验充分度: ⭐⭐⭐⭐⭐ 跨架构、跨规模、跨数据、跨调度、跨优化器、跨 batch 全打了一遍，还顺手验证了 CV，覆盖度非常诚意。
- 写作质量: ⭐⭐⭐⭐ 动机讲得通顺，理论与实验之间通过 noise-to-signal ratio 这条线串得紧凑；但 river-valley 的术语依赖大量前作，新读者要先做功课。
- 价值: ⭐⭐⭐⭐⭐ 一行代码 + 一个超参就能在 MoE 预训练 + wsd 调度（即现代主流配置）拿到稳定增益，落地价值极高，可能成为 LLaMA 后训练 recipe 里的下一个默认插件。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization](turning_stale_gradients_into_stable_gradients_coherent_coordinate_descent_with_i.md)
- [\[ICML 2026\] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models](bounded_hyperbolic_tangent_a_stable_and_efficient_alternative_to_pre-layer_norma.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)

</div>

<!-- RELATED:END -->
