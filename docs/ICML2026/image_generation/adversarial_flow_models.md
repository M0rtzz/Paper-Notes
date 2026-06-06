---
title: >-
  [论文解读] Adversarial Flow Models
description: >-
  [ICML 2026][图像生成][Adversarial Training] 作者在 GAN 训练目标上加一个最优传输正则 $\|G(z)-z\|^2$，把 GAN 的"任意搬运图"约束成 Wasserstein-2 最优搬运图，让纯 transformer 上的对抗训练第一次能稳定收敛并端到端做单步生成…
tags:
  - "ICML 2026"
  - "图像生成"
  - "Adversarial Training"
  - "Flow Matching"
  - "一步生成"
  - "Optimal Transport"
  - "DiT"
---

# Adversarial Flow Models

**会议**: ICML 2026  
**arXiv**: [2511.22475](https://arxiv.org/abs/2511.22475)  
**代码**: 论文末提到 "The code is available at this repository"（有）  
**领域**: 图像生成 / 扩散与流匹配 / GAN  
**关键词**: Adversarial Training, Flow Matching, 一步生成, Optimal Transport, DiT

## 一句话总结
作者在 GAN 训练目标上加一个最优传输正则 $\|G(z)-z\|^2$，把 GAN 的"任意搬运图"约束成 Wasserstein-2 最优搬运图，让纯 transformer 上的对抗训练第一次能稳定收敛并端到端做单步生成，ImageNet-256 上 1NFE FID 刷到 2.38（XL/2）和 1.94（112 层）。

## 研究背景与动机

**领域现状**：少步/单步图像生成主要靠两条路：(1) 从预训练 flow matching 教师蒸馏一致性模型 / sCM / MeanFlow / Shortcut 等；(2) 用对抗训练做最后润色（GAN-style refinement）。两条路通常都还得保留 flow 主干。

**现有痛点**：一致性方法即便目标是单步生成，也得在所有时间步上传播一致性约束，这会"吃掉"模型容量、累积传播误差、并因为 pointwise / moment matching 损失而图像偏糊。纯 GAN 训练在标准 transformer 上极不稳定，要么靠卷积 + 复杂 trick（R3GAN），要么要冻结特征网络（GAT），无法享受 DiT / 大模型的 scaling 红利。

**核心矛盾**：作者点出 GAN 失稳的根本原因——adversarial 目标只约束生成分布要匹配数据分布，但不约束 $z \mapsto x$ 的具体搬运图。理论上存在无穷多有效搬运图，初始化 + 训练随机性会让生成器在它们之间不停漂移。

**本文目标**：用单一目标（不依赖蒸馏 / 不依赖教师 / 不依赖特征网络），在标准 DiT 架构上稳定做单步 / 少步对抗训练，同时享受 flow 的 deterministic transport 性质。

**切入角度**：把 Brenier 定理引入：在 Gaussian 源 + 二次代价下，最优传输图是唯一的。如果在 GAN 之上再加一个鼓励 $G(z)$ 离 $z$ 近的损失，就能在所有"有效搬运图"中锁定唯一的 Wasserstein-2 最优传输图，从而消除生成器漂移。

**核心 idea**：用 $\mathcal{L}_{\mathrm{ot}}^G = \mathbb{E}_z[\|G(z)-z\|^2/n]$ 作为额外正则项的 GAN，加上一个带 EMA 归一化的反向传播 trick，让对抗训练在 DiT 上从零训练单步 / 少步生成模型。

## 方法详解

### 整体框架
模型仍是 GAN：生成器 $G$ 把 Gaussian 噪声 $z\in\mathbb{R}^n$ 直接映成图像 latent $G(z) \in \mathbb{R}^n$，判别器 $D$ 区分真假，用 relativistic 损失 + R1/R2 梯度惩罚（用有限差分近似）+ logit centering 惩罚。在生成器端额外加一个最优传输损失 $\mathcal{L}_{\mathrm{ot}}^G$，并对来自判别器的梯度做 EMA 归一化，使 $\lambda_{\mathrm{ot}}$ 可跨模型规模复用。多步 / 任意步通过引入源时间步 $s$、目标时间步 $t$ 和线性插值 $x_s = (1-s)x + s z$ 自然扩展。架构上采用未改动的标准 DiT，单步时去掉时间步投影，判别器与生成器几乎对称，只多一个 [CLS] token。

### 关键设计

1. **最优传输正则 + Brenier 锚定**:

    - 功能：在 GAN 的边际匹配目标之上额外锁定 transport map 的"形状"，让生成器收敛到唯一的 Wasserstein-2 最优传输映射。
    - 核心思路：对生成器加 $\mathcal{L}_{\mathrm{ot}}^G=\mathbb{E}_z\big[\tfrac{1}{n}\|G(z)-z\|^2_2\big]$，在多步设定下推广为 $\mathbb{E}_{x,z,s,t}\big[\tfrac{1}{n\,w(s,t)}\|G(x_s,s,t)-x_s\|^2_2\big]$，权重 $w(s,t)=\max(|s-t|,\delta)$。$\lambda_{\mathrm{ot}}$ 必须做调度：太小逃不出局部极小，太大会被推向恒等映射；本文采用按训练进度衰减的策略。
    - 设计动机：消除"GAN 训练不收敛"的真正病因——目标函数欠定。Brenier 定理保证唯一最优传输图，OT 正则把 GAN 优化变成在所有有效搬运图中"选最近的那个"，使训练曲线和不同随机初始化下的结果都稳定可复现（一维高斯混合实验中能给出完全一致的映射）。

2. **梯度归一化（gradient normalization in backward path）**:

    - 功能：让 $\lambda_{\mathrm{ot}}$ 这个超参数在 B/2 → XL/2 → 112 层模型之间通用，不必每个 size 重新搜。
    - 核心思路：把 $D(G(z))$ 改写成 $D(\phi(G(z)))$，$\phi$ 在前向是恒等、在反向把 $\partial \mathcal{L}_{\mathrm{adv}}^G/\partial G(z)$ 用 EMA 跟踪到的梯度范数归一化，再除以 $\sqrt{n}$。可视为把 Adam 的二阶矩思想搬到 backward 路径上。
    - 设计动机：对抗损失从 $D$ 反传的梯度幅值受架构、初始化、$\lambda_{\mathrm{gp}}$ 强烈影响，原本 Adam 的自适应缩放能"吸收"幅值差异；但加了 $\lambda_{\mathrm{ot}}$ 后两个损失的相对比例变得重要，必须先把对抗梯度归一化到统一尺度。

3. **任意步训练 + 深度递归的单步模型**:

    - 功能：让同一框架既支持纯单步生成，也支持几步生成乃至任意源/目标时间步之间的搬运；并通过 transformer block 重复把单步模型做得很深以匹配多步模型的参数量。
    - 核心思路：训练时 $s\sim\mathcal{U}(0,1),\ t\sim\mathcal{U}(0,s)$，生成器接收 $(x_s, s, t)$，残差形式写作 $G(x_s,s,t) = x_s - (s-t)\,g(x_s,s,t)$（类似 velocity 预测）。判别器仅依赖 $(x_t, t)$，绝不能 condition on 源样本——否则 $x,z$ 独立采样导致目标无法满足、训练发散。深度极深的单步模型用 transformer block repetition：每次复用 hidden state，加一个轻量"重复 ID embedding"区分迭代，整体仍做端到端单步训练，无任何中间监督。
    - 设计动机：与一致性方法相比，本文 $G$ 直接通过 $D$ 学习目标分布，不需要 propagate 一致性，所以可以只在 1-NFE 这一组特定时间步训练；同时极深单步模型规避了"重复进出 data space → projection 误差"的问题，把多步模型的容量优势收编到单步推理路径上。

### 损失函数 / 训练策略
判别器损失 $\mathcal{L}_{\mathrm{AF}}^D = \mathcal{L}_{\mathrm{adv}}^D + \lambda_{\mathrm{gp}}(\mathcal{L}_{r_1}^D + \mathcal{L}_{r_2}^D) + \lambda_{\mathrm{cp}}\mathcal{L}_{\mathrm{cp}}^D$，其中 R1/R2 用 $\epsilon=0.01$ 的有限差分代替二阶导，仅对 25% batch 计算；生成器损失 $\mathcal{L}_{\mathrm{AF}}^G = \mathcal{L}_{\mathrm{adv}}^G + \lambda_{\mathrm{ot}}\mathcal{L}_{\mathrm{ot}}^G$。AdamW，$\beta_1=0,\beta_2=0.9$，lr $1\times10^{-4}$，batch 256，EMA 0.9999，遵循 MeanFlow 的尺寸定义（B/M/L/XL，patch=2）。生成器和判别器同 size，分别用独立 dataloader。Guidance 通过额外 $\mathcal{L}_{\mathrm{cg}}^G=-\mathbb{E}[C(\mathrm{interp}(G(z,c),z',t'),t',c)]$ 实现，必须在时间步上累积梯度才能复现 CFG 行为。

## 实验关键数据

### 主实验
ImageNet-256（32×32×4 VAE latent）类条件生成，FID-50k 对全 train set 评估，主要对比 1NFE / 2NFE / 4NFE。

| 模型 | NFE | 参数 / 深度 | FID-50k | 备注 |
|------|----|------------|---------|------|
| AF B/2 (本文) | 1 | 28 层 | 接近 sCM XL/2 | 容量被保留下来用于一步生成 |
| AF XL/2 (本文) | 1 | 28 层 | **2.38** | 1NFE 新 SOTA |
| AF XL/2 (本文，深度递归) | 1 | 56 层 | **2.08** | 超过 28 层 2NFE 等价对照 |
| AF XL/2 (本文，深度递归) | 1 | 112 层 | **1.94** | 超过 28 层 4NFE 等价对照 |
| sCM / iMM / MeanFlow / AYF 等 | 1 | 同 size | 高于本文 | 一致性家族 |
| R3GAN / GAT 等纯对抗 | 1 | 卷积 / 非标准 transformer | 较弱或不可比 | 需要冻结特征网络或非标准架构 |

### 消融实验

| 配置 | 现象 | 解读 |
|------|------|------|
| 无 $\mathcal{L}_{\mathrm{ot}}$，任意 $\lambda_{\mathrm{gp}}$ | 训练发散 | OT 正则是稳定 DiT 上对抗训练的必要条件 |
| $\lambda_{\mathrm{ot}}$ 过小 | 容易陷入局部极小 | 不足以约束搬运图，行为退化为 GAN |
| $\lambda_{\mathrm{ot}}$ 过大 | 推向 $G(z)\approx z$ | 分布匹配被牺牲 |
| 固定 $\lambda_{\mathrm{ot}}$ vs 衰减 | 衰减更优 | 前期约束 transport，后期让 GAN 微调分布 |
| 不做梯度归一化 | $\lambda_{\mathrm{ot}}$ 需逐 size 重搜 | EMA 归一化让超参在 B → XL → 112 层全程通用 |
| $D(\cdot, z)$ 即 condition on 源 | 训练振荡 / 发散 | 由于 $x,z$ 独立采样，该目标在数学上不可满足 |
| 简单 classifier guidance $C(G(z,c),c)$ | 与无 guidance 几乎相同 | 类别边界清晰时分类器无梯度，guidance 失效；必须用时间步条件分类器 + 沿 flow 累积梯度 |

### 关键发现
- 不靠教师蒸馏、不靠特征网络、不靠改架构，纯标准 DiT 上的对抗训练能稳定从零训练并在 ImageNet 拿到 1NFE SOTA，OT 正则是关键开关。
- 在 guidance-free 设定下本文反过来还能超过 flow matching；作者归因为 $L_2$ 不是流形度量、forward KL 强 mode-coverage 容易产生 OOD 样本，而 GAN 判别器更接近感知度量、JS 距离对异常值更鲁棒。
- 深度递归单步模型超越多步模型表明：模型有效深度是单步生成 fidelity 的瓶颈，而非"步数本身"——这给"单步 vs 多步"之争提供了新的解读。

## 亮点与洞察
- 把 GAN 训练不稳定的病因明确归到"目标欠定"上，再用 Brenier 给出唯一最优传输图作为锚点，是一个干净、可证明且能直接落地的视角，比一致性家族的"propagate consistency"思路更轻盈。
- 时间步条件分类器引导（$C(x_{t'}, t', c)$）模拟了 CFG 沿 flow 累积梯度的效果，让单步对抗模型也享受到 CFG 风格的可控生成；这个 trick 可以直接搬到任何单步 / 少步 GAN 框架。
- 反向路径的 EMA 梯度归一化是一个被低估的"超参 reduce search"工程小技巧——把多目标 loss 的相对比例从 $\lambda$ 调到 $D$ 的输出尺度后，新增 loss 的 weight 选择就解耦了模型大小。
- 深度递归单步训练在概念上对消"flow 必须多步"的固有偏见，给"用容量换 NFE"提供了一个全新的设计点。

## 局限与展望
- 数据集仍局限于 ImageNet-256 类条件，未在大规模 text-to-image / video 上做大规模验证；作者只在 motivation 引用 Lin et al. 2025 暗示可扩展。
- $\lambda_{\mathrm{ot}}$ 的衰减调度仍需手动设计；虽然梯度归一化让超参跨 size 通用，但调度形状还需要进一步研究。
- 当生成器走到 transport map 唯一性失效的区域（如多模态分布的分界），OT 正则可能与 GAN 目标产生张力，理论上未严格分析。
- 极深单步模型 (112 层 + repetition) 的训练成本和稳定性仍依赖较小的 lr 和较低的 OT 衰减下限，工程上对 batch / hardware 仍有要求。

## 相关工作与启发
- **vs 一致性家族（CM / sCM / iMM / MeanFlow / AYF / Shortcut）**：他们用一致性约束沿 flow 传播，需要在所有时间步训练；本文直接在目标时间步训练，省下容量、避免误差累积。
- **vs R3GAN / GAT 等纯对抗复兴工作**：他们靠卷积 + 特殊设计 / 冻结特征网络；本文用标准 DiT，唯一改动是在 $D$ 加 [CLS] token。
- **vs 蒸馏 (Salimans & Ho / Liu et al.)**：本文不需要 teacher，可端到端从零训练。
- **vs 蒸馏 + 对抗微调（如 Lin et al. 2025）**：他们用对抗做最后的精炼；本文证明对抗本身就足够做主训练，省掉两阶段流程。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Brenier 锚定 + 反向梯度归一化的组合既清晰又能解释失稳病因，且第一次在标准 DiT 上跑通从零对抗。
- 实验充分度: ⭐⭐⭐⭐ ImageNet-256 多 size 多 NFE 系统对比 + 大量超参 / 配置消融，但缺少大规模 T2I / video 验证。
- 写作质量: ⭐⭐⭐⭐⭐ 病因分析→数学动机→实现 trick→大量消融，全篇有"教科书式"的论证流。
- 价值: ⭐⭐⭐⭐⭐ 直接挑战"少步生成必须 distill / 必须一致性"的主流路径，为后续大规模生成模型设计开了一条新路径。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[ICLR 2026\] TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](../../ICLR2026/image_generation/twinflow_realizing_one-step_generation_on_large_models_with_self-adversarial_flo.md)
- [\[CVPR 2025\] Instant Adversarial Purification with Adversarial Consistency Distillation](../../CVPR2025/image_generation/instant_adversarial_purification_with_adversarial_consistency_distillation.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)

</div>

<!-- RELATED:END -->
