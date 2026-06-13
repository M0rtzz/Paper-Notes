---
title: >-
  [论文解读] Adversarial Flow Models
description: >-
  [ICML 2026][图像生成][Adversarial Training] 作者在 GAN 训练目标上加一个最优传输正则 $\|G(z)-z\|^2$，把 GAN 的"任意搬运图"约束成 Wasserstein-2 最优搬运图，让纯 transformer 上的对抗训练第一次能稳定收敛并端到端做单步生成，ImageNet-256 上 1NFE FID 刷到 2.38（XL/2）和 1.94（112 层）。
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
本文骨架仍是一个 GAN：生成器 $G$ 把高斯噪声 $z\in\mathbb{R}^n$ 直接映成图像 latent $G(z)\in\mathbb{R}^n$，判别器 $D$ 用 relativistic 损失加 R1/R2 梯度惩罚（有限差分近似）和 logit centering 来区分真假。关键改动是在生成器端补一个最优传输正则，把"GAN 能匹配分布但不约束搬运图"这个失稳病因堵住，再配一套反向路径的梯度归一化让超参跨模型规模通用。整个框架既能做纯单步生成，也能通过引入源时间步 $s$、目标时间步 $t$ 与线性插值扩展到任意步搬运，而架构本身用的是未改动的标准 DiT。

### 关键设计

**1. 最优传输正则 + Brenier 锚定：把欠定的对抗目标钉到唯一搬运图上**

GAN 失稳的根因在于对抗目标只要求生成分布匹配数据分布，却不约束 $z\mapsto x$ 这条搬运图的具体形状——理论上存在无穷多有效搬运图，初始化和训练噪声会让生成器在它们之间漂移。本文加一项最优传输损失 $\mathcal{L}_{\mathrm{ot}}^G=\mathbb{E}_z\big[\tfrac{1}{n}\|G(z)-z\|^2_2\big]$，鼓励 $G(z)$ 离源 $z$ 尽量近；在多步设定下推广为 $\mathbb{E}_{x,z,s,t}\big[\tfrac{1}{n\,w(s,t)}\|G(x_s,s,t)-x_s\|^2_2\big]$，权重 $w(s,t)=\max(|s-t|,\delta)$。之所以有效，是因为 Brenier 定理保证在高斯源 + 二次代价下最优传输图唯一，于是 OT 正则把 GAN 优化变成"在所有有效搬运图里选最近那个"，消除了漂移——一维高斯混合实验里不同随机初始化都能收敛到完全一致的映射。$\lambda_{\mathrm{ot}}$ 必须按训练进度衰减：太小逃不出局部极小、退化回普通 GAN，太大则把 $G$ 推向恒等映射、牺牲分布匹配。

**2. 反向路径梯度归一化：让 $\lambda_{\mathrm{ot}}$ 一个值通吃所有模型规模**

加了 $\mathcal{L}_{\mathrm{ot}}$ 后，对抗损失与 OT 损失的相对比例变得敏感，而对抗损失从 $D$ 反传的梯度幅值又受架构、初始化、$\lambda_{\mathrm{gp}}$ 强烈影响——原本 Adam 的自适应缩放能吸收这种幅值差异，现在却会让 $\lambda_{\mathrm{ot}}$ 必须逐 size 重搜。解决办法是把 $D(G(z))$ 改写成 $D(\phi(G(z)))$，其中 $\phi$ 前向是恒等、反向把 $\partial\mathcal{L}_{\mathrm{adv}}^G/\partial G(z)$ 用 EMA 跟踪到的梯度范数归一化再除以 $\sqrt{n}$。本质上是把 Adam 的二阶矩思想搬到 backward 路径上，先把对抗梯度归一化到统一尺度，使 $\lambda_{\mathrm{ot}}$ 在 B/2 → XL/2 → 112 层之间一个值通用，省掉逐规模调参。

**3. 任意步训练 + 深度递归的单步模型：用容量换 NFE**

同一框架要同时支持单步、几步乃至任意源/目标时间步之间的搬运。训练时采样 $s\sim\mathcal{U}(0,1),\ t\sim\mathcal{U}(0,s)$，生成器接收 $(x_s,s,t)$，写成残差形式 $G(x_s,s,t)=x_s-(s-t)\,g(x_s,s,t)$（类似 velocity 预测）；判别器只依赖 $(x_t,t)$，绝不能 condition on 源样本，否则 $x,z$ 独立采样会让目标在数学上不可满足、训练发散。与一致性方法相比，这里 $G$ 直接通过 $D$ 学目标分布、无需沿 flow 传播一致性，因此可以只在 1-NFE 这组特定时间步上训练，省下容量也避免误差累积。为了在单步推理路径上吃到多步模型的容量优势，本文把单步模型做得极深——用 transformer block repetition 复用 hidden state，每次迭代加一个轻量"重复 ID embedding"区分，整体仍端到端单步训练、无任何中间监督，从而规避了"反复进出 data space → projection 误差"的问题。

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
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[CVPR 2025\] Instant Adversarial Purification with Adversarial Consistency Distillation](../../CVPR2025/image_generation/instant_adversarial_purification_with_adversarial_consistency_distillation.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)

</div>

<!-- RELATED:END -->
