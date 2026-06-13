---
title: >-
  [论文解读] From Navigation to Refinement: Revealing the Two-Stage Nature of Flow-based Diffusion Models through Oracle Velocity
description: >-
  [CVPR 2026][图像生成][Flow Matching] 这篇论文给 rectified flow 的边际速度场推出了一个在高斯先验 + 有限数据集下的**闭式 oracle 解**，并用它揭示 flow-based 扩散模型的训练目标天然分成「导航阶段」(早期、被多个数据模式混合引导，负责全局布局/泛化) 和「精修阶段」(后期、被最近邻单个样本主导，负责细节/记忆) 两段，再用这个两阶段视角统一解释了 timestep shifting、CFG interval、latent space 设计这些经验技巧为什么有效。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "Flow Matching"
  - "oracle velocity"
  - "记忆与泛化"
  - "两阶段训练"
  - "timestep schedule"
---

# From Navigation to Refinement: Revealing the Two-Stage Nature of Flow-based Diffusion Models through Oracle Velocity

**会议**: CVPR 2026  
**arXiv**: [2512.02826](https://arxiv.org/abs/2512.02826)  
**代码**: 项目页 https://maps-research.github.io/from-navigation-to-refinement/  
**领域**: 扩散模型 / 图像生成 / 生成模型理论  
**关键词**: Flow Matching, oracle velocity, 记忆与泛化, 两阶段训练, timestep schedule

## 一句话总结
这篇论文给 rectified flow 的边际速度场推出了一个在高斯先验 + 有限数据集下的**闭式 oracle 解**，并用它揭示 flow-based 扩散模型的训练目标天然分成「导航阶段」(早期、被多个数据模式混合引导，负责全局布局/泛化) 和「精修阶段」(后期、被最近邻单个样本主导，负责细节/记忆) 两段，再用这个两阶段视角统一解释了 timestep shifting、CFG interval、latent space 设计这些经验技巧为什么有效。

## 研究背景与动机
**领域现状**：Flow Matching（FM）/ rectified flow 已成为训练 SOTA 扩散模型的事实标准——线性插值 $x_t = \alpha_t x_1 + \sigma_t x_0$，模型回归速度场。但 FM 目标本身被普遍认为「不可解」(intractable)，因为真值速度场 $u_t(x_t)$ 拿不到，实践中都退而用梯度等价的条件版 CFM，以单个样本 $x_1$ 构造条件路径。

**现有痛点**：模型的「记忆 vs 泛化」行为一直没被讲清。过去工作有两个局限：(1) 多在「从纯噪声采样」的设定下研究；(2) 多在小规模/低分辨率数据集（FFHQ、CIFAR-10）上验证。而当数据规模到 ImageNet 级别，从头记忆训练样本几乎不可能发生，这些结论就不再适用。Song et al. 观察到 ImageNet 规模模型「从不同时间点续采」会出现记忆/泛化的分叉：从靠近先验的早期续采得到新样本，从靠近数据的后期续采则倾向复现训练图——但**为什么**会这样，缺一个原理性解释。

**核心矛盾**：大家把 FM 目标当成一个「黑箱不可解的回归任务」，于是无法从训练目标本身去解释模型行为。如果能写出 oracle 目标的精确形式，记忆/泛化、学习难度、各种采样技巧就都能从「模型在每个时刻到底被要求拟合什么」推出来。

**切入角度**：作者指出，在三条温和假设下——高斯先验、有限数据集近似 $p_{\text{data}}$、rectified flow 线性插值——边际速度场其实**有闭式表达**。于是可以在样本空间任意点精确算出 oracle 速度，把 FM 训练还原成一个有真值标签的监督学习问题来剖析。

**核心 idea**：用闭式 oracle 速度场作显微镜，发现 oracle 目标沿时间 $t$ 天然分两段（导航 / 精修），并以此为统一框架解释模型的记忆-泛化行为与一系列采样实践。

## 方法详解
本文不是提出新模型，而是一套**分析框架**：先推闭式 oracle 速度，再用它刻画两阶段训练目标，最后把模型行为和经验技巧都挂回这两个阶段。下面按「核心数学结果 → 两阶段刻画 → 实验探针 → 技巧再解释」展开。

### 关键设计

**1. 高斯先验下的闭式 oracle 速度场：把「不可解」的 FM 目标算出来**

痛点是 FM 目标 $\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,p_t(x_t)} \| v_t(x_t;\theta) - u_t(x_t) \|^2$ 里的真值 $u_t(x_t)$ 通常拿不到，只能用 CFM 近似。作者把数据分布写成有限数据集上的经验混合分布，对路径边际（一个高斯混合）用贝叶斯法则、再取给定 $x_t$ 的条件期望 $u_t^*(x_t,t) := \mathbb{E}[u_t(x_t\mid x_1)\mid x_t]$，得到闭式解（Theorem 2.1）：

$$u_t^*(x_t,t) = A_t \sum_{i=1}^{N} \gamma_i(x_t,t)\, x_1^{(i)} + B_t\, x_t$$

其中 $A_t = \dot\alpha_t - \alpha_t \dot\sigma_t/\sigma_t$，$B_t = \dot\sigma_t/\sigma_t$，而归一化后验权重

$$\gamma_i(x_t,t) = \frac{\exp\!\big(-\|x_t - \alpha_t x_1^{(i)}\|^2 / 2\sigma_t^2\big)}{\sum_{j=1}^{N} \exp\!\big(-\|x_t - \alpha_t x_1^{(j)}\|^2 / 2\sigma_t^2\big)}$$

是一个 softmax 形式的「$x_t$ 距离每个数据点 $x_1^{(i)}$ 有多近」的权重。这一步之所以关键，是它把 oracle 目标变成「$N$ 个数据点的加权平均 + 一个随当前位置走的项」——后续所有结论都从 $\gamma_i$ 怎么随 $t$ 变化推出来。类条件生成时，只在该类子集 $\{x_1^{(i)}\}_{i\in I_y}$ 上算 $u_t^*(x_t,t\mid y)$ 即可

**2. 两阶段训练目标：导航 vs 精修，转折点由 $D$、$N$、$\sigma_t$ 共同决定**

有了 $\gamma_i$ 的闭式，就能看清 oracle 目标沿 $t$ 的结构变化。在靠近先验的早期 $t\in[0,0.1]$（**导航阶段**），$\gamma_i$ 比较平、多个数据点都有权重，oracle 速度是「多个相关数据模式的混合」，把模型推向一个全局布局；过了 $t\approx 0.1$（**精修阶段**），top-1 后验权重迅速饱和到 1，oracle 速度坍缩到**单个最近邻样本**主导，几乎退化成 CFM 目标 $x_1 - x_0$。论文用两条曲线验证：(a) $u_t^*$ 与 CFM 目标 $(x_1-x_0)$ 的 MSE 只在 $t\in[0,0.1]$ 明显发散、之后高度重合；(b) top-1 后验权重在 $t>0.1$ 后迅速逼近 1。

为什么饱和这么快？因为在 $D$ 维空间里 $\gamma_i$ 指数项的平方距离随 $D$ 线性增长、再除以 $2\sigma_t^2$，$D$ 越大、$\sigma_t$ 越小，样本间哪怕微小的距离差也会被指数放大成压倒性的权重差——于是后验在最近邻上「尖峰化」。由此得到转折点的依赖关系：**数据维度 $D$ 越大转折越早、样本量 $N$ 越大转折越晚**。ImageNet $256^2$ 在 latent $D\in\{4096,8192\}$、$N\approx 1400$ 下，转折恰好落在 $t\approx 0.1$。这条结论很有用——同一个 rectified flow 目标，在不同数据集上有效训练目标其实不一样

**3. 混合采样探针：用「oracle 走前段、模型走后段」把记忆/泛化挂到具体阶段**

光有训练目标的两阶段还不够，作者要证明**模型行为**也分两段，并把它和记忆/泛化对上。设计了一个混合采样：从高斯先验出发，先用 oracle 速度 $u_t^*$ 走到切换点 $t_{\text{switch}}$，之后切到训练好的模型速度。用 oracle 走前段的好处是保证中间状态严格落在插值分布上，从而把「模型预测不完美」这个混杂因素隔离掉。结果很干净：全程用 oracle 会确定性地检索回某个训练样本；$t_{\text{switch}}\in(0.2,1.0]$（在精修阶段才切）模型几乎复现训练轨迹、产出近似训练图（**记忆**）；$t_{\text{switch}}\in[0,0.2]$（在导航阶段就切）强先验扰动让模型无法反推原始轨迹，从而偏离训练实例、表现出**泛化**。配合可视化中间预测（单步 Euler 到 $t=1$）可见：$t=0$ 时预测坍缩成类均值（鲨鱼偏蓝、熊猫黑白），早期主要导航全局布局（$t\approx0.2$ 稳定），后期只精修局部细节。值得注意，模型行为的转折 $t\approx0.2$ 比训练目标的转折 $t\approx0.1$ 略滞后，作者推测是模型切到一致目标后还需额外时间余量来纠正累积误差

**4. 两阶段视角统一再解释经验技巧：timestep shift / CFG interval / latent space**

最后把框架落到实践。作者还观察到两阶段「学习难度不同」：训练 MSE 的发散集中在导航阶段，精修阶段两种目标曲线重合；且**导航性能几乎与模型容量无关，精修才显著受益于更大模型/更长训练**——Tab. 1 的换模型实验直接证实（在固定 NFE 下把 Stage 1 换成小模型几乎不掉点，把 Stage 2 换成小模型 gFID 大幅恶化）。基于此重新解释三件事：(i) **timestep shifting** $t_m = s t_n / (1+(s-1)t_n)$ 本质是调节两阶段的采样步分配，$s<1$ 给导航多分步、适度偏向导航能得到更好样本；(ii) **CFG interval** 把引导限制在子区间，最优区间集中在导航→精修的**早中精修段**，且扩展区间时要排除最初 $[0,0.1]$（极噪声下放大引导会干扰全局布局形成）；(iii) **latent space**：语义对齐的 VA-VAE（DINO VF loss）oracle loss 在精修阶段平滑收敛成抛物线状，纯重建的 SD-VAE 则呈波浪状，说明模式组织更清晰的 latent 同时利于导航与精修、收敛更快

### 损失函数 / 训练策略
本文不改训练目标，仍用标准 rectified flow / CFM；分析对象是基于 VA-VAE 与 SD-VAE latent 训练的 LightningDiT-XL/1（及 B 变体），ImageNet $256^2$，多数模型 100 epoch（部分 800 epoch 作 8× 算力对照）。所有「oracle」量都用闭式 $u_t^*$ 离线精确计算，不需额外训练。

## 实验关键数据

### 主实验：两阶段下的模型容量分配（Tab. 1）
固定总 NFE（每个子区间 25 步均匀采样、无 CFG），交换 Stage 1 / Stage 2 所用模型，看 gFID@50K。结论：换小模型做 Stage 1（导航）几乎不掉点，换小模型做 Stage 2（精修）大幅恶化——容量主要被精修吃掉。

| 采样切分 | Stage 1 模型 | Stage 2 模型 | gFID@50K ↓ |
|----------|--------------|--------------|------------|
| [0,0.1]+[0.1,1.0] | XL | XL | 2.94 |
| [0,0.1]+[0.1,1.0] | Base | XL | 3.71 |
| [0,0.1]+[0.1,1.0] | XL | Base | 11.26 |
| [0,0.1]+[0.1,1.0] | Base | Base | 12.45 |
| [0,0.2]+[0.2,1.0] | XL | XL | 2.60 |
| [0,0.2]+[0.2,1.0] | Base | XL | 4.47 |
| [0,0.2]+[0.2,1.0] | XL | Base | 9.24 |
| [0,0.2]+[0.2,1.0] | Base | Base | 12.01 |

可见 Base→XL（仅换精修）从 12.45→3.71，而 XL→Base（仅退化精修）反而到 11.26，两条对比印证「精修是容量瓶颈、导航不是」。

### 分析实验：CFG interval 扫描（Tab. 3，LightningDiT-B/1，$\omega=2.5$）
| CFG 区间 | gFID@50K ↓ | CFG 区间 | gFID@50K ↓ |
|----------|-----------|----------|-----------|
| None | 12.99 | [0.0,1.0] | 10.79 |
| [0.0,0.1] | 6.33 | [0.1,0.2] | 5.21 |
| [0.1,0.3] | 3.54 | [0.1,0.5] | 2.82 |
| [0.1,0.6] | 2.80 | [0.1,0.7] | 2.86 |
| [0.4,0.5] | 10.39 | [0.9,1.0] | 12.20 |

最优区间集中在早中精修段（[0.1,0.6] 取到最低 2.80），过晚或仅在最初导航段都差。

### 关键发现
- **转折点 $t\approx 0.1$ 由 $D$、$N$、$\sigma_t$ 共同决定**：$D$ 越大转折越早、$N$ 越大转折越晚；ImageNet latent $D\in\{4096,8192\}$、$N\approx1400$ 解释了实测的 $t\approx0.1$。
- **timestep shift（Tab. 2）**：把约 22%（uniform）的早期步占比适度提到 28%–34%（$s=0.7$/$0.5$）时 gFID 最好（12.46 / 12.23），偏太多（$s=0.1$，72%）反而劣化到 19.91——导航要多给一点但不能过头。
- **模型行为转折（$t\approx0.2$）滞后于训练目标转折（$t\approx0.1$）**：模型需余量纠正切换后的累积误差。
- **latent 结构影响精修收敛**：VA-VAE 平滑收敛、SD-VAE 波浪状，语义对齐 latent 收敛更快。

## 亮点与洞察
- **把「不可解」的 FM 目标写成闭式 oracle**：这是全文支点——一旦能精确算 $u_t^*$，记忆/泛化、学习难度、采样技巧都能从「每个 $t$ 模型被要求拟合什么」推出来，而不再靠经验观察堆砌。
- **后验权重 $\gamma_i$ 的 softmax 视角非常直观**：把「靠近先验=多模式混合、靠近数据=最近邻独占」解释成一个温度随 $\sigma_t$ 收缩的 softmax 尖峰化过程，顺带给出 $D$/$N$ 的定量依赖，可迁移到任何线性插值扩散框架。
- **oracle-走前段的混合采样是个干净探针**：用解析速度隔离「模型不完美」这个混杂因素，让记忆/泛化能被切换点 $t_{\text{switch}}$ 单变量地复现，方法论上很值得借鉴。
- **一个框架统一解释三类技巧**：timestep shift 是分配采样步、CFG interval 是「在精修早中段加引导最划算」、latent 设计是「模式组织清晰利于两阶段」——把零散调参经验收敛成同一原理。

## 局限与展望
- **假设较强**：闭式 oracle 依赖高斯先验、有限数据集、线性插值三条假设；论文也承认演示用了合成单位高斯样本，真实数据分布会略偏离（虽行为相似）。非线性 schedule / 非高斯先验下结论是否成立需进一步验证。
- **主要在 ImageNet 类条件、latent 扩散上验证**：文中虽附录补了 Flux.1，但对文生图、视频等大规模条件生成的两阶段转折点与技巧最优区间是否同样成立，仍待系统检验。
- **oracle loss 的绝对量不可跨 latent 直接比**（作者已注明只看曲线形状），因此 VA-VAE vs SD-VAE 的优劣更多是定性结论。
- **可改进方向**：既然导航对容量不敏感、精修才吃容量，自然引出「非对称容量分配 / 阶段化蒸馏」——前段用小模型或更少步、后段集中算力，论文已用 Tab. 1 给出可行性信号但未做成完整训练方案。

## 相关工作与启发
- **vs 闭式扩散目标的已有工作**：此前也有人在不同扩散公式下推过闭式目标，本文聚焦 rectified flow 的边际速度场，并把闭式解专门用来揭示「两阶段」这一新结构与记忆/泛化的归因，而非仅作为采样工具。
- **vs 记忆/泛化分析（小数据集路线）**：过去多在 FFHQ/CIFAR-10、从头采样设定下研究，本文指出大规模数据下「从头记忆」几乎不可能，真正的记忆来自精修阶段从训练轨迹续采，把结论搬到了 ImageNet 规模。
- **vs Song et al. 的续采观察**：他们经验地发现早/晚续采的记忆-泛化分叉，本文用两阶段 oracle 目标给出了原理性解释（导航=泛化、精修=记忆）。
- **vs timestep shifting / CFG interval 原始提法**：这些技巧原本各有动机（高分辨率噪声不均衡、避免全程过度引导），本文把它们统一重述为「调节两阶段算力分配 / 在精修早中段加引导」。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把被认为不可解的 FM 目标写成闭式 oracle，并由此揭示两阶段结构，视角新且解释力强
- 实验充分度: ⭐⭐⭐⭐ ImageNet 上多组定量验证（容量分配/timestep/CFG/latent）齐全，但跨任务（文生图/视频）覆盖有限
- 写作质量: ⭐⭐⭐⭐⭐ 推导清晰、图表与结论对应、takeaway 收束到位
- 价值: ⭐⭐⭐⭐⭐ 给扩散模型训练动态提供了可操作的原理框架，对采样调参与架构设计都有直接指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](../../ICML2026/image_generation/stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[CVPR 2026\] VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation](vde_training-free_accelerating_rectified_flow_model_via_velocity_decomposition_a.md)
- [\[CVPR 2026\] FAGER: Factually Grounded Evaluation and Refinement of Text-to-Image Models](fager_factually_grounded_evaluation_and_refinement_of_text-to-image_models.md)

</div>

<!-- RELATED:END -->
