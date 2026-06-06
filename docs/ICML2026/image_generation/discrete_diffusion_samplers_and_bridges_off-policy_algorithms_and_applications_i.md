---
title: >-
  [论文解读] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces
description: >-
  [ICML2026][图像生成][离散扩散采样器] 本文把连续空间扩散采样里成熟的 off-policy RL 训练技巧（replay buffer、重要性加权、MCMC 探索）首次系统迁移到离散扩散采样器，并进一步推广到 data-to-energy 离散 Schrödinger 桥…
tags:
  - "ICML2026"
  - "图像生成"
  - "离散扩散采样器"
  - "off-policy RL"
  - "轨迹平衡"
  - "Schrödinger 桥"
  - "VQ-VAE 后验采样"
---

# Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces

**会议**: ICML2026  
**arXiv**: [2602.05961](https://arxiv.org/abs/2602.05961)  
**代码**: https://github.com/mmacosha/offpolicy-discrete-diffusion-samplers-and-bridges  
**领域**: image_generation / 扩散采样 / 离散扩散 / 摊销采样 / off-policy RL  
**关键词**: 离散扩散采样器、off-policy RL、轨迹平衡、Schrödinger 桥、VQ-VAE 后验采样

## 一句话总结
本文把连续空间扩散采样里成熟的 off-policy RL 训练技巧（replay buffer、重要性加权、MCMC 探索）首次系统迁移到离散扩散采样器，并进一步推广到 data-to-energy 离散 Schrödinger 桥，在 Ising/Potts、离散化 GMM 等多模分布上显著缓解 mode collapse，最后用它在 VQ-VAE 的离散潜空间里做 data-free 的条件图像生成（后验采样）。

## 研究背景与动机

**领域现状**：从未归一化能量 $p(x) \propto e^{-E(x)}$ 中采样长期由 MCMC、AIS、SMC 主导。近几年，扩散采样器（diffusion samplers）作为一种摊销式采样方法兴起：用一个学到的扩散过程把简单先验 $p_0$ 推向目标 $p_{\text{target}}$，训练目标通常是某种 path-space 散度（trajectory balance、log-variance 等）。在连续空间里，引入 off-policy RL（replay buffer、importance weighting、MCMC 探索）已被证明能显著提升模式覆盖（Sendera et al. 2024; Choi et al. 2026）。

**现有痛点**：离散版本（masked / uniform discrete diffusion sampler，如 MDNS, Zhu et al. 2025; Sanokowski et al. 2025a）发展明显滞后——它们几乎全部使用 on-policy 训练（直接从当前模型采轨迹算损失），在低温 Ising、Potts、高维 GMM 这种强多模目标上极易 mode collapse，最终只覆盖单一模式。

**核心矛盾**：on-policy 训练只看模型自己采到的轨迹，模型一旦偏向某个模式就会强化这个偏向；要打破这个反馈环必须让训练数据"超出当前策略覆盖"——这正是连续扩散采样器里 off-policy 技术的角色，但还没人把它干净地引进离散设定。同时，Zhang et al. 2022a 的 GFlowNet 工作其实早就在做"等价于离散扩散采样"的 off-policy 训练，只是这条线和近期的 masked diffusion sampler 一直没被联系起来。

**本文目标**：(i) 在统一的 second-moment 损失框架下，把 TB / LV 目标和 replay buffer / 重要性加权 / MCMC exploration 干净地搬到离散扩散采样器；(ii) 把这套方法继续推广到 data-to-energy 的离散 Schrödinger 桥（即一端只有能量、另一端只有样本时的桥）；(iii) 拓出一个新应用——离散潜空间生成模型（VQ-VAE）的 data-free 后验采样。

**切入角度**：作者注意到 Berner et al. 2026 的连续空间统一理论可以平移到离散，trajectory balance 又恰好把未知归一化常数吸收进可学标量 $c$，这意味着 GFlowNet 那一脉的 off-policy 思想可以无缝套上离散 diffusion sampler 的 mask/uniform kernel——技术桥梁本质上是把连续 SDE 的 path-space 损失换成离散 Markov 链的轨迹比损失。

**核心 idea**：所有需要的就是一个统一的二阶矩损失 $\mathcal{L}_{\mathcal{P}} = \mathbb{E}_{\mathcal{P}}\big[(\log \tfrac{p_0 \otimes \overrightarrow{p}_\theta^{\otimes N}}{p_{\text{target}} \otimes \overleftarrow{p}^{\otimes N}} - c)^2\big]$，然后通过精心设计的轨迹分布 $\mathcal{P}$（buffer / 重要性加权 / MCMC 精修）实现 off-policy，把它从连续扩散采样器搬到离散扩散采样器和 Schrödinger 桥。

## 方法详解

### 整体框架

输入：能量函数 $E: \mathcal{S} \to \mathbb{R}$，其中 $\mathcal{S} = \{1, \dots, C\}^d$ 是离散序列空间；目标是从 $p_{\text{target}}(x) \propto e^{-E(x)}$ 采样。模型是一个前向（去噪）核 $\overrightarrow{p}_\theta(X_{n+1} \mid X_n)$，配合事先选定的后向（加噪）核 $\overleftarrow{p}$（masking 或 uniform 离散扩散），从 $p_0$ 出发经 $N$ 步 Markov 链到达 $X_N \in \mathcal{S}$，希望 $X_N$ 的边缘分布等于 $p_{\text{target}}$。

训练 = 用某分布 $\mathcal{P}$ 采轨迹 $X_{0:N}$，最小化 $\mathcal{L}_{\mathcal{P}}$（公式 3）。当 $c$ 取可学标量时是 **trajectory balance (TB)**；当 $c$ 取 batch 内经验均值时是 **log-variance (LV)**。$\mathcal{P}$ 的选择就是 on-policy / off-policy 的分水岭。

桥的情形（§3）：把固定的 $p_0$ 换成任意分布，再让 $\overleftarrow{p}_\varphi$ 也参数化，用 IPF 迭代（公式 6a/6b）交替拟合 $\overrightarrow{\mathcal{P}}, \overleftarrow{\mathcal{P}}$，data-to-energy 情形下 (6b) 用上面的 LV 变体来训。参考过程取 uniform 离散扩散（masking 桥问题不适用）。

应用（§4）：在 VQ-VAE 的离散潜空间 $z \in \{1, \dots, 8\}^{16}$ 上，给定预训练自回归先验 $p_{\text{latent}}(z)$、确定性解码器 $f$ 和分类似然 $p(y \mid f(z))$，把后验采样 $p(z \mid y) \propto p_{\text{latent}}(z) \cdot p(y \mid f(z))$ 当成新的离散能量采样问题，用同一套 sampler 训。

### 关键设计

1. **统一二阶矩损失 + on/off-policy 解耦**：

    - 功能：用同一个目标 $\mathcal{L}_{\mathcal{P}}$ 同时支持 TB 和 LV，并允许任意轨迹分布 $\mathcal{P}$。
    - 核心思路：损失 $\mathcal{L}_{\mathcal{P}} = \mathbb{E}_{\mathcal{P}}[(\log \tfrac{p_0 \otimes \overrightarrow{p}_\theta^{\otimes N}}{p_{\text{target}} \otimes \overleftarrow{p}^{\otimes N}} - c)^2]$ 在 $\mathcal{P}$ 满支撑时的唯一最小值仍是 $p_0 \otimes \overrightarrow{p}_\theta^{\otimes N} = p_{\text{target}} \otimes \overleftarrow{p}^{\otimes N}$，所以可以自由替换 $\mathcal{P}$ 做探索而不改变最优解；当 $\mathcal{P} = p_0 \otimes \overrightarrow{p}_\theta^{\otimes N}$ 时梯度与 reverse KL 同向，对应传统 on-policy 训练。
    - 设计动机：on-policy 训练高方差且会被早期模式 lock-in，而把损失写成"轨迹比的平方"这种形式后，$\mathcal{P}$ 就像 RL 里的 behavior policy，可以随便换成 buffer 或 MCMC 精修的轨迹，从而引入探索。

2. **重要性加权 replay buffer**：

    - 功能：把以前 rollout 出来的终末态 $X_N$ 存进 buffer，按其"目标重要性"重新采样训练轨迹。
    - 核心思路：每个 buffer 元素入库时算并存储重要性权重 $w = e^{-E(X_N)} \prod_n \overleftarrow{p}(X_n \mid X_{n+1}) / [p_0(X_0) \prod_n \overrightarrow{p}_\theta(X_{n+1} \mid X_n)]$（公式 4），训练时按 $w$ 加权抽样；之后再用 $\overleftarrow{p}^{\otimes N}$ 反向 unroll 出完整轨迹喂损失。
    - 设计动机：纯 uniform buffer 已能稳住快速变化的策略；重要性加权进一步把训练算力倾斜到"目标概率高、但模型自己采到的概率低"的样本——这正是 on-policy 训练永远到不了的高价值区域，对多模分布的模式覆盖至关重要。

3. **MCMC exploration 精修 buffer**：

    - 功能：在 buffer 样本喂训练之前，用以 $p_{\text{target}}$ 为平稳分布的 MCMC kernel（如 Metropolis-Hastings、Swendsen-Wang、Hamming-ball proposal）迭代若干步精修。
    - 核心思路：MCMC 只需调用能量函数 $E$，不需调用模型，几乎不增 GPU 开销；对 Ising/Potts 这种有结构的能量可以选 Swendsen-Wang 这种"会跨模式跳"的 proposal，对一般离散密度用 1-Hamming-ball MH。算法 1 把"模型 rollout → 入 buffer → MCMC 精修 → 加权采样训练"串成一个完整的训练循环。
    - 设计动机：buffer 只能记忆模型自己见过的模式，而 MCMC 可以在能量梯度的引导下走到模型没见过的模式上去；二者结合既保证训练数据来自"真目标附近"，又能在低温/强多模设定下打破 mode collapse（Table 1 中只有带 MCMC 的方法在 Ising $\beta=1.2$ 没 collapse 到单模）。

### 损失函数 / 训练策略

主目标用 trajectory balance：$\mathcal{L}_{\text{TB}} = \mathbb{E}_{\mathcal{P}}[(\log \tfrac{p_0 \overrightarrow{p}_\theta^{\otimes N}}{p_{\text{target}} \overleftarrow{p}^{\otimes N}} - \log Z_\phi)^2]$，$\log Z_\phi$ 是可学常数，吸收未知归一化项。log-variance 版本把 $c$ 换成 batch 内经验均值，免去额外参数。桥设定下用 IPF 交替更新前后向核，data-to-energy 步用 LV 变体。采样推理时允许变时间离散化：训练时多步 mask，推理时可改成"每步只 unmask 一格"，平衡训练显存和推理质量（§A.1）。温度退火（target temperature annealing）默认开，对所有 sampler 公平。

## 实验关键数据

### 主实验

Ising / Potts 模型（$16 \times 16$ toroidal lattice，5 次平均；MDNS = 当前 SOTA on-policy 离散 diffusion sampler）：

| 设置 | 方法 | ELBO ↑ | EUBO ↓ | Sinkhorn ↓ | Magnetisation err ↓ |
|------|------|--------|--------|------------|----------------------|
| Ising $\beta=0.6$ | MDNS (WDCE, on-policy) | 310.18 | 341.82 | 48.71 | 0.41 |
| Ising $\beta=0.6$ | LV (on-policy) | 309.77 | 422.53 | 116.96 | 0.97 (collapse) |
| Ising $\beta=0.6$ | TB + Buffer | 310.42 | 310.56 | 3.59 | 0.04 |
| Ising $\beta=0.6$ | **TB + Buffer + MCMC** | **310.43** | **310.55** | **3.47** | **0.02** |
| Ising $\beta=1.2$ | MDNS / on-policy | 614.42 | >1100 | ~127 | 1.00 (severe collapse) |
| Ising $\beta=1.2$ | **TB + Buffer + MCMC** | **615.03** | **615.14** | **0.02** | **0.03** |
| Potts $q=3, \beta=1.2$ | MDNS | 620.23 | 680.52 | 99.95 | 0.58 |
| Potts $q=3, \beta=1.2$ | **TB + Buffer + MCMC** | **620.73** | **621.30** | **12.37** | **0.03** |

离散化合成密度（每维 Gray 码 8 bit，5 次平均）：

| 目标 | 方法 | ELBO ↑ | MMD ↓ | Sinkhorn ↓ |
|------|------|--------|-------|------------|
| 40GMM ($d=32$) | MDNS | -16.66 | 0.17 | 349.31 |
| 40GMM ($d=32$) | TB (on-policy) | -2.47 | 0.40 | 2142.65 (collapse) |
| 40GMM ($d=32$) | TB + Buffer | -5.97 | 0.07 | 114.11 |
| 40GMM ($d=32$) | **TB + Buffer + MCMC** | **-7.13** | **0.04** | **4.25** |
| ManyWell ($d=80$) | MDNS | 41.52 | 0.04 | 1.82 |
| ManyWell ($d=80$) | **TB + Buffer + MCMC** | 48.74 | 0.04 | 1.36 |

VQ-VAE 后验采样（MNIST，16 维潜空间，8-word codebook，似然 = 数字属于 odd / even / 等于某数）：on-policy LV 和 off-policy LV 都能正确生成目标类别的图像（Fig. 5），off-policy 多样性更好；这是首次把离散扩散采样器接入预训练生成模型潜空间做 data-free 条件生成。

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| on-policy (TB / LV) | 在 $\beta=1.2$ Ising、40GMM $d=32$ 上 Sinkhorn 比 off-policy 高 1-3 个数量级，magnetisation ≈ 0.95 | 严重 mode collapse |
| TB + Buffer | 大部分温度下显著好转，但 Ising $\beta=1.2$ 仍偶发 collapse（Sink 50.94 ± 62.35） | 单靠 buffer 在最低温下不稳定 |
| TB + Buffer + MCMC | 所有温度下均稳定，Sinkhorn 与真 MH MCMC 同量级甚至更好 | MCMC 探索是低温区的关键 |
| LV vs TB | TB 在大部分温度上略优于 LV，但 LV 更省（不学 $\log Z$） | 与连续空间结论一致 |
| MCMC 选 Swendsen-Wang vs Hamming-ball | 结构化能量上 Swendsen-Wang 显著快 | §D.1.3 |
| Schrödinger 桥（on vs off-policy） | 简单桥（3GMM↔4GMM）两者都行；难桥（10GMM↔40GMM）on-policy collapse、off-policy 成功 | Fig. 4 |

### 关键发现

- 三大 off-policy 组件中，**MCMC exploration 是低温/强多模设定下唯一能稳住的关键**——单靠 buffer 在 Ising $\beta=1.2$ 仍会偶发崩塌；MCMC 一旦接入，结果稳定到与长跑 MH MCMC 同量级。
- on-policy 训练在所有困难任务上几乎一致地 collapse 到单一模式（magnetisation 接近 1，Sinkhorn 比 off-policy 高 1–3 个数量级），从而验证了把连续空间 off-policy 经验迁移到离散设定的必要性。
- TB 在能写出未知 $\log Z$ 的设定下普遍略优于 LV；但 LV 在 Schrödinger 桥的 data-to-energy 步骤里更自然（不需额外可学常数）。
- VQ-VAE 后验采样实验首次证明，离散 diffusion sampler 可以作为预训练生成模型的"通用后验插件"，无需对原模型 fine-tune 或反传梯度。

## 亮点与洞察

- 论文把"GFlowNet（Zhang et al. 2022a）== 离散扩散采样器（MDNS, Zhu et al. 2025）"这条历史悬而未决的等价关系点透了——这意味着 GFlowNet 三年多积累的 off-policy 训练经验整套可用，paper 实际上是在补一段被两个社群各自忽略的桥梁。
- 把"未知归一化常数 $Z$"吸收进 TB 的可学标量 $c$ 是个非常巧妙的工程小动作：让 second-moment 损失既能在已知 $Z$（如桥的 reference）下退化为 KL，又能在 $Z$ 未知（采样问题）下直接学，免去了所有 importance reweight 的 bias 校正讨论。
- "MCMC 几乎免费"的观察值得迁移：因为 MCMC 只调能量、不调模型，对任何 amortised sampler 都可以无痛接上做训练时探索。这对"模型贵、能量便宜"的设定（蛋白质能量、组合优化）尤其有意义。
- VQ-VAE 后验采样把"在预训练离散潜空间里做条件生成"做成了 sampling problem：未来可以接 LLM 的 token 空间做 controllable generation——本质上是 RL fine-tuning 的另一种实现方式，但完全 data-free。

## 局限与展望

- 主战场仍是合成（Ising / Potts / 离散 GMM）和小规模 VQ-VAE/MNIST，没有 scale 到大型离散潜空间（如真实图像 VQ-GAN、大型 LM 的 token 空间），高维下 MCMC 探索的实际效率未充分检验。
- MCMC 在训练循环里跑得不够长以收敛（作者自己点明），这意味着 MCMC 主要起"良性扰动"作用而非"真后验校正"，在拓扑更复杂的能量上有效性有待验证。
- 一致用 trajectory balance / log-variance 这两类二阶矩损失，没有与基于 SMC、重要性自适应等替代摊销方案做直接比较。
- 桥实验只到二维 Gray-coded 16-bit，data-to-energy IPF 在更高维的稳定性尚需更系统的验证。
- 后验采样仅用了分类似然（odd/even/=k），更复杂条件（OCR、ROI mask）下能否仍 data-free 地学好后验是开放问题。

## 相关工作与启发

- **vs MDNS (Zhu et al. 2025)**：MDNS 是当前离散 diffusion sampler 的 on-policy 代表（用 weighted denoising cross-entropy）。本文用统一 second-moment 框架把它纳入对比，并展示 off-policy 在低温多模设定下系统性更强，是其直接增强版。
- **vs GFlowNet (Zhang et al. 2022a; Bengio et al. 2021/2023)**：GFlowNet 在三年前就用 TB + off-policy 在离散 EBM 上做摊销采样，但社群一直没把它与 masked discrete diffusion sampler 联系起来。本文显式做了这个 unification，让两条路径的工具互通。
- **vs 连续扩散采样 off-policy 工作 (Sendera et al. 2024; Choi et al. 2026)**：连续设定用 SDE，本文证明同一套 buffer + 重要性加权 + MCMC 三件套换成离散 Markov 链一样有效——这种"换 kernel 不换思想"的迁移很优雅。
- **vs 离散 Schrödinger 桥 (Kim et al. 2025a; Ksenofontov & Korotin 2025)**：之前的离散桥工作要求两端都有样本（data-to-data）。本文在 Tamogashev & Malkin (2026) 的连续 data-to-energy IPF 基础上拓到离散，是该方向第一篇。
- **vs Outsourced diffusion samplers (Venkatraman et al. 2025)**：Venkatraman 等人在连续潜空间（VAE/GAN/CNF）做了 data-free 后验采样，本文把同一思路推到 VQ-VAE 的离散潜空间，扩展了"摊销后验"的可用模型族。

## 评分
- 新颖性: ⭐⭐⭐⭐ 单点技术（TB/LV/buffer/MCMC）都来自已有连续空间或 GFlowNet 工作，本文亮点在"系统迁移 + 统一框架 + 桥/VQ-VAE 两个新应用"，更像高质量整合而非全新方法。
- 实验充分度: ⭐⭐⭐⭐ Ising、Potts、40GMM、ManyWell、Schrödinger 桥、VQ-VAE 后验六个场景全覆盖，消融详尽（buffer / MCMC / 损失 / 调度 / 离散步数都有），但缺真实大规模生成模型验证。
- 写作质量: ⭐⭐⭐⭐⭐ 把 GFlowNet 与 discrete diffusion sampler 的历史关系厘清，统一框架公式干净，章节衔接清楚，是这块新手友好的好综合参考。
- 价值: ⭐⭐⭐⭐ 给离散摊销采样研究者一份"立刻可上手"的工具箱；VQ-VAE 后验采样这条线对 controllable generation 社群有启发，但短期内主要受益者还是物理 / 组合优化 / probabilistic ML 社群。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](../../NeurIPS2025/image_generation/fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)
- [\[ICML 2026\] Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges](transferable_multi-bit_watermarking_across_frozen_diffusion_models_via_latent_co.md)
- [\[NeurIPS 2025\] Guided Diffusion Sampling on Function Spaces with Applications to PDEs](../../NeurIPS2025/image_generation/guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)
- [\[ICML 2026\] Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)

</div>

<!-- RELATED:END -->
