---
title: >-
  [论文解读] Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion
description: >-
  [ICML 2026][图像生成][可信多模态融合] 本文提出 GMF：用 Diffusion Schrödinger Bridge + Rectified Flow 在潜空间估计每个模态的"传输修正成本"（初始速度平方 $\|v_\theta(z,0)\|^2$）…
tags:
  - "ICML 2026"
  - "图像生成"
  - "可信多模态融合"
  - "Schrödinger 桥"
  - "Rectified Flow"
  - "传输能量"
  - "证据学习"
---

# Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion

**会议**: ICML 2026  
**arXiv**: [2605.31193](https://arxiv.org/abs/2605.31193)  
**代码**: 暂无  
**领域**: 多模态VLM / 可信融合 / 生成式几何  
**关键词**: 可信多模态融合, Schrödinger 桥, Rectified Flow, 传输能量, 证据学习

## 一句话总结
本文提出 GMF：用 Diffusion Schrödinger Bridge + Rectified Flow 在潜空间估计每个模态的"传输修正成本"（初始速度平方 $\|v_\theta(z,0)\|^2$），作为一个**与分类器置信度解耦**的几何可靠性信号来动态加权多模态融合，从而打破"模型自己评判自己"的循环依赖，在传感器噪声和语义冲突下显著优于基于置信度的可信融合基线。

## 研究背景与动机

**领域现状**：可信多模态融合（Trustworthy Multimodal Fusion）的主流路线是 dynamic fusion——先独立处理每个模态，再按"模态质量"加权聚合预测。代表方法 TMC、QMF、PDF、DBF 等都用分类器输出（熵、证据、Dirichlet 浓度）作为质量评分。

**现有痛点**：深度网络存在严重的过度自信问题（Guo et al. 2017）。在严重噪声、OOD 或语义冲突场景下，分类器可能"自信地错"——输出概率很尖锐但答案是错的。基于置信度的方法把"我有多自信"当作"输入有多干净"，无法识别这种 confident-but-wrong 的失败模式。

**核心矛盾**：可靠性评估和被评估的预测来自同一个模型，形成**循环依赖**（circular dependency）——用预测去判断预测的可靠性。当分类器被欺骗时，所有依赖其输出的可靠性度量也一起失效。

**本文目标**：构造一个**独立于分类器决策边界**的可靠性信号，使得即使分类器被严重噪声或冲突输入欺骗，融合机制仍能正确识别坏模态并降低其权重。

**切入角度**：作者把"模态质量"重新定义为**潜空间几何偏离度**——干净样本聚在数据流形上，OOD/噪声样本远离流形。如何度量"远离"？用最优传输：把样本传输到一个参考分布所需的"修正功"。

**核心 idea**：用 Diffusion Schrödinger Bridge 学一条从潜特征到参考分布的传输路径，用 Rectified Flow 把路径拉直成单步线性预测；**初始速度平方** $\|v_\theta(z,0)\|^2$ 就是一个高效的"几何不可靠分"——干净样本传输代价低，噪声/冲突样本传输代价高，且这个度量与分类器 logits 完全解耦。

## 方法详解

### 整体框架

输入 $M$ 个模态 $\{x^{(m)}\}_{m=1}^M$，每个模态先经编码器得到潜特征 $z^{(m)} = E^{(m)}(x^{(m)}) \in \mathbb{R}^d$。GMF 在潜空间上做两件事：

1. **模态内几何评估**（intra-modal）：用 modality-specific 的 rectified flow 速度场 $v_\theta^{(m)}$ 估计 $z^{(m)}$ 到一个**类无关参考先验** $\mathcal{P}_{\text{prior}}$ 的传输能量 $\mathcal{E}_{\text{intra}}^{(m)} = \|v_\theta^{(m)}(z^{(m)},0)\|_2^2$，作为模态的"内在质量分"。

2. **模态间几何评估**（inter-modal）：对每个有向对 $(a \to b)$ 训练一个 cross-modal 速度场 $v_\Phi^{(a \to b)}$，把 $z^{(a)}$ 映到 $b$ 的流形上，残差 $\|\Phi_{a \to b}(z^{(a)}) - z^{(b)}\|_2^2$ 度量模态间语义一致性。

随后用一个"竞争-交互"门控机制把这两类几何成本合并成融合权重 $w^{(m)}$，与证据 $\mathbf{e}^{(m)} = \text{Softplus}(z^{(m)} W_{\text{cls}}^{(m)})$ 一起组装成 Dirichlet 参数 $\boldsymbol{\alpha} = \sum_m w^{(m)} \mathbf{e}^{(m)} + \mathbf{1}$，喂给 evidential 分类头。训练时几何分支和决策分支梯度路径分离，避免互相污染。

### 关键设计

1. **模态内传输能量作为几何可靠性分**：

    - 功能：给出一个**与分类器输出无关**的标量来度量 $z^{(m)}$ 离干净流形多远。
    - 核心思路：用 Schrödinger Bridge 形式化 $\min_v \int_0^1 \mathbb{E}\|v_t\|^2 dt$，但直接求解需要迭代积分太慢；改用 Rectified Flow 把路径线性化——在 $z_t = (1-t)z_0 + t z_1$ 上回归常速度 $z_1 - z_0$，目标 $\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, z_0, z_1}\|v_\theta(z_t, t) - (z_1 - z_0)\|^2$。推理时只需评一次 $v_\theta(z, 0)$，把 $\|v_\theta(z, 0)\|_2^2$ 当作单步学到的"修正分"。
    - 设计动机：(1) 单步推理低延迟，能在线部署；(2) $v_\theta$ 在 $z$（path 的源点）上求值，推理与训练分布一致；(3) 干净样本在流形上、需要的修正小，噪声/缺失样本偏离流形、修正大——形成与置信度**正交**的失败检测器。

2. **跨模态传输残差作为语义冲突门控**：

    - 功能：在不需要解码器的前提下，直接在潜空间度量两个模态在语义上是否对得上。
    - 核心思路：为每个有向对 $(a \to b)$ 单独学一个 $v_\Phi^{(a \to b)}$，用 $\mathcal{L}_{\text{inter}}^{(a \to b)} = \mathbb{E}\|v_\Phi^{(a \to b)}(z_t, t) - (z^{(b)} - z^{(a)})\|^2$ 训练；推理时一步把 $z^{(a)}$ 投影成 $\hat{z}^{(a \to b)} = z^{(a)} + v_\Phi^{(a \to b)}(z^{(a)}, 0)$，残差 $\mathcal{E}_{\text{inter}}^{(a \to b)} = \|\hat{z}^{(a \to b)} - z^{(b)}\|_2^2$ 越大说明 $a$ 和 $b$ 的语义越对不上。论文还在理论上证明（Thm 4.5）：如果两个模态分别落在不同类的流形上，残差有下界 $(\delta - 2\epsilon)^2$（Geometric Barrier Principle）。
    - 设计动机：把"两个模态是否说同一件事"从分类器层（容易被两个都过自信的输出骗）下沉到潜空间几何层，得到一个分类器无法伪造的外部判据。

3. **竞争-交互融合权重**：

    - 功能：把内在质量分 $\mathcal{E}_{\text{intra}}$ 和跨模态一致性 $\mathcal{E}_{\text{inter}}$ 组合成最终融合权重 $w^{(m)}$，并被理论 Thm 4.4 解释为一个熵正则最小化的 Gibbs 解。
    - 核心思路：先按 Boltzmann 分配竞争基分 $\beta_{\text{comp}}^{(m)} = \exp(-\mathcal{E}_{\text{intra}}^{(m)}/\tau) / \sum_k \exp(-\mathcal{E}_{\text{intra}}^{(k)}/\tau)$；再用 interaction gate $\gamma_{\text{int}}^{(m)} = \lambda \sum_{k \neq m} r^{(k)} \exp(-\mathcal{E}_{\text{inter}}^{(k \to m)}/\kappa)$ 收集"可靠邻居"的几何投票，其中 $r^{(k)} = \sigma(\theta_r - \mathcal{E}_{\text{intra}}^{(k)})$ 是邻居本身的可靠性软门；加 $\epsilon_\gamma$ 数值稳定后归一化得 $w^{(m)} = \beta_{\text{comp}}^{(m)} \tilde{\gamma}_{\text{int}}^{(m)} / \sum_j \beta_{\text{comp}}^{(j)} \tilde{\gamma}_{\text{int}}^{(j)}$。
    - 设计动机：竞争项保证"自身干净"的模态拿高权重，交互项把"被可靠邻居否认"的模态权重指数压低——这正对应推论 4.6 的"冲突模态指数抑制"。两层合并使得**自信但与他人冲突**的坏模态被双重削权，从而打破循环依赖。

### 损失函数 / 训练策略

总目标 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{geo}} \mathcal{L}_{\text{geo}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$。其中：

- $\mathcal{L}_{\text{geo}} = \sum_m \mathcal{L}_{\text{intra}}^{(m)} + \sum_{a \neq b} \mathcal{L}_{\text{inter}}^{(a \to b)}$ 训练所有 RF 速度场；
- $\mathcal{L}_{\text{task}}$ 是 evidential 交叉熵 + KL 正则（往均匀 Dirichlet 拉，惩罚证据不足的过自信）；
- $\mathcal{L}_{\text{reg}} = (1 - \rho)^\zeta \cdot \text{KL}(\text{Dir}(\boldsymbol{\alpha}) \| \text{Dir}(\mathbf{1}))$ 用全局一致性系数 $\rho = \frac{1}{M(M-1)} \sum_{a \neq b} \exp(-\mathcal{E}_{\text{inter}}^{(a \to b)}/\kappa)$ 在跨模态分歧大时强制预测分布趋向均匀。

关键训练 trick：在算融合权重时对 $\mathcal{E}_{\text{intra}}, \mathcal{E}_{\text{inter}}$ 做 stop-gradient（`sg`），让 $\mathcal{L}_{\text{task}}$ 只更新编码器与分类头、$\mathcal{L}_{\text{geo}}$ 只更新速度场，避免任务梯度污染几何度量。

## 实验关键数据

### 主实验

四个 benchmark：NYU Depth V2（RGB-D）、UPMC FOOD-101（图文）、MVSA-Single（图文情感）、PneumoniaMNIST（X-ray + 报告）。和 10 个基线对比（含 TMC、QMF、PDF、DBF、UAW-EEF 等）。

**传感器噪声鲁棒性**（NYU/Food-101，加 Gaussian 噪声 $\sigma \in \{1.0, 2.0\}$，或 50% 模态缺失）：

| 数据集 | 场景 | Concat | QMF | PDF | DBF | UAW-EEF | **GMF** |
|--------|------|--------|-----|-----|-----|---------|---------|
| NYU | Clean | 68.5 | 71.2 | 72.5 | 72.3 | 71.8 | 71.9 |
| NYU | $\sigma=2.0$ | 28.4 | 45.8 | 47.5 | 49.1 | 50.2 | **55.2** |
| NYU | Incomplete | 35.8 | 56.4 | 58.2 | 60.3 | 61.5 | **64.8** |
| Food-101 | $\sigma=2.0$ | 30.2 | 48.6 | 51.2 | 52.4 | 53.1 | **58.7** |
| Food-101 | Incomplete | 41.2 | 78.5 | 80.6 | 81.3 | 82.4 | **85.4** |

干净数据上 GMF 与最优持平，**噪声越严重领先越大**——验证了"几何信号在过自信场景下仍有效"的假设。

**语义冲突安全性**（MVSA-Single，shuffle 图文对制造冲突）：

| 方法 | Rejection Rate ↑ | Avg Entropy ↑ | CDR (AUROC) ↑ |
|------|------------------|---------------|---------------|
| QMF | 18.5% | 0.52 | 56.8 |
| PDF | 21.3% | 0.58 | 60.1 |
| DBF | 35.2% | 0.94 | 71.2 |
| **GMF** | **76.8%** | **1.85** | **89.4** |

冲突拒绝率比次优方法 DBF 高 **41.6 pp**，AUROC 高 18.2 pp——说明纯靠预测空间证据的方法（QMF/PDF/DBF）在"两个都过自信但矛盾"的输入上几乎检测不到冲突，而 GMF 的跨模态传输残差能直接拍出冲突。

**医学风险分层**（PneumoniaMNIST）：GMF 准确率 91.2%，与正确性的 Pearson 相关 $r=0.78$（次优 DBF 仅 0.61），ECE 降到 0.068（次优 0.095）。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full GMF | $\sigma=2.0$ 准确率 55.2% | 完整模型 |
| 用 predictive entropy 替代 $\mathcal{E}_{\text{intra}}$ | 36.8% | 掉 18.4 pp；MI(可靠性, 置信度) 从 0.10 飙到 0.67——验证统计度量与置信度高度耦合，几何度量解耦 |
| 用 cosine similarity 替代 flow-based $\mathcal{E}_{\text{inter}}$ | 冲突检测大幅下降 | 说明语义冲突在潜空间是**非线性几何畸变**，线性度量抓不到 |
| 1-step RF vs 多步 ODE 积分 | 准确率几乎相同 | 验证 RF 拉直假设成立，单步速度估计够用 |
| 先验 $\mathcal{P}_{\text{prior}}$：$\mathcal{N}(0,I)$ vs $\mathcal{N}(0,\Sigma)$ vs Laplace | 性能稳定 | 几何信号主要来自学到的传输结构，对先验选择不敏感 |

### 关键发现

- **几何信号的独立性是关键**：消融中把 $\mathcal{E}_{\text{intra}}$ 换成 entropy，准确率立刻掉到 statistical baseline 水平且 MI 飙升，说明"打破循环依赖"不是论文嘴上说说，是真的需要一个外部信号。
- **冲突检测的指数门控被实证支持**：图 2(b) 显示 $w^{(m)}$ 关于 $\mathcal{E}_{\text{inter}}$ 大致服从 $e^{-\mathcal{E}_{\text{inter}}/\kappa}$ 的指数衰减，干净对集中在 $\mathcal{E}_{\text{inter}} < 5$，冲突对集中在 $> 9$，几何屏障定理在实验上得到验证。
- **单步 RF 在精度损失可忽略的前提下把延迟拉到与 PDF/DBF 同档**，让 GMF 在 safety-critical 实时场景可用。

## 亮点与洞察

- **"循环依赖"这个 framing 抓得很准**：之前的可信融合工作虽然各自加 trick，但都没把"用预测评估预测"这一根本结构性弱点点破。一旦把它点出来，"必须用一个 prediction-free 的外部信号"就成了几乎自明的设计原则，本文的方法只是这个原则下的一种具体实例化——这种"先重塑问题再给方法"的论文写法很值得学。
- **生成模型当作几何探针，而非生成器**：Schrödinger Bridge / Rectified Flow 本来是 generative model，本文却只取它训练好后**一次前向的速度向量模长**作为标量探针，完全不做采样。这种"用生成模型副产物当判别信号"的思路可以迁移到 OOD 检测、对抗样本检测、医学异常检测等任何需要"流形偏离度"的任务。
- **几何屏障定理把可解释性钉死在假设上**：Thm 4.5 用 $(\delta - 2\epsilon)^2$ 这种几何量给出了冲突模态被指数压制的下界，配合 Thm 4.4 把融合权重证明成熵正则最小化的唯一解——整套理论虽然假设很强（latent regularity + cross-modal Lipschitz），但提供了一个干净的"几何 → 权重 → 安全"的可解释链路。
- **训练时分离梯度路径**这个工程细节非常关键，否则任务损失会把速度场拽偏，让传输能量重新和分类器耦合，循环依赖就回来了。

## 局限与展望

- 作者承认的局限：理论 Thm 4.5 依赖"潜空间类流形可分（concentration + metric separability）"以及"跨模态映射局部 $\xi$-语义一致"这两个强假设，在表示学习失败或模态本身没对齐的场景下不成立。
- **跨模态速度场数量随 $M$ 平方增长**（每个有向对一个 $v_\Phi^{(a \to b)}$），对 $M \geq 4$ 的多传感器系统参数和训练成本都会膨胀，论文只做了 2 模态实验，可扩展性未验证。
- **类无关参考先验** $\mathcal{P}_{\text{prior}}$ 抹掉了类信息，对于"同类内分布转移"（如同一类的不同 domain）能否区分需要进一步验证；论文说先验选择不敏感，但没测先验完全 misspecified 的情况。
- **MVSA-Single 上 76.8% 的 Rejection Rate 听起来高，但意味着仍有 23% 的冲突未被检出**，在真正 safety-critical 的医疗/自驾上还远未达到可部署的可靠性。
- 改进方向：(1) 用 amortized cross-modal field（一个网络条件在 $(a, b)$ 上）替代 pair-specific，把 $O(M^2)$ 参数压到 $O(M)$；(2) 把传输能量作为外部信号接入 LLM-based fusion，给 VLM 加一个潜空间几何 sanity check；(3) 把"几何屏障"扩展到时间维度，做视频/时序多模态的可信融合。

## 相关工作与启发

- **vs QMF / PDF / TMC**（基于证据/熵的统计可信融合）：他们都从分类器输出推质量分，本文从潜空间几何推质量分；本文的核心优势就是打破他们共有的循环依赖，劣势是需要额外训练 $M + M(M-1)$ 个速度场。
- **vs DBF / UAW-EEF**（显式建模分歧的近期 SOTA）：他们已经意识到要建模冲突，但分歧仍然是在 belief mass / classifier logit 上算的，仍困在预测空间；本文的跨模态残差 $\mathcal{E}_{\text{inter}}$ 在潜空间做几何对齐，提供了一个分类器无法伪造的外部冲突信号——MVSA 上 Rejection Rate 76.8% vs 35.2% 的差距就是这种"换战场"的红利。
- **vs Evidential Deep Learning 系列**：EDL 用 Dirichlet 强化"不知道"的表达，但本质上还是分类器内省；GMF 保留了 EDL 的 evidence head 但把权重生成器换成几何模块，可看作"EDL + 外部几何 prior"的混合体。
- **vs 用 diffusion/flow 做 OOD 检测**（Pinaya 2022 等）：之前用 reconstruction likelihood 或采样轨迹长度，要么需要解码器要么需要多步采样；本文用 Rectified Flow 的单步初始速度作为标量代理，是这条线在效率与多模态融合场景的具体推进。

## 评分
- 新颖性: ⭐⭐⭐⭐ "把可信融合的失败诊断为循环依赖，并用 RF/SB 的几何探针打破它"是一个清爽且未被前人系统化的视角，理论+方法+实验闭环完整。
- 实验充分度: ⭐⭐⭐⭐ 4 个 benchmark、10 个基线、3 类压力测试（噪声 / 冲突 / 缺失 / 医学）覆盖到位，消融把"几何 vs 统计"的解耦量化（MI 0.10 vs 0.67）做得很有说服力；但模态数都 ≤2，没验证 $M \geq 3$ 的可扩展性。
- 写作质量: ⭐⭐⭐⭐ "循环依赖"叙事把动机讲得非常清楚，理论部分（Thm 4.4 / 4.5）虽假设强但与实验呼应紧密；公式排版较密集，部分符号（$\rho$ 同时被局部/全局使用）需要细读才能厘清。
- 价值: ⭐⭐⭐⭐ 可信多模态融合在自动驾驶/医疗诊断/机器人是刚需，本文提出的"用生成模型副产物当外部可靠性信号"的范式有较强通用性，工程上几何分支可以模块化嵌到现有 dynamic fusion 系统。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges](../../ICLR2026/image_generation/contact_wasserstein_geodesics_for_non-conservative_schrödinger_bridges.md)
- [\[ICLR 2026\] Branched Schrödinger Bridge Matching](../../ICLR2026/image_generation/branched_schrödinger_bridge_matching.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](../../CVPR2026/image_generation/careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [\[ICML 2026\] Geometry-Aware Tabular Diffusion](geometry-aware_tabular_diffusion.md)
- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)

</div>

<!-- RELATED:END -->
