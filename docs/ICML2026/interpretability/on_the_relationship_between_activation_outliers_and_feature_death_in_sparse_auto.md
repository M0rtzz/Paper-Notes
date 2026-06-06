---
title: >-
  [论文解读] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders
description: >-
  [ICML 2026][可解释性][稀疏自编码器] 本文指出 SAE 中"死特征"问题的真正根源不是训练动力学而是激活分布的几何性质——用 $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 量化"维度级离群"严重程度…
tags:
  - "ICML 2026"
  - "可解释性"
  - "稀疏自编码器"
  - "feature death"
  - "激活离群"
  - "mean-centering"
  - "TopK SAE"
---

# On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders

**会议**: ICML 2026  
**arXiv**: [2605.31518](https://arxiv.org/abs/2605.31518)  
**代码**: 无  
**领域**: 可解释性
**关键词**: 稀疏自编码器, feature death, 激活离群, mean-centering, TopK SAE

## 一句话总结
本文指出 SAE 中"死特征"问题的真正根源不是训练动力学而是激活分布的几何性质——用 $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 量化"维度级离群"严重程度，从初始化就解析地预测死率（454 个模型-层组合上 Spearman $\rho=0.82\sim0.89$），并证明仅用 mean-centering 就能把 AlphaFold3/ESM3 等高 $\gamma$ 模型的死率从 70%+ 降到接近零。

## 研究背景与动机

**领域现状**：稀疏自编码器（SAE）是机制可解释性的主流工具，把神经网络激活映射到高维稀疏字典空间（$n>d$），每个字典方向代表一个可解释概念。架构上有 ReLU-SAE、TopK-SAE、JumpReLU-SAE 等变体，本文以 TopK-SAE 为主。

**现有痛点**：同一套 SAE 配置（架构、字典大小、稀疏度、AuxK 都一致）在 GPT-2 上死特征率 <5%，到 AlphaFold3 上却高达 72%。即使在 ESM3 单个模型内部，不同层的死率也在 20%–80% 之间剧烈波动。死特征意味着字典容量被严重浪费，幸存特征不得不"挤"进更多概念，反而让 SAE 本想消除的 superposition 回流。

**核心矛盾**：以往的复活技术（AuxK、Ghost Gradient、Resampling）都把死特征视为"训练动力学问题"——既然卡住了就用各种 trick 推一把。但在最严重的模型上这些 trick 全部失效，说明问题根本不在训练。

**本文目标**：(1) 找到一个能跨模态、跨模型预测死率的可解释诊断量；(2) 解释为什么 AuxK 等复活方法在高死率模型上失效；(3) 给出原则性的预处理方案并说明何时必须使用。

**切入角度**：作者发现高死率层都有一个共同的激活模式——少数维度的均值显著大于 per-token 标准差（"高均值低方差"的维度级离群），这与量化领域研究的 token 级离群（个别 token 的尖峰）完全不同。这种几何性质在初始化时就已经决定了大部分特征的命运。

**核心 idea**：死特征是激活几何问题而非训练问题——用 $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 一个标量就能预测死率，而 mean-centering（用激活均值初始化 bias）就能从根上消除离群导致的死。

## 方法详解

### 整体框架

论文围绕三件事展开：把单一标量 $\gamma$ 解析地与死率挂钩、把训练时的复活机制拆成快慢两条路径、给出一个"零额外计算开销"的预处理（mean-centering）。

输入：任意预训练神经网络（GPT-2、Pythia、DINOv3、ESM3、AlphaFold3、Evo2 等）某一层的激活分布；
输出：(1) 训练前就能算出的 $\gamma$ 诊断值；(2) 由 $\gamma$ 直接预测的初始死率公式；(3) 一个修改 SAE bias 初始化的一行代码。

形式上，TopK-SAE 的标准结构为：

$\mathbf{z}_{\text{pre}}=\mathbf{W}_{\text{enc}}(\mathbf{x}-\mathbf{b})+\mathbf{b}_{\text{enc}}$，
$\mathbf{z}=\text{TopK}(\text{ReLU}(\mathbf{z}_{\text{pre}}))$，
$\hat{\mathbf{x}}=\mathbf{W}_{\text{dec}}^{\top}\mathbf{z}+\mathbf{b}$。

死特征有两条路径：**dead-by-ReLU**（pre-activation 在任何输入上都为负）和 **dead-by-TopK**（pre-activation 为正但永远进不了 top-$k$）。

### 关键设计

1. **$\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 诊断量与解析死率公式**：

    - 功能：用一个标量量化"激活均值"相对"每 token 方差"的占比，进而在训练开始前就给出死特征率的闭式预测。
    - 核心思路：把单 token 激活分解为 $\mathbf{x}=\bm{\mu}+(\mathbf{x}-\bm{\mu})$，则 pre-activation 拆成常数 shift 项 $\mathbf{w}_i\cdot\bm{\mu}$ 和随输入变化的 signal 项 $\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})$。当 $\gamma$ 大时 shift 主导 signal：与 $\bm{\mu}$ 反向对齐的特征 pre-activation 永远为负（dead-by-ReLU），与 $\bm{\mu}$ 强正向对齐的特征在每个输入上都激活，只有与 $\bm{\mu}$ 近似正交的特征才真正响应输入。把 shift 和 signal 都视为随机单位向量在固定方向上的投影，用高维概率近似（详细推导在 Appendix B）得到 $P(\text{dead-by-ReLU})=\Phi(-C/\gamma)$，其中 $C=\Phi^{-1}(1-1/N)\approx 4.26$（$N=10^5$ 评估样本）；TopK 情形下生存门槛抬高到 shift 分布的 $(1-k/n)$ 分位数 $t_k=\Phi^{-1}(1-k/n)$，得到 $P(\text{dead-by-TopK})\approx \Phi(t_k-C/\gamma)$。实际计算 $\gamma$ 前先对激活做 per-token LayerNorm，剥离不同 token 之间的尺度差异。
    - 设计动机：以往工作要么用 token 级离群（kurtosis），要么干脆没有诊断量。$\gamma$ 是真正"维度级"的几何量，跨模态在 454 个模型-层组合上 Spearman $\rho$ 高达 0.89（dead-by-TopK）/ 0.82（dead-by-ReLU），且不需要任何拟合参数。实践者甚至可以在投入算力训练 SAE 之前就预测是否会出现严重死特征。

2. **两条死特征复活路径与"bias 学 $\bm{\mu}$ 是瓶颈"**：

    - 功能：解释训练中死特征如何（以及为什么不能）自行复活，揭示 AuxK 等已有方法的作用域上限。
    - 核心思路：在合成数据上把 SAE 的 bias 冻结/不冻结、加/不加 AuxK 进行消融。**Dead-by-TopK** 的复活靠的是 alive feature 在训练中收敛后下调自己的激活幅度，让原本卡在第 $k+1$ 名的特征挤进来——这条路径在 $\sim$200K 步内就能完成，bias 冻结也不受影响；**Dead-by-ReLU** 的复活则只能靠 bias 慢慢吸收 $\bm{\mu}$，因为只有 bias 能把恒为负的 pre-activation 抬到零以上。问题是 bias 学 $\bm{\mu}$ 的速度严重依赖 $\gamma$：$\gamma\le 5$ 时 200K 步就能学到 99%，$\gamma\approx 20$ 时 2M 步只到 90%，$\gamma\ge 30$ 时 2M 步只到 50–70%。直观原因是特征权重作用于输入、效果随输入幅度放大，而 bias 是直接相加、不被输入放大，所以 $\|\bm{\mu}\|$ 越大 bias 越追不上；alive feature 一旦学到 $\bm{\mu}$ 又会进一步压低 bias 的梯度。AuxK 的隐藏作用其实是抑制"附带死亡"——TopK 复活过程中部分 alive feature 在缩小激活时被压到零以下变成新的 dead-by-ReLU，AuxK 给 dead-by-TopK 提供梯度让它们稳住而不滑入 dead-by-ReLU；但 AuxK 完全不加速 bias 学习，所以面对从初始化就 dead-by-ReLU 的特征束手无策，这就解释了为什么 AuxK 在 $\gamma$ 中等时有效、$\gamma$ 高时失效。
    - 设计动机：以往工作隐式假设"死特征 = 训练动力学坏了"，所以一直在做"如何注入更多梯度"的工作。这里第一次把复活机制按死亡路径解耦，证明高 $\gamma$ 下根本不需要更好的复活技术——只要让 bias 一开始就处在 $\bm{\mu}$ 的位置即可。

3. **Mean-centering：用激活均值初始化 bias**：

    - 功能：一行代码层面的预处理，把 SAE 的 bias 初始化为激活的几何中位数（默认）或算术均值，从而把 pre-activation 中的 shift 项直接消掉。
    - 核心思路：令 $\mathbf{b}=\bm{\mu}$ 后，pre-activation 退化为 $z_i=\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})+b_{\text{enc}}$，shift 项 $\mathbf{w}_i\cdot\bm{\mu}$ 消失，所有特征的 pre-activation 都围绕零分布、只随输入变化，从初始化就不存在因离群导致的死。默认采用 geometric median 而非 arithmetic mean，因为某些模型的激活分布偏斜较重；这一选择在 Appendix D.5 有逐模型对比。等价于在 runtime 做 mean subtraction，但折进 bias 初始化后没有任何额外推理开销。注意它只消除"离群型死"，对极少数因为方差集中在小维度子空间的层（蛋白/基因模型的少数层）仍有残留死，需要再用 PCA whitening（Appendix E）处理。
    - 设计动机：mean-centering 在 Bricken 2023b、Gao 2024 里其实出现过，但用得不一致、也没有清晰的判据。$\gamma$ 正好给出"何时必须 center"的原则——高 $\gamma$ 必做，低 $\gamma$ 可选——把这一步从经验 trick 升级为有解析依据的预处理。

### 损失函数 / 训练策略

仍是标准 TopK-SAE 训练目标（重构 MSE + TopK 稀疏化），$k$、字典大小、学习率等超参在跨模型对比中保持一致；mean-centering 不修改 loss，只动 bias 的初始化。所有合成实验取 10 个种子取平均，真实数据上对 454 个模型-层组合统一用 mid-network 层做训练。

## 实验关键数据

### 主实验：$\gamma$ 在合成与真实数据上预测死率

| 数据 | 指标 | dead-by-ReLU | dead-by-TopK | 备注 |
|------|------|--------------|--------------|------|
| 合成激活（控制 $\gamma$） | Spearman $\rho$ | 1.0 | 1.0 | $\Phi(-C/\gamma)$ 曲线几乎完美对齐 |
| 454 个真实模型-层（语言/视觉/蛋白/基因） | Spearman $\rho$ | 0.82 | 0.89 | 无任何拟合参数 |
| AlphaFold3 mid layer | 死特征率 | — | 98% → <5% | mean-centering 后 |
| ESM3 mid layer | 死特征率 | — | 83% → ≈0 | mean-centering 后 |

### 消融：mean-centering vs baseline vs AuxK（ESM3 L24，$\gamma\approx 8$）

| 配置 | 训练末死特征率 | 可解释生物概念数 |
|------|----------------|------------------|
| baseline | ≈75% | 73 (dict=8192) |
| baseline + AuxK | ≈25%（plateau） | — |
| LayerNorm + $\sqrt{d}$ rescale | ≈20% | 少于 baseline |
| mean-centering（dict=2048） | ≈0 | **100** |
| mean-centering（dict=8192） | ≈0 | 更高 |

### 合成 ground-truth feature 恢复（$\gamma=40$）

| 配置 | MMCS（最大余弦相似度均值） |
|------|----------------------------|
| baseline | 0.38 |
| mean-centering | 0.97 |

### 关键发现

- **$\gamma$ 是真正的"训练前可算"诊断量**：在 454 个真实模型-层上无拟合就能达到 $\rho\approx 0.89$，意味着可以先算 $\gamma$ 再决定是否投入算力训练 SAE。
- **bias 学习是高 $\gamma$ 下复活的瓶颈**：$\gamma\ge 30$ 时 bias 在 2M 步内只能学到 $\bm{\mu}$ 的 50–70%，dead-by-ReLU 因此长期卡在 75–90%。
- **AuxK 真正在做的事是抑制 collateral death**：它给 dead-by-TopK 提供梯度让其稳住，而不是真的复活初始化时就 dead-by-ReLU 的特征。
- **mean-centering 用 4× 更小的字典超过 baseline**：ESM3 上 dict=2048 的 mean-centered SAE（100 个概念）已超过 dict=8192 的 baseline（73 个概念），训练算力大幅下降。
- **mean-centering 还稳定了对学习率的敏感性**：baseline 在 LR sweep 中死率方差极大，mean-centered 几乎保持一致的低死率。
- **理论稍微高估 dead-by-ReLU**：当激活分布重尾（用 per-dim kurtosis 可以直接诊断），实际中最大 signal 可以超过 Gaussian 假设下的 $C\approx 4.26$，从而救活一部分本应被判死的特征。

## 亮点与洞察

- **把"训练问题"重新框定为"几何问题"**：长期以来 SAE 社区都在调 AuxK / Ghost Gradient / Resampling，这篇论文证明在最难的模型上这些 trick 全部失效，本质是因为大部分特征在初始化时就死了——这是研究框架的转向，不只是一个新方法。
- **一个 0 参数的解析公式打败一堆经验诊断**：之前社区用 kurtosis 等指标都是 token 级的，而 $\gamma$ 抓住了 dimension-level outlier 这个真正的来源，公式纯粹从高维几何推出来不含拟合参数，却跨四个模态都能预测死率。这种"先验+几何"的范式可以迁移到其他需要诊断初始化问题的场景。
- **bias 与 weight 的学习速度不对称**：权重作用于输入、效果随输入幅度放大，而 bias 直接相加。这个观察可以解释很多归一化/中心化技术为何有效，是个可复用的直觉。
- **AuxK 的真实作用被重新解释**：之前都以为 AuxK 在"复活"特征，本文细分后发现它其实在"防止 alive 特征滑入 dead"，这种"现象→机制重判"的细致拆解值得借鉴。
- **MMCS 从 0.38 → 0.97**：合成数据上字典几乎完美对齐 ground-truth，说明 mean-centering 不只是降死率，而是真的让 SAE 学到了对的方向，从可解释性角度直接受益。

## 局限与展望

- **极少数层 mean-centering 不够**：蛋白和基因模型的部分层即使 center 后仍有残留死特征，因为方差集中在少数方向，需要再做 PCA whitening（Appendix E），这意味着 mean-centering 只是"必要而非充分"的预处理。
- **理论假设了 Gaussian signal**：实际激活重尾时 $\Phi(-C/\gamma)$ 会高估死率，作者用 per-dim kurtosis 兜底，但还没有一个统一的公式同时容纳重尾激活。
- **仅在 mid-network 单层做了系统对比**：跨层、跨任务的迁移性虽然有 Appendix 数据但不够全面，尤其是不同层 $\gamma$ 差异很大时如何统一选预处理仍开放。
- **未与最新的"low-rank attention"假说（Wang 2025）做正面对比**：Wang 等把死率归因于注意力激活的低秩结构，本文归因于维度级离群，两种几何因子的相互作用没有系统消融。
- **应用层面的延伸方向**：可以把 $\gamma$ 反过来用作"激活归一化"或"模型架构正则"的目标——直接把训练时的 $\gamma$ 压低，可能从源头减少下游 SAE 训练成本，甚至有助于量化。

## 相关工作与启发

- **vs AuxK / Ghost Grad / Resampling (Gao 2024; Bricken 2023b)**：它们都试图给死特征"打梯度"复活；本文证明在高 $\gamma$ 下根本不是梯度问题而是 bias 距离问题，AuxK 真正在做的事是抑制 collateral death 而非真复活。本文优势是给出原则性的预处理，劣势是不能处理"非离群型"残留死。
- **vs token-level outlier 研究 (Sun 2024; Dettmers 2022)**：他们关注的是量化中个别 token 的尖峰；本文关注的是 dimension-level outlier——在每个 token 上都偏离零的固定维度，几何性质完全不同。
- **vs Lu et al. 2025（ESMFold 维度离群）/ Wang et al. 2025（低秩死特征）**：Lu 等首先观察到 ESMFold 的同类离群但未给出诊断/解决方案；Wang 等把死率归到 attention 激活的低秩结构。本文是第一次给出跨模态可预测的诊断量 + 解析公式 + 极简解决方案。
- **vs 早期 SAE 工作（Bricken 2023b; Gao 2024 用过 mean-centering）**：他们用得不一致，没有判据。本文用 $\gamma$ 把"何时必须 center"原则化，把它从经验 trick 升级为理论支持的标准流程。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把死特征从"训练动力学问题"重新框定为"激活几何问题"，从高维概率推出无拟合参数的死率公式，是范式级的视角转变。
- 实验充分度: ⭐⭐⭐⭐⭐ 454 个真实模型-层 + 控制 $\gamma$ 的合成实验 + 多模态（语言/视觉/蛋白/基因）+ bias 冻结消融，证据链非常完整。
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导和 figure 配合极佳，关键洞察用一句话能讲清，附录补全了所有可质疑的细节（重尾激活、几何中位数、跨层迁移）。
- 价值: ⭐⭐⭐⭐⭐ 直接给出可立即落地的 mean-centering 方案 + 训练前诊断 $\gamma$，对所有训 SAE 的可解释性研究者来说是必读的工程改进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](sparse_autoencoders_are_topic_models.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](../../CVPR2026/interpretability/beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)

</div>

<!-- RELATED:END -->
