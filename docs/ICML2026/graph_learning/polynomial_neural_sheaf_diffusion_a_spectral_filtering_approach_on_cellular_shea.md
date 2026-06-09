---
title: >-
  [论文解读] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves
description: >-
  [ICML 2026][图学习][神经层扩散] PolyNSD 把 Sheaf 神经网络的"一步空间扩散"换成对归一化 sheaf 拉普拉斯的可学习 $K$ 阶多项式谱滤波器，用 Chebyshev 三项递推稳定计算，单层就拥有 $K$-hop 感受野和可控的低/带/高通响应…
tags:
  - "ICML 2026"
  - "图学习"
  - "神经层扩散"
  - "Chebyshev 多项式滤波"
  - "异质图"
  - "过平滑"
  - "对角传输映射"
---

# Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves

**会议**: ICML 2026  
**arXiv**: [2512.00242](https://arxiv.org/abs/2512.00242)  
**代码**: 无  
**领域**: 图神经网络 / Sheaf 神经网络 / 谱图过滤 / 异质性图  
**关键词**: 神经层扩散, Chebyshev 多项式滤波, 异质图, 过平滑, 对角传输映射

## 一句话总结
PolyNSD 把 Sheaf 神经网络的"一步空间扩散"换成对归一化 sheaf 拉普拉斯的可学习 $K$ 阶多项式谱滤波器，用 Chebyshev 三项递推稳定计算，单层就拥有 $K$-hop 感受野和可控的低/带/高通响应；意外的发现是只用对角 restriction maps 就能超越所有需要稠密大维 stalk 的现有 NSD，参数、显存、运行时间都大幅下降。

## 研究背景与动机
**领域现状**：GNN 在同质图上很成功，但在异质图（相邻节点类别不同）和深层堆叠（过平滑）下都会出问题。一类解药是 cellular sheaf 理论：给每个节点配一个局部特征空间（stalk），给每条边配一个线性 restriction map，由此构造的 sheaf Laplacian 可以做"传输感知"扩散，比各向同性消息传递更适合异质性。Bodnar 等 2022 年的 Neural Sheaf Diffusion (NSD) 把 sheaf Laplacian 学出来，实现了 SOTA。

**现有痛点**：NSD 本质是一步空间传播 $X^{(t+1)}=X^{(t)}-\sigma(\Delta_{\mathcal{F}(t)}(I_{nd}\otimes W_1^{(t)})X^{(t)}W_2^{(t)})$，存在四个结构问题：(i) 每层只走一步，长程依赖必须靠堆深层，进一步加剧过平滑；(ii) 依赖稠密的 per-edge restriction maps（diagonal/bundle/general），参数和显存都被 stalk 维度 $d$ 拖累；(iii) 训练里需要 SVD 之类的归一化或分解，数值不稳；(iv) 模型性能高度依赖大 stalk 维数，无法解耦"精度"与"计算代价"。

**核心矛盾**：sheaf Laplacian 的"传输感知 + partial order"优势是空间局部的，但要把它用到多跳交互上必须重复堆层；而堆层又触发过平滑和昂贵的 sheaf 重学。换句话说，**长程传播被绑死在层深上**。

**本文目标**：(1) 在保留 sheaf transport 表达力的前提下，让单层就能做 $K$-hop 混合；(2) 把扩散的频率响应从"隐式低通"改成可学习；(3) 解耦性能与 stalk 维度，让 diagonal restriction 就够用；(4) 给出与标准 GNN 同阶的稳定多项式实现。

**切入角度**：作者借鉴 ChebNet/GPRGNN 这类经典谱多项式滤波器——既然图 Laplacian 上 Chebyshev 递推已经被验证为稳定且高效，那 sheaf Laplacian 是对称半正定矩阵，理论上同一套谱泛函分析（$p(L)=Up(\Lambda)U^\top$）完全适用。

**核心 idea**：用 $p(L)=\sum_{k=0}^K c_k L^k$ 替代单步 $(aI+bL)$ 扩散，并通过谱重缩放 $\widetilde{L}=2L/\lambda_\text{max}-I$ 让 Chebyshev 基稳定有界，把 sheaf 扩散一次性升级为可控的谱滤波层。

## 方法详解

### 整体框架
PolyNSD 想解决的是"sheaf 神经网络的长程能力被绑死在层深上"这个矛盾：sheaf Laplacian 的传输感知扩散很适合异质图，但要覆盖多跳就得堆层，堆层又触发过平滑和昂贵的 sheaf 重学。它的做法是把每个节点先 lift 到一个 $d$ 维 stalk、由 sheaf learner（一个 MLP）从相邻特征对学出 edge-wise restriction map 并拼出 sheaf Laplacian $L\in\mathbb{R}^{Nd\times Nd}$，然后不再做单步扩散，而是在 $L$ 上学一个 $K$ 阶多项式谱滤波器，让单层就拥有 $K$-hop 感受野和可控的频率响应。整个算子与现有 SheafNN 兼容，是即插即用的传播层替换。

### 关键设计

**1. Sheaf Laplacian 上的 Chebyshev 多项式谱滤波：把单步扩散换成单层 $K$-hop 滤波器**

Bodnar 的原始 NSD 本质是 $X^{(t+1)}=X^{(t)}-\sigma(\Delta_{\mathcal{F}}(\dots))$，等价于在 Laplacian 上做 $aI+bL$ 的一阶变换，单层只走一跳，长程依赖只能靠堆深层换来。PolyNSD 的出发点是：$L$ 是对称半正定矩阵，$L=U\Lambda U^\top$，因此多项式 $p(L)=\sum_{k=0}^K c_k L^k$ 通过谱泛函就等于 $Up(\Lambda)U^\top$——它在第 $i$ 个 sheaf Fourier 模态上的乘子正好是 $p(\lambda_i)$。这样只要让 $p$ 的形状可学，单层就能自由实现低通（$p$ 单调递减、平滑）、带通（$p$ 带状、提取中频）或高通（$p$ 递增、保留 disagreement），而频率选择性由系数自己学出来。更关键的是 **Proposition 1** 保证了空间局部性：$(p(L))_{vu}=0$ 当 $\text{dist}_G(v,u)>K$，也就是 $K$ 阶多项式严格对应 $K$-hop 混合，既不需要堆 $K$ 层、也不需要重复学 sheaf。

直接学单项式系数 $\{c_k\}$ 在数值上是灾难（Vandermonde 病态），所以这里改在重缩放后的算子上用第一类 Chebyshev 多项式 $T_k(\xi)=\cos(k\arccos\xi)$ 作基，并用三项递推稳定计算，最终的滤波器写成 $p_\theta(\widetilde{L})=\sum_{k=0}^K \alpha_k T_k(\widetilde{L})$。

**2. 谱重缩放 + 凸混合系数：让 $K$ 再大也稳定、且只衰减不放大 disagreement**

Chebyshev 基只在 $[-1,1]$ 上有界，一旦 $|\xi|>1$ 就指数增长，所以必须先把谱压回这个区间，做仿射缩放 $\widetilde{L}=2L/\lambda_\text{max}-I$。其中归一化的 $\Delta_\mathcal{F}$ 直接取 $\lambda_\text{max}=2$ 即安全；未归一化的 $L_\mathcal{F}$ 用 Gershgorin 上界或几步 power iteration 估 $\lambda_\text{max}$，避免每步做特征分解。缩放之后 $|T_k(\xi)|\le 1$，再把组合系数取成凸混合 $\alpha=\text{softmax}(\eta)$，就能让 $\|p_\theta(\widetilde{L})\|_2\le 1$，得到一个非膨胀算子。

这一步不只是数值技巧，它直接换来了稳定性保证。**Proposition 2** 证明：若 $0\le p(\lambda)\le 1$ 在 $\sigma(L)$ 上成立，则 Dirichlet 能量 $\langle p(L)x,Lp(L)x\rangle=\sum_i\lambda_i p(\lambda_i)^2\hat x_i^2\le\sum_i\lambda_i\hat x_i^2=\langle x,Lx\rangle$，即滤波器只能 damp 节点间的 disagreement、不会放大它，训练在 $K$ 很大时仍不发散。又因为多项式滤波器互相对易 $p(L)q(L)=(pq)(L)$，堆多层等价于学一个更高阶的多项式，这让"频率选择性"成了可分析、可控的一等设计。

**3. High-pass skip + gated residual + 对角 restriction：把表达力从大 stalk 转移到频率响应**

纯扩散天然带低通偏置，对异质图不利，所以 PolyNSD 额外并入一条高通分量 $h_\text{hp}=x-\lambda_\text{max}^{-1}Lx$，与多项式输出线性合成 $z=p_\theta(\widetilde{L})x+\alpha_\text{hp}h_\text{hp}$。因为 $\widetilde{L}$ 与 $L$ 共享特征基，整条线性映射在 eigenbasis 上仍是对角的，等效乘子为 $m(\lambda)=p_\theta(2\lambda/\lambda_\text{max}-1)+\alpha_\text{hp}(1-\lambda/\lambda_\text{max})$，单个标量 $\alpha_\text{hp}$ 就能解析地把频率响应往高通方向掰。最后用 gated residual $x^+=(I+\tanh\varepsilon)x-\phi(z)$ 输出，其中 $\phi$ 是 1-Lipschitz 非线性，保证整层 Lipschitz 常数有显式上界，允许加深而不爆梯度。

这套组合让 restriction map 可以退到最便宜的对角形式：参数只剩 $K+1$ 个标量加 sheaf learner，单层代价 $\mathcal{O}(K\cdot\text{nnz}(L)\cdot C)$。传统 NSD 必须靠大 stalk 维和 bundle/general restriction 来补表达力，而这里把表达力转移到"频率响应可学"上，对角 restriction 就能匹敌稠密形式——这正是后面实验里"性能不再依赖大 stalk"的来源。为验证这一点不是多项式滤波单独的功劳，论文还专门构造了去掉 sheaf transport（stalk=1、transport=identity）的 **PolySpectralGNN** 作为消融对照。

### 损失函数 / 训练策略
任务是标准监督节点分类（交叉熵），重点不在 loss 而在传播算子。训练有几个落地要点：$\lambda_\text{max}$ 用 power iteration 估以避开每步特征分解；凸混合 $\alpha=\text{softmax}(\eta)$ 把 $\|p\|_\infty\le 1$ 直接写进参数化；谱重缩放后整层算子 Lipschitz $\le 1$，允许加深而不爆梯度；对角 restriction 作为默认形式，bundle/general 仅用于对比。

## 实验关键数据

### 主实验
9 个节点分类数据集（同质 Cora/Citeseer/Pubmed，异质 Texas/Wisconsin/Film/Squirrel/Chameleon/Cornell）。前三名按颜色标。这里选最异质（同质度 0.11）的 Texas 和最同质（0.81）的 Cora 摘出来对比：

| 模型 | Texas (异质 0.11) | Wisconsin | Squirrel | Chameleon | Cora (同质 0.81) | Citeseer | Pubmed |
|------|------------------|-----------|----------|-----------|------------------|----------|--------|
| **DiagPolySD (本文)** | **90.00±4.68** | 88.63±3.59 | **56.61±2.06** | **71.45±2.03** | 88.79±1.13 | 77.74±1.26 | 89.70±0.32 |
| BundlePolySD | 89.74±5.32 | 89.41±4.04 | 55.76±2.02 | 71.18±1.46 | 88.33±1.34 | 77.57±1.55 | **89.75±0.34** |
| Diag-NSD (Bodnar) | 85.67±6.95 | 88.63±2.75 | 54.78±1.81 | 68.68±1.73 | 87.14±1.06 | 77.14±1.85 | 89.42±0.43 |
| Gen-NSD (Bodnar) | 82.97±5.13 | 89.21±3.84 | 53.17±1.31 | 67.93±1.58 | 87.30±1.15 | 76.32±1.65 | 89.33±0.35 |
| GGCN | 84.86±4.55 | 86.86±3.29 | 55.17±1.58 | 71.14±1.84 | 87.95±1.05 | 77.14±1.45 | 89.15±0.37 |
| H2GCN | 84.86±7.23 | 87.65±4.98 | 36.48±1.86 | 60.11±2.15 | 87.87±1.20 | 77.11±1.57 | 89.49±0.38 |
| GPRGNN | 78.38±4.36 | 82.94±4.21 | 31.61±1.24 | 46.58±1.71 | 87.95±1.18 | 77.13±1.67 | 87.54±0.38 |
| GCN | 55.14±5.16 | 51.76±3.06 | 53.43±2.01 | 64.82±2.24 | 86.98±1.27 | 76.50±1.36 | 88.42±0.50 |

DiagPolySD 在最异质的 Texas 上比 Diag-NSD 提升 4.3 点、比 GCN 提升 35 点；在 Squirrel/Chameleon 上把 SOTA 推到 56.6 / 71.5；在同质数据上仍持平或微胜。

### 消融实验

| 配置 | 关键变化 | 现象 |
|------|---------|------|
| Full DiagPolySD ($K>1$) | 完整模型 | 全部 9 数据集 top-3 |
| $K=1$（退化为 NSD） | 多项式阶降为 1 | 等价于 $aI+bL$，回到 Bodnar 单层扩散 |
| 不做谱重缩放 | 直接学 $\{c_k\}$ 单项式 | Vandermonde 病态，训练发散 |
| 去掉 high-pass skip | 只用多项式响应 | 低通偏置变强，异质数据掉点 |
| 去掉 gated residual | 用普通残差 | 深层梯度不稳 |
| PolySpectralGNN | stalk=1, identity transport | 异质数据明显劣化（Texas 64.6 vs 90.0），证明 sheaf transport 在异质性下不可替代 |
| Bundle/General restriction | 换稠密 restriction | 与对角接近，性能不再依赖大 stalk |

### 关键发现
- **对角 restriction 就够用**：传统 NSD 需要 bundle/general 才能挤出性能，参数和显存都被 stalk 维拖累；PolyNSD 用对角就持平甚至超过——把"表达力"从 restriction 维度转移到了多项式频率响应，是范式级的解耦。
- **$K>1$ 一致带来收益**：把多项式阶从 1 推到更高在所有数据集上都有提升，验证"单层 $K$-hop"比"堆 $K$ 层 NSD"既快又准。
- **sheaf 在异质性下不可替代**：PolySpectralGNN（去掉 sheaf 只留 Chebyshev 滤波）在 Texas/Wisconsin/Cornell 这种极异质数据上落后 20+ 点，说明 transport-aware 才是异质性图的关键，多项式滤波只是把它放大。
- **谱重缩放是稳定性必要条件**：去掉 $\widetilde{L}=2L/\lambda_\text{max}-I$ 训练直接发散，凸混合 $\text{softmax}$ 系数也是把 $\|p\|_2\le 1$ 落到代码层面的关键。

## 亮点与洞察
- **把 ChebNet 的"老把戏"搬到 sheaf Laplacian** 是看似显然但工程上不平凡的工作：sheaf Laplacian 维度是 $Nd\times Nd$，估 $\lambda_\text{max}$、做三项递推、保证谱在 $[-1,1]$ 都需要仔细处理；论文把这些细节都跑通了。
- **范式 overturn：性能 ≠ 大 stalk**：以往社区都默认 NSD 要靠大 stalk + general restriction 才能赢，PolyNSD 用对角 restriction + 多项式滤波直接打破这个共识，把 sheaf 框架从"贵族"拉成"平民"，路径效仿了 ChebNet 把谱图卷积平民化的故事。
- **频率响应可解释**：能学到的 $p_\theta(\lambda)$ 可以直接画出来——异质数据上学出来的多带通甚至高通响应，同质数据上学出来的偏低通——为 sheaf 模型提供了 spectral interpretability，论文还做了 oversquashing 的长程影响衰减分析。
- **可迁移设计模板**：(1) "稀疏算子 + 谱重缩放 + Chebyshev 递推" 三件套适用于任何对称 PSD 算子（GNN/隐式神经网络/PDE solver）；(2) "high-pass skip + gated residual" 的组合可缓解任何低通偏置的扩散模型。

## 局限与展望
- **依赖 $\lambda_\text{max}$ 的准确估计**：归一化 $\Delta_\mathcal{F}$ 用 $\lambda_\text{max}=2$ 安全，但对未归一化 $L_\mathcal{F}$ 用 power iteration 估，迭代次数与图谱性质相关，对极端图谱可能不稳。
- **Chebyshev 是默认基但不是唯一选择**：论文承认实现 basis-agnostic，但只系统验证了 Chebyshev；其他正交基（Jacobi、Bernstein）在不同数据上是否更优还有待探索。
- **$K$ 是超参，没有自适应机制**：理想情况下 $K$ 应该按图直径或任务自动选；现在还是手调。
- **sheaf learner 仍是 MLP，存在 instability 风险**：multi-edge consistency 等更复杂结构没有特别处理。
- **大规模图的工程化**：$L$ 是 $Nd\times Nd$，$K$ 次稀疏乘对超大图仍有挑战，论文没在 OGB 级别图上验证。

## 相关工作与启发
- **vs Bodnar et al.（NSD 2022）**：NSD 是 $K=1$ 的特例，PolyNSD 把它扩展到任意阶并附带稳定性证明，把"性能依赖大 stalk"打破。
- **vs ChebNet（Defferrard 2016）**：本质同套谱多项式滤波技巧，但搬到 sheaf Laplacian 上获得了"传输感知 + 频率可控"的双重表达力。
- **vs GPRGNN / FAGCN / H2GCN**：这些是异质图的 SOTA 多项式或频率响应 baseline，但它们用图 Laplacian 而非 sheaf Laplacian；本文在 Texas/Wisconsin/Squirrel 上把它们都甩开 20+ 点，量化了 sheaf transport 的增益。
- **vs PolySpectralGNN（自家 ablation）**：把 sheaf 拿掉只留谱多项式，性能显著下滑，是用 ablation 反过来证明"sheaf 不只是浪费参数"的优雅设计。
- **启发**：把"任何空间扩散模型 → 谱多项式版本"的思路套用到 Hodge Laplacian、normalised attention matrix、graph wavelet operator 上，可能都有立竿见影的收益。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是凭空发明新算子，而是把 ChebNet 的成熟谱滤波框架严格移植到 sheaf Laplacian 上，并且发现"对角 restriction 就够"这一反共识结论；移植+解耦的工作量与洞察都到位。
- 实验充分度: ⭐⭐⭐⭐⭐ 9 个数据集（同质 + 6 种异质）、4 种 restriction、与 14+ baseline 全对比；ablation 把 sheaf vs. 多项式、$K$、谱重缩放、high-pass 一一拆开，还做 oversmoothing/oversquashing/连续 ODE 扩展。
- 写作质量: ⭐⭐⭐⭐ 公式密集但层次清晰，Proposition 1 / 2 给出空间局部性和能量单调性两条理论保证；Fig. 1 把 10 步 pipeline 画得很清楚。
- 价值: ⭐⭐⭐⭐ 把 sheaf 神经网络从"昂贵小众"推向"便宜可用"，并且开放了"在 sheaf Laplacian 上设计谱滤波器"这条研究路径，社区影响可期。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Deep Neural Sheaf Diffusion](deep_neural_sheaf_diffusion.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)
- [\[ICML 2026\] Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach](rethinking_feature_alignment_in_generalist_graph_anomaly_detection_a_relational_.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)

</div>

<!-- RELATED:END -->
