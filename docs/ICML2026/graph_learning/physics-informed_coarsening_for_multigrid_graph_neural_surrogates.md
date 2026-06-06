---
title: >-
  [论文解读] Physics-Informed Coarsening for Multigrid Graph Neural Surrogates
description: >-
  [ICML 2026][图学习][图神经网络] 本文为固体力学有限元仿真训练了一个 Encoder-Processor-Decoder 多重网格 GNN 代理模型，核心创新是把"粗化（downsampling）时选哪些节点"从几何启发式（FPS）/学习注意力换成"按动量守恒方程的离散残差打分再 TopK"…
tags:
  - "ICML 2026"
  - "图学习"
  - "图神经网络"
  - "物理残差"
  - "固体力学代理"
  - "网格粗化"
  - "长时滚动"
---

# Physics-Informed Coarsening for Multigrid Graph Neural Surrogates

**会议**: ICML 2026  
**arXiv**: [2605.31013](https://arxiv.org/abs/2605.31013)  
**代码**: [项目主页](https://sites.google.com/view/physics-informed-coarsening)  
**领域**: 科学计算 / 图神经 PDE 代理 / 固体力学  
**关键词**: 多重网格 GNN, 物理残差, 固体力学代理, 网格粗化, 长时滚动

## 一句话总结
本文为固体力学有限元仿真训练了一个 Encoder-Processor-Decoder 多重网格 GNN 代理模型，核心创新是把"粗化（downsampling）时选哪些节点"从几何启发式（FPS）/学习注意力换成"按动量守恒方程的离散残差打分再 TopK"，从而把粗层算力倾斜到应力集中、接触界面、大变形等动力学关键区域，在 DeformingPlate 上把 rollout RMSE 从 SOTA 的 $11.46\times 10^{-3}$ 降到 $6.5\times 10^{-3}$（提升约 43%）。

## 研究背景与动机

**领域现状**：用神经网络替代 FEM 做 PDE 仿真已经在流体（Navier-Stokes、湍流、翼型）上跑出了几个数量级的加速，主流架构是 MeshGraphNet 系的 Encode-Process-Decode 图神经网络，配合多重网格/U-Net 风的分层消息传递（MultiScale MeshGraphNets、BSMS-GNN、Multi-Scale GNN、HCMT、UNISOMA）来缓解深层 GNN 的过平滑、提升长程信息传播。

**现有痛点**：（i）固体力学被严重低估——它和流体不一样，存在大变形、塑性、接触、应力集中这些强非线性局部现象，主流 benchmark（基本全是流体）测不出来；（ii）多重网格架构里"粗化时选哪些节点保留"这个核心设计普遍用 farthest point sampling（FPS）这类纯几何启发式，或者用学习到的 attention 分数——前者完全不看物理，会均匀铺满整个 domain，把算力浪费在没什么物理活动的安静区域；后者训练时容易跑偏。

**核心矛盾**：粗层节点数有限（本文固定 50%），如果按几何均匀分布，应力集中和接触界面这些"动力学关键但空间上局部"的区域分到的粗层算力就不够，长时 rollout 时这些区域的误差最先发散，反过来污染整个解。

**本文目标**：（i）设计一个粗化准则，让粗层节点优先落在"物理上重要"的区域；（ii）这个准则要在 quasi-static 超弹、瞬态非线性弹性、含接触的弹塑性这几种 regime 都能用；（iii）配套发布固体力学 benchmark 弥补缺口。

**切入角度**：作者借了经典 FEM 里"residual-based adaptive mesh refinement"的老想法——FEM 自适应加密就是按 PDE 残差大的地方加密网格。把这个思路平移到 GNN 多重网格：粗化时按动量守恒方程的离散残差打分。

**核心 idea**：用"动量守恒方程的离散残差范数"作为节点重要性得分，TopK 选出残差最大的节点构成粗图，让多重网格层级自然向应力集中区/接触界面/大变形区倾斜。

## 方法详解

### 整体框架
输入：时刻 $t$ 的 3D 非结构网格 $\mathcal{G}=(\mathcal{V},\mathcal{E})$ 及节点物理场 $\bm{u}^t$（位移/速度/位置）。
输出：下一步增量 $\bm{u}^{t+1}=\bm{u}^t+\Phi_\theta(\bm{u}^t,\mathcal{G})$，残差式时间推进，便于长时 rollout。

主干为 Encoder–Processor–Decoder。Encoder 用逐点 MLP 把节点特征升到隐维度 $h$；Decoder 把最终隐特征映回 $\mathbb{R}^3$。Processor 在隐空间交替执行三种算子：
（i）GraphNet 块 $\mathrm{GN}$ 做细网格消息传递（MeshGraphNet 标准更新规则，先更新边再聚合到节点）；
（ii）Downsampling 块 $\mathrm{DN}$ 把细图压成 $n_s=0.5n$ 的粗图；
（iii）Upsampling 块 $\mathrm{UP}$ 用 KNN 把粗图特征插值回细图，与细层特征融合后再过若干 GraphNet 块。整体形成 $\mathcal{G}\to\tilde{\mathcal{G}}\to\mathcal{G}_c\to\tilde{\mathcal{G}}$ 的 U-Net 式调度，粗层负责扩大有效感受野、传播全局信息。

关键 twist 在 $\mathrm{DN}$ 内部：它不是按几何选节点，而是借用主 Decoder $\phi_{\mathrm{dec}}$ 临时把当前隐特征解码成物理量，算出每个节点的"动量守恒残差"$\bm{r}_i^t$，再按 $s_i^t=\|\bm{r}_i^t\|_2$ 排序 TopK。

### 关键设计

1. **基于动量守恒残差的节点物理打分**：

    - 功能：给每个节点算一个标量得分 $s_i^t$，刻画"该节点附近预测出的物理场对动量守恒方程的违背程度"，作为节点重要性的 a posteriori 指标。
    - 核心思路：先用主 Decoder $\hat{\bm{u}}^t=\phi_{\mathrm{dec}}(\tilde{\mathcal{G}})$ 把当前隐图解码到物理空间（这一步是"借用"，不参与最终预测），用预测场 $\hat{\bm{u}}^t$ 算应力 $\hat{\bm{\sigma}}^t$；瞬态情形下残差为 $\bm{r}_i^t=\rho_i\ddot{\hat{\bm{u}}}_i^t-(\nabla_h\cdot\hat{\bm{\sigma}}^t)_i-\rho_i\mathbf{b}_i^t$，准静态情形丢掉惯性项变为平衡残差 $\bm{r}_i^t=-(\nabla_h\cdot\hat{\bm{\sigma}}^t)_i-\rho_i\mathbf{b}_i^t$；散度 $\nabla_h\cdot$ 用固定的 mesh-based 离散算子重构，得分取 $s_i^t=\|\bm{r}_i^t\|_2$。残差每个时刻自回归重算一次。
    - 设计动机：FEM 里残差大的地方就是物理上"算不准/动得猛"的地方，把这个准则平移到 GNN 粗化，等价于让多重网格层级自动捕捉应力集中、接触界面、边界过渡这些"哪怕几何上稀疏但物理上关键"的区域；而且这是个无监督信号，不需要额外标签。一个值得注意的工程选择是**复用主 Decoder** 来算物理量（而不是独立训一个 scoring decoder），消融显示独立 decoder 反而掉点，说明共享 decoder 强迫主分支学到"物理上自洽"的表征。

2. **TopK 物理引导节点选择 + KNN 重网格化**：

    - 功能：把得分向量 $\bm{s}\in\mathbb{R}^n$ 转成 $n_s$ 个粗节点的下标集合 $\mathcal{V}_c$，并重建粗图边集 $\mathcal{E}_c$。
    - 核心思路：节点选择有两种 —— 确定性 TopK $\mathcal{I}=\mathrm{TopK}(\bm{s}^t,n_s)$（保留得分最高的 $n_s$ 个），或者按 $p_i=s_i/\sum_j s_j$ 做 categorical 采样的概率式选择。边集构建也有两种 —— 继承细网格的诱导子图 $\mathcal{E}_c=\{(i,j)\in\mathcal{E}\mid i,j\in\mathcal{V}_c\}$，或在被选节点上用欧氏 KNN 重新拉一张图（"remeshing"）。实验最佳组合是 **TopK + remeshing**：TopK 比概率采样在 rollout RMSE 上从 $13.1\times 10^{-3}$ 一路降到 $6.5\times 10^{-3}$，remeshing 比继承连通性更稳定。
    - 设计动机：TopK 在物理引导下比随机采样更激进，所有粗层算力都给最关键的几个区域；KNN 重网格化避免了"残差大的节点恰好被原始边断开"的拓扑碎片化，对长程信息传播至关重要——这点和 BSMS（拓扑 bi-stride 池化）形成鲜明对比，BSMS 一定要保连通性，但作者发现纯按物理选+重连反而更好。

3. **Encoder-Processor-Decoder + KNN 上采样融合**：

    - 功能：把粗图处理后的全局信息送回细图，同时保留细层局部精度。
    - 核心思路：粗层经过若干 GraphNet 块得到 $\tilde{\mathcal{G}}_c^{n_s\times h}$ 后，用 $k$-NN（在物理空间欧氏距离意义下）把每个细节点的特征插值成它最近 $k$ 个粗节点特征的加权和，再和细层原始特征拼接/融合，最后再过若干细层 GraphNet 块做局部精修。时间推进采用残差式 $\bm{u}^{t+1}=\bm{u}^t+\Phi_\theta(\bm{u}^t,\mathcal{G})$，对应经典数值积分格式，对长时 rollout 稳定性帮助很大。
    - 设计动机：传统单尺度 GNN 受限于消息传递的 $k$-hop 半径，捕捉不到全局耦合；多重网格的粗层一跳跨越远距离，但只有粗层是不够的（会丢失局部应力梯度）。Encoder-Processor-Decoder + 残差时间步是 MeshGraphNet 沿用下来的成熟设计，本文的新颖度全压在 Processor 内部的"物理引导粗化"上，证明换粗化策略就能在不改主干的情况下打过 7 个强 baseline。

### 损失函数 / 训练策略
直接监督下一步状态预测，逐节点 MSE 损失。AdamW 优化器，所有 baseline 训 30 个 epoch（约 $10^6$ 步）、同一 protocol，每个配置跑 3-5 个 seed 取均值；所有多重网格模型固定粗化比例为 50%（粗图节点数 = 细图的一半），controlling for capacity，把变量隔离到"粗化策略"本身。NVIDIA A100。

## 实验关键数据

### 主实验：DeformingPlate 上对比 7 个 SOTA

| 方法 | Rollout RMSE ($\times 10^{-3}$) ↓ | 1-step RMSE ($\times 10^{-3}$) ↓ | #Params ↓ |
|------|------|------|------|
| MeshGraphNets | 12.75 | 0.10 | 2.8M |
| BSMS-GNN | 16.60 | 0.15 | 2.1M |
| Transolver++ | 29.80 | 1.00 | 722K |
| Transformer GNN | 24.97 | 1.20 | 3.5M |
| Multi-Scale GNN | 15.7 | 0.10 | 3.1M |
| HCMT | 12.97 | 0.14 | 2.53M |
| UNISOMA | 11.46 | 0.16 | 2.85M |
| **Ours (Physics-informed Multigrid)** | **6.50** | **0.095** | 2.9M |

Rollout 误差直接对半砍（从 UNISOMA 的 11.46 → 6.50，提升约 43%），1-step 误差也是全场最低；参数量持平，没有靠加参数取胜。

### 消融实验：多重网格架构下不同采样策略对比

| 采样策略 | Rollout RMSE ($\times 10^{-3}$) ↓ | 1-step RMSE ($\times 10^{-5}$) ↓ |
|------|------|------|
| BSMS（拓扑 bi-stride） | 16.60 | 15 |
| FPS（无 remeshing） | 15.0 | 10.31 |
| Attention-based | 8.1 | 17.10 |
| FPS（带 remeshing） | 8.0 | 9.74 |
| Physics-informed + 概率采样 | 13.1 | 11.32 |
| **Physics-informed + TopK（Ours）** | **6.5** | **9.57** |

跨数据集泛化（Table 4，与 MGN/FPS 对比）：BeamSimple 上 Rollout $1.44\times 10^{-1}$（vs MGN 1.72, FPS 1.56）；SpindleUpsetting 上 1-step $11.24\times 10^{-3}$（vs MGN 11.92, FPS 11.66），rollout 略输 FPS（2.96 vs 2.60），说明残差式打分在塑性+接触的极端 regime 里没有压倒性优势但仍 competitive。粗层宽度消融（Table 5）：$h_c=128$ vs $h_c=256$，加宽反而掉点（6.59→15.72×10⁻³），说明粗层 capacity 已经够，瓶颈在选谁、不在多大。

### 关键发现
- **粗化策略 > 架构容量**：同样主干下，把 FPS 换成 physics-informed TopK 就把 rollout 误差从 8.0 降到 6.5；而 MeshGraphNet 在没有多重网格的情况下达到 12.75，说明"分层"和"选谁"两件事是叠加增益。
- **TopK 显著胜过概率采样**（13.1 → 6.5）：在 50% 这种高粗化率下，随机性引入的方差损害大于探索的收益。
- **决策器复用很重要**：用独立 decoder 算物理打分会掉点，共享 decoder 强迫表征同时支持"预测下一步"和"算残差"，反而促进物理自洽。
- **打分应该包含边界/接触节点**：消融显示"all nodes scoring"稳胜"only normal nodes"，因为反作用力和约束节点本身就携带很强的物理信号。
- **加粗层宽度无效**：$h_c=256$ 比 128 反而更差，提示在 mesh-based 模型上盲目加 capacity 不如改采样准则。

## 亮点与洞察
- **把经典 FEM 的 residual-based adaptive refinement 平移到 GNN 多重网格的粗化**：这个 analogy 既给了方法可解释性（"残差大的地方=物理活动剧烈"），也直接复用了几十年的数值分析直觉。在"用 ML 替代数值方法"的浪潮里这种"借经典数值方法的设计哲学回来指导 ML 架构"的范式特别值得关注。
- **得分函数零额外参数**：物理残差完全由主 Decoder 输出 + 固定离散散度算子算出，不引入新的可学习模块，却带来巨大的归纳偏置。
- **Decoder 共享是一个有用的 trick**：让同一个 decoder 同时承担"最终输出"和"中间打分"两个任务，自然地把"物理一致性"作为辅助正则注入主分支——这个 idea 可以迁移到任何需要"中间监督信号"但又不想增加参数的场景。
- **TopK + KNN 重网格化的组合**：放弃保连通性、用 KNN 重新连边，反而比保连通性好，提示在多重网格 GNN 里"拓扑保真度"可能没"语义相关性"重要。

## 局限与展望
- 作者承认：**残差打分对"复杂材料、扭曲网格、富接触场景"非常 nontrivial**——需要离散散度重构和访问有物理意义的场，本质上"把经典数值方法的零件搬了一部分回来"，不再是纯 ML black-box。
- 在 SpindleUpsetting（重塑性+接触）的 rollout 上反而输给纯 FPS，说明在极端非线性 regime 残差信号本身可能噪声很大；这暴露了"用主 Decoder 解码物理量"的脆弱性——如果主预测不准，残差也不准，可能引发恶性循环。
- 只在 50% 这一个固定粗化率下评测，没看不同压缩率下的曲线；多层级嵌套（粗-更粗-超粗）也没有研究。
- 改进方向：把残差打分与梯度向量场结合（不只看模长还看方向）；自适应粗化率（动力学剧烈时多保点）；和 PINN 损失结合，让残差既用来打分也用来正则化预测。

## 相关工作与启发
- **vs MeshGraphNets**：本文的 Encoder-Processor-Decoder 主干和残差时间步直接继承自 MGN，但加了多重网格层级和物理引导粗化；rollout 从 12.75 → 6.50，纯结构性改进。
- **vs BSMS-GNN（拓扑 bi-stride 池化）**：BSMS 用图拓扑做 bi-stride 池化保连通性，本文用物理残差 TopK + KNN 重连边，结果上 BSMS 16.60 → Ours 6.50，证明"语义相关性"压倒"拓扑保真度"。
- **vs FPS-based 多重网格（Multi-Scale GNN/Garnier 2024）**：纯几何均匀覆盖 vs 物理引导聚焦，在固体力学这种"局部应力集中决定全局动力学"的场景下，物理引导明显占优。
- **vs UNISOMA**：UNISOMA 用固定数量的 slice token 做注意力压缩，在尖锐接触/应力集中处会丢失局部细节；本文显式按残差选节点，恰好补上了这个短板。
- **vs PINN**：PINN 把残差当 loss 监督预测，本文把残差当架构信号引导粗化——同一信号、两种用法，可以组合。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](quantile-free_uncertainty_quantification_in_graph_neural_networks.md)
- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)
- [\[AAAI 2026\] On Stealing Graph Neural Network Models](../../AAAI2026/graph_learning/on_stealing_graph_neural_network_models.md)
- [\[ICML 2026\] Deep Neural Sheaf Diffusion](deep_neural_sheaf_diffusion.md)

</div>

<!-- RELATED:END -->
