---
title: >-
  ICML2026 图学习方向9篇论文解读
description: >-
  9篇ICML2026的图学习方向论文解读，涵盖图神经网络、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "图学习"
  - "论文解读"
  - "论文笔记"
  - "图神经网络"
  - "扩散模型"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**🧪 ICML2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (21)](../../ACL2026/graph_learning/index.md) · [📷 CVPR2026 (8)](../../CVPR2026/graph_learning/index.md) · [🔬 ICLR2026 (21)](../../ICLR2026/graph_learning/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/graph_learning/index.md) · [🧠 NeurIPS2025 (52)](../../NeurIPS2025/graph_learning/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/graph_learning/index.md)

🔥 **高频主题：** 图神经网络 ×2 · 扩散模型 ×2

**[Anchor-guided Hypergraph Condensation with Dual-level Discrimination](anchor-guided_hypergraph_condensation_with_dual-level_discrimination.md)**

:   AHGCDD 把超图凝聚 (HGC) 从"先训练结构生成器、再匹配训练轨迹"的解耦范式重写为端到端框架：用 Heat-Kernel-PageRank 把结构信息塞进初始化特征、用 anchor-guided 思路按特征距离合成稀疏可学的超边，再用粗+细双级判别损失 (类原型 MMD + 实例级对比) 代替昂贵的 HNN 重训练，在 5 个超图基准上 ≥SOTA 同时最高 144× 加速。

**[Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)**

:   本文把经典谱 GNN 的单变量特征值滤波器 $g(\lambda_i)$ 推广为双变量滤波器 $g(\lambda_i,\lambda_j)$，把信号从节点域抬到节点对域，理论上能逼近 Local 2-GNN（超越 1-WL），并通过低秩张量分解避开 $n^2\times n^2$ 显式计算，在异质图节点分类和子结构计数上拿到强结果。

**[Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)**

:   本文把图扩散反向 SDE 的采样轨迹看成 Riemannian 统计流形上的参数曲线，用 Fisher-Rao 度量推出一个无需训练的 Drift Variation Score (DVS) 来度量轨迹的局部"信息曲率"，并据此自适应缩放步长，使每步在信息流形上前进等长，从而在分子（QM9/ZINC250k）和图（Planar/SBM/Ego）生成中以更少步数取得更高 FCD / MMD 保真度。

**[Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)**

:   R-GFM 把"不同 hop 数"的子图当作上层 Graph-of-Graphs 的节点，再用一套动态 MoE 路由把每个 GoG 分配到曲率最匹配的 Riemannian 流形（双曲 / 欧氏 / 球面），同时解决了现有图基础模型固定 receptive field 与单一 Euclidean 嵌入两个先天缺陷，下游最高带来 49% 相对提升。

**[On the Expressive Power of GNNs to Solve Linear SDPs](on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)**

:   本文从 Weisfeiler–Leman 层级的角度首次刻画了学习线性 SDP 解所需的最小 GNN 表达力，证明标准的变量-约束二部图消息传递（VC-WL）和高阶 VC-2-WL 都不够，而 2-FWL 等价的 VC-2-FWL 架构足以仿真 PDHG 求解器的更新步骤，并在合成与 SDPLIB 上把高质量预测用作 warm-start，最多带来约 80% 的加速。

**[Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)**

:   PolyNSD 把 Sheaf 神经网络的"一步空间扩散"换成对归一化 sheaf 拉普拉斯的可学习 $K$ 阶多项式谱滤波器，用 Chebyshev 三项递推稳定计算，单层就拥有 $K$-hop 感受野和可控的低/带/高通响应；意外的发现是只用对角 restriction maps 就能超越所有需要稠密大维 stalk 的现有 NSD，参数、显存、运行时间都大幅下降。

**[Quantile-Free Uncertainty Quantification in Graph Neural Networks](quantile-free_uncertainty_quantification_in_graph_neural_networks.md)**

:   QpiGNN 提出"无需分位输入、无需后处理"的 GNN 节点级预测区间框架，用双头 GNN（一头预测均值、一头预测半宽）配合直接优化"覆盖率+区间宽度"的标签级联合损失，在 19 个合成/真实数据集上平均覆盖率提高 22%、区间宽度收窄 50%。

**[Structure-Centric Graph Foundation Model via Geometric Bases](structure-centric_graph_foundation_model_via_geometric_bases.md)**

:   SCGFM 把跨域图基础模型重写为度量测度空间上的"三角测量"问题：学一组 $K$ 个可训练几何基 $\{B_k\}$，每个图用其与各基的 Gromov–Wasserstein 距离 $\delta_k$ softmax 得到一组结构坐标 $\mathbf{w}$，再用基上的 OT plan 把节点特征汇聚到统一维度，从而摆脱"必须对齐节点特征空间"的传统 GFM 桎梏，在 in-domain 与 OOD 少样本图/节点分类上都打过 baseline。

**[Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)**

:   本文把 CNF 公式建模成「子句–文字超图 + 子句关联图」，并在变量级把表示拆成极性不变与极性等变两部分，再用极性翻转一致性正则训练，把 unsat-core 变量预测精度显著拉高一档。
