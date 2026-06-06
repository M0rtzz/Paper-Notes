---
title: >-
  [论文解读] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport
description: >-
  [CVPR2026][优化/理论][剪枝] 将视觉 token 裁剪建模为最优传输（OT）下的分布对齐问题，通过最小化完整与裁剪后 token 集合间的 2-Wasserstein 距离…
tags:
  - "CVPR2026"
  - "优化/理论"
  - "剪枝"
  - "optimal transport"
  - "Wasserstein distance"
  - "submodular optimization"
  - "training-free"
  - "MLLM efficiency"
---

# OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport

**会议**: CVPR2026  
**arXiv**: [2602.20205](https://arxiv.org/abs/2602.20205)  
**代码**: [xiwenc1/OTPrune](https://github.com/xiwenc1/OTPrune)  
**领域**: 优化  
**关键词**: visual token pruning, optimal transport, Wasserstein distance, submodular optimization, training-free, MLLM efficiency

## 一句话总结

将视觉 token 裁剪建模为最优传输（OT）下的分布对齐问题，通过最小化完整与裁剪后 token 集合间的 2-Wasserstein 距离，以 Gaussian 代理 + log-det 子模目标 + 贪心 Cholesky 选择实现 training-free、$O(mk^2)$ 复杂度的高效裁剪，在 11 个多模态基准上取得 SOTA 精度-效率折中。

## 研究背景与动机

**MLLM 视觉 token 冗余严重**：单张图像可产生数百个 patch-level token，而 Transformer self-attention 的计算量与序列长度呈平方关系，导致推理代价极高。

**经验表明 70-90% 视觉 token 可裁剪**：已有研究证实大部分视觉 token 存在冗余，裁剪后对准确率影响很小，这为 token pruning 提供了充分动机。

**现有裁剪方法的局限**：基于 attention 的方法（FastV、VTW）在 attention 值不能准确反映 token 重要性时失效；基于 calibration 的方法（FitPrune）依赖外部数据集且泛化性差；fine-tuning 方法（M³）计算成本高。

**DivPrune 只关注多样性，忽视全局代表性**：DivPrune 通过选择彼此不相似的 token 来降低冗余，但多样性（diversity）不等于代表性（representativeness）——选出的子集可能无法保留原始 token 分布的协方差结构和语义覆盖。

**分布对齐假设的实证支持**：作者在 LLaVA 1.5-7B 上用多种子集选择策略验证了 OT 距离与下游任务性能的 Spearman 秩相关性很强，即 OT 距离越小、下游性能越好。

**需要一个 principled、training-free、task-agnostic 的裁剪框架**：既保留局部多样性又确保全局分布对齐，同时保持计算可行性和理论保证。

## 方法详解

### 整体框架

给定 vision encoder 输出的 $m$ 个 token $\bm{V} \in \mathbb{R}^{m \times d}$，OTPrune 选出 $k$ 个 token 的子集 $\mathcal{C}$，使裁剪后 token 分布 $Q$ 尽可能逼近完整 token 分布 $P$。核心流程：

1. 将 $P$, $Q$ 用 Gaussian 代理近似，得到 2-Wasserstein 距离的闭式解
2. 推导 log-det 子模代理目标函数
3. 通过贪心 Cholesky 分解高效求解

### 关键设计 1：OT 分布对齐建模

- **功能**：将 token pruning 建模为分布近似问题，最小化完整 token 集 $P$ 与裁剪子集 $Q$ 之间的 2-Wasserstein 距离
- **核心思路**：$\min_Q \mathcal{W}_2^2(P, Q) = \inf_{\pi \in \Pi(P,Q)} \mathbb{E}_{(x,y) \sim \pi} [\|x - y\|_2^2]$，直接在嵌入空间衡量两个分布的几何差异
- **设计动机**：与仅追求多样性的 DivPrune 不同，Wasserstein 距离同时编码了局部多样性和全局代表性——保留的子集不仅彼此不同，还能忠实近似原始特征流形的几何结构

### 关键设计 2：Gaussian 代理 + log-det 子模目标

- **功能**：用零均值 Gaussian 近似 token 分布 $P \approx \mathcal{N}(0, \Sigma)$, $Q \approx \mathcal{N}(0, \Sigma_\mathcal{C})$，将 Wasserstein 距离转化为协方差矩阵的 trace 目标，再用 $\gamma$-log-det 算子取下界
- **核心思路**：Gaussian 假设下 $\mathcal{W}_2^2$ 有闭式解 $\text{tr}(\Sigma) + \text{tr}(\Sigma_\mathcal{C}) - 2\text{tr}((\Sigma^{1/2} \Sigma_\mathcal{C} \Sigma^{1/2})^{1/2})$。通过对特征做单位方差归一化后，最大化 trace 目标。再利用 $\log\det(\bm{I} + \gamma \bm{X}) \leq \gamma \text{tr}(\bm{X})$ 的下界关系，得到代理目标 $\max_\mathcal{C} \log\det(\bm{I} + \gamma \Sigma^{1/2} \Sigma_\mathcal{C} \Sigma^{1/2})$
- **设计动机**：log-det 函数是可证明的单调子模函数（monotone submodular），保证贪心算法具有 $(1 - 1/e)$ 近似比，且避免了直接计算矩阵平方根的高开销

### 关键设计 3：Sylvester 变换 + Cholesky 贪心选择

- **功能**：利用 Sylvester 行列式恒等式将 $d \times d$ 矩阵运算转化为 $k \times k$ 矩阵运算，然后通过增量 Cholesky 分解贪心选取 token
- **核心思路**：预计算 $\tilde{\bm{V}} = \bm{V}\bm{V}^\top$，则目标变为 $\max_\mathcal{C} \log\det(\bm{I} + \tilde{\gamma} \tilde{\bm{V}}_\mathcal{C} \tilde{\bm{V}}_\mathcal{C}^\top)$。每次贪心迭代选择使目标增量最大的 token $j = \arg\max_{i \notin \mathcal{C}} d_i^2$，并通过 Cholesky 系数增量更新剩余 token 的增益
- **设计动机**：总复杂度 $O(mk^2)$，远低于暴力枚举的 $\binom{m}{k}$；子模性保证贪心解质量有理论下界；实验表明实际近似比远超 $(1 - 1/e) \approx 0.632$

### 关键设计 4：超参数 $\gamma$ 的鲁棒性

- **功能**：$\gamma$ 控制 OT 损失与 token 选择正则化之间的平衡
- **核心思路**：实验表明 $\gamma = 0.01$ 在 LLaVA 1.5-7B/13B 和 LLaVA 1.6-7B 上均表现良好，性能对 $\gamma$ 不敏感
- **设计动机**：无需大量调参即可部署，增强了方法的实用性

## 实验关键数据

### 表 1：11 个多模态基准上的主要对比（LLaVA 1.5-7B，保留 ~9.8% token）

| 方法 | TFLOP (ratio%) | GQA (EM) | MMB (Acc) | MME (P-score) | POPE (F1) | SQA (EM) | Avg. Rank ↓ | One-shot Rank ↓ |
|------|----------------|----------|-----------|---------------|-----------|----------|-------------|-----------------|
| Original | 3.228 (100%) | 61.96 | 64.09 | 1506 | 85.84 | 69.41 | – | – |
| VTW | 0.603 (18.5%) | 38.94 | 21.31 | 681 | 25.35 | 65.29 | 8.23 | 3.41 |
| FastV | 0.514 (15.7%) | 38.73 | 20.62 | 696 | 32.84 | 65.15 | 8.41 | 3.59 |
| DivPrune | 0.512 (15.6%) | 56.85 | 59.19 | 1328 | 86.02 | 68.27 | 3.23 | 1.64 |
| **OTPrune** | **0.512 (15.6%)** | **56.62** | **60.40** | **1368** | 79.59 | **68.42** | **2.36** | **1.36** |
| M³ (finetune) | 0.512 (15.6%) | 60.81 | 65.81 | 1391 | 86.33 | 64.65 | 2.68 | – |

**要点**：OTPrune 在无需训练/校准的 one-shot 方法中排名最优（Avg. Rank 2.36 vs DivPrune 3.23），甚至接近需要微调的 M³。

### 表 2：合成数据上 OTPrune vs DivPrune 的 Win Rate 和 Optimality Ratio

| 配置 (m, d, k) | 方法 | Win Rate (%) | Optimality Ratio (%) |
|----------------|------|-------------|---------------------|
| (20, 10, 5) | DivPrune | 60.14 | 74.53 |
| | **OTPrune** | **94.56** | **87.81** |
| (30, 20, 10) | DivPrune | 84.62 | 76.48 |
| | **OTPrune** | **99.49** | **93.98** |
| (100, 50, 30) | DivPrune | 88.77 | 92.20 |
| | **OTPrune** | **99.99** | **100.00** |
| (1000, 256, 100) | DivPrune | 97.59 | 97.71 |
| | **OTPrune** | **100.00** | **100.00** |

**要点**：OTPrune 在各维度配置下均大幅领先 DivPrune，高维场景下 Win Rate 和 Optimality Ratio 均达到 100%，远超理论下界 $(1-1/e) \approx 63.2\%$。

### 补充：LLaVA 1.5-13B 和 LLaVA 1.6-7B 结果

- **LLaVA 1.5-13B**：OTPrune Avg. Rank 1.27（DivPrune 2.14），One-shot Rank 1.18
- **LLaVA 1.6-7B**：OTPrune Avg. Rank 1.23（DivPrune 1.77），在仅约 11% 计算量下保持最佳性能
- **Wilcoxon 秩检验**：OTPrune vs DivPrune，p-value = 0.028，统计显著

## 亮点与洞察

1. **视角新颖**：首次将视觉 token 裁剪严格建模为 OT 分布对齐问题，从"选不同的 token"升级为"选能代表整体分布的 token"，理论动机清晰
2. **理论完备**：从 Wasserstein 距离 → Gaussian 代理 → log-det 子模目标 → Cholesky 贪心，每一步推导链条严谨，子模性和 $(1-1/e)$ 近似保证将裁剪从启发式提升为有原理保证的优化
3. **Training-free 且即插即用**：无需微调、无需校准数据集，直接在推理时使用，适用于 LLaVA 1.5-7B/13B 和 1.6-7B 等多种架构
4. **假设验证充分**：通过 Spearman 相关性分析明确证明了"OT 距离越小 → 下游性能越好"这一核心假设
5. **计算高效**：$O(mk^2)$ 复杂度，在保留约 10-15% token 情况下仅需原始 FLOP 的 ~15%

## 局限与展望

1. **Gaussian 近似的局限**：token 分布在高维空间中可能并不服从 Gaussian，尤其对于语义丰富的图像，多模态分布可能更合适（如 Gaussian Mixture）
2. **零均值假设较强**：实际 token embedding 通常不以零为中心，忽略均值匹配可能在某些场景下引入偏差
3. **仅评估了 LLaVA 系列**：未在其他 MLLM 架构（如 Qwen-VL、InternVL）上验证泛化性
4. **预计算 $\tilde{\bm{V}} = \bm{V}\bm{V}^\top$ 的开销**：当 $m$ 很大时（如 LLaVA 1.6 的 2000+ token），$m \times m$ 矩阵的存储和计算可能成为瓶颈
5. **未考虑文本-视觉交互**：裁剪完全基于视觉 token 内部的分布结构，未利用文本 query 信息进行条件裁剪，可能在特定任务上不如 attention-based 方法
6. **固定裁剪比例**：one-shot 设定下所有图像使用相同裁剪比例，未根据图像复杂度自适应调整

## 相关工作与启发

- **DivPrune**（CVPR 2025）: 基于 DPP 的多样性驱动裁剪，是 OTPrune 的直接前身和主要对比对象。OTPrune 的核心改进在于从"多样性"提升为"分布对齐"
- **FastV / VTW**: 基于 attention score 的裁剪方法，在高裁剪率下性能急剧下降，说明 attention 不是可靠的 token 重要性指标
- **FitPrune**: 基于校准集的方法，依赖外部数据且泛化性差
- **M³**: 需要微调的方法，精度高但计算成本大
- **Optimal Transport 在 ML 中的应用**: OT 已广泛用于生成模型（WGAN）、域适应等，本文将其引入 token 裁剪是一个自然且优雅的迁移
- **子模优化**: log-det 子模函数在传感器放置、特征选择等领域有成熟理论，本文借助这一工具实现了有理论保证的贪心选择

**启发**：分布对齐的思路可推广到其他 token/patch 选择场景（如 ViT 的 token 蒸馏、视频帧选择、点云下采样）；Wasserstein 距离作为裁剪质量指标比 attention score 更 principled。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将 token pruning 建模为 OT 问题，理论推导从 Wasserstein → Gaussian 代理 → log-det 子模目标的链条完整优雅
- **实验充分度**: ⭐⭐⭐⭐ — 11 个基准、3 个 LLaVA 变体、合成数据验证、消融实验、统计检验均覆盖，但缺少非 LLaVA 架构验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 从动机假设验证到理论推导到实验分析，逻辑链条清晰严谨，图表设计专业
- **价值**: ⭐⭐⭐⭐ — Training-free 即插即用、有理论保证的 token 裁剪方法，对 MLLM 高效推理有直接实用价值，OT 视角也为后续工作打开了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)
- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)
- [\[NeurIPS 2025\] FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling](../../NeurIPS2025/optimization/fedrts_federated_robust_pruning_via_combinatorial_thompson_sampling.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](../../ICML2026/optimization/distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)
- [\[CVPR 2025\] Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](../../CVPR2025/optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)

</div>

<!-- RELATED:END -->
