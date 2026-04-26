---
title: >-
  [论文解读] Discrete Diffusion Trajectory Alignment via Stepwise Decomposition
description: >-
  [ICLR2026][医学图像][扩散模型] 提出 SDPO（Stepwise Decomposition Preference Optimization），将离散扩散模型的轨迹对齐问题分解为逐步后验对齐子问题，避免了在整条去噪链上反传梯度的困难，在 DNA 序列设计、蛋白质逆折叠和语言建模三个任务上均显著超越现有方法。
tags:
  - ICLR2026
  - 医学图像
  - 扩散模型
  - preference optimization
  - RLHF
  - trajectory alignment
  - stepwise decomposition
---

# Discrete Diffusion Trajectory Alignment via Stepwise Decomposition

**会议**: ICLR2026  
**arXiv**: [2507.04832](https://arxiv.org/abs/2507.04832)  
**代码**: [hanjq17/discrete-diffusion-sdpo](https://github.com/hanjq17/discrete-diffusion-sdpo)  
**领域**: medical_imaging  
**关键词**: discrete diffusion, preference optimization, RLHF, trajectory alignment, stepwise decomposition  
**作者**: Jiaqi Han, Austin Wang, Minkai Xu, Wenda Chu, Meihua Dang, Haotian Ye, Huayu Chen, Yisong Yue, Stefano Ermon (Stanford, Caltech, Tsinghua)  

## 一句话总结

提出 SDPO（Stepwise Decomposition Preference Optimization），将离散扩散模型的轨迹对齐问题分解为逐步后验对齐子问题，避免了在整条去噪链上反传梯度的困难，在 DNA 序列设计、蛋白质逆折叠和语言建模三个任务上均显著超越现有方法。

## 背景与动机

离散扩散模型（discrete diffusion models）在序列数据建模上展现了巨大潜力，覆盖 DNA 序列设计、蛋白质逆折叠乃至文本生成等多个领域。然而，如何像大语言模型的 RLHF 一样，对预训练的离散扩散模型进行奖励对齐（alignment），仍是一个未解决的核心问题。

现有方法面临的主要困难：

1. **离散性导致梯度反传困难**：离散扩散模型的采样链由序列级离散随机变量组成，奖励通常仅定义在最终干净序列 $\mathbf{x}_0$ 上，将梯度反传到整条去噪链计算代价极高且不稳定
2. **似然计算不精确**：对齐整条轨迹的联合分布 $p_\theta(\mathbf{x}_{0:T})$ 时，精确计算似然和评估奖励都不可行，导致次优性能
3. **在线 RL 采样开销大**：离散扩散的链式采样使得在线 RL 方法（如 DRAKES、diffu-GRPO）每一步训练都需要昂贵的在线采样

## 核心问题

如何设计一种高效的离线偏好优化方法，使其既能精确计算似然和奖励、又能兼容任意奖励函数，从而有效对齐预训练的离散扩散模型？

## 方法详解

### 逐步分解（Stepwise Decomposition）

核心思想：将整条扩散轨迹 $p_\theta(\mathbf{x}_{0:T})$ 的对齐问题分解为 $T$ 个子问题，每个子问题独立对齐第 $t$ 步的因式化后验近似 $\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t)$。

对于每个扩散步 $t$，子问题为：

$$\max_{\hat{p}_\theta} \mathbb{E}_{\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t,\mathbf{c})}[r(\mathbf{x}_0,\mathbf{c})] - \beta_t D_{\mathrm{KL}}[\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t,\mathbf{c}) \| \hat{p}_{\mathrm{ref}}(\mathbf{x}_0|\mathbf{x}_t,\mathbf{c})]$$

这样做的好处：

- 后验 $\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t)$ 可以高效且精确地计算似然
- 奖励直接定义在干净序列 $\mathbf{x}_0$ 上，无需对中间变量的有偏估计
- 支持任意奖励函数，不局限于 Bradley-Terry 模型

### 理论等价性（Theorem 4.1）

关键定理：每步子问题的最优解 $\{\hat{p}^*(\mathbf{x}_0|\mathbf{x}_t)\}_{t=1}^T$ 所诱导的联合分布 $p^*(\mathbf{x}_{0:T})$ 同时也是原始轨迹对齐目标的最优解，对应的链奖励为逐步奖励的加和形式 $\hat{r}(\mathbf{x}_{0:T})=\beta\sum_{t=1}^T r_t(\mathbf{x}_{t-1};\mathbf{x}_t)$。

### 广义逐步对齐（Distribution Matching）

通过分布匹配的方式优化每步后验，最终损失函数为交叉熵形式：

$$\mathcal{L}(\theta) = -\mathbb{E}_{t,\mathbf{c},\mathbf{x}_0,q(\mathbf{x}_t|\mathbf{x}_0)} \sum_{i=1}^N \left( \frac{\exp(r(\mathbf{x}_0^{(i)},\mathbf{c}))}{\sum_j \exp(r(\mathbf{x}_0^{(j)},\mathbf{c}))} \cdot \log \frac{\exp(\tilde{r}_\theta(\mathbf{x}_0^{(i)},\mathbf{x}_t^{(i)},\mathbf{c},\beta_t))}{\sum_j \exp(\tilde{r}_\theta(\mathbf{x}_0^{(j)},\mathbf{x}_t^{(j)},\mathbf{c},\beta_t))}\right)$$

其中隐式奖励 $\tilde{r}_\theta$ 基于模型与参考模型的 log-likelihood 之差。$w(t)=1-\alpha_t$ 用于均摊损失到每个 token。

### 迭代标注（Iterative Labeling）

训练过程中可迭代生成新样本并用奖励模型标注，逐步提升训练数据质量，进一步增强性能。

## 实验关键数据

### DNA 序列设计

| 方法 | Pred-Activity ↑ | ATAC-Acc ↑ | 3-mer Corr ↑ | JASPAR Corr ↑ |
|------|:---:|:---:|:---:|:---:|
| DRAKES | 5.61 | 92.5% | 0.887 | 0.911 |
| diffu-GRPO | 5.86 | 33.0% | 0.783 | 0.903 |
| **SDPO** | **6.30** | **94.8%** | **0.900** | **0.936** |

- 相比最强 RL 基线 DRAKES，预测活性提升 **12.3%**
- 同时保持高 ATAC 准确率和自然序列特征

### 蛋白质逆折叠

| 方法 | Pred-ddG ↑ | %(ddG>0) ↑ | scRMSD ↓ | Success Rate ↑ |
|------|:---:|:---:|:---:|:---:|
| DRAKES | 1.095 | 86.4% | 0.918 | 78.6% |
| diffu-GRPO | 1.286 | 76.8% | 1.192 | 37.2% |
| **SDPO** | **1.400** | **87.1%** | 0.938 | 75.5% |

- Pred-ddG 大幅领先所有基线；diffu-GRPO 虽 ddG 高但 scRMSD 严重退化（奖励过优化）

### 语言建模（LLaDA-8B-Instruct）

| 方法 | AlpacaEval LC ↑ | AlpacaEval WR ↑ | GSM8K ↑ | IFEval ↑ |
|------|:---:|:---:|:---:|:---:|
| Instruct | 10.6% | 6.8% | 78.6 | 52.9 |
| D2-DPO | 12.1% | 7.5% | 78.1 | 53.8 |
| **SDPO** | **14.2%** | **8.7%** | **81.2** | **55.1** |

- GSM8K 从 78.6 提升至 81.2，超过 LLaMA-3-8B 的 RL 后训练结果

### 训练效率

每步训练平均耗时：DRAKES 6.02s，diffu-GRPO 1.51s，**SDPO 仅 0.77s**（无需在线采样）。

## 亮点

1. **理论优雅**：将轨迹对齐分解为逐步后验对齐，并严格证明二者最优解等价，理论扎实
2. **通用性强**：兼容任意奖励函数，不局限于 Bradley-Terry 等简化奖励模型；当 $N=2$ 且使用 BT 奖励时退化为 DPO 的特殊情况
3. **高效离线**：无需在线策略采样，训练速度比 DRAKES 快约 8 倍
4. **跨领域验证**：在 DNA 设计、蛋白质工程、语言建模三个差异巨大的领域均一致优于基线
5. **迭代标注**：仅需少量额外标注（相比 DRAKES 的 128k，SDPO 用 15k 就达到更高奖励）即可持续提升

## 局限与展望

1. **蛋白质任务中 Success Rate 略低于 DRAKES**（75.5% vs 78.6%），说明在 scRMSD 维度上仍有优化空间
2. **因式化后验近似**（$\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t)=\prod_i \hat{p}_\theta(\mathbf{x}_0^{(i)}|\mathbf{x}_t)$）忽略了 token 间的依赖关系，可能在长序列上引入误差
3. **语言模型实验规模有限**：仅在 LLaDA-8B 上验证，未扩展到更大模型或更多 benchmark
4. **Monte-Carlo 估计的偏差**：$N$ 有限时分区函数的估计仍有偏，表中显示 $N$ 从 25 到 100 性能趋于饱和但未下降
5. **迭代标注需要可调用的奖励模型**，限制了在纯偏好数据场景下的应用

## 与相关工作的对比

| 维度 | DPO / D2-DPO | DRAKES | diffu-GRPO | SDPO |
|------|:---:|:---:|:---:|:---:|
| 在线/离线 | 离线 | 在线 | 在线 | 离线 |
| 奖励类型 | Bradley-Terry | 任意 | 任意 | 任意 |
| 优化粒度 | 轨迹级 | 轨迹级 | 轨迹级 | **逐步** |
| 训练效率 | 高 | 低 | 中 | **高** |
| 似然精确性 | 近似 | 近似 | 近似 | **精确** |
| 奖励过优化风险 | 低 | 中 | **高** | 低 |

- 与 D2-DPO/VRPO 相比：SDPO 不依赖 BT 模型假设，支持任意奖励且性能更强
- 与 DRAKES 相比：SDPO 离线训练，效率高 ~8x，DNA 任务上预测活性高 12.3%
- 与 diffu-GRPO 相比：SDPO 无奖励过优化问题（diffu-GRPO 蛋白质任务 Success Rate 仅 37.2%）
- 与推理时 guidance（CG/SMC/TDS）相比：SDPO 是训练时优化，采样时无额外开销

## 启发与关联

1. **逐步分解思想的迁移**：该分解策略具有普适性，可能迁移到连续扩散模型的对齐、甚至其他多步决策优化场景
2. **与 DPO 的统一视角**：SDPO 在 $N=2$ + BT 奖励下退化为 DPO，提供了 DPO 在扩散模型上的更一般化框架
3. **面向生物序列设计**：DNA/蛋白质任务上的显著提升表明该方法特别适合序列级奖励明确的生物工程应用
4. **大语言扩散模型的偏好优化**：在 LLaDA 上的验证为离散扩散语言模型的后训练提供了可行路径

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (逐步分解 + 等价性证明是全新视角)
- 实验充分度: ⭐⭐⭐⭐⭐ (三个差异较大领域 + 丰富消融)
- 写作质量: ⭐⭐⭐⭐⭐ (理论推导清晰，实验组织有条理)
- 价值: ⭐⭐⭐⭐⭐ (为离散扩散模型对齐建立了新基准)

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](ultra-fast_language_generation_via_discrete_diffusion_divergence_instruct.md)
- [\[ICLR 2026\] Causal Interpretation of Neural Network Computations with Contribution Decomposition](causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)
- [\[AAAI 2026\] TAlignDiff: Automatic Tooth Alignment assisted by Diffusion-based Transformation Learning](../../AAAI2026/medical_imaging/taligndiff_automatic_tooth_alignment_assisted_by_diffusion-based_transformation_.md)
- [\[ICLR 2026\] Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)
- [\[ICML 2025\] GenMol: A Drug Discovery Generalist with Discrete Diffusion](../../ICML2025/medical_imaging/genmol_a_drug_discovery_generalist_with_discrete_diffusion.md)

<!-- RELATED:END -->
