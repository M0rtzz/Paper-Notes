---
title: >-
  [论文解读] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling
description: >-
  [医学图像] HSO 提出了一种层次化调度优化器，通过双层优化框架（上层全局搜索最优初始化策略 + 下层局部优化调度精炼），在仅 8 秒一次性优化代价下实现扩散模型极低 NFE 下的 SOTA 免训练采样质量。
tags:
  - 医学图像
---

# Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling

## 论文信息

- **会议**: AAAI 2026
- **arXiv**: [2511.11688](https://arxiv.org/abs/2511.11688)
- **代码**: [https://github.com/chappy0/HSO.git](https://github.com/chappy0/HSO.git)
- **领域**: 扩散模型加速 / 采样调度优化
- **关键词**: 扩散模型, 采样加速, 调度优化, 双层优化, 免训练, FID, 低NFE

## 一句话总结

HSO 提出了一种层次化调度优化器，通过双层优化框架（上层全局搜索最优初始化策略 + 下层局部优化调度精炼），在仅 8 秒一次性优化代价下实现扩散模型极低 NFE 下的 SOTA 免训练采样质量。

## 研究背景与动机

扩散概率模型在图像生成质量上设立新标杆，但迭代采样过程的高 NFE（网络函数评估次数）严重阻碍实时应用。调度优化是一种免训练加速策略，通过在固定小 NFE 下找到最优时间步分布来最大化采样质量。

理想的调度优化方法应同时满足四个核心原则：

| 原则 | 规则法 | 感知优化 | 理论优化 | **HSO** |
|------|--------|---------|---------|---------|
| 适应性 | ✗ | ✓ | ✓ | ✓ |
| 有效性 | ✗ | ✓ | ✗ | ✓ |
| 实用鲁棒性 | ✓ | ✗ | ✗ | ✓ |
| 计算效率 | ✓ | ✗ | ✓ | ✓ |

现有范式的问题：
- **规则法**（如 EDM schedule）：固定公式无法适配不同模型和 NFE
- **感知优化**（如 AutoDiffusion）：需反复生成图像，计算代价高（~1.1天）
- **理论优化**（如 DM-NonUni）：非凸优化景观易陷入局部最优，忽略鲁棒性

## 方法详解

### 整体框架：双层优化

HSO 将搜索全局最优调度的问题分解为两个可处理的子问题：

**下层：局部优化（调度精炼）**
$$\Lambda_{\text{opt}}(\boldsymbol{\psi}) = \arg\min_\Lambda \mathcal{J}_{\text{lower}}(\Lambda | \Lambda_{\text{init}}(\boldsymbol{\psi}))$$

- 从上层提供的初始点 $\Lambda_{\text{init}}$ 出发
- 使用信赖域约束优化算法
- 由 MEP 目标函数引导

**上层：全局搜索（初始化策略）**
$$\boldsymbol{\psi}^* = \arg\min_\boldsymbol{\psi} \mathcal{F}_{\text{upper}}(\Lambda_{\text{opt}}(\boldsymbol{\psi}))$$

- 在低维超参数空间 $\mathbb{R}^3$ 中搜索（$\boldsymbol{\psi} = (\rho, \tilde{\sigma}_{\min}, \tilde{\sigma}_{\max})$）
- 使用差分进化等种群演化算法
- 由 SPF 适应度函数评估

**迭代交替过程**：上层提案候选策略群 → 每个候选下层执行局部优化 → 评估反馈上层 → 进化下一代。

### 关键设计

#### 1. 中点误差代理 (MEP - Midpoint Error Proxy)

核心理论贡献。基于概率流 ODE 的全局生成误差：

$$x_\epsilon = \frac{\sigma_\epsilon}{\sigma_T}x_T + \sigma_\epsilon \sum_{i=0}^{N-1}\int_{\lambda_i}^{\lambda_{i+1}}e^\lambda f(\lambda)d\lambda$$

**混合中点近似**（Lemma 1）：将可精确积分的指数项 $e^\lambda$ 与神经网络项 $f(\lambda)$ 分离，仅对 $f(\lambda)$ 做中点近似：

$$\int_{\lambda_i}^{\lambda_{i+1}}e^\lambda f(\lambda)d\lambda \approx f(\lambda_{i+\frac{1}{2}})(e^{\lambda_{i+1}} - e^{\lambda_i})$$

局部截断误差为 $O(h^3)$，与标准中点法同阶，但数值更稳定。

**全局误差界**（Theorem 1）：

$$\|{\tilde{x}_{\epsilon,MEP} - x_0}\| \leq C + \sigma_\epsilon \tilde{\eta} \sum_{i=0}^{N-1} \tilde{\epsilon}_{t(\lambda_{i+\frac{1}{2}})}(e^{\lambda_{i+1}} - e^{\lambda_i})$$

最小化调度依赖的求和项即得 MEP 目标：

$$\mathcal{J}_{\text{MEP}}(\Lambda) = \sum_{i=0}^{N-1} \tilde{\epsilon}_{t(\lambda_{i+\frac{1}{2}})}(e^{\lambda_{i+1}} - e^{\lambda_i})$$

**MEP 的优势**：
- 求解器无关（不绑定 UniPC 等特定求解器）
- 线性时间 $O(N)$ 计算
- 通过精确积分指数项避免数值不稳定

#### 2. 间距惩罚适应度 (SPF - Spacing-Penalized Fitness)

解决无约束优化产生病态接近时间步的问题：

$$\mathcal{F}_{\text{SPF}}(\psi|N) = \mathcal{J}_{\text{MEP}}(\Lambda_{\text{opt}}) + \gamma L_{\text{penalty}}(\Lambda_{\text{opt}}|N)$$

惩罚项：
$$L_{\text{penalty}} = \sum_{i=0}^{N-1}\max(0, d_{\min}(N) - |t(\lambda_{i+1}) - t(\lambda_i)|)^2$$

$d_{\min}(N)$ 自适应 NFE 预算：NFE=4 时为 0.15（强制大间距），NFE=20 时为 0.01（允许精细步进）。

### 损失函数

HSO 不涉及训练损失，而是优化目标函数。核心为 MEP 目标（下层）和 SPF 适应度（上层）的协同。

## 实验

### 实验设置

- 模型：Stable Diffusion v2.1-base
- 求解器：UniPC, DDIM
- 数据集：LAION-Aesthetics 6.5+（~30K对）、MS-COCO 2017 val（30K）、ImageNet 512x512 val（50K）
- 指标：FID（Fréchet Inception Distance）
- 搜索空间：$\rho \in [3,16]$, $t_\epsilon \in [0.01,0.03]$, $t_{\max} \in [0.96,1.0]$

### 主实验表格

**FID 对比（HSO vs. DM-NonUni）**：

| 数据集 | 求解器 | NFE | DM-NonUni FID | **HSO FID** |
|--------|--------|-----|-------------|-----------|
| LAION | UniPC | 4 | 18.96 | **15.71** |
| LAION | UniPC | 5 | 13.91 | **11.94** |
| LAION | DDIM | 4 | 68.92 | **24.77** |
| LAION | DDIM | 5 | 35.38 | **17.17** |
| MS-COCO | UniPC | 4 | 27.50 | **23.26** |
| MS-COCO | DDIM | 5 | 30.12 | **23.15** |
| ImageNet 512 | UniPC | 4 | 20.75 | **17.20** |
| ImageNet 512 | DDIM | 4 | 41.51 | **19.78** |

在 DDIM 上提升尤为显著（NFE=4 时 LAION 从 68.92→24.77），验证了 MEP 求解器无关性的优势。

### 适应性验证

**NFE 适应性**：不同 NFE 的最优参数呈复杂非单调关系（$\rho^*$ 在 NFE=8 时峰值 12.42），证明固定规则法不可行。

**模型适应性**：在 PixArt-α 上 FID 从 37.65 降至 18.05，验证跨模型泛化。

### 鲁棒性验证

| 条件 | 调度示例 | 最小间距 | FID |
|------|---------|---------|-----|
| 无惩罚 | [999, 70, 9, 9] | 0.0 | 165.48（崩溃） |
| 有 SPF | [959, 716, 370, 30] | 243.0 | 19.76±0.25 |

无 SPF 时出现"尾部聚集"（时间步坍缩到 [9,9]），FID 灾难性退化。

### 消融实验

| 配置 | UniPC FID | DDIM FID |
|------|----------|---------|
| 基线 (DM-NonUni) | 18.07 | 71.57 |
| +双层搜索 | 11.44 | 29.22 |
| +MEP (无双层) | 26.33 | 37.27 |
| **完整 HSO** | **15.70** | **24.80** |

### 计算效率

| 方法 | 范式 | 准备代价 | NFE | FID |
|------|------|---------|-----|-----|
| **HSO** | 层次优化 | **~8秒** | 4 | 15.71 |
| AutoDiffusion | 进化搜索 | ~1.1天 | 4 | 17.86 |
| DM-NonUni | 局部优化 | ~1秒 | 4 | 18.96 |
| LCM | 蒸馏 | ~1.33天 | 4 | 11.10 |
| Mean Flows | 流匹配 | ~60天 | 1 | 3.43 |

HSO 在免训练方法中以 8 秒代价达到最优性能，比训练方法少数个数量级的准备时间。

### 关键发现

- 双层框架的协同效果：上层找到更好的初始化使下层避免局部最优，二者缺一不可
- MEP 的"可预见权衡"：在 UniPC 特定任务上略逊于绑定求解器的目标，但在 DDIM 上大幅优于，体现了通用性的价值
- SPF 是实用部署的必要保障：理论最优≠实际最优，时间步聚集会导致灾难

## 亮点与洞察

1. **问题分解思路优雅**：将 $N$ 维非凸优化分解为 3 维全局搜索 + $N$ 维局部精炼，突破维数灾难
2. **MEP 的理论贡献扎实**：通过精确积分指数项 + 仅近似网络项，在保持相同阶误差的同时提升数值稳定性
3. **SPF 填补了理论与实践的鸿沟**：直接指出理论最优调度在实践中可能崩溃的问题，并给出自适应解决方案
4. **8 秒 vs. 1.1 天**：与 AutoDiffusion 相比，HSO 以 4 个数量级的搜索加速达到更好性能，极具实际部署价值

## 局限性

- 仅在 Stable Diffusion v2.1 和 PixArt-α 上验证，更新的模型（如 SD3、FLUX）未测试
- NFE≥10 的场景改善幅度可能较小（随着 NFE 增加边际收益递减）
- 搜索范围$\boldsymbol{\psi}$的边界仍需人工设定，存在一定启发式成分
- 论文分类为 medical_imaging 存疑——该论文是通用的扩散模型加速方法，与医学影像无直接关联
- $d_{\min}(N)$ 的线性启发式虽有效但缺乏理论依据

## 相关工作

- **训练式加速**：知识蒸馏 (LCM), 一致性模型 (Consistency Models), Mean Flows
- **免训练加速**：求解器设计 (DPM-Solver, UniPC, DDIM), 模型级优化 (量化/缓存)
- **调度优化**：规则法 (EDM schedule), 感知优化 (AutoDiffusion, OMS-DPM), 理论优化 (DM-NonUni)

## 评分

⭐⭐⭐⭐⭐ (5/5)

- 理论贡献扎实（MEP 的推导和证明完整），工程价值极高（8秒优化成本）
- 四个设计原则的系统分析框架清晰
- 实验覆盖三个大规模基准、两种求解器、适应性/鲁棒性/效率全方位验证
- 双层优化的构思简洁优雅，SPF 解决了一个被忽视但重要的实际问题

<!-- RELATED:START -->

## 相关论文

- [Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](../../NeurIPS2025/medical_imaging/raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)
- [Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](../../NeurIPS2025/medical_imaging/scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/medical_imaging/multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)

<!-- RELATED:END -->
