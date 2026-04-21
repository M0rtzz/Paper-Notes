---
title: >-
  [论文解读] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing
description: >-
  [ICML2025][randomized smoothing] 基于 randomized smoothing 框架提出"代价敏感认证半径"（cost-sensitive certified radius），首次实现可扩展到大模型与高维数据的代价敏感对抗鲁棒性认证与训练，在保持整体准确率的同时显著提升对高代价误分类的鲁棒性。
tags:
  - ICML2025
  - randomized smoothing
  - cost-sensitive robustness
  - certified defense
  - adversarial examples
  - cost matrix
---

# Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing

**会议**: ICML2025  
**arXiv**: [2310.08732](https://arxiv.org/abs/2310.08732)  
**代码**: [AppleXY/Cost-Sensitive-RS](https://github.com/AppleXY/Cost-Sensitive-RS)  
**领域**: 对抗鲁棒性 / 可认证防御  
**关键词**: randomized smoothing, cost-sensitive robustness, certified defense, adversarial examples, cost matrix

## 一句话总结
基于 randomized smoothing 框架提出"代价敏感认证半径"（cost-sensitive certified radius），首次实现可扩展到大模型与高维数据的代价敏感对抗鲁棒性认证与训练，在保持整体准确率的同时显著提升对高代价误分类的鲁棒性。

## 研究背景与动机

现有对抗防御方法（经验防御如对抗训练、可认证防御如 randomized smoothing）默认所有误分类代价相同，但在现实场景中不同误分类的代价往往差异巨大：
- **医学诊断**：将恶性肿瘤误判为良性远比反过来更危险
- **自动驾驶**：将行人误判为背景的后果远大于将背景误判为行人

此前唯一的代价敏感认证方法 (Zhang & Evans, 2019) 基于凸松弛，无法扩展到深度网络和大扰动场景。本文目标：在 randomized smoothing 这一可扩展框架下，提供代价敏感的鲁棒性认证与训练算法。

## 方法详解

### 1. 代价矩阵与代价敏感设定

定义代价矩阵 $\mathbf{C} \in \mathbb{R}_{\geq 0}^{m \times m}$，其中 $C_{jk} > 0$ 表示将类 $j$ 误分类为类 $k$ 会产生不可忽视的代价。对于种子类 $y$，其敏感目标类集合为 $\Omega_y = \{k \in [m] : C_{yk} > 0\}$。

### 2. 代价敏感认证半径（核心贡献）

在经典 randomized smoothing 的认证半径基础上，提出两种新的认证半径：

**Groupwise 代价敏感认证半径**：

$$r_{\text{cs-group}}(\mathbf{x}; \Omega_y) = \frac{\sigma}{2}\left[\Phi^{-1}\left(\max_{k \in [m]} [h_\theta(\mathbf{x})]_k\right) - \Phi^{-1}\left(\max_{k \in \Omega_y} [h_\theta(\mathbf{x})]_k\right)\right]$$

**Pairwise 代价敏感认证半径**：

$$r_{\text{cs-pair}}(\mathbf{x}; j) = \frac{\sigma}{2}\left[\Phi^{-1}\left(\max_{k \in [m]} [h_\theta(\mathbf{x})]_k\right) - \Phi^{-1}\left([h_\theta(\mathbf{x})]_j\right)\right]$$

其中 $h_\theta(\mathbf{x})$ 是 smoothed classifier 对各类的预测概率，$\Phi^{-1}$ 是标准正态分布的逆 CDF。

**关键定理 (Theorem 4.2)**：在合理假设下，$r_{\text{cs-pair}} \geq r_{\text{cs-group}} \geq r_{\text{standard}}$，即代价敏感认证半径严格不小于标准认证半径，当 $|\Omega_y|$ 越小优势越明显。

### 3. 基于蒙特卡洛的认证算法

提出 Algorithm 1 (Certify_Group) 和 Algorithm 2 (Certify_Pair)：
- 通过高斯采样估计各类预测概率
- 对 $p_A$（主类概率）计算 $(1-\alpha/2)$ 下置信界
- 对 $p_B$（敏感目标类概率）通过 union bound 计算 $(1-\alpha/(2|\Omega_y|))$ 上置信界
- 最终返回 $\max(\hat{r}_{\text{std}}, \hat{r}_{\text{cs}})$ 以确保最紧证书

### 4. Margin-CS 训练方法

设计了直接优化代价敏感认证半径的训练目标：

$$\min_\theta \left\{ \mathbb{E}_{(\mathbf{x},y)} \mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}+\boldsymbol{\delta}), y) + \lambda_1 \mathbb{E}_{\mathcal{D}_n} \mathcal{L}_M(r_{\text{cs-group}}; 0, \gamma_1) + \lambda_2 \mathbb{E}_{\mathcal{D}_s} \sum_{j \in \Omega_y} C_{yj} \mathcal{L}_M(r_{\text{cs-pair}}; 0, \gamma_2) \right\}$$

- 第一项：标准交叉熵 + 高斯噪声增强，保持整体准确率
- 第二项：对非敏感样本优化 groupwise 认证半径
- 第三项：对敏感样本按代价加权优化 pairwise 认证半径
- 使用 hinge loss $\mathcal{L}_M$ 替代不可微的认证半径，确保数值稳定性

## 实验关键数据

### CIFAR-10 (ResNet-56, $\epsilon=0.5$, $\sigma=0.5$)

| 方法 | Acc(%) ↑ | Rob_cs(%) ↑ | Rob_cost ↓ |
|------|----------|-------------|------------|
| Gaussian | 65.4 | 22.3 | 4.99 |
| SmoothAdv | 66.9 | 27.1 | 4.94 |
| MACER | 65.9 | 27.3 | 5.27 |
| SmoothAdv-CS | 66.1 | 53.5 | 3.12 |
| **Margin-CS** | **67.5** | **54.8** | **3.04** |

S-Pair 设定下 Margin-CS 的 Rob_cs 达到 **92.4%**，远超 Gaussian 的 50.4%。

### Imagenette (S-Seed 设定)

| 方法 | Acc(%) | Rob_cs(%) | Rob_cost |
|------|--------|-----------|----------|
| Gaussian | 80.3 | 64.6 | 3.67 |
| SmoothAdv-CS | 76.1 | 68.9 | 2.24 |
| **Margin-CS** | **79.6** | **81.1** | **1.35** |

### HAM10k 医学数据集（恶性/良性分类，代价比 10:1）

| 方法 | Acc(%) | Rob_cs(%) | Rob_cost | Precision(%) | Recall(%) |
|------|--------|-----------|----------|--------------|-----------|
| Gaussian | 82.9 | 11.8 | 1.56 | 51.0 | 15.0 |
| MACER | 82.7 | 21.1 | 1.41 | 50.0 | 25.0 |
| **Margin-CS** | **83.2** | **34.4** | **1.17** | **52.0** | **41.3** |

### 与 Zhang & Evans (2019) 对比 (CIFAR-10, $\sigma=0.25$)

| 方法 | Acc(%) | Rob_cs(%) |
|------|--------|-----------|
| Zhang & Evans | 61.2 | 92.4 |
| **Margin-CS** | **80.9** | **93.5** |

Margin-CS 在准确率上高出近 20 个百分点，鲁棒性也更优。

## 亮点与洞察

1. **首次可扩展的代价敏感认证**：之前唯一方法 (Zhang & Evans 2019) 无法处理大数据集和深度网络，本文基于 randomized smoothing 克服了这一限制
2. **认证半径严格提升**：理论证明代价敏感半径 ≥ 标准半径，且 $|\Omega_y|$ 越小提升越显著
3. **精巧的置信界构造**：通过 union bound 对敏感类分别估计上置信界，而非简单使用 $1-p_A$ 作为 $p_B$ 的上界，获得更紧的证书
4. **实际医学场景验证**：HAM10k 上 Recall 从基线 15% 提升至 41.3%，对恶性肿瘤漏诊率大幅降低
5. **灵活的训练框架**：Margin-CS 通过分组阈值精细控制不同数据子群的优化，实现准确率与鲁棒性的更优权衡

## 局限与展望

1. **仅限 $\ell_2$ 范数**：当前认证仅支持 $\ell_2$ 扰动，$\ell_\infty$ 等其他范数下的扩展尚待研究
2. **代价矩阵需先验**：实际应用中代价矩阵需要领域专家指定，如何自动学习代价矩阵是开放问题
3. **认证半径仍然保守**：蒙特卡洛采样引入的统计误差使得实际认证半径弱于理论值
4. **大扰动下失效**：$\epsilon > 1.5$ 时所有方法的认证率都急剧下降
5. **计算开销**：推理时需多次高斯采样（$n=10^5$ 量级），不适合实时部署

## 相关工作与启发

- **Randomized Smoothing** (Cohen et al., 2019)：本文的认证框架基础
- **SmoothAdv** (Salman et al., 2019)：对抗训练 + smoothing
- **MACER** (Zhai et al., 2020)：直接优化认证半径
- **Zhang & Evans (2019)**：唯一的代价敏感认证方法（凸松弛，不可扩展）
- 启发：代价敏感思想可扩展到其他认证框架（如 IBP、CROWN），以及多目标对抗鲁棒性

## 评分
- 新颖性: ⭐⭐⭐⭐ (代价敏感 + randomized smoothing 的结合新颖，理论扎实)
- 实验充分度: ⭐⭐⭐⭐ (CIFAR-10/Imagenette/ImageNet/HAM10k 四数据集，多 baseline 对比，消融实验完整)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，理论推导严谨，图示直观)
- 价值: ⭐⭐⭐⭐ (填补了可扩展代价敏感认证的空白，对医学等高风险场景有实际意义)

<!-- RELATED:START -->

## 相关论文

- [Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning](../../NeurIPS2025/llm_evaluation/cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)
- [A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](../../NeurIPS2025/llm_evaluation/a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)
- [Deep Cost Ray Fusion for Sparse Depth Video Completion](../../ECCV2024/llm_evaluation/deep_cost_ray_fusion_for_sparse_depth_video_completion.md)
- [CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/llm_evaluation/counselbench_llm_mental_health_qa.md)
- [Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation](faster_and_stronger_when_ann-snn_conversion_meets_parallel_spiking_calculation.md)

<!-- RELATED:END -->
