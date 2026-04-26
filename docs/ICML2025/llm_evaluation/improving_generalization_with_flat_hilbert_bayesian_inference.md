---
title: >-
  [论文解读] Improving Generalization with Flat Hilbert Bayesian Inference
description: >-
  [ICML2025][Bayesian Inference] 提出 Flat Hilbert Bayesian Inference (FHBI)，将 SAM 的平坦性概念从有限维欧氏空间推广到无限维再生核希尔伯特空间 (RKHS)，并与粒子采样贝叶斯推断结合，在 VTAB-1K 基准上以 73.7% 平均 Top-1 准确率全面超越九个基线方法。
tags:
  - ICML2025
  - Bayesian Inference
  - Sharpness-Aware Minimization
  - RKHS
  - SVGD
  - LoRA
  - 泛化性
---

# Improving Generalization with Flat Hilbert Bayesian Inference

**会议**: ICML2025  
**arXiv**: [2410.04196](https://arxiv.org/abs/2410.04196)  
**代码**: 待确认  
**领域**: 贝叶斯优化  
**关键词**: Bayesian Inference, Sharpness-Aware Minimization, RKHS, SVGD, LoRA, 泛化性

## 一句话总结

提出 Flat Hilbert Bayesian Inference (FHBI)，将 SAM 的平坦性概念从有限维欧氏空间推广到无限维再生核希尔伯特空间 (RKHS)，并与粒子采样贝叶斯推断结合，在 VTAB-1K 基准上以 73.7% 平均 Top-1 准确率全面超越九个基线方法。

## 研究背景与动机

贝叶斯推断通过后验分布建模不确定性，但现有方法（如 SVGD）仅逼近经验后验 $p(\theta|\mathcal{S})$，容易过拟合训练集。另一方面，SAM 通过最小化损失函数的锐度来寻找平坦极小值以提升泛化性，但仅限于单模型、有限维欧氏空间。

**核心动机**：能否将 SAM 的平坦极小化思想与粒子贝叶斯方法结合，在函数空间（RKHS）中逼近 **总体后验** $p(\theta|\mathcal{D})$ 而非经验后验，从而同时获得：

1. 贝叶斯方法的不确定性量化能力
2. 平坦极小化带来的泛化提升
3. 粒子间的交互多样性

## 方法详解

### 总体后验 vs 经验后验

定义经验后验和总体后验：

$$p(\theta|\mathcal{S}) \propto \exp(-\mathcal{L}_\mathcal{S}(\theta))p(\theta), \quad p(\theta|\mathcal{D}) \propto \exp(-\mathcal{L}_\mathcal{D}(\theta))p(\theta)$$

Proposition 4.1 证明总体后验是下述优化问题的解：

$$\mathbb{Q}^* = \min_{\mathbb{Q} \ll \mathbb{P}_\theta} \left\{ \mathbb{E}_{\theta \sim \mathbb{Q}}[\mathcal{L}_\mathcal{D}(\theta)] + D_{\text{KL}}(\mathbb{Q} \| \mathbb{P}_\theta) \right\}$$

即从 $\mathbb{Q}^*$ 中采样的粒子集成能最优地最小化总体损失，避免过拟合。

### 函数空间泛化界 (Theorem 4.2)

将已有的欧氏空间泛化界推广到 RKHS $\mathcal{H}^d$：

$$\tilde{L}_\mathcal{D}(f) \leq \max_{\|f'-f\|_{\mathcal{H}^d} \leq \rho} \tilde{L}_\mathcal{S}(f') + \mathcal{O}\left(\sqrt{\frac{\log(1+1/\rho^2)+\log(n/\delta)}{n-1}}\right)$$

关键贡献在于处理了 RKHS 的**无限维**特性（如 RBF 核），不能直接套用依赖维度的已有结果。

### 贝叶斯推断泛化界 (Theorem 4.3)

将函数空间锐度连接到贝叶斯推断——总体后验的 KL 散度可被经验后验的最坏情况 KL 散度上界控制：

$$D_{\text{KL}}(q_{[I+f]} \| p(\theta|\mathcal{D})) \leq \max_{\|f'-f\|_{\mathcal{H}^d} \leq \rho} D_{\text{KL}}(q_{[I+f']} \| p(\theta|\mathcal{S})) + \mathcal{O}(\cdot)$$

### FHBI 算法

基于上述理论，FHBI 采用**两步迭代**更新：

**Step 1 - 对抗扰动（上升步）**：在 RKHS 中沿函数梯度方向找到最坏扰动

$$\hat{f}_k^* = \rho \frac{\nabla_f D_{\text{KL}}(q_{[I+f]} \| p(\cdot|\mathcal{S}))|_{f=f_k}}{\|\nabla_f D_{\text{KL}}(q_{[I+f]} \| p(\cdot|\mathcal{S}))|_{f=f_k}\|_{\mathcal{H}^d}}$$

**Step 2 - 函数下降步**：在扰动位置计算梯度并更新

$$f_{k+1} = f_k - \epsilon \nabla_f D_{\text{KL}}(q_{[I+f]} \| p(\cdot|\mathcal{S}))|_{f=f_k+\hat{f}_k^*}$$

实际实现中，维护 $m$ 个粒子 $\{\theta_i\}_{i=1}^m$，每个粒子的更新涉及所有粒子的信息交互：

- **锐度最小化**：每个粒子寻找平坦区域（类似 SAM）
- **角度排斥力**：促进粒子梯度方向多样化（$\nabla_{\theta_j}\mathcal{L}(\theta_j) \cdot \nabla_{\theta_k}\mathcal{L}(\theta_k)$ 最小化）
- **空间排斥力**：核梯度项 $\nabla_\theta k(\theta, \theta_j)$ 防止粒子坍缩

**统一视角**：FHBI 是 SAM 和 SVGD 的推广——$\rho=0$ 退化为 SVGD，$m=1$ 退化为 SAM。

## 实验关键数据

在 VTAB-1K 基准（19个数据集，涵盖 Natural/Specialized/Structured 三类）上，使用 ViT-B/16 + LoRA 微调：

| 方法 | Natural (7) | Specialized (4) | Structured (8) | 平均 |
|------|------------|-----------------|----------------|------|
| AdamW | 79.1 | 84.3 | 59.0 | 72.0 |
| SAM | 80.1 | 83.2 | 56.0 | 70.5 |
| DeepEns | 79.3 | 83.9 | 42.8 | 67.0 |
| BayesTune | 80.5 | 84.9 | 59.3 | 72.2 |
| SVGD | 79.8 | 84.6 | 56.3 | 70.9 |
| SADA-JEM | 80.3 | 84.7 | 58.6 | 72.1 |
| **FHBI** | **82.4** | **86.9** | **61.6** | **73.7** |

- FHBI 在全部三个域（Natural、Specialized、Structured）均取得最优
- 平均 Top-1 准确率 73.7%，超过最佳基线 BayesTune 1.5 个百分点
- 在难度较高的 Structured 数据集上优势更明显

**校准误差 (ECE)**：FHBI 在多个数据集上也取得最低 ECE，说明模型置信度校准更好。

## 亮点与洞察

1. **理论贡献扎实**：首次将 SAM 的泛化界从有限维推广到无限维 RKHS，非平凡（需处理无限维度问题）
2. **优雅的统一视角**：FHBI 统一了 SAM（单粒子平坦性）和 SVGD（多粒子后验逼近），揭示了两者的内在联系
3. **三重多样性机制**：锐度+角度排斥+空间排斥，三重力促进粒子多样化
4. **实验全面**：19 个数据集 × 9 个基线，在 Top-1 准确率和 ECE 两个指标上均验证有效
5. **与 LoRA 自然结合**：仅对轻量 LoRA 参数运行多粒子推断，计算开销可控

## 局限与展望

1. **计算成本**：$m$ 个粒子意味着 $m$ 倍前向/反向传播+ 核矩阵计算，扩展到大规模模型时开销不可忽视
2. **仅评估 ViT-B/16 + LoRA**：未在更大模型（ViT-L、LLM）或其他 PEFT 方法上验证
3. **核函数选择敏感性**：论文使用 RBF 核，未充分探讨其他核函数的影响
4. **理论-实践差距**：Lemma 4.4 依赖 $\|f\|$ 足够小的近似，实际中该条件的满足程度未详细讨论
5. **仅图像分类任务**：未在 NLP、目标检测等任务上验证泛化性

## 相关工作与启发

- **SAM** (Foret et al., 2021)：FHBI 的欧氏空间特例 ($m=1$)
- **SVGD** (Liu & Wang, 2016)：FHBI 的无扰动特例 ($\rho=0$)
- **BayesTune** (Kim et al., 2023)：贝叶斯微调基线
- **SA-BNN** / **SADA-JEM**：锐度感知贝叶斯方法
- **启发**：将优化手段（锐度感知）系统性提升到函数空间层面，是贝叶斯深度学习的重要方向

## 评分

- 新颖性: ⭐⭐⭐⭐ （RKHS 泛化界+统一 SAM/SVGD 视角）
- 实验充分度: ⭐⭐⭐⭐ （19 数据集 × 9 基线，多指标）
- 写作质量: ⭐⭐⭐⭐ （理论推导清晰，图示直观）
- 价值: ⭐⭐⭐⭐ （贝叶斯推断+泛化理论的有意义融合）

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces](function_encoders_a_principled_approach_to_transfer_learning_in_hilbert_spaces.md)
- [\[ICML 2025\] IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness](promoting_ensemble_diversity_with_interactive_bayesian_distributional_robustness.md)
- [\[ICML 2025\] Set-Valued Predictions for Robust Domain Generalization](set_valued_predictions_for_robust_domain_generalization.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](../../NeurIPS2025/llm_evaluation/bayesian_evaluation_of_large_language_model_behavior.md)
- [\[ICML 2025\] Hyperband-based Bayesian Optimization for Black-box Prompt Selection](hyperband-based_bayesian_optimization_for_black-box_prompt_selection.md)

<!-- RELATED:END -->
