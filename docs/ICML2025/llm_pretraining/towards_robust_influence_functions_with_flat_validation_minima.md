---
title: >-
  [论文解读] Towards Robust Influence Functions with Flat Validation Minima
description: >-
  [ICML 2025][influence function] 揭示影响函数 (IF) 在含噪训练数据上失效的根本原因不在于 Hessian 逆近似不准（先前研究的焦点），而在于验证损失的**尖锐度**导致损失变化估计失真，理论推导出 IF 误差上界与验证风险尖锐度的联系，并设计出专用于平坦验证极小值的新 IF 形式 (FVM)。
tags:
  - ICML 2025
  - influence function
  - flat minima
  - SAM
  - noisy label detection
  - data attribution
---

# Towards Robust Influence Functions with Flat Validation Minima

**会议**: ICML 2025  
**arXiv**: [2505.19097](https://arxiv.org/abs/2505.19097)  
**代码**: [GitHub](https://github.com/Virusdoll/IF-FVM)  
**领域**: 可解释AI / 训练数据影响  
**关键词**: influence function, flat minima, SAM, noisy label detection, data attribution

## 一句话总结

揭示影响函数 (IF) 在含噪训练数据上失效的根本原因不在于 Hessian 逆近似不准（先前研究的焦点），而在于验证损失的**尖锐度**导致损失变化估计失真，理论推导出 IF 误差上界与验证风险尖锐度的联系，并设计出专用于平坦验证极小值的新 IF 形式 (FVM)。

## 研究背景与动机

**领域现状**：影响函数 (IF) 是评估单个训练样本对模型预测影响的核心工具。标准 IF 公式为 $\mathcal{I}(z_\text{tr}; z_\text{val}) = g_{z_\text{val}}^\top H_\text{tr}^{-1} g_{z_\text{tr}}$，通过两步近似——参数变化估计（Newton step）和损失变化估计（一阶展开）——来避免昂贵的 leave-one-out 重训练。

**现有痛点**：IF 在含噪训练数据上表现极差——但这恰恰是它最需要发挥作用的场景（区分好数据和坏数据）。无论使用一阶近似（TracIn）还是二阶近似（LiSSA），IF 的失效模式是一样的。

**核心矛盾**：先前研究聚焦于改进 Hessian 逆近似精度，但本文发现问题不在这里。即使完美估计了参数变化 $\Delta\theta$，如果验证损失景观是尖锐的（sharp），一阶展开也会产生巨大的估计差距。

**本文目标** 从根本上修复 IF 在含噪数据上的估计可靠性问题。

**切入角度**：将 IF 失效重新归因到损失变化估计（第二步），利用 SAM 寻找平坦验证极小值。

**核心 idea**：在平坦验证极小值上计算影响函数，并用二阶展开替代一阶展开来估计损失变化。

## 方法详解

### 整体框架

本文分两层：
1. **理论分析**：建立 IF 估计误差 ↔ 验证风险尖锐度的上界关系
2. **方法设计**：提出 VM (Validation Minima) 和 FVM (Flat Validation Minima) 两种新 IF 估计器

### 关键设计

1. **IF 估计误差上界定理 (Theorem 3.2)**:
    - 功能：理论解释 IF 在含噪数据上为什么失效
    - 核心思路：定义估计误差为符号判断错概率 $\mathcal{E}(\mathcal{I}) = \mathbb{P}[\text{sgn}(\mathcal{I}) \neq \text{sgn}(\mathcal{I}^*)]$，证明上界：
    $$\mathcal{E}(\mathcal{I}) \leq \exp\left(-\frac{2\mu^2}{\hat{R}_\text{val}^\gamma(\theta)^2}\right)$$
    其中 $\hat{R}_\text{val}^\gamma(\theta) = \max_{\|\Delta\| \leq \gamma} \hat{R}_\text{val}(\theta + \Delta)$ 含验证风险和尖锐度，$\mu$ 表征 IF 判别力下界
    - 设计动机：上界说明需要同时减小验证风险和尖锐度

2. **标准 IF 在平坦极小值上的失效分析**:
    - 功能：解释不能简单地 SAM + 标准 IF
    - 核心思路：两个问题——(1) 参数变化估计基于训练集 Hessian，但参数在验证集极小值处，错位；(2) 收敛模型梯度近零均值，一阶项 $g_\text{val}^\top H_\text{tr}^{-1} g_{z_\text{tr}}$ 期望趋零，$\mu \to 0$
    - 设计动机：Figure 3 实验直观证明——SAM 训后标准 IF 的 AUC 反而下降

3. **新 IF 形式 (VM/FVM)**:
    - 功能：专为平坦验证极小值设计的影响估计
    - 核心思路：同时修复两步——
        - **参数变化**：在平坦验证极小值上重新定义扰动，Newton step 使用**验证集 Hessian** $\tilde{H}_\text{val}$ 而非训练集 Hessian
        - **损失变化**：用**二阶近似**替代一阶：$\ell(z_\text{val}, \tilde{\theta}_{z_\text{tr}}) - \ell(z_\text{val}, \tilde{\theta}) \approx \frac{1}{2} \Delta\theta^\top \nabla^2 \ell(z_\text{val}, \tilde{\theta}) \Delta\theta$
        - 最终全验证集 IF：$\mathcal{I}(z_\text{tr}, S_\text{val}) = \tilde{g}_{z_\text{tr}}^\top \tilde{H}_\text{val}^{-1} \tilde{g}_{z_\text{tr}}$
    - 设计动机：二阶项不受梯度趋零影响（Hessian 不趋零），Hessian 在验证集上计算解决对齐问题

### 损失函数 / 训练策略

- **VM**：在验证集上用 ERM 训练到极小值后计算新 IF
- **FVM**：在验证集上用 SAM 训练寻找平坦极小值后计算新 IF
- SAM 实现使用 LPF-SGD，$M=1$

## 实验关键数据

### 噪声标签检测：CIFAR-10N/100N（ROC AUC %）

| 方法 | CIFAR-10N Aggre | CIFAR-10N Random | CIFAR-10N Worst | CIFAR-100N Noisy |
|------|:-:|:-:|:-:|:-:|
| LiSSA | 59.74±2.91 | 59.78±2.77 | 65.75±0.39 | 57.48±1.70 |
| TracIn | 53.91±5.85 | 61.61±0.74 | 65.74±2.32 | 56.13±2.51 |
| GEX | 87.38±1.21 | 91.11±0.53 | 93.28±0.10 | **90.17±0.70** |
| DataInf | 58.50±3.98 | 54.50±2.32 | 55.49±1.45 | 53.69±1.35 |
| **VM** | 95.18±0.15 | 95.92±0.10 | 95.88±0.13 | 89.77±0.08 |
| **FVM** | **96.14±0.06** | **96.63±0.03** | **96.46±0.08** | 90.80±0.04 |

### 噪声标签重标注（Top-1 Accuracy %）

| 方法 | Aggre | Random | Worst | CIFAR-100N |
|------|:---:|:---:|:---:|:---:|
| LiSSA | 5.28 | 9.04 | 19.32 | 0.28 |
| TracIn | 37.08 | 53.28 | 50.66 | 20.11 |
| GEX | 30.19 | 54.03 | 80.35 | 22.41 |
| **VM** | 94.17 | 91.94 | 85.01 | 58.13 |
| **FVM** | **94.63** | **92.46** | **86.09** | **70.61** |

FVM 在 CIFAR-100N 重标注上达到 70.61%，是先前最佳 GEX (22.41%) 的 3 倍以上。

### 文本生成影响力识别（Llama-2-13B + LoRA）

| 任务 | 方法 | ROC AUC | Recall |
|------|------|:---:|:---:|
| Sentence Transformations | TracIn | 94.95±6.14 | 70.97±25.18 |
| Sentence Transformations | DataInf | 99.58±1.96 | 96.18±9.33 |
| Sentence Transformations | **VM** | **99.97±0.16** | **99.10±2.79** |
| Math Problems | TracIn | 78.50±17.77 | 26.61±39.95 |
| Math Problems | DataInf | 99.86±0.68 | 97.37±6.97 |
| Math Problems | **FVM** | **99.99±0.07** | **99.38±1.65** |

### 关键发现

- FVM 在 CIFAR-10N 所有设定上大幅超越基线（比 GEX 高 3-6% AUC），且方差极小
- IF 性能与验证集准确率高度相关（Figure 2），直接验证理论
- 标准 IF 在 SAM 训后反而退化（Figure 3a），证明必须用新 IF 形式
- VM 已大幅优于标准 IF，FVM 进一步在大多数设定上更优

## 亮点与洞察

1. **颠覆传统认知**：IF 失败不在 Hessian 近似，而在损失变化估计的尖锐性
2. **理论与实验完美对应**：Theorem 3.2 的预测通过 Figure 2/3 精确验证
3. **简洁修复**：核心改动两处——验证集上训练 + 二阶替代一阶
4. **性能不是渐进改善而是质变**：重标注从 ~22% 到 ~70%

## 局限与展望

- SAM 训练验证集需要额外计算成本
- 二阶 IF 比一阶复杂，大规模应用受限
- 需要干净验证集（某些场景难获取）
- 对非噪声数据场景的改善程度待系统验证

## 相关工作与启发

- **Koh & Liang (2017) 标准 IF**：本文的改进基础
- **TracIn (Pruthi et al., 2020)**：一阶 IF 近似，同受梯度趋零影响
- **GEX (Kim et al., 2023)**：利用梯度期望做归因，需 32 次调优（FVM 仅需 1 次）
- **SAM (Foret et al., 2021)**：提供了寻找平坦极小值的优化工具
- 启发：损失景观几何性质对梯度基方法的影响被低估

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 识别了被忽视十年的 IF 核心失败模式
- 实验充分度: ⭐⭐⭐⭐⭐ 四类任务、多数据集、详细消融
- 写作质量: ⭐⭐⭐⭐ Figure 2/3 的诊断设计精巧
- 价值: ⭐⭐⭐⭐⭐ 重标注能力的突破开辟新应用空间

<!-- RELATED:START -->

## 相关论文

- [Superposition Yields Robust Neural Scaling](../../NeurIPS2025/llm_pretraining/superposition_yields_robust_neural_scaling.md)
- [ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](../../NeurIPS2025/llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [On the Clean Generalization and Robust Overfitting in Adversarial Training from Two Theoretical Views: Representation Complexity and Training Dynamics](on_the_clean_generalization_and_robust_overfitting_in_adversarial_training_from_.md)
- [RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization](../../ICLR2026/llm_pretraining/recon_robust_symmetry_discovery_via_explicit_canonical_orientation_normalization.md)

<!-- RELATED:END -->
