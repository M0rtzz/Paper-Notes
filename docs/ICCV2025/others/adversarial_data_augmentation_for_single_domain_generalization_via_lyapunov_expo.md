---
title: >-
  [论文解读] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents
description: >-
  [ICCV 2025][单域泛化] 提出 LEAwareSGD 优化器，利用 Lyapunov 指数（LE）动态调节学习率，将模型训练引导至"混沌边缘"附近，从而在单源域泛化任务中实现更广泛的参数空间探索和更强的跨域泛化能力。 单域泛化（SDG）是域泛化领域中最具挑战性的设置——仅用一个源域的数据训练模型…
tags:
  - "ICCV 2025"
  - "单域泛化"
  - "对抗数据增强"
  - "Lyapunov 指数"
  - "混沌边缘"
  - "学习率调节"
---

# Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents

**会议**: ICCV 2025  
**arXiv**: [2507.04302](https://arxiv.org/abs/2507.04302)  
**代码**: 无  
**领域**: 优化  
**关键词**: 单域泛化, 对抗数据增强, Lyapunov 指数, 混沌边缘, 学习率调节

## 一句话总结

提出 LEAwareSGD 优化器，利用 Lyapunov 指数（LE）动态调节学习率，将模型训练引导至"混沌边缘"附近，从而在单源域泛化任务中实现更广泛的参数空间探索和更强的跨域泛化能力。

## 研究背景与动机

单域泛化（SDG）是域泛化领域中最具挑战性的设置——仅用一个源域的数据训练模型，期望其能泛化到未见的目标域。现有 SDG 方法主要依赖数据增强技术，其中对抗数据增强（ADA）通过生成模拟域偏移的扰动样本来增强模型鲁棒性。

然而现有对抗增强方法存在关键缺陷：它们倾向于引入**局部化扰动**，无法充分探索参数空间的全局结构。从 t-SNE 可视化（Figure 1）可以清晰看到，ADA、ME-ADA、AdvST 等方法的参数轨迹在训练初期即集中在有限区域，限制了模型学习跨域可泛化特征的能力。

作者从动力系统理论中获得灵感：将神经网络训练视为离散时间动力系统中的状态转移过程，利用 Lyapunov 指数（LE）衡量系统对扰动的敏感性，并将训练引导至"混沌边缘"——这一临界状态在稳定性与适应性之间取得最优平衡，是可泛化特征最有可能出现的区域。

## 方法详解

### 整体框架

LEAwareSGD 是一个基于 LE 反馈的优化器，与对抗数据增强策略结合使用。训练过程交替进行两步：(1) 生成对抗样本模拟域偏移；(2) 用 LE 引导的学习率更新模型参数。整体目标是一个 min-max 优化问题。

### 关键设计

1. **基于 LE 的模型扰动分析**：将标准梯度下降视为动力系统 $\theta_{t+1} = \theta_t - \eta_t \nabla L(\theta_t)$，引入微小扰动 $\delta\theta_0$，分析其传播规律。通过一阶 Taylor 展开：

$$\delta\theta_{t+1} = (I - \eta_t H[L(\theta_t)]) \delta\theta_t$$

递归展开后，Lyapunov 指数定义为：

$$\text{LE} = \lim_{t \to \infty} \frac{1}{t} \ln\left(\frac{\|\delta\theta_t\|}{\|\delta\theta_0\|}\right)$$

由此推导出 LE 的上下界均由学习率 $\eta_i$ 和 Hessian 矩阵 $H[L(\theta_i)]$ 决定，建立了 LE 与学习率之间的数学联系。

2. **LE 引导的学习率调节**：核心创新。当 $\Delta\text{LE}_t = \text{LE}_t - \text{LE}_{t-1} > 0$（系统趋向混沌边缘），降低学习率以深入探索该区域：

$$\eta_{t+1} = \eta_t \cdot \exp(-\beta \cdot \Delta\text{LE}_t), \quad \text{if } \Delta\text{LE}_t > 0$$

当 $\Delta\text{LE}_t \leq 0$ 时保持学习率不变。$\beta$ 是控制灵敏度的超参数。设计动机在于：LE 增大意味着系统靠近混沌边缘，此时可泛化特征更可能出现，需要减速深入探索。

3. **LE 感知的对抗数据增强**：将 LEAwareSGD 与对抗增强结合，联合优化目标为：

$$\min_\theta \max_\omega \mathbb{E}_{(x,y) \sim \mathcal{D}_S} [\ell(\theta; \tau(x;\omega), y) - \lambda d_\theta(\tau(x;\omega), x)] + \frac{\gamma}{2} \|\theta\|_2^2$$

其中 $\tau(x;\omega)$ 是语义变换，$\lambda$ 平衡对抗损失与特征一致性，$\gamma$ 控制权重衰减。权重衰减项确保 Hessian 矩阵近似正定，使 LE 趋于负值以维持训练稳定性。

### 损失函数 / 训练策略

- 使用标准交叉熵 + L2 正则化
- PACS: batch size 16, $\gamma$=5e-4, lr=5e-4, 50 epochs
- OfficeHome: batch size 32, $\gamma$=1e-4, lr=1e-4, 50 epochs
- DomainNet: batch size 128, $\gamma$=1e-5, lr=1e-3, 200 epochs
- backbone 统一使用 ResNet-18

## 实验关键数据

### 主实验

| 方法 | PACS Avg | OfficeHome Avg | DomainNet Avg |
|------|---------|---------------|--------------|
| ERM | 57.80 | 43.60 | 23.77 |
| ADA | 61.11 | 44.75 | 24.26 |
| ME-ADA | 60.22 | 45.35 | 24.63 |
| AdvST | 67.06 | 52.60 | 27.22 |
| PSDG | 67.14 | 47.05 | 26.28 |
| **LEAwareSGD** | **69.46** | **54.38** | **28.15** |

*在三个基准上均取得 SOTA，PACS 上超越 AdvST 2.40%，OfficeHome 超越 1.78%。*

### 消融实验

| 数据比例 | AdvST | LEAwareSGD | 提升 |
|---------|-------|-----------|------|
| 10% | 49.26 | 58.73 | **+9.47** |
| 20% | 53.55 | 61.78 | +8.23 |
| 50% | 60.18 | 66.44 | +6.26 |

*PACS 低数据场景下的对比。LEAwareSGD 在仅 10% 数据时仍能大幅超越 AdvST，展现极强的低资源泛化能力。*

| 优化器 | A | C | P | S | Avg |
|--------|------|------|------|------|------|
| Adam | 76.52 | 71.15 | 64.06 | 53.98 | 66.43 |
| AdamW | 76.68 | 71.93 | 62.03 | 56.68 | 66.83 |
| SGD | 76.65 | 74.92 | 62.47 | 54.18 | 67.06 |
| **LEAwareSGD** | **79.17** | **77.16** | **65.05** | **57.78** | **69.46** |

*与常用优化器对比（PACS）。Adam 系列因倾向于尖锐极小值而泛化较差。*

### 关键发现

- LE 动态曲线（Figure 3）显示 LEAwareSGD 在所有域上均使 LE 值稳定在接近零的区域（混沌边缘），而其他方法波动更大或 LE 过低
- LEAwareSGD 作为插件可普遍提升现有对抗增强方法：ADA 提升 0.52-5.15%，ME-ADA 提升 2.30-7.00%，AdvST 提升 1.78-2.40%
- 在 ResNet-34/50/101/152 上均验证了一致的泛化提升，ResNet-152 在 PACS 上达到 75.34%
- 训练时间仅比 AdvST 略增（PACS: 1.99h vs 1.90h）

## 亮点与洞察

- 首次将 Lyapunov 指数引入域泛化领域，建立了"混沌边缘训练"的理论框架
- LEAwareSGD 是一个通用优化器，可无缝融入任何对抗数据增强流程
- 低数据场景下（10%）的巨大优势（+9.47%）表明混沌边缘训练特别有助于从有限数据中提取可泛化特征

## 局限与展望

- LE 的计算需要维护扰动副本并前向传播两次，引入额外计算开销
- 当前框架使用表格搜索确定最优 $\beta$ 和 $\gamma$，自适应调参策略有待探索
- DomainNet Quickdraw 域效果略低于 SimDE，可能需要域特异性的增强策略

## 相关工作与启发

- 与 SAM/GSAM 等关注损失面平坦度的优化器不同，LEAwareSGD 从动力系统稳定性出发调节学习率
- "混沌边缘"概念来自动力系统理论，此前主要在 RNN 和残差网络的稳定性分析中使用，本文首次将其用于直接控制训练
- 对其他需要域泛化的任务（医学影像、自动驾驶）有直接借鉴价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Curvature Enhanced Data Augmentation for Regression](../../ICML2025/others/curvature_enhanced_data_augmentation_for_regression.md)
- [\[CVPR 2025\] Gradient-Guided Annealing for Domain Generalization](../../CVPR2025/others/gradient-guided_annealing_for_domain_generalization.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](../../ICLR2026/others/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)
- [\[ACL 2025\] Is Linguistically-Motivated Data Augmentation Worth It?](../../ACL2025/others/is_linguistically-motivated_data_augmentation_worth_it.md)
- [\[ICML 2025\] Set-Valued Predictions for Robust Domain Generalization](../../ICML2025/others/set_valued_predictions_for_robust_domain_generalization.md)

</div>

<!-- RELATED:END -->
