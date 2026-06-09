---
title: >-
  [论文解读] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents
description: >-
  [ICCV 2025][单域泛化] 提出 LEAwareSGD 优化器，利用 Lyapunov 指数（LE）动态调节学习率，将模型训练引导至"混沌边缘"附近，从而在单源域泛化任务中实现更广泛的参数空间探索和更强的跨域泛化能力。
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
**领域**: 其他  
**关键词**: 单域泛化, 对抗数据增强, Lyapunov 指数, 混沌边缘, 优化器

## 一句话总结

提出 LEAwareSGD 优化器，利用 Lyapunov 指数（LE）动态调节学习率，引导模型训练在混沌边缘附近，在对抗数据增强框架下实现更广泛的参数空间探索，显著提升单域泛化（SDG）性能。

## 研究背景与动机

单域泛化（SDG）旨在仅用单一源域训练出能泛化到未知目标域的模型，核心困难在于训练数据多样性不足以及域偏移幅度大。现有 SDG 方法主要依赖数据增强技术：

**对抗数据增强**（ADA, ME-ADA, AdvST 等）：生成模拟域偏移的扰动样本增强鲁棒性。然而这些方法的扰动往往是局部的，无法有效探索全局参数空间，导致模型捕获可泛化特征的能力受限。

**生成模型方法**（PDEN 等）：通过合成样本扩展源域分布，但计算开销大且生成质量难以保证。

作者从动力系统理论获得启发：将神经网络训练视为参数空间中的离散时间动力系统，每次参数更新是一次状态转移。**混沌边缘**（edge of chaos）是秩序与混沌之间的临界状态，此处系统能同时保持稳定性与适应性。Lyapunov 指数（LE）是量化系统混沌程度的经典指标——LE > 0 表示混沌（扰动指数级增长），LE < 0 表示稳定（扰动衰减）。在混沌边缘（LE ≈ 0⁻），模型既不会过拟合也不会发散，最有利于学习泛化特征。

然而，现有优化器（SGD, Adam 等）没有考虑系统动态状态，缺乏根据训练稳定程度主动调节的机制。

## 方法详解

### 整体框架

LEAwareSGD 将 Lyapunov 指数计算集成到对抗数据增强优化中：(1) 计算参数扰动的传播来估计 LE；(2) 根据 LE 变化动态调节学习率；(3) 在对抗数据增强框架中联合优化。

### 关键设计

1. **基于 LE 的模型扰动分析**：引入初始扰动 $\delta\theta_0$，得到扰动后参数 $\tilde{\theta_t} = \theta_t + \delta\theta_t$。通过一阶 Taylor 展开得到扰动传播公式：

$$\delta\theta_{t+1} = (I - \eta_t H[L(\theta_t)]) \delta\theta_t$$

递归展开后代入 LE 定义 $LE = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\|\delta\theta_t\|}{\|\delta\theta_0\|}$，建立 LE 与学习率 $\eta$ 及 Hessian 矩阵 $H$ 的关系。LE 受此二者联合决定。

2. **LE 引导的学习率调节**：核心创新。计算 LE 变化量 $\Delta LE_t = LE_t - LE_{t-1}$：

    - 当 $\Delta LE_t > 0$（模型接近混沌边缘），降低学习率以在该区域深度探索：
    $\eta_{t+1} = \eta_t \cdot \exp(-\beta \cdot \Delta LE_t)$
    - 当 $\Delta LE_t \leq 0$，保持学习率不变
   
   参数 $\beta$ 控制调节灵敏度。这种设计确保模型在 LE 上升（接近泛化特征丰富区域）时放慢脚步充分探索，而非快速通过。

3. **对抗数据增强联合优化**：采用标准的 minimax 框架，内循环最大化变换后样本的损失（对抗增强），外循环最小化模型在增强样本上的损失。加入权重衰减正则化 $\frac{\gamma}{2}\|\theta\|_2^2$ 确保 Hessian 近似正定，从而 LE 倾向为负，促进训练稳定。

### 损失函数 / 训练策略

$$\min_\theta \max_\omega \mathbb{E}_{(x,y)\sim\mathcal{D}_S} [\ell(\theta; \tau(x;\omega), y) - \lambda d_\theta(\tau(x;\omega), x)] + \frac{\gamma}{2}\|\theta\|_2^2$$

- $\ell$：预测损失
- $d_\theta$：原始样本与变换样本的特征距离
- $\lambda$：对抗损失与特征一致性权衡
- $\gamma$：权重衰减系数

训练交替进行：先生成对抗样本模拟域偏移，再用 LEAwareSGD 更新参数。

## 实验关键数据

### 主实验

在 PACS、OfficeHome、DomainNet 三个标准 SDG 基准上评估，backbone 为 ResNet-18。

| 方法 | PACS Avg. | OfficeHome Avg. | DomainNet Avg. |
|------|----------|----------------|---------------|
| ERM | 57.80 | 43.60 | 23.77 |
| ADA | 61.11 | 44.75 | 24.26 |
| ME-ADA | 60.22 | 45.35 | 24.63 |
| AdvST | 67.06 | 52.60 | 27.22 |
| PSDG | 67.14 | 47.05 | 26.28 |
| **LEAwareSGD (Ours)** | **69.46** | **54.38** | **28.15** |

*在所有三个基准上均取得最优结果，PACS 上超越第二名 2.32%。*

### 消融实验

| 优化器 | PACS A | C | P | S | Avg. |
|-------|-------|---|---|---|------|
| Adam | 76.52 | 71.15 | 64.06 | 53.98 | 66.43 |
| AdamW | 76.68 | 71.93 | 62.03 | 56.68 | 66.83 |
| RMSprop | 71.62 | 71.08 | 59.09 | 47.55 | 62.34 |
| SGD | 76.65 | 74.92 | 62.47 | 54.18 | 67.06 |
| **LEAwareSGD** | **79.17** | **77.16** | **65.05** | **57.78** | **69.46** |

*LEAwareSGD 比标基准 SGD 高 2.40%，自适应优化器（Adam 等）表现明显较差。*

**低数据场景**（10% 数据量下 PACS）：LEAwareSGD 达 58.73%，AdvST 仅 49.26%，提升 **9.47%**。

### 关键发现

- LE 值在训练过程中趋近于 0 的负值区域（混沌边缘），相比其他方法的 LE 波动更小更稳定
- LEAwareSGD 可作为即插即用组件提升不同对抗增强方法：ADA +0.52%, ME-ADA +2.30%, AdvST +2.40%（PACS）; OfficeHome 上 ME-ADA +7.00%
- 不同 ResNet backbone（18/34/50/101/152）均有效，ResNet-152 达 75.34%
- 计算开销仅略高于 AdvST（PACS平均 1.99h vs 1.90h），低于 ADA（2.13h）

## 亮点与洞察

- 首次将动力系统理论中的 Lyapunov 指数引入单域泛化优化，建立了混沌边缘与模型泛化能力之间的理论联系
- 方法简洁优雅：仅需在标准 SGD 基础上增加一个 LE 反馈机制调节学习率
- t-SNE 可视化直观展示了 LEAwareSGD 比其他方法探索了更广泛的参数空间

## 局限与展望

- LE 的计算依赖 Hessian 矩阵，在大规模模型上可能存在近似误差
- 当前仅在分类任务上验证，检测/分割等任务有待探索
- DomainNet Quickdraw 域效果略逊于 SimDE（6.70% vs 6.85%），高度抽象域可能需要专门增强策略

## 相关工作与启发

- SAM 和 GSAM 通过最小化损失面锐度促进泛化，LEAwareSGD 从动力系统角度提供了互补视角
- 将 LE 作为优化反馈信号的思路可推广到其他需要平衡稳定性与探索性的训练场景
- 混沌边缘理论为理解"什么样的训练动态有利于泛化"提供了新的理论框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐
# Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents

**会议**: ICCV 2025  
**arXiv**: [2507.04302](https://arxiv.org/abs/2507.04302)  
**代码**: 无  
**领域**: optimization  
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
