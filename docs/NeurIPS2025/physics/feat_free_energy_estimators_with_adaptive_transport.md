---
title: >-
  [论文解读] FEAT: Free Energy Estimators with Adaptive Transport
description: >-
  [NeurIPS 2025][物理/科学计算][自由能估计] 提出 FEAT 框架，利用随机插值学习两个热力学系统之间的传输映射，基于 escorted Jarzynski 等式和 controlled Crooks 定理提供一致、最小方差的自由能差估计器及变分上下界，统一了平衡与非平衡方法。
tags:
  - "NeurIPS 2025"
  - "物理/科学计算"
  - "自由能估计"
  - "随机插值"
  - "Jarzynski 等式"
  - "Crooks 定理"
  - "变分界"
---

# FEAT: Free Energy Estimators with Adaptive Transport

**会议**: NeurIPS 2025  
**arXiv**: [2504.11516](https://arxiv.org/abs/2504.11516)  
**代码**: [GitHub](https://github.com/jiajunhe98/FEAT)  
**领域**: 计算物理 / 分子模拟  
**关键词**: 自由能估计, 随机插值, Jarzynski 等式, Crooks 定理, 变分界

## 一句话总结

提出 FEAT 框架，利用随机插值学习两个热力学系统之间的传输映射，基于 escorted Jarzynski 等式和 controlled Crooks 定理提供一致、最小方差的自由能差估计器及变分上下界，统一了平衡与非平衡方法。

## 研究背景与动机

### 现有痛点

**现有痛点**：自由能估计是统计力学、化学、生物和机器学习中的基本挑战（如配分函数计算、配体结合自由能）

### 领域现状

**领域现状**：经典方法（FEP、BAR、TI）依赖平衡采样或中间系统，在高维空间中分布重叠不足时失效

### 核心矛盾

**核心矛盾**：Jarzynski 等式引入非平衡轨迹，但估计器方差大

### 解决思路

**解决思路**：近年深度学习方法（normalizing flows + targeted FEP、neural TI）有进展，但非平衡方法在深度学习框架中仍未被充分探索

### 补充说明

**补充说明**：FEAT 的定位**：利用随机插值高效学习传输，通过 escorted Jarzynski 和 Crooks 定理提供更灵活、更低方差的估计器

## 方法详解

### 整体框架

FEAT 的核心流程：

1. **学习传输**：给定两个端点系统 $S_a$ 和 $S_b$ 的样本，用随机插值 $I_t = \alpha_t x_a + \beta_t x_b + \gamma_t \epsilon$ 学习速度场 $v_t^\psi$ 和能量梯度 $\nabla U_t^\theta$
2. **计算广义功**：沿学习到的 SDE 路径计算（校正后的）广义功 $\widetilde{W}^v$
3. **估计自由能差**：利用 escorted Jarzynski 等式给出变分界，利用非平衡 BAR 给出最小方差估计

### 关键设计

1. **随机插值学习传输**:
    - 功能：用两个神经网络分别学习速度场 $v_t^\psi(x)$ 和得分函数 $\nabla U_t^\theta(x)$
    - 核心思路：速度场通过回归损失 $\mathcal{L}_v = \mathbb{E}[|v_t^\psi(I_t) - \dot{I}_t|^2]$ 训练；得分函数通过去噪得分匹配 $\mathcal{L}_U^{\text{DSM}} = \mathbb{E}[|\nabla U_t^\theta(I_t) - \gamma_t^{-1}\epsilon|^2]$ 训练
    - 设计动机：随机插值同时学习传输和能量插值，无需预定义能量路径；训练过程不需要 Langevin 动力学模拟

2. **边界条件校正 + FB RND 估计**:
    - 功能：处理学习到的 $U_0^\theta \neq U_a$、$U_1^\theta \neq U_b$ 的情况，用前向-后向 Radon-Nikodym 导数避免散度计算
    - 核心思路：校正广义功加入端点能量修正项；FB RND 形式消除了对 $\nabla \cdot v_t$ 的计算需求且对离散化误差鲁棒
    - 设计动机：实际中边界条件难以精确满足；散度计算开销大且引入精度问题

### 损失函数 / 训练策略

- 总训练损失：$\mathcal{L}_v(\psi) + \mathcal{L}_U^{\text{DSM}}(\theta) + \mathcal{L}_U^{\text{TSM,0}}(\theta) + \mathcal{L}_U^{\text{TSM,1}}(\theta)$
- TSM（Target Score Matching）用 $\nabla U_a(x_a)$ 和 $\nabla U_b(x_b)$ 作为端点监督信号，改善边界条件
- 估计器层次：变分下界 (ELBO) ≤ IWAE 下界 ≤ $\Delta F$ ≤ IWAE 上界 ≤ 变分上界 (EUBO)
- 非平衡 BAR：迭代求解 $C = \Delta F$，给出最小方差估计

## 实验关键数据

### 主实验（表格）

| 方法 | GMM (2D) 误差 | 丙氨酸二肽 误差 | φ⁴ 量子场论 误差 |
|------|-------------|---------------|----------------|
| Targeted FEP | 0.15 | 较大 | 较大 |
| Neural TI | 0.08 | 中等 | 中等 |
| **FEAT (BAR)** | **0.02** | **最小** | **最小** |

### 消融实验

- FEAT-BAR vs FEAT-Jarzynski：BAR 一致优于单向 Jarzynski，方差更低
- 有/无 TSM 损失：TSM 显著改善边界条件匹配精度
- FB RND vs 散度形式：FB RND 对离散化更鲁棒，且计算成本更低
- ODE ($\sigma_t = 0$) vs SDE ($\sigma_t > 0$)：SDE 在高维问题中方差更小

### 关键发现

- FEAT 在所有测试场景中显著优于 targeted FEP 和 neural TI
- 子框架关系：静态采样 ($\sigma_t=0, v_t=0$) = FEP；ODE 传输 ($\sigma_t=0$) = targeted FEP/CNF；完美传输 = TI
- 量子场论实验（$\phi^4$ 理论）展示了 FEAT 在物理基础问题上的适用性

## 亮点与洞察

- **理论统一**：首次在单一框架下统一 FEP、BAR、targeted FEP、TI 和 Jarzynski 方法
- 非平衡 BAR 估计器兼具一致性和最小方差性质
- FB RND 消除散度计算的技巧具有广泛适用性
- 从机器学习角度看，FEAT 架起了变分推断（ELBO/EUBO）和统计物理（Jarzynski/Crooks）的桥梁

## 局限与展望

- 需要两端系统的精确样本，不适用于只有单端样本的场景
- 神经网络训练质量直接影响估计精度
- 大分子系统的扩展性有待验证（实验中最大的系统是丙氨酸二肽）
- 离散化误差虽然通过 FB RND 缓解但未完全消除

## 相关工作与启发

- 与 neural TI (Máté et al.) 的关系：neural TI 是 FEAT 在完美传输极限下的特例
- 与 normalizing flow 方法的关系：CNF 是 FEAT 在 $\sigma_t=0$ 时的 ODE 特例
- 对药物设计中的相对结合自由能计算具有直接应用价值

## 评分

- 理论创新：⭐⭐⭐⭐⭐
- 实验验证：⭐⭐⭐⭐
- 实用价值：⭐⭐⭐⭐
- 写作质量：⭐⭐⭐⭐
- 综合评分：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Universal Neural Optimal Transport](../../ICML2025/physics/universal_neural_optimal_transport.md)
- [\[NeurIPS 2025\] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)
- [\[NeurIPS 2025\] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)

</div>

<!-- RELATED:END -->
