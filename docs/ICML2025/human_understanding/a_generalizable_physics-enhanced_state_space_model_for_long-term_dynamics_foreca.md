---
title: >-
  [论文解读] A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments
description: >-
  [ICML 2025][人体理解][状态空间模型] 提出 Phy-SSM，将部分已知的物理知识融入深度状态空间模型（SSM），通过动力学分解（已知/未知矩阵）和物理状态正则化，实现对噪声大、不规则采样数据的长期动力学精准预测与外推。
tags:
  - ICML 2025
  - 人体理解
  - 状态空间模型
  - 部分物理知识
  - 长期动力学预测
  - 不规则采样
  - 物理增强机器学习
---

# A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments

**会议**: ICML 2025  
**arXiv**: [2507.10792](https://arxiv.org/abs/2507.10792)  
**代码**: [GitHub](https://github.com/511205787/Phy_SSM-ICML2025)  
**领域**: 人类理解  
**关键词**: 状态空间模型, 部分物理知识, 长期动力学预测, 不规则采样, 物理增强机器学习

## 一句话总结

提出 Phy-SSM，将部分已知的物理知识融入深度状态空间模型（SSM），通过动力学分解（已知/未知矩阵）和物理状态正则化，实现对噪声大、不规则采样数据的长期动力学精准预测与外推。

## 研究背景与动机

现实中的动力学系统（自动驾驶、流行病传播、气候科学等）通常受物理法则支配，但完整的物理方程往往难以获取。已有方法面临的核心困境：

**物理知识不完整**：传统 PINN 或 Hamiltonian 网络假设物理定律完全已知，但复杂系统（如车辆在恶劣天气下的运行）的完整动力学方程无法用第一性原理推导

**数据质量差**：传感器故障、时钟不同步导致数据含噪且不规则采样

**长期外推能力弱**：已有部分物理增强方法（如 GOKU、PI-VAE）基于 Neural ODE，严重依赖初始条件，缺乏动态修正机制，在长期外推任务中性能急剧下降

**现有方法局限**：SINDy Autoencoder 等方法依赖有限差分估计导数，仅适用于无噪声规则数据；基于 NODE 的方法在捕获非线性时变系统的长程相关性方面存在困难

**核心问题**：如何在物理知识仅部分可知、数据含噪且不规则采样的条件下，提升长期动力学预测的准确性与泛化能力？

## 方法详解

### 整体框架

Phy-SSM 由三个核心组件组成：

1. **顺序编码器（Sequential Encoder）**：基于简化的结构化 SSM，编码观测序列并估计隐状态的后验分布
2. **Phy-SSM 单元（核心）**：将部分已知的物理动力学分解为已知和未知的状态矩阵，结合 HiPPO 记忆机制建模未知动力学
3. **解码器（Decoder）**：把物理隐状态映射为最终输出

工作流程：编码器从含噪不规则观测中提取隐状态后验 → Phy-SSM 单元利用上一时刻隐状态和控制输入预测下一时刻的物理一致隐状态 → 解码器输出预测轨迹。

### 关键设计

#### 1. 动力学分解（Dynamics Decomposition）

这是本文最核心的创新。将系统动力学 $f(z, u)$ 分解为已知部分 $f_{\text{knw}}$ 和未知部分 $f_{\text{unk}}$：

$$\frac{dz(t)}{dt} = f_{\text{knw}}(z(t), u(t)) + f_{\text{unk}}(z(t), u(t))$$

进一步，通过**状态扩展**（引入非线性项 $\psi(z)$）将非线性系统转化为线性 SSM 形式：

$$\frac{d\bar{z}(t)}{dt} = (A_{\text{knw}}(t) + A_{\text{unk}}(t))\bar{z}(t) + B_{\text{unk}}(t)u(t)$$

其中 $\bar{z} = [z^\top, \psi(z)^\top]^\top$ 是扩展状态，$A_{\text{knw}}$ 编码已知物理、$A_{\text{unk}}$ 由网络学习。这种分解的好处是：（i）利用矩阵运算提升效率；（ii）直接将物理知识嵌入矩阵结构中。

#### 2. 知识掩码机制（Knowledge Mask）

引入二值掩码 $M \in \{0,1\}^{d_{\bar{z}} \times d_{\bar{z}}}$ 对网络学到的未知矩阵施加**硬约束**：

$$A_{\text{unk}}(t) = M_A \odot \tilde{A}_{\text{unk}}(t)$$

掩码中 1 表示对应位置允许学习（未知动力学），0 表示阻断更新（已知/不相关部分）。以摆系统为例，只有描述角加速度的第二行需要学习，其余行全部被掩码屏蔽为零。这保证了物理约束作为硬约束被严格执行。

#### 3. 结构化 SSM 建模未知动力学

使用多层结构化 SSM（基于 S5 架构）以隐状态 $z$ 和控制输入 $u$ 为输入，近似学习未知连续函数 $\tilde{A}_{\text{unk}}(t)$ 和 $\tilde{B}_{\text{unk}}(t)$。关键优势：

- **HiPPO 记忆机制**：高效保留历史信息，捕获长程依赖
- **连续时间建模**：天然支持不规则采样数据
- **并行扫描训练**：通过 parallel scan 加速训练

#### 4. 编码器的后验估计

编码器是简化的结构化 SSM，引入记忆变量 $h(t)$ 捕获长序列相关性。每个时刻的隐状态后验取决于当前观测和前一时刻记忆：

$$z(t_i) \mid x(t_i) \sim \mathcal{N}(\hat{\mu}_z(t_i), \text{diag}(\hat{\sigma}_z^2(t_i)))$$

这种设计使插值阶段中，模型能基于后续观测动态修正预测，而非完全依赖初始条件。

#### 5. 物理状态正则化

先验分布的均值和方差由 Phy-SSM 单元生成，直接嵌入物理知识。在插值阶段利用后验动态修正；在外推阶段通过自回归预测。

#### 6. 理论保证——分解唯一性

**命题 1**：在 $A_{\text{knw}}$ 和 $A_{\text{unk}}$ 的支撑集不重叠（由知识掩码保证）的条件下，最小化目标函数时动力学分解的解是唯一的。这保证了训练过程中已知和未知部分不会相互干扰。

### 损失函数 / 训练策略

整体目标函数由两部分组成：

$$\mathcal{L} = \mathcal{L}_{\text{VAE}} + \lambda \mathcal{L}_{\text{reg}}$$

- **VAE 损失** $\mathcal{L}_{\text{VAE}}$：包含重建损失 $\mathcal{L}_{\text{recon}}$（观测重建质量）和 KL 散度 $\mathcal{L}_{\text{KL}}$（先验与后验分布距离），权重系数 $\beta$ 控制 KL 项强度
- **物理状态正则化** $\mathcal{L}_{\text{reg}}$：先验采样 $z(t_i)$ 与后验采样 $z^*(t_i)$ 之间的欧氏距离惩罚，权重系数 $\lambda$ 控制

正则化项具有双重作用：（1）约束编码器输出遵循物理动力学；（2）引导 Phy-SSM 单元学习更准确的、与全局轨迹一致的未知动力学，显著提升外推性能。

## 实验关键数据

### 主实验

在三个真实应用上评估：无人机状态预测（高频不规则采样 573-1915 Hz）、COVID-19 疫情预测（10% 数据缺失）、车辆运动预测（nuScenes 数据集）。

**无人机状态预测（Drone）**

| 方法 | 插值 MAE(×10⁻¹)↓ | 插值 MSE(×10⁻¹)↓ | 外推 MAE(×10⁻¹)↓ | 外推 MSE(×10⁻¹)↓ |
|------|-----|-----|-----|-----|
| S5 | 1.059 | 0.309 | 8.426 | 17.333 |
| ContiFormer | 1.446 | 0.374 | 4.059 | 5.092 |
| GOKU | 3.293 | 2.738 | 3.456 | 3.130 |
| **Phy-SSM (Ours)** | **1.002** | **0.222** | **2.733** | **1.798** |

**COVID-19 疫情预测**

| 方法 | 插值 MAE(×10⁻¹)↓ | 插值 MSE(×10⁻²)↓ | 外推 MAE(×10⁻¹)↓ | 外推 MSE(×10⁻¹)↓ |
|------|-----|-----|-----|-----|
| S5 | 0.861 | 1.057 | 5.212 | 4.560 |
| ContiFormer | 0.830 | 1.059 | 6.882 | 9.147 |
| GOKU | 1.019 | 1.667 | 6.140 | 7.918 |
| **Phy-SSM (Ours)** | **0.795** | **1.032** | **1.998** | **0.692** |

**车辆运动预测（nuScenes，域外外推）**

| 方法 | ADE↓ | FDE↓ | Speed Error↓ | Accel Error(×10¹)↓ |
|------|------|------|------|------|
| Wayformer | 8.842 | 8.810 | 46.233 | 76.267 |
| SDVAE | 7.050 | 8.235 | 2.689 | 2.065 |
| PIVAE | 7.569 | 8.519 | 2.381 | 2.081 |
| **Phy-SSM (Ours)** | **6.206** | **7.197** | **2.398** | **2.043** |

### 消融实验

| 配置 | 外推 MAE(×10⁻¹)↓ | 外推 MSE(×10⁻¹)↓ | 说明 |
|------|-----|-----|------|
| 无 Phy-SSM 单元 + 无正则化（纯数据驱动 SSM） | 8.426 | 17.333 | 外推严重退化 |
| 有 Phy-SSM 单元 + 无正则化 | 3.008 | 2.176 | 物理嵌入显著提升外推 |
| **完整模型（Phy-SSM 单元 + 正则化）** | **2.733** | **1.798** | 正则化进一步约束隐状态 |

### 关键发现

1. **外推性能差距巨大**：纯数据驱动 S5 在外推 MSE 上达 17.333，Phy-SSM 仅 1.798，降低近 10 倍
2. **物理增强方法全面优于数据驱动方法**：在域外车辆预测中，所有 PEML 方法在 Speed/Accel/Jerk 指标上远优于 Wayformer 等数据驱动 SOTA
3. **正则化的双刃剑效应**：移除正则化后插值性能略有提升（过拟合观测），但外推显著退化——正则化引导模型学习泛化的物理动力学而非过拟合训练数据
4. **COVID-19 外推提升最为显著**：外推 MSE 从次优 S5 的 4.560 降至 0.692，降幅超 85%，体现了对时变动力学的建模能力

## 亮点与洞察

1. **动力学分解 + 知识掩码**：以简洁优雅的方式将部分物理知识作为硬约束嵌入 SSM，避免了 PINN 式软约束的松散性，理论上保证分解唯一性
2. **SSM 替代 NODE 的合理性**：针对 NODE 依赖初始条件、长期预测退化的本质缺陷，选用 SSM 的 HiPPO 记忆和连续时间特性来捕获长程依赖
3. **通用性框架**：同一方法在自动驾驶（运动学）、流行病学（SIR 模型）、无人机控制（刚体动力学）三个差异极大的领域均有效，展现了框架的通用性
4. **物理状态正则化**：巧妙地在 VAE 框架中约束先验-后验对齐，使物理一致性从模型架构延伸到训练目标
5. **摆系统 Walk-Through**：通过具体的摆系统实例清晰展示了状态扩展、矩阵分解、掩码设计的完整流程，极具工程参考价值

## 局限与展望

1. **需要领域专家手动设计**：$A_{\text{knw}}$ 矩阵和知识掩码 $M$ 需要根据具体系统手动构造，自动化程度不足，迁移到新领域时需要物理建模知识
2. **线性化假设**：通过状态扩展将非线性转为线性 SSM，但对高度非线性系统（如湍流）可能需要极高维扩展状态
3. **评估指标有限**：外推长度为 60-200 步，更长时间尺度的性能尚未验证
4. **计算开销未充分讨论**：多层结构化 SSM + 矩阵分解的计算复杂度相比简单 S5 或 Mamba 的增量未给出
5. **可拓展方向**：可尝试自动发现部分物理结构（结合 SINDy 思想）、引入注意力机制增强时变 $A_{\text{unk}}$ 的表达能力、探索多智能体交互场景

## 相关工作与启发

- **PINN/HNN/LNN 系列**：完全依赖已知物理，本文放松到部分已知，更贴近现实
- **GOKU / PI-VAE**：同属部分物理增强 VAE，但基于 NODE 建模未知动力学，长期外推弱；本文用 SSM 替代 NODE 是关键改进
- **S4/S5/Mamba**：纯数据驱动 SSM 缺乏物理归纳偏置，外推能力有限；本文首次在 SSM 中嵌入部分物理知识
- **SINDy Autoencoder**：从数据中自动发现物理定律，但依赖有限差分、不适用于噪声数据；与本文互补——未来可结合 SINDy 自动发现 + Phy-SSM 预测

**对自身研究的启发**：动力学分解 + 掩码的思路可泛化为任何"部分知识嵌入"场景——只要能把问题拆分为已知结构和未知部分，就可用类似框架。这在人体运动预测、社交行为建模等人类理解任务中有直接应用前景。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | SSM + 部分物理知识分解是新颖组合，掩码机制简洁有效 |
| 理论深度 | 4 | 提供了分解唯一性的理论分析 |
| 实验充分性 | 4.5 | 三个真实场景 + 完善的消融 + 敏感性分析 |
| 写作质量 | 4 | Walk-Through 示例出色，整体结构清晰 |
| 实用价值 | 3.5 | 需要手动构造物理矩阵，工程落地门槛较高 |
| **综合** | **4** | 扎实的工作，在物理增强深度学习方向有明确贡献 |

<!-- RELATED:START -->

## 相关论文

- [XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](../../AAAI2026/human_understanding/xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)
- [Homogeneous Dynamics Space for Heterogeneous Humans](../../CVPR2025/human_understanding/homogeneous_dynamics_space_for_heterogeneous_humans.md)
- [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation](../../CVPR2025/human_understanding/learning_phase_distortion_with_selective_state_space_models_for_video_turbulence.md)
- [Reactivation: Empirical NTK Dynamics Under Task Shifts](reactivation_empirical_ntk_dynamics_under_task_shifts.md)

<!-- RELATED:END -->
