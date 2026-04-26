# Information Shapes Koopman Representation

**会议**: ICLR 2026  
**arXiv**: [2510.13025](https://arxiv.org/abs/2510.13025)  
**代码**: https://github.com/Wenxuan52/InformationKoopman  
**领域**: 模型压缩  
**关键词**: koopman operator, information bottleneck, dynamical systems, representation learning, von neumann entropy

## 总结

本文从信息瓶颈（Information Bottleneck, IB）的视角重新审视 Koopman 算子的有限维表示学习问题。Koopman 算子将非线性动力系统提升为无穷维线性演化，但实际应用需要在有限维子空间中近似，导致表示学习面临"简洁性 vs 表达力"的根本矛盾。作者证明：(1) 潜在互信息控制预测误差的上界，但过度最大化会导致模态坍塌（mode collapse）；(2) von Neumann 熵可防止坍塌并保持有效维度。基于此，提出了一个信息论 Lagrangian 公式，统一平衡时间一致性（temporal coherence）、预测充分性（predictive sufficiency）和结构一致性（structural consistency）三大目标，并推导出可计算的损失函数。在物理仿真、视觉控制和图结构动力学三类任务上均优于现有 Koopman 方法。

## 动机

1. **Koopman 算子的无穷维困境**：Koopman 算子理论上可将非线性动态线性化，但其无穷维特性使得在深度网络中寻找合适的有限维子空间极为困难，现有方法常出现不稳定或模态坍塌
2. **缺乏通用的表示学习原则**：已有工作依赖领域先验（对称性、守恒律等）来约束 Koopman 表示，但缺少一般性的指导原则来平衡简洁性与表达力
3. **信息瓶颈视角的自然适配**：IB 框架天然适合描述"压缩输入同时保留预测信息"的权衡，但标准 IB 未考虑动态系统的线性演化约束
4. **潜在空间的线性约束更严格**：与 VAE 不同，Koopman 学习要求潜在空间不仅编码当前状态，还需支持线性前向传播，对表示施加了更强的结构约束
5. **简单增加维度无法解决问题**：先前研究表明，盲目增大潜在空间维度并不能提升性能，反而可能破坏时间一致性
6. **误差在自回归预测中累积放大**：Koopman 表示中的微小偏差会随时间步传播和放大，需要理论工具来量化和控制这种累积误差

## 方法

### 信息流分析

作者首先建立 Koopman 表示的概率视角。给定初始状态 $x_0$，Koopman 表示诱导的轨迹分布为：

$$p^{KR}(x_{1:t}|x_0) = \int p(z_0|x_0) \prod_{n=1}^{t} p(z_n|z_{n-1}) p(x_n|z_n) dz_{0:t}$$

其中编码器 $p(z_0|x_0)$ 将状态映射到潜在空间，线性高斯转移 $p(z_n|z_{n-1}) = \mathcal{N}(z_n|\mathcal{K}z_{n-1}, \Sigma)$ 实现 Koopman 演化，解码器 $p(x_n|z_n)$ 重建状态。

### 自回归误差界

核心理论贡献是证明了真实轨迹与 Koopman 轨迹之间的差异由逐步信息损失控制：

$$\|p(x_{1:t}|x_0) - q^{KR}(x_{1:t}|x_0)\|_{TV} \leq \sqrt{\frac{1}{2}\sum_{n=1}^{t}(I(x_{n-1};x_n) - I(z_{n-1};z_n)) + \mathcal{E}}$$

信息间隙 $I(x_{n-1};x_n) - I(z_{n-1};z_n)$ 直接度量了 Koopman 近似丢失的动态耦合信息。

### 信息分解与谱性质

将互信息 $I(z_t; x_t)$ 分解为三个成分：
- **时间一致信息** $I(z_{t-n}; z_t)$：对应特征值 $|\lambda| \approx 1$ 的 Koopman 模态，代表可长期保持的信息
- **快速耗散信息** $I(z_t; x_{t-1}|z_{t-n})$：对应 $|\lambda| < 1$ 的模态，随时间指数衰减
- **残差信息** $I(z_t; x_t|x_{t-1})$：无谱对应，属于噪声等不可预测成分，可压缩

### 信息论 Lagrangian

基于以上分析，提出统一优化目标：

$$\max_z \alpha \log I(z_{t-n};z_t) - \beta I(z_t;x_t|z_{t-n}) + \gamma S\left(\frac{\mathcal{C}}{\text{tr}(\mathcal{C})}\right) + \log p(x_t|z_t)$$

其中 $\alpha$ 项保持时间一致性，$\beta$ 项压缩耗散和残差成分，$\gamma$ 项通过 von Neumann 熵 $S(\cdot)$ 防止模态坍塌保持有效维度，最后一项为重建损失。

### 可计算损失函数

将 Lagrangian 转化为实用损失：时间一致信息通过闭式互信息或 InfoNCE 计算；结构一致性通过 $\mathbb{E}_{p_\theta(z_n|x_n)}[\log q_\psi(z_n|z_{n-1})]$（即 Koopman 线性转移的似然）实现；von Neumann 熵从小批量的归一化协方差矩阵计算。整个框架与架构无关，可适配 VAE 和 AE 两种结构。

## 实验

### 表1：物理仿真任务性能对比（NRMSE ↓ / SSIM ↑ / SDE ↓）

| 任务 | 指标 | VAE | KAE | KKR | PFNN | **Ours** |
|------|------|-----|-----|-----|------|----------|
| Lorenz 63 | 5-NRMSE | 0.005 | 0.006 | 0.004 | 0.005 | **0.003** |
| Lorenz 63 | 50-NRMSE | 0.019 | 0.023 | 0.017 | 0.017 | **0.013** |
| Lorenz 63 | KLD | 1.047 | 0.464 | 0.342 | 0.293 | **0.285** |
| Kármán Vortex | 5-NRMSE | 0.127 | 0.149 | 0.114 | 0.075 | **0.068** |
| Kármán Vortex | 5-SSIM | 0.743 | 0.719 | 0.868 | 0.920 | **0.936** |
| Kármán Vortex | SDE | 0.538 | 0.620 | 0.799 | 0.278 | **0.256** |
| Dam Flow | 50-NRMSE | 0.034 | 0.046 | 0.031 | – | **0.026** |
| Dam Flow | SDE | 0.563 | 0.488 | 0.373 | – | **0.244** |
| ERA5 Weather | 5-NRMSE | – | 0.055 | 0.058 | 0.049 | **0.028** |
| ERA5 Weather | 5-SSIM | – | 0.666 | 0.664 | 0.697 | **0.867** |

### 表2：消融实验——各正则项对 Pendulum 流形学习的影响

| 配置 | 时间一致性 (α) | 结构一致性 (β) | von Neumann 熵 (γ) | 流形质量 |
|------|:-:|:-:|:-:|------|
| 完整模型 | ✓ | ✓ | ✓ | 最接近真实 $\mathcal{S}^1 \times \mathbb{R}$ |
| α=0 | ✗ | ✓ | ✓ | 退化为散点，无几何结构 |
| β=0 | ✓ | ✗ | ✓ | 流形坍塌，丧失动力学结构 |
| γ=0 | ✓ | ✓ | ✗ | 保留 $\mathcal{S}^1$ 但丢失 $\mathbb{R}$ 维度 |
| 仅增大 α | ↑↑ | ✓ | ✗ | 表示集中于 $\mathcal{S}^1$ 分量 |
| α + γ | ✓ | ✓ | ✓ | 恢复完整 $\mathcal{S}^1 \times \mathbb{R}$ |

## 亮点

- **理论深度突出**：首次建立了 Koopman 表示的信息论框架，将互信息与自回归误差界、谱性质严格关联，揭示了 MI 促进简洁性但可能导致模态坍塌、von Neumann 熵维持表达力的对偶关系
- **信息分解具有洞察力**：将潜在信息分解为时间一致/快速耗散/残差三个成分并与 Koopman 特征值对应，为理解动力系统表示提供了新的分析工具
- **架构无关的通用框架**：所提 Lagrangian 可适配 VAE/AE 结构，且在物理仿真、视觉控制、图结构动力学三类不同任务上均有效
- **消融实验直观有力**：通过 Pendulum 流形可视化清晰展示了每个正则项的作用，理论预测与实验观察完美吻合

## 局限

- **计算开销未充分讨论**：von Neumann 熵需要计算协方差矩阵的特征分解，在高维潜在空间中可能成为瓶颈
- **超参数调节依赖经验**：Lagrangian 乘子 $\alpha, \beta, \gamma$ 的选择对性能有显著影响，但论文未提供系统的选择指南
- **实验规模相对有限**：物理仿真任务维度适中（最大 64×64×2），未在更大规模或更复杂的实际系统上验证
- **线性 Koopman 假设的局限**：框架本质假设潜在演化为线性，对强非线性或混沌系统（如湍流）的适用边界未深入分析
- **缺乏与现代基础模型的对比**：未与基于 Transformer 的时序预测方法（如 FourCastNet）比较

## 相关工作

- **Koopman 算子学习**：KAE (Pan et al., 2023) 是经典的 Koopman 自编码器；KKR (Bevanda et al., 2023) 基于核方法；PFNN (Cheng et al., 2025) 针对混沌系统设计了 Poincaré 流结构——本文从信息论角度统一和超越了这些方法
- **信息瓶颈方法**：Tishby et al. (2000) 提出经典 IB 框架；β-VAE (Burgess et al., 2018) 将 IB 引入变分自编码器——本文将 IB 扩展到动力系统的时序 Koopman 表示
- **动力系统表示学习**：E2C (Banijamali et al., 2019) 和 PCC (Levine et al., 2020) 从 VAE 出发学习可控表示——本文通过信息论约束获得更好的流形结构
- **谱分析与有效维度**：von Neumann 熵在量子信息中用于度量纠缠——本文创新性地将其引入 Koopman 表示防止模态坍塌

## 评分

| 维度 | 分数 (1-10) |
|------|:-----------:|
| 新颖性 | 8 |
| 理论深度 | 9 |
| 实验充分性 | 7 |
| 写作质量 | 8 |
| 实用价值 | 7 |
| **总分** | **7.8** |

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](../../NeurIPS2025/interpretability/shap_values_via_sparse_fourier_representation.md)

<!-- RELATED:END -->
