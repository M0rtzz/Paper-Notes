---
title: >-
  [论文解读] PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement
description: >-
  [CVPR 2026][科学计算][rPPG] 从Navier-Stokes方程出发，通过严格数学推导揭示rPPG脉搏信号遵循二阶阻尼谐振子模型，其离散解形式等价于因果卷积算子，从而为TCN架构的选择提供了第一性原理依据，设计出仅0.29M参数的PHASE-Net在多个数据集上达到SOTA。
tags:
  - CVPR 2026
  - 科学计算
  - rPPG
  - 物理信息网络
  - 时间卷积网络
  - 血流动力学
  - Navier-Stokes
  - 轻量模型
---

# PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement

**会议**: CVPR 2026  
**arXiv**: [2509.24850](https://arxiv.org/abs/2509.24850)  
**代码**: [GitHub](https://github.com/Alex036225/PhaseNet)  
**领域**: 科学计算 / 远程生理信号估计  
**关键词**: rPPG, 物理信息网络, 时间卷积网络, 血流动力学, Navier-Stokes, 轻量模型

## 一句话总结

从Navier-Stokes方程出发，通过严格数学推导揭示rPPG脉搏信号遵循二阶阻尼谐振子模型，其离散解形式等价于因果卷积算子，从而为TCN架构的选择提供了第一性原理依据，设计出仅0.29M参数的PHASE-Net在多个数据集上达到SOTA。

## 研究背景与动机

**领域现状**：远程光电容积脉搏波（rPPG）通过普通摄像头捕捉皮肤血容量微变化来提取心率等生理信号，是非接触生理监测的关键技术。深度学习方法（PhysNet、PhysFormer、RhythmMamba等）已成为主流范式。

**现有痛点**：

1. 现有深度学习模型大多是启发式设计——将rPPG视为通用时空信号处理任务，架构选择依赖经验试错
2. 缺乏物理理论基础导致模型可能过拟合数据集特定噪声模式，跨域泛化差
3. 头部运动和光照变化产生的伪影远强于真实脉搏信号，"黑箱"模型难以提供可靠性保证

**核心矛盾**：高性能深度学习模型 vs 缺乏物理可解释性和理论保证。

**本文目标** 能否从物理第一性原理出发，设计一个架构本身就是信号物理规律直接体现的rPPG模型？

**切入角度**：从Navier-Stokes方程推导血流脉搏动力学，严格证明TCN是物理正确的架构选择。

**核心 idea**：rPPG信号的物理动力学等价于因果卷积，因此TCN不是启发式选择而是物理必然。

## 方法详解

### 整体框架

视觉编码器（3个EST Block，每个含ZAS模块）提取时空特征 → 自适应空间滤波器（ASF）生成空间注意力掩码并聚合+计算时间差分 → 门控时间卷积网络（GTCN）建模长程时间动态 → rPPG波形输出。

### 关键设计

1. **物理推导链：从Navier-Stokes到TCN**

    - 出发点：Beer-Lambert定律建立像素变化ΔI(t)与皮下血容量ΔV(t)的线性关系，血管顺应性进一步将ΔV(t)与局部血压脉动z(t)关联
    - 从Navier-Stokes方程线性化 → 1D动量+连续性方程 → 消去速度变量得阻尼波动方程 $\frac{\partial^2 p'}{\partial t^2} + \alpha \frac{\partial p'}{\partial t} = c^2 \frac{\partial^2 p'}{\partial x^2}$
    - 固定观测点x₀处退化为二阶ODE（阻尼谐振子）：$\ddot{z} + \alpha \dot{z} + \omega^2 z = u(t)$
    - 半隐式Euler离散化 → LTI状态空间模型 → **Proposition 1**证明其解为因果卷积 $z_t = \sum_{m=0}^{\infty} g[m] \cdot a_{t-m}$ → **Proposition 2**证明FIR即可以任意精度ε近似IIR → TCN是该物理过程的精确计算实现
    - 意义：首次建立从血流动力学第一原理到具体网络架构的完整逻辑链

2. **Zero-FLOPs Axial Swapper (ZAS)**

    - 对feature map的后k=⌊pC⌋个通道执行块内空间转置（将H×W分成b×b块后做矩阵转置），其余通道不变
    - 关键性质：自逆性（ZAS(ZAS(X))=X保证可逆和梯度稳定）、能量守恒（‖ZAS(X)‖₂=‖X‖₂, 1-Lipschitz避免信号放大）
    - 设计动机：零FLOPs、零参数即可注入跨区域空间交互，增强远距面部区域的特征混合

3. **自适应空间滤波器 (ASF)**

    - 对每帧通过轻量卷积生成空间logit图 → spatial softmax归一化为注意力掩码Mₜ → 加权聚合空间维度得到1D特征向量zₜ
    - 同时计算一阶时间差分 $\mathbf{v}_t = \mathbf{z}_t - \mathbf{z}_{t-1}$ 编码脉搏"速度"
    - 输出 = [zₜ, vₜ] 通道拼接，既保留空间纯化的强度信息又编码短时时间变化
    - 设计动机：前额/面颊SNR高但其他区域以噪声为主 → 全局平均池化(GAP)是次优的

4. **门控时间卷积网络 (GTCN)**

    - 双路因果扩张TCN：一路tanh激活、一路sigmoid门控 → 逐元素乘法融合
    - 物理意义：实现Proposition 1&2中推导出的因果卷积运算，建模长程时间动态

### 损失函数 / 训练策略

负Pearson相关损失：$\mathcal{L}_{\text{pred}} = -\frac{\sum_t (\hat{y}_t - \bar{\hat{y}})(y_t - \bar{y})}{\sqrt{\sum_t (\hat{y}_t - \bar{\hat{y}})^2 \sum_t (y_t - \bar{y})^2}}$，直接优化预测波形与GT的形态相似性。

## 实验关键数据

### 主实验（域内评估）

| 方法 | UBFC MAE↓ | UBFC RMSE↓ | PURE MAE↓ | PURE RMSE↓ | BUAA MAE↓ | MMPD MAE↓ | 参数量 |
|------|-----------|------------|-----------|------------|-----------|-----------|--------|
| PhysNet | 2.95 | 3.67 | 2.10 | 2.60 | 10.89 | 4.80 | 大 |
| PhysFormer | 0.92 | 2.46 | 1.10 | 1.75 | 8.45 | 11.99 | 大 |
| RhythmFormer | 0.50 | 0.78 | 0.27 | 0.47 | 9.19 | 4.69 | 中 |
| Contrast-Phys+ | 0.21 | 0.80 | 0.48 | 0.98 | - | - | 中 |
| Style-rPPG | 0.17 | 0.41 | 0.39 | 0.62 | - | - | 中 |
| LST-rPPG | 0.16 | 0.57 | 0.32 | 0.62 | - | - | 中 |
| **PHASE-Net** | **0.15** | **0.53** | **0.14** | **0.35** | **5.89** | **4.78** | **0.29M** |

### 消融实验（跨域泛化，Leave-One-Out）

| 方法 | Others→U MAE↓ | Others→P MAE↓ | Others→B MAE↓ | Others→M MAE↓ |
|------|---------------|---------------|---------------|---------------|
| PhysFormer | 10.29 | 19.75 | 22.09 | 13.90 |
| RhythmFormer | 14.71 | 21.11 | 6.04 | 16.14 |
| EfficientPhys | 12.87 | 7.15 | 32.30 | 12.87 |
| **PHASE-Net** | **10.04** | **2.86** | - | - |

### 关键发现

- PURE上MAE 0.14 bpm，比RhythmFormer(0.27)减半——物理先验的归纳偏置显著提升精度
- 仅0.29M参数即达SOTA——理论严谨与极致轻量的统一
- 跨域泛化Others→PURE MAE 2.86 bpm，大幅优于PhysFormer(19.75)和RhythmFormer(21.11)——物理先验增强泛化
- BUAA/MMPD等挑战性数据集上，PhysFormer等出现负相关(R<0)，PHASE-Net仍保持正相关

## 亮点与洞察

- **首次从第一性原理推导rPPG网络架构**：从Navier-Stokes → ODE → SSM → 因果卷积 → TCN的完整数学证明链，将架构选择从经验升级为物理必然
- **ZAS零FLOPs模块**：纯排列操作即增强跨区域特征交互，自逆性和能量守恒保证训练稳定性的数学证明优雅
- **ASF的时间差分设计**：将空间聚合和时间微分统一在一个模块中，为下游物理模型提供"位置+速度"的完整状态信息
- **理论严谨+工程极简的范式**：0.29M参数说明好的归纳偏置可以大幅减少模型复杂度

## 局限与展望

- 物理推导依赖多个简化假设（层流、线性化、单点观测、弹性恢复力近似），在极端运动或非典型血管条件下假设可能不成立
- ZAS的块大小b和通道比例p需手动设定，缺少自适应机制
- 未在VIPL-HR等大规模野外数据集上验证
- 跨域泛化表格部分数据缺失(Others→B, Others→M)，不够完整

## 相关工作与启发

- **vs PhysFormer/RhythmMamba**：这些方法用Transformer/SSM建模时序，属于通用序列模型；PHASE-Net从物理角度证明因果卷积(TCN)才是rPPG任务的正确计算原语
- **vs PINN传统范式**：经典PINN将物理方程嵌入损失函数；PHASE-Net的创新在于用物理规律约束网络架构本身——"物理决定结构"而非"物理约束训练"
- 启发：这种"从PDE推导出网络结构"的方法论可推广到其他有明确物理模型的信号处理任务（如地震波、声学信号）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从第一性原理推导rPPG网络架构，方法论意义重大
- 实验充分度: ⭐⭐⭐⭐ 4个数据集域内+跨域评估，消融完整，但部分跨域数据缺失
- 写作质量: ⭐⭐⭐⭐⭐ 推导严谨、从物理到架构的逻辑链清晰流畅
- 价值: ⭐⭐⭐⭐⭐ 物理驱动架构设计范式有普适意义，0.29M参数的极致效率适合部署

<!-- RELATED:START -->

## 相关论文

- [EHETM: High-Quality and Efficient Turbulence Mitigation with Events](high-quality_and_efficient_turbulence_mitigation_with_events.md)
- [DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/scientific_computing/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/scientific_computing/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)
- [F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](../../NeurIPS2025/scientific_computing/f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)

<!-- RELATED:END -->
