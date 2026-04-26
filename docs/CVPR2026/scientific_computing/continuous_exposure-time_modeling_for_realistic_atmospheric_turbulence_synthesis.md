---
title: >-
  [论文解读] Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis
description: >-
  [CVPR 2026][科学计算][大气湍流合成] 提出曝光时间依赖的调制传递函数（ET-MTF），将曝光时间建模为连续变量，构建了大规模合成湍流数据集 ET-Turb（5083视频、200万帧），显著提升湍流复原模型在真实数据上的泛化能力。
tags:
  - CVPR 2026
  - 科学计算
  - 大气湍流合成
  - 曝光时间建模
  - 调制传递函数(MTF)
  - 点扩散函数(PSF)
  - 湍流图像复原
---

# Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis

**会议**: CVPR 2026  
**arXiv**: [2603.01398](https://arxiv.org/abs/2603.01398)  
**代码**: [有](https://github.com/Jun-Wei-Zeng/ET-Turb)  
**领域**: 科学计算  
**关键词**: 大气湍流合成, 曝光时间建模, 调制传递函数(MTF), 点扩散函数(PSF), 湍流图像复原  

## 一句话总结

提出曝光时间依赖的调制传递函数（ET-MTF），将曝光时间建模为连续变量，构建了大规模合成湍流数据集 ET-Turb（5083视频、200万帧），显著提升湍流复原模型在真实数据上的泛化能力。

## 研究背景与动机

大气湍流通过折射率的随机波动，对远距离成像引入几何扭曲（tilt）和曝光时间相关的模糊（blur），严重影响遥感、视频监控、天文观测等应用。学习方法的性能高度依赖训练数据的真实性，而获取大规模配对真实湍流数据极其昂贵，因此合成数据集至关重要。

现有合成方法的核心缺陷在于对曝光时间的处理过于粗糙：

- **固定曝光方法**：大量方法对所有样本使用单一曝光设置，导致模糊统计特性单一，无法反映真实成像中的时间变化性
- **二值曝光方法**：部分方法仅区分"短曝光"和"长曝光"两种模式，使用对应的 $\text{MTF}_{\text{SE}}$ 和 $\text{MTF}_{\text{LE}}$，忽略了中间曝光时间产生的平滑过渡
- **物理仿真方法**：气体灶等物理装置受限于短光路，多步相位屏方法计算开销巨大

这些限制导致合成数据与真实湍流存在显著域差距，训练出的模型泛化能力受限。

## 方法详解

### 整体框架

论文将湍流退化建模为：$I(\mathbf{x}) = \mathcal{B}_\tau(\mathcal{T}(J(\mathbf{x})))$，其中 $\mathcal{T}$ 为几何扭曲算子（与曝光时间无关），$\mathcal{B}_\tau$ 为曝光时间依赖的模糊算子。整个合成流程分三步：

1. **ET-MTF 推导**：从 Azoulay 理论出发，推导连续曝光时间依赖的调制传递函数
2. **PSF 导出**：从 ET-MTF 得到去除 tilt 影响的纯模糊点扩散函数
3. **模糊宽度场**：将标量模糊宽度扩展为空间变化场，结合光学湍流统计约束

### 关键设计

#### 1. 曝光时间依赖的 MTF（ET-MTF）

**功能**：建立从短曝光到长曝光的连续平滑 MTF 模型。

**核心思路**：基于 Azoulay 的有限曝光 MTF 理论，引入有效相干长度 $\rho_p(\tau)$ 概念。短曝光时湍流在物理口径 $D$ 内冻结，长曝光时传感器积分多个湍流状态，等效于更大的口径 $D + v_w \tau$：

$$\text{MTF}_{\text{ET}}(\boldsymbol{\xi}, \tau) = e^{-\left(\frac{\lambda \|\boldsymbol{\xi}\|}{\rho_p(\tau)}\right)^{5/3}}$$

$$\rho_p(\tau) = 1 + 0.35 \left(\frac{r_0}{D + v_w \tau}\right)^{1/3}$$

其中 $r_0$ 是 Fried 参数，$v_w$ 是风速。随着 $\tau$ 增大，$\rho_p(\tau)$ 平滑减小，MTF 在高频段衰减加快，自然产生从弱模糊到强模糊的连续过渡。

**设计动机**：现有 $\text{MTF}_{\text{SE}}$ 和 $\text{MTF}_{\text{LE}}$ 仅定义了两个极端状态，中间过渡无物理建模。直接经验插值缺乏物理可解释性。

#### 2. 模糊宽度重参数化

**功能**：将 ET-MTF 从仅依赖曝光时间扩展到同时依赖局部模糊宽度。

**核心思路**：利用 PSF 的半高全宽（FWHM）定义模糊宽度 $\omega \approx \frac{0.49 \lambda f}{r_0}$，将 $r_0$ 反解代入有效相干长度：

$$\rho_p(\omega, \tau) = 1 + 0.28 \left(\frac{\lambda f}{\omega(D + v_w \tau)}\right)^{1/3}$$

最终 ET-MTF 同时由空间位置（通过 $\omega$）和时间（通过 $\tau$）共同决定。

**设计动机**：原始 $\rho_p(\tau)$ 在图像平面上空间均匀，但真实湍流因局部折射率波动表现为空间变化的模糊模式。

#### 3. 空间变化模糊宽度场

**功能**：为每个空间位置分配不同的模糊宽度，实现空间非均匀模糊建模。

**核心思路**：将模糊宽度建模为空间相关的随机场 $\mathcal{W}(\mathbf{x}, \tau)$，其均值和标准差由光学湍流理论约束：

$$\mathcal{W}(\mathbf{x}, \tau) = \max(\epsilon, \bar{\omega}(\tau) + \sigma_\omega(\tau) \mathcal{R}(\mathbf{x}))$$

其中 $\bar{\omega}(\tau)$ 和 $\sigma_\omega(\tau)$ 均是 $\tau$ 的函数（由详细的物理公式给出），$\mathcal{R}(\mathbf{x})$ 是经低通滤波的零均值单位方差高斯随机场，$\epsilon > 0$ 保证非负性。

最终的空间变化模糊操作为：

$$\mathcal{B}_\tau(I_T(\mathbf{x})) = \text{PSF}_{\text{ET}}(\mathbf{x}, \mathcal{W}(\mathbf{x}, \tau), \tau) * I_T(\mathbf{x})$$

#### 4. 帧间相关性建模

**功能**：将单帧合成扩展到视频序列，建模湍流退化的时间演化。

**核心思路**：采用 Taylor 冻结流假设，将湍流视为被平均风平移的准静态折射率场：

$$\mathcal{H}(J_t(\mathbf{x})) = \mathcal{H}\left(J_0\left(\mathbf{x} - \frac{f \mathbf{v}_w t}{L}\right)\right)$$

在扩展的退化场上沿风向平移即可生成时间相关的视频帧。

### 损失函数 / 训练策略

本文核心贡献在于数据集构建而非网络训练。ET-Turb 数据集设计了 12 种湍流配置，系统性覆盖不同光学和大气条件：

- **参数空间**：传播距离 30-1000m、焦距 0.1-1m、F 数 2.8-24、$C_n^2$ 范围 $0.5 \times 10^{-14}$ 到 $300 \times 10^{-14}$ m$^{-2/3}$、风速 1-10 m/s、曝光时间 0.5-40ms
- **数据规模**：5,083 个视频，2,005,835 帧，分为 3,988 训练 / 1,095 测试
- **真实数据集**：ET-Turb-Real 包含 74 个视频，来自 3 种不同成像设备

## 实验关键数据

### 主实验

在真实湍流数据上评估不同合成数据集训练的模型（无参考指标，越低越好）：

| 训练数据集 | TSR-WGAN NIQE↓ | TSR-WGAN BRISQUE↓ | TMT NIQE↓ | TMT BRISQUE↓ | DATUM NIQE↓ | DATUM BRISQUE↓ | MambaTM NIQE↓ | MambaTM BRISQUE↓ |
|---|---|---|---|---|---|---|---|---|
| TMT-dynamic | 4.231 | 52.502 | 4.361 | 58.581 | 4.219 | 54.921 | 4.217 | 55.062 |
| ATSyn-dynamic | 4.224 | 54.462 | 4.483 | 59.707 | 4.308 | 59.126 | 4.247 | 56.876 |
| **ET-Turb** | **4.190** | **50.981** | **4.221** | **56.691** | **4.204** | **54.070** | **4.212** | **55.050** |

ET-Turb 在全部 4 个模型 × 2 个指标共 8 项评测中取得 7 项最优。

### 消融实验

不同曝光建模策略的对比（MambaTM 模型）：

| 曝光策略 | NIQE↓ | BRISQUE↓ |
|---|---|---|
| 固定曝光 τ=1ms | 4.355 | 55.457 |
| 二值 MTF_SE/LE | 4.297 | 55.123 |
| **连续 ET-MTF** | **4.212** | **55.050** |

### 关键发现

1. **固定曝光训练的模型**难以恢复强模糊，因为训练数据中缺乏曝光变化
2. **二值 MTF 模型**有所改善但仍存在残余模糊，说明其对中间曝光范围覆盖不足
3. **连续 ET-MTF** 产生最自然、视觉一致的复原效果，证明连续建模的关键作用
4. ET-Turb 训练的模型在零样本迁移到真实数据时，避免了其他数据集训练模型常见的建筑文字变形、远处电线杆失真等伪影

## 亮点与洞察

1. **物理建模的简洁优雅**：通过"有效口径 = 物理口径 + 风速×曝光时间"这一直觉概念，自然地桥接了短/长曝光 MTF，物理意义清晰
2. **重参数化技巧**：将 Fried 参数 $r_0$ 替换为模糊宽度 $\omega$，巧妙引入空间变化性
3. **数据集设计思路**：12 种配置 × 7 个物理参数的系统化采样，比随机采样更能覆盖真实场景的多样性
4. **评估设计合理**：使用无参考指标在真实数据上评估，避免了合成数据测合成数据的循环论证

## 局限与展望

1. **Taylor 冻结流假设**的有效性受限于短曝光时间尺度，对极端条件可能失效
2. 仅考虑了各向同性湍流模型，真实大气（尤其近地面）可能呈各向异性
3. 合成数据仅包含模糊和几何扭曲，未建模散射、色散等其他大气效应
4. 曝光时间限制在 0.5-40ms，超长曝光场景（如天文观测）可能需要不同建模
5. 可结合可学习的曝光时间调度策略，做端到端的退化感知训练

## 评分

⭐⭐⭐⭐ 4/5

在湍流合成这个相对窄的领域做出了扎实的物理建模贡献。ET-MTF 的推导有清晰的物理根基，数据集设计周全，实验评估充分（4个SOTA模型×3个数据集的交叉验证）。扣分点在于这是一个数据集/仿真工具论文，缺少模型架构创新；此外消融实验中指标提升幅度有限（NIQE 从 4.297→4.212），虽然视觉效果差异更明显。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)
- [\[CVPR 2026\] EHETM: High-Quality and Efficient Turbulence Mitigation with Events](high-quality_and_efficient_turbulence_mitigation_with_events.md)
- [\[NeurIPS 2025\] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](../../NeurIPS2025/scientific_computing/gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)
- [\[NeurIPS 2025\] EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](../../NeurIPS2025/scientific_computing/eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)
- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](../../NeurIPS2025/scientific_computing/physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)

<!-- RELATED:END -->
