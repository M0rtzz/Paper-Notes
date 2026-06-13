---
title: >-
  [论文解读] Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions
description: >-
  [ECCV 2024][3D视觉][事件相机] 提出 Deblur e-NeRF，通过物理精确的像素带宽模型来建模事件相机的运动模糊，首次实现从运动模糊的事件流中直接有效地重建无模糊 NeRF。 领域现状：事件相机因其高时间分辨率、高动态范围和低延迟等特性，在高速运动和低光照场景中具有优势。近年来，从事件流重建 NeRF（如…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "事件相机"
  - "NeRF"
  - "运动模糊"
  - "像素带宽模型"
  - "低光照"
---

# Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions

**会议**: ECCV 2024  
**arXiv**: [2409.17988](https://arxiv.org/abs/2409.17988)  
**代码**: [https://github.com/wengflow/deblur-e-nerf](https://github.com/wengflow/deblur-e-nerf)  
**领域**: 3D视觉  
**关键词**: 事件相机, NeRF, 运动模糊, 像素带宽模型, 低光照

## 一句话总结

提出 Deblur e-NeRF，通过物理精确的像素带宽模型来建模事件相机的运动模糊，首次实现从运动模糊的事件流中直接有效地重建无模糊 NeRF。

## 研究背景与动机

**领域现状**：事件相机因其高时间分辨率、高动态范围和低延迟等特性，在高速运动和低光照场景中具有优势。近年来，从事件流重建 NeRF（如 Robust e-NeRF）已取得显著进展。

**现有痛点**：与普遍的认知相反，事件相机在高速或低光照条件下同样会遭受运动模糊。这是因为事件传感器像素的带宽有限，且带宽与入射光强度大致成正比。然而，现有的事件 NeRF 重建方法和事件模拟器都未考虑事件运动模糊的完整非线性行为。

**核心矛盾**：事件运动模糊的根本原因是像素模拟电路带宽有限——它限制了最小事件检测延迟、最大可检测变化频率和最大事件生成速率。在高速或低光照条件下，这一限制尤为严重。

**本文目标** 在高速运动或低光照条件下，从有运动模糊的事件流直接重建无模糊的 NeRF。

**切入角度**：作者从事件传感器的物理电路分析出发，构建了完整的像素带宽模型来精确描述事件运动模糊的非线性特性。

**核心 idea**：用物理精确的像素带宽模型在 Analysis-by-Synthesis 框架中建模事件运动模糊，从而"透过"模糊重建清晰 NeRF。

## 方法详解

### 整体框架

Deblur e-NeRF 基于 Robust e-NeRF 的 Analysis-by-Synthesis 框架，核心扩展是引入物理精确的像素带宽模型来描述事件运动模糊。训练时，从 NeRF 渲染出辐射值，通过带宽模型合成运动模糊后的有效对数辐射，再与实际事件流进行一致性优化。重建后通过改进的 translated-gamma 校正消除歧义。

### 关键设计

1. **像素带宽模型 (Pixel Bandwidth Model)**:

    - **功能**：精确建模事件传感器像素模拟电路的频率响应特性，输出运动模糊后的有效对数辐射 $\log L_{blur}$。
    - **核心思路**：将像素电路建模为单位增益的 4 阶非线性时不变 (NLTI) 低通滤波器，以状态空间形式表示：
    $\dot{\mathbf{x}}(t) = A(u(t))\mathbf{x}(t) + B(u(t))u(t), \quad \mathbf{y}(t) = C\mathbf{x}(t)$
      该滤波器由三个级联阶段组成：(a) 2 阶 NLTI 低通滤波器——建模对数光感受器的瞬态响应，其阻尼比 $\zeta$ 和固有角频率 $\omega_n$ 是输入的复杂非线性函数，带宽与 $\exp(u)=L$ 大致成正比；(b) 1 阶 LTI 低通滤波器——建模源跟随器缓冲器的响应，截止频率为常数 $\omega_{c,sf}$；(c) 1 阶 LTI 低通滤波器——建模差分放大器响应，截止频率 $\omega_{c,diff}$。
    - **设计动机**：事件模糊源于像素电路的物理限制，只有精确建模电路行为才能正确处理模糊。

2. **运动模糊有效对数辐射合成 (Motion-Blurred Log-Radiance Synthesis)**:

    - **功能**：给定 NeRF 渲染的 $\log L_{sig}$，合成像素"看到"的模糊信号 $\log L_{blur}$。
    - **核心思路**：首先将连续时间非线性模型线性化并离散化，得到离散时间 4 阶线性时变 (LTV) 滤波器：
    $\mathbf{x}[k+1] = A_d[k]\mathbf{x}[k] + B_d[k]u[k] + \tilde{B}_d[k]u[k+1]$
      然后求解其瞬态响应的数值解，将 $\mathbf{y}[k]$ 近似为过去与当前输入的加权和：$\mathbf{y}[k] \approx \sum_{i=k_0}^{k} \hat{\mathbf{w}}[i] u[i]$，其中 $\hat{\mathbf{w}}[i]$ 为归一化权重。
    - **重要采样策略**：提出从变换指数分布 $T_i \sim \text{Exp}(t_k - t_i; \omega_{c,dom,min})$ 中采样输入时间戳，近似权重函数分布，将采样集中在权重大的位置，在低光照时最为有效。

3. **阈值归一化全变差损失 (Threshold-Normalized Total Variation Loss)**:

    - **功能**：替代 Robust e-NeRF 的梯度损失，更好地正则化大面积无纹理区域。
    - **核心思路**：惩罚预测的运动模糊对数辐射在事件间隔内的变化量：
    $\ell_{tv}(\mathbf{e}) = \left|\frac{\delta \log \hat{L}_{blur}}{\bar{C}}\right|$
      其中 $\bar{C} = \frac{1}{2}(C_{-1} + C_{+1})$ 为平均对比度阈值，$\delta \log \hat{L}_{blur}$ 为两个采样时间点之间的变化。
    - **设计动机**：相比梯度损失，全变差损失对均匀区域施加更强的均匀性约束，并通过阈值归一化实现对任意阈值的泛化。

4. **Translated-Gamma 校正**:

    - **功能**：消除预测有效辐射中的未知黑电平偏移和 gamma 不准确性。
    - **核心思路**：对重建后的 $\hat{\mathbf{L}}$ 进行平移-伽马校正：$\hat{\mathbf{L}}_{sig,corr} = \mathbf{b} \odot \hat{\mathbf{L}}^a - \mathbf{c}$，通过 Levenberg-Marquardt 非线性最小二乘优化参数 $a, \mathbf{b}, \mathbf{c}$。
    - **设计动机**：事件相机仅观测辐射变化而非绝对值，且黑电平（暗电流等效辐射）未知，需要额外校正。

### 损失函数 / 训练策略

- **主重建损失**：Huber 范数变体的阈值归一化差分损失 $\ell_{diff}$，权重 $\lambda_{diff}$
- **正则化损失**：阈值归一化全变差损失 $\ell_{tv}$，权重 $\lambda_{tv}$
- 支持像素带宽模型参数 $\Omega$ 与 NeRF 参数 $\Theta$ 的联合优化
- 训练批大小仅为 baseline 的 1/8

## 实验关键数据

### 主实验（合成数据 - 不同相机速度）

| 条件 | 指标 | Deblur e-NeRF | Robust e-NeRF | 提升 |
|------|------|---------------|---------------|------|
| 速度 0.125× | PSNR↑ | 28.71 | 28.31 | +0.40 |
| 速度 1× (默认) | PSNR↑ | 28.41 | 26.11 | +2.30 |
| 速度 4× (高速) | PSNR↑ | 27.48 | 22.18 | **+5.30** |
| 无模糊上界 | PSNR↑ | 29.43 | 28.48 | +0.95 |

### 实验数据（真实序列 - EDS 数据集）

| 序列 | 指标 | Deblur e-NeRF | Robust e-NeRF | E2VID+NeRF |
|------|------|---------------|---------------|------------|
| 08_peanuts_running | PSNR↑ | **18.27** | 18.00 | 14.85 |
| 08_peanuts_running | LPIPS↓ | **0.503** | 0.507 | 0.595 |
| 11_all_characters | PSNR↑ | **16.53** | 15.91 | 13.12 |
| 11_all_characters | LPIPS↓ | **0.511** | 0.552 | 0.627 |

### 消融实验（光照强度影响）

| 光照 (lux) | 指标 | Deblur e-NeRF | Robust e-NeRF | 差距 |
|------------|------|---------------|---------------|------|
| 100,000 (强光) | PSNR↑ | 28.73 | 27.62 | +1.11 |
| 1,000 (正常) | PSNR↑ | 28.41 | 26.11 | +2.30 |
| 10 (极低光) | PSNR↑ | 28.62 | 22.72 | **+5.90** |

### 关键发现

- 运动模糊越严重（高速 / 低光照），Deblur e-NeRF 相对现有方法的优势越大，在速度 4× 时 PSNR 超过 Robust e-NeRF 5.3 dB
- 即使在极低光照（10 lux）下，本文方法的性能几乎不下降（28.62 vs 无模糊上界 29.43），而 Robust e-NeRF 严重退化
- 即使在无运动模糊的理想条件下，提出的全变差损失仍比梯度损失提升约 1 dB，证明其独立价值
- 支持像素带宽模型参数的联合优化，无需精确先验

## 亮点与洞察

- **首次揭示并解决事件相机运动模糊问题在 NeRF 重建中的影响**：打破了"事件相机不受运动模糊影响"的常见误解
- **物理精确建模**：从电路层面出发建模带宽限制，而非简单的线性近似，具有极强的物理可解释性
- **全变差损失的通用价值**：即使在无模糊场景中也能改善重建质量，说明正则化改进具有独立意义
- **开源了改进的事件模拟器和合成数据集**，为后续研究提供基础设施

## 局限与展望

- 黑电平 $L_{dark}$ 未知时无法分离信号辐射和暗电流，只能重建有效辐射
- 训练批大小受限（仅为 baseline 的 1/8），提升计算效率后可能获得更好结果
- 像素带宽模型参数依赖于特定的传感器和偏置设置，跨设备泛化性有待验证
- 真实数据实验场景有限，仅在 EDS 数据集的两个序列上测试

## 相关工作与启发

- **vs Robust e-NeRF**: 同属事件 NeRF 重建，但 Robust e-NeRF 完全忽略了事件运动模糊；本文通过带宽模型扩展了其事件生成模型，在严重模糊场景中 PSNR 提升可达 5+ dB
- **vs E2VID + NeRF**: 先将事件重建为视频帧再做 NeRF 的两阶段方法，性能远不如端到端方法
- **vs 图像去模糊 NeRF**: 图像模糊可简单建模为曝光时间内平均（LTI 低通滤波），但事件模糊是非线性且与光照相关的，需要 NLTI 滤波器建模

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从物理电路角度建模事件相机运动模糊并应用于 NeRF 重建
- 实验充分度: ⭐⭐⭐⭐ 合成和真实实验覆盖多种速度/光照条件，但真实数据规模较小
- 写作质量: ⭐⭐⭐⭐ 物理建模阐述清晰，数学推导完整
- 价值: ⭐⭐⭐⭐ 挑战了事件相机不受模糊影响的常见假设，开源代码和数据集推动领域发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Invertible Neural Warp for NeRF](invertible_neural_warp_for_nerf.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)
- [\[ECCV 2024\] Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-observations for High-Quality Sparse-View Reconstruction](deceptive-nerf3dgs_diffusion-generated_pseudo-observations_for_high-quality_spar.md)
- [\[ECCV 2024\] RoGUENeRF: A Robust Geometry-Consistent Universal Enhancer for NeRF](roguenerf_a_robust_geometry-consistent_universal_enhancer_for_nerf.md)
- [\[ECCV 2024\] The NeRFect Match: Exploring NeRF Features for Visual Localization](the_nerfect_match_exploring_nerf_features_for_visual_localization.md)

</div>

<!-- RELATED:END -->
