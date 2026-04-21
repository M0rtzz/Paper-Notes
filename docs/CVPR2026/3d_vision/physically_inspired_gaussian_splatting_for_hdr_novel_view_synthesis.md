---
title: >-
  [论文解读] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis
description: >-
  [CVPR 2026][3D视觉][HDR新视角合成] 提出PhysHDR-GS——一个物理渲染启发的HDR新视角合成框架：将高斯颜色分解为固有反射率和可调环境光照，通过图像-曝光(IE)分支和高斯-光照(GI)分支互补捕获HDR细节，跨分支HDR一致性损失提供无GT的显式HDR监督，光照引导梯度缩放解决曝光偏差的梯度饥饿问题，在多个基准上优于HDR-GS 2.04dB且保持76FPS实时渲染。
tags:
  - CVPR 2026
  - 3D视觉
  - HDR新视角合成
  - 3DGS
  - 物理渲染启发
  - 双分支架构
  - 光照引导梯度缩放
---

# Physically Inspired Gaussian Splatting for HDR Novel View Synthesis

**会议**: CVPR 2026  
**arXiv**: [2603.28020](https://arxiv.org/abs/2603.28020)  
**代码**: https://huimin-zeng.github.io/PhysHDR-GS/  
**领域**: 3D视觉 / HDR新视角合成  
**关键词**: HDR新视角合成, 3DGS, 物理渲染启发, 双分支架构, 光照引导梯度缩放

## 一句话总结
提出PhysHDR-GS——一个物理渲染启发的HDR新视角合成框架：将高斯颜色分解为固有反射率和可调环境光照，通过图像-曝光(IE)分支和高斯-光照(GI)分支互补捕获HDR细节，跨分支HDR一致性损失提供无GT的显式HDR监督，光照引导梯度缩放解决曝光偏差的梯度饥饿问题，在多个基准上优于HDR-GS 2.04dB且保持76FPS实时渲染。

## 研究背景与动机

**领域现状**：HDR新视角合成(HDR-NVS)通过融合不同曝光的LDR视图来重建高动态范围场景。从NeRF到3DGS的演进显著加速了HDR-NVS，HDR-GS用球谐拟合HDR颜色+MLP色调映射，GaussHDR统一3D/2D色调映射并融合双分支LDR输出。

**现有痛点**：(1) **外观纠缠**：物体外观由材料属性和环境条件（直接/间接光照）共同决定，仅缩放传感器曝光时间无法分解这些因素、也无法反映光照依赖的外观变化——曝光变化 $\Delta t$ 导致全局强度变化，而环境光变化 $\Delta L_a$ 导致局部外观变化（如幸运猫铭牌处的反射）；(2) **隐式HDR监督**：HDR真值通常不可用，HDR内容的监督只能通过约束tone-mapped的LDR结果间接实现，但色调映射压缩动态范围使得异常/饱和HDR值无法被有效约束；(3) **曝光偏差梯度饥饿**：色调映射曲线在极端（过曝/欠曝）区域斜率极小，对应区域的高斯primitive累积梯度远小于正常曝光区域，难以达到densification阈值，导致这些区域表示不足。

**核心矛盾**：现有HDR-NVS方法遵循传统HDR成像管线——用曝光+色调映射模拟不同亮度级别的2D图像，但不在3D空间中建模光照，场景的环境依赖属性被忽视。

**本文目标** (1) 分解曝光和环境光照对外观的不同影响；(2) 无HDR GT情况下的显式HDR监督；(3) 极端曝光区域高斯的梯度饥饿和不足densification。

**切入角度**：从物理渲染方程出发——将高斯颜色建模为固有反射率 $H_r$ 和环境光照 $L_a$ 的函数，曝光 $t$ 和光照 $L_a$ 互补地调制动态范围。

**核心 idea**：将3DGS的颜色分解为反射率和光照，用曝光调制图像(IE)和光照调制高斯(GI)两个分支互补捕获HDR细节，跨分支一致性损失和光照引导梯度缩放解决HDR监督和梯度饥饿问题。

## 方法详解

### 整体框架
PhysHDR-GS将每个高斯的颜色分解为固有反射率 $H_r$（场景内在属性，曝光不变）和环境光照 $L_a$（可调节），通过MLP合成HDR颜色 $\mathbf{c} = g(L_a, H_r)$。基于此，框架包含两个互补分支：(1) **IE分支**：在渲染出的HDR图像上施加曝光缩放 $I_{HDR} \times t$，模拟标准相机观测；(2) **GI分支**：用光照调制器调整3D高斯的环境光照，渲染重光照HDR图像 $\hat{I}_{HDR}$，捕获光照依赖的外观变化。两分支的HDR输出经tone mapper融合为最终LDR结果。

### 关键设计

1. **物理辐射合成（IE+GI双分支）**:

    - 功能：互补地覆盖更高的动态范围
    - 核心思路：基于简化渲染方程 $L_o(\mathbf{x},\omega_o) = L_e(\mathbf{x}) + L_a(\mathbf{x}) H_r(\mathbf{x},\omega_o)$。**IE分支**：分别建模 $H_r$ 和 $L_a$，用MLP $g$ 合成高斯颜色 $\mathbf{c}=g(L_a,H_r)$，渲染HDR图像后施加曝光 $t$ 做全局缩放——覆盖不同亮度带，将中间调区域拉入相机响应范围。**GI分支**：引入光照调制器 $\hat{L}_a = \varphi(L_a, l)$，用虚拟光照 $\hat{L}_a$ 替换 $L_a$ 重新合成重光照颜色 $\hat{\mathbf{c}}=g(\hat{L}_a, H_r)$——通过局部调整辐射强度避免饱和
    - 设计动机：曝光 $t$ 提供全局动态范围控制，环境光照 $L_a$ 提供局部光照依赖变化——两者的响应模式互补，结合可覆盖更高动态范围

2. **跨分支HDR一致性损失**:

    - 功能：在无HDR真值情况下为HDR内容提供显式监督
    - 核心思路：对每个视图，将光照级别 $l$ 设为曝光 $t$（使两分支亮度可比），对IE分支的 $I_{HDR} \times t$ 和GI分支的 $\hat{I}_{HDR}$ 施加高斯模糊后计算L1一致性损失：$\mathcal{L}_{\text{cons}} = \|\mathcal{G}(I_{HDR} \times t) - \mathcal{G}(\hat{I}_{HDR})\|_1$。高斯模糊避免惩罚未对齐的细节，损失约束两分支在整体光照和低频结构上一致
    - 设计动机：仅靠LDR监督无法约束饱和/异常HDR值（被tone mapping压缩了）。两个不同建模路径（曝光vs光照）产生的HDR结果应当一致——这种自监督信号弥补了HDR GT的缺失

3. **光照引导梯度缩放**:

    - 功能：缓解过曝/欠曝区域高斯的梯度饥饿，防止表示不足
    - 核心思路：观察到高斯接收的梯度与光照偏差 $\Delta L_a = |L_a - \hat{L}_a|$ 正相关（过曝/欠曝区域光照偏差大、梯度小）。提出缩放因子 $s_a = s \cdot \sigma(|L_a - \hat{L}_a|) + 1$（$\sigma$为sigmoid，$s$为超参数），将densification判据修改为 $\mathbb{I}_i(s_a) \frac{1}{M_i}\sum_k \|\frac{\partial \mathcal{L}_k}{\partial \mu_{i,k}^{\text{ndc}}}\|_2 > \tau_p$。光照偏差越大的高斯获得越大的梯度放大，帮助其达到splitting阈值
    - 设计动机：标准3DGS的densification基于屏幕空间梯度阈值，但tone mapping曲线在极端区域斜率接近零，使得这些区域的高斯累积微小梯度、永远无法被split/clone——导致过曝/欠曝区域出现under-densified模糊。梯度缩放直接补偿了这种系统性偏差

4. **Cross-fusion色调映射器**:

    - 功能：融合IE和GI分支的LDR输出
    - 核心思路：tone mapper $f$ 包含两个轻量MLP——$f_{tm}$ 对每个HDR输入进行全局+局部色调映射产生两对LDR预测，$f_{mix}$ 对两对LDR做交叉融合：$I_{LDR}^{IG} = f_{mix}(I_{LDR}^{glo}, \hat{I}_{LDR}^{loc})$ 和 $I_{LDR}^{GI} = f_{mix}(I_{LDR}^{glo}, I_{LDR}^{loc})$，最终LDR = 两者相加
    - 设计动机：全局tone mapping保持整体亮度一致，局部tone mapping保留细节，交叉融合让两个分支的互补信息在LDR域中也能相互补充

### 损失函数 / 训练策略
总损失 $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{rec}} + \lambda_2 \mathcal{L}_{\text{cons}} + \lambda_3 \mathcal{L}_{\text{unit}}$，其中重建损失 $\mathcal{L}_{\text{rec}} = \gamma \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{D-SSIM}}$（$\gamma=0.2$）对三个LDR输出计算。$\lambda_1=1, \lambda_2=0.5, \lambda_3=0$（合成数据0.5）。前10k迭代冻结融合MLP仅训练tone mapping MLP。训练30k迭代，单张A6000 GPU。

## 实验关键数据

### 主实验（HDR-NeRF-Real, exp3设置）

| 方法 | LDR-OE PSNR↑ | LDR-NE PSNR↑ | LPIPS↓ |
|------|-------------|-------------|--------|
| HDR-NeRF | 34.27 | 32.15 | 0.074 |
| HDR-GS | 34.87 | 31.02 | 0.029 |
| GaussHDR | 36.05 | 33.49 | 0.017 |
| GaussHDR† | 36.32 | 33.84 | 0.014 |
| **Ours†** | **36.91** | **34.15** | **0.012** |

注：Ours†(Scaffold-GS)在LDR-OE上比GaussHDR†高0.59dB。

### 合成数据结果（HDR-NeRF-Syn, exp3设置）

| 方法 | LDR-OE PSNR↑ | LDR-NE PSNR↑ | HDR PSNR↑ |
|------|-------------|-------------|-----------|
| HDR-GS | 40.28 | 27.07 | 17.51 |
| GaussHDR† | 43.87 | 42.74 | 39.08 |
| **Ours†** | **44.26** | **43.19** | **39.21** |

### 消融实验（HDR-NeRF-Real, exp3）

| 配置 | LDR-OE PSNR | LDR-NE PSNR |
|------|-------------|-------------|
| IE branch only | 36.18 | 33.38 |
| + GI branch | 36.27 (+0.09) | 33.46 (+0.08) |
| + HDR-cons | 36.43 (+0.16) | 33.84 (+0.38) |
| + I-GS | **36.91** (+0.48) | **34.15** (+0.31) |

### 效率对比

| 方法 | 渲染(ms) | FPS | 训练(min) | 显存(MB) |
|------|---------|-----|-----------|---------|
| HDR-NeRF | 4189 | 0.24 | 500 | 11049 |
| HDR-GS | 9 | 117 | 10 | 5014 |
| GaussHDR | 19 | 53 | 28 | 5596 |
| Ours | **13** | **76** | 15 | **3274** |

### 关键发现
- **光照引导梯度缩放(I-GS)贡献最大**——单独带来0.48dB提升，说明过曝/欠曝区域的梯度饥饿确实是HDR-NVS的关键瓶颈
- **HDR一致性损失带来显著提升**——特别在novel exposure(LDR-NE)上提升0.38dB，说明无GT的HDR自监督有效弥补了tone mapping的信息损失
- **GI分支单独贡献较小但与其他组件协同效果好**——定性分析显示它主要改善了光照依赖的外观(如桌面反射)和纹理失真
- **效率优异**——Ours比GaussHDR快1.43倍(76fps vs 53fps)，显存仅3274MB(GaussHDR 5596MB)，训练时间15min
- 在LPIPS感知指标上Ours†在所有基准上均为最优，说明物理建模有助于感知质量

## 亮点与洞察
- **"曝光调图像，光照调高斯"的对偶设计**是本文核心洞察——曝光 $t$ 是2D域的全局缩放，光照 $L_a$ 是3D域的局部调制，两者互补覆盖HDR。这种从物理渲染方程自然推导出的双分支设计比之前的工程设计更有理论根基
- **梯度饥饿的发现和解决方案**具有普适价值——任何涉及非线性映射（如gamma校正、tone mapping）的3DGS优化都可能存在类似的梯度衰减问题。光照偏差作为梯度缩放的代理变量这一发现可迁移到其他场景
- **跨分支自监督**的思路——两个不同路径建模同一物理量（HDR辐射），强制它们一致提供了无GT的显式监督。这种设计可迁移到其他缺少GT的3D重建任务
- **实时高效**——76FPS+3274MB显存，比HDR-NeRF快322倍、比GaussHDR快1.43倍且显存更少

## 局限与展望
- 环境光照假设为均匀半球照明(uniform hemispherical illumination)，对方向性强光源（如点光源/聚光灯）建模不够精确
- 反射率 $H_r$ 和光照 $L_a$ 的分解依赖MLP，可能存在固有的模糊性——同一观测可由多组 $(H_r, L_a)$ 解释
- 光照调制器 $\varphi$ 是数据驱动的，对超出训练曝光范围的光照条件泛化能力有限
- 仅在多曝光静态场景上评估，对动态场景、单曝光设置的效果未知
- GI分支单独贡献有限(0.09dB)，说明光照调制的效果可能受限于训练数据中光照变化的多样性

## 相关工作与启发
- **vs HDR-GS**: HDR-GS用球谐拟合HDR颜色+MLP做曝光条件化tone mapping，不建模3D光照。PhysHDR-GS将颜色分解为反射率+光照，在3D空间中显式建模光照
- **vs GaussHDR**: GaussHDR统一3D/2D色调映射并融合双分支LDR输出，是纯工程导向设计。PhysHDR-GS从物理渲染方程出发，双分支有物理含义（曝光=2D全局 vs 光照=3D局部）
- **vs NeRF-based HDR方法**: HDR-NeRF等方法训练和推理都极慢(4189ms/frame)。PhysHDR-GS继承3DGS的效率优势(13ms/frame)

## 评分
- 新颖性: ⭐⭐⭐⭐ 从物理渲染方程推导出IE+GI双分支设计有理论优雅性，梯度饥饿的发现和光照引导缩放是有实际价值的新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准×两种曝光设置×两种backbone，消融完整，效率分析详尽
- 写作质量: ⭐⭐⭐⭐ 从物理渲染方程到方法设计的推导清晰，图表丰富直观
- 价值: ⭐⭐⭐⭐ 梯度饥饿发现和光照引导缩放对3DGS社区有普适价值，但HDR-NVS领域相对小众

<!-- RELATED:START -->

## 相关论文

- [SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](../../ICCV2025/3d_vision/sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](geodesicnvs_flow_matching_novel_view_synthesis.md)
- [Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting](hierarchical_visual_relocalization_with_nearest_view_synthesis_from_feature_gaus.md)
- [PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis](physgaia_a_physics-aware_benchmark_with_multi-body_interactions_for_dynamic_nove.md)
- [PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)

<!-- RELATED:END -->
