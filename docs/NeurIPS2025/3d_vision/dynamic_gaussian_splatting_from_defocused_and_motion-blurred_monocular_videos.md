---
title: >-
  [论文解读] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos
description: >-
  [NeurIPS 2025][3D视觉][3D Gaussian Splatting] 提出统一框架，通过可学习模糊核卷积联合建模散焦模糊和运动模糊，结合动态高斯致密化策略和未见视角约束，从模糊单目视频中实现高质量动态 3DGS 新视角合成。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 3D Gaussian Splatting
  - 动态场景重建
  - 散焦模糊
  - 运动模糊
  - 新视角合成
---

# Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos

**会议**: NeurIPS 2025  
**arXiv**: [2510.10691](https://arxiv.org/abs/2510.10691)  
**代码**: [hhhddddddd/dydeblur](https://github.com/hhhddddddd/dydeblur)  
**领域**: 3d_vision  
**关键词**: 3D Gaussian Splatting, 动态场景重建, 散焦模糊, 运动模糊, 新视角合成  

## 一句话总结

提出统一框架，通过可学习模糊核卷积联合建模散焦模糊和运动模糊，结合动态高斯致密化策略和未见视角约束，从模糊单目视频中实现高质量动态 3DGS 新视角合成。

## 背景与动机

从单目视频重建动态场景并合成新视角是 3D 视觉的重要问题。现有方法存在核心瓶颈：

1. **模糊类型割裂**：散焦模糊（defocus）和运动模糊（motion blur）的成因完全不同——前者源于景深限制，后者源于曝光时间内的相对运动。现有方法只能处理其中一种
2. **模糊核估计困难**：虽然两种模糊都可建模为核卷积，但从动态场景中准确估计逐像素模糊核极其困难
3. **动态高斯不完整**：基于追踪点初始化的动态高斯在遮挡区域存在缺失

## 核心问题

如何设计统一框架同时处理散焦模糊和运动模糊的单目视频，实现高保真清晰新视角合成？

## 方法详解

### 动态高斯表示与致密化

采用 Shape-of-Motion 框架分别建模静态和动态高斯。动态高斯通过 $\mathrm{SE}(3)$ 运动基的线性组合表示运动：

$$\mathrm{T}_{t_0 \to t} = \sum_{b=0}^{N_b} \mathbf{w}^{(b)} \mathrm{T}_{t_0 \to t}^{(b)}$$

变换后的动态高斯位置和旋转：

$$\mu_t = \mathrm{R}_{t_0 \to t} \mu_{t_0} + \mathrm{t}_{t_0 \to t}, \quad \mathrm{R}_t = \mathrm{R}_{t_0 \to t} \mathrm{R}_{t_0}$$

**动态高斯致密化 (DGD)**：仅用规范帧可见的追踪点初始化动态高斯，然后从所有观测帧的深度图重投影补充动态区域，通过前景重映射将观测帧高斯变换回规范帧：

$$\mu_{t_0}^g = (\mathrm{R}_{t_0 \to t}^{G'})^{-1} (\mu_t^g - \mathrm{t}_{t_0 \to t}^{G'})$$

在训练 $N_d = 2500$ 次迭代后执行一次致密化。

### 统一模糊合成

无论散焦还是运动模糊，都统一建模为逐像素核卷积：

$$\tilde{B}(x) = \sum_{x_i \in \mathcal{N}(x)} \tilde{I}(x_i) k_x(x_i), \quad \text{s.t.} \sum_{x_i} k_x(x_i) = 1$$

**模糊预测网络 (BP-Net)**：4 层 CNN，输入包括相机嵌入 $e(i)$、场景特征 $f_{\text{scene}}$（由渲染图像、深度、运动掩码编码）和像素坐标位置编码 $p(x)$，同时预测模糊核 $k_x$ 和模糊强度 $m_x$：

$$k_x, m_x = F_\Theta(e(i), f_{\text{scene}}(x), p(x))$$

输出模糊图像通过锐利图像和模糊图像混合：

$$\hat{B}(x) = (1 - m_x) \cdot \tilde{I}(x) + m_x \cdot \tilde{B}(x)$$

**模糊感知稀疏约束**：防止对轻微模糊区域过度模糊化。定义核中心权重目标：

$$c_x = \text{sigmoid}(\text{scale} \cdot (1 - \text{sg}(m_x)))$$

$$\mathcal{L}_{\text{spa}} = \mathcal{L}_1(c_x, k_x(c))$$

模糊强度高 → 中心权重目标低 → 允许分散核；模糊强度低 → 中心权重目标高 → 强制集中核。

### 未见视角约束

通过几何和外观信息从训练视角合成周围的未见视角，减轻单目视频的过拟合：

$$p_t = K P_t^{-1} P_s D_s(p_s) K^{-1} p_s$$

生成平行未见视角（相邻训练视角间插值）和垂直未见视角（沿相机轨迹法向偏移），每 $N_u = 5$ 次迭代引入一次。

### 总损失

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{geo}} + \mathcal{L}_{\text{smo}} + \mathcal{L}_{\text{spa}}$$

重建损失 $\mathcal{L}_{\text{rec}} = (1-\beta)\mathcal{L}_1(\hat{B}, B) + \beta \mathcal{L}_{\text{ssim}}(\hat{B}, B)$，$\beta = 0.2$。

## 实验关键数据

### D2RF 散焦模糊 + DyBluRF 运动模糊

| 方法 | 散焦 PSNR↑ | 散焦 SSIM↑ | 散焦 LPIPS↓ | 运动 PSNR↑ | 运动 SSIM↑ | 运动 LPIPS↓ | 训练时间 |
|------|-----------|-----------|------------|-----------|-----------|------------|---------|
| D3DGS | 22.54 | 0.715 | 0.215 | 21.54 | 0.675 | 0.287 | 10 min |
| SoM | 28.32 | 0.784 | 0.164 | 26.21 | 0.823 | 0.109 | 10 min |
| D2RF | 27.04 | 0.808 | 0.128 | 23.67 | 0.745 | 0.120 | 48 hrs |
| DyBluRF | 26.24 | 0.788 | 0.159 | 24.53 | 0.864 | 0.079 | 48 hrs |
| De4DGS | 28.49 | 0.791 | 0.154 | 26.62 | 0.871 | 0.059 | 20 hrs |
| **Ours** | **29.39** | **0.859** | **0.078** | **27.01** | **0.876** | **0.056** | **1 hr** |

散焦数据集上 PSNR 比 De4DGS 高 0.9 dB，LPIPS 降低 49%；训练仅需 1 小时 vs 20 小时。

### 消融实验

| 配置 | 散焦 PSNR | 运动 PSNR |
|------|----------|----------|
| w/o 稀疏约束 | 29.03 | 26.63 |
| w/o Shortcut | 29.12 | 26.74 |
| w/o DGD | 29.19 | 26.53 |
| w/o 未见视角 | 29.09 | 26.66 |
| **完整方法** | **29.39** | **27.01** |

模糊核大小 $K=9$ 为最优平衡点，$K>9$ 提升不明显。

## 亮点

1. **首个统一框架**：同时处理散焦和运动模糊的动态场景重建
2. **模糊强度-核稀疏性耦合**：通过 $m_x$ 自适应约束核分布，物理上合理
3. **极高效率**：1 小时训练 + 65 FPS 渲染，比 NeRF 方案快 48×
4. 未见视角合成策略有效缓解单目视频过拟合

## 局限与展望

- 依赖 2D 先验（深度估计、分割），先验误差会传播
- 大幅非刚性运动模糊场景仍会失败
- 需要逐场景优化，无法泛化到新场景
- 模糊核大小固定为 $9 \times 9$，可能无法覆盖大范围模糊

## 与相关工作的对比

- **vs De4DGS / DyBluRF**：这些方法通过曝光时间内多帧加权模拟运动模糊，无法处理散焦；本文用核卷积统一两者
- **vs D2RF**：D2RF 用分层 DoF volume rendering 处理散焦，但无法处理运动模糊且极慢（48h）
- **vs Shape-of-Motion**：SoM 是清晰视频的 SOTA，本文在其基础上增加模糊建模，从模糊输入恢复清晰场景

## 启发与关联

模糊核预测与强度预测的解耦设计值得借鉴——显式建模"一个像素有多模糊"和"怎么模糊"两个维度。未见视角合成约束是解决单目重建过拟合的通用策略。

## 评分

- ⭐ 新颖性: 8/10 — 统一散焦+运动模糊的动态 3DGS 为首创，模糊感知稀疏约束设计巧妙
- ⭐ 实验充分度: 8/10 — 两个数据集覆盖两种模糊类型，消融完整
- ⭐ 写作质量: 7/10 — 结构清晰但部分细节（未见视角生成）描述较简略
- ⭐ 价值: 8/10 — 实用价值高，统一框架+高效率，对模糊视频重建有直接推动

<!-- RELATED:START -->

## 相关论文

- [DGH: Dynamic Gaussian Hair](dgh_dynamic_gaussian_hair.md)
- [CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](../../ICCV2025/3d_vision/comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [Dynamic Neural Radiance Field from Defocused Monocular Video](../../ECCV2024/3d_vision/dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images](../../CVPR2025/3d_vision/cocogaussian_leveraging_circle_of_confusion_for_gaussian_splatting_from_defocuse.md)

<!-- RELATED:END -->
