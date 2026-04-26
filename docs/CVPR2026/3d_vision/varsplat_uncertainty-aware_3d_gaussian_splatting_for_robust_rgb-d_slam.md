---
title: >-
  [论文解读] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM
description: >-
  [CVPR2026][3D视觉][3D Gaussian Splatting] 提出 VarSplat，首个在3DGS-SLAM中学习**逐splat外观方差** $\sigma^2$ 并通过全方差定律渲染**逐像素不确定性图** $V$ 的系统，将不确定性统一应用于跟踪、子图配准和回环检测，在4个数据集上取得鲁棒且领先的性能。
tags:
  - CVPR2026
  - 3D视觉
  - 3D Gaussian Splatting
  - SLAM
  - uncertainty modeling
  - RGB-D
  - alpha合成
---

# VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM

**会议**: CVPR2026  
**arXiv**: [2603.09673](https://arxiv.org/abs/2603.09673)  
**代码**: [项目主页](https://anhthuan1999.github.io/varsplat/)  
**领域**: 3d_vision  
**关键词**: 3D Gaussian Splatting, SLAM, uncertainty modeling, RGB-D, alpha合成

## 一句话总结

提出 VarSplat，首个在3DGS-SLAM中学习**逐splat外观方差** $\sigma^2$ 并通过全方差定律渲染**逐像素不确定性图** $V$ 的系统，将不确定性统一应用于跟踪、子图配准和回环检测，在4个数据集上取得鲁棒且领先的性能。

## 背景与动机

3DGS-SLAM 通过光栅化各向异性高斯实现快速可微分渲染，在重建质量和速度上远超 NeRF-SLAM。然而现有方法普遍存在一个关键缺陷：**测量可靠性未被显式建模**。当场景中出现低纹理区域、透明表面、反射表面或深度不连续边界时，均匀的光度权重会导致位姿估计产生漂移。

现有不确定性建模方案的不足：

- **几何端不确定性**（如 CG-SLAM 的深度方差、UncLe-SLAM 的逐像素深度不确定性）：仅建模几何维度，忽略外观不稳定性
- **预训练预测器**（如 WildGS-SLAM 基于 DINOv2 特征预测不确定性图）：依赖外部模型，无法端到端优化
- **射线终止概率**（如 Uni-SLAM 的 termination-probability field）：不确定性不来自光栅化器本身

VarSplat 的核心想法是：**直接学习每个高斯的外观方差 $\sigma_i^2$，通过全方差定律和 alpha 合成传播到逐像素不确定性 $V$，在单次光栅化 pass 中完成**。

## 核心问题

1. 如何在3DGS中显式建模外观不确定性，且不引入额外网络或预训练模型？
2. 如何将逐splat方差高效传播为逐像素不确定性图？
3. 如何将不确定性统一用于SLAM三个关键环节：跟踪、配准、回环检测？

## 方法详解

### 3.1 逐像素不确定性渲染

**Splat 表示扩展。** 在标准3DGS的基础上，每个 Gaussian $G_i$ 除了均值位置 $\mu_i$、不透明度 $\alpha_i$、尺度 $s_i$、协方差 $\Sigma_i$ 和球谐颜色 $c_i$，额外增加一个**外观方差参数** $\sigma_i^2 \in \mathbb{R}^3$（三通道），表示该 splat 颜色均值附近的不确定程度。每个子图定义为：

$$P^s = \{G_i^s(\mu_i, \Sigma_i, \alpha_i, s_i, c_i, \sigma_i^2) \mid i=1,\ldots,N^s\}$$

**直觉理解。** $\sigma_i^2$ 与空间协方差 $\Sigma_i$（定义几何范围）及SH系数（定义均值外观）完全不同。即使SH正确建模了视角相关的均值颜色，在深度不连续/遮挡边界/反射区域，微小视角变化就会改变重叠splat的可见性和alpha权重，导致不一致的颜色观测——此时 $\sigma_i^2$ 会学习到较大的值。

**标准 alpha 合成。** 透射率和权重：

$$w_i = T_i \alpha_i, \quad T_i = \prod_{j}^{i-1}(1-\alpha_j)$$

渲染颜色 $C$ 和深度 $D$：

$$C = \sum_i w_i c_i, \quad D = \sum_i w_i z_i$$

**基于全方差定律的方差渲染。** 对随机变量 $X$（像素颜色）和条件变量 $Z$（3D高斯），全方差定律将逐像素方差分解为两项：

$$\text{Var}[X] = \mathbb{E}[\text{Var}[X|Z]] + \text{Var}(\mathbb{E}[X|Z])$$

- **第一项**（期望逐splat方差）：通过 alpha 合成 $\sum_i w_i \sigma_i^2$ 直接得到
- **第二项**（splat均值的方差）：利用二阶矩公式 $\sum_i w_i c_i^2 - (\sum_i w_i c_i)^2$

合并得到最终的逐像素不确定性 $V$：

$$V = \sum_i w_i(\sigma_i^2 + c_i^2) - \left(\sum_i w_i c_i\right)^2$$

**关键优势：** $V$ 的计算与颜色/深度渲染共享同一个单次光栅化 pass，不需要额外的前向传递或蒙特卡洛采样，保持了实时效率。

### 3.2 建图（Mapping）

**子图管理。** 遵循 LoopSplat/Gaussian-SLAM 的子图策略：当相机移动超过子图质心的空间阈值或累计跟踪不确定性超过预设限制时创建新子图。首帧通过深度反投影初始化高斯，后续帧在未观测区域添加或合并重叠的高斯。

**建图损失函数：**

$$\mathcal{L}_{\text{map}} = \lambda_{\text{color}} \cdot \mathcal{L}_{\text{color}} + \lambda_{\text{depth}} \cdot \mathcal{L}_{\text{depth}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}} + \lambda_{\text{var}} \cdot \mathcal{L}_{\text{var}}$$

其中：

- **颜色损失**：L1 + SSIM 加权组合 $\mathcal{L}_{\text{color}} = (1-\lambda_{\text{SSIM}})\|\hat{I}-I\|_1 + \lambda_{\text{SSIM}}(1-\text{SSIM}(\hat{I},I))$
- **深度损失**：$\mathcal{L}_{\text{depth}} = \|\hat{D}-D\|_1$
- **正则化**：控制高斯尺度 $\mathcal{L}_{\text{reg}} = \|\hat{s}-s\|_1$
- **方差损失**：基于高斯负对数似然

**从零学习方差。** 受 ActiveNeRF 的似然视角启发，方差损失采用高斯负对数似然形式：

$$\mathcal{L}_{\text{var}} = \frac{1}{2V}\left(\|\hat{I}-I\|_2^2 + \|\hat{D}-D\|_2^2\right) + \log(V)$$

设计要点：
- 使用 **平方 L2（MSE）** 而非 L1 作为残差，因为 L1 对应 Laplace 分布的 scale 参数，会破坏高斯模型假设
- 同时纳入颜色和深度残差，使方差反映几何+外观的综合可靠性
- 梯度分析：$\frac{\partial \mathcal{L}_{\text{var}}}{\partial \sigma_i^2} = \frac{\partial \mathcal{L}_{\text{var}}}{\partial V} \cdot w_i$，方差通过 alpha 权重 $w_i$ 传播到每个 splat

### 3.3 下游位姿估计

**不确定性归一化权重。** 采用中位数中心化对数缩放，对像素级和子图级方差分别计算权重：

$$\widetilde{w}_p = \exp[-(\log V - \widetilde{V})/\tau], \quad \widetilde{V} = \text{median}_\Omega(\log V)$$

$$\widetilde{w}_s = \exp[-(\log \sigma^2 - \widetilde{\sigma^2})/\tau], \quad \widetilde{\sigma^2} = \text{median}(\log \sigma^2)$$

方差大于中位数的像素/splat 权重衰减，可靠区域获得更强的监督。$\tau > 0$ 控制权重的锐度。

**跟踪（Tracking）。** 给定输入帧 $(I,D)$，估计当前位姿 $T_j$。RGB 图像更易受视角、低纹理和遮挡影响，因此用不确定性权重自适应约束：

$$\mathcal{L}_{\text{track}} = \sum \lambda_c (\widetilde{w_p} \odot \|\hat{I}-I\|_1) + (1-\lambda_c)\|\hat{D}-D\|_1$$

关键：跟踪时**冻结方差参数**并停止 $\widetilde{w_p}$ 的梯度，避免与位姿优化冲突。

**回环检测（Loop Detection）。** 利用逐splat方差 $\sigma_i^2$ 调制子图相似度。计算方差加权后的不透明度比率：

$$r = \frac{\sum_j \widetilde{w_s} \alpha_j}{\sum_j \alpha_j}, \quad \text{sim} = \text{cross\_sim} \odot (r_q \cdot r_{db})$$

该比率编码子图中剩余多少可靠外观信息，无需逐子图惩罚。

**配准（Registration）。** 检测到回环后，将查询关键帧定位到数据库子图中，用不确定性权重调制光度损失：

$$\mathcal{L}_{\text{registration}} = \sum \widetilde{w_p} \odot \|\hat{I}-I\|_1 + \|\hat{D}-D\|_1$$

**全局合并。** 通过 TSDF 融合合并所有子图，用融合几何初始化全局高斯中心，最终用 $\mathcal{L}_{\text{color}}$ 精炼（此阶段不使用不确定性权重，因为不稳定区域已在前面环节被控制）。

## 实验关键数据

### 跟踪性能（ATE RMSE ↓，cm）

| 数据集 | 最佳基线 | VarSplat | 提升 |
|--------|---------|----------|------|
| Replica (8场景均值) | LoopSplat: 0.26 | **0.23** | ~12% |
| ScanNet++ (5场景均值) | LoopSplat: 2.05 | **1.69** | ~18% |
| TUM-RGBD (5场景均值) | LoopSplat: 3.33 | **3.20** | ~4% |
| ScanNet (6场景均值) | Loopy-SLAM: 7.7 | **6.5** | ~16% |

### 渲染与重建性能

| 指标 | 数据集 | VarSplat | 对比（LoopSplat） |
|------|--------|----------|------------------|
| PSNR ↑ | Replica | 37.15 | 36.63 |
| SSIM ↑ | Replica | 0.986 | 0.985 |
| LPIPS ↓ | Replica | 0.109 | 0.112 |
| Depth L1 ↓ | Replica | 0.50 | 0.51 |
| F1 ↑ | Replica | 90.2% | 90.4% |
| NVS PSNR ↑ | ScanNet++ | **21.33** | 21.30 |

### 消融实验

不确定性在三个环节的逐步叠加效果（ScanNet 6场景均值 ATE RMSE）：

- 无不确定性：8.20 → +跟踪：7.63 → +回环：7.49 → +配准（全部启用）：**6.53**，总提升约 **20%**

运行时间（Replica/Room0，A100 80GB）：Mapping 1.9s/帧，Tracking 2.0s/帧，与 LoopSplat（1.2s/1.8s）相当。

## 亮点

1. **数学优雅**：通过全方差定律将逐splat方差传播到逐像素不确定性，无需蒙特卡洛采样或额外前向传递，完全在单次光栅化 pass 中完成
2. **端到端学习**：方差 $\sigma_i^2$ 作为可微分参数与位姿和高斯参数联合优化，不依赖预训练模型
3. **统一不确定性应用**：同一套方差信号贯穿跟踪（像素级）、回环（子图级）、配准（像素级）三个环节
4. **冻结策略**：跟踪和回环时冻结方差参数，避免梯度冲突，设计合理
5. **鲁棒性提升明显**：在真实数据集（ScanNet/ScanNet++/TUM-RGBD）上比基线稳定得多

## 局限与展望

1. **仅支持 RGB-D 输入**：未扩展到纯 RGB（单目/双目）场景，限制应用范围
2. **计算开销增加**：Mapping 从 LoopSplat 的 1.2s/帧增至 1.9s/帧（+58%），对实时性敏感的应用可能不友好
3. **合并阶段丢弃不确定性**：TSDF 融合后的全局精炼未使用 $V$，可能损失最终重建质量
4. **方差建模假设**：使用各向同性的逐通道方差 $\sigma_i^2 \in \mathbb{R}^3$，未建模通道间协方差
5. **动态场景**：未考虑动态物体的处理，在动态环境下可能失效

## 与相关工作的对比

| 方法 | 不确定性类型 | 来源 | 在线学习 | 单次pass |
|------|------------|------|---------|---------|
| CG-SLAM | 深度方差 | 几何驱动 | ✓ | ✓ |
| Uni-SLAM | 射线终止概率 | 隐式场 | ✓ | ✗ |
| WildGS-SLAM | DINOv2特征图 | 预训练 | ✗ | ✓ |
| ActiveNeRF | 逐像素方差 | 神经网络 | ✓ | ✗ |
| **VarSplat** | **逐splat外观方差** | **全方差定律** | **✓** | **✓** |

VarSplat 的核心优势在于：不确定性直接来自3DGS表示本身（而非外部模型），通过闭式公式传播（而非采样），且在线端到端优化（而非后处理）。

## 启发与关联

- **方差冻结策略**值得借鉴：在不同阶段选择性地冻结/训练方差参数，避免梯度冲突，这对多任务联合优化有普适指导意义
- 全方差定律的分解 $V = \mathbb{E}[\text{Var}] + \text{Var}(\mathbb{E})$ 可推广到3DGS的其他属性（如语义、法线）的不确定性估计
- 类似的不确定性加权思路可迁移到 3DGS-based 的其他任务：自由视点合成中的主动视角选择、场景补全、语义分割等

## 评分

- 新颖性: ⭐⭐⭐⭐ — 全方差定律+alpha合成的公式推导干净优雅，是3DGS不确定性建模的自然而新颖的方案
- 实验充分度: ⭐⭐⭐⭐⭐ — 4个数据集（合成+真实），与12+基线对比，消融覆盖三个下游任务和方差训练策略
- 写作质量: ⭐⭐⭐⭐ — 公式推导清晰，动机阐述充分，实验组织有条理
- 价值: ⭐⭐⭐⭐ — 为3DGS-SLAM系统提供了一种高效实用的不确定性建模范式，有较强的方法论贡献

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [\[CVPR 2026\] DROID-W: DROID-SLAM in the Wild](droid-slam_in_the_wild.md)
- [\[CVPR 2026\] FastGS: Training 3D Gaussian Splatting in 100 Seconds](fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [\[CVPR 2026\] Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](spectral_defense_against_resource-targeting_attack_in_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists](speeding_up_the_learning_of_3d_gaussians_with_much_shorter_gaussian_lists.md)

<!-- RELATED:END -->
