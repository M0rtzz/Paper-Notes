---
title: >-
  [论文解读] P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising
description: >-
  [ECCV 2024][3D视觉][点云去噪] 提出 P2P-Bridge，将点云去噪建模为 Schrödinger Bridge 问题，学习噪声点云到干净点云之间的最优传输计划，首次引入数据到数据（而非数据到噪声）的扩散范式，在合成数据和真实室内场景（ScanNet++、ARKitScenes）上均大幅超越现有方法。
tags:
  - ECCV 2024
  - 3D视觉
  - 点云去噪
  - Schrödinger Bridge
  - 扩散模型
  - 最优传输
  - DINOv2
---

# P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising

**会议**: ECCV 2024  
**arXiv**: [2408.16325](https://arxiv.org/abs/2408.16325)  
**代码**: [项目页](https://p2p-bridge.github.io) (有)  
**领域**: 3D视觉  
**关键词**: 点云去噪, Schrödinger Bridge, 扩散模型, 最优传输, DINOv2  

## 一句话总结

提出 P2P-Bridge，将点云去噪建模为 Schrödinger Bridge 问题，学习噪声点云到干净点云之间的最优传输计划，首次引入数据到数据（而非数据到噪声）的扩散范式，在合成数据和真实室内场景（ScanNet++、ARKitScenes）上均大幅超越现有方法。

## 研究背景与动机

- **领域现状**: 点云去噪是3D视觉的基础预处理任务，深度学习方法（ScoreDenoise、MAG、PD-Flow）已展现比传统方法更优的性能，但主要在合成高斯噪声假设下训练
- **现有痛点**: 真实世界扫描仪（LiDAR、手机）产生的噪声远比各向同性高斯噪声复杂，包括离群点簇、ghost点、边缘flare等效应，现有方法在真实场景中性能显著退化
- **核心矛盾**: 传统扩散模型使用高斯先验（data-to-noise），无法学习传感器特有的噪声特性；且现有方法使用的距离度量随点云大小非线性缩放，阻碍模型扩展
- **本文解决什么**: 如何设计一个可以学习数据特定噪声特性的去噪框架，使其在合成噪声和真实室内场景噪声上都表现优异
- **切入角度**: 将去噪问题重新建模为 Schrödinger Bridge 问题——寻找噪声点云到干净点云之间的最优传输路径
- **核心idea**: 用 data-to-data 扩散桥代替传统的 data-to-noise 扩散过程，配合最短路径插值实现无序点云间的有意义插值，并引入 DINOv2 语义特征辅助去噪

## 方法详解

### 整体框架

P2P-Bridge 将去噪建模为反向扩散过程：噪声点云 $\tilde{\mathcal{P}}$ 作为先验分布 $p_{\text{prior}}$，干净点云 $\mathcal{P}$ 作为数据分布 $p_{\text{data}}$。通过训练一个网络学习从 $\tilde{\mathcal{P}}$ 到 $\mathcal{P}$ 的最优传输计划，推理时通过 DDPM 采样迭代去噪。模型基于 PVCNN 架构，支持坐标、RGB 和 DINOv2 特征作为输入。

### 关键设计

**模块一：可控扩散桥（Tractable Diffusion Bridge）**

将噪声-干净点云对视为 Schrödinger Bridge 的配对边界数据。令漂移 $\mathbf{f} := 0$ 并使用线性扩散调度 $g^2(t)$，后验分布具有解析形式：

$$q(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_T) = \mathcal{N}(\mathbf{x}_t; \mu_t(\mathbf{x}_0, \mathbf{x}_T), \Sigma_t)$$

其中后验均值和方差为：

$$\mu_t = \frac{\bar{\sigma}_t^2}{\bar{\sigma}_t^2 + \sigma_t^2} \mathbf{x}_0 + \frac{\sigma_t^2}{\bar{\sigma}_t^2 + \sigma_t^2} \mathbf{x}_T, \quad \Sigma_t = \frac{\sigma_t^2 \bar{\sigma}_t^2}{\sigma_t^2 + \bar{\sigma}_t^2}$$

其中 $\sigma_t^2 = \int_0^t g^2(\tau) d\tau$，$\bar{\sigma}_t^2 = \int_t^1 g^2(\tau) d\tau$。这将复杂的 SB 问题简化为可训练的框架。

**模块二：无序点云的最短路径插值**

由于点云是无序的，$\mu_t$ 描述的插值需要定义合理的点对应关系。采用 PointMixup 的最短路径插值，寻找噪声点云和干净点云之间的最优双射分配：

$$\phi^* = \arg\min_{\phi \in \Phi} \sum_{i=1}^{N} \|\mathbf{x}_T^i - \mathbf{x}_0^{\phi(i)}\|_2$$

当桥的随机性消失（$g^2(t) \to 0$）时，桥 SDE 退化为最优传输 ODE：

$$d\mathbf{x}_t = \frac{g^2(t)}{\sigma_t^2} (\mathbf{x}_t - \mathbf{x}_0) dt$$

该分配只需对每个数据对计算一次，后续训练中可直接复用排序后的干净点云。

**模块三：语义增强特征嵌入**

创新性地引入 DINOv2 逐点特征：利用相机位姿和内参将像素级 DINOv2 特征投影到噪声点云上，为每个点提供高层语义信息。网络架构基于 PVCNN（PointVoxel-CNN），增加了多头全局注意力和特征嵌入模块，使用1×1卷积映射输入特征到高维空间。

### 损失函数 / 训练策略

- **噪声预测损失**: 标准的扩散模型噪声预测目标

$$\mathcal{L} = \|\epsilon_\theta(\mathbf{x}_t, t) - \frac{\mathbf{x}_t - \mathbf{x}_0}{\sigma_t}\|_2^2$$

- **DDPM 采样**: 推理时使用迭代 DDPM 采样，仅需 3 步函数评估即可获得良好结果
- **Patch 处理**: 对大规模室内场景采用 patch 方式处理，对重叠区域取平均坐标（而非直接拼接+FPS采样），有效减少 patch 边界伪影
- **训练配置**: 物体数据集用 PU-Net，场景数据集 batch size 32，最多100K步
- **时间步条件**: 使用正弦位置编码对时间步 $t$ 进行条件化，全局特征通过自适应组归一化注入

## 实验关键数据

### 主实验（物体级去噪，PU-Net 数据集，CD×10⁴）

| 方法 | 10K点 1%噪声 CD | 10K点 3%噪声 CD | 50K点 3%噪声 CD | 50K点 3%噪声 P2M |
|------|:---:|:---:|:---:|:---:|
| ScoreDenoise | 2.52 | 4.71 | 1.93 | 1.04 |
| MAG | 2.50 | 4.69 | 1.93 | 1.05 |
| PD-Flow | 2.13 | 5.19 | 3.90 | 2.86 |
| I-PFN | 2.31 | 5.49 | 2.54 | 1.65 |
| **P2P-Bridge** | **2.28** | **3.99** | **1.56** | **0.84** |

高噪声（3%）下优势显著：50K点 CD 1.56 vs ScoreDenoise 1.93（-19%），P2M 0.84 vs 1.04（-19%）。

### 真实室内场景（ScanNet++ Apple LiDAR + 3DMatch，指标×10⁴）

| 方法 | 特征 | CD | P2M |
|------|------|:---:|:---:|
| Bilateral | XYZ | 64.28 | 63.51 |
| ScoreDenoise | XYZ | 58.78 | 57.99 |
| PD-Flow | XYZ | 54.02 | 53.14 |
| I-PFN | XYZ | 52.31 | 51.49 |
| **P2P-Bridge** | XYZ | 35.56 | 34.78 |
| **P2P-Bridge** | XYZ+RGB | 35.17 | 34.39 |
| **P2P-Bridge** | XYZ+RGB+DINO | **34.88** | **34.11** |

ScanNet++ 上 CD 从次优 I-PFN 的 52.31 降至 34.88（-33%），优势巨大。

### 关键发现

- **data-to-data 范式在真实噪声上优势巨大**: 合成高斯噪声下各方法差距不大，但真实室内场景中 P2P-Bridge 大幅领先，验证了学习数据特定噪声特性的重要性
- **仅3步去噪即可获得优秀结果**: DDPM 采样的鲁棒性使模型对去噪步数不敏感
- **DINOv2 语义特征有效**: 加入 DINOv2 特征后在 ScanNet++ 上进一步降低 CD（从35.56到34.88），高层语义信息有助于区分结构边界
- 基于高斯噪声训练的方法在 patch 处理时产生严重伪影（边界点被误判为离群值），而 P2P-Bridge 通过坐标平均策略和真实噪声训练有效避免
- 在未见过的 PC-Net 数据集上泛化良好，适应性优于竞争方法

## 亮点与洞察

- **问题建模优雅**: 将去噪重构为 Schrödinger Bridge 的最优传输问题，理论基础扎实，将扩散模型从"噪声到数据"扩展为"数据到数据"
- **最短路径插值关键**: 巧妙解决了无序点云间插值的核心技术难题，且分配只需计算一次，训练高效
- **真实场景评估开创性**: 首次系统在 ScanNet++ 和 ARKitScenes 等真实扫描数据上评估点云去噪方法，填补了领域评估空白
- **语义辅助去噪**: DINOv2 特征的引入为点云去噪提供了新维度，不再局限于几何特征

## 局限与展望

- 需要配对的噪声-干净训练数据，获取成本较高（依赖高精度 Faro 扫描仪）
- 最优分配 $\phi^*$ 的计算为 $O(N^3)$，对大规模点云可能成为瓶颈
- 无法处理点云缺失/不完整区域，需结合点云补全方法
- 大规模场景的 patch 划分策略仍需优化，patch 间一致性有待提升
- 未探索无监督或自监督变体，降低对配对数据的依赖

## 相关工作与启发

- **ScoreDenoise / MAG**: 基于得分匹配的代表方法，在高斯噪声下效果好但真实场景退化严重
- **PD-Flow**: 基于 normalizing flow 的方法，在真实噪声上也有合理表现但存在 patch 伪影
- **I²-SB (Schrödinger Bridge for Images)**: 图像翻译中的 SB 应用，本文借鉴其可控桥框架
- **PointMixup**: 提供了无序点云间有意义插值的理论基础
- **启发**: SB 范式可推广到点云补全、形状生成、点云配准等其他3D数据到数据转换任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将 Schrödinger Bridge 应用于点云去噪，data-to-data 扩散范式具有开创性
- **实验充分度**: ⭐⭐⭐⭐⭐ — 合成+真实数据集全面评估，多种baseline对比，消融实验覆盖关键设计
- **写作质量**: ⭐⭐⭐⭐ — 数学推导清晰，从 SDE 到可控框架的过渡自然，图示直观
- **实用价值**: ⭐⭐⭐⭐⭐ — 仅3步去噪、代码开源、真实场景效果显著，实际应用前景广阔

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models](sediff_structure_extraction_for_domain_adaptive_depth_estimation_via_denoising_d.md)
- [\[ICCV 2025\] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising](../../ICCV2025/3d_vision/noise2score3d_tweedies_approach_for_unsupervised_point_cloud_denoising.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation](cloudfixer_test-time_adaptation_for_3d_point_clouds_via_diffusion-guided_geometr.md)

<!-- RELATED:END -->
