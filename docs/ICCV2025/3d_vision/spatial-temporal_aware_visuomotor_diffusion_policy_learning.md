---
title: >-
  [论文解读] Spatial-Temporal Aware Visuomotor Diffusion Policy Learning
description: >-
  [ICCV 2025][3D视觉][视觉模仿学习] 提出 4D Diffusion Policy（DP4），通过动态高斯世界模型为扩散策略注入3D空间和4D时空感知能力，在17个仿真任务和3个真实机器人任务上大幅超越基线（Adroit +16.4%, DexArt +14%, RLBench +6.45%, 真实任务 +8.6%）。
tags:
  - ICCV 2025
  - 3D视觉
  - 视觉模仿学习
  - 扩散策略
  - 3D高斯溅射
  - 4D时空感知
  - 世界模型
  - 灵巧操作
---

# Spatial-Temporal Aware Visuomotor Diffusion Policy Learning

**会议**: ICCV 2025  
**arXiv**: [2507.06710](https://arxiv.org/abs/2507.06710)  
**代码**: [项目主页](https://ZhenyangLiu.github.io/DP4)  
**领域**: 3D视觉 / 机器人操作  
**关键词**: 视觉模仿学习, 扩散策略, 3D高斯溅射, 4D时空感知, 世界模型, 灵巧操作

## 一句话总结

提出 4D Diffusion Policy（DP4），通过动态高斯世界模型为扩散策略注入3D空间和4D时空感知能力，在17个仿真任务和3个真实机器人任务上大幅超越基线（Adroit +16.4%, DexArt +14%, RLBench +6.45%, 真实任务 +8.6%）。

## 研究背景与动机

视觉模仿学习（Visual Imitation Learning）是训练机器人执行复杂任务的有效方法，已在物体抓取、腿足运动、灵巧操作等领域取得显著进展。然而，现有方法存在两个核心瓶颈：

**依赖行为克隆**：主流方法通过监督历史轨迹进行行为克隆（Behavior Cloning），仅被动模仿专家动作，无法真正理解环境的物理结构

**缺乏时空感知**：既无法准确捕获当前场景的3D空间结构（如物体的精确位置和几何），也无法建模交互过程中的4D时空动态变化（如物体运动趋势）

这意味着，当场景发生变化或需要精确物体交互时，传统方法容易失败。例如，在钉锤子任务中，仅模仿挥锤动作而不理解钉子的3D位置，会导致频繁偏移。

**核心问题**：如何让扩散策略模型不仅学习"怎么动"，还理解"环境长什么样"以及"环境会怎么变化"？

DP4 的回答是：构建一个动态高斯世界模型（Dynamic Gaussian World Model），将3D空间结构和4D时空动态显式编码到策略学习过程中。

## 方法详解

### 整体框架

DP4 的流程如下：

1. **输入**：单视角 RGB-D 图像
2. **3D 点云构建**：将 RGB-D 转为 3D 点云
3. **多层次特征提取**：提取局部（Local）和全局（Global）3D表征
4. **扩散策略**：以多层次3D表征和机器人状态为条件，通过条件扩散模型生成轨迹
5. **高斯世界模型**（仅训练时）：从点云构建3DGS重建当前场景 → 预测未来场景 → 提供3D/4D监督

关键洞察：高斯世界模型仅在训练时参与，推理时不增加额外开销。

### 关键设计 1：多层次3D空间感知

**局部3D表征**：
- 对点云进行裁剪（保留感兴趣区域，如机械臂附近），使用 FPS 下采样到 512 个点
- 通过轻量 MLP 网络（3D Local Encoder）编码为紧凑局部特征

**全局3D表征**：
- 将完整点云转为 $100^3$ 的体素（Voxel）
- 通过 3D 卷积编码器（3D Global Encoder，U-Net 结构）输出全局特征 $v^{(t)} \in \mathbb{R}^{100^3 \times 64}$

**可泛化高斯回归器（Generalizable Gaussian Regressor）**：
- 从全局体素特征回归高斯原语参数 $\theta^{(t)} = (\mu, c, r, s, \sigma)$，分别表示位置、颜色、旋转、缩放和不透明度
- 通过可微分 tile-based 栅格化渲染 RGB 和 Depth 图像
- 使用3D重建损失增强空间感知：

$$\mathcal{L}_{3D} = \sum_{\mathbf{p}} \|\mathbf{C}^{(t)}(\mathbf{p}) - \mathbf{C}^{*(t)}\|_2^2 + \|\mathbf{D}^{(t)}(\mathbf{p}) - \mathbf{D}^{*(t)}\|_2^2$$

### 关键设计 2：4D时空感知

在3D高斯世界模型的基础上叠加时间维度的建模：

- **可变形 MLP（Deformable MLP）**：给定当前高斯参数 $\theta^{(t)}$ 和动作 $a^{(t)}$，预测参数变化量 $\Delta\theta^{(t)}$
- **未来场景重建**：$\theta^{(t+1)} = \theta^{(t)} + \Delta\theta^{(t)}$，然后渲染未来时刻的 RGB 和 Depth
- **4D 一致性监督**：将预测的未来场景与真实未来场景对齐

$$\mathcal{L}_{4D} = \sum_{\mathbf{p}} \|\mathbf{C}^{(t+1)}(\mathbf{p}) - \mathbf{C}^{*(t+1)}\|_2^2 + \|\mathbf{D}^{(t+1)}(\mathbf{p}) - \mathbf{D}^{*(t+1)}\|_2^2$$

这迫使3D表征不仅编码当前场景结构，还隐式学习物理世界的动态规律。

### 关键设计 3：扩散决策模块

采用基于卷积网络的扩散策略（DDIM 噪声调度）：

- 以多层次3D表征 $r$ 和机器人状态 $q$ 为条件
- 从高斯噪声 $a^K$ 经过 $K$ 步去噪生成无噪声动作 $a^0$
- 训练时预测 2 步观测、生成 4 步动作、执行最后 3 步

### 损失函数

总损失为三部分加权组合：

$$\mathcal{L}_{DP4} = \mathcal{L}_{action} + \lambda_{3D} \mathcal{L}_{3D} + \lambda_{4D} \mathcal{L}_{4D}$$

其中 $\lambda_{3D} = 0.1$，$\lambda_{4D} = 0.01$。训练策略：前 500 步冻结 Deformable MLP（warm-up），之后联合训练所有模块。

## 实验关键数据

### 主实验：仿真任务成功率 (%)

**Adroit 仿真（灵巧手操作）**：

| 方法 | Hammer | Door | Pen | 平均 |
|------|--------|------|-----|------|
| IBC | 0 | 0 | 9 | 3.0 |
| BCRNN | 0 | 0 | 9 | 3.0 |
| Diffusion Policy | 48 | 50 | 25 | 41.0 |
| DP3 | 100 | 62 | 43 | 68.3 |
| **DP4 (本文)** | **100** | **80** | **75** | **84.7** |

**DexArt 仿真（关节物体操作）**：

| 方法 | Laptop | Faucet | Bucket | Toilet | 平均 |
|------|--------|--------|--------|--------|------|
| DP | 69 | 23 | 58 | 46 | 49.0 |
| DP3 | 83 | 63 | 82 | 46 | 68.5 |
| **DP4 (本文)** | **92** | **84** | **90** | **64** | **82.5** |

**RLBench 仿真（多任务操作，10个任务166个变体）**：

| 方法 | 第一组5任务平均 | 第二组5任务平均 |
|------|----------------|----------------|
| PerAct | 30.4 | 10.4 |
| GNFactor | 47.5 | 16.0 |
| ManiGaussian | 57.1 | 33.2 |
| **DP4 (本文)** | **63.3** | **39.9** |

### 消融实验（Adroit）

| RGB监督 | Depth监督 | 4D动态 | Hammer | Door | Pen |
|---------|----------|--------|--------|------|-----|
| ✗ | ✗ | ✗ | 94.0 | 64.0 | 45.0 |
| ✔ | ✗ | ✗ | 96.0 | 68.0 | 48.0 |
| ✔ | ✔ | ✗ | 98.0 | 75.0 | 72.0 |
| ✔ | ✔ | ✔ | **100.0** | **80.0** | **75.0** |

### 真实机器人实验

| 方法 | 抓瓶子 | 叠杯子 | 倒水 | 平均 |
|------|--------|--------|------|------|
| DP | 36.0 | 44.0 | 28.0 | 36.0 |
| DP3 | 42.0 | 62.0 | 34.0 | 46.0 |
| **DP4 (本文)** | **48.0** | **72.0** | **44.0** | **54.6** |

### 关键发现

1. **4D监督贡献最大**：从消融看，Pen 任务在加入4D监督后成功率从 47% 跃升至 75%，说明动态感知对精细操作至关重要
2. **推理无额外开销**：高斯世界模型仅在训练时参与，推理时间仅增加约 0.1 秒（如 Hammer 从 6.40s 到 6.57s）
3. **超参数敏感度**：$\lambda_{3D}=0.1$, $\lambda_{4D}=0.01$ 为最优配置，过大过小都会降低性能
4. **单视角即可工作**：不需要多相机设置，仅用一个 RGB-D 相机即可完成全部任务

## 亮点与洞察

1. **世界模型作为训练辅助而非推理依赖**：这是非常精巧的设计——高斯世界模型仅在训练时提供额外监督信号，推理时完全去掉，既享受了结构化监督的好处，又不增加部署成本
2. **从"被动模仿"到"主动理解"**：传统 BC 只关心输出层的动作正确，DP4 同时要求中间表征能重建3D场景和预测未来场景，强制特征学有用的物理信息
3. **3D + 4D 分层设计**：3D损失确保空间结构感知，4D损失在此基础上叠加时间动态，逐层递进非常合理
4. **可泛化高斯回归器**：直接从体素特征回归高斯参数，避免了逐场景优化 3DGS 的高昂成本，实现了泛化能力
5. **实验覆盖全面**：17 个仿真任务 + 173 个变体 + 3 个真实任务，包括灵巧手、关节物体、可变形物体等多种挑战

## 局限性

1. **单视角限制**：仅使用单个 RGB-D 相机，渲染质量有限（论文也承认渲染缺乏细节），对遮挡严重的场景可能失效
2. **真实任务成功率仍有提升空间**：最佳真实任务成功率为 72%（叠杯子），倒水仅 44%，距离实际部署仍有差距
3. **仅预测一步未来**：4D监督只建模 $t \to t+1$ 的转换，未考虑更长时间跨度的动态预测
4. **依赖深度信息**：需要 RGB-D 而非纯 RGB 输入，限制了纯视觉相机场景的适用性
5. **训练成本**：需要 H100 80GB GPU，训练 3000 个 epoch，计算资源需求较高

## 相关工作与启发

- **3D Diffusion Policy (DP3)**：DP4 的直接前身，使用点云3D表征但缺乏时空感知，DP4 在其上增加了高斯世界模型
- **ManiGaussian**：同样使用高斯溅射做机器人操作，但作为 PerAct 风架构的感知模块，DP4 则将其融入扩散策略框架
- **GNFactor**：用 NeRF 做泛化因子，DP4 改用 3DGS 获得更高效的渲染和更好的扩展性
- **Diffusion Policy**：基础框架，DP4 在保留扩散策略优势的同时引入结构化的3D/4D监督
- **World Model 路线**：Dreamer 系列在潜空间做未来预测，DP4 选择在显式3D表示空间做预测，物理可解释性更强

**启发**：将世界模型作为训练时的辅助监督而非推理组件，是一个值得推广的范式——可以将昂贵的结构化知识"蒸馏"进特征表征中。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 3DGS 世界模型引入扩散策略做训练监督的思路很新颖
- **技术深度**: ⭐⭐⭐⭐ — 3D/4D 分层监督设计合理，可变形高斯场预测未来有技术含量
- **实验充分度**: ⭐⭐⭐⭐⭐ — 17仿真+3真实任务，消融充分，可视化丰富
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富，动机阐述到位
- **实用价值**: ⭐⭐⭐⭐ — 推理不增加额外开销，单视角即可工作，有实际部署潜力
- **综合推荐**: ⭐⭐⭐⭐ — 扎实的工作，有效地将3D视觉和世界模型引入机器人策略学习

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting](7dgs_unified_spatialtemporalangular_gaussian_splatting.md)
- [Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](../../NeurIPS2025/3d_vision/trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)
- [SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space](../../ICML2025/3d_vision/se3-equivariant_diffusion_policy_in_spherical_fourier_space.md)
- [Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics](../../CVPR2026/3d_vision/efficient_hybrid_se3-equivariant_visuomotor_flow_policy_via_spherical_harmonics_.md)

<!-- RELATED:END -->
