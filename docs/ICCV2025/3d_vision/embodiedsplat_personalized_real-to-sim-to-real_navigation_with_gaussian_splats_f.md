---
title: >-
  [论文解读] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 提出 EmbodiedSplat，一个利用 iPhone 手机拍摄视频 → 3D 高斯溅射重建 mesh → 在 Habitat-Sim 中微调导航策略 → 部署到真实世界的完整流程，在真实场景 ImageNav 任务上比零样本基线提升 20%-40% 绝对成功率，sim-vs-real 相关系数达 0.87-0.97。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯溅射
  - 具身导航
  - sim-to-real迁移
  - 场景重建
  - 个性化策略训练
  - ImageNav
---

# EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device

**会议**: ICCV 2025  
**arXiv**: 2509.17430  
**代码**: https://gchhablani.github.io/embodied-splat (项目页面)  
**领域**: 3d_vision  
**关键词**: 3D高斯溅射, 具身导航, sim-to-real迁移, 场景重建, 个性化策略训练, ImageNav

## 一句话总结

提出 EmbodiedSplat，一个利用 iPhone 手机拍摄视频 → 3D 高斯溅射重建 mesh → 在 Habitat-Sim 中微调导航策略 → 部署到真实世界的完整流程，在真实场景 ImageNav 任务上比零样本基线提升 20%-40% 绝对成功率，sim-vs-real 相关系数达 0.87-0.97。

## 研究背景与动机

具身 AI 的训练和评估主要依赖仿真环境，但存在三大挑战：

**合成环境缺乏真实感**：HSSD 等合成数据集在风格和复杂度上与真实世界差距大，导致 sim-to-real 迁移困难

**真实场景采集成本高**：HM3D、Matterport3D 等高保真扫描依赖昂贵的专业设备和劳动密集型流程

**无法覆盖部署环境的多样性**：预训练数据集无法预见所有可能的部署场景，当机器人部署在大学、商场等新环境时，策略性能会显著下降

**核心问题**：能否用低成本的手机视频捕获部署环境，生成足够好的 3D 网格来微调导航策略，实现有效的 sim-to-real 迁移？

**本文的关键创新在于**：不是追求最好的重建质量，而是研究重建质量与导航性能之间的关系——即"多好的 mesh 就够用了？"

## 方法详解

### 整体框架（四阶段流水线）

1. **场景捕获**：iPhone 13 Pro Max + Polycam 应用录制 RGB-D 视频（20-30 分钟/场景），Nerfstudio 处理后采样 1000 帧对齐的 RGB-深度帧和位姿
2. **Mesh 重建**：使用 DN-Splatter 训练 3D 高斯溅射（30,000 迭代），通过 Poisson 重建生成 mesh；同时对比 Polycam 直接导出的 mesh
3. **Sim 训练**：将 mesh 转换为 .glb 格式加载到 Habitat-Sim，生成 ImageNav episode 并训练/微调策略
4. **Real 部署**：在 Stretch 机器人上部署策略进行真实场景导航

### 关键技术选择

- **DN-Splatter**：使用深度-法线正则化来提升 mesh 质量。传感器深度权重 $\lambda_d = 0.2$，启用深度平滑和法线损失
- **法线编码器**：经验性选择 Metric3D-V2 而非 Omnidata，因其产生更高质量的 mesh
- **Episode 生成**：HM3D/HSSD 数据集每个训练场景生成 10,000 个 episode；自采集场景仅生成 1,000 个训练 + 100 个评估 episode
- **评估指标**：成功率（SR）——在最大步数前停在目标位置 1m 以内

### 训练策略

- **零样本**：直接在 HM3D（800 个训练场景）或 HSSD（134 个训练场景）上预训练 600M-1200M 步
- **微调**：在预训练策略基础上，在单个重建场景上微调仅 20M 步（学习率 2.5e-6 给 LSTM 策略，6e-7 给视觉编码器）
- **过拟合**：从头在单个场景上训练 ~100M 步（用于验证是否需要大规模预训练）

## 实验关键数据

### 主实验：真实世界 lounge 场景导航成功率

| 策略 | 预训练数据 | Mesh 类型 | 成功率 (10 episodes) |
|---|---|---|---|
| 零样本 | HM3D (real) | — | 50% |
| 零样本 | HSSD (synthetic) | — | 10% |
| 微调 | HM3D → DN mesh | DN-Splatter | **70%** |
| 微调 | HM3D → Polycam | Polycam | **70%** |
| 微调 | HSSD → DN mesh | DN-Splatter | 40% |
| 微调 | HSSD → Polycam | Polycam | 50% |
| 过拟合(无预训练) | — → Polycam | Polycam | 50% |
| 过拟合(无预训练) | — → DN mesh | DN-Splatter | 10% |

**核心发现**：HM3D 预训练 + 微调，成功率从 50% → 70%（**+20%**）；HSSD 预训练 + 微调，成功率从 10% → 50%（**+40%**）。

### 消融：仿真中微调效果

| 场景 | HM3D 零样本 SR | HM3D 微调 SR |
|---|---|---|
| conf_a (DN mesh) | 85% | 95%+ |
| conf_b (DN mesh) | 88% | 95%+ |
| classroom (DN mesh) | 53% | 90%+ |
| lounge (DN mesh) | 50% | 90%+ |
| classroom (Polycam) | 42% | 90%+ |
| lounge (Polycam) | 76% | 90%+ |

微调后所有场景仿真成功率均达 90%+，**仅需额外 20M 步（对比预训练的 600M 步）**。

### 分析：Sim-to-Real 相关性

- DN mesh 的 SRCC（Sim-vs-Real Correlation）为 **0.87-0.97**
- 说明仿真中的性能改善可以可靠预测真实世界的改善
- 场景规模（平均最短路径距离）与零样本成功率负相关
- PSNR 与成功率正相关

### 关键发现

1. **不需要大规模预训练也能有非零真实世界成功率**：仅在 Polycam mesh 上过拟合的策略在真实世界达到 50% 成功率
2. **真实数据预训练远优于合成**：HM3D 零样本 50% vs HSSD 零样本 10%
3. **持续训练 HM3D 在 400M 步后对自采集场景的零样本性能开始下降或停滞**
4. **Polycam mesh 在视觉保真度上更好**（直接使用原始图像），但 DN mesh（开源）也具有竞争力

## 亮点与洞察

1. **低成本可扩展方案**：iPhone 拍 20-30 分钟 + DN-Splatter 训练 1-2 小时→ 可用的仿真场景，远低于 Matterport 的成本
2. **个性化部署的范式**：不是追求通用的、万能的策略，而是快速捕获特定部署场景并在其中微调——实际部署中这可能比追求更大的预训练数据更实用
3. **系统性分析**：不仅展示了方法，还深入分析了重建质量 vs 导航性能、预训练数据 vs 迁移性能等多个维度的关系，为后续研究提供了有价值的 insights
4. **手持拍摄即可（无需云台）**：对比 MuSHRoom 数据集使用云台，本文证明手持手机也足够

## 局限性

1. **真实世界评估规模小**：仅在 lounge 一个场景上做了 10 个 episode 的真实世界测试，统计可靠性存疑
2. **仅验证 ImageNav 任务**：未扩展到 ObjectNav、移动操作等更复杂任务
3. **场景规模有限**：自采集场景为 1-3 个房间级别，建筑级别的大型场景重建质量和导航效果未知
4. **DN mesh 的视觉保真度**：GS 学习的颜色与实际照片有差距，过拟合在 DN mesh 上的策略真实世界成功率仅 10%（vs Polycam 的 50%）
5. **缺少与 Phone2Proc 等同类方法的直接对比**

## 相关工作与启发

- **Phone2Proc**：使用 iPhone RoomPlan API 生成布局再程序化生成场景，但需后处理和多变体生成；本文直接重建整个场景
- **GaussNav / SplatNav**：用 GS 做导航但非端到端/未在真实机器人上验证
- **对未来具身 AI 的启示**：低成本 3D 重建 + 快速微调可能成为机器人部署的标准流程——先扫描再部署

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — 首次系统验证 GS → Habitat → Real 的完整个性化导航流程
- **实验完整性**: ⭐⭐⭐ — 真实世界评估规模小，但分析维度丰富
- **实用性**: ⭐⭐⭐⭐⭐ — 流程简单、成本低、可立即实践
- **写作质量**: ⭐⭐⭐⭐ — 组织清晰，研究问题明确

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](../../CVPR2025/3d_vision/mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [\[ICLR 2026\] D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping](../../ICLR2026/3d_vision/d-rex_differentiable_real-to-sim-to-real_engine_for_learning_dexterous_grasping.md)
- [\[ICCV 2025\] HouseTour: A Virtual Real Estate A(I)gent](housetour_a_virtual_real_estate_aigent.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacherguided_diffusion_for_3d_gaussian_s.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)

<!-- RELATED:END -->
