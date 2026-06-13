---
title: >-
  [论文解读] ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping
description: >-
  [CVPR 2025][机器人][hand-object interaction] 提出多层遮挡（MLO）表示学习 3D 手-物遮挡关系，并将 Objaverse 大规模 3D 物体数据整合进训练，实现首个支持灵巧双手操作 + 可泛化物体外观的手-物操作视频生成框架。 领域现状： 基于扩散模型的手-物体交互（HOI）图像生成…
tags:
  - "CVPR 2025"
  - "机器人"
  - "hand-object interaction"
  - "video generation"
  - "multi-layer occlusion"
  - "扩散模型"
  - "Objaverse"
---

# ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping

**会议**: CVPR 2025  
**arXiv**: [2412.16212](https://arxiv.org/abs/2412.16212)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: hand-object interaction, video generation, multi-layer occlusion, diffusion model, Objaverse

## 一句话总结

提出多层遮挡（MLO）表示学习 3D 手-物遮挡关系，并将 Objaverse 大规模 3D 物体数据整合进训练，实现首个支持灵巧双手操作 + 可泛化物体外观的手-物操作视频生成框架。

## 研究背景与动机

**领域现状**: 基于扩散模型的手-物体交互（HOI）图像生成已取得初步成功，方法通常利用深度图、法线图、手部骨架等 2D 条件信号引导生成。

**现有痛点**:
1. **遮挡关系建模不足**: 现有方法只依赖 2D 条件信号，无法处理手指间自遮挡和手-物互相遮挡，导致生成结果出现穿透伪影和结构错误。
2. **物体泛化能力差**: HOI 视频数据集通常只包含十余种物体类别，训练后的模型难以泛化到训练集外的多样化物体。
3. **视频时序一致性**: 从图像到视频的扩展需要额外的时序一致性机制。

**核心矛盾**: 手部有极高的自由度和精细的关节结构，2D 条件信号是病态问题，无法唯一确定 3D 遮挡关系；同时 HOI 视频数据极度稀缺。

**本文切入角度**: 设计 3D 感知的 MLO 表示替代 2D 条件信号，同时引入 Objaverse 解决数据稀缺问题，通过多数据集联合训练策略统一学习。

## 方法详解

### 整体框架

ManiVideo 基于条件扩散模型，输入为双手 MANO 模型参数 $(θ,β)$ 和物体 3D 模型的运动序列，输出为时序一致的手-物操作视频。流程包括：
1. 将原始手-物信号转换为 MLO 表示（遮挡无关法线图 + 遮挡置信图）和物体表示（外观 + 几何）
2. 将 MLO 表示以两种方式嵌入 UNet（初始噪声 + Transformer block）
3. 将物体表示通过 AppearanceNet 和几何 embedding 注入 UNet
4. 两阶段训练：图像阶段 → 时序阶段

### 关键设计

**1. 多层遮挡（MLO）表示**
- **功能**: 将手-物 3D 模型分解为多个独立层（物体、掌心、拇指、食指、中指、无名指、小指），每层独立渲染遮挡无关法线图 $H$，补偿被遮挡的隐藏区域。
- **核心思路**: 受多平面图像（MPI）启发，构建从远到近的多层 3D 结构。同时引入遮挡置信图 $D$（基于深度图），深色区域表示遮挡越严重。模型通过 $D$ 区分可见/隐藏区域，利用 $H$ 修复被遮挡部分。
- **设计动机**: 2D 条件信号（深度图、mask 等）是病态输入，无法表示手指密集排列时的遮挡关系。MLO 提供完整的 3D 视角，使模型能感知被遮挡手指的几何。

**2. MLO 的双重嵌入**
- **功能**: 将 MLO 结构以两种方式注入 UNet：(a) 通过 Pose Guider（4层卷积）提取 $H$ 的特征并加到初始噪声 $z_t' = z_t + G([H])$；(b) 将 $H$ 和 $D$ 拼接后经卷积+MLP 提取 embedding $E_F$，通过交叉注意力注入 Transformer block。
- **核心思路**: 初始层学习粗粒度空间对应，深层 Transformer block 感知复杂遮挡关系。
- **设计动机**: 消融实验表明仅使用单一嵌入方式（w/o MLO*）会导致穿透伪影，双重嵌入互补地利用 MLO 信息。

**3. 物体表示与 Objaverse 集成**
- **功能**: 为每个 Objaverse 物体渲染 6 视角外观图像 $O_I$，加上背景参考图 $O_B$，通过 AppearanceNet $R$ 注入 UNet。同时渲染物体法线图 $H_o$ 并采样点云 $P \in \mathbb{R}^{2048 \times 3}$，提取几何 embedding $E_N$ 经交叉注意力注入。
- **核心思路**: 利用 Objaverse 800K+ 3D 模型的规模优势，随机生成旋转 $Q$ 和平移 $L$ 运动轨迹模拟物体运动，弥补 HOI 视频数据的稀缺。
- **设计动机**: HOI 数据集物体种类极少（~15 类），仅凭 HOI 数据训练会导致模型过拟合特定物体的纹理动态，Objaverse 提供了丰富的物体外观和几何多样性。

### 损失函数 / 训练策略

- **两阶段训练**: 图像阶段冻结时序层训练约 20K 迭代，时序阶段冻结图像层添加时序层训练约 30K 迭代。
- **多数据集混合**: 每次迭代从 Objaverse 数据、HOI 视频数据、人体数据中等比采样。Objaverse 数据不含手部，将 MLO 中手相关层置零；人体数据不含物体，将所有条件置零。
- **学习率**: 图像阶段 $1 \times 10^{-5}$，时序阶段 $8 \times 10^{-6}$，Adam 优化器。
- **人体扩展**: 可选提取人体骨架 $S$ 通过额外 Pose Guider $G_1$ 注入 UNet，支持人体中心的 HOI 视频生成。

## 实验关键数据

### 主实验

| 方法 | DexYCB FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | MPJPE↓ |
|---|---|---|---|---|---|
| HOGAN | 64.74 | 0.102 | 29.50 | 0.896 | 60.95 |
| ADiff | 53.95 | 0.093 | 29.96 | 0.903 | 59.12 |
| CDiff | 84.74 | 0.127 | 28.27 | 0.835 | 68.01 |
| **ManiVideo** | **49.96** | **0.079** | **30.10** | **0.913** | **57.30** |

自采数据集：

| 方法 | FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | MPJPE↓ |
|---|---|---|---|---|---|
| ADiff | 39.91 | 0.127 | 29.17 | 0.898 | 37.45 |
| CDiff | 45.50 | 0.133 | 28.33 | 0.883 | 42.89 |
| **ManiVideo** | **37.70** | **0.113** | **29.59** | **0.905** | **32.89** |

### 消融实验

| 配置 | FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | MPJPE↓ |
|---|---|---|---|---|---|
| w/o Objaverse | 61.60 | 0.121 | 27.99 | 0.895 | 37.33 |
| w/o MLO | 46.67 | 0.115 | 28.26 | 0.869 | 39.41 |
| w/o MLO* (仅初始噪声) | 40.60 | 0.117 | 28.30 | 0.881 | 34.02 |
| **完整模型** | **37.70** | **0.113** | **29.59** | **0.905** | **32.89** |

### 关键发现

1. **MLO 对遮挡建模至关重要**: 去掉 MLO 后 FID 从 37.70 升至 46.67，SSIM 从 0.905 降至 0.869，尤其在手指密集排列和弯曲不可见的场景下差距显著。
2. **双重嵌入互补**: 仅使用初始噪声嵌入（w/o MLO*）虽优于完全不用 MLO，但仍存在穿透问题，说明 Transformer block 的深层遮挡关系感知不可或缺。
3. **Objaverse 提升物体泛化**: 不用 Objaverse 时 FID 劣化最严重（61.60），模型过拟合训练集物体纹理。

## 亮点与洞察

- **首个框架**: 首次实现同时支持灵巧双手操作和可泛化物体的 HOI 视频生成
- **3D 感知条件设计**: MLO 表示将 2D 条件信号的病态问题转化为 3D 完整建模，思路新颖且有效
- **多数据集统一训练**: 巧妙地将 Objaverse（物体）、HOI 视频（交互）、人体数据通过条件置零统一到同一框架
- **实用价值**: 支持人体中心的 HOI 视频生成，可用于数字人和 VR 场景

## 局限与展望

- 性能受驱动信号精度制约（依赖 MANO 拟合质量）
- 复杂物体纹理的泛化仍受合成-真实域差距影响
- 可考虑 4D 表示（时空一致的物体外观建模）进一步提升纹理一致性
- 当前 HOI 训练数据规模仍然有限（722 视频，15 物体），扩大数据规模有望进一步提升

## 相关工作与启发

- **HOGAN**: GAN-based 方法，用流变形实现 HOI 图像编辑，但光流无法捕捉 3D 遮挡
- **Affordance Diffusion**: 给定物体参考图生成 HOI 图像，但只使用 2D mask 条件
- **Animate Anyone**: ManiVideo 的骨架设计参考了其 AppearanceNet + UNet 架构
- **Objaverse**: 800K+ 3D 模型的大规模数据集，本文首次将其系统性整合到 HOI 视频生成训练中

## 评分 ⭐⭐⭐⭐

**创新性**: ⭐⭐⭐⭐ MLO 表示和多数据集统一训练都是新颖设计  
**实验充分度**: ⭐⭐⭐⭐ 两个数据集对比 + 完善的消融实验  
**写作质量**: ⭐⭐⭐⭐ 结构清晰，图示直观  
**实用价值**: ⭐⭐⭐⭐ 在数字人和 VR 领域有直接应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [\[CVPR 2025\] ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning](maniptrans_efficient_dexterous_bimanual_manipulation_transfer_via_residual_learn.md)
- [\[ECCV 2024\] GraspXL: Generating Grasping Motions for Diverse Objects at Scale](../../ECCV2024/robotics/graspxl_generating_grasping_motions_for_diverse_objects_at_scale.md)
- [\[CVPR 2025\] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)
- [\[ECCV 2024\] Learning Cross-Hand Policies of High-DOF Reaching and Grasping](../../ECCV2024/robotics/learning_cross-hand_policies_of_high-dof_reaching_and_grasping.md)

</div>

<!-- RELATED:END -->
