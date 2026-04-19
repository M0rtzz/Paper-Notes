# DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image

**会议**: CVPR 2025  
**arXiv**: [2503.19373](https://arxiv.org/abs/2503.19373)  
**代码**: https://hygenie1228.github.io/DeClotH/ (项目页)  
**领域**: 3D视觉 / 扩散模型  
**关键词**: 3D服装重建, 人体重建, 模板正则化, DMTet, ClothDiffusion

## 一句话总结
提出 DeClotH，用 DMTet 表示+3D 模板正则化（SMPLicit/SMPL）分别重建服装和人体，并训练 ClothDiffusion（ControlNet 微调）生成纯服装图像作为 SDS 损失指导，在单张图像下实现可分解的 3D 服装+人体重建，在 4D-DRESS 上超越所有基线。

## 研究背景与动机

**领域现状**：从单张图像重建 3D 人体+服装是时尚、VR 等应用的核心需求。现有方法要么联合重建服装和人体（分离困难），要么依赖参数化模型（表达力有限）。

**现有痛点**：(1) 联合重建时服装和人体几何严重耦合，无法单独编辑服装或人体；(2) 被遮挡区域（如背面、被服装遮挡的人体）无几何先验；(3) 标准 StableDiffusion 的 SDS 损失会生成混合了服装和人体的内容，而不是纯服装。

**本文目标** 从单张图像分别重建高质量的 3D 服装和人体，可独立编辑。

**切入角度**：用 3D 模板模型（SMPLicit 服装模板、SMPL 人体模板）作为几何正则化先验处理遮挡；训练专门的 ClothDiffusion 生成纯服装图像提供更精确的 SDS 引导。

**核心 idea**：DMTet 表示 + 3D 模板正则化处理遮挡 + ClothDiffusion 专用 SDS 损失实现可分解的单图 3D 服装+人体重建。

## 方法详解

### 关键设计

1. **模板正则化**：用 SMPLicit 服装模板和 SMPL 人体模板通过 Chamfer distance 正则化 DMTet 重建，使被遮挡区域也能获得合理的几何形状。消融显示无模板正则化时 CD 从 3.902 暴涨到 8.245

2. **ClothDiffusion SDS**：ControlNet 微调为条件生成纯服装图像（以服装轮廓+人体骨架为条件），替代标准 SD 的 SDS 损失。标准 SD 会在纯服装位置生成人体皮肤等内容，干扰服装独立重建

3. **DMTet 表示**：可微分四面体网格，支持拓扑变化，适合表示复杂服装形状

## 实验关键数据（4D-DRESS）

| 方法 | CD↓(服装) | CD↓(服装+人体) | NC↓(服装+人体) |
|------|---------|-------------|-------------|
| BCNet | 4.387 | 3.925 | 0.090 |
| SMPLicit | 4.080 | 3.605 | 0.091 |
| **DeClotH** | **3.902** | **3.292** | **0.079** |

## 评分
- 新颖性: ⭐⭐⭐⭐ ClothDiffusion 和模板正则化的组合有效
- 实验充分度: ⭐⭐⭐⭐ 与多种基线对比+详细消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 可分解重建对时尚/虚拟试穿有直接价值

<!-- RELATED:START -->

## 相关论文

- [GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data](../../NeurIPS2025/image_generation/geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)
- [DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](../../ICCV2025/image_generation/dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [FaceCraft4D: Animated 3D Facial Avatar Generation from a Single Image](../../ICCV2025/image_generation/facecraft4d_animated_3d_facial_avatar_generation_from_a_single_image.md)

<!-- RELATED:END -->
