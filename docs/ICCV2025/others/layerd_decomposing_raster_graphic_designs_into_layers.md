---
title: >-
  [论文解读] LayerD: Decomposing Raster Graphic Designs into Layers
description: >-
  [ICCV 2025][图层分解] 提出 LayerD，通过迭代提取未遮挡顶层和背景补全来分解栅格图形设计为可编辑图层，并利用图形设计的域先验（纹理平坦区域）进行精炼，同时提出了基于 DTW 的层级评估协议。
tags:
  - ICCV 2025
  - 图层分解
  - 图形设计
  - 迭代抠图
  - 调色板优化
  - 评估协议
---

# LayerD: Decomposing Raster Graphic Designs into Layers

**会议**: ICCV 2025  
**arXiv**: [2509.25134](https://arxiv.org/abs/2509.25134)  
**代码**: [https://cyberagentailab.github.io/LayerD/](https://cyberagentailab.github.io/LayerD/)  
**领域**: 其他（图形设计 / 图像分解）  
**关键词**: 图层分解, 图形设计, 迭代抠图, 调色板优化, 评估协议

## 一句话总结

提出 LayerD，通过迭代提取未遮挡顶层和背景补全来分解栅格图形设计为可编辑图层，并利用图形设计的域先验（纹理平坦区域）进行精炼，同时提出了基于 DTW 的层级评估协议。

## 研究背景与动机

设计师在 Photoshop/PowerPoint 等工具中以图层为基本单位创建和编辑图形设计。一旦合成为栅格图像，图层信息丢失，使编辑和再利用变得困难。将栅格图像逆向分解为图层序列（即合成的逆问题）可以重新启用基于图层的编辑工作流。

然而图形设计的图层分解面临独特挑战：

**元素多样性**：图形设计包含排版、装饰、矢量图形、插图甚至自然图像素材的混合

**与自然图像的差异**：直接应用自然场景分解方法会导致不期望的分解（如照片素材中的物体被错误分离）或伪影（背景光照影响纯色矢量图形）

**固有的 ill-posed 性**：存在多种可能的解，一个图层可以被任意分割为多个图层，导致评估困难

现有方法（如 MULAN、Accordion）采用堆叠流水线（检测→分割→排序→补全），虽然可以利用各阶段预训练模型，但不可避免地累积误差。

## 方法详解

### 整体框架

LayerD 将分解任务公式化为**迭代的顶层抠图 + 背景补全**过程。从前到后（$m=M$ 到 $m=1$）逐层提取：
1. 抠图模型 $F_\theta$ 预测当前图像的顶层 alpha map
2. 背景补全模型 $G_\phi$ (LaMa) 补全被提取区域
3. 通过逆 alpha blending 计算前景 RGB 值
4. 重复直到 alpha map 无显著前景

这种设计将检测、分割和图层排序**统一为一个端到端任务**，避免了堆叠流水线的误差累积。

### 关键设计

1. **迭代顶层抠图 (Iterative Top-Layer Matting)**：
   
   训练一个 trimap-free 的抠图模型（基于 BiRefNet + Swin-L），在 Crello 数据集上通过有监督学习。训练目标是明确的"未被遮挡的顶层"——通过检查图层的遮挡关系，将所有未遮挡图层的 alpha map 合并为单一目标。这种清晰的目标定义消除了训练中的歧义。

   关键改进：在训练数据中加入背景补全模型处理过的样本（先用 LaMa 补全顶层区域再作为输入），使抠图模型对补全伪影鲁棒，弥合训练-推理的分布差异。

   损失函数：$\mathcal{L} = \lambda_{BCE}\mathcal{L}_{BCE} + \lambda_{IoU}\mathcal{L}_{IoU} + \lambda_{SSIM}\mathcal{L}_{SSIM}$，先用全部损失训练，后期仅用 SSIM 损失提升边界质量。

2. **调色板精炼 (Palette-based Refinement)**：

   利用图形设计的域先验——**大量纹理平坦（flat）区域**（如纯色背景、矢量形状、文字）：

   - **背景精炼**：将补全目标区域按连通域分割，计算周围区域的颜色梯度。若零梯度区域占主导，提取主色调（palette），将补全结果映射到最近的 palette 颜色（在 Lab 空间）。消除补全模型在平坦区域产生的伪影。
   
   - **前景精炼**：同样分析连通域的颜色梯度，对被分类为平坦区域的部分：从原图或中间补全背景中提取与 palette 颜色匹配的区域，若与预测 alpha 的重叠度超过阈值，用该区域生成新的 alpha mask。显著改善边界质量和细装饰层（线条、边框）的检测。

3. **前景颜色估计 (Foreground Color Estimation)**：

   不同于现有方法简单用分割 mask 替换原图 alpha，LayerD 通过已知的 alpha map 和补全背景，利用逆 alpha blending 精确计算前景 RGB：

   $$\hat{l}_m^C = \frac{\hat{x}_m^C - \hat{x}_{m-1}^C \odot (1 - \hat{l}_m^A)}{\hat{l}_m^A}$$

   对于透明像素（alpha < 1），这比直接替换能更好地处理前景-背景混合。

### 损失函数 / 训练策略

- 抠图模型基于 BiRefNet + Swin-L，在 Crello 训练集上训练 60 epochs，batch size 12
- 背景补全使用现成的 LaMa 模型，不做微调
- 最大迭代次数设为 3
- 调色板最大颜色数：前景 10 色，背景 2 色

## 实验关键数据

### 主实验

基于自定义评估协议（DTW 对齐 + 层级编辑距离），在 Crello 测试集上评估：

| 方法 | RGB L1↓ (0 edits) | Alpha IoU↑ (0 edits) | RGB L1↓ (3 edits) | Alpha IoU↑ (3 edits) |
|------|-------------------|---------------------|-------------------|---------------------|
| YOLO baseline | ~0.055 | ~0.42 | ~0.045 | ~0.52 |
| VLM baseline | ~0.050 | ~0.45 | ~0.040 | ~0.55 |
| BiRefNet (无额外训练) | ~0.045 | ~0.50 | ~0.038 | ~0.58 |
| LayerD (无文本训练) | ~0.032 | ~0.60 | ~0.025 | ~0.68 |
| LayerD + Hi-SAM | ~0.035 | ~0.57 | ~0.028 | ~0.65 |
| **LayerD** | **~0.030** | **~0.62** | **~0.023** | **~0.70** |

注：数据从论文 Figure 5 读取的近似值。LayerD 在所有指标和编辑距离下均为最优。

### 消融实验

| 配置 | RGB L1↓ | Alpha IoU↑ |
|------|---------|-----------|
| Naive (mask 替换) | baseline | baseline |
| + 逆 blending 颜色估计 | ↓ 显著改善 | - |
| + 背景精炼 | ↓↓ 大幅改善 | ↑ 后续层也改善 |
| + 前景精炼 | ↓↓ | ↑ 边界质量提升 |

额外发现：
- LayerD 单独使用比 LayerD + Hi-SAM 效果更好（专门训练的抠图模型优于通用文字分割）
- 包含文字层训练的 LayerD 在"排除文字"评估中也略优于不含文字训练版本（文字本质上是矢量形状的变体）

### 关键发现

- **统一流水线优于组件堆叠**：LayerD 统一了检测+分割+排序为迭代抠图的单一任务，在所有指标上系统性优于 YOLO 和 VLM baseline
- **域先验至关重要**：调色板精炼利用了图形设计中大量平坦区域的特性，显著消除伪影
- **训练-推理对齐有效**：在训练中加入补全后的图像作为输入，提升了对补全伪影的鲁棒性
- **评估协议创新**：基于 DTW 的层级对齐 + 编辑距离的度量方式，比逐层像素比较更合理
- **前景精炼改善了细节**：特别是对细线条和装饰边框的检测，plain 抠图模型容易失败
- LayerD 可以泛化到 FLUX.1 生成的图形设计图像

## 亮点与洞察

1. **问题定义到位**：图形设计图层分解是一个实际且重要的问题，先前工作极少
2. **简洁而有效**：用迭代抠图统一多个子任务，远比多阶段管道优雅
3. **评估协议贡献**：ill-posed 任务需要恰当的评估方式，DTW 对齐 + 编辑距离是一个有见地的设计
4. **域知识驱动优化**：调色板精炼虽然简单但极为有效，体现了对图形设计领域的深入理解
5. **应用价值明确**：分解后可直接进行颜色转换、平移、缩放等图层级编辑

## 局限与展望

- 最大迭代次数固定为 3，对于层数更多的复杂设计可能不够
- 透明图层的估计未被重点处理（论文明确排除）
- 调色板精炼对渐变色和复杂纹理的图形设计效果有限
- 仅在 Crello 数据集上训练和评估，泛化到其他设计风格（如印刷品、UI 设计）待验证
- 未考虑图层间的语义分组（如 logo 由图标+文字组成，是否作为一个图层）

## 相关工作与启发

- MULAN：自然图像分解，使用开放词汇检测+零样本分割的堆叠方案
- Accordion：同时期基于 VLM 的图形设计分解工作，但代码和模型未开源
- LaMa：高质量图像补全模型，LayerD 的关键组件
- BiRefNet：trimap-free 抠图模型，LayerD 的主干网络
- Color segmentation 方法：相关但目标不同（半透明颜色层 vs 对象图层）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 迭代顶层抠图的统一框架简洁优雅，调色板精炼巧妙利用域先验
- **实验充分度**: ⭐⭐⭐⭐ 多 baseline 对比、消融、定性分析充分，但仅一个数据集
- **写作质量**: ⭐⭐⭐⭐⭐ 问题定义清晰，方法描述详细，图示丰富直观
- **价值**: ⭐⭐⭐⭐ 对创意工作流有实际应用价值，评估协议对社区有贡献

<!-- RELATED:START -->

## 相关论文

- [Low-Rank Interconnected Adaptation across Layers](../../ACL2025/others/low-rank_interconnected_adaptation_across_layers.md)
- [Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss](../../ICML2025/others/enhancing_certified_robustness_via_block_reflector_orthogonal_layers_and_logit_a.md)
- [A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)
- [C4D: 4D Made from 3D through Dual Correspondences](c4d_4d_made_from_3d_through_dual_correspondences.md)
- [Membership Inference Attacks with False Discovery Rate Control](membership_inference_attacks_with_false_discovery_rate_control.md)

<!-- RELATED:END -->
