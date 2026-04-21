---
title: >-
  [论文解读] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs
description: >-
  [3D视觉] GaussianProperty 提出了一个免训练框架，利用 SAM 分割和 GPT-4V 识别能力，通过全局-局部推理模块和多视角投票策略，将物理属性（密度、弹性模量、摩擦系数等）赋予 3D Gaussians，支持物理仿真和机器人抓取两大下游任务。
tags:
  - 3D视觉
---

# GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2412.11258](https://arxiv.org/abs/2412.11258)
- **代码**: [项目页面](https://Gaussian-Property.github.io)
- **领域**: 3D视觉
- **关键词**: 3D Gaussian Splatting, 物理属性估计, 大型多模态模型, 机器人抓取, 动态仿真

## 一句话总结
GaussianProperty 提出了一个免训练框架，利用 SAM 分割和 GPT-4V 识别能力，通过全局-局部推理模块和多视角投票策略，将物理属性（密度、弹性模量、摩擦系数等）赋予 3D Gaussians，支持物理仿真和机器人抓取两大下游任务。

## 研究背景与动机

从视觉数据中估计物理属性是计算机视觉和图形学中的关键任务，对增强现实、物理仿真和机器人抓取等应用至关重要。然而该领域仍未被充分探索，原因包括：

**标注数据获取困难**：物理属性（如密度、弹性模量）无法从视觉表面直接观测，缺乏标注的真实标签数据

**预测任务存在歧义**：仅从有限的可观察表面推断内在物理属性具有固有模糊性

**现有方法局限性**：已有工作大多针对特定类型的材料属性（如质量或硬度），需要特定任务的标注数据，泛化能力有限

作者观察到，人类能够根据视觉线索预测物理属性，这种能力源于将外观与已知材料的先验知识关联。而大型多模态模型（LMMs）如 GPT-4V 拥有大量先验知识，展示出类人的视觉识别能力，可用于物理属性估计。

## 方法详解

### 整体框架

GaussianProperty 的整体流程分为三个阶段：
1. **部件级分割**：利用 SAM 对多视角图像进行部件级语义分割
2. **物理属性匹配**：通过全局-局部推理模块和 GPT-4V 识别材料并查询物理属性
3. **2D到3D投影**：利用多视角投票策略将 2D 物理属性提升到 3D Gaussians

### 关键设计一：部件级分割

对每张观测图像 $I \in \mathcal{I}^N$，输入 $32 \times 32$ 的点提示网格到 SAM，获取不同层级的精确分割掩码 $\mathbf{M}$。通过 IoU 分数、稳定性分数和掩码重叠率剔除冗余掩码，得到语义层级的部件级分割。

### 关键设计二：全局-局部物理属性推理

直接用全局图像查询物理属性效果不佳，模型难以建立全局与局部的关联。因此设计了三部分推理策略：

- **材料候选库**：构建包含 15 种材料族和 600+ 种具体材料的候选库，简化 LMM 的材料检索
- **全局-局部组合推理**：同时提供原始全局图像、带红色掩码标注的分割图、以及局部部件裁剪图，帮助 GPT-4V 建立部件与整体的关联
- **渐进式提示引导**：设计分步提示，先让 LMM 描述部件外观，再识别材料并输出质量密度 $\rho$、杨氏模量 $E$、泊松比 $P$ 等物理参数

### 关键设计三：基于投票的 2D 到 3D 提升

将每个 3D Gaussian $\mathbf{s}$ 通过相机参数投影到各可见的 2D 图像：

$$u, v = \mathbf{K} \cdot [\mathbf{R}|\mathbf{t}] \cdot \mathbf{s}$$

利用 Gaussian 估计的深度判断可见性，然后采用频率投票策略确定最终属性：

$$\hat{a} = \arg\max_a \sum_{i=1}^N \mathbb{I}(a_i = a)$$

其中 $N$ 为视角数，$a_i$ 为第 $i$ 个视角观测到的属性。

### 下游应用一：材料敏感的机器人抓取

基于预测的物理属性计算抓取力的上下界：

$$F_{\min} = \sum_{i=1}^M \frac{1}{2}\rho(i)V(i)g\left(\frac{\cos\theta}{\mu(s)} - \sin\theta\right)$$

$$F_{\max} = \min\left[A\sigma_y(s), \frac{1}{2}AE(s)d(s)\kappa_{\max}(s)\right]$$

确保抓取力足够抬起物体且不造成变形。

### 下游应用二：基于物理的动态仿真

将预测的物理属性（密度、杨氏模量、泊松比、材料类型）直接赋予 3D Gaussians，结合 MPM（Material Point Method）实现物理驱动的动态仿真，无需手动标注。

## 实验

### 主实验：材料分割

| 方法 | ABO 平均 mIoU | MVImgNet 平均 mIoU |
|------|--------------|-------------------|
| NeRF2Physics | 25.59 | 4.02 |
| **Ours** | **55.83** | **34.83** |

在 ABO 和 MVImgNet 数据集上，GaussianProperty 在所有材料类别上大幅超越 NeRF2Physics。

### 消融实验：全局-局部推理与投票策略

| 全局-局部推理 | 投票策略 | 平均 mIoU(%) |
|-------------|---------|-------------|
| ✗ | ✓ | 22.17 |
| ✓ | ✗ | 51.28 |
| ✓ | ✓ | **55.83** |

两个关键设计均显著贡献于最终性能，全局-局部推理提升 29 个百分点，投票策略额外提升 4.5 个百分点。

### 机器人抓取实验

| 方法 | PUR(%) | NDR(%) | SR(%) |
|------|--------|--------|-------|
| MinGF (固定最小力) | 低 | 高 | 低 |
| MidGF (固定中等力) | 中 | 中 | 中 |
| MaxGF (固定最大力) | 高 | 低 | 低 |
| **Ours (自适应力)** | **高** | **高** | **最高** |

在 16 个真实物体的抓取实验中，自适应抓取力策略在成功率上超越所有固定力基线。

### 关键发现
- GPT-4V 具备从视觉推断材料属性的能力，但需要精心设计的提示策略
- 全局-局部组合信息对准确识别材料至关重要
- 多视角投票有效避免单视角的偶发错误
- 物理属性注解使 3DGS 可无缝集成于仿真和机器人系统

## 亮点与洞察

1. **免训练框架**：完全利用 SAM 和 GPT-4V 的零样本能力，无需额外训练
2. **首次探索 LMM 用于 3D 物理属性估计**：展示了视觉基础模型在物理理解上的潜力
3. **实际应用价值**：在真实机器人平台验证了材料敏感抓取的有效性
4. **可扩展设计**：材料候选库和推理模块可灵活适配不同下游任务

## 局限性
- 依赖 GPT-4V 的 API 调用，成本较高且受限于模型的知识覆盖范围
- 对于外观相似但材料不同的物体（如金属漆面 vs 真金属），可能产生误判
- 投票策略假设视角间的属性预测是独立的，未考虑空间连续性
- 动态仿真中的物理属性为离散赋值，未建模材料的连续渐变

## 相关工作
- NeRF2Physics：利用 LLM 提出候选材料并通过零样本核回归估计物理属性
- Make-it-Real：推理物体的 PBR 材料属性用于纹理生成
- PhysGaussian：将牛顿物理学整合到 3DGS 中实现动态仿真，但需手动赋予物理属性
- SAM / GPT-4V：分别提供精确分割和多模态理解能力

## 评分
- **创新性**: ⭐⭐⭐⭐ — 首个将 LMM 用于 3D 物理属性估计的免训练框架
- **实用性**: ⭐⭐⭐⭐ — 真实机器人抓取验证，应用前景广阔
- **实验完整度**: ⭐⭐⭐⭐ — 材料分割/动态仿真/机器人抓取三方面验证
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机明确

<!-- RELATED:START -->

## 相关论文

- [DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness](dso_aligning_3d_generators_with_simulation_feedback_for_physical_soundness.md)
- [SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)
- [Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)

<!-- RELATED:END -->
