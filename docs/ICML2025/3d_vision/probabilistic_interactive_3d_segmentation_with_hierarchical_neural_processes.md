---
title: >-
  [论文解读] Probabilistic Interactive 3D Segmentation with Hierarchical Neural Processes
description: >-
  [ICML 2025][3D视觉][交互式3D分割] NPISeg3D提出了首个基于层次化神经过程（Hierarchical Neural Processes）的概率交互式3D分割框架，通过场景级和物体级双层潜变量结构以及概率原型调制器，在少量点击下实现了优于AGILE3D的分割精度…
tags:
  - "ICML 2025"
  - "3D视觉"
  - "交互式3D分割"
  - "神经过程"
  - "层次化潜变量"
  - "不确定性估计"
  - "少样本泛化"
---

# Probabilistic Interactive 3D Segmentation with Hierarchical Neural Processes

**会议**: ICML 2025  
**arXiv**: [2505.01726](https://arxiv.org/abs/2505.01726)  
**代码**: [https://jliu4ai.github.io/NPISeg3D_projectpage/](https://jliu4ai.github.io/NPISeg3D_projectpage/)  
**领域**: 3D视觉  
**关键词**: 交互式3D分割, 神经过程, 层次化潜变量, 不确定性估计, 少样本泛化

## 一句话总结
NPISeg3D提出了首个基于层次化神经过程（Hierarchical Neural Processes）的概率交互式3D分割框架，通过场景级和物体级双层潜变量结构以及概率原型调制器，在少量点击下实现了优于AGILE3D的分割精度，同时提供可靠的不确定性估计。

## 研究背景与动机

**领域现状**：交互式3D分割通过用户提供的点击（正/负点击）在复杂3D场景中生成精确的物体mask。现有方法如InterObject3D、AGILE3D主要基于确定性模型，关注注意力机制和多目标分割能力。

**现有痛点**：（1）**少样本泛化不足**：用户期望少量点击就获得精确分割，但现有方法在稀疏输入下泛化能力有限，尤其在复杂场景和多样物体中；（2）**缺少不确定性估计**：现有方法（如AGILE3D）完全忽视了预测不确定性，无法告诉用户哪些区域的预测可能不可靠，这在高风险场景（如医学影像、自动驾驶）中是严重缺陷。

**核心矛盾**：确定性模型天然无法量化预测不确定性，而简单的概率模型（如单一潜变量的NP）在复杂多物体场景中难以同时捕获全局场景结构和物体级特征。

**本文目标**：构建一个概率框架，同时解决少样本泛化和不确定性估计两个挑战。

**切入角度**：神经过程（Neural Processes）天然适合少样本泛化和不确定性估计——将用户点击视为context set，未标注3D点视为target set。但单层NP在多物体场景中表达能力不足。

**核心 idea**：引入层次化潜变量结构（场景级 + 物体级），并通过概率原型调制器将潜变量信息注入点击原型，实现"Scene→Objects→Clicks"的多粒度信息流。

## 方法详解

### 整体框架
输入为3D点云场景 $\mathbf{S} \in \mathbb{R}^{N \times 6}$（坐标+颜色）和用户点击集合。点编码器提取场景特征 $\mathbf{X}_T$（target set）和点击原型 $\mathbf{X}_C$（context set）。通过层次化潜变量推断和概率原型调制后，生成分割mask。推理时通过多次Monte Carlo采样实现不确定性估计。

### 关键设计

1. **层次化潜变量结构**:

    - 功能：捕获多粒度的场景上下文信息，增强少样本泛化
    - 核心思路：引入场景级潜变量 $\mathbf{z}_s$ 建模全局上下文和物体间关系，以及物体级潜变量 $\mathbf{z}_o^m$ 捕获各物体的细粒度特征。场景级潜变量通过scene-level aggregator推断——先对每个物体的点击原型求平均得到物体级原型，再汇聚为场景级原型，通过Transformer和MLP生成 $\mathcal{N}(\mu_s, \sigma_s)$。物体级潜变量以 $\mathbf{z}_s$ 为条件：$[\mu_o^m, \sigma_o^m] = \text{MLP}(\alpha \mathbf{z}_s + (1-\alpha) \sum_i \mathbf{X}_C^{m,i})$，其中 $\alpha$ 平衡场景级和物体级信息
    - 设计动机：单一潜变量在多物体复杂场景中难以同时建模全局结构和物体细节，层次结构让信息从粗到细逐层传递

2. **概率原型调制器（Probabilistic Prototype Modulator）**:

    - 功能：将物体级潜变量注入点击原型，增强其上下文感知能力和不确定性建模
    - 核心思路：通过FiLM风格的特征调制 $\tilde{\mathbf{X}}_C^{m,i,j} = \gamma(\mathbf{z}_o^{m,j}) \odot \mathbf{X}_C^{m,i} + \beta(\mathbf{z}_o^{m,j})$，其中 $\gamma, \beta$ 由MLP从第j次Monte Carlo采样的 $\mathbf{z}_o^{m,j}$ 生成。每个点击原型被物体级上下文"调制"，从而获得更强的分类能力
    - 设计动机：原始点击原型只有局部信息，通过潜变量调制注入全局和物体级语义。多次采样产生不同的调制结果，自然实现不确定性量化

3. **基于ELBO的训练目标**:

    - 功能：联合优化分割精度和潜变量分布
    - 核心思路：训练时用target set推断后验分布 $q(\mathbf{z}_s|\mathbf{X}_T)$ 和 $q(\mathbf{z}_o^m|\mathbf{z}_s, \mathbf{X}_T^m)$，通过KL散度约束先验（从context set推断）逼近后验。总损失 $\mathcal{L} = \mathcal{L}_{seg} + \lambda_{kl}(D_{KL}^{scene} + \sum_m D_{KL}^{object_m})$，分割损失用Dice+CE
    - 设计动机：变分推断框架确保先验在推理时能从少量点击有效推断有意义的潜变量

### 损失函数 / 训练策略
总损失包含分割损失（Dice + Cross-Entropy）和两级KL散度正则化。$\lambda_{kl}$ 平衡分割精度与潜变量分布约束。推理时从先验采样 $N_{z_o}$ 次Monte Carlo样本，通过余弦相似度计算分割logits并取均值。不确定性通过多次采样的方差获得。

## 实验关键数据

### 主实验

| 数据集 (ScanNet40→) | 指标 | NPISeg3D | AGILE3D | 提升 |
|---------------------|------|----------|---------|------|
| S3DIS-A5 (多物体) | Avg IoU↑ | 90.5 | 88.3 | +2.2 |
| S3DIS-A5 (多物体) | Avg NoC↓ | 5.0 | 6.2 | -1.2 |
| KITTI-360 (多物体) | Avg IoU↑ | 48.5 | 44.3 | +4.2 |
| KITTI-360 (多物体) | Avg NoC↓ | 17.0 | 18.2 | -1.2 |
| ScanNet (单物体) | Avg IoU↑ | 88.2 | 87.1 | +1.1 |
| Replica (多物体) | Avg IoU↑ | 88.5 | 86.9 | +1.6 |

### 消融实验

| 配置 | Avg IoU (S3DIS) | 说明 |
|------|----------------|------|
| Full NPISeg3D | 90.5 | 完整模型 |
| w/o 层次潜变量 | ~88.5 | 去掉场景级z_s后退化至单层NP |
| w/o 概率原型调制 | ~89.0 | 直接用原始点击原型 |
| w/o KL正则化 | ~88.0 | 潜变量分布不受约束 |

### 关键发现
- 在域外数据上优势最大：KITTI-360（户外LiDAR）上比AGILE3D高4.2% IoU，说明概率框架增强了泛化能力
- 层次潜变量贡献最大：场景级z_s对多物体分割尤其关键，因为它编码了物体间的空间关系
- 不确定性估计可靠：高不确定性区域与实际分割错误区域高度相关，可有效指导用户后续点击

## 亮点与洞察
- **NP用于分割的首次尝试**：将交互式分割优雅地映射到NP的context/target框架中（点击=context，未标注点=target），这个formulation可以迁移到2D交互式分割、医学图像分割等场景
- **FiLM调制+概率采样**：概率原型调制器巧妙地把FiLM（feature-wise linear modulation）和Monte Carlo采样结合，单一机制同时实现上下文增强和不确定性估计，设计非常简洁
- **层次结构的必要性**：多物体场景需要全局理解（哪些物体在哪里）和局部理解（每个物体的具体特征），这个洞察对其他需要多粒度推理的任务也有启示

## 局限与展望
- 实验使用ScanNetV2训练，未探索大规模点云数据预训练的可能性
- Monte Carlo采样增加了推理计算量，实时交互场景中的效率需要优化
- 不确定性估计虽然可靠，但论文未探索如何自动利用不确定性来选择下一次最优点击位置（可以做主动学习）
- 层次结构固定为两层，更深的层次（如part-level）可能在更复杂场景中有用

## 相关工作与启发
- **vs AGILE3D (Yue et al. 2023)**：AGILE3D用注意力机制做多物体分割但完全是确定性的，NPISeg3D通过概率框架实现了更好的泛化和不确定性估计，尤其在域外数据上
- **vs 标准NP (Garnelo et al. 2018)**：标准NP只有单一全局潜变量，在多物体场景中表达力不足；NPISeg3D通过层次化潜变量解决了这个问题
- **vs InterPCSeg (Zhang et al. 2024)**：InterPCSeg集成语义分割网络做测试时纠正，但仍是确定性的，且不提供不确定性信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将NP引入交互式3D分割，层次化设计合理且有效
- 实验充分度: ⭐⭐⭐⭐ 4个数据集、多个baselines、消融实验和不确定性分析完整
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，数学推导严谨
- 价值: ⭐⭐⭐⭐ 概率交互式分割是重要方向，框架有广泛迁移潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](../../ICCV2025/3d_vision/easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[CVPR 2026\] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](../../CVPR2026/3d_vision/gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)
- [\[ECCV 2024\] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](../../ECCV2024/3d_vision/click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)
- [\[ICML 2025\] ReferSplat: Referring Segmentation in 3D Gaussian Splatting](refersplat_referring_segmentation_in_3d_gaussian_splatting.md)
- [\[ICML 2025\] Thickness-aware E(3)-Equivariant 3D Mesh Neural Networks](thickness-aware_e3-equivariant_3d_mesh_neural_networks.md)

</div>

<!-- RELATED:END -->
