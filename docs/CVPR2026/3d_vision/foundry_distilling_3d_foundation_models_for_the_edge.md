---
title: >-
  [论文解读] Foundry: Distilling 3D Foundation Models for the Edge
description: >-
  [CVPR 2026][3D视觉][基础模型蒸馏] 提出 Foundation Model Distillation（FMD）范式和 Foundry 框架，通过 compress-and-reconstruct 目标让学生模型学习一组可学习的 SuperToken 来压缩教师的潜空间基向量，生成的单一蒸馏模型在分类、分割、少样本等多任务上保持通用性，同时将 FLOPs 从 478G 降至最低 137G。
tags:
  - CVPR 2026
  - 3D视觉
  - 基础模型蒸馏
  - 3D点云
  - SuperToken
  - 表征空间压缩
  - 边缘部署
---

# Foundry: Distilling 3D Foundation Models for the Edge

**会议**: CVPR 2026  
**arXiv**: [2511.20721](https://arxiv.org/abs/2511.20721)  
**代码**: 无  
**领域**: 3D视觉 / 模型压缩  
**关键词**: 基础模型蒸馏, 3D点云, SuperToken, 表征空间压缩, 边缘部署

## 一句话总结

提出 Foundation Model Distillation（FMD）范式和 Foundry 框架，通过 compress-and-reconstruct 目标让学生模型学习一组可学习的 SuperToken 来压缩教师的潜空间基向量，生成的单一蒸馏模型在分类、分割、少样本等多任务上保持通用性，同时将 FLOPs 从 478G 降至最低 137G。

## 研究背景与动机

**领域现状**：自监督学习（SSL）预训练的基础模型已成为强大的通用特征提取器，在 3D 点云领域（如 Point-BERT、Point-JEPA）表现尤为突出，广泛应用于机器人、自动驾驶、AR/VR 等场景。这些模型通过在大规模无标注数据上预训练，获得了对各种下游任务的强泛化能力。

**现有痛点**：这些基础模型体量巨大（数亿参数+二次注意力复杂度），在边缘设备（如机器人、AR 头显）上无法运行。即使是现代 GPU，在处理 30 万点的中等规模点云时也可能 OOM。现有的知识蒸馏（KD）方法虽然可以创建高效的学生模型，但它们产出的是"专家模型"——擅长特定任务但失去了基础模型核心的下游无关通用性。

**核心矛盾**：标准知识蒸馏在特定任务的 logit 上训练学生，创建的学生只继承了教师在该任务上的行为，不具备跨任务迁移能力。这违背了基础模型的核心价值——通用表征能力。一个理想的蒸馏方法应该保留教师的整个表征空间，而非仅保留其在特定任务上的输出。

**本文目标**：设计一种新的蒸馏范式，将大型 SSL 基础模型压缩为紧凑、高效、忠实的代理模型，保留其通用表征能力。

**切入角度**：不直接模仿教师的特征嵌入（feature mimicry），而是通过信息瓶颈强制学生学习教师潜空间的紧凑基向量——先压缩成少量 SuperToken，再从中重建教师的完整 token 级表征。

**核心 idea**：用"压缩-重建"代替"模仿"，让学生学到的不是教师的某个输出，而是能高效表示教师整个潜空间的一组基向量。

## 方法详解

### 整体框架

Foundry 的蒸馏过程分三步：(1) **教师前向传播**——冻结的预训练教师处理输入点云，产出目标表征 $\mathbf{Y} \in \mathbb{R}^{c \times d}$；(2) **学生压缩与重建**——DSO 模块将 $c$ 个 token 压缩为 $s \ll c$ 个 SuperToken，轻量学生编码器处理后，CAU 模块从中重建教师的完整表征 $\hat{\mathbf{Y}} \in \mathbb{R}^{c \times d}$；(3) **蒸馏优化**——最小化 $\mathcal{L}_{distillation} = \text{SmoothL1}(\hat{\mathbf{Y}}, \mathbf{Y})$。

### 关键设计

1. **动态 SuperToken 优化模块（Dynamic Supertoken Optimization, DSO）**:

    - 功能：将输入的 $c$ 个 token 压缩为 $s$ 个可学习的 SuperToken
    - 核心思路：维护一组随机初始化的可学习 SuperToken $\mathbf{S} \in \mathbb{R}^{s \times d}$，作为潜空间的基向量集合。通过交叉注意力机制（SuperToken 为 query，输入 token 为 key/value）计算硬分配矩阵 $\text{CAM}_{j,i} = 1$ 当 $i = \arg\max_k \frac{\mathbf{q}_k \cdot \mathbf{k}_j}{\sqrt{d}}$。然后对每个 SuperToken，聚合所有被分配到它的 token 的 value 向量的均值来更新自身：$\mathbf{S}_{updated} = \frac{\text{CAM}^T \mathbf{V}}{\text{sum}(\text{CAM}^T, \text{axis}=1)}$。使用 Gumbel-Softmax 保证可微性
    - 设计动机：与静态 K-Means 聚类不同，可学习的 SuperToken 通过端到端训练能适应蒸馏目标，学到真正信息密集的潜空间基。语义分组在位置编码加入前进行，确保 SuperToken 基于内容而非位置进行特征压缩

2. **交叉注意力上采样模块（Cross-Attention Upsampling, CAU）**:

    - 功能：从压缩的 SuperToken 重建教师的完整 token 级表征
    - 核心思路：复用 DSO 阶段计算的分配矩阵 CAM 作为路由机制。每个原始 token 位置通过 CAM 查找对应的 SuperToken 的更新表征，然后与原始输入 token 做残差连接，最后通过 MLP 映射到教师的表征维度：$\hat{\mathbf{Y}} = \text{MLP}(\mathbf{T} + \text{CAM} \cdot \mathbf{S}_{encoder\_out})$
    - 设计动机：残差连接至关重要——它重新注入了压缩过程中可能丢失的局部高频细节信息，确保高保真重建。CAM 的复用避免了额外的计算开销

3. **门控压缩机制（Gated Compression, 可选）**:

    - 功能：在推理时实现动态、按需的计算预算控制
    - 核心思路：添加一个 2 层 MLP 门控网络，对每个输入 token 预测融合概率 $\pi_i$。只有 $\pi_i > r$（用户定义阈值）的 token 才通过 DSO 压缩，其余 token 绕过压缩直接与 SuperToken 一起送入学生编码器。训练时加入正则项 $\mathcal{L}_{gate} = -\lambda_{gate} \sum_i \pi_i$ 鼓励更多压缩
    - 设计动机：不同场景对精度和速度的需求不同。门控机制允许部署时通过调整阈值 $r$ 在精度和计算量之间灵活权衡，无需重新训练

### 损失函数 / 训练策略

- 核心蒸馏损失：$\mathcal{L}_{distillation} = \text{SmoothL1}(\hat{\mathbf{Y}}, \mathbf{Y})$
- 门控版本：$\mathcal{L} = \mathcal{L}_{distillation} + \mathcal{L}_{gate}$
- 训练在 ShapeNet55 上进行 150 epochs，学生编码器初始化自教师权重，可选冻结与否
- 教师均为 ViT-S 架构的 Point-JEPA 模型

## 实验关键数据

### 主实验（通用模型 vs 专家模型）

| 方法 | ShapeNet55 分类 Acc | ShapeNetPart 分割 mIoU_C/mIoU_I |
|--------|------|------|
| 教师 (Point-JEPA) | 90.54 | 83.91/85.73 |
| Foundry (通用, 16 SuperToken) | 89.87 | 81.87/84.82 |
| 专家-分类 (KD蒸馏) | 75.09 | - |
| 专家-分割 (KD蒸馏) | - | 61.88/65.72 |

### SuperToken 机制消融

| 方法 | ShapeNet55 Acc |
|------|---------|
| Foundry (可学习 DSO+CAU) | 89.68 |
| KMeans-Student (静态聚类) | 76.08 |
| FPS-Student (预采样) | 87.56 |

### 关键发现

- **FMD 通用模型完胜专家蒸馏**：单一通用学生在两个任务上都保持高性能（89.87%/81.87%），而专家学生在其原生任务上反而崩溃（分类专家仅 75.09%、分割专家仅 61.88%），证明了 FMD 范式的优越性
- **可学习 SuperToken 远优于静态方法**：DSO 比 K-Means 高出 13.6%（89.68 vs 76.08），证明端到端学习基向量的必要性
- **极致压缩仍保持有效**：仅用 1 个 SuperToken 时，Foundry 在 10-shot 分类中仍达 91.8%，接近教师的 96.1%
- **边缘部署可行**：FLOPs 从 478G 降至 137-178G（$s$=1~16），延迟从 0.09s 降至 0.05-0.06s。在 6GB GPU 上处理 30 万点大场景，教师和 ToMe 均 OOM，Foundry 仅需 4.0GB
- 蒸馏损失与下游精度高度相关（明确的反相关），$s \leq 4$ 后收益递减，意味着仅需非常少的 SuperToken 即可充分跨越教师的潜空间

## 亮点与洞察

- **范式创新：从 task-specific KD 到 representation distillation**：这是本文最重要的贡献。传统 KD 创建的是任务专家，而 FMD 创建的是基础模型的微型代理。这种范式差异对所有需要在边缘部署基础模型的场景都有意义
- **compress-and-reconstruct 的信息瓶颈设计**：强制学生通过极窄的 SuperToken 瓶颈重建教师表征，比直接 L2 特征模仿更能捕捉潜空间的结构。这一思路可迁移到 2D 图像基础模型（如 DINOv2、SAM）的蒸馏
- **与现有 token 压缩方法兼容**：Foundry 的 FMD 框架可以与 ToMe、PiToMe、PatchMerger 组合使用并进一步提升性能

## 局限与展望

- 仅验证了单一教师（Point-JEPA ViT-S），对其他 3D 基础模型（如 Point-MAE、PointGPT）和更大规模架构（ViT-B/L）的泛化性未知
- 仅覆盖 3D 点云领域，向 2D 图像、视频基础模型的扩展是重要的未来方向
- 门控机制虽然提供了推理时的灵活性，但其性能对训练时 $\lambda_{gate}$ 的选择敏感
- DSO 的硬分配可能限制了表征质量——被错误分配的 token 无法修正（虽然 CAU 的残差连接部分弥补了这一点）
- 未在真实的边缘设备（如 Jetson、手机）上进行端到端性能评测

## 相关工作与启发

- **vs TinyCLIP/CLIP-KD**：这些方法蒸馏的是 CLIP 的跨模态对齐能力（一种特定能力），而 Foundry 蒸馏的是 SSL 模型的整个表征空间（通用能力），理念根本不同
- **vs ToMe/PiToMe**：ToMe 在推理时在线合并 token 来加速，而 Foundry 是离线蒸馏创建全新的学生模型。两者可以互补——实验显示用 FMD 框架蒸馏 ToMe 学生可以获得最佳效果
- **vs 3DLST**：3DLST 也使用可学习 supertoken，但目标是特定分割任务的推理加速，而非创建通用代理模型。Foundry 将其思路升级为蒸馏目标

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 提出了 FMD 新范式并清晰论证了与传统 KD 和直接特征模仿的本质区别
- 实验充分度: ⭐⭐⭐⭐⭐ 6+1 个数据集, 通用vs专家对比, 多种 SuperToken 数量, 门控变体, 计算量分析, 大场景测试
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精炼、实验层次清晰、与相关工作的区分到位
- 价值: ⭐⭐⭐⭐⭐ 对 3D 基础模型的边缘部署有直接推动作用，FMD 范式具有广泛迁移潜力

<!-- RELATED:START -->

## 相关论文

- [NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration](nanosd_edge_efficient_foundation_model_for_real_time_image_restoration.md)
- [AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](ava_bench_atomic_visual_ability_vfm.md)
- [E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [Global-Aware Edge Prioritization for Pose Graph Initialization](global-aware_edge_prioritization_for_pose_graph_initialization.md)

<!-- RELATED:END -->
