---
title: >-
  [论文解读] Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot
description: >-
  [ECCV 2024][3D视觉][人体网格恢复] Multi-HMR是首个单阶段多人全身（含手部和面部表情）3D人体网格恢复方法，使用ViT骨干网络和交叉注意力预测头（HPH），结合新的CUFFS合成数据集解决手部姿态学习困难，在多人和全身两类基准上同时达到SOTA。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "人体网格恢复"
  - "多人全身姿态估计"
  - "单阶段检测"
  - "SMPL-X"
  - "合成数据集"
---

# Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot

**会议**: ECCV 2024  
**arXiv**: [2402.14654](https://arxiv.org/abs/2402.14654)  
**代码**: [https://github.com/naver/multi-hmr](https://github.com/naver/multi-hmr)  
**领域**: 3D视觉  
**关键词**: 人体网格恢复, 多人全身姿态估计, 单阶段检测, SMPL-X, 合成数据集

## 一句话总结

Multi-HMR是首个单阶段多人全身（含手部和面部表情）3D人体网格恢复方法，使用ViT骨干网络和交叉注意力预测头（HPH），结合新的CUFFS合成数据集解决手部姿态学习困难，在多人和全身两类基准上同时达到SOTA。

## 研究背景与动机

**领域现状**：人体网格恢复（HMR）经过多年发展已在多个方面取得进展。单人HMR方法（如HMR、HMR2.0）从裁剪图像回归SMPL参数；全身方法（如PIXIE、Hand4Whole）通过多级裁剪分别处理身体、手部、面部；多人方法（如ROMP、BEV、PSVT）实现了单阶段多人检测但仅限body-only。然而没有任何方法同时解决了四大需求：全身预测、多人处理、相机空间定位、相机内参自适应。

**现有痛点**：(1) 多人+全身的组合特别困难——手部和面部通常在自然图像中分辨率很低，单阶段方法难以从全图直接学到细粒度的手部姿态。(2) 已有的全身方法依赖多级裁剪流程（先检测人、再裁出手/脸区域），这增加计算量且无法端到端学习。(3) 现有多人方法不支持全身预测（面部表情+手部姿态）。(4) 大多数方法使用固定的相机假设，无法适应不同的相机内参。

**核心矛盾**：单阶段方法虽然高效且可端到端训练，但从低分辨率的全局特征中回归手部和面部的细粒度参数极其困难；多级裁剪方法能捕获细节但引入了检测流水线的误差传播和效率问题。

**本文目标** (1) 如何在单阶段框架中同时实现多人检测和全身（含手指+面部）参数回归？(2) 如何解决训练数据中手部信息不足、不可见的问题？

**切入角度**：作者采用简洁的Transformer-first设计——ViT作为骨干，交叉注意力头让每个检测到的人体token关注全图特征来预测全身参数。同时引入CUFFS合成数据集，专门包含近距离拍摄、手部清晰可见的全身人体图像来增强训练。

**核心 idea**：用ViT+交叉注意力头实现单阶段多人全身网格回归，配合近距离手部合成数据集弥补手部训练数据不足。

## 方法详解

### 整体框架

输入一张RGB图像，ViT骨干提取patch级别的token嵌入 $\mathbf{E} \in \mathbb{R}^{H/P \times W/P \times D}$。第一步通过CenterNet范式预测人体中心热力图检测人物。第二步，检测到的中心token作为query进入Human Perception Head（HPH），通过交叉注意力聚合全图特征后回归SMPL-X参数（身体姿态 $\boldsymbol{\theta}$、体型 $\boldsymbol{\beta}$、面部表情 $\boldsymbol{\alpha}$）和深度 $t_z$。可选地编码相机光线方向进行相机感知预测。

### 关键设计

1. **Human Perception Head（HPH）**:

    - 功能：从全图特征中高效地回归每个检测到的人体的全身SMPL-X参数
    - 核心思路：对 $N$ 个检测到的人体，初始化 $N$ 个query向量 $\mathbf{q}_n = (\mathbf{E}_{i,j} \oplus \bar{\mathbf{x}}) + \mathbf{p}_{ij}$，其中拼接了token嵌入和均值body参数。通过 $L=2$ 层交叉注意力块依次处理：$\mathbf{Q}^l = \text{MLP}_l(\text{SA}_l(\text{CA}_l[\mathbf{Q}^{l-1}, \mathbf{E}]))$，其中CA让每个人的query关注所有图像token，SA让不同人的query互相交互
    - 设计动机：相比传统的逐token独立回归器，交叉注意力能让预测头访问全局上下文信息（如其他身体部位的线索），且不同人之间的自注意力有助于处理遮挡和深度排序。实验证明HPH比迭代回归器收敛更快、性能更好

2. **CUFFS合成数据集（Close-Up Frames of Full-Body Subjects）**:

    - 功能：提供手部姿态多样且清晰可见的近距离全身训练数据
    - 核心思路：使用HumGen3D在Blender中渲染60k张合成图像。人体放置在距相机约2.5米处确保手部可见，采样来自BEDLAM/AGORA/UBody的人体姿态，额外引入InterHand数据集的丰富手部姿态进行手部替换增强。通过SMPL-X到HumGen3D的网格对应关系实现精确的GT标注
    - 设计动机：现有合成数据集（BEDLAM、AGORA）中人物通常距镜头较远、手部仅占几个像素，且手部姿态缺乏多样性。单阶段方法不依赖手部裁剪，因此需要训练数据本身就包含清晰的手部细节

3. **可选相机嵌入**:

    - 功能：当相机内参可用时，提升3D空间定位精度
    - 核心思路：对每个patch中心 $(u_i, v_j)$，通过 $\mathbf{r}_{i,j} = \mathbf{K}^{-1}[u_i, v_j, 1]^T$ 计算光线方向，取前两个分量进行Fourier编码，拼接到骨干输出的token嵌入上 $\mathbf{E} \doteq \mathbf{E} \oplus \mathbf{E}_K$
    - 设计动机：相机内参直接影响3D到2D的投影关系，简单线性嵌入效果反而下降，而Fourier编码光线方向配合焦距归一化带来一致的性能提升

### 损失函数 / 训练策略

总损失为 $\mathcal{L} = \mathcal{L}_\text{det} + \mathcal{L}_\text{params} + \lambda(\mathcal{L}_\text{mesh} + \mathcal{L}_\text{reproj})$。检测用二值交叉熵；回归用L1损失作用于SMPL-X参数、偏移量和深度；网格损失直接监督顶点3D坐标；2D重投影损失提供额外的约束。深度预测在对数空间中进行，使用归一化nearness参数化保证对焦距变化的鲁棒性。DINOv2初始化ViT骨干，batch size 8，学习率5e-5，400k迭代训练。

## 实验关键数据

### 主实验

多人body-only基准（3DPW上MPJPE / MuPoTs上PCK3D）：

| 方法 | 3DPW PA-MPJPE↓ | MuPoTs PCK3D↑ | CMU MPJPE↓ |
|------|----------------|---------------|------------|
| ROMP | 47.3 | 69.9 | 108.1 |
| BEV | 46.9 | 70.2 | 105.3 |
| PSVT | 45.7 | - | 97.7 |
| **Multi-HMR-448** | 43.8 | **80.6** | - |
| **Multi-HMR** | **41.7** | **85.0** | **82.8** |

全身基准EHF（PVE-All / PVE-Hands）：

| 方法 | PVE-All↓ | PA-PVE-All↓ | PVE-Hands↓ |
|------|----------|-------------|------------|
| Hand4Whole | 76.8 | 50.3 | 39.8 |
| OSX | 70.8 | 48.7 | 53.7 |
| **Multi-HMR** | **44.2** | **32.7** | **36.4** |

### 消融实验

| 配置 | MuPoTs PCK3D↑ | 3DPW MPJPE↓ | EHF PVE↓ | 说明 |
|------|---------------|-------------|----------|------|
| HRNet + Reg | 65.8 | 83.2 | 143.1 | CNN骨干+迭代回归 |
| ViT-S + Reg | 70.1 | 80.2 | 90.6 | ViT骨干+迭代回归 |
| ViT-S + HPH | 70.9 | 80.1 | 80.1 | ViT骨干+HPH |
| ViT-B + HPH | 76.3 | 73.5 | 55.3 | 更大骨干 |
| +CUFFS | 76.0 | 72.9 | 49.8 | 手部显著提升 |

### 关键发现

- ViT骨干显著优于同参数量的HRNet（EHF PVE从143.1降到90.6），ViT的全局注意力对全身预测至关重要
- HPH比迭代回归器收敛更快且性能更好，query间的自注意力也有正面贡献
- CUFFS数据集主要提升手部指标（EHF-H从47.4降到40.5），对其他指标影响很小
- DINOv2预训练一致优于其他预训练方式，收敛也更快
- 即使ViT-S在448×448分辨率下也具有竞争力（30fps实时）

## 亮点与洞察

- **四合一的统一框架**：首次在单个模型中实现多人+全身+相机空间+相机感知四大功能
- **合成数据的惊人效果**：纯合成训练优于混入真实数据——这挑战了"需要真实数据"的直觉
- **ViT+单阶段=赢家**：标准ViT架构可以直接受益于视觉社区的自监督预训练进展
- **HPH的设计简洁有效**：添加仅2层的交叉注意力显著优于复杂的多级流水线

## 局限与展望

- Patch级别检测限制了密集人群场景（两人中心在同一patch内会碰撞）
- 头部被遮挡时检测困难（因为使用头部作为主关节点）
- 远距离人物的手部和面部预测仍有改进空间
- SMPL-X的关节树旋转表示可能导致末端肢体误差累积
- 可以探索每个patch多query方案来处理拥挤场景

## 相关工作与启发

- **ROMP/BEV**：单阶段多人方法，使用HRNet骨干，仅body-only
- **OSX**：单人全身方法，使用ViT但依赖关键点重采样高分辨率特征
- **BEDLAM**：证明了合成数据可以训练出SOTA模型
- 启发：简洁的架构（ViT+交叉注意力）加上恰当的合成数据，就能打败复杂的多级流水线

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个四合一统一框架，CUFFS数据集设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖6+评估基准，消融全面（架构/数据/损失/相机/分辨率）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设置详尽
- 价值: ⭐⭐⭐⭐ 实用性强，代码开源，推动多人全身HMR普及

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](../../AAAI2026/3d_vision/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](global-to-pixel_regression_for_human_mesh_recovery.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] Divide and Fuse: Body Part Mesh Recovery from Partially Visible Human Images](divide_and_fuse_body_part_mesh_recovery_from_partially_visible_human_images.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)

</div>

<!-- RELATED:END -->
