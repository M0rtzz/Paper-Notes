---
title: >-
  [论文解读] GAP: Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation
description: >-
  [CVPR2026][3D视觉][双臂操控] GAP利用预训练3D几何基础模型（π³）提取3D特征，融合2D语义和本体感知，通过条件扩散联合预测未来动作序列和未来3D pointmap，在RoboTwin 2.0和真实双臂实验中达到SOTA。
tags:
  - CVPR2026
  - 3D视觉
  - 双臂操控
  - 3D几何先验
  - 扩散策略
  - 点云预测
  - 模仿学习
---

# GAP: Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation

**会议**: CVPR2026  
**arXiv**: [2602.23814](https://arxiv.org/abs/2602.23814)  
**代码**: https://github.com/Chongyang-99/GAP.git  
**领域**: 3d_vision  
**关键词**: 双臂操控, 3D几何先验, 扩散策略, 点云预测, 模仿学习

## 一句话总结
GAP利用预训练3D几何基础模型（π³）提取3D特征，融合2D语义和本体感知，通过条件扩散联合预测未来动作序列和未来3D pointmap，在RoboTwin 2.0和真实双臂实验中达到SOTA。

## 研究背景与动机

**领域现状**：双臂操控（bimanual manipulation）需要策略同时生成两只机械臂的协调动作，涉及精密装配、形变物体操作和杂乱环境交互。当前主流方法包括：基于2D的ACT（action chunking + DETR Transformer）、扩散策略DP，以及引入3D的DP3（点云输入）。

**现有痛点**：
   - **2D方法缺乏空间感知**：ACT、DP等方法依赖2D特征，无法显式推理3D空间关系、遮挡和接触。在需要精确空间推理的双臂任务中表现不佳
   - **3D方法依赖显式点云**：DP3等需要深度相机生成点云，但真实世界中高质量点云获取需要精确标定、对噪声和遮挡敏感。2D→3D提升方法（如back-projection）分辨率低、工程开销大
   - **缺乏预测性3D推理**：现有方法只感知当前3D状态，不预测动作执行后的3D变化，限制了长horizon规划能力

**核心矛盾**：双臂操控需要3D感知能力来推理空间关系，但显式获取3D信息（点云）在真实场景中不够可靠。同时，仅感知当前状态不足以支持需要预测未来几何变化的复杂操控。

**本文目标**：能否直接利用3D几何基础模型从RGB图像获取隐式3D特征，绕过显式点云管线？能否通过联合预测未来3D结构来增强策略的空间理解和长horizon规划？

**切入角度**：最近3D几何基础模型（如DUSt3R、VGGT、π³）能从RGB图像快速、鲁棒地重建稠密3D结构。作者将π³作为感知backbone，其latent特征天然包含丰富的3D几何信息——不需要显式生成点云，直接用latent做策略条件。更进一步，通过预测"未来3D latent"迫使模型学习3D-aware的前瞻推理。

**核心 idea**：用预训练3D几何基础模型的latent作为3D先验，联合去噪未来动作和未来3D pointmap来实现RGB-only的3D-aware双臂操控策略。

## 方法详解

### 整体框架
输入：5帧历史RGB帧 $V$、当前帧 $I_t$、当前本体感知 $p_t \in \mathbb{R}^{14}$（双臂各6关节角+1 gripper状态）。输出：未来N步双臂动作序列 $a_{t:t+N} \in \mathbb{R}^{N \times 14}$ 和未来第N步的3D pointmap $P_{t+N} \in \mathbb{R}^{H \times W \times 4}$。

Pipeline分四阶段：(1) 三路并行编码器提取特征 → (2) Transformer融合为统一上下文 → (3) 条件扩散解码器联合去噪 → (4) 分别解码为动作和pointmap。

### 关键设计

1. **Geometry 3D Encoder（π³编码器）**

    - 功能：从时序RGB帧中提取包含3D几何信息的特征
    - 核心思路：从历史帧 $V$ 中均匀采样5帧，与当前帧 $I_t$ 拼成6帧序列。送入π³编码器（多视图几何模型），每帧被patchify为 $14 \times 14$ 个patch。提取backbone最后两层特征拼接得到1024维的3D几何特征 $\mathbf{f}_{3d}$。注意：只用π³的encoder部分，不经过decoding heads
    - 设计动机：π³是预训练的3D几何基础模型，其latent天然编码了多视图/多帧的3D几何关系。与显式点云相比，latent特征更鲁棒（不受标定误差、深度噪声影响），且是feed-forward的

2. **Semantics 2D Encoder（DINOv3编码器）**

    - 功能：从当前帧提取高层语义特征
    - 核心思路：当前帧 $I_t$ 通过DINOv3编码器，划分为 $16 \times 16$ 个patch，得到1024维语义特征 $\mathbf{f}_{2d}$
    - 设计动机：3D几何特征捕捉的是空间结构，但缺乏任务相关的高层语义理解。DINOv3提供物体级别的语义先验（如识别哪个物体需要操作），与几何特征互补

3. **State Encoder（MLP编码器）**

    - 功能：编码机器人本体感知状态
    - 核心思路：简单MLP将 $p_t \in \mathbb{R}^{14}$ 映射到1024维嵌入 $\mathbf{f}_p$

4. **Semantic-Geometric Fusion（语义-几何融合）**

    - 功能：将三路异构特征融合为统一的上下文表示
    - 核心思路：三个1024维特征 $[\mathbf{f}_{3d}, \mathbf{f}_{2d}, \mathbf{f}_p]$ 沿token维度拼接，送入**4层DETR encoder**做深度融合。输出为统一的Semantic-Geometric Fused Context $\mathbf{f}_c$
    - 设计动机：DETR encoder的自注意力机制让三种模态的特征充分交互——3D几何特征告诉语义特征"物体在哪里"，语义特征告诉几何特征"哪个物体重要"，本体感知约束"当前机器人能做什么"

5. **Joint Diffusion Decoder（联合扩散解码器）**

    - 功能：以 $\mathbf{f}_c$ 为条件，联合去噪生成未来动作序列和未来3D latent
    - 核心思路：采用DETR decoder结构实现条件扩散。训练时，前向过程给clean target $x_0 = \{a_{t:t+N}, \mathbf{f}_{t+N}, P_{t+N}\}$ 加高斯噪声得到 $x_k$；反向过程中decoder预测clean target $\hat{x}_0$。损失函数为L1损失：
    $\mathcal{L} = \mathbb{E}_{k, x_0, \epsilon}\left[\|{\hat{a}_{t:t+N}} - a_{t:t+N}\|_1 + \lambda\|\hat{\mathbf{f}}_{t+N} - \mathbf{f}_{t+N}\|_1 + \gamma\|\hat{P}_{t+N} - P_{t+N}\|_1\right]$
    - **两种预测目标**：
        - Future Action Chunk: $\mathbb{R}^{N \times 14}$，N步双臂动作（6-DoF关节+1-DoF gripper × 2臂）
        - Future 3D Pointmap Latent: $\mathbf{f}_{t+N} \in \mathbb{R}^{H/14 \times W/14 \times 1024}$，由π³的dense head解码为 $P_{t+N} \in \mathbb{R}^{H \times W \times 4}$（x,y,z + confidence）
    - 设计动机：联合预测未来3D结构迫使模型学会"如果执行这些动作，3D场景会变成什么样"——这是一种隐式的world model。只监督horizon $N$ 处的最终状态（而非逐步预测），强制模型推理整个动作序列的累积效果，增强长horizon规划

6. **Pseudo-GT生成策略**

    - 功能：为3D latent目标生成稳定的监督信号
    - 核心思路：不能简单地对单帧调用π³（噪声大、不稳定）。对数据集中每帧 $s$，均匀采样 $n$ 个历史帧组成时序窗口 $\{V, I_s\}$ 送入π³ encoder，只保留 $I_s$ 对应的latent $\mathbf{f}_s$。训练时target设为 $\mathbf{f}_{t+N}$
    - 设计动机：时序窗口的联合处理大幅稳定了3D latent特征质量

### 损失函数 / 训练策略
- 训练采用标准扩散框架，按ACT风格的action chunking策略
- 2D方法（含本文）训练200-600 epochs，3D方法3000 epochs，batch size 32
- 100条专家demo（仿真）或50条teleoperation demo（真实）
- 推理时从高斯噪声出发，迭代 $K$ 步去噪，3D pointmap解码可选择性跳过以提高效率

## 实验关键数据

### 主实验 - RoboTwin 2.0 仿真（三类任务平均成功率%）

| 方法 | Dominant-select (16任务) | Sync-bimanual (8任务) | Seq-coordinate (8任务) |
|------|------------------------|-----------------------|------------------------|
| ACT (2D) | 34.1 | 32.4 | 29.4 |
| DP (2D) | 44.4 | 37.1 | 33.6 |
| DP3 (3D点云) | 61.2 | 42.0 | 42.0 |
| G3Flow (3D+语义) | 54.3 | 43.2 | 40.5 |
| RDT (1.2B参数) | 49.5 | 44.6 | 41.2 |
| Xu et al. (2D+预测) | 55.1 | 47.5 | 44.9 |
| **GAP (Ours)** | **63.2** | **51.3** | **50.4** |

### 消融实验（4任务平均成功率%）

| 2D Semantic | 3D Geometric | Geometric Imagination | 成功率 Avg. |
|:-----------:|:------------:|:---------------------:|:-----------:|
| ✓ | ✓ | ✓ | **25.1** |
| ✗ | ✓ | ✓ | 24.4 |
| ✓ | ✓ | ✗ | 23.6 |
| ✓ | ✗ | ✗ | 21.0 |

### 真实世界实验（成功率%，20 trials/task）

| 任务 | ACT | DP | Xu et al. | **Ours** |
|------|-----|----|-----------|----------|
| Place Empty Cup | 70 | 70 | 75 | **80** |
| Place Dual Shoes | 0 | 10 | 15 | **20** |
| Hanging Mug | 0 | 0 | 5 | **20** |
| Scan Object | 25 | 20 | 35 | **40** |
| **平均** | 23.8 | 25 | 32.5 | **40** |

### 关键发现
- **3D几何感知极其重要**：去掉3D Geometric Module + Geometric Imagination后成功率从25.1%降到21.0%（-16.3%相对下降），是三个模块中影响最大的
- **Geometric Imagination（预测未来3D）是核心创新**：单独去掉它使成功率从25.1%降到23.6%（-6.0%），说明预测未来3D结构确实帮助策略学到更好的3D理解
- **仅用RGB输入超越显式点云方法**：GAP（RGB only）在Dominant-select任务上超越DP3（63.2% vs 61.2%），证明预训练3D基础模型的latent可以替代显式点云
- **同步双臂任务优势明显**：在Place Dual Shoes上GAP达43.3%，DP3仅17.7%，说明GAP对双臂协调推理更好
- **数据效率优异**：低数据量（10-20 demo）时，GAP利用预训练特征已有明显学习信号，而DP完全失败（0%成功率）
- **真实世界Hanging Mug**：ACT和DP完全失败（0%），GAP达20%——该任务需要精确3D几何推理来定位杯耳和挂钩的空间关系

## 亮点与洞察
- **将3D基础模型的latent直接作为策略条件是一个优雅的范式**：绕过了点云获取的工程复杂性，π³的latent本身就编码了丰富的3D几何，这一思路可以推广到任何需要3D感知的机器人任务
- **联合预测动作和未来3D结构是隐式的world model**：不需要单独训练world model，通过扩散过程中的联合去噪自然地将3D预测能力注入策略网络。只预测horizon末尾的3D状态而非逐步预测，巧妙地降低了计算和监督复杂度
- **语义-几何融合用DETR encoder**：简单但有效，让不同模态（3D几何、2D语义、本体感知）通过自注意力充分交互，避免了手工设计融合方式
- **Pseudo-GT生成策略**（时序窗口稳定π³输出）可以迁移到任何使用3D基础模型做监督的下游任务

## 局限与展望
- **单步预测horizon**：只预测horizon $N$ 处的3D状态，缺乏multi-step 3D trajectory的预测。对超长horizon任务可能不足——可扩展为预测多个未来时间步的3D结构
- **缺乏持久3D记忆**：无法跨episode积累3D知识，每次推理独立处理当前窗口。可引入persistent 3D memory（借鉴Wang et al.的continuous 3D perception model）
- **真实世界成功率仍较低**：Hanging Mug最高也只有20%，说明复杂精密操控仍有很大进步空间。可能需要更多demo数据或更好的sim-to-real transfer
- **推理效率**：π³和DINOv3两个大型backbone + 扩散迭代去噪，计算开销较大。论文未报告推理时间，实际部署可能需要蒸馏或加速
- **泛化到未见任务/物体**：实验仅在RoboTwin benchmark已知任务上评估，未测试zero-shot泛化

## 相关工作与启发
- **vs DP3**: DP3直接用点云作为输入，需要深度相机和点云预处理。GAP只需RGB输入+预训练3D模型latent，在多数任务上反超DP3，说明隐式3D表示比显式点云更robust
- **vs G3Flow**: G3Flow将2D语义特征投影到3D点云上。GAP则在latent空间做语义-几何融合，避免了3D投影带来的量化误差和标定依赖
- **vs Xu et al.**: Xu et al.联合预测动作和未来2D帧。GAP将预测目标从2D帧升级为3D pointmap，更符合操控任务的3D本质——消融实验证明3D预测比2D预测更有效
- **vs ACT**: ACT是GAP的框架基础（DETR架构 + action chunking），但GAP加入了3D基础模型特征和geometric imagination，将ACT的平均成功率从约32%提升到约55%

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将3D几何基础模型的latent作为操控策略的核心感知先验+联合预测未来3D结构
- 实验充分度: ⭐⭐⭐⭐⭐ 仿真32个任务三类别、6个baseline、消融实验、数据效率分析、真实世界4任务验证
- 写作质量: ⭐⭐⭐⭐ 动机清晰、框架图直观，但某些实验细节需对照补充材料
- 价值: ⭐⭐⭐⭐ 在双臂操控中引入3D基础模型是重要方向，联合3D预测的paradigm有广泛影响力

<!-- RELATED:START -->

## 相关论文

- [Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [Action-guided Generation of 3D Functionality Segmentation Data](action-guided_generation_of_3d_functionality_segmentation_data.md)
- [Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation](../../ICLR2026/3d_vision/ctrlshift_high-quality_geometry-aware_object_manipulation_in_visual_generation.md)
- [Pano360: Perspective to Panoramic Vision with Geometric Consistency](pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)
- [Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)

<!-- RELATED:END -->
