---
title: >-
  [论文解读] HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery
description: >-
  [CVPR 2025][3D视觉][人体网格恢复] 提出HeatFormer——一种基于Transformer的神经优化器，通过将SMPL参数估计转化为热力图生成与对齐问题，实现对多视角图像中人体形状和姿态的迭代优化恢复，在Human3.6M上达到29.5mm MPJPE的SOTA精度，且对视角数量、相机配置和遮挡具有强鲁棒性。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "人体网格恢复"
  - "多视角"
  - "神经优化器"
  - "热力图"
  - "Transformer"
---

# HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery

**会议**: CVPR 2025  
**arXiv**: [2412.04456](https://arxiv.org/abs/2412.04456)  
**代码**: [https://vision.ist.i.kyoto-u.ac.jp/research/heatformer/](https://vision.ist.i.kyoto-u.ac.jp/research/heatformer/) (项目页面)  
**领域**: 3D视觉  
**关键词**: 人体网格恢复, 多视角, 神经优化器, 热力图, Transformer

## 一句话总结

提出HeatFormer——一种基于Transformer的神经优化器，通过将SMPL参数估计转化为热力图生成与对齐问题，实现对多视角图像中人体形状和姿态的迭代优化恢复，在Human3.6M上达到29.5mm MPJPE的SOTA精度，且对视角数量、相机配置和遮挡具有强鲁棒性。

## 研究背景与动机

**领域现状**：人体网格恢复（HMR）主流方法大多基于单目图像，通过前馈回归网络直接预测SMPL参数。部分多视角方法通过特征聚合或体素化来融合多视角信息，但本质上仍是一次推理得到结果。

**现有痛点**：(1) 单目方法受限于深度歧义和遮挡——在真实场景中，人经常被桌椅、沙发等物体遮挡，或自遮挡，单视角无法恢复被遮挡部分。(2) 现有多视角方法多为前馈架构，仅在输入阶段使用一次多视角信息，无法在估计过程中反复参考视觉证据进行修正。(3) 大多数方法依赖固定的相机配置，限制了实际部署灵活性。

**核心矛盾**：如何设计一个方法既能充分利用多视角互补信息（特别是跨视角解决遮挡），又不依赖于特定的相机数量和配置？前馈方法只"看一次"输入就输出结果，缺乏自我修正能力；传统优化方法虽能迭代修正，但需要精确的2D-3D对应，且不易端到端训练。

**本文目标** 设计一个神经优化器，能够在推理过程中迭代利用多视角图像反馈来逐步修正SMPL估计，同时对视角数量和相机配置保持不可知。

**切入角度**：将SMPL参数估计重新表述为热力图生成与对齐问题。热力图提供稳定的空间梯度（比关键点坐标更适合反向传播），使得端到端神经优化成为可能。通过Transformer的编码器-解码器架构，将解码器的多次前向推理展开为迭代优化。

**核心 idea**：用Transformer解码器的重复前向推理作为展开的迭代优化，在每步中通过热力图交叉注意力对齐当前SMPL估计与多视角观测。

## 方法详解

### 整体框架

输入为多视角图像，输出为各视角对应的SMPL形状参数 $\beta \in \mathbb{R}^{10}$ 和姿态参数 $\theta \in \mathbb{R}^{24 \times 3}$。流程分为：(1) 用ViT提取每个视角的图像特征，用AdaFuse提取多关节热力图；(2) HeatEncoder将每个视角的多关节热力图整合为单一特征，与图像特征拼接；(3) Decoder接收当前SMPL估计生成的热力图作为query，与编码器输出做交叉注意力，输出参数修正量；(4) 更新SMPL参数，重复步骤3共3次。

### 关键设计

1. **HeatEncoder（热力图编码器）**:

    - 功能：将每个视角的 $k$ 个关节热力图整合为一个统一的特征表示
    - 核心思路：对每个视角的热力图集合 $P \in \mathbb{R}^{k \times H \times W}$，先按空间位置切分为patch，每个patch被展平为token，token附带关节顺序编码和空间位置编码。所有token加上一个CLS token输入自注意力模块，CLS token通过自注意力聚合所有关节的空间信息，输出一个整合后的热力图特征。对每个视角独立执行此操作后，将整合热力图与ViT图像特征按空间位置拼接，形成最终的token序列。
    - 设计动机：直接用关键点坐标作为特征不利于梯度传播（梯度不连续）。热力图是关键点位置的2D概率分布，提供平滑的空间梯度，天然适合端到端优化。通过自注意力跨关节整合，可以学习关节间的空间协调关系。

2. **Decoder作为神经优化器**:

    - 功能：通过重复前向推理实现SMPL参数的迭代优化
    - 核心思路：每次迭代中，将当前SMPL参数实例化网格→提取关节→在每个视角投影为2D热力图→通过HeatEncoder编码→作为Decoder的query。Decoder通过交叉注意力将query（当前估计的热力图）与key/value（输入图像的热力图+特征）对齐，输出张量经过平均池化和MLP得到SMPL参数的修正量（差值）。将修正量加到当前估计上，再重新生成热力图进入下一次迭代。实验发现3-4次迭代即可收敛。
    - 设计动机：传统优化需要显式梯度计算，计算慢且可能陷入局部最优。将优化过程"展开"为Transformer的重复前向推理，既继承了优化的自我修正能力，又保持了神经网络的端到端可训练性和推理速度。

3. **视角无关的估计与灵活性**:

    - 功能：使模型对相机数量和配置完全不可知
    - 核心思路：输出每个视角依赖的SMPL参数（而非单一3D估计），这些参数可以通过选择与下游应用最接近的视角、或按2D重投影质量挑选最优估计来得到最终结果。由于Transformer的token数量可变，同一模型可以在不同数量的视角上训练和推理。当相机标定可用时利用外参优化热力图，当不可用时降级为弱透视相机模型。
    - 设计动机：固定相机配置严重限制实际应用——不同房间、不同安装条件会导致配置各异。视角无关设计让模型可以"即插即用"。

### 损失函数 / 训练策略

总损失为六项之和：$\mathcal{L} = \lambda_{3D}\mathcal{L}_{3D} + \lambda_{2D}\mathcal{L}_{2D} + \mathcal{L}_{smpl} + \mathcal{L}_{hm} + \lambda_v\mathcal{L}_v + \lambda_{adv}\mathcal{L}_{adv}$，其中 $\mathcal{L}_{3D}$ 和 $\mathcal{L}_{2D}$ 是3D/2D关节MSE损失，$\mathcal{L}_{smpl}$ 是SMPL参数监督，$\mathcal{L}_{hm}$ 是步长加权热力图损失（权重随迭代递增 [0.001, 0.003, 0.005]），$\mathcal{L}_v$ 是顶点损失（防止身体扭曲），$\mathcal{L}_{adv}$ 是对抗先验损失。先在单视角上预训练HeatEncoder，再冻结HeatEncoder和ViT训练整个模型。

## 实验关键数据

### 主实验

| 方法 | 类型 | Human3.6M MPJPE ↓ | Human3.6M PA-MPJPE ↓ | MPI-INF-3DHP MPJPE ↓ | MPI-INF-3DHP PCK ↑ |
|------|------|-------------------|----------------------|----------------------|-------------------|
| HMR2.0 | 单目 | 44.8 | 33.6 | - | - |
| PaFF | 多视角 | 33.0 | 26.9 | 48.4 | 98.6 |
| U-HMR | 多视角 | 31.0 | 22.8 | - | - |
| **HeatFormer (iter3)** | 多视角 | **30.7** | 23.3 | **39.8** | **99.5** |
| **HeatFormer (iter4)** | 多视角 | **29.5** | **22.4** | 40.6 | **99.5** |

### 消融实验

| 配置 | Human3.6M MPJPE ↓ | PA-MPJPE ↓ | MPVPE ↓ |
|------|-------------------|-----------|---------|
| iter1 | 34.9 | 26.2 | 41.9 |
| iter2 | 31.2 | 23.6 | 37.5 |
| iter3 | 30.7 | 23.3 | 37.0 |

**跨数据集泛化（BEHAVE遮挡数据集）：**

| 方法 | Protocol1 MPJPE ↓ | Protocol2 MPJPE ↓ |
|------|-------------------|-------------------|
| HMR2.0a | 72.2 | 48.1 |
| HMR2.0a+scoreHMR | 72.9 | 49.0 |
| **HeatFormer (iter3)** | **51.1** | **34.2** |

### 关键发现

- 迭代次数从1到3的MPJPE从34.9降到30.7，说明神经优化的迭代修正确实有效且收敛快速
- 在BEHAVE遮挡数据集上，HeatFormer大幅超越单目方法HMR2.0（且后者预训练数据更多），证明多视角利用对遮挡场景至关重要
- 跨数据集评估（仅在Human3.6M训练，MPI-INF-3DHP测试）上大幅超过U-HMR（56.0 vs 73.2 MPJPE），说明U-HMR过拟合到特定数据集，而神经优化formulation带来更好泛化

## 亮点与洞察

- **热力图作为优化中间表示**：传统方法用关键点坐标做2D-3D对齐，梯度不连续；用热力图替代，提供平滑空间梯度使端到端神经优化成为可能。这个"回归基础"的思路非常巧妙。
- **展开的优化即重复推理**：将Transformer decoder的多次前向pass等同于优化的迭代步骤，每步都能参考原始输入（通过cross-attention），是一种优雅的将优化融入前馈网络的方式。可以迁移到其他需要迭代对齐的任务。
- **视角无关设计**：输出视角依赖的估计而非全局估计，利用了Transformer对token数量灵活的特性，实现了真正的即插即用部署。

## 局限与展望

- 未利用时序信息，在视频监控场景中未充分发挥连续帧的约束
- 依赖AdaFuse进行多视角感知的2D关节检测作为前置步骤，其精度会影响后续优化
- 训练数据主要来自Human3.6M和MPI-INF-3DHP，场景多样性有限
- 推理时需要3-4次解码器前向pass，相比单次前馈方法计算量更大

## 相关工作与启发

- **vs SPIN**: SPIN也结合回归和优化，但其优化阶段使用传统SMPLify且仅限单目，HeatFormer将优化内化为神经网络推理且原生支持多视角
- **vs PyMAF**: PyMAF也在网络内做迭代对齐反馈，但基于特征像素对齐而非热力图，且不支持多视角
- **vs U-HMR**: U-HMR用Transformer融合多视角特征但是前馈架构，在跨数据集评估中明显过拟合

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个用于HMR的神经优化器，热力图对齐formulation新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集评估+遮挡分析+跨数据集泛化+视角数量分析，非常充分
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 对老年监护、安全监控等固定多相机应用有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](prompthmr_promptable_human_mesh_recovery.md)
- [\[CVPR 2025\] MEGA: Masked Generative Autoencoder for Human Mesh Recovery](mega_masked_generative_autoencoder_for_human_mesh_recovery.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](../../ICCV2025/3d_vision/ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](../../ECCV2024/3d_vision/global-to-pixel_regression_for_human_mesh_recovery.md)
- [\[ICML 2025\] ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](../../ICML2025/3d_vision/adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)

</div>

<!-- RELATED:END -->
