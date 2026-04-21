---
title: >-
  [论文解读] Reconstructing Humans with a Biomechanically Accurate Skeleton
description: >-
  [CVPR 2025][3D视觉][人体姿态估计] HSMR 首次实现从单张图像估计生物力学准确的骨骼模型（SKEL）参数，通过伪标签迭代精炼策略解决无真值训练数据的困难，在标准人体姿态估计基准上匹配 HMR2.0 的性能，在极端姿态场景（MOYO 瑜伽数据集）上 MPJPE 大幅领先超过 18mm，同时有效避免不自然的关节旋转。
tags:
  - CVPR 2025
  - 3D视觉
  - 人体姿态估计
  - 生物力学骨骼
  - SKEL模型
  - 参数化人体模型
  - 伪标签精炼
---

# Reconstructing Humans with a Biomechanically Accurate Skeleton

**会议**: CVPR 2025  
**arXiv**: [2503.21751](https://arxiv.org/abs/2503.21751)  
**代码**: https://isshikihugh.github.io/HSMR/  
**领域**: 3D视觉 / 人体重建  
**关键词**: 人体姿态估计, 生物力学骨骼, SKEL模型, 参数化人体模型, 伪标签精炼

## 一句话总结

HSMR 首次实现从单张图像估计生物力学准确的骨骼模型（SKEL）参数，通过伪标签迭代精炼策略解决无真值训练数据的困难，在标准人体姿态估计基准上匹配 HMR2.0 的性能，在极端姿态场景（MOYO 瑜伽数据集）上 MPJPE 大幅领先超过 18mm，同时有效避免不自然的关节旋转。

## 研究背景与动机

**领域现状**：3D人体姿态估计近年取得巨大进展，以 SMPL 为代表的参数化人体模型被广泛使用。从 HMR 到 HMR2.0，基于 Transformer 的回归方法在标准基准上不断刷新 SOTA。然而，这些方法的输出主要服务于视觉应用（动画、AR/VR），在生物力学领域的采用非常有限。

**现有痛点**：SMPL 及其后续模型使用简化的骨骼设计——每个关节都建模为球窝关节（ball-and-socket joint），拥有三个自由度。这导致两个问题：(1) 运动树不符合真实人体解剖结构；(2) 额外的自由度允许模型预测不自然的关节角度（如膝盖反向弯曲），使输出与生物力学仿真不兼容。

**核心矛盾**：视觉上看起来合理的姿态在生物力学层面可能是不合法的。SMPL 的过度参数化让网络可以通过不自然的关节旋转来降低2D/3D关节误差，却牺牲了物理合理性。

**本文目标**：使用生物力学准确的 SKEL 模型替代 SMPL，直接从单张图像回归 SKEL 参数。核心挑战是没有任何图像-SKEL参数配对的训练数据。

**切入角度**：SKEL 模型与 SMPL 共享相同的表面网格拓扑，这使得可以将现有 SMPL 伪标签转换为 SKEL 参数。但直接转换质量有限，需要在训练过程中迭代精炼。

**核心 idea**：借鉴 SPIN 的 "optimization-in-the-loop" 思想，设计 SKELify 优化过程，在训练中周期性地用网络预测作为初始化、对齐2D关键点来精炼伪标签质量。

## 方法详解

### 整体框架

输入是一张裁剪的人物图像，经过 ViT 骨干网络提取特征，然后通过 Transformer 头回归 SKEL 模型的姿态参数 $q \in \mathbb{R}^{46}$、体型参数 $\beta \in \mathbb{R}^{10}$ 和相机参数 $\pi$。SKEL 模型根据参数输出皮肤网格（6890顶点）和骨骼网格。训练数据通过 SMPL-to-SKEL 转换获得初始伪标签，然后在训练中通过 SKELify 迭代精炼。

### 关键设计

1. **连续旋转表示替代欧拉角**:

    - 功能：解决 SKEL 的欧拉角参数不适合直接回归的问题
    - 核心思路：网络输出连续旋转表示 $q_{\text{cont}}$，先通过 Gram-Schmidt 转换为旋转矩阵 $q_{\text{mat}}$（在此表示上计算参数损失），再转换为欧拉角 $q_{\text{Euler}}$ 输入 SKEL 模型。这样既避免了欧拉角的万向节锁和不连续问题，又保持了与 SKEL 的兼容性
    - 设计动机：直接回归欧拉角会导致训练不稳定，连续旋转表示已被证明对回归任务更友好

2. **SKELify 伪标签迭代精炼**:

    - 功能：在无真值情况下逐步提升训练数据的标签质量
    - 核心思路：对每张训练图像，以当前 HSMR 网络预测 $(q^{\text{reg}}, \beta^{\text{reg}})$ 为初始化，优化 SKEL 参数使其3D关节投影对齐2D关键点真值。优化目标包含三项：2D重投影误差 $E_{\text{kp2D}}$（带鲁棒核函数）、体型先验 $E_{\text{shape}} = \|\beta\|^2$、以及基于生物力学关节极限的姿态先验 $E_{\text{pose}} = \sum_i \exp(l_i - q_i) + \exp(q_i - u_i)$。优化结果替换原始伪标签用于后续训练
    - 设计动机：初始的 SMPL-to-SKEL 转换存在大量失败案例（手臂穿透、脊柱扭曲等）。使用网络预测作为优化初始值比随机初始化或直接从 SMPL 转换更接近良好解，形成良性循环

3. **生物力学关节极限约束**:

    - 功能：确保预测的姿态符合人体关节的自然活动范围
    - 核心思路：SKEL 模型为每个关节自由度定义了明确的上下限（如膝关节：0°伸展到135°屈曲）。SKEL 的姿态空间只有46维（SMPL 是72维），每个参数对应单一自由度的欧拉角。这本身就隐式约束了关节旋转的合法范围，在 SKELify 中还通过指数惩罚项 $E_{\text{pose}}$ 软性施加极限约束
    - 设计动机：SMPL 的球窝关节设计允许膝盖横向/反向弯曲等不自然运动，实验证实 HMR2.0 等方法频繁违反关节极限

### 损失函数 / 训练策略

总损失包含四项：参数损失 $\mathcal{L}_q = \|q_{\text{mat}} - q_{\text{mat}}^*\|_2^2$ 和 $\mathcal{L}_\beta = \|\beta - \beta^*\|_2^2$（仅在伪标签可用时使用），以及3D和2D关键点损失 $\mathcal{L}_{\text{kp3D}} = \|X - X^*\|_1$、$\mathcal{L}_{\text{kp2D}} = \|\pi(X) - x^*\|_1$。使用 HMR2.0 的大规模训练数据（Human3.6M、COCO、MPII 等）。SKELify 精炼以批处理方式周期性执行。

## 实验关键数据

### 主实验

| 数据集 | 指标 | HSMR | HMR2.0 | 差异 |
|--------|------|------|--------|------|
| COCO | PCK@0.05 ↑ | 0.85 | 0.86 | -0.01 |
| 3DPW | MPJPE ↓ | 81.5 | 81.3 | +0.2 |
| 3DPW | PA-MPJPE ↓ | 54.8 | 54.3 | +0.5 |
| Human3.6M | MPJPE ↓ | 50.4 | 50.0 | +0.4 |
| MOYO | MPJPE ↓ | 104.5 | 123.3 | **-18.8** |
| MOYO | PA-MPJPE ↓ | 79.6 | 90.4 | **-10.8** |
| MOYO | MPVPE ↓ | 120.1 | 142.2 | **-22.1** |

### 消融实验

| 配置 | COCO@0.05 | 3DPW MPJPE | MOYO MPJPE |
|------|-----------|------------|------------|
| HMR2.0 + SKEL fit (两阶段) | 0.78 | 81.0 | 130.5 |
| HSMR (端到端) | 0.85 | 81.5 | 104.5 |

关节违规频率（MOYO，膝关节超过阈值比例）：

| 方法 | 10° | 20° | 30° |
|------|-----|-----|-----|
| SMPL方法 | 高频违规 | 高频违规 | 高频违规 |
| HSMR | 极低 | 极低 | 几乎不违规 |

### 关键发现

- HSMR 在标准基准（3DPW、H36M）上与 HMR2.0 差距在 0.5mm 以内，证明 SKEL 模型的约束性并未牺牲常规场景的性能
- 在极端姿态的 MOYO 数据集上 MPJPE 提升 18.8mm，说明生物力学约束对困难姿态有强正则化效果
- 两阶段方法（先 HMR2.0 再 SKEL fitting）效果差且慢（3分钟/帧），远不如端到端 HSMR
- 所有回归 SMPL 参数的方法都存在显著的关节旋转违规，HSMR 几乎完全避免了这个问题

## 亮点与洞察

- **用约束换泛化**：直觉上更受限的模型（46维 vs 72维）应该性能更差，但实验表明适当的约束反而提升了泛化能力，特别是在分布外的极端姿态上。这是一个重要的认知——过度参数化可能是人体重建方法的隐性弱点
- **伪标签循环精炼**：在无真值的情况下通过"网络预测→优化→更新标签→再训练"的循环逐步提升数据质量，这个策略可以推广到其他缺乏标注的任务
- **揭示了 SMPL 方法的系统性问题**：关节旋转违规的量化分析很有说服力，为整个社区指出了一个被忽视的问题方向

## 局限与展望

- 目前仅处理单人单帧，未扩展到多人或视频
- SKEL 模型没有手和面部的细粒度建模
- 伪标签精炼仍然无法保证100%正确，有些SMPL-to-SKEL转换的失败案例可能遗留
- 未来可结合 AddBiomechanics 等生物力学数据集获得真正的 SKEL 真值
- 若结合时序模型，生物力学骨骼的约束在运动分析中价值更大

## 相关工作与启发

- **vs HMR2.0**: 架构和训练数据几乎相同，唯一区别是 SKEL vs SMPL。在常规数据集平手、极端姿态大幅领先，说明骨骼设计本身就是一个重要的设计维度
- **vs HybrIK**: HybrIK 引入逆运动学约束，但仍基于 SMPL 的自由度设计，无法从根本上避免不自然旋转
- **vs SKEL fitting**: 直接对 SMPL 输出做 SKEL 拟合不仅慢（3min/帧）而且效果差，说明端到端学习的优势明显

## 评分

- 新颖性: ⭐⭐⭐⭐ SKEL 模型不是本文提出的，但首次将其引入端到端回归框架并解决了无标签训练问题
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集评估、关节违规分析、两阶段baseline对比都很全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，实验对比公平，问题定位精准
- 价值: ⭐⭐⭐⭐ 为人体重建引入生物力学维度，有潜力影响运动分析和康复医学等应用领域

<!-- RELATED:START -->

## 相关论文

- [Reconstructing People, Places, and Cameras](reconstructing_people_places_and_cameras.md)
- [Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild](../../ECCV2024/3d_vision/reloo_reconstructing_humans_dressed_in_loose_garments_from_monocular_video_in_th.md)
- [PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)
- [Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning](reconstructing_close_human_interaction_with_appearance_and_proxemics_reasoning.md)

<!-- RELATED:END -->
