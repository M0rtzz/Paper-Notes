---
title: >-
  [论文解读] Stereo Any Video: Temporally Consistent Stereo Matching
description: >-
  [ICCV2025][3D视觉][视频立体匹配] 提出Stereo Any Video框架，通过融合单目视频深度基础模型先验(Video Depth Anything)、全对全配对相关(all-to-all-pair correlation)和时序凸上采样(temporal convex upsampling)三大核心模块，在不依赖相机位姿或光流的前提下实现空间精确且时序一致的视频立体匹配，在多个数据集零样本设定下达到SOTA。
tags:
  - ICCV2025
  - 3D视觉
  - 视频立体匹配
  - 时序一致性
  - 单目深度先验
  - 代价体
  - 视差估计
---

# Stereo Any Video: Temporally Consistent Stereo Matching

**会议**: ICCV2025  
**arXiv**: [2503.05549](https://arxiv.org/abs/2503.05549)  
**代码**: [项目页面](https://tomtomtommi.github.io/StereoAnyVideo/)  
**领域**: 3D视觉/立体匹配  
**关键词**: 视频立体匹配, 时序一致性, 单目深度先验, 代价体, 视差估计

## 一句话总结
提出Stereo Any Video框架，通过融合单目视频深度基础模型先验(Video Depth Anything)、全对全配对相关(all-to-all-pair correlation)和时序凸上采样(temporal convex upsampling)三大核心模块，在不依赖相机位姿或光流的前提下实现空间精确且时序一致的视频立体匹配，在多个数据集零样本设定下达到SOTA。

## 研究背景与动机

**任务定义**：视频立体匹配从校正后的左右图像序列中估计逐帧视差图，用于3D场景重建，对自动驾驶、机器人导航、AR/VR等下游任务至关重要。

**现有方法局限**：
   - **基于图像的方法**（RAFT-Stereo、IGEV-Stereo等）：不利用时序信息，直接应用于视频时存在闪烁和伪影。
   - **基于视频的方法**（CODD、TemporalStereo、BiDAStereo等）：依赖辅助信息（相机位姿或光流）进行时序对齐，在动态场景和复杂相机运动下，辅助模块本身精度不足会成为瓶颈。

**核心观察**：视频生成领域的研究表明，稳定的特征表示是保证时序连贯的关键。作者从"特征鲁棒性与稳定性"这一新视角出发设计框架，而非依赖外部辅助信号。

**关于单目深度图的实证分析**：作者比较了DepthCrafter（单目视频深度）和RAFT-Stereo在Sintel上的表现——DepthCrafter虽然视觉上更一致（人类评分更高），但空间和时序误差远大于RAFT-Stereo（EPE: 8.68 vs 1.42），说明单目深度是"一致但不准确的"，因此只利用其特征作为先验，而非直接使用深度图。

## 方法详解

### 整体架构
输入为校正后的立体视频序列 $\{I_L^t, I_R^t\}_{t=1}^{T}$，采用级联(cascaded)流程从低分辨率逐步恢复全分辨率视差，包含三个核心模块：

### 1. 融合基础模型先验的特征提取
- **双路特征提取**：使用可训练的卷积编码器提取图像特征和上下文特征（残差块结构），同时利用冻结的Video Depth Anything (VDA)提取深度特征。
- **特征融合**：通过浅层卷积适配器(adapter)下采样VDA的深度特征（32通道），与卷积特征（96通道）拼接形成128通道的稳定特征表示。
- **为什么用VDA-S而非VDA-L**：VDA-L参数量是VDA-S的13倍(381.8M vs 28.4M)，但性能提升微乎其微，且大骨干需要更多适配参数，增加训练复杂度。

### 2. All-to-All-Pair Correlation (全对全配对相关)
- **传统方法**：在第$n$次迭代中，先用上一次视差$d_{n-1}$将右特征warp到左特征坐标，然后在局部搜索窗口内计算单方向的点积相关：$C_n(x,y) = \langle F_L(x,y), \hat{F}_R(x+r_x, y+r_y) \rangle$
- **本文方法**：引入双向对应关系，计算两个搜索窗口内所有潜在匹配点之间的相似度：$C_n(x,y) = \langle F_L(x+r_x, y+r_y), \hat{F}_R(x+r_x, y+r_y) \rangle$
- **优势**：(1) 增强匹配验证、降低歧义；(2) 通过密集对应关系强制匹配平滑性；(3) 不依赖光流，避免其精度瓶颈。
- **代价体维度变化**：从 $H' \times W' \times (2r_x+1)(2r_y+1)$ 扩展到 $H' \times W' \times (2r_x+1)^2(2r_y+1)^2$。

### 3. 时序代价聚合
- **MLP编码器**：将高维代价体压缩为紧凑表示 $E_n = \text{MLP}(C_n)$。
- **3D-GRU迭代更新**：使用可分离3D卷积在空间和时间维度聚合信息，集成super kernels和时空注意力机制，迭代更新视差。
- **时序凸上采样(Temporal Convex Upsampling)**：核心创新之一。每个高分辨率像素由其在3帧(当前帧 $\pm 1$)内的 $3 \times 3 \times 3 = 27$ 个低分辨率邻居通过可学习权重加权组合得到：
  $$\mathbf{w} = \text{softmax}(\text{Conv3d}(h_n))$$
  $$D_n^t = \alpha \cdot \sum_{ijk} \mathbf{w}_{ijk} \odot \text{unfold}([d_n^{t-1}, d_n^{t}, d_n^{t+1}])$$
  这种设计使上采样过程本身就具有时序一致性，而传统方法只做空间上采样。

### 4. 损失函数
- 采用纯图像级L1损失，权重系数 $\gamma=0.9$，不使用OPW或TGM等时序损失：
  $$\mathcal{L} = \sum_{t=1}^{T} \sum_{n=1}^{N} \gamma^{N-n} \|D_{gt}^t - D_n^t\|$$
- 作者发现时序损失在视频立体匹配中会引入精度-一致性的trade-off，收益不明显。

## 实验关键数据

### 训练配置
- 先在SceneFlow上训练120K迭代，再在Dynamic Replica + Infinigen SV + Virtual KITTI2混合数据上微调80K迭代
- 训练序列长度T=5，评估序列长度T=20；GRU迭代次数训练N=10，评估N=20
- 总训练约6天(A100 GPU)

### 零样本泛化结果(仅SceneFlow训练, Table 2)

| 数据集 | 指标 | BiDAStereo | MonSter | **Ours** |
|--------|------|------------|---------|----------|
| Sintel Final | TEPE↓ | 1.26 | 1.70 | **1.07** |
| Dynamic Replica | EPE↓ | 0.65 | 0.45 | **0.25** |
| Infinigen SV | TEPE↓ | 1.99 | 1.65 | **1.65** |
| Virtual KITTI2 | TEPE↓ | 1.02 | 0.73 | **0.74** |

### 混合数据训练结果(Table 3)

| 数据集 | 指标 | BiDAStereo | FoundationStereo | **Ours** |
|--------|------|------------|-----------------|----------|
| Spring | TEPE↓ | 0.90 | 1.78 | **0.77** |
| Sintel Final | TEPE↓ | 1.33 | — | **0.99** |
| KITTI Depth | TEPE↓ | 0.42 | 0.40 | **0.35** |

仅用合成数据训练的模型超越了在真实数据(含同域数据)上训练的方法至少15%。

### 消融实验核心发现(Table 4)

| 组件 | 变体 | Sintel TEPE↓ | DR TEPE↓ |
|------|------|-------------|----------|
| 先验 | 无深度先验 | 1.47 | 0.092 |
| 先验 | VDA-S ✓ | 1.27 | 0.083 |
| 相关 | 1D+2D局部 | 1.27 | 0.083 |
| 相关 | 1D+2D全对全 ✓ | 1.21 | 0.076 |
| 上采样 | 双线性 | 1.26 | 0.085 |
| 上采样 | 时序凸上采样 ✓ | 1.04 | 0.067 |
| 注意力 | 时空注意力 ✓ | 1.07 | 0.057 |

### 效率分析(Table 6, 720×1280分辨率, 20帧)

| 方法 | 参数量(M) | 显存(G) | MACs(T) |
|------|----------|---------|---------|
| DynamicStereo | 20.5 | 35.1 | 182.3 |
| BiDAStereo | 12.2 | 41.1 | 186.6 |
| **Ours** | 9.4 [+28.4冻结] | 41.1 | 303.4 |

可训练参数最少(9.4M)，但因使用基础模型先验，计算量最大(303.4T MACs)。

## 亮点与洞察

1. **视角新颖**：从"特征稳定性"而非"辅助信号对齐"出发解决时序一致性，绕开了光流/位姿估计的精度瓶颈。

2. **"一致但不准确"的实证**：通过定量+主观实验证明单目视频深度的局限性（EPE远大于立体匹配），为"只用特征、不用深度图"提供了令人信服的理由。

3. **时序凸上采样设计精巧**：将传统的空间凸上采样扩展到3D，每个像素融合时间邻域的27个邻居，用softmax权重实现端到端可学习的时序平滑。

4. **极强的零样本泛化**：仅在合成数据上训练，在真实室内/室外场景都表现出色，甚至超越在同域真实数据上训练的方法。

5. **简洁的损失设计**：只用L1损失就实现了强时序一致性，说明时序一致性主要来自架构设计（特征提取+代价聚合+上采样）而非时序损失的显式约束。

## 局限与展望

1. **计算开销大**：MACs为BiDAStereo的1.63倍(303.4T vs 186.6T)，主要源于冻结的VDA骨干，限制了实时应用。
2. **模型规格单一**：当前仅有一个版本，作者计划开发大/轻量两个版本的model zoo。
3. **训练数据仍限于合成**：虽然泛化能力强，但未探索在真实数据上训练的效果。
4. **序列长度受限**：训练T=5、评估T=20，更长视频的一致性效果未充分验证。
5. **3D相关失败**：消融显示3D搜索窗口（跨时间维度搜索）效果很差（TEPE 4.51），说明帧间未对齐时无法直接跨时间构建相关，这个问题的解决或许能进一步提升性能。

## 相关工作与启发

- **RAFT-Stereo [Lipson2021]**：迭代立体匹配范式的基础架构，本文在此基础上扩展到视频场景。
- **BiDAStereo [Jing2024]**：基于光流对齐的时序立体匹配方法，是本文最直接的对比基线。
- **Video Depth Anything [Chen2025]**：视频单目深度基础模型，提供冻结特征先验。
- **FoundationStereo [Wen2025]**：同样利用Depth Anything特征的图像级方法，本文将此思路扩展到视频。
- **MonSter [Cheng2025]**：融合单目深度图和特征的双分支方法，但为图像级。
- 对idea生成的启发：(1) 基础模型先验+轻量适配的范式可推广到其他视频3D任务；(2) "不用深度图、只用特征"的思路可能适用于其他需要深度-立体融合的场景。

## 评分
- 新颖性: ⭐⭐⭐⭐ (全对全相关+时序凸上采样有较强原创性)
- 实验充分度: ⭐⭐⭐⭐⭐ (7个数据集、详尽消融、定性定量+用户研究)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，动机论证充分)
- 价值: ⭐⭐⭐⭐ (视频立体匹配新标杆，泛化能力突出)

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] TC-Stereo: Temporally Consistent Stereo Matching](../../ECCV2024/3d_vision/temporally_consistent_stereo_matching.md)
- [\[ICCV 2025\] BANet: Bilateral Aggregation Network for Mobile Stereo Matching](banet_bilateral_aggregation_network_for_mobile_stereo_matchi.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [\[CVPR 2025\] FoundationStereo: Zero-Shot Stereo Matching](../../CVPR2025/3d_vision/foundationstereo_zero-shot_stereo_matching.md)

<!-- RELATED:END -->
