---
title: >-
  [论文解读] SAP: Segment Any 4K Panorama
description: >-
  [CVPR 2026][图像分割][全景分割] 提出 SAP（Segment Any 4K Panorama），通过将全景图转化为沿球面固定轨迹采样的透视伪视频序列，解决 SAM2 流式记忆机制在 360° 图像上的结构性失配问题，并合成 183K 实例标注的 4K 全景图进行微调，在真实世界全景基准上实现零样本 mIoU +17.2 的提升。
tags:
  - CVPR 2026
  - 图像分割
  - 全景分割
  - SAM2
  - 4K 高分辨率
  - 拓扑-记忆对齐
  - 透视视频重构
---

# SAP: Segment Any 4K Panorama

**会议**: CVPR 2026  
**arXiv**: [2603.12759](https://arxiv.org/abs/2603.12759)  
**代码**: 有 (Project Page)  
**领域**: 全景图像分割  
**关键词**: 全景分割, SAM2, 4K 高分辨率, 拓扑-记忆对齐, 透视视频重构

## 一句话总结

提出 SAP（Segment Any 4K Panorama），通过将全景图转化为沿球面固定轨迹采样的透视伪视频序列，解决 SAM2 流式记忆机制在 360° 图像上的结构性失配问题，并合成 183K 实例标注的 4K 全景图进行微调，在真实世界全景基准上实现零样本 mIoU +17.2 的提升。

## 研究背景与动机

随着 360° 相机在机器人、AR/VR 和具身智能中的普及，对全景图像高质量实例分割的需求日益增长。然而现有分割基础模型（如 SAM/SAM2）面临三大挑战：

1. **分辨率损失**：SAM 系列仅支持 $1024^2$ 输入，4K 2:1 全景图（$4096 \times 2048$）被压缩至 $1024 \times 512$ 并填充，丢失大量细节
2. **几何畸变**：等距柱状投影（ERP）引入严重的极区畸变和左右接缝不连续
3. **结构性假设违反**：SAM2 的流式记忆机制假设连续帧对应平滑相机运动和重叠视觉内容，而 ERP 全景图**没有内在的时间顺序**——滑动窗口裁剪破坏了物理视点连续性

OmniSAM 尝试直接在 ERP 上使用滑动窗口，但畸变和接缝问题仅是表面症状。本文的关键洞察：**球面全景图从根本上违反了流式记忆模型的结构性假设**，需要从拓扑-记忆对齐角度解决。

## 方法详解

### 整体框架

SAP 的流程分为四步：
1. 将 ERP 全景图 + 提示点投影为固定轨迹的透视伪视频序列
2. 使用经微调的 SAM2 处理伪视频，生成逐帧分割掩码
3. 将透视帧掩码反投影并融合回 ERP 平面
4. 输出最终全景分割结果

### 关键设计

1. **全景图到透视视频转换**：给定 ERP 图像 $I^{ERP} \in \mathbb{R}^{H \times W \times 3}$，通过三阶段几何投影生成 $N$ 个透视视图：
    - 定义相机内参（FoV $\beta = 90°$, 焦距 $f = \frac{L-1}{2\tan(\beta/2)}$）
    - 像素反投影为射线方向：$\mathbf{r}^{cam} \propto \mathbf{K}^{-1}[u,v,1]^T$
    - 旋转至世界坐标并转换为球面坐标采样：$[x,y,z]^T = \text{Normalize}(\mathbf{R}_i \mathbf{K}^{-1}[u,v,1]^T)$

2. **列优先锯齿扫描轨迹**：这是本文最核心的设计创新。相比行优先扫描，列优先具有**无限循环**性质：从任意起始点出发，沿上下往复运动可准确返回起始点。形式化地，第 $j$ 列的访问顺序为：
    $$\mathcal{O}_j = \begin{cases} (j,1),(j,2),\dots,(j,N_{pitch}), & j \bmod 2 = 1 \\ (j,N_{pitch}),\dots,(j,2),(j,1), & j \bmod 2 = 0 \end{cases}$$
    连续帧之间仅一个角度维度变化（yaw 或 pitch），确保类似视频的平滑过渡。采样网格由 FoV 和重叠率 $r=0.5$ 决定：$\Delta_{yaw} = \beta_h(1-r)$, $N_{yaw} = \lceil 360°/\Delta_{yaw} \rceil$。

3. **循环扩展支持任意起始点**：将轨迹复制 2 倍（$2 \cdot N$ 帧），训练时随机采样起始索引 $s \in \{0, \dots, N-1\}$，提取连续 $N$ 帧窗口，保证任何窗口都覆盖全部视点至少一次。

4. **183K 合成数据集**：使用 InfiniGen 引擎花费 40,000 GPU 小时合成 183,440 张 $4096 \times 2048$ 分辨率的全景图，共 6,409,732 个实例掩码。物体尺寸分布：小型 37.84%、中型 25.70%、大型 36.47%。

5. **提示点投影**：用户在 ERP 上的提示点 $\mathbf{p} = (u_p, v_p)$ 先转为球面方向向量 $\mathbf{d} = [\cos\theta_p\cos\phi_p, \cos\theta_p\sin\phi_p, \sin\theta_p]^T$，再投影至各透视帧判断可见性。

### 掩码融合

逐帧掩码通过最大值融合回 ERP 平面：

$$M^{ERP}(u,v) = \max_{i: (u,v) \in \mathcal{V}_i} \tilde{M}_i(u,v)$$

### 损失函数 / 训练策略

- 基于 SAM2 (Hiera-Large 编码器) 构建
- **冻结图像编码器**，仅更新 memory attention、memory encoder、mask decoder 和 prompt encoder
- 混合训练：合成全景数据 + SAM2 原始训练数据（SA-1B + SA-V）以防灾难性遗忘
- AdamW 优化器，batch size 128，lr $2 \times 10^{-4}$（余弦调度），weight decay 0.1，梯度裁剪 0.1

## 实验关键数据

### PAV-SOD 真实世界 4K 全景基准（零样本）

| 方法 | 1-click Overall | 1-click Small | 1-click Large | 3-click Overall |
|------|----------------|--------------|--------------|----------------|
| SAM2-tiny | 51.6 | 46.3 | 49.1 | 82.2 |
| SAM2-tiny+scan | 65.1 | 49.6 | 70.0 | 83.0 |
| **SAP-tiny** | **75.8** | **53.9** | **79.7** | **84.8** |
| Δ(SAP-SAM2) | **+24.2** | +7.6 | +30.6 | +2.6 |
| SAM2-large | 66.3 | 50.7 | 64.4 | 84.3 |
| SAM2-large+scan | 69.0 | 58.4 | 73.8 | 84.1 |
| **SAP-large** | **77.3** | **61.1** | **81.7** | **86.1** |
| Δ(SAP-SAM2) | **+11.0** | +10.4 | +17.3 | +1.8 |

### InfiniGen 合成 4K 全景基准

| 方法 | 1-click Overall | 1-click Small | 1-click Large | 3-click Overall |
|------|----------------|--------------|--------------|----------------|
| SAM2-base | 62.0 | 57.6 | 59.8 | 81.4 |
| SAP-base | **81.8** | **72.3** | **89.6** | **88.9** |
| Δ(SAP-SAM2) | **+19.8** | +14.7 | +29.8 | +7.5 |
| SAM2-large | 62.8 | 59.7 | 60.7 | 81.4 |
| SAP-large | **81.9** | **72.5** | **90.7** | **89.0** |
| Δ(SAP-SAM2) | **+19.1** | +12.8 | +30.0 | +7.6 |

### 关键发现

- SAP 在所有模型尺寸上均大幅超越 SAM2，**四种模型平均 +17.2 mIoU**（PAV-SOD 1-click）
- 单纯的扫描策略（无微调）在 tiny 模型上即可带来 +13.5 提升，但微调后改善更大且更一致
- 大物体改进最显著（PAV-SOD tiny: +30.6），说明跨视图传播对大物体尤为关键
- 在 HunyuanWorld（卡通风格 8K 全景）上，不微调直接扫描反而性能下降，微调必不可少
- 消融实验确认：混合 SAM2 原始数据训练显著改善泛化（PAV-SOD: 67.3 → 77.3）

## 亮点与洞察

- **问题定义精准**：将"ERP 畸变和接缝不连续"重新理解为"拓扑-记忆对齐"问题，从根本上解释了 SAM2 的失败原因
- **列优先锯齿轨迹设计精巧**：满足无限循环约束，保证任意起始点的全覆盖，在工程层面非常优雅
- **大规模合成数据有效**：183K 合成图 + 微调 SAM2 不仅在合成测试集上有效，在真实世界数据上同样表现优秀，证实了合成-真实迁移的可行性
- **与现有工作的本质区别**：OmniSAM 在 ERP 上直接滑窗，SAP 则在透视空间操作，完全避免畸变

## 局限与展望

- 大量透视帧（$N_{yaw} \times N_{pitch}$ 帧 × 2 循环）带来较高推理成本
- 固定 FoV $90°$ 和重叠率 $50\%$ 是手动选择的，未自适应优化
- 仅评估了 SAM2 一种基础模型，未测试其他分割基础模型
- 实例跨帧一致性依赖 SAM2 记忆机制，复杂遮挡场景可能仍有困难
- 合成数据虽有效但域差距仍在，尤其在小物体上改进相对有限

## 相关工作与启发

- **SAM2 [Meta 2024]**：视频分割基础模型，提供流式记忆机制——本文的基础
- **OmniSAM [2024]**：在 ERP 上使用 SAM2 滑窗做语义分割，本文的改进对象
- **InfiniGen [2024]**：用于生成大规模合成全景图的数据引擎
- **Trans4PASS / PanoFormer**：变形嵌入/切线贴片方法处理球面畸变
- 启发：拓扑-记忆对齐的思路可推广至其他球面/柱面/鱼眼等非标准几何的基础模型适配

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 综合评价 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [Live Interactive Training for Video Segmentation](live_interactive_training_for_video_segmentation.md)
- [DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](dsflash_comprehensive_panoptic_scene_graph_generation_in_realtime.md)
- [SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)
- [Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera](unified_spherical_frontend_learning_rotation-equivariant_representations_of_sphe.md)

<!-- RELATED:END -->
