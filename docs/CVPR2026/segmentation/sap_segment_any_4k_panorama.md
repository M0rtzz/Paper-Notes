---
title: >-
  [论文解读] SAP: Segment Any 4K Panorama
description: >-
  [CVPR 2026][语义分割][全景分割] 提出 SAP（Segment Any 4K Panorama），通过将全景图转化为沿球面固定轨迹采样的透视伪视频序列，解决 SAM2 流式记忆机制在 360° 图像上的结构性失配问题，并合成 183K 实例标注的 4K 全景图进行微调，在真实世界全景基准上实现零样本 mIoU +17.2 的提升。
tags:
  - "CVPR 2026"
  - "语义分割"
  - "全景分割"
  - "SAM2"
  - "4K 高分辨率"
  - "拓扑-记忆对齐"
  - "透视视频重构"
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

SAP 要解决的是 SAM/SAM2 这类分割基础模型搬到 360° 全景图上水土不服的问题。它的整条链路是：先把一张 ERP 全景图连同用户提示点，按球面上一条固定轨迹采样成一段「透视伪视频」；再用微调过的 SAM2 逐帧分割这段伪视频；最后把各透视帧的掩码反投影、按最大值融合回 ERP 平面，得到最终全景分割。关键洞察在于：SAM2 的流式记忆假设相邻帧来自平滑相机运动、画面有重叠，而 ERP 全景图本身没有时间顺序，直接滑窗会破坏这个假设——SAP 用「造一段真正连续的透视视频」来对齐它。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["ERP 4K 全景图 + 用户提示点"] --> B["全景图→透视视频<br/>几何投影去畸变 + 提示点同步投影"]
    B --> C["列优先锯齿扫描轨迹<br/>帧间只变一个角度，循环扩展支持任意起点"]
    C --> D["微调 SAM2 逐帧分割<br/>183K 合成数据按轨迹对齐微调"]
    D --> E["掩码反投影与最大值融合<br/>各帧掩码拼回 ERP，重叠处取 max"]
    E --> F["4K 全景实例分割"]
```

### 关键设计

**1. 全景图→透视视频：把没有时序的全景造成有时序的视频**

SAM2 的记忆机制吃的是「帧间平滑、内容重叠」的视频，而 ERP 全景图既有极区畸变和左右接缝、又没有内在时间顺序。SAP 的第一步是用几何投影把全景造成伪视频：给定 ERP 图像 $I^{ERP}\in\mathbb{R}^{H\times W\times3}$，先定相机内参（FoV $\beta=90°$、焦距 $f=\frac{L-1}{2\tan(\beta/2)}$），把每个像素反投影成射线方向 $\mathbf{r}^{cam}\propto\mathbf{K}^{-1}[u,v,1]^T$，再旋到世界坐标、转成球面坐标采样 $[x,y,z]^T=\text{Normalize}(\mathbf{R}_i\mathbf{K}^{-1}[u,v,1]^T)$，生成 $N$ 个无畸变的透视视图。这样 SAM2 面对的是它熟悉的透视图，畸变与接缝从源头消失。用户在 ERP 上点的提示点用同一套变换矩阵同步投影：提示点 $\mathbf{p}=(u_p,v_p)$ 先转成球面方向向量 $\mathbf{d}=[\cos\theta_p\cos\phi_p,\cos\theta_p\sin\phi_p,\sin\theta_p]^T$，再投到各透视帧上，凡满足 $z_c>0$ 且落在帧内的帧都被判为「提示可见」，从而在正确的帧里触发分割。

**2. 列优先锯齿扫描轨迹：保证任意起点都能回到原点**

透视视图按什么顺序排成「视频」决定了帧间是否平滑，这是全文最核心的设计。核心约束是轨迹必须支持「无限循环」——从任意起点出发都能沿同一条轨迹精确回到该点，而行优先扫描做不到、只有列优先能满足。形式上第 $j$ 列的访问顺序为
$$\mathcal{O}_j = \begin{cases} (j,1),(j,2),\dots,(j,N_{pitch}), & j \bmod 2 = 1 \\ (j,N_{pitch}),\dots,(j,2),(j,1), & j \bmod 2 = 0 \end{cases}$$
相机先沿当前 yaw 列纵向扫一遍 pitch，再进一个 yaw 步、反向纵扫，逐列锯齿往复；这样连续两帧只变一个角度维度（yaw 或 pitch），过渡像真视频一样平滑。采样网格由 FoV 和重叠率 $r=0.5$ 定：$\Delta_{yaw} = \beta_h(1-r)$、$N_{yaw} = \lceil 360°/\Delta_{yaw} \rceil$。为了让模型不依赖固定起点，SAP 利用这条轨迹天然的循环性，把它首尾相接复制一倍得到 $2\cdot N$ 帧；训练时随机采起始索引 $s\in\{0,\dots,N-1\}$、截取连续 $N$ 帧作窗口，无论从哪帧开始都至少覆盖全部视点一次，模型对起点不敏感。

**3. 183K 合成数据集与轨迹对齐微调：补足 4K 全景标注的缺口**

真实 4K 全景的实例标注极稀缺。SAP 用 InfiniGen 引擎花 40,000 GPU 小时合成了 183,440 张 $4096\times2048$ 全景图，共 6,409,732 个实例掩码（物体尺寸分布：小型 37.84%、中型 25.70%、大型 36.47%）。这批数据用来按同一条扫描轨迹微调 SAM2，让它适应全景内容的同时不丢原有分割能力——微调时冻结图像编码器、只更新记忆相关模块与解码器，并混入 SAM2 原始数据（SA-1B + SA-V）防止灾难性遗忘。消融显示：仅靠扫描策略而不微调，在卡通风 8K 全景上反而掉点，轨迹对齐微调才是稳定跨视图传播的关键。

**4. 掩码反投影与最大值融合：把各帧结果拼回全景**

微调后的 SAM2 在伪视频上逐帧输出掩码后，需拼回 ERP 平面。每帧掩码按其透视参数反投影回球面、再映射到 ERP 网格；由于相邻帧有 50% 重叠，同一 ERP 像素会被多帧覆盖，SAP 用逐像素取最大值融合 $M^{ERP}(u,v) = \max_{i:\,(u,v)\in\mathcal{V}_i}\tilde{M}_i(u,v)$——只要任一覆盖帧判该像素为前景就保留，避免单帧漏检造成的空洞。

### 损失函数 / 训练策略

SAP 基于 SAM2（Hiera-Large 编码器）构建，冻结图像编码器，只更新 memory attention、memory encoder、mask decoder 和 prompt encoder。训练用合成全景数据 + SAM2 原始数据（SA-1B + SA-V）混合，防止灾难性遗忘；优化器 AdamW，batch size 128，lr $2\times10^{-4}$（余弦调度），weight decay 0.1，梯度裁剪 0.1。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [\[CVPR 2026\] Live Interactive Training for Video Segmentation](live_interactive_training_for_video_segmentation.md)
- [\[CVPR 2026\] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)
- [\[CVPR 2026\] Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera](unified_spherical_frontend_learning_rotation-equivariant_representations_of_sphe.md)
- [\[CVPR 2026\] Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](boundary_segment_action_segmentation.md)

</div>

<!-- RELATED:END -->
