---
title: >-
  [论文解读] Rethink Sparse Signals for Pose-guided Text-to-Image Generation
description: >-
  [ICCV 2025][图像生成][Sparse Pose] 提出 SP-Ctrl（Spatial-Pose ControlNet），通过可学习空间姿态表示（SPR）和关键点概念学习（KCL）两个核心策略，使稀疏信号（OpenPose）在姿态引导文生图任务中达到与稠密信号（深度图、DensePose）相当的姿态对齐精度，同时保留稀疏信号在多样性生成和跨物种泛化上的天然优势。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Sparse Pose"
  - "OpenPose"
  - "ControlNet"
  - "Keypoint Concept Learning"
  - "Spatial Controllable Generation"
---

# Rethink Sparse Signals for Pose-guided Text-to-Image Generation

**会议**: ICCV 2025  
**arXiv**: [2506.20983](https://arxiv.org/abs/2506.20983)  
**代码**: [GitHub](https://github.com/) (论文提及 "Codes are available here")  
**领域**: 图像生成 / 姿态引导生成 / ControlNet  
**关键词**: Sparse Pose, OpenPose, ControlNet, Keypoint Concept Learning, Spatial Controllable Generation

## 一句话总结
提出 SP-Ctrl（Spatial-Pose ControlNet），通过可学习空间姿态表示（SPR）和关键点概念学习（KCL）两个核心策略，使稀疏信号（OpenPose）在姿态引导文生图任务中达到与稠密信号（深度图、DensePose）相当的姿态对齐精度，同时保留稀疏信号在多样性生成和跨物种泛化上的天然优势。

## 研究背景与动机

### 问题背景
姿态引导文生图（T2I）生成是一个重要任务，涉及人体/动物中心的图像生成、动画驱动和条件 3D 生成等应用。近期方法倾向使用**稠密信号**（depth map、normal map、DensePose、SMPL）来获得更精确的姿态控制。然而稠密信号存在明显缺陷：
1. **不灵活**：通常需要从参考图像提取，难以创建和编辑
2. **与文本矛盾**：对物体形状和轮廓施加强约束，可能与文本提示冲突导致生成质量下降

### 稀疏信号的优势
- **形状无关**：作为解剖学抽象，不限制具体形态
- **类别无关**：关键点定义在物种间共享（如哺乳动物），提供统一姿态表示
- **可操作性**：不依赖参考图像，创建和编辑自由度高

### 核心挑战
稀疏信号（如 OpenPose）的姿态控制精度远不如稠密信号——关键问题在于：
1. **OpenPose 本为可视化设计**：RGB 颜色编码几乎不提供有效信息，甚至可能混淆关键点的区分
2. **稀疏性导致空间感知困难**：点状和线段指令对模型来说难以精确感知和跟随

## 方法详解

### 整体框架
SP-Ctrl 构建在 Stable Diffusion v1.5 + ControlNet 之上，包含三个组件：(1) 空间姿态嵌入模块渲染学习后的空间姿态表示；(2) 可学习关键点文本嵌入用于关键点概念学习；(3) ControlNet adapter 注入空间条件到冻结的 SD 模型。

### 关键设计一：Spatial-Pose Representation (SPR)
将 OpenPose 的固定 RGB 关键点颜色替换为**可学习嵌入**。具体地，空间姿态嵌入模块 $\mathcal{G}(\cdot; \phi)$ 将随机初始化向量 $\boldsymbol{E}_0 = \{\boldsymbol{e}_k \in \mathbb{R}^{1 \times C}\}_{k=1}^N$ 映射为关键点嵌入：

$$\boldsymbol{E}_{kpt} = \mathcal{G}(\boldsymbol{E}_0; \phi)$$

$\mathcal{G}$ 是由堆叠线性层和 LayerNorm 组成的 MLP，其参数 $\phi$ 通过去噪任务的梯度自适应优化。骨架嵌入设为全 1 向量 $\boldsymbol{e}_{sks} = \mathbf{1}^{1 \times C'}$。最终渲染多通道骨架姿态图 $\boldsymbol{I}_{sp} \in \mathbb{R}^{H \times W \times C'}$ 作为 ControlNet 输入。

关键发现：
- 随机初始化 >> 文本嵌入初始化（mAP 高 0.88%，因文本和空间表示空间存在差异）
- 固定 $\boldsymbol{E}_0$ + 可学习 $\mathcal{G}$ >> 可学习 $\boldsymbol{E}_0$（更稳定的优化）
- 甚至 1 通道嵌入就能取得竞争性表现——说明学习后的表示高度富有表达力

### 关键设计二：Keypoint Concept Learning (KCL)
受交叉注意力图与关键点位置的空间关联性启发，引入可学习关键点 token 并通过热图约束增强注意力对齐：

1. **引入关键点 token**：为每个关键点描述（如 eye、nose、elbow）添加新文本 token $\{\langle \boldsymbol{k}_i \rangle\}_{i=1}^N$，对应可学习嵌入 $\mathcal{V}_{kpt} = \{\boldsymbol{v}^*_i \in \mathbb{R}^{768}\}_{i=1}^N$

2. **热图约束损失**：提取交叉注意力图 $\mathcal{M}_{kpt}$，鼓励其与关键点高斯热图 $\mathcal{H}$ 对齐：

$$\mathcal{L}_{ht} = \frac{1}{|\mathcal{M}_{kpt}|} \cdot \frac{1}{H'W'} \sum_{v_i \geq 1} \|(\mathcal{M}_i - \mathcal{H}_i)\|^2$$

3. **梯度截断**：对 noisy image query $Q$ 的梯度进行 detach 以避免信息泄露导致的外观坍塌

### 训练目标
联合优化空间嵌入模块、关键点嵌入和 ControlNet adapter：

$$\phi^*, \boldsymbol{V}_{kpt}^*, \Theta^* = \arg\min_{\phi, v_i^*, \Theta} \mathcal{L}_{ldm} + \eta \cdot \mathcal{L}_{ht}$$

其中 $\eta = 0.1$，热图约束仅在 250∼500 时间步、U-Net 第 3 个 Transformer block 上计算。

## 实验

### 主实验结果

| 数据集 | 方法 | Pose mAP↑ | FID↓ | CLIP-Score↑ | Detection AP.75↑ |
|--------|------|-----------|------|-------------|-------------------|
| AP-10K | T2I-Adapter | 48.16 | 27.29 | 25.52 | 24.23 |
| AP-10K | ControlNet | 44.25 | 19.40 | 24.77 | 24.35 |
| AP-10K | **SP-Ctrl** | **55.63** | **18.52** | 23.86 | **25.10** |
| Human-Art | ControlNet | 45.26 | 26.69 | 27.84 | 8.18 |
| Human-Art | HumanSD† | 49.92 | 35.18 | 27.35 | 8.29 |
| Human-Art | GRPose† | 50.93 | 28.85 | 27.95 | 6.51 |
| Human-Art | **SP-Ctrl** | **51.11** | 29.30 | 25.94 | **9.11** |

在 AP-10K 上 SP-Ctrl 的姿态 mAP 比 ControlNet 基线提升 **11.38%**，FID 低 0.88；在 Human-Art 上达到与 SOTA GRPose 相当的 mAP（51.11 vs 50.93），**无需额外预训练姿态估计器**。

### 消融实验

| 方法 | Pose mAP↑ | FID↓ | CLIP-Score↑ |
|------|-----------|------|-------------|
| ControlNet (baseline) | 44.25 | 19.40 | 24.77 |
| + Spatial Pose (SPR) | 52.85 | 19.67 | 24.62 |
| + KCL | 51.34 | 18.94 | 24.09 |
| **SP-Ctrl (两者结合)** | **55.63** | **18.52** | 23.86 |

SPR 单独提升 8.60% mAP，KCL 单独提升 7.09% mAP，两者结合时效果叠加，达到最优 55.63%。

### KCL 细粒度消融

| 组件 | mAP↑ |
|------|------|
| ControlNet + OpenPose (baseline) | 44.25 |
| + $\mathcal{V}_{kpt}$ (仅关键点token) | 50.38 |
| + $\mathcal{V}_{kpt}$ + $\mathcal{L}_{ht}$ (热图约束) | **51.34** |

关键点 token 本身提升 6.13%，热图约束进一步提升。时间步 250∼500 对概念形成最关键；U-Net 第 3 个 Transformer block 的注意力图贡献最大。

### 稀疏 vs 稠密信号对比
SP-Ctrl 使用稀疏信号达到的 mAP 与稠密信号（depth map）方法可比，同时在**图像多样性**和**跨物种泛化**上具有天然优势——同一套稀疏关键点可驱动不同物种的生成。

## 亮点与洞察
- **重新审视稀疏信号的价值**：在稠密信号主导的趋势中，证明稀疏信号经过合理增强后可达到竞争性精度，同时保留形状/类别无关的优势
- **OpenPose 颜色编码的负面影响**：实验证明固定 RGB 颜色甚至会混淆关键点识别，学习后的嵌入即使只有 1 个通道也更有效
- **利用交叉注意力的空间对应性**：发现去噪 U-Net 的 cross-attention 自然与关键点位置存在空间对齐，通过热图约束显式强化这一对应
- **极低额外推理开销**：SPR 和 KCL 在推理时几乎不增加计算量（SPR 只是替换输入表示，KCL 只添加少量 text token）
- **跨物种泛化**：同一模型可从鱼类关键点生成鸟类图像，展现稀疏信号的类别无关性

## 局限性
- 基于 SD v1.5，未在更新的扩散模型架构上验证
- CLIP-Score 由于去除了新引入的关键点 token 而略有下降（约 0.68），虽然视觉质量良好
- 关键点定义仍需预先指定，无法自动发现新类别的关键点
- 在遮挡严重或关键点缺失较多的情况下表现未详细分析

## 相关工作
- **空间可控 T2I**：ControlNet、T2I-Adapter、GLIGEN
- **姿态引导信号**：OpenPose（稀疏）、DensePose/SMPL（稠密）、图学习（GRPose）
- **姿态增强策略**：HumanSD（感知损失）、GRPose（图学习）、SpaText（可学习嵌入）

## 评分
- 新颖性：⭐⭐⭐⭐ — 重新审视稀疏信号并提出两个简洁有效的增强策略
- 技术深度：⭐⭐⭐⭐ — 对 OpenPose 编码问题和注意力机制的分析深入
- 实验充分度：⭐⭐⭐⭐ — 动物+人体两个数据集，详尽的消融实验
- 实用价值：⭐⭐⭐⭐ — 推理无额外开销，与 ControlNet 兼容，跨物种泛化性好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion](dual_recursive_feedback_on_generation_and_appearance_latents_for_pose-robust_tex.md)
- [\[ICCV 2025\] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](tera_rethinking_text-guided_realistic_3d_avatar_generation.md)
- [\[NeurIPS 2025\] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation](../../NeurIPS2025/image_generation/scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](lusd_localized_update_score_distillation_for_text-guided_image_editing.md)
- [\[CVPR 2025\] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis](../../CVPR2025/image_generation/exploring_sparse_moe_in_gans_for_text-conditioned_image_synthesis.md)

</div>

<!-- RELATED:END -->
