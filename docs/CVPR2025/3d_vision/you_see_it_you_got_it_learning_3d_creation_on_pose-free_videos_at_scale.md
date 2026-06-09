---
title: >-
  [论文解读] You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale
description: >-
  [CVPR 2025][3D视觉][多视图扩散] 本文提出 See3D，一个在大规模互联网视频（320M 帧/16M 视频片段）上训练的无位姿视觉条件多视图扩散模型，通过自动化数据筛选管线和时间依赖的视觉条件设计，实现了零样本开放世界 3D 生成能力。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "多视图扩散"
  - "无位姿训练"
  - "视频数据"
  - "3D生成"
  - "大规模学习"
---

# You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale

**会议**: CVPR 2025  
**arXiv**: [2412.06699](https://arxiv.org/abs/2412.06699)  
**代码**: [项目页面](https://vision.baai.ac.cn/see3d)  
**领域**: 3D视觉  
**关键词**: 多视图扩散, 无位姿训练, 视频数据, 3D生成, 大规模学习

## 一句话总结

本文提出 See3D，一个在大规模互联网视频（320M 帧/16M 视频片段）上训练的无位姿视觉条件多视图扩散模型，通过自动化数据筛选管线和时间依赖的视觉条件设计，实现了零样本开放世界 3D 生成能力。

## 研究背景与动机

- **3D 数据的稀缺性**：现有 3D 生成模型依赖昂贵的 3D "金标签"（如 Objaverse 的 0.8M 物体）或 2D 扩散先验，性能受限于有限的 3D 数据规模。构建大规模 3D 数据集对学术界仍是负担。
- **视频作为 3D 数据源**：人类的 3D 感知来自多视图观察而非特定 3D 表示。互联网视频提供了海量、多样、低成本的多视图图像源。
- **核心挑战**：(1) 从原始视频中筛选出 3D 感知数据（静态场景+大范围视角变化），(2) 在无 3D 几何和位姿标注的情况下学习通用 3D 先验。
- **位姿标注的瓶颈**：现有多视图扩散模型通常需要精确相机位姿作为条件输入，但为网络规模视频标注位姿代价极高或不可行。
- **核心思路**："You See it, You Got it"——仅通过观看视频内容获取 3D 知识。提出纯 2D 归纳的视觉条件（而非 3D 归纳的位姿条件），使模型能在无位姿标注的视频上大规模训练。

## 方法详解

### 整体框架

See3D 包含三个核心组件：
1. **数据筛选管线**：从 25.48M 原始视频中自动筛选出 15.99M 个 3D 感知视频片段
2. **视觉条件多视图扩散模型**：无需位姿标注，通过时间依赖的视觉条件学习相机控制
3. **基于 warping 的 3D 生成框架**：利用 See3D 进行长序列新视角合成

### 关键设计

**1. 自动化视频数据筛选管线**
- **功能**：从海量互联网视频中识别并提取静态场景+大视角变化的 3D 感知数据
- **核心思路**：四步管线——(a) 时空下采样提高效率，(b) Mask R-CNN 语义识别动态物体（人、动物等），(c) 光流估计精确过滤动态区域，(d) 关键点轨迹追踪过滤小视角变化视频。最终从 Pexels、Artgrid、Airvuz、Skypixel 四个来源获取 320M 帧的 WebVi3D 数据集
- **设计动机**：动态内容会扭曲场景几何，小视角变化提供不了足够的 3D 观测。人工标注验证显示管线准确率达 88.6%

**2. 时间依赖的视觉条件**
- **功能**：在无位姿标注情况下隐式控制相机运动，且能泛化到下游任务的特定视觉条件
- **核心思路**：对目标图像 $G$ 进行三步处理：(1) 随机不规则 masking 减少对像素信号的直接依赖，(2) 添加时间依赖噪声 $C_t = \sqrt{\bar{\alpha}_{t'}} (1-M)X_0 + \sqrt{1-\bar{\alpha}_{t'}} \epsilon$，通过函数 $t' = f(t)$ 调控信号泄露程度，(3) 时间依赖混合 $V_t = W_t \cdot C_t + (1-W_t) \cdot X_t$，使模型在大时间步依赖视觉线索、小时间步依赖 $X_t$，减少域差距
- **设计动机**：直接使用视频帧作为条件会导致对视频数据的过拟合（信号泄露），而纯噪声又会丧失相机控制。时间依赖的噪声+混合机制在两者间取得平衡，使条件成为通用的"视觉暗示"

**3. 迭代稀疏像素级深度对齐**
- **功能**：在 warping-based 3D 生成中纠正几何估计误差，防止累积退化
- **核心思路**：在迭代生成过程中，通过关键点匹配建立锚点视图间的对应关系，利用单目深度估计+稀疏对齐恢复稠密深度图，用于 warp 生成后续视图。包含像素级深度尺度对齐和全局度量深度恢复两步
- **设计动机**：直接使用单目深度会因尺度歧义和估计误差导致 warp 图像的畸变和拉伸，误差在迭代中累积严重

### 损失函数

标准条件扩散训练目标：
$$\mathbb{E}_{X_0, Y_0, \epsilon, t}\left[\|\epsilon_\theta(X_t, Y_0, V_t, t) - \epsilon\|_2^2\right]$$

其中 $V_t$ 为视觉条件，$Y_0$ 为参考视图，损失仅在目标图像上计算。

## 实验关键数据

### 主实验：单视图 3D 重建（GSO 数据集）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 训练数据 |
|------|:---:|:---:|:---:|:---:|
| Zero123++ | 中等 | 中等 | 中等 | Objaverse (0.8M) |
| SV3D | 中高 | 中高 | 中低 | Objaverse + 视频 |
| **See3D** | **最高** | **最高** | **最低** | **WebVi3D (16M)** |

### 稀疏视图重建（DTU 数据集）

| 方法 | PSNR ↑ | SSIM ↑ | 训练数据 |
|------|:---:|:---:|:---:|
| PixelNeRF | 低 | 低 | 3D 数据 |
| ZeroNVS | 中等 | 中等 | 3D + pose |
| **See3D** | **最高** | **最高** | **视频 (无 pose)** |

### 关键发现

- See3D 仅用视频数据训练，在多个基准上超越了在昂贵 3D 数据集（需要位姿标注）上训练的模型
- WebVi3D 数据集规模比 Objaverse 大 20 倍，比 RealEstate10K 大 200 倍
- 视觉条件无需位姿标注，且能零样本迁移到 warping-based 生成和 3D 编辑任务
- 数据规模是关键——消融实验显示更多数据持续提升性能
- 模型在物体级和场景级 3D 生成上均有效

## 亮点与洞察

1. **数据驱动的 3D 学习范式**：证明了大规模视频数据可以替代昂贵的 3D 标注数据，为 3D 生成的可扩展训练开辟了新路径
2. **视觉条件的通用性**：时间依赖噪声+混合机制使同一条件格式适用于训练（视频）和推理（warped images）的不同域
3. **自动化数据管线的实用价值**：88.6% 准确率的管线可持续从不断增长的互联网视频中获取 3D 数据
4. **"无位姿"训练的突破**：完全消除了位姿标注需求，使训练可以扩展到任意规模

## 局限与展望

- 视频数据主要来自自然场景，对人造物体和室内场景的覆盖可能不足
- 无位姿训练的相机控制精度可能不如有位姿方法
- 迭代生成的长序列仍可能累积误差
- 未来可结合视频预训练和少量 3D 微调实现更好的几何精度

## 相关工作与启发

- **Zero123/Zero123++**：在 Objaverse 上训练的多视图生成，但受限于数据规模
- **SV3D**：结合视频和 3D 数据，但仍需位姿
- **SDS 系列**：利用 2D 扩散先验优化 3D，但 2D→3D 转换质量差
- **Emu3**：视频理解中的场景分割方法，本文用类似管线做数据筛选
- 启发：AI 系统可以像人类一样"通过看来学习 3D"，大规模多视图观察是可扩展 3D 理解的关键

## 评分

⭐⭐⭐⭐⭐ — 这是一个有潜力改变 3D 生成范式的工作。数据规模（16M 视频片段/320M 帧）和无位姿训练的组合使其具有极强的可扩展性。视觉条件设计优雅且通用。在多个基准上超越了用昂贵 3D 数据训练的方法。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Perceptual Inductive Bias is What You Need Before Contrastive Learning](perceptual_inductive_bias_is_what_you_need_before_contrastive_learning.md)
- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[ICCV 2025\] Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels](../../ICCV2025/3d_vision/do_it_yourself_learning_semantic_correspondence_from_pseudo-labels.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering](../../NeurIPS2025/3d_vision/you_can_trust_your_clustering_model_a_parameter-free_self-boosting_plug-in_for_d.md)

</div>

<!-- RELATED:END -->
