---
title: >-
  [论文解读] MVSMamba: Multi-View Stereo with State Space Model
description: >-
  [NeurIPS 2025][多视图立体] 提出MVSMamba，首个基于Mamba架构的多视图立体(MVS)网络，通过参考视角中心的动态扫描策略实现高效的视内和视间全局全方向特征聚合，在DTU和Tanks-and-Temples上以最优效率达到SOTA性能。
tags:
  - NeurIPS 2025
  - 多视图立体
  - Mamba
  - 状态空间模型
  - 特征匹配
  - 深度估计
---

# MVSMamba: Multi-View Stereo with State Space Model

**会议**: NeurIPS 2025  
**arXiv**: [2511.01315](https://arxiv.org/abs/2511.01315)  
**代码**: https://github.com/JianfeiJ/MVSMamba  
**领域**: 3D视觉  
**关键词**: 多视图立体, Mamba, 状态空间模型, 特征匹配, 深度估计

## 一句话总结

提出MVSMamba，首个基于Mamba架构的多视图立体(MVS)网络，通过参考视角中心的动态扫描策略实现高效的视内和视间全局全方向特征聚合，在DTU和Tanks-and-Temples上以最优效率达到SOTA性能。

## 研究背景与动机

多视图立体(MVS)旨在从标定的多视图图像重建密集3D几何，核心依赖高质量的跨视图特征匹配。鲁棒的特征表示是可靠匹配的基础。

现有方法的发展脉络和局限：
- **CNN-based方法**（CasMVSNet等）：效率高但受限于局部感受野，在无纹理、反光等困难区域表现不佳
- **Transformer-based方法**（TransMVSNet、MVSFormer++等）：引入全局依赖建模提升性能，但注意力机制的二次复杂度带来效率问题。即便使用线性注意力、极线窗口注意力等优化，仍需多轮self-attention和cross-attention交替计算，开销大。或者依赖参数庞大的预训练ViT（如MVSFormer++使用39.48M参数）

核心问题：**如何在保持高性能的同时最小化计算成本？**

Mamba作为状态空间模型的高效变体，具有线性复杂度的长距离依赖建模能力，为解决这一问题提供了理想方案。但Mamba的1D序列化扫描不能直接适配MVS中一对多的特征匹配需求，需要专门设计扫描策略。

## 方法详解

### 整体框架

MVSMamba基于coarse-to-fine范式，整体流程为：(1) FPN编码器提取多尺度特征；(2) 在FPN的底层（第0尺度）插入Dynamic Mamba module (DM-module)进行参考-源特征间的全局交互；(3) 在FPN解码器的第1尺度插入Simplified DM-module (SDM-module)进行单视图内特征增强；(4) 基于增强后的金字塔特征通过warping、代价体构建和正则化进行由粗到精的深度估计。

### 关键设计

1. **Dynamic Mamba Module (DM-module)**: 核心创新。对每对参考-源特征，将源特征拼接到参考特征的上、下、左、右四个方向，形成四个拼接特征图。分别用四个不同方向的skip scan（步长为2）扫描这四个拼接图，生成四条方向序列。关键是**参考中心**设计：扫描方向从参考视角出发朝向源视角，使源特征能学习到来自参考特征的全局表示。四条序列分别送入四个独立的Mamba block处理，再merge回2D特征图。步长2使序列长度为像素总数的1/4，提升效率。

2. **Reference-Centered Dynamic Scanning（参考中心动态扫描）**: 扫描起始坐标(h_k, w_k)根据源图像索引k动态更新。对不同源视图，参考特征中每个像素的扫描方向不同，因此当K≥5（至少4个源视图）时，参考特征可获得全方向(omnidirectional)的全局感受野。这解决了单一扫描方向导致特征聚合缺乏全方向性的问题，且无需像Transformer那样多轮self/cross-attention交替计算。

3. **Simplified DM-module (SDM-module)与多尺度聚合**: SDM-module仅处理单视图特征（不需要参考-源拼接），直接扫描输入特征生成四个方向序列。DM-module仅部署在第0尺度捕获视间交互，SDM-module部署在第1尺度做视内增强。实验发现在更多尺度上添加DM/SDM不带来额外增益（因为第0尺度的交互已通过decoder传播到所有尺度），确保了效率。

4. **特征合并(Feature Merging)**: 将四条增强序列逆扫描回四个拼接特征图，分别从参考区域和源区域提取增强特征，对参考特征的四个方向分量求和得到最终的全方向增强特征。

### 损失函数 / 训练策略

在每个尺度上使用交叉熵损失监督概率体积。DTU训练：5视图×512×640，batch 4，15 epoch，lr=0.001分阶段衰减。BlendedMVS微调用于Tanks-and-Temples评估：11视图×576×768，batch 2，15 epoch。高分辨率训练：5视图×1024×1280，10 epoch。深度假设数: 32-16-8-4（由粗到精），深度间隔: 2-1-1-0.5。使用Adam优化器。最终点云通过动态融合策略获得。

## 实验关键数据

### 主实验

**DTU数据集（点云重建质量+效率）**：

| 方法 | 类型 | Overall ↓ | Acc. ↓ | Comp. ↓ | GPU(G) ↓ | Time(s) ↓ | Params(M) ↓ |
|------|------|-----------|--------|---------|----------|-----------|-------------|
| MVSMamba* (Ours) | Mamba | **0.280** | **0.308** | **0.252** | 2.82 | 0.11 | 1.31 |
| MVSFormer++ | Trans. | 0.281 | 0.309 | 0.252 | 4.71 | 0.23 | 39.48 |
| ET-MVSNet | Trans. | 0.291 | 0.329 | 0.253 | 2.91 | 0.16 | 1.09 |
| CasMVSNet | CNN | 0.355 | 0.324 | 0.385 | 4.48 | 0.18 | 0.93 |

性能最佳且效率最优（GPU内存最低、推理速度最快、参数量仅1.31M，远小于MVSFormer++的39.48M）。综合排名(Avg. Rank)第一(2.50)。

**Tanks-and-Temples（泛化性评估）**：

| 方法 | Intermediate Mean ↑ | Advanced Mean ↑ |
|------|---------------------|-----------------|
| MVSMamba (Ours) | **67.67** | **43.32** |
| MVSFormer++ | 67.18 | 41.60 |
| GoMVS | 66.44 | 43.07 |
| GeoMVSNet | 65.89 | 41.52 |

在Intermediate和Advanced两个集合上均取得最佳F-score。

### 消融实验

| 配置 | Overall ↓ | MAE ↓ | GPU(G) | Time(s) | 说明 |
|------|-----------|-------|--------|---------|------|
| Full MVSMamba | **0.287** | **5.21** | 2.82 | 0.11 | 完整模型 |
| w/o DM | 0.295 | 5.58 | 2.82 | 0.104 | 去掉DM-module，性能明显下降 |
| w/o SDM | 0.289 | 5.45 | 2.82 | 0.097 | 去掉SDM，影响较小 |
| w/o MLP | 0.293 | 5.23 | 2.82 | 0.108 | 去掉MLP增强 |
| w/ VMamba scan | 0.291 | 5.30 | 2.82 | 0.13 | 用四方向扫描替换 |
| w/ EVMamba scan | 0.298 | 5.81 | 2.82 | 0.11 | Skip scan但无动态策略 |
| w/ JamMa scan | 0.301 | 6.01 | 2.82 | 0.11 | Joint scan不适合MVS |

### 关键发现

- DM-module贡献最大（去掉后Overall从0.287升至0.295），因为它在FPN底层捕获了视间长距离依赖并传播到所有尺度
- 动态扫描策略大幅优于VMamba四方向扫描、EVMamba skip scan和JamMa joint scan，验证了参考中心+动态起始坐标的设计
- 四个Mamba block不共享权重优于共享（仅增加0.1M参数），说明不同扫描方向需要学习不同信息
- 参考中心扫描优于源中心扫描，因为源特征需要从参考特征学习一致的全局表示
- DM-module仅在第0尺度部署即可，在更多尺度添加反而不涨甚至掉点

## 亮点与洞察

- 首次将Mamba引入MVS领域，证明线性复杂度的SSM可以达到Transformer性能的同时效率远优
- 参考中心动态扫描策略是核心创新：巧妙地将Mamba的1D扫描适配到MVS的一对多匹配需求，不同源视图提供不同扫描方向使参考特征获得全方向感受野
- 与MVSFormer++对比：性能相当(0.280 vs 0.281)但参数量少30倍(1.31M vs 39.48M)，推理速度快2倍(0.11s vs 0.23s)，GPU内存少40%(2.82G vs 4.71G)
- 在Tanks-and-Temples Advanced上也取得SOTA，证明泛化性强

## 局限与展望

- Mamba是因果模型（仅利用前文信息），虽然通过四方向扫描缓解但仍非完全双向
- 当前DM-module仅在最低分辨率部署，高分辨率下Mamba的效率优势可能更显著但未探索
- 仅验证了coarse-to-fine范式，与迭代更新范式（如RAFT-Stereo衍生方法）的结合值得探索
- 动态扫描策略依赖至少4个源视图才能达到全方向覆盖

## 相关工作与启发

- VMamba将四方向扫描引入视觉任务，MVSMamba进一步发展为参考中心+动态起始坐标
- TransMVSNet首次引入Transformer到MVS，但需交替self/cross-attention，效率低
- JamMa提出Joint Mamba用于特征匹配，但其joint scan不适合MVS的一对多场景
- EVMamba的skip scan提升效率但感受野受限，MVSMamba通过动态起始坐标解决

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐
