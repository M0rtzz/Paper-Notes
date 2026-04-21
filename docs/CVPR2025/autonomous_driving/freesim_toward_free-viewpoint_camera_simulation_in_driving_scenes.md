---
title: >-
  [论文解读] FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes
description: >-
  [CVPR 2025][自动驾驶][自由视角仿真] 本文提出FreeSim，通过将挑战性的偏离轨迹新视角生成问题重新表述为生成式图像增强问题，配合piece-wise高斯重建的训练数据构造和渐进式视角扩展策略，首次实现了驾驶场景中超过3米横向偏移的高质量自由视角渲染。
tags:
  - CVPR 2025
  - 自动驾驶
  - 自由视角仿真
  - 3D高斯溅射
  - 扩散模型
  - 渐进式重建
  - 图像增强
  - 驾驶仿真
---

# FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes

**会议**: CVPR 2025  
**arXiv**: [2412.03566](https://arxiv.org/abs/2412.03566)  
**代码**: [drive-sim.github.io/freesim](https://drive-sim.github.io/freesim)  
**领域**: 自动驾驶  
**关键词**: 自由视角仿真, 3D高斯溅射, 扩散模型, 渐进式重建, 图像增强, 驾驶仿真

## 一句话总结

本文提出FreeSim，通过将挑战性的偏离轨迹新视角生成问题重新表述为生成式图像增强问题，配合piece-wise高斯重建的训练数据构造和渐进式视角扩展策略，首次实现了驾驶场景中超过3米横向偏移的高质量自由视角渲染。

## 研究背景与动机

### 领域现状
逼真的驾驶仿真是自动驾驶开发的基础组件。随着3D高斯溅射（3DGS）的出现，基于重建的驾驶场景仿真取得了快速进展（如PVG、StreetGS等），在已录制轨迹上实现了高质量渲染。

### 现有痛点
1. **偏离轨迹的渲染质量急剧下降**：现有方法仅能沿录制轨迹高质量渲染，当视角偏离录制轨迹（如横移3米模拟换道）时，渲染结果出现严重退化（鬼影、模糊、失真）
2. **缺乏偏离轨迹的GT数据**：车辆只能沿一条轨迹行驶，无法同时获取多条并行轨迹的图像，传统多视角生成模型（如从RealEstate10k、CO3D等数据集训练）所需的训练样本不可用
3. **现有生成方法受限**：纯生成方法如SGD仅处理旋转变化、FreeVS受限于LiDAR覆盖范围、GAN监督的UniSim在大偏移时效果有限

### 核心矛盾
如何在没有偏离轨迹GT数据的情况下，生成高质量、一致的自由视角渲染图像？

### 切入角度
关键洞察：如果视角偏移很小，从已重建的辐射场渲染的结果虽有退化但仍可辨识。从轻度退化的图像恢复高质量图像，远比从位姿变换条件直接生成新视角要容易得多。因此将新视角生成重新表述为**图像增强**任务。

### 核心idea
分两步走：(1) 构造匹配的"退化-高质量"训练数据对来训练生成式增强模型（利用piece-wise高斯重建+外推渲染+高斯扰动）；(2) 渐进式地将偏离轨迹视角加入重建训练集，从小偏移逐步扩展到大偏移，确保每次新增视角的渲染仅有轻微退化。

## 方法详解

### 整体框架
FreeSim是一个"生成-重建"混合系统。生成部分：基于构造的训练数据训练ControlNet增强模型，将退化渲染恢复为高质量图像。重建部分：以PVG为基础，渐进式地添加偏离轨迹视角的生成图像到训练集中进行重建，从小偏移（0.5m）逐步扩展到大偏移（3m+）。

### 关键设计

#### 1. 训练数据构造（Training Data Construction）

- **功能**：在没有偏离轨迹GT的情况下，构造匹配的"退化图像-高质量图像"训练对来训练增强模型
- **核心思路**：
    - **Piece-wise高斯重建**：将Waymo每条完整轨迹分割为多个短sub-segment（20帧），每个sub-segment快速重建一个小规模高斯场（<2分钟/segment，最多1M primtives），使整个Waymo（1150个场景）重建仅需40小时/8GPU
    - **外推渲染模拟退化**：对每个sub-segment保留最后4帧作为测试帧进行外推渲染（而非插值），模拟偏离轨迹视角的退化模式（侧向摄像头沿行进方向运动≈前向摄像头横向偏移）
    - **高斯扰动增强多样性**：随机采样部分高斯primitive施加平移噪声（最大0.2m）和旋转噪声（最大15°），模拟"目标鬼影"等退化模式
    - 总共构造约**150万**训练样本
- **设计动机**：直接使用位姿变换条件的多视角生成不可行（缺乏多轨迹数据）。将任务转化为图像增强后，只需要构造与偏离轨迹退化模式匹配的训练数据即可。piece-wise重建既高效又能自然产生外推退化模式

#### 2. 生成式增强模型

- **功能**：将退化渲染图像增强为高质量图像
- **核心思路**：
    - 基于Stable Diffusion v1.5，集成两个ControlNet分支：一个处理退化图像条件 $\mathbf{I}_d$，另一个处理可选的稀疏LiDAR投影条件 $\mathbf{I}_l$
    - 两个ControlNet的输出特征相加后融入UNet各分辨率层
    - 去除CLIP文本嵌入的交叉注意力
    - Image blending策略：训练时以0.1概率将退化图像与GT混合（$\alpha=0.5$），防止模型只学习修复严重退化
- **设计动机**：退化图像提供了强先验（几何结构大致正确），LiDAR条件补充近距离的精确深度信息。双ControlNet设计让两种条件独立编码、灵活组合
- **损失函数**：标准扩散训练损失 $\mathcal{L} = \mathbb{E}[\|\epsilon_\theta(z_t; c_d, c_l, t) - \epsilon\|_2^2]$

#### 3. 渐进式重建策略（Progressive Reconstruction）

- **功能**：将增强能力从小偏移扩展到大偏移，避免直接处理严重退化的渲染
- **核心思路**：
    - 先用标准PVG对录制轨迹进行预重建
    - 每隔5k迭代，将所有视角横向平移一个步长（默认0.5m），用增强模型生成新视角图像，加入训练集
    - 冻结新训练集，优化高斯场至近收敛，然后再次平移
    - 如此反复，从录制轨迹渐进扩展到远离轨迹的视角
    - 最终对渲染结果再用增强模型做后处理，消除滚动快门失真和生成随机性导致的轻微模糊
- **设计动机**：如果直接在大偏移位置渲染，图像可能完全损坏，增强模型无法有效恢复。渐进式策略确保每次新增的视角仅有轻微退化，增强模型可以从容处理

## 实验关键数据

### 主实验 — Waymo Open Dataset

| 方法 | 录制轨迹 PSNR↑ | 偏移1m FID↓ | 偏移2m FID↓ | 偏移3m FID↓ |
|------|---------------|------------|------------|------------|
| StreetGS | 28.01 | 25.8 | 35.4 | 47.6 |
| EmerNeRF | 29.18 | 32.3 | 40.2 | 49.8 |
| PVG (baseline) | 29.19 | 22.9 | 34.3 | 47.5 |
| **FreeSim** | 28.32 | **14.6** | **17.0** | **18.6** |

FreeSim在3m偏移时FID仅18.6，相比PVG baseline降低了**60.8%**。

### 消融实验

| 配置 | @1m FID | @2m FID | @3m FID |
|------|---------|---------|---------|
| Non-progressive | 20.1 | 26.3 | 29.7 |
| w/o LiDAR | 15.5 | 18.5 | 21.3 |
| Step size 1.0m | 14.5 | 16.9 | 18.4 |
| Default (0.5m) | **14.6** | **17.0** | **18.6** |

### 关键发现

1. **渐进式重建至关重要**：无渐进策略时3m偏移FID从18.6恶化至29.7，说明直接处理大偏移退化不可行
2. **LiDAR条件对大偏移更重要**：在3m偏移时，无LiDAR的FID从18.6升至21.3，但在1m时差异较小
3. **步长≤1m通常产生良好结果**：过大步长（1.5m）在大偏移时效果变差
4. **增强模型具有跨方法泛化性**：虽然训练数据仅用PVG构造，但可应用于StreetGS等其他重建方法的退化渲染
5. 后增强处理可有效消除高频区域的模糊（如树木、近处车辆）

## 亮点与洞察

1. **问题重新表述极为巧妙**：将"偏离轨迹视角生成"→"退化图像增强"的重新表述，既绕开了缺乏多轨迹GT的核心困难，又将问题简化到现有扩散模型容易处理的范畴
2. **数据构造策略完整且可扩展**：piece-wise高斯重建+外推渲染+高斯扰动三管齐下，高效构造了150万训练样本，成本可控
3. **渐进式策略思想通用**：从小偏移到大偏移的渐进扩展思想不仅适用于驾驶仿真，也可推广到其他需要外推渲染的场景重建任务
4. **实际应用意义重大**：自由视角仿真是真正实用的驾驶仿真器所必需的能力，FreeSim朝这个目标迈出了重要一步

## 局限性

1. 录制轨迹上的渲染质量有轻微下降（PSNR从29.19降至28.32），生成过程不可避免地引入微小不一致
2. 渐进式重建增加了总训练时间，每个场景需要额外的多轮生成-重建
3. 扩散生成的随机性导致不同视角间可能存在细节纹理不一致
4. 滚动快门失真问题虽通过后处理缓解，但未从根本解决
5. 实验仅选取了16个场景，更大规模验证需要进一步工作

## 相关工作与启发

- **与SGD的对比**：SGD基于参考图像和深度图的扩散生成，主要处理旋转变化，难以应对大空间平移。FreeSim通过退化-增强的方式突破了这一限制
- **与FreeVS的对比**：FreeVS使用LiDAR投影作为伪图像条件，受限于LiDAR覆盖范围。FreeSim将LiDAR仅作为辅助条件，主条件是退化渲染图像，覆盖更全面
- **Piece-wise重建的效率启示**：将大场景分割为小片段重建的策略在数据构造中极为高效（6倍加速），这一思路可用于其他大规模场景的预处理
- **对World Model的补充**：相比DriveDreamer4D等基于world model的方法，FreeSim基于重建+增强的方式可能更适合合成非常规轨迹（如视角升高等world model未见过的运动模式）

## 评分

⭐⭐⭐⭐⭐ (5/5)

问题重要且困难，核心思路（退化增强+渐进扩展）优雅且有效。实验结果在大偏移场景下大幅超越所有基线，消融实验详尽。整体完成度高，对驾驶仿真领域有重要推动作用。

<!-- RELATED:START -->

## 相关论文

- [Generating Multimodal Driving Scenes via Next-Scene Prediction](generating_multimodal_driving_scenes_via_next-scene_prediction.md)
- [Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](scenario_dreamer_vectorized_latent_diffusion_for_generating_driving_simulation_e.md)
- [SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)
- [M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)
- [LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes](limoe_mixture_of_lidar_representation_learners_from_automotive_scenes.md)

<!-- RELATED:END -->
