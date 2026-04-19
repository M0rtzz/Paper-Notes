---
title: >-
  [论文解读] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving
description: >-
  [CVPR 2025][自动驾驶][扩散模型] 本文提出DiffusionDrive，通过截断扩散策略（将去噪步骤从20步减少到2步）和级联扩散解码器，首次将扩散模型成功应用于端到端自动驾驶的实时多模态轨迹规划，在NAVSIM数据集上以88.1 PDMS刷新记录，同时保持45 FPS的实时速度。
tags:
  - CVPR 2025
  - 自动驾驶
  - 扩散模型
  - 端到端自动驾驶
  - 多模态轨迹规划
  - 截断扩散策略
  - 实时规划
---

# DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving

**会议**: CVPR 2025  
**arXiv**: [2411.15139](https://arxiv.org/abs/2411.15139)  
**代码**: [hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive)  
**领域**: 自动驾驶  
**关键词**: 扩散模型, 端到端自动驾驶, 多模态轨迹规划, 截断扩散策略, 实时规划

## 一句话总结
本文提出DiffusionDrive，通过截断扩散策略（将去噪步骤从20步减少到2步）和级联扩散解码器，首次将扩散模型成功应用于端到端自动驾驶的实时多模态轨迹规划，在NAVSIM数据集上以88.1 PDMS刷新记录，同时保持45 FPS的实时速度。

## 研究背景与动机

端到端自动驾驶近年来取得了显著进展，主流方法（如Transfuser、UniAD、VAD）通常从ego-query回归单条轨迹，但这种范式忽略了驾驶行为的内在不确定性和多模态特性。VADv2引入大规模固定锚点词表（4096-8192个锚点）来离散化连续动作空间，但受限于锚点的数量和质量，难以覆盖词表外场景，且计算开销巨大。

扩散模型在机器人策略学习中展现了强大的多模态动作分布建模能力。然而，直接将vanilla扩散策略应用于自动驾驶面临两大问题：(1) 需要20步去噪，FPS从60降至7，无法满足实时需求；(2) 从不同高斯噪声采样的轨迹严重重叠，存在模态坍塌问题。

核心洞察是：与从随机高斯噪声开始去噪不同，人类驾驶遵循固定的驾驶模式，然后根据实时交通状况动态调整。因此，可以将先验驾驶模式嵌入扩散策略中，从锚定的高斯分布（而非标准高斯分布）开始去噪，从而大幅减少去噪步骤。

## 方法详解

### 整体框架
DiffusionDrive由感知模块和扩散解码器两部分组成。感知模块可以集成各种已有的端到端感知架构（UniAD、VAD、Transfuser等），接收不同传感器输入（相机、LiDAR）。扩散解码器从锚定高斯分布中采样噪声轨迹，通过增强的场景上下文交互逐步去噪，生成最终的多模态规划轨迹。

### 关键设计

1. **截断扩散策略（Truncated Diffusion Policy）**:

    - 核心思路：不从纯高斯噪声开始，而是从锚定高斯分布开始去噪
    - 通过K-Means对训练集轨迹聚类得到少量锚点（仅需20个，相比VADv2的8192个减少了400倍），然后在锚点周围添加少量高斯噪声，形成锚定高斯分布
    - 训练时截断扩散调度（50/1000），仅在锚点附近扩散
    - 推理时从锚定高斯分布开始，仅需2步去噪（相比vanilla的20步减少10倍）
    - 每个锚点同时预测分类得分和去噪轨迹，最终选择得分最高的轨迹作为输出
    - 推理灵活性：推理时采样数量可以动态调整，不必等于训练时的锚点数

2. **级联扩散解码器（Cascade Diffusion Decoder）**:

    - 基于Transformer架构，替代了UNet
    - 通过可变形空间交叉注意力（Deformable Spatial Cross-attention）与BEV和透视图特征交互
    - 与感知模块输出的agent/map查询进行交叉注意力
    - 使用Timestep Modulation层编码扩散时间步信息
    - 级联机制：堆叠多个解码器层，逐步精化轨迹重建
    - 跨去噪时间步共享参数，减少参数量（从102M降至60M）

3. **训练目标**:

    - 轨迹重建损失：L1重建损失，仅对与ground truth最近的锚点（正样本）计算
    - 分类损失：BCE损失，区分正负样本
    - 总损失 = Σ[y_k * L_rec + λ * BCE]

### 损失函数 / 训练策略
- 对每个训练样本，找到距离ground truth轨迹最近的锚点作为正样本
- 正样本计算轨迹L1重建损失，所有样本计算BCE分类损失
- 使用AdamW优化器，学习率6×10⁻⁴，在8块4090 GPU上训练100个epoch
- 总batch size 512，无测试时增强

## 实验关键数据

### 主实验
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| NAVSIM navtest | PDMS | 88.1 | 86.5 (Hydra-MDP-W-EP) | +1.6 |
| NAVSIM navtest | EP | 82.2 | 78.7 (Hydra-MDP-W-EP) | +3.5 |
| NAVSIM navtest | DAC | 96.2 | 96.0 (Hydra-MDP-W-EP) | +0.2 |
| nuScenes | Avg L2 (m) | 0.57 | 0.61 (SparseDrive) | -6.6% |
| nuScenes | Avg Collision (%) | 0.08 | 0.08 (SparseDrive) | 持平 |

### 消融实验
| 配置 | 关键指标 (PDMS) | 说明 |
|------|---------|------|
| Transfuser (baseline) | 84.0 | 单模态回归 MLP |
| TransfuserDP (vanilla diffusion) | 84.6 (+0.6) | 20步去噪，FPS=7，模态多样性11% |
| TransfuserTD (truncated diffusion) | 85.7 (+1.7) | 2步去噪，FPS=27，模态多样性70% |
| DiffusionDrive (完整) | 88.1 (+4.1) | 2步去噪，FPS=45，模态多样性74% |
| 无空间交叉注意力 | 55.1 | 性能严重退化，说明空间交互至关重要 |
| 仅1步去噪 | 87.9 | 仅1步已可获得良好性能 |
| 10个采样噪声 | 84.9 | 少量采样即可获得不错效果 |
| 40个采样噪声 | 88.2 | 更多采样覆盖更多潜在动作空间 |

### 关键发现
- 截断扩散策略同时解决了模态坍塌和计算开销两大问题：模态多样性从11%提升至70%，去噪步骤从20减至2
- 扩散解码器比UNet参数更少（60M vs 102M），但性能更好（88.1 vs 85.7 PDMS）
- 空间交叉注意力是最关键的设计，移除后PDMS从87.1骤降至55.1
- DiffusionDrive可以生成高质量的多模态轨迹（如变道、避障），这在单模态方法中不可能实现

## 亮点与洞察
- 首次将扩散模型成功应用于端到端自动驾驶的实时规划，解决了"扩散模型太慢"这一根本瓶颈
- 截断扩散策略的设计直觉非常自然：人类驾驶不是从随机出发，而是基于既有模式微调
- 仅用20个锚点就超越了8192个锚点的VADv2系列方法，说明生成式建模比离散化更高效
- 推理灵活性：训练时的锚点数和推理时的采样数解耦，可以根据算力动态调整
- 实时性能：45 FPS on 4090，远优于vanilla扩散策略的7 FPS

## 局限与展望
- 主要在非反应式仿真（non-reactive simulation）中评估，未在真实闭环驾驶中验证
- 锚点由K-Means静态聚类得到，未来可探索自适应锚点生成
- 扩散解码器目前仅与BEV特征交互（在Transfuser设置下），可以扩展到更丰富的场景表示
- 0-shot泛化能力（如新城市、新天气条件）尚未充分验证
- nuScenes上的实验表明简单场景中提升有限，方法的优势在复杂场景中更明显

## 相关工作与启发
- 与Diffusion Policy（机器人领域）的关系：本文的截断扩散策略是对vanilla扩散策略的重要改进，引入了驾驶领域的先验知识
- 与VADv2/Hydra-MDP（采样自词表范式）的对比：本文展示了连续生成式建模比离散化更高效
- TDPM（图像生成中的截断去噪）启发了截断思路，但本文引入了显式的驾驶先验（锚点），而非隐式中间分布
- 该方法的核心思想（用领域先验替代纯噪声起点）可推广到其他需要实时决策的机器人任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 截断扩散策略的设计直觉巧妙，首次将扩散模型推向实时自动驾驶
- 实验充分度: ⭐⭐⭐⭐ 在NAVSIM和nuScenes上的量化和可视化实验充分，但缺少真实部署结果
- 写作质量: ⭐⭐⭐⭐⭐ 从Transfuser到DiffusionDrive的渐进式讲述逻辑清晰，图表精美
- 价值: ⭐⭐⭐⭐⭐ 解决了扩散模型在自动驾驶中的核心瓶颈，兼具理论创新和实用价值

<!-- RELATED:START -->

## 相关论文

- [Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)
- [World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)
- [Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)

<!-- RELATED:END -->
