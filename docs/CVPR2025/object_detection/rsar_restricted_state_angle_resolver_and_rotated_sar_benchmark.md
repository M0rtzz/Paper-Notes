---
title: >-
  [论文解读] RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark
description: >-
  [CVPR 2025][目标检测][旋转目标检测] 本文从维度映射的统一视角重新审视旋转目标检测中的角度解码器，揭示现有方法忽略单位圆约束导致的预测偏差，提出 Unit Cycle Resolver（UCR），并借助 UCR 构建了目前最大的多类别旋转 SAR 目标检测数据集 RSAR。
tags:
  - CVPR 2025
  - 目标检测
  - 旋转目标检测
  - SAR图像
  - 角度边界不连续
  - 单位圆约束
  - 弱监督
---

# RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark

**会议**: CVPR 2025  
**arXiv**: [2501.04440](https://arxiv.org/abs/2501.04440)  
**代码**: https://github.com/zhasion/RSAR  
**领域**: 目标检测 / 遥感  
**关键词**: 旋转目标检测, SAR图像, 角度边界不连续, 单位圆约束, 弱监督

## 一句话总结
本文从维度映射的统一视角重新审视旋转目标检测中的角度解码器，揭示现有方法忽略单位圆约束导致的预测偏差，提出 Unit Cycle Resolver（UCR），并借助 UCR 构建了目前最大的多类别旋转 SAR 目标检测数据集 RSAR。

## 研究背景与动机

**领域现状**：旋转目标检测以 $(cx, cy, w, h, \theta)$ 表示检测框，在遥感、3D检测和场景文字检测中广泛应用。光学遥感领域已有大量进展（DOTA 等数据集推动），但 SAR（合成孔径雷达）领域的旋转目标检测进展缓慢。

**现有痛点**：(1) SAR 领域缺乏大规模旋转标注数据集，标注成本高昂且耗时；(2) 旋转目标检测的核心挑战——角度边界不连续问题——仍未完全解决。已有的角度解码方案（PSC 通过相移编码、ACM 通过复指数函数）虽然将一维角度映射到多维编码空间以解决边界不连续，但忽略了编码状态必须满足的单位圆约束。

**核心矛盾**：现有角度编码方法独立预测各维度的编码值，预测结果可能偏离单位圆，导致多对一映射——不同编码状态通过线性缩放可对应同一角度，使优化空间不必要地复杂化并引入预测偏差。

**本文目标**：(1) 提出改进的角度解码器提高角度预测精度，特别是弱监督旋转检测；(2) 利用改进方法高效构建大规模 SAR 旋转目标检测数据集。

**切入角度**：作者从维度映射的统一视角重新审视 PSC 和 ACM——一维映射存在边界不连续，二维映射对应 ACM（$\cos\theta + j\sin\theta$），三维映射对应 PSC。这个统一视角清楚揭示了"缺乏单位圆约束"这一共同缺陷。

**核心 idea**：通过添加简单的单位圆约束损失 $\mathcal{L}_{uc}$，确保预测的角度编码状态满足单位圆（二维）或椭圆（三维）约束条件，消除多对一映射带来的优化困难。

## 方法详解

### 整体框架
UCR 是一个即插即用的角度解码模块，适用于任何旋转目标检测器的角度预测头。以弱监督方法 H2RBox-v2 为例，模型从水平框标注学习预测旋转框，UCR 替换其原始角度解码器。训练时通过总损失 $\mathcal{L} = \mathcal{L}_{cls} + \lambda_{reg}\mathcal{L}_{reg} + \lambda_{uc}\mathcal{L}_{uc}$ 联合优化分类、回归和单位圆约束。

### 关键设计

1. **维度映射的统一视角（Unified Perspective of Dimensional Mapping）**:

    - 功能：将 PSC 和 ACM 两种看似不同的角度解码方法统一到同一个数学框架中，揭示共同缺陷。
    - 核心思路：角度边界不连续问题的根源是一维空间中角度范围两端的值应该相等但距离很远。解决方案是维度映射：二维映射 $m_1 = \cos\theta, m_2 = \sin\theta$（ACM）；三维映射满足 $\sum_{i=1}^3 m_i^2 = 3/2$ 和 $\sum_{i=1}^3 m_i = 0$（PSC 是一种有效解）。关键发现：无论哪种映射，编码状态都必须落在单位圆/椭圆上，但现有方法独立预测各维度值不保证这一约束。
    - 设计动机：统一视角不仅阐明了已有方法的数学本质，更重要的是暴露了"独立预测导致偏离单位圆"这一共同缺陷。

2. **单位圆约束损失（Unit Circle Constraint Loss）**:

    - 功能：约束模型预测的角度编码状态满足单位圆条件，消除多对一映射带来的优化困难。
    - 核心思路：$\mathcal{L}_{uc} = |n/2 - \sum_{i=1}^n m_i^2| + \sigma(n) |\sum_{i=1}^n m_i|$，其中 $n$ 为映射维度。二维时（$n=2$），$\sigma(2)=0$，损失简化为 $|1 - m_1^2 - m_2^2|$，即约束编码状态在单位圆上。三维时（$n=3$），$\sigma(3)=1$，额外约束编码值之和为零。
    - 设计动机：约束损失将编码状态的解空间从整个 $n$ 维空间限制到单位圆/椭圆上，消除了线性缩放导致的多对一映射，简化了优化目标，提高了角度预测精度。

3. **无效区域机制（Invalid Region Mechanism）**:

    - 功能：处理训练初期编码值随机性大的问题，提高训练稳定性。
    - 核心思路：定义无效区域 $\sum_{i=1}^n m_i^2 < m_{invalid}$，当预测的编码状态在单位圆中心附近（模很小）时，仅施加单位圆约束损失而不施加角度回归损失。因为中心附近的编码值到角度的映射不稳定，先让约束损失将预测推向单位圆附近，再进行角度回归。
    - 设计动机：训练初期编码预测值接近零（随机初始化），此时强行做角度回归会导致梯度不稳定。无效区域让模型先学会"在什么范围内预测"，再学会"预测什么值"。

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{cls} + \lambda_{reg}\mathcal{L}_{reg} + \lambda_{uc}\mathcal{L}_{uc}$。UCR 即插即用，可应用于全监督和弱监督检测器。用于 RSAR 数据集构建时，UCR 增强的 H2RBox-v2 生成旋转伪标注，再经人工校准得到最终标注。

## 实验关键数据

### 主实验
弱监督旋转检测在 RSAR、DOTA-v1.0 和 HRSC 上的表现：

| 方法 | 角度解码器 | 维度 | 监督 | RSAR AP50 | DOTA AP50 | HRSC AP50 |
|------|----------|------|------|----------|----------|----------|
| FCOS (R-50) | - | - | 旋转框 | 66.66 | 71.44 | 89.26 |
| H2RBox-v2 | ACM | 2D | 水平框 | 65.34 | 72.37 | 89.58 |
| H2RBox-v2 | PSC | 3D | 水平框 | 65.16 | 72.31 | 89.30 |
| H2RBox-v2 | **UCR** | **2D** | **水平框** | **69.21** | **73.22** | **89.73** |
| H2RBox-v2 | **UCR** | **3D** | **水平框** | **68.33** | **73.99** | **89.74** |

### 消融实验
| 配置 | RSAR mAP | DOTA mAP | 说明 |
|------|---------|---------|------|
| ACM (无约束) | 30.64 | 41.05 | 基线 |
| PSC (无约束) | 30.91 | 40.69 | 基线 |
| UCR 2D | 32.25 | 42.65 | +1.6/+1.6 |
| UCR 3D | **32.64** | **43.10** | +1.7/+2.1 |

### 关键发现
- UCR 在弱监督设置下大幅提升角度预测精度，RSAR 上 AP50 从 65.34 提升至 69.21（2D UCR），甚至超越全监督 FCOS 的 66.66
- 在 DOTA-v1.0 上，弱监督 UCR（73.99 AP50）超越全监督 FCOS（71.44 AP50），证明角度解码器设计的重要性
- 三维映射 UCR 略优于二维映射 UCR，但差异不大，二维映射计算更简单
- RSAR 包含 95,842 张图像和 183,534 个标注实例，是目前最大的多类别旋转 SAR 检测数据集

## 亮点与洞察
- **统一视角的理论贡献**：将 PSC 和 ACM 统一到维度映射框架中，这种"先统一再发现缺陷"的研究范式非常优雅，揭示了一个简单却被忽视的约束条件
- **极简但有效的改进**：仅添加一个约束损失项就能显著提升角度预测精度，改进对整体框架的侵入性极低
- **数据集构建的高效策略**：用改进的弱监督模型生成伪标注+人工校准，比从头标注效率高得多，是利用 AI 辅助标注的成功案例

## 局限与展望
- UCR 的约束是软约束（通过损失函数），无法保证预测严格落在单位圆上，可以探索硬约束（如归一化操作）
- RSAR 数据集中部分类别（如飞机）由于方向模糊被排除，数据集覆盖范围有待扩展
- 仅在旋转框检测上验证，未测试在实例分割等需要角度预测的其他任务上的效果
- 无效区域的阈值 $m_{invalid}$ 作为超参数需要调整

## 相关工作与启发
- **vs ACM**: ACM 用复指数函数编码角度（二维映射），但独立预测 cos 和 sin 不保证在单位圆上。UCR 添加约束损失解决此问题
- **vs PSC**: PSC 用相移编码（三维映射），同样忽略约束。UCR 的统一视角揭示两者本质相同且共享相同缺陷
- **vs GWD/KLD**: 基于高斯分布的方法从不同角度缓解边界问题，但未从根本上解决。UCR 直接约束编码空间，更加直接

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一视角和单位圆约束是简洁有力的贡献，但改进本身相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ 在 RSAR、DOTA、HRSC 三个数据集上验证，提供了全新的 RSAR 数据集
- 写作质量: ⭐⭐⭐⭐ 统一视角的阐述清晰，图示直观帮助理解
- 价值: ⭐⭐⭐⭐ RSAR 数据集对 SAR 旋转检测领域有重要推动作用，UCR 简洁有效可广泛使用

<!-- RELATED:START -->

## 相关论文

- [Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)
- [BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](../../NeurIPS2025/object_detection/delving_into_cascaded_instability_a_lipschitz_continuity_view_on_image_restorati.md)
- [OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](../../NeurIPS2025/object_detection/overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)
- [Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding](../../AAAI2026/object_detection/learning_procedural-aware_video_representations_through_state-grounded_hierarchy.md)

<!-- RELATED:END -->
