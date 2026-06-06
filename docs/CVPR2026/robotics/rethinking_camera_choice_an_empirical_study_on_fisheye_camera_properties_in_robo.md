---
title: >-
  [论文解读] Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation
description: >-
  [CVPR 2026][机器人][鱼眼相机] 首次系统性地对腕部鱼眼相机在机器人操作模仿学习中的特性进行实证研究，围绕空间定位、场景泛化和硬件泛化三个核心问题揭示了宽视场角的优势与局限，并提出 Random Scale Augmentation (RSA) 策略解决跨相机迁移中的尺度过拟合问题。
tags:
  - "CVPR 2026"
  - "机器人"
  - "鱼眼相机"
  - "机器人操作"
  - "模仿学习"
  - "视场角"
  - "泛化性"
---

# Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation

**会议**: CVPR 2026  
**arXiv**: [2603.02139](https://arxiv.org/abs/2603.02139)  
**作者**: Han Xue, Min Nan, Xiaotong Liu, Wendi Chen, Yuan Fang, Jun Lv, Cewu Lu, Chuan Wen (上海交通大学, 东南大学, 中科大等)
**项目页**: [robo-fisheye.github.io](https://robo-fisheye.github.io/)
**领域**:机器人
**关键词**: 鱼眼相机, 机器人操作, 模仿学习, 视场角, 泛化性

## 一句话总结

首次系统性地对腕部鱼眼相机在机器人操作模仿学习中的特性进行实证研究，围绕空间定位、场景泛化和硬件泛化三个核心问题揭示了宽视场角的优势与局限，并提出 Random Scale Augmentation (RSA) 策略解决跨相机迁移中的尺度过拟合问题。

## 研究背景与动机

鱼眼相机凭借超广视场角（FoV > 180°）在机器人操作中的应用快速增长，但学术界对鱼眼相机如何影响策略学习的理解远远落后于其实际部署。

**现有问题**：
- 鱼眼相机引入的强径向畸变对视觉编码器的影响尚不明确
- 宽 FoV 在不同场景复杂度下的实际收益缺乏量化分析
- 不同鱼眼镜头之间的策略迁移（硬件泛化）存在系统性失败，但根因不清
- 缺乏涵盖仿真和真实世界的系统性基准来指导鱼眼数据集的大规模采集

**核心动机**：建立首个系统性实证研究框架，回答三个关键研究问题：

**RQ1 - 空间定位**：宽 FoV 能否增强策略的空间定位能力？

**RQ2 - 场景泛化**：鱼眼相机如何影响对新背景的泛化？

**RQ3 - 硬件泛化**：策略能否在不同鱼眼镜头之间迁移？

## 方法详解

### 实验平台
- **真实世界**：Flexiv Rizon 4 七轴机械臂 + DH AG-160-95 自适应夹爪，通过 Meta Quest 3 头显遥操作采集示教数据
- **仿真环境**：在 MuJoCo 中实现两阶段投影管线（先渲染全景图再投影为鱼眼视图），可精确控制镜头参数
- **任务设计**：3 个真实世界任务（Pick Cup、Fold Towel、Hang Chinese Knot）+ 改编 Robomimic/MimicGen 的 6 个仿真任务
- **评估指标**：采用多阶段归一化评分（normalized score），比二元成功率更细粒度

### RQ1：空间定位分析
**假设**：鱼眼的宽 FoV 通过捕获更多静态环境特征作为视觉锚点来增强策略定位，因此性能应强烈依赖于场景视觉复杂度。

**实验设计**：对比特征贫乏（纯色背景）vs 特征丰富（纹理布/杂物）环境下的策略表现，使用无状态（state-free）策略以隔离视觉定位能力。

**关键发现**：鱼眼 + 丰富场景组合使策略可以仅依赖视觉输入完成高精度操作，隐式编码了机器人与环境的空间关系，使显式本体感觉状态（proprioception）变得冗余。

### RQ2：场景泛化的 Scaling Law
**假设**：鱼眼策略能更有效地利用场景多样性，随训练场景数 $N$ 增加表现出更陡峭的性能提升曲线。

**实验设计**：固定总数据量（如 200 条轨迹），仅增加独立训练场景数 $N$（从 1 到 8），在完全未见过的测试背景上零样本评估。使用 32 种不同背景纹理。

**关键发现**：鱼眼相机的宽 FoV 相当于隐式数据增强，使策略更好地利用场景多样性。真实世界中，鱼眼策略仅需 8 个多样训练场景即可达到近乎完美的分数（0.988）。

### RQ3：硬件泛化与 RSA
**问题根因**：策略对特定镜头的绝对像素尺度产生过拟合（Scale Overfitting）。部署到新镜头时，物体在图像中的尺度变化导致策略误判深度——放大导致"undershoot"（感知更近），缩小导致"overshoot"（感知更远）。

**Random Scale Augmentation (RSA)**：
- 训练时从均匀分布随机采样缩放因子 $s$（如 0.7~1.3）
- $s > 1$ 时缩小图像并填充黑边（zoom-out 效果）
- 迫使网络学习相对空间关系（如目标相对于夹爪的尺度）而非绝对像素大小
- 简单、即插即用，无需修改网络架构

## 实验关键数据

### Table 1: RQ1 - 真实世界空间定位（归一化分数，无状态策略）

| 任务 | 相机类型 | 贫乏场景 | 丰富场景 | 增益 |
|---|---|---|---|---|
| Pick Cup | Fisheye (State-free) | 0.525 | 0.800 | **+0.275** |
| Fold Towel | Fisheye (State-free) | 0.100 | 0.700 | **+0.600** |
| Hang Chinese Knot | Fisheye (State-free) | 0.200 | 0.500 | **+0.300** |

鱼眼 + 丰富场景在所有任务上均大幅超越贫乏场景，Fold Towel 增益最大达 **+0.600**。仿真中鱼眼在丰富场景的 SR 为 0.66，较 Pinhole 的 0.34 提升 **+0.32**。

### Table 2: RQ3 - RSA 尺度敏感性分析（仿真，归一化分数）

| 缩放因子 $S$ | 效果 | Baseline | RSA |
|---|---|---|---|
| 0.70 | 强烈放大 | 0.000 | **0.900** |
| 0.85 | 适度放大 | 0.950 | **1.000** |
| 1.00 | 训练尺度 | 1.000 | 1.000 |
| 1.15 | 适度缩小 | 0.750 | **0.975** |
| 1.30 | 强烈缩小 | 0.650 | **1.000** |

Baseline 在尺度偏移时呈"倒 V 形"急剧衰减（$S=0.70$ 时直降为 0），RSA 在全尺度范围保持 0.9+ 稳健表现。

### 补充数据

**RQ2 场景泛化（真实世界 Pick Cup）**：

| 训练场景数 $N$ | Pinhole | Fisheye |
|---|---|---|
| 1 | 0.081 | 0.556 |
| 4 | 0.238 | 0.869 |
| 8 | 0.181 | **0.988** |

Fisheye 的 scaling 曲线远陡于 Pinhole，$N=8$ 时近乎完美；Pinhole 在 $N=8$ 时反而下降。

**RQ3 真实硬件跨相机迁移**：

| 镜头 | FoV | 尺度变化 | Baseline | RSA |
|---|---|---|---|---|
| 训练镜头 | 180° | 1.0x | 1.000 | 1.000 |
| 窄镜头 | 150° | ~1.2x (放大) | 0.500 | **0.950** |
| 广角镜头 | 220° | ~0.8x (缩小) | 0.003 | **0.600** |

Baseline 在广角镜头上几乎完全失败（0.003），RSA 将其提升至 0.600。

## 亮点与洞察

- **首个系统性实证研究**：填补了鱼眼相机在机器人操作策略学习中缺乏系统分析的空白，三个研究问题层层递进，结论具有实操指导意义
- **场景复杂度的关键作用**：揭示了"鱼眼有用"的前提条件——必须在视觉特征丰富的环境中才能充分发挥宽 FoV 的定位优势，纯色背景下改善有限
- **隐式数据增强效应**：鱼眼相机的宽 FoV 在腕部移动时天然引入更大的视角变化，等价于场景级数据增强，这是其泛化优势的根本来源
- **Scale Overfitting 的诊断与修复**：精准定位跨相机迁移失败的根因为尺度过拟合，提出的 RSA 策略极其简洁（仅需随机缩放+黑边填充），但效果显著
- **实用指南**：为大规模鱼眼数据集采集提供了三条明确建议——在丰富环境采集、最大化场景多样性、使用 RSA 训练

## 局限性

- **仅限腕部视角**：所有实验基于 wrist-mounted 鱼眼相机，未探索第三人称视角或多视角融合的场景
- **任务范围有限**：3 个真实世界任务 + 6 个仿真任务，未涵盖灵巧操作、长时序或高精度装配等更复杂场景
- **RSA 的局限**：广角镜头（220°）迁移仅达 0.600，距完美仍有差距；极端焦距变化下仿真表现仅 0.06，说明 RSA 不能完全解决所有硬件差异
- **未考虑畸变矫正**：未探索先做几何矫正再训练策略的方案，这可能是更直接的跨相机迁移路径
- **仅评估模仿学习**：未涉及强化学习或在线适应方案，RSA 在 RL 范式下的效果未知

## 相关工作

- **鱼眼相机在机器人中的应用**：FisheyeStereoNet（鱼眼深度估计）、BiFuse/OmniFusion（全景深度）→ 聚焦感知层面，缺乏对策略学习的系统分析
- **机器人操作中的相机选择**：UMI/ALOHA（腕部相机方案）、RoVi-Aug/MimicGen（视觉增强）→ 均使用针孔相机，未考虑 FoV 的影响
- **域适应与泛化**：Domain Randomization、Random Crop Augmentation → RSA 可视为尺度维度的域随机化，但针对性更强
- **本文定位**：首次从策略学习角度系统研究相机模型选择的影响，填补了"相机→感知→策略"链路中相机选择环节的研究空白

## 评分

- 新颖性: ⭐⭐⭐⭐ — 作为实证研究本身方法创新有限，但研究问题的提出和 RSA 的发现具有实际价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 仿真+真实世界双轨验证，6+3 任务，消融实验设计严谨，控制变量清晰
- 写作质量: ⭐⭐⭐⭐ — 三个RQ结构清晰，假设-验证-结论的组织方式易于跟随
- 价值: ⭐⭐⭐⭐ — 为鱼眼数据集大规模采集提供了可直接执行的指南，RSA 策略简洁实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[ICLR 2026\] When would Vision-Proprioception Policies Fail in Robotic Manipulation?](../../ICLR2026/robotics/when_would_vision-proprioception_policies_fail_in_robotic_manipulation.md)
- [\[CVPR 2026\] CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding](cyclemanip_enabling_cyclic_task_manipulation_via_effective_historical_percepti.md)

</div>

<!-- RELATED:END -->
