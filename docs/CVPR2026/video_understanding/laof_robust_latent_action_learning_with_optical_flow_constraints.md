---
title: >-
  [论文解读] LAOF: Robust Latent Action Learning with Optical Flow Constraints
description: >-
  [CVPR 2026][视频理解][潜动作学习] 提出LAOF框架，利用智能体的光流作为伪监督信号约束潜动作学习，使潜动作表示对干扰更鲁棒，在LIBERO和PROCGEN上显著超越无监督基线，且在无标签条件下匹配或超越使用1%动作标签的监督方法。
tags:
  - CVPR 2026
  - 视频理解
  - 潜动作学习
  - 光流约束
  - 具身智能
  - 模仿学习
  - 视频预训练
---

# LAOF: Robust Latent Action Learning with Optical Flow Constraints

**会议**: CVPR 2026  
**arXiv**: [2511.16407](https://arxiv.org/abs/2511.16407)  
**代码**: [GitHub](https://github.com/XizoB/LAOF)  
**领域**: 视频理解  
**关键词**: 潜动作学习, 光流约束, 具身智能, 模仿学习, 视频预训练

## 一句话总结

提出LAOF框架，利用智能体的光流作为伪监督信号约束潜动作学习，使潜动作表示对干扰更鲁棒，在LIBERO和PROCGEN上显著超越无监督基线，且在无标签条件下匹配或超越使用1%动作标签的监督方法。

## 研究背景与动机

从大规模无动作标签视频中学习潜动作表示是构建可扩展具身基础模型的关键路径。LAPO范式通过逆动力学模型（IDM）+前向动力学模型（FDM）的自编码框架联合训练潜动作，已在LAPA、GR00T N1等大规模具身模型中得到应用。

核心问题：LAPO隐含假设连续帧间的所有变化都由智能体的动作引起，但真实世界视频中存在大量**与动作无关的干扰**（如移动的背景物体、随机环境变化），且纯重建目标可能使潜动作与视觉外观纠缠。

现有解决方案：
- 添加少量动作标签监督（LAOM、villa-X）：在极端标签稀缺时交替训练不稳定，容易过拟合
- 离散化VQ-VAE：创建信息瓶颈但表达力受限

核心洞察：**光流提供了像素级的帧间运动信息**，天然抑制静态背景并强调运动物体，且预训练光流模型已有强跨场景泛化能力。光流可作为与动作高度相关的伪监督信号，无需人工标注。

## 方法详解

### 整体框架

三阶段训练pipeline：
1. **预训练**：在无标签视频上联合训练IDM+FDM+光流解码器
2. **蒸馏**：将IDM知识蒸馏到只接受当前帧的潜动作策略
3. **微调**：用少量动作标签训练动作解码器（潜动作→物理动作）

### 关键设计

1. **光流伪监督约束**:
    - 功能：通过光流解码器约束潜动作捕获真实物理运动
    - 核心思路：增加专用光流解码器 $d_{flow}: \mathcal{Z} \rightarrow \mathcal{F}_{rgb}$，直接将潜动作映射为光流特征。光流伪标签由预训练RAFT模型生成，转换为RGB格式后通过DINOv2编码。预训练损失：$\mathcal{L}_{pretrain} = \mathcal{L}_{reconstruction} + \mathcal{L}_{flow}$
    - 设计动机：光流与动作高度相关（moving object = action outcome），作为辅助解码目标可以约束潜动作空间的物理一致性，避免潜动作退化为外观编码

2. **RGB格式光流处理**:
    - 功能：使光流兼容DINOv2视觉编码器
    - 核心思路：光流向量(u,v)转换为极坐标→方向映射HSV色相、幅度映射饱和度和亮度→标准HSV→RGB转换。幅度归一化：$m_{norm} = \min(1.0, m/(\sigma\sqrt{H^2+W^2}))$
    - 设计动机：统一使用DINOv2处理观测和光流，避免额外编码器

3. **物体中心光流（Object-Centric）**:
    - 功能：在有动态干扰的场景中提取智能体相关光流
    - 核心思路：对静态背景场景（如机器人操作），全局光流已经自然关注智能体运动。对动态干扰场景（如游戏），使用LangSAM生成物体遮罩，过滤无关运动：$f_{rgb,t}^{sam} = mask_t \odot f_{rgb,t}^{all}$
    - 设计动机：不同场景自适应选择全局vs物体中心光流，兼顾通用性

### 损失函数 / 训练策略

- 纯LAOF：$\mathcal{L}_{pretrain} = \mathcal{L}_{reconstruction} + \mathcal{L}_{flow}$
- LAOF-Action（含少量标签）：$\mathcal{L}_{pretrain} = \mathcal{L}_{reconstruction} + (1-\lambda)\mathcal{L}_{flow} + \lambda\mathcal{L}_{action}$，$\lambda = M/(N+M)$
- 蒸馏：$\mathcal{L}_{distillation} = \|\pi(\hat{z}_t|s_t,l_t) - z_t\|_2$
- 微调：$\mathcal{L}_{action} = \|d_{action}(\hat{a}_t|z_t) - a_t\|_2$

## 实验关键数据

### 主实验 — LIBERO模仿学习

| 方法 | SPATIAL成功率 | OBJECT成功率 | GOAL成功率 | LONG成功率 | 平均提升 |
|------|-------------|-------------|-----------|-----------|---------|
| LAPO | 80.4% | 81.2% | 84.0% | 44.7% | 基线 |
| CoMo | 74.1% | 87.6% | 80.8% | 49.9% | +0.5 |
| CoMo w/ OF | 76.2% | **89.7%** | 82.6% | **57.9%** | +4.0 |
| **LAOF** | **82.5%** | 85.3% | **87.2%** | 52.0% | **+4.2** |
| LAOM-Action (1%标签) | 86.0% | 91.1% | 86.3% | 61.6% | +8.7 |
| **LAOF-Action (1%标签)** | **88.2%** | **95.9%** | **88.6%** | **63.7%** | **+11.5** |

### 消融实验 — 光流约束位置

| 配置 | 效果 | 说明 |
|------|------|------|
| 直接连接到潜动作 | 最优 | 光流解码器直接从z解码 |
| 通过FDM去约束 | 次优 | 间接约束效力减弱 |
| 无光流约束 | 基线 | LAPO原始方法 |

### 标签比例扩展实验

| 动作标签比例 | LAOF-Action vs LAOM-Action |
|-------------|---------------------------|
| 0% | LAOF ≥ LAOM-Action@1% |
| 1% | LAOF-Action 显著超越 |
| 5% | 仍有提升 |
| 10% | 光流约束仍有效 |

### 关键发现

- 无监督LAOF匹配甚至超越使用1%标签的LAOM-Action，证明光流伪监督的强效性
- 光流约束在标签比例增加到10%时仍然有效，说明两种信号互补而非冗余
- 连续潜动作一致优于离散VQ-VAE表示（两个benchmark均有验证）
- 提出的潜动作评估指标与下游任务性能高度相关（Pearson相关系数0.83/0.73）

## 亮点与洞察

- 光流作为伪监督信号的选择既自然又有效——像素级运动捕获是动作的直接视觉结果
- RGB格式光流统一了观测和运动的处理流程，仅需单一视觉编码器
- LAOF-Action的自适应权重设计（$\lambda=M/(N+M)$）随标签比例自动平衡两种信号
- 作为LAPO范式的扩展，可直接集成到现有具身基础模型训练流程中

## 局限与展望

- 依赖预训练光流模型（RAFT），光流估计错误会传播为噪声标签
- 物体中心光流依赖LangSAM的分割质量，复杂场景可能失效
- 仅在LIBERO（机器人操作）和PROCGEN（2D游戏）上验证，真实世界场景未测试
- 三阶段训练pipeline增加了工程复杂度

## 相关工作与启发

- **vs LAPO**: LAPO隐含静态背景假设，LAOF通过光流显式处理动态干扰
- **vs LAOM**: LAOM需要动作标签且交替训练不稳定；LAOF用无标签光流获得更稳定的训练
- **vs FlowVLA (并发)**: FlowVLA将光流离散化为token用于世界模型训练；LAOF使用连续光流约束学习潜动作，目标不同

## 评分

- 新颖性: ⭐⭐⭐⭐ 光流作为潜动作伪监督的idea自然且有效，但核心贡献是实验验证而非概念突破
- 实验充分度: ⭐⭐⭐⭐⭐ LIBERO+PROCGEN，连续vs离散，标签比例扫描，消融详细
- 写作质量: ⭐⭐⭐⭐ 方法清晰，问题定义精确，三阶段流程条理分明
- 价值: ⭐⭐⭐⭐ 对具身基础模型预训练有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [BAT: Learning Event-based Optical Flow with Bidirectional Adaptive Temporal Correlation](../../AAAI2026/video_understanding/bat_learning_event-based_optical_flow_with_bidirectional_adaptive_temporal_corre.md)
- [Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](../../ICCV2025/video_understanding/unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](../../ICCV2025/video_understanding/prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)
- [Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)

<!-- RELATED:END -->
