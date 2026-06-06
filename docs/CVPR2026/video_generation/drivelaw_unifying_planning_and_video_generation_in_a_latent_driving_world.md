---
title: >-
  [论文解读] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World
description: >-
  [CVPR 2026][视频生成][世界模型] 提出 DriveLaW，一个通过共享潜在空间将视频生成与运动规划统一的驾驶世界模型，将视频生成器的中间潜在特征直接注入扩散规划器，在 nuScenes 视频预测和 NAVSIM 规划基准上同时达到 SOTA。
tags:
  - "CVPR 2026"
  - "视频生成"
  - "世界模型"
  - "自动驾驶规划"
  - "潜在空间"
  - "扩散策略"
---

# DriveLaW: Unifying Planning and Video Generation in a Latent Driving World

**会议**: CVPR 2026  
**arXiv**: [2512.23421](https://arxiv.org/abs/2512.23421)  
**代码**: [https://github.com/xiaomi-research/drivelaw](https://github.com/xiaomi-research/drivelaw)  
**领域**: 视频生成  
**关键词**: 世界模型, 自动驾驶规划, 视频生成, 潜在空间, 扩散策略

## 一句话总结

提出 DriveLaW，一个通过共享潜在空间将视频生成与运动规划统一的驾驶世界模型，将视频生成器的中间潜在特征直接注入扩散规划器，在 nuScenes 视频预测和 NAVSIM 规划基准上同时达到 SOTA。

## 研究背景与动机

世界模型通过学习驾驶场景的时序演化来应对真实世界的长尾挑战，但当前方法将世界模型的角色限制在三个间接层面：(1) 数据生成器——合成稀有场景数据或作为闭环仿真环境；(2) 监督信号——预测未来视觉/可达性信号来监督规划；(3) 并行生成——在统一架构中共同生成视频和轨迹但仍是解耦过程。

核心矛盾：**即使在"统一"架构中，视频生成器和规划器仍作为独立模块运行**——Epona 和 DriveVLA-W0 分别训练视频生成和策略头，未利用生成器内部潜在表示作为规划状态。视频生成器虽然从大规模数据中学到了丰富的场景语义、物体动力学和物理规律，但这些知识被"浪费"在渲染上而未传导给规划器。

核心洞察：**视频生成器的内部激活编码了丰富的、时序连贯的场景理解——这正是规划所需的表示。** DriveLaW 将生成器从"渲染器"重新定位为"特征提取器"，将其去噪后的潜在特征直接作为规划器的条件输入。

## 方法详解

### 整体框架

DriveLaW 由两个核心组件链式连接：(1) DriveLaW-Video——时空视频生成器，包含时空 VAE 和 Video DiT（扩散 Transformer），输入历史观测和动作，输出去噪后的视频潜在特征；(2) DriveLaW-Act——轻量级 Action DiT 扩散规划器，以视频潜在特征为条件，通过 flow matching 生成未来轨迹。两者通过三阶段渐进训练策略优化。

### 关键设计

1. **链式生成-规划架构（Chained Design）**:

    - 功能：将视频生成器的表示直接传导给规划器
    - 核心思路：不同于并行设计（视频和轨迹各自独立输出），DriveLaW 将 Video DiT 去噪后的潜在特征 $z$ 直接注入 Action DiT 作为条件。这些潜在特征从大规模视频预训练中学到了场景语义、智能体动力学和物理规律的紧凑表示。Action DiT 以标准 DiT 架构实现，用 flow matching 目标训练
    - 设计动机：相比并行设计，链式设计有三个优势：(a) 充分利用大规模视频预训练学到的表示；(b) 训练时避免视频生成和规划之间的梯度干扰；(c) 级联确保生成的视觉细节和规划轨迹之间的一致性

2. **噪声重注入机制（Noise Reinjection）**:

    - 功能：平衡激进压缩与视觉保真度
    - 核心思路：在时空 VAE 的高压缩比下，去噪早期阶段可能产生结构不一致和模糊（尤其高速场景）。噪声重注入在去噪早期探索并选择最优生成路径——对中间去噪结果重新注入受控噪声，让模型重新探索替代路径
    - 设计动机：高保真视频合成和实时稳定规划存在内在张力。高压缩 VAE 对规划效率有利但损害视觉质量，噪声重注入是两者之间的调节器

3. **三阶段渐进训练策略**:

    - 功能：协调视频生成和规划的优化
    - 核心思路：(a) 第一阶段——学习长时运动：训练 Video DiT 生成粗粒度视频，建立时序动力学理解；(b) 第二阶段——精炼空间细节：在更高分辨率或更精细的去噪步骤下微调视频质量；(c) 第三阶段——链式规划：冻结 Video DiT，将其潜在特征链接到 Action DiT，训练规划器
    - 设计动机：直接端到端训练会导致视频生成和规划目标冲突。渐进策略让每个组件在其最佳学习窗口内优化

### 损失函数 / 训练策略

Video DiT 使用标准扩散损失（去噪目标），Action DiT 使用 flow matching 目标生成轨迹。三阶段训练中第三阶段冻结 Video DiT 参数，仅训练 Action DiT。

## 实验关键数据

### 主实验

**nuScenes 视频生成**

| 方法 | FID↓ | FVD↓ | 说明 |
|------|------|------|------|
| 之前 SOTA | 基线 | 基线 | 各类世界模型和视频生成器 |
| **DriveLaW-Video** | **-33.3%** | **-1.8%** | 大幅领先 |

**NAVSIM 规划基准（PDMS）**

| 方法 | PDMS | 说明 |
|------|------|------|
| 之前 SOTA（世界模型方法） | 基线 | 各种世界模型+规划方法 |
| **DriveLaW-Act** | **新纪录** | 无需后训练(RL)或后处理(scorers) |

### 消融实验

| 配置 | FID | PDMS | 说明 |
|------|-----|------|------|
| 仅 BEV 特征→规划 | 更高 | 更低 | 传统 BEV 表示 |
| 仅 VLM 特征→规划 | 中等 | 中等 | 视觉-语言模型特征 |
| **视频潜在特征→规划** | **最低** | **最高** | 视频生成器的表示最优 |
| 并行设计 | 中等 | 中等 | 生成和规划独立输出 |
| **链式设计** | **最低** | **最高** | 潜在特征传导给规划器 |

### 关键发现

- 视频生成器的潜在表示优于 BEV 和 VLM 特征作为规划输入——证明从大规模视频预训练中学到的表示有独特价值
- 链式设计相比并行设计在两个任务上都更优，验证了表示传导优于独立输出
- 噪声重注入机制在高速场景下显著减少模糊和结构不一致
- 无需 RL 后训练或评分器后处理即达到 NAVSIM SOTA，说明视频先验已足够强

## 亮点与洞察

- **"视频生成器即特征提取器"** 是深刻的范式转换：将生成模型从端输出器重新定位为中间表示提供者，跨越了"生成"和"理解"的边界
- **链式 vs 并行**的对比令人信服：即使在"统一"架构中，信息流的方向和耦合方式至关重要
- **三阶段训练**巧妙避免了多目标冲突——先分别优化再级联微调的策略具有通用性

## 局限与展望

- 链式设计意味着规划延迟受视频生成速度限制，实时性可能不足
- 当前仅单视图视频生成，多视图一致性未涉及
- 仅在 nuScenes 和 NAVSIM 上验证，真实闭环驾驶部署的鲁棒性待测试
- 视频生成器的错误会直接传播到规划器（误差级联）

## 相关工作与启发

- **vs Epona**: 并行生成视频和轨迹，解耦设计未利用生成器内部表示
- **vs DriveVLA-W0**: 也用混合 Transformer 生成两种模态，但仍是并行输出流
- **vs DiffusionDrive**: 纯扩散规划，无视频生成的世界理解先验

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将视频生成器的中间潜在表示作为规划状态，链式设计有原创性
- 实验充分度: ⭐⭐⭐⭐⭐ 双任务 SOTA + 表示对比消融 + 架构设计消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但三阶段训练细节可更详尽
- 价值: ⭐⭐⭐⭐⭐ 为自动驾驶世界模型提供了新范式，来自小米 EV 有实际应用背景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[ICML 2026\] OLAF-World: Orienting Latent Actions for Video World Modeling](../../ICML2026/video_generation/olaf-world_orienting_latent_actions_for_video_world_modeling.md)
- [\[CVPR 2026\] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)
- [\[CVPR 2026\] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)
- [\[CVPR 2025\] World2Act: Latent Action Post-Training via Skill-Compositional World Models](../../CVPR2025/video_generation/world2act_latent_action_post-training_via_skill-compositional_world_models.md)

</div>

<!-- RELATED:END -->
