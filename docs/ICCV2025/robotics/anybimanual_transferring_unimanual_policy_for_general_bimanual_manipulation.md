---
title: >-
  [论文解读] AnyBimanual: Transferring Unimanual Policy for General Bimanual Manipulation
description: >-
  [ICCV 2025][机器人][双臂操控] 提出 AnyBimanual，一个即插即用的框架，通过技能管理器和视觉对齐器将预训练的单臂操控策略迁移到通用双臂操控场景，在仅有少量双臂示范的情况下实现显著的多任务泛化能力。
tags:
  - ICCV 2025
  - 机器人
  - 双臂操控
  - 策略迁移
  - 技能原语
  - 视觉对齐
  - 行为克隆
---

# AnyBimanual: Transferring Unimanual Policy for General Bimanual Manipulation

**会议**: ICCV 2025  
**arXiv**: [2412.06779](https://arxiv.org/abs/2412.06779)  
**代码**: https://anybimanual.github.io/  
**领域**: 机器人操控  
**关键词**: 双臂操控, 策略迁移, 技能原语, 视觉对齐, 行为克隆

## 一句话总结

提出 AnyBimanual，一个即插即用的框架，通过技能管理器和视觉对齐器将预训练的单臂操控策略迁移到通用双臂操控场景，在仅有少量双臂示范的情况下实现显著的多任务泛化能力。

## 研究背景与动机

双臂操控系统在家庭服务、机器人手术、工业装配等场景中扮演重要角色。相比单臂系统，双臂系统工作空间更大、能完成更复杂的任务（如一只手稳定目标，另一只手操作），但面临关键瓶颈：

**数据昂贵**：双臂操控的动作空间维度极高，遥操作示范数据采集需要专用系统、额外传感器和精密校准，人力成本极高

**泛化困难**：受限于数据量，直接学习的双臂策略难以泛化到多种任务

**现有方案的局限**：
   - 基于 LLM/VLM 的高层规划方法：受制于预定义的低层执行器，无法处理接触密集型任务
   - 固定角色分配（stabilizer/actor）：协作模式不灵活
   - 参数化原子动作：难以由人工指定，限制了部署场景

**关键洞察**：单臂策略（如 PerAct、RVT）已通过大规模参数和训练数据展现出令人印象深刻的跨任务泛化能力。双臂任务往往由多个单臂子任务组合而成，因此可以挖掘和迁移单臂策略中的通用操控知识。

## 方法详解

### 整体框架

AnyBimanual 由两个核心模块组成：技能管理器（Skill Manager）处理语言分支，视觉对齐器（Visual Aligner）处理视觉分支。两个预训练的单臂策略模型分别预测左右臂动作。

### 关键设计

1. **技能管理器（Skill Manager）**

    - 维护一个离散的技能原语集合 $\mathcal{Z} = \{z_1, z_2, ..., z_K\}$，每个技能 $z_k \in \mathbb{R}^D$ 是隐式嵌入
    - 将双臂的语言嵌入表达为技能原语的线性组合加上任务导向补偿：
    $\hat{l}_t^{left} = \sum_{k=1}^K \hat{w}_{k,t}^{left} z_k + \epsilon_t^{left}, \quad \hat{l}_t^{right} = \sum_{k=1}^K \hat{w}_{k,t}^{right} z_k + \epsilon_t^{right}$
    - 使用多模态 Transformer 动态预测每步的组合权重：$(\\hat{w}_t^{left}, \epsilon_t^{left}, \hat{w}_t^{right}, \epsilon_t^{right}) = f_\theta(v_t, l, p_t)$
    - 技能原语可用预训练单臂策略的语言模板 token 初始化，缓解域差距
    - 设计动机：如双臂交接（Handover）任务，可分解为左臂"放置"、右臂"拾取"两个单臂技能

2. **可泛化技能表征学习**

    - 通过稀疏正则化鼓励用最少的技能原语重构语言嵌入：
    $\mathcal{L}_{skill} = \|\hat{w}^{left}\|_1 + \|\hat{w}^{right}\|_1 + \lambda_\epsilon(\|\epsilon^{left}\|_{2,1} + \|\epsilon^{right}\|_{2,1})$
    - $\ell_1$ 范数强制稀疏选择，使每个技能表征捕获独立的基础运动
    - 补偿项的 $\ell_{2,1}$ 范数确保仅在必要时引入体现特定知识

3. **视觉对齐器（Visual Aligner）**

    - 生成两个空间软掩码来分解体素空间，使每个臂的视觉输入对齐到其预训练阶段的分布：
    $v_t^{left} = (\hat{v}_t^{left} \odot v_t) \oplus v_t, \quad v_t^{right} = (\hat{v}_t^{right} \odot v_t) \oplus v_t$
    - 通过最大化 Jensen-Shannon 散度实现互斥分割：
    $\mathcal{L}_{voxel} = -D_{KL}(\hat{v}_t^{left}\|\hat{v}_t^{right})/2 - D_{KL}(\hat{v}_t^{right}\|\hat{v}_t^{left})/2$
    - 直觉：双臂异步协作时，左右臂关注工作空间的不同区域，互斥分割自然地将双臂场景还原为单臂配置

### 总体训练目标

$$\mathcal{L}_{total} = \mathcal{L}_{BC} + \lambda_{skill}\mathcal{L}_{skill} + \lambda_{voxel}\mathcal{L}_{voxel}$$

其中 $\mathcal{L}_{BC}$ 是行为克隆的交叉熵损失。该框架是模型无关的，支持 Transformer 策略和扩散策略等不同架构。

## 实验关键数据

### 主实验（RLBench2, 100 demonstrations）

| 方法 | straighten rope | sweep dustpan | press buttons | put in fridge | 平均 |
|------|----------------|---------------|---------------|---------------|------|
| PerAct2 | 8 | 52 | 41 | 16 | 14.67 |
| PerAct2+Pretrain | 17 | 55 | 40 | 22 | - |
| PerAct-LF | 11 | 47 | 9 | 7 | - |
| **PerAct+AnyBimanual** | **24** | **57** | **14** | **26** | **32.00** |

总体平均成功率：AnyBimanual (32.00%) vs PerAct2 (14.67%)，提升 **17.33%**。

### 消融实验

| Row | Skill Manager | Visual Aligner | Long | Generalized | Sync | Average |
|-----|---------------|----------------|------|-------------|------|---------|
| 1 | - | - | 16.29 | 23.50 | 3.50 | 14.67 |
| 2 | ✗ | ✗ | 19.57 | 25.50 | 9.50 | 16.75 |
| 3 | ✗ | ✓ | 21.57 | 44.00 | 15.50 | 19.75 |
| 4 | ✓ | ✗ | 23.71 | 42.00 | 17.00 | 25.67 |
| 5 | ✓ | ✓ | 27.29 | 44.00 | 25.00 | **32.00** |

### 关键发现

- AnyBimanual 将 PerAct2 SOTA 提升了 17.33%，在长时序、多变体和同步协调任务上优势尤为明显
- 作为即插即用方法，也能提升 PerAct-LF（+72.76%）和 RVT-LF（+39.41%）的性能
- 技能管理器对长时序任务提升最大（+8.92%），视觉对齐器对泛化和同步任务贡献显著
- 两个模块协同效果 > 各自之和：从 14.67% 到 32.00%
- 真实世界 9 个双臂任务平均成功率 84.62%，展现强实用性
- 可视化显示技能管理器能动态调度合理的技能组合，视觉对齐器有效分解体素空间

## 亮点与洞察

- **框架设计优雅**：核心 idea 简洁——双臂 = 两个单臂的协调组合，通过稀疏分解和互斥掩码实现
- **模型无关性**：支持 PerAct、RVT 等不同单臂策略，实用性强
- **数据效率**：仅需 20-100 个双臂示范即可迁移，大幅降低数据需求
- **可解释性好**：技能调度权重和体素分解mask均可可视化，便于理解模型行为

## 局限与展望

- 对需要精确旋转的简单任务（如 Rotate Toothbrush）表现不佳（20%成功率）
- 在短时序简单任务（如 Lift ball）上引入额外复杂性可能略降性能
- 技能原语数 $K$ 需要手动设定
- 当前仅在 RLBench2 和受限真实场景测试，未在更大规模基准上验证
- 依赖关键帧提取质量

## 相关工作与启发

- 与 PerAct2 的固定架构设计形成对比，展示了"迁移而非重新训练"的范式优势
- 稀疏技能分解的思想可推广到其他多agent协作场景
- 互斥视觉掩码的设计灵感来自双臂异步协作的观察，可扩展到多臂系统

## 评分
- 新颖性: ⭐⭐⭐⭐ 将单臂迁移双臂的框架设计新颖，技能调度和视觉对齐思路清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 12仿真+9真实任务，多基线对比，消融完整，可视化丰富
- 写作质量: ⭐⭐⭐⭐ 问题动机明确，方法描述系统化
- 价值: ⭐⭐⭐⭐⭐ 高度实用的即插即用方案，对机器人操控研究有重要推动

<!-- RELATED:START -->

## 相关论文

- [MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](../../CVPR2025/robotics/momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)
- [iManip: Skill-Incremental Learning for Robotic Manipulation](imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation](../../CVPR2025/robotics/lift3d_policy_lifting_2d_foundation_models_for_robust_3d_robotic_manipulation.md)
- [Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding](adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)

<!-- RELATED:END -->
