---
title: >-
  [论文解读] A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning
description: >-
  [CVPR 2025][机器人][预训练视觉模型] 通过系统评估发现DINO/iBOT在机器人任务上优于MAE但在非物体中心(NOC)数据上性能退化，原因是丧失了物体中心表示能力。提出SlotMIM方法，通过语义瓶颈（减少原型数量促进objectness涌现）和跨视图一致性正则+slot级对比学习，使模型在NOC数据上也能学到物体中心表示，仅用241K样本即超越用>1M样本的MVP/VC-1。
tags:
  - CVPR 2025
  - 机器人
  - 预训练视觉模型
  - 机器人学习
  - 物体中心表示
  - 自监督学习
  - 数据类型
---

# A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning

**会议**: CVPR 2025  
**arXiv**: [2503.06960](https://arxiv.org/abs/2503.06960)  
**代码**: https://github.com/CVMI-Lab/SlotMIM  
**领域**: 机器人学 / 视觉预训练  
**关键词**: 预训练视觉模型, 机器人学习, 物体中心表示, 自监督学习, 数据类型

## 一句话总结

通过系统评估发现DINO/iBOT在机器人任务上优于MAE但在非物体中心(NOC)数据上性能退化，原因是丧失了物体中心表示能力。提出SlotMIM方法，通过语义瓶颈（减少原型数量促进objectness涌现）和跨视图一致性正则+slot级对比学习，使模型在NOC数据上也能学到物体中心表示，仅用241K样本即超越用>1M样本的MVP/VC-1。

## 研究背景与动机

**领域现状**：预训练视觉模型(PVM)已成为机器人学习的基础构建块。主流做法（MVP、VC-1）是MAE+自中心视角(ego-centric)数据预训练。

**现有痛点**：两个未被质疑的假设——(1) MAE是最优预训练方法；(2) ego-centric数据是最佳选择。实验发现均不成立：DINO/iBOT在操作和感知任务上全面优于MAE；ImageNet（物体中心）数据效果优于ego-centric数据。

**核心矛盾**：DINO/iBOT虽在物体中心数据上表现最优，但在场景中心/ego-centric等NOC数据上严重退化。原因：它们难以从NOC数据中学到物体中心表示——而objectness与操作任务成功率高度相关（相关系数0.72）。

**核心idea**：设计SlotMIM——通过减少原型数量（语义瓶颈）促进objectness涌现，加跨视图一致性学习语义原型，再用slot级对比学习增强区分性。使得PVM能在任何类型数据上都学到物体中心表示。

## 方法详解

### 整体框架

基于iBOT扩展：(1) 减少原型数量（8192→512）创建信息瓶颈→patch聚类涌现objectness；(2) 加跨视图patch一致性损失→原型获得语义意义；(3) 按原型将patch pooling为slot→slot级MoCo对比学习。

### 关键设计

1. **语义瓶颈（Representation Bottleneck）**：

    - 功能：大幅减少聚类原型数量
    - 核心发现：iBOT用8192原型捕获细粒度模式但缺乏语义，减少到512后objectness自然涌现（Fig.4a：从纹理级聚类变为物体级聚类）
    - 设计动机：信息瓶颈迫使模型学习组合性的物体概念而非低级模式

2. **跨视图一致性（Cross-view Consistency）**：

    - 功能：强制同一图像不同增强视图的对应patch共享相同原型
    - 核心思路：用ROIAlign对齐两个视图的重叠区域，对匹配的patch对计算跨视图交叉熵：$\mathcal{L}_{patch}^{cross}$
    - 设计动机：iBOT的within-view MIM损失不提供视图不变性引导，导致原型缺乏语义一致性

3. **Slot级对比学习**：

    - 功能：将patch按原型分配pooling为object-level slot特征，在slot间做MoCo对比
    - 核心思路：$\mathbf{s}_{\theta,i} = h_\theta(\sum_j p_\theta(\mathbf{v}_j)_i \mathbf{z}_{\theta,j})$，同一原型的slot在两个视图中形成正对
    - 设计动机：patch级学习不足以区分物体，slot级对比学习提升物体级区分性

### 关键发现：逆向缩放

MAE性能随数据增加而提升，但DINO/iBOT从241K扩展到1.28M时性能反而下降！原因：自监督学习目标过度压缩表示，丢失了操作任务需要的低级视觉信息。SlotMIM避免了这个问题——因为它在ego-centric数据上学到的是细粒度部件而非粗粒度物体，不会过度压缩。

## 实验关键数据

### 主实验：241K预训练对比

| 方法 | 数据 | Franka Kitchen | Meta-World | VOC Jacc | ADE mIoU |
|------|------|:-:|:-:|:-:|:-:|
| MAE | Ego-241K | 34.2 | 36.0 | 37.1 | 40.3 |
| DINO | INet-241K | 38.5 | 42.8 | 42.2 | 44.5 |
| iBOT | INet-241K | 40.1 | 43.3 | 43.2 | 48.2 |
| **SlotMIM** | **COCO+-241K** | **42.0** | **44.1** | **43.9** | **49.1** |

### 消融实验：各组件贡献

| 配置 | mask | cross | within | slot | k-NN | ADE | Jacc |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 仅DINO cross-view | ✗ | ✓ | ✗ | ✗ | 45.1 | 47.4 | 42.5 |
| +MIM | ✓ | ✓ | ✗ | ✗ | 44.9 | 48.6 | 42.3 |
| +within-view+slot | ✓ | ✓ | ✓ | ✓ | **46.2** | **49.1** | **43.9** |

### 关键发现

- **241K SlotMIM > 1M+ MVP/VC-1**：仅用1/4数据量即超越利用百万级ego数据的SOTA方法
- **数据类型应匹配任务**：ego数据最适合操作任务，场景数据最适合导航，物体数据最适合感知——"一种数据打天下"不可取
- **Objectness与操作性能相关系数0.72**：这是选择预训练方法的关键信号

## 亮点与洞察

- **挑战两大流行假设**：MAE并非最优PVM方法、ego数据并非最佳选择。这对整个robot learning社区有重要指导意义
- **"逆向缩放"现象的发现与解释**：数据增多反而性能下降——因为自监督学习过度压缩。这是对scaling law的重要补充
- **SlotMIM的自适应性**：在不同类型数据上学到不同粒度的objectness（物体中心→粗粒度物体，ego→细粒度部件），体现了方法的灵活性

## 局限与展望

- Slot的数量（即原型数）需要人工设定，不同数据最优值不同（ImageNet: 1024, COCO: 512）
- 跨视图匹配使用ROIAlign处理裁剪对应，可能不适合大尺度形变
- 6个任务的评估虽然多样，但每个任务的实验规模受限（3 seeds × 有限demos）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 系统性研究视角+SlotMIM方法创新（瓶颈+跨视图+slot对比），挑战领域假设
- 实验充分度: ⭐⭐⭐⭐⭐ 4种数据类型×5种方法×6种任务×多scale，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ Fig.1/2/4/6信息量巨大，从系统分析到方法设计逻辑链清晰
- 价值: ⭐⭐⭐⭐⭐ 为PVM+robot learning社区提供了数据中心的新视角和强效方法

<!-- RELATED:START -->

## 相关论文

- [Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation](mitigating_the_human-robot_domain_discrepancy_in_visual_pre-training_for_robotic.md)
- [Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation](think_small_act_big_primitive_prompt_learning_for_lifelong_robot_manipulation.md)
- [MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)
- [SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)
- [Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](../../ICML2025/robotics/hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)

<!-- RELATED:END -->
