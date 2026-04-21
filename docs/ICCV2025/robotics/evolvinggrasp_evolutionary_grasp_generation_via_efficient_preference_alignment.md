---
title: >-
  [论文解读] EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment
description: >-
  [ICCV 2025][机器人][灵巧抓取] 提出 EvolvingGrasp，通过 Handpose-wise Preference Optimization (HPO) 和 Physics-Aware Consistency Model (PCM) 实现灵巧抓取姿态的高效进化式生成与人类偏好对齐，在四个基准数据集上取得 SOTA，并实现 30 倍加速。
tags:
  - ICCV 2025
  - 机器人
  - 灵巧抓取
  - 偏好对齐
  - 一致性模型
  - 扩散模型
  - 物理约束
---

# EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment

**会议**: ICCV 2025  
**arXiv**: [2503.14329](https://arxiv.org/abs/2503.14329)  
**代码**: https://evolvinggrasp.github.io/ (有)  
**领域**: Robotics / Dexterous Grasping  
**关键词**: 灵巧抓取, 偏好对齐, 一致性模型, 扩散模型, 物理约束

## 一句话总结

提出 EvolvingGrasp，通过 Handpose-wise Preference Optimization (HPO) 和 Physics-Aware Consistency Model (PCM) 实现灵巧抓取姿态的高效进化式生成与人类偏好对齐，在四个基准数据集上取得 SOTA，并实现 30 倍加速。

## 研究背景与动机

灵巧机器人手在复杂环境中的泛化能力受限于训练数据多样性不足。现实世界场景的无限多样性使得预定义所有抓取策略不切实际。现有方法可分为：
- **优化方法**：通过力闭合状态优化手部姿态，但计算开销大
- **学习方法**：直接回归映射特征到抓取姿态，但存在模式坍塌问题
- **生成方法**：如 DexGrasp Anything 使用扩散模型，但需要数百步采样且无法对齐人类偏好

核心挑战在于：(1) 现有方法无法在部署后持续适应，无法处理训练分布外的变化；(2) 扩散模型的迭代采样和物理约束计算导致效率低下；(3) 缺乏与人类抓取习惯的偏好对齐机制。本文受进化思想启发——系统通过持续反馈学习，从成功和失败中迭代改进——提出进化式抓取生成框架。

## 方法详解

### 整体框架

EvolvingGrasp 由两个核心模块组成：
1. **Handpose-wise Preference Optimization (HPO)**：将偏好对齐形式化为后验概率优化，使模型从成功和失败的抓取样本中迭代学习
2. **Physics-Aware Consistency Model (PCM)**：蒸馏扩散教师模型为轻量一致性模型，集成物理约束保证生成物理可行性

给定物体点云 $O \in \mathbb{R}^{N \times 3}$，目标是从后验分布 $P(x|O)$ 生成高成功率、低穿透的灵巧抓取姿态，其中姿态参数包含关节角度 $\theta_h \in \mathbb{R}^{24}$、全局平移 $T_{global} \in \mathbb{R}^3$ 和全局旋转 $R_{global} \in SO(3)$。

### 关键设计

**HPO（Handpose-wise Preference Optimization）**：
- 首次将 DPO 引入灵巧抓取领域，并扩展为更灵活的形式
- 标准 DPO 要求成对偏好数据（正/负各一个），HPO 放宽了这一限制，允许正负样本数量不等
- 通过 Bradley-Terry 模型建模偏好概率，优化目标使成功抓取的概率增大、失败抓取的概率减小
- 偏好选择支持仿真评估（六方向稳定性测试）或人在回路选择
- 使用 LoRA 进行轻量微调以实现高效偏好对齐

**Physics-Aware Consistency Model (PCM)**：
- **Physics-Aware Distillation**：先训练扩散教师模型，再蒸馏为一致性学生模型，在蒸馏损失中加入三类物理约束：
    - 表面拉力（Surface Pulling Force）：维持手指与物体的稳定接触
    - 外部穿透排斥力（External Penetration Repulsion）：防止手指穿透物体
    - 自穿透排斥力（Self-Penetration Repulsion）：避免手指间碰撞
- **Physics-Aware Sampling**：在采样过程中通过物理约束的梯度修正采样均值，引导轨迹朝向物理可行的姿态

### 损失函数 / 训练策略

总体训练分为三阶段：
1. **扩散预训练**：标准噪声预测损失训练教师模型
2. **物理感知蒸馏**：$\mathcal{L}_{PAD} = \mathcal{L}_{CD} + \sum \alpha_i L_{PA_i}$，一致性蒸馏损失加物理约束
3. **偏好微调**：HPO 损失通过 LoRA 轻量微调整个模型
    - 从仿真中收集成功/失败样本，成功样本为正例、失败为负例
    - 在线迭代：随着更多样本生成，持续提升抓取性能

## 实验关键数据

### 主实验

在 DexGraspNet、MultiDex、RealDex、DexGRAB 四个数据集上评估：

| 方法 | DexGraspNet Suc.6↑ | MultiDex Suc.6↑ | RealDex Suc.6↑ | DexGRAB Suc.6↑ | 时间↓ |
|------|-------|-------|-------|-------|-------|
| UniDexGrasp | 33.9 | 21.6 | 27.1 | 20.8 | 0.46s |
| DexGrasp Any. | 53.6 | 72.2 | 34.6 | 56.5 | 32.91s |
| Ours w/o HPO (4-step) | 63.8 | 75.3 | 51.6 | 55.6 | 1.41s |
| **Ours (4-step)** | **65.2** | **76.8** | **50.6** | **57.7** | **1.41s** |
| Ours (8-step) | 65.4 | 80.3 | 64.4 | 60.8 | 2.71s |
| Real-time (2-step) | 55.2 | 63.7 | 46.5 | 48.9 | **0.06s** |

与 SOTA 方法 DexGrasp Anything 相比实现 **30 倍加速**（32s → 1.41s），同时成功率大幅提升。

### 消融实验

在 MultiDex 数据集上验证各模块贡献（4-step）：

| 配置 | CM | PGD | PGS | HPO | Suc.6↑ | Pen.↓ |
|------|:---:|:---:|:---:|:---:|--------|-------|
| a | ✓ | | | | 60.0 | 14.0 |
| b | ✓ | ✓ | | | 64.3 | 12.5 |
| e | ✓ | ✓ | ✓ | | 75.3 | 13.1 |
| **f** | ✓ | ✓ | ✓ | ✓ | **76.8** | **13.0** |

- 物理约束蒸馏 (PGD) 将 Suc.6 从 60.0 提升到 64.3 (+4.3)
- 物理约束采样 (PGS) 带来最大提升至 75.3 (+11.0)
- HPO 偏好对齐进一步微调至 76.8

### 关键发现

1. 随着 fine-tuning epoch 增加，Suc.6 指标持续改善，穿透深度整体呈下降趋势
2. 从退化数据集训练的次优模型出发，通过 HPO 的进化微调最终超越原始模型精度
3. 无需物理引导的实时模式（2-step）仅需 0.06s，适用于实时应用场景
4. 在真实 ShadowHand 机器人上成功部署，验证了进化抓取能力

## 亮点与洞察

1. **首次将 DPO 引入灵巧抓取**，并扩展为无需严格配对的 HPO，更适合机器人场景
2. **一致性模型 + 物理约束**的结合思路巧妙：既保证少步生成的效率，又通过蒸馏和采样双重物理约束保证合理性
3. **进化自我提升**：模型在部署后可以通过自身生成的成功/失败样本持续改进，无需额外标注
4. 30 倍加速是实际工程价值的重要突破——从 32s 到 1.41s，使实时抓取成为可能

## 局限与展望

1. 偏好微调可能降低生成多样性——偏向对齐的策略可能限制探索空间
2. 目前偏好数据来自仿真（六方向测试），迁移到复杂真实场景时偏好定义可能需要调整
3. 物理约束（穿透力等）依赖已知物体几何信息，对未知物体的泛化能力待验证
4. LoRA 微调的超参数（rank、learning rate）对不同场景的敏感性未充分讨论

## 相关工作与启发

- **DexGrasp Anything**：物理约束扩散模型，速度慢但质量高，是本文的重要基线
- **Diffusion-DPO**：将 DPO 扩展到多步 MDP 用于扩散模型偏好对齐，本文 HPO 的直接灵感来源
- **Consistency Models (CM/sCMs)**：一致性模型框架实现少步采样，本文在此基础上加入物理约束
- 启发：偏好学习 + 物理约束的组合可推广到其他具身操作任务（如装配、工具使用）

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 4.5 |
| 写作质量 | 4 |
| 实用价值 | 4.5 |
| 总评 | 4 |

<!-- RELATED:START -->

## 相关论文

- [DexVLG: Dexterous Vision-Language-Grasp Model at Scale](dexvlg_dexterous_vision-language-grasp_model_at_scale.md)
- [SemGrasp: Semantic Grasp Generation via Language Aligned Discretization](../../ECCV2024/robotics/semgrasp_semantic_grasp_generation_via_language_aligned_discretization.md)
- [Embodied Representation Alignment with Mirror Neurons](embodied_representation_alignment_with_mirror_neurons.md)
- [PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency](pacgdc_label-efficient_generalizable_depth_completion_with_projection_ambiguity_.md)
- [Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context](../../ACL2025/robotics/dice_idiomaticity.md)

<!-- RELATED:END -->
