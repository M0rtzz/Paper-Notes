---
title: >-
  [论文解读] EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds
description: >-
  [ICCV 2025][人体理解][egocentric vision] 提出EgoAgent，一个统一的预测式智能体模型，在单个Transformer中同时学习表征第一人称视觉观测、预测未来世界状态和生成3D人体动作。
tags:
  - ICCV 2025
  - 人体理解
  - egocentric vision
  - agent model
  - world model
  - 3D human motion prediction
  - joint embedding predictive architecture
---

# EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds

**会议**: ICCV 2025  
**arXiv**: [2502.05857](https://arxiv.org/abs/2502.05857)  
**代码**: [https://github.com/zju3dv/EgoAgent](https://github.com/zju3dv/EgoAgent)  
**领域**: human_understanding  
**关键词**: egocentric vision, agent model, world model, 3D human motion prediction, joint embedding predictive architecture

## 一句话总结

提出EgoAgent，一个统一的预测式智能体模型，在单个Transformer中同时学习表征第一人称视觉观测、预测未来世界状态和生成3D人体动作。

## 研究背景与动机

人类通过感知-行动循环持续与环境交互，同时获得视觉感知、预测世界动态和行动决策三种能力。认知科学的Common Coding Theory指出感知和行动深度交织、共享表征空间。然而，现有方法将这三种能力拆分为独立任务分别建模：

1. 视觉表征学习（如DINO、DoRA）——学习世界观测的高层表征
2. 世界模型（如JEPA）——学习状态转移的预测表征
3. 动作预测（如siMLPe）——预测未来人体动作

这种割裂的方式无法捕捉能力之间的内在关联。核心挑战在于：人类与世界的交互涉及感知→行动→观测的连续循环，观测和动作在时间和因果上紧密耦合，如何设计学习框架和监督信号来捕捉这种复杂依赖关系？

## 方法详解

### 整体框架

EgoAgent采用联合嵌入-行动-预测架构(JEAP)，将第一人称视频帧和3D人体姿态编码为交替的"状态-动作-状态-动作"token序列，通过因果注意力机制处理。框架包含两个不对称分支：predictor分支预测未来状态和动作，observer分支从原始观测中提取目标状态。以InternLM作为基础架构（不加载预训练权重），支持300M和1B两种规模。

### 关键设计

1. **交错联合预测 (Interleaved Joint Prediction)**: 在每个时间步$t$，构建结构化token序列：图像token $i_t$、动作查询token $q_a$、动作token $a_t$、状态查询token $q_s$。利用因果注意力机制，$q_a$整合$i_{[0:t]}$和$a_{[0:t-1]}$预测当前动作$A'_t$；$q_s$整合$i_{[0:t]}$和$a_{[0:t]}$预测下一世界状态$S'_{t+1}$。设计动机：显式建模"观测→触发行动→影响下一状态"的因果和时间依赖。

2. **时间不对称的预测器-观察器架构 (Temporally Asymmetric Predictor-Observer)**: 观察器分支仅处理图像输入，提取当前帧特征用于自监督表征学习，以及下一帧特征作为状态预测的监督信号。观察器参数通过预测器的EMA更新。关键优势：查询式设计将共享的状态/表征组件与预测器的动作组件解耦，避免梯度冲突。公式：$\mathcal{L}_{pred}(t) = \mathcal{L}_{dino}(S'_{t+1}, sg[S_{t+1}])$，$\mathcal{L}_{act}(t) = \mathcal{L}_1(A'_t, A_t)$。

3. **语义特征空间学习 (Learning in Semantic Feature Space)**: 使用可学习卷积层将图像投射为连续语义嵌入，而非使用VQGAN等重建型tokenizer的离散token。设计动机：人类基于抽象概念而非像素进行预测，语义特征空间更符合认知过程。实验证明VQGAN的像素级潜空间导致世界状态预测和视觉表征能力的显著下降。

### 损失函数 / 训练策略

整体目标函数：

$$\mathcal{L} = \frac{1}{t}\sum_{k=0}^{t}(\lambda_{rep}\mathcal{L}_{rep} + \lambda_{pred}\mathcal{L}_{pred} + \lambda_{act}\mathcal{L}_{act})$$

- $\mathcal{L}_{rep}$：自监督表征损失，通过同一帧的不同裁剪视图进行DINO式对比学习（$\lambda_{rep}=2$）
- $\mathcal{L}_{pred}$：状态预测损失，预测器与观察器输出的DINO损失（$\lambda_{pred}=1$）
- $\mathcal{L}_{act}$：动作预测损失，预测3D姿态与真值的L1损失（$\lambda_{act}=3$）

训练在WalkingTours和Ego-Exo4D两个数据集上进行。每5帧采样一张图，保留所有3D姿态。300M模型用32卡A100训练25小时，1B模型用48卡训练60小时。batch size 1920，基础学习率6e-4，FP16加速。

## 实验关键数据

### 主实验

**三任务综合性能对比**：

| 方法 | 世界状态预测 Top1/mAP | 动作预测 MPJPE↓(30fps) | 视觉表征 ImgNet-1K Top1 |
|------|---------------------|----------------------|----------------------|
| DoRA | 30.15/45.01 | - | 34.52 |
| DINO | 28.24/43.42 | - | 22.18 |
| siMLPe | - | 13.33 | - |
| Diffusion Policy-T | - | 25.92 | - |
| **EgoAgent-300M** | **43.01/58.06** | **12.92** | **34.65** |
| **EgoAgent-1B** | **46.43/61.96** | **12.51** | **35.84** |

EgoAgent-1B在世界状态预测上超越DoRA +16.28% Top1，在动作预测上改进siMLPe -0.82 MPJPE，在ImageNet-1K上超越DoRA +1.32%。

### 消融实验

**联合学习消融（14400 iterations）**：

| 设置 | 状态预测Top1 | 动作MPJPE↓ | 表征ImgNet-100 Top1 |
|------|------------|-----------|-------------------|
| 完整模型 | 37.77 | 14.49 | 41.64 |
| 去掉$\mathcal{L}_{pred}$ | - | 14.70 | 39.12 |
| 去掉$\mathcal{L}_{act}$ | 34.86 | - | 39.92 |
| 去掉$\mathcal{L}_{rep}$ | 25.90 | 14.49 | - |
| 仅$\mathcal{L}_{pred}$ | 33.23 | - | - |
| 仅$\mathcal{L}_{act}$ | - | 14.32 | - |
| 仅$\mathcal{L}_{rep}$ | - | - | 40.80 |
| 像素级潜空间(无rep) | 20.62 | 13.57 | 1.00 |
| 像素级潜空间(有rep) | 15.63 | 16.25 | 31.20 |

**TriFinger机器人操控**：

| 方法 | Reach Cube | Move Cube |
|------|-----------|-----------|
| DINO | 78.03% | 47.42% |
| DoRA | 82.40% | 48.13% |
| **EgoAgent-1B** | **85.72%** | **57.66%** |

### 关键发现

- **三任务互益**：移除任何一个任务都会降低其他两个任务的性能，证实了联合学习的互补性
- **表征是基础**：去掉$\mathcal{L}_{rep}$对状态预测的伤害最大（-11.87% Top1），说明表征是预测和行动的基础
- **语义 vs 像素**：VQGAN像素级潜空间几乎消灭了视觉表征能力（Top1仅1.00%），证明语义空间的优越性
- **动作多样性对预测的贡献**：相同观测下不同姿态条件能检索到正确反映运动动态的未来图像

## 亮点与洞察

1. **认知科学启发的架构设计**：将Common Coding Theory转化为可计算的JEAP架构，动作查询和状态查询的交错放置完美契合因果时间序列
2. **从零训练LLM架构**：不使用语言预训练权重，证明视觉感知和预测能力可以仅从视觉-动作数据中学习
3. **三任务依赖关系的深入分析**：表征→(预测,行动)→表征构成正循环，但单独的预测或行动无法提升表征
4. **EMA观察器的优雅解耦**：查询式设计使观察器不处理动作模态也能通过EMA稳定更新

## 局限与展望

1. 当前仅使用粗粒度3D身体姿态，未包含手部精细表征，限制了在物体操控等精细任务上的能力
2. 缺少长期记忆机制，20帧滑动窗口可能不足以处理需要长时间依赖的任务
3. 训练数据来自Ego-Exo4D的自动标注姿态，噪声较大
4. 世界状态预测通过特征检索评估，而非直接生成未来帧，应用场景有限
5. 可探索结合语言指令实现目标导向的行动规划

## 相关工作与启发

- **JEPA (LeCun)**: 联合嵌入预测架构的理论基础，本文将其扩展为包含行动预测的反应式智能体
- **DoRA**: 在第一人称视频上的目标级表征学习，是视觉表征的主要对比基线
- **MC-JEPA**: 自监督学习结合光流估计学习内容和运动动态，本文在此基础上加入动作模态
- **Common Coding Theory**: 认知科学中感知与行动共享表征空间的理论，为本文统一框架提供理论依据

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个统一表征、预测和行动的第一人称智能体模型，JEAP架构设计优雅
- **实验充分度**: ⭐⭐⭐⭐ 三任务全面评估，消融研究深入揭示任务间依赖关系
- **写作质量**: ⭐⭐⭐⭐ 认知科学视角的叙事引人入胜，方法描述清晰
- **价值**: ⭐⭐⭐⭐ 为具身智能的统一建模提供了重要参考，开源代码和模型

<!-- RELATED:START -->

## 相关论文

- [ShowUI: One Vision-Language-Action Model for GUI Visual Agent](../../CVPR2025/human_understanding/showui_one_vision-language-action_model_for_gui_visual_agent.md)
- [SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)
- [GENMO: A GENeralist Model for Human MOtion](genmo_a_generalist_model_for_human_motion.md)
- [FREE-Merging: Fourier Transform for Efficient Model Merging](free-merging_fourier_transform_for_efficient_model_merging.md)
- [What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities](../../ICML2025/human_understanding/what_limits_virtual_agent_application_omnibench_a_scalable_multi-dimensional_ben.md)

<!-- RELATED:END -->
