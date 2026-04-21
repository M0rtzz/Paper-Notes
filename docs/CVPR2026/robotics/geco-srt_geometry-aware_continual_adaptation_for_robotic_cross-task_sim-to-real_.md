---
title: >-
  [论文解读] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer
description: >-
  [CVPR 2026][机器人] 提出一种基于几何感知的持续适应方法 GeCo-SRT，通过从局部几何特征中提取跨域/跨任务不变知识，在多次 sim-to-real 迁移中实现知识积累，从而高效适应新任务。
tags:
  - CVPR 2026
  - 机器人
---

# GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer

**会议**: CVPR 2026  
**arXiv**: [2602.20871](https://arxiv.org/abs/2602.20871)  
**代码**: 无  
**领域**: 机器人

## 一句话总结

提出一种基于几何感知的持续适应方法 GeCo-SRT，通过从局部几何特征中提取跨域/跨任务不变知识，在多次 sim-to-real 迁移中实现知识积累，从而高效适应新任务。

## 背景与动机

1. **Sim-to-real 差距是制约机器人学习的核心瓶颈**：仿真中训练的策略因视觉渲染和物理动力学差异，部署到真实世界时性能严重下降（Direct Deploy 平均成功率仅 3.1%）。
2. **现有方法将每次 sim-to-real 迁移视为独立过程**：System Identification 依赖手工建模、Domain Randomization 需要专家调参，且均仅支持单任务迁移，每个新任务都需从零开始，成本高昂。
3. **缺乏跨任务知识积累机制**：不同操作任务间共享大量几何基元（如边缘、平面），但已有方法未利用这些可迁移的几何先验来加速后续任务的适应。
4. **持续学习中的灾难性遗忘**：在顺序学习多个任务时，新任务训练会覆盖旧任务知识，标准 PER 方法基于任务损失优先采样，忽略了专家级别的知识保护。

## 方法详解

### 3.1 基于人类校正的 Sim-to-Real 迁移框架

对每个任务 $i$，首先在仿真中用模仿学习训练基础扩散策略 $\pi^{b_i}$（含点云编码器 $E_p^{b_i}$ 和扩散策略头 $\pi_h^{b_i}$），使用 L2 扩散损失：

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{(a_t, o_t) \sim \mathcal{D}_{sim}^{e_i}, \epsilon \sim \mathcal{N}(0,I)} \left[ \| \epsilon - \epsilon_\theta(a_t^k, k, o_t) \|^2 \right]$$

将基础策略部署到真实世界，通过 human-in-the-loop 共享自主框架收集人类校正轨迹 $\mathcal{D}_{real}^{h_i}$。冻结基础策略参数，引入共享感知残差模块 $E_p^s$（即 Geo-MoE），用仿真与真实世界数据的混合回放缓冲区 $D_{buf}^{m_i}$ 持续更新。

### 3.2 几何感知混合专家模块 (Geo-MoE)

核心洞察：**局部几何特征具有双重不变性**——

- **域不变性**：局部几何（如平面度、线性度）在仿真和真实世界中结构一致，不受纹理/材质差异影响
- **任务不变性**：几何基元（如边缘、角点）在不同操控任务间共享（如"抓取方块"与"堆叠方块"共享方块平面特征）

具体实现：

1. 使用 k-NN 从输入点云 $P$ 中采样局部点组 $g_i$
2. 通过局部 PCA 估算每组的几何特征（平面度、线性度、显著性）
3. 轻量门控网络 $G$ 产生路由权重 $w_i = \text{Softmax}(G(g_i))$
4. $M$ 个并行专家网络加权融合：

$$g_i' = \sum_{j=1}^{M} w_{i,j} \cdot \text{Expert}_j(g_i)$$

5. 聚合所有组特征得到校正残差向量 $g_{res}'$，与冻结编码器输出拼接：$\hat{f} = \text{Concat}(E_p^{b_i}(o), g_{res}')$

训练损失：

$$L_{total} = \text{MSE}(\hat{a}, a) + \alpha L_{balance}$$

其中 $L_{balance}$ 为负载均衡损失，防止门控塌缩。

### 3.3 几何专家引导的优先经验回放 (Geo-PER)

标准 PER 基于任务损失优先采样，忽略了空闲专家的知识退化。Geo-PER 将优先级度量从任务损失转移到**专家利用率**：

对历史样本 $i \in \mathcal{R}$，记录其专家激活向量 $W_i = \{w_{i,1}, \dots, w_{i,M}\}$，计算当前新任务的平均专家利用率 $U^{\text{new}}$，动态更新采样优先级：

$$P_i \propto \sum_{j=1}^{M} w_{i,j} \cdot \frac{1}{u_j^{\text{new}} + \epsilon}$$

直觉：如果某专家 $j$ 在当前任务中利用率低（$u_j^{\text{new}}$ 小），则高度激活该专家的历史样本获得更高采样概率，确保空闲专家参数持续参与梯度更新以对抗遗忘。

## 实验结果

### 实验设置

使用 4 个机器人操控任务序列验证：Pick Cube → Stack Cube → Pick Banana → Plug Insert。真实硬件为 Xarm5 机械臂 + Rotiq2F140 夹爪 + 双 RealSense 深度相机。每个任务收集 60 条人类校正轨迹。

### 单任务 Sim-to-Real 迁移

| 方法 | Pick Cube | Stack Cube | Pick Banana | Plug Insert | 平均 SR |
|------|-----------|------------|-------------|-------------|---------|
| Direct Deploy | 5.7% | 0% | 6.7% | 0% | 3.1% |
| Action Residual | 16.7% | 3.3% | 13.3% | 0.0% | 9.2% |
| Transic | 66.7% | 30.0% | 23.3% | 33.3% | 38.3% |
| **Geo-MoE** | **80.0%** | **43.3%** | **40.0%** | **36.7%** | **50.0%** |

Geo-MoE 比最强基线 Transic 平均高出 11.7%，验证了几何感知特征在弥补观察域差距中的优势。

### 跨任务持续学习

| 方法 | Pick Cube SR↑ | N-NBT↓ | Stack Cube SR↑ | N-NBT↓ | Pick Banana SR↑ | N-NBT↓ | Plug Insert SR↑ | 平均 SR↑ | 平均 N-NBT↓ |
|------|-------------|--------|--------------|--------|---------------|--------|----------------|---------|------------|
| Naive Fine-tuning | 16.7% | 100% | 3.3% | 100% | 13.3% | 100% | 3.3% | 9.2% | 75.0% |
| Transic + PER | 76.7% | 81.2% | 30.0% | 72.3% | 20.0% | 66.5% | 33.3% | 40.0% | 55.0% |
| Geo-MoE + PER | 83.3% | 34.6% | 50.0% | 76.8% | 46.7% | 7.1% | 43.3% | 55.7% | 29.6% |
| Geo-MoE + EWC | 80.0% | 70.8% | 36.7% | 77.3% | 20.0% | 50.0% | 16.7% | 38.3% | 49.5% |
| **GeCo-SRT** | **86.7%** | **28.3%** | **53.3%** | **72.0%** | **60.0%** | **5.5%** | **53.3%** | **63.3%** | **26.5%** |

GeCo-SRT 平均成功率 63.3%，比基线高 52%；平均遗忘率 N-NBT 仅 26.5%，远优于其他方法。

### 数据效率

使用仅 20 条校正轨迹时，持续学习方法在 Pick Cube 上达到 76.6% 成功率，几乎等同于从零训练使用 60 条轨迹的效果。整体上仅需 **1/6 数据量**即可匹配基线的全数据性能。

## 亮点

- **范式创新**：首次将 sim-to-real 迁移从"单次孤立"提升为"持续跨任务"范式，实现知识积累
- **几何双重不变性**：巧妙利用局部几何特征的域不变性+任务不变性作为可迁移知识的载体
- **专家级遗忘防护**：Geo-PER 从专家利用率角度重新定义优先级，精准保护空闲专家的专业知识
- **实际验证充分**：在真实机器人（Xarm5）上完成 4 个任务的完整持续迁移实验，非纯仿真验证
- **显著数据效率**：1/6 数据即可达到基线全数据性能，对真实世界数据稀缺场景意义重大

## 局限性

- **仅聚焦观察域差距**：方法主要通过几何特征弥补视觉观察的 sim-to-real gap，对复杂动力学差异（如摩擦力、接触力）的适应能力有限
- **任务数量有限**：实验仅涉及 4 个任务的顺序学习，更长任务序列下的可扩展性未验证
- **依赖人类校正**：每个任务仍需人类操作员收集 60 条校正轨迹，完全自主化程度不足
- **任务相似性依赖**：跨任务迁移效果高度依赖几何相似性，几何差异大的任务间可能产生负迁移

## 评分

- ⭐⭐⭐⭐ 新颖性：首次提出持续跨任务 sim-to-real 范式，几何不变性作为可迁移知识载体的洞察新颖
- ⭐⭐⭐⭐ 实用性：显著降低新任务适应的数据成本，对真实机器人部署有直接价值
- ⭐⭐⭐ 实验充分度：真机实验完整但任务规模偏小（4个任务），缺少更多样化场景
- ⭐⭐⭐ 写作质量：结构清晰、动机充分，但部分公式符号不够一致

<!-- RELATED:START -->

## 相关论文

- [Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](rcnf_robot_conditioned_normalizing_flow_anomaly.md)
- [Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](actiongeometry_prediction_with_3d_geometric_prior.md)
- [MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](mergevla_crossskill_model_merging_toward_a_general.md)

<!-- RELATED:END -->
