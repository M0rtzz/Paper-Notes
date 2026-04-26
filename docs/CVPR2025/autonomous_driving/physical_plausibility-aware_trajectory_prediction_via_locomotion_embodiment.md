---
title: >-
  [论文解读] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment
description: >-
  [CVPR 2025][自动驾驶][轨迹预测] 提出 Locomotion Embodiment 框架，利用物理模拟器中的人形运动生成评估轨迹的物理合理性，通过可微的 LocoVal 函数替代不可微的物理模拟器来训练轨迹预测网络，并在推理时过滤不合理轨迹。
tags:
  - CVPR 2025
  - 自动驾驶
  - 轨迹预测
  - 物理合理性
  - 运动生成
  - 物理模拟器
  - 可微代理
---

# Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment

**会议**: CVPR 2025  
**arXiv**: [2503.17267](https://arxiv.org/abs/2503.17267)  
**代码**: [GitHub](https://github.com/ImIntheMiddle/EmLoco)  
**领域**: 自动驾驶  
**关键词**: 轨迹预测, 物理合理性, 运动生成, 物理模拟器, 可微代理

## 一句话总结

提出 Locomotion Embodiment 框架，利用物理模拟器中的人形运动生成评估轨迹的物理合理性，通过可微的 LocoVal 函数替代不可微的物理模拟器来训练轨迹预测网络，并在推理时过滤不合理轨迹。

## 研究背景与动机

人体轨迹预测(HTP)旨在预测行人未来移动路径，对自动驾驶、机器人和安防系统至关重要。现有方法虽然利用了人体姿态信息，但存在以下不足：

- **姿态信息利用不充分**：现有方法仅将姿态作为额外输入隐式学习，预测的轨迹可能与观测到的人体姿态物理不一致
- **数据驱动的局限**：需要大规模的姿态-轨迹配对数据集覆盖所有可能组合，但这类数据集不存在
- **瞬时观测场景**：当仅有少量过去帧可用时（如行人突然从障碍物后出现），基于长期观测的方法失效
- **随机预测的多样性-合理性矛盾**：minMSE 损失仅监督最佳预测头，导致其余头欠拟合；MSE 损失则迫使所有预测收敛到单一真值，牺牲多样性

## 方法详解

### 整体框架

框架包含两个训练阶段和一个推理阶段：(1) 训练 LocoVal 函数——使用物理模拟器中的 PACER 运动生成器获取轨迹合理性评分作为真值，训练可微的 MLP 代理函数；(2) 训练 HTP 网络——使用 EmLoco 损失和标准 MSE 损失联合训练；(3) 推理——使用 LocoVal 过滤器滤除不合理轨迹。

### 关键设计1: LocoVal 函数 — 可微的物理合理性代理

**功能**: 替代不可微的物理模拟器，在训练和推理中评估轨迹的物理合理性。

**核心思路**: 在物理模拟器(IsaacGym)中使用 PACER 运动生成器控制人形沿给定轨迹行走。当轨迹物理不合理（如急转弯），人形无法跟随，累积奖励 $\Omega$ 较低。训练 MLP 网络 $\mathcal{V}$ 从可用的图像级线索（未来轨迹 $\tau_s$、初始姿态 $\mathbf{j}_0$、根关节速度 $\mathbf{v}_{\text{root},0}$）估计该奖励：

$$\mathcal{L}_\mathcal{V} = \text{MSE}(\mathcal{V}(\tau_s, \mathbf{h}'_0), \mathcal{G}(\tau_s, \mathbf{h}_0))$$

训练时使用合理配对和随机不合理配对，使 $\mathcal{V}$ 能区分合理与不合理的姿态-轨迹组合。

**设计动机**: (1) 物理模拟器不可微，无法直接反向传播；(2) 模拟器需要完整的物理状态（关节角速度等），在图像场景中不可用。$\mathcal{V}$ 用少量可用线索即可估计合理性，且推理开销可忽略。

### 关键设计2: EmLoco 损失 — 联合监督所有预测头

**功能**: 利用物理合理性先验同时训练所有预测头，避免 minMSE 导致的欠拟合问题。

**核心思路**: EmLoco 损失定义为 $\mathcal{L}_E = -\mathcal{V}(\hat{\tau}_f, \mathbf{h}'_0)$，鼓励预测合理轨迹。与标准 MSE 损失组合：

$$\mathcal{L}_\text{total} = \mathcal{L}_T + \alpha \mathcal{L}_E$$

对于多头随机预测，$\mathcal{L}_T$ 使用 $\min_k \text{MSE}(\hat{\tau}_f^k, \tau_f)$（仅监督最佳头），而 **$\mathcal{L}_E$ 对所有头的预测取平均**，实现全头同时优化。

**设计动机**: MSE 损失是数据项（拟合真值），EmLoco 损失是正则化项（融入物理先验）。EmLoco 不需要唯一真值，因此可以在保持预测多样性的同时提升所有头的合理性，解决了随机 HTP 中效率与多样性的矛盾。

### 关键设计3: LocoVal 过滤器 — 即插即用的推理时过滤

**功能**: 在推理时评估并过滤多头预测中的不合理轨迹。

**核心思路**: 对所有预测轨迹计算合理性评分，低于阈值 $\lambda$ 的轨迹被过滤：保留所有 $\mathcal{V}(\hat{\tau}_f^k, \mathbf{h}'_0) \geq \lambda$ 的轨迹；若全部低于阈值，保留评分最高的一条。

**设计动机**: 可与任何预训练的随机 HTP 网络即插即用使用，无需重新训练。

### 损失函数

$$\mathcal{L}_\text{total} = \underbrace{\min_k \text{MSE}(\hat{\tau}_f^k, \tau_f)}_{\text{minMSE (data term)}} + \alpha \underbrace{(-\frac{1}{K}\sum_k \mathcal{V}(\hat{\tau}_f^k, \mathbf{h}'_0))}_{\text{EmLoco (physics prior)}}$$

其中 $\alpha = 100$。

## 实验关键数据

### 主实验: 标准设置 (9帧观测, 确定性预测)

| 方法 | JTA ADE↓ | JTA FDE↓ | JRDB ADE↓ | JRDB FDE↓ |
|------|---------|---------|----------|----------|
| EqMotion | 1.13 | 2.39 | 0.40 | 0.77 |
| Social-Trans | 1.11 | 2.26 | 0.40 | 0.76 |
| **Ours** | **0.97** | **1.91** | **0.37** | **0.72** |

### 多头随机预测 (5头/20头)

| 方法 | ADE (5/20) | FDE (5/20) | χ² Vel. | χ² Acc. |
|------|-----------|-----------|---------|---------|
| Social-Trans | 1.86/2.14 | 3.51/4.26 | 0.134/0.169 | 0.009/0.009 |
| **Ours** | **1.68/1.80** | **3.34/3.56** | **0.100/0.087** | **0.002/0.003** |

### LocoVal 过滤器效果 (即插即用, ETH/UCY)

| 方法 | ADE/FDE (无过滤) | ADE/FDE (有过滤) |
|------|----------------|----------------|
| EqMotion | 0.36/0.55 | **0.35/0.53** |

### 关键发现

- 在JTA数据集上确定性ADE从1.11降至**0.97**（-12.6%），FDE从2.26降至**1.91**（-15.5%）
- EmLoco损失对所有预测头的联合优化使 $\chi^2$ 距离（速度、加速度等物理指标）大幅改善
- 即使 JRDB 使用的是估计而非真实 3D 姿态，方法仍有效提升，证明了对噪声输入的鲁棒性
- LocoVal 过滤器可即插即用提升预训练模型性能

## 亮点与洞察

1. **首次将物理模拟器中的人形控制引入轨迹预测**，建立了姿态-轨迹物理一致性的评估标准
2. **EmLoco损失的巧妙设计**：作为正则化项不需真值标签，因此可同时优化所有预测头
3. **物理模拟器仅用于训练前阶段**，推理时零额外成本
4. **方法论的模型无关性**：既适用于确定性也适用于随机 HTP 方法

## 局限与展望

- LocoVal 函数基于 AMASS 运动数据训练，可能对极端运动(奔跑、跳跃)泛化不佳
- 当前只考虑了单人的物理合理性，未建模多人交互的物理约束
- 鸟瞰图场景(ETH/UCY)无法使用 3D 姿态，仅能使用 LocoVal 过滤器
- 未来可探索将环境物理约束(如地形、障碍物)也纳入合理性评估

## 相关工作与启发

- **Social-Transmotion**: 基于 Transformer 的社交轨迹预测，支持姿态输入
- **PACER**: 基于强化学习的物理人形运动控制器
- **EqMotion**: 基于等变网络的随机轨迹预测方法
- **MATRIX**: 提出 $\chi^2$ 距离评估轨迹物理合理性

## 评分

⭐⭐⭐⭐ — 创新性强，首次将物理模拟与轨迹预测结合，理论优雅（可微代理替代不可微模拟器），实验全面（多数据集、多设置、消融）。EmLoco 损失对多头训练的改进有清晰的理论解释和实验验证。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[CVPR 2025\] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)
- [\[CVPR 2025\] SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction](socialmoif_multi-order_intention_fusion_for_pedestrian_trajectory_prediction.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](../../ICCV2025/autonomous_driving/donut_a_decoder-only_model_for_trajectory_prediction.md)

<!-- RELATED:END -->
