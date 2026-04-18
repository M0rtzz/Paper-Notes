---
title: >-
  [论文解读] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation
description: >-
  [CVPR 2026][3D视觉][扩散策略] 针对扩散策略多步去噪慢、Flow Matching 单步快但模式平均导致碰撞的问题，提出 Ada3Drift：在训练阶段构造 drifting field 将预测吸引到最近 expert demonstration 并排斥其他模式，配合多尺度场聚合和 sigmoid 调度损失过渡，实现 1 NFE 推理下保持多模态动作分布，在 Adroit/Meta-World/RoboTwin 和真实机器人上达到 SOTA。
tags:
  - CVPR 2026
  - 3D视觉
  - 扩散策略
  - Flow Matching
  - 单步推理
  - 多模态动作分布
  - 3D visuomotor policy
---

# Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation

**会议**: CVPR 2026  
**arXiv**: [2603.11984](https://arxiv.org/abs/2603.11984)  
**代码**: 待确认  
**领域**: 3D 视觉 / 机器人操作  
**关键词**: 扩散策略, Flow Matching, 单步推理, 多模态动作分布, 3D visuomotor policy

## 一句话总结
针对扩散策略多步去噪慢、Flow Matching 单步快但模式平均导致碰撞的问题，提出 Ada3Drift：在训练阶段构造 drifting field 将预测吸引到最近 expert demonstration 并排斥其他模式，配合多尺度场聚合和 sigmoid 调度损失过渡，实现 1 NFE 推理下保持多模态动作分布，在 Adroit/Meta-World/RoboTwin 和真实机器人上达到 SOTA。

## 研究背景与动机
扩散策略（Diffusion Policy）通过多步迭代去噪生成动作轨迹，天然支持多模态动作分布（如"左绕"或"右绕"），但推理需要 10-100 次网络前向传播（NFE），实时性差。Flow Matching（FM）将噪声到动作的映射建模为 ODE 直线路径，理论上 1 步即可完成推理。

然而 FM 的 1-step 推理存在根本矛盾：当训练数据包含多种可行动作模式时，条件 Flow Matching 的速度场是所有模式的加权平均——从同一噪声点出发的多条 ODE 路径被平均为一条"中间"路径。这条平均路径往往不对应任何真实可行方案，在 3D 操作中直接表现为轨迹穿越障碍物或撞到环境。

核心矛盾：**如何在单步推理（1 NFE）的计算约束下保持多模态动作分布？** 本文利用离线训练与在线推理的计算预算不对称性——训练时间可以"奢侈"，推理时间必须"节俭"——将多模态保持的计算开销全部转移到训练阶段。

## 方法详解

### 整体思路
Ada3Drift 的核心设计哲学：**在训练阶段引入 drifting field，让每个预测被"吸引"到最近的 expert demonstration 并被其他模式"排斥"，从而在 1-step 网络中隐式编码多模态结构**。推理时仅需 1 次前向传播。

### 关键设计

1. **Training-Time Drifting Field**:

    - 核心目标：为训练样本的每个预测构造一个辅助向量场，使其漂移到正确的模式
    - 匹配机制：计算预测 $\hat{a}_i$ 与 batch 内所有 expert demonstration $a_j^*$ 之间的双向亲和度（bidirectional affinity）：
    $A_{ij} = \sqrt{A_{ij}^{\text{row}} \cdot A_{ij}^{\text{col}}}$
      其中 $A^{\text{row}}$ 为预测侧 softmax（预测选择最佳 expert），$A^{\text{col}}$ 为 expert 侧 softmax（expert 选择最近预测）。几何平均确保匹配是双向一致的
    - 吸引场 $V^+$：将预测拉向匹配的 expert demonstration
    - 排斥场 $V^-$：将预测推离非匹配的 expert（防止模式坍缩为单一模式）
    - 最终 drifting field：$V = V^+ + V^-$，叠加在标准 flow matching 目标上

2. **Multi-Scale Field Aggregation**:

    - 问题：单一温度 $\tau$ 的 softmax 亲和度只能捕获特定空间尺度的模式分离
    - 方案：使用多组温度 $\tau \in \{0.02, 0.05, 0.2\}$，分别产生不同粒度的 drifting field
        - 低温（$\tau=0.02$）：尖锐匹配，捕获局部精细模式
        - 高温（$\tau=0.2$）：平滑匹配，捕获全局粗粒度模式
    - 自归一化聚合：各尺度权重 $\lambda_{\tau_l}$ 通过场的范数自适应归一化，避免手动调参：
    $V_{\text{agg}} = \sum_l \lambda_{\tau_l} V_{\tau_l}$

3. **Sigmoid-Scheduled Loss Transition**:

    - 动机：直接从训练开始就施加 drifting field 会导致不稳定（网络初期预测随机，drifting 方向不可靠）
    - 方案：前 70% 训练使用标准 MLE loss（粗粒度学习整体动作分布），后 30% 渐进引入 drift sharpening loss：
    $w_{\text{drift}} = \sigma\left(\frac{e - 0.7E}{0.05E}\right)$
      其中 $\sigma$ 为 sigmoid 函数，$e$ 为当前 epoch，$E$ 为总 epoch 数
    - 效果：平滑过渡避免训练震荡，前期建立全局分布理解，后期锐化模式边界

4. **Timestep-Free 1D U-Net**:

    - 关键洞察：标准扩散/FM 模型中的时间步嵌入（timestep embedding）用于告诉网络当前去噪进度，但 Ada3Drift 推理时不需要迭代去噪，时间步始终为 0
    - 方案：移除 U-Net 中的时间步条件模块，简化网络结构
    - 3D 感知编码：使用 PointNet++ 编码 3D 点云观测，通过 FiLM（Feature-wise Linear Modulation）注入 1D U-Net 的各层
    - 输入：将动作序列展平为 1D 序列，U-Net 在动作序列维度上操作

### 损失函数
$$\mathcal{L} = (1 - w_{\text{drift}}) \cdot \mathcal{L}_{\text{MLE}} + w_{\text{drift}} \cdot \mathcal{L}_{\text{drift}}$$

其中 $\mathcal{L}_{\text{MLE}}$ 为标准 flow matching 回归损失，$\mathcal{L}_{\text{drift}}$ 包含吸引和排斥项。

## 实验关键数据

### 主实验
| 方法 | NFE | Adroit (Avg SR%) | Meta-World (Avg SR%) | RoboTwin (Avg SR%) |
|------|-----|-------------------|----------------------|---------------------|
| Diffusion Policy | 100 | 74.2 | 82.5 | 68.3 |
| Flow Matching (1-step) | 1 | 58.7 | 71.3 | 52.1 |
| DDIM (10-step) | 10 | 70.8 | 79.6 | 64.5 |
| **Ada3Drift (Ours)** | **1** | **78.5** | **85.2** | **72.8** |

关键发现：
- Ada3Drift 以 1 NFE 超越了需要 100 NFE 的 Diffusion Policy
- 相比 naive Flow Matching 1-step，成功率提升约 +20%
- 推理计算量约为标准扩散策略的 1/10

### 真实机器人
在真实 Franka Panda 机械臂上验证 3 个任务，成功率均超越 Diffusion Policy baseline，且推理延迟 <10ms（满足实时控制要求）。

### 消融实验
| 配置 | Drifting Field | Multi-Scale | Sigmoid Schedule | Avg SR% |
|------|---------------|-------------|-----------------|---------|
| Vanilla FM | ✗ | ✗ | ✗ | 58.7 |
| + Drifting | ✓ | ✗ | ✗ | 69.4 |
| + Multi-Scale | ✓ | ✓ | ✗ | 73.1 |
| **Full Model** | ✓ | ✓ | ✓ | **78.5** |

- Drifting field 贡献最大（+10.7%），验证了核心机制有效性
- 多尺度聚合额外 +3.7%，说明多粒度模式捕获的重要性
- Sigmoid 调度 +5.4%，证明渐进过渡对训练稳定性关键

## 亮点与洞察
- **计算不对称性利用**：将多模态保持的开销从推理时间转移到训练时间，是一个优雅的工程设计哲学
- **双向亲和度**：几何平均的双向匹配比单向 softmax 更鲁棒，避免多对一退化
- **无需额外推理开销**：drifting field 仅在训练时使用，推理网络结构与 vanilla FM 完全相同
- **理论清晰**：从 ODE 速度场的平均化问题出发，drifting field 的引入有明确的几何直觉

## 局限与展望
- Batch 内匹配依赖 batch 大小，batch 太小时模式覆盖不全
- 双向亲和度计算复杂度为 $O(B^2)$（B 为 batch size），超大 batch 时开销显著
- 多温度 $\tau$ 的选择目前为手动设定，可探索自适应学习
- 仅验证了 3D 点云输入，未探索 RGB 图像输入场景
- 排斥场的强度参数需要针对不同任务调整

## 相关工作与启发
- 与 Diffusion Policy (Chi et al., 2023) 的关键区别：1 NFE vs 100 NFE，且保持多模态
- 与 Consistency Model 的对比：Consistency Model 蒸馏需要预训练扩散模型，Ada3Drift 端到端训练
- 双向匹配灵感可能来自 optimal transport 中的 Sinkhorn 算法
- drifting field 概念可推广到其他需要多模态输出的生成任务（如多解姿态估计）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ training-time drifting field 是全新概念，计算不对称性利用思路原创
- 实验充分度: ⭐⭐⭐⭐ 三个仿真平台+真实机器人+完整消融，但缺少与 Consistency Policy 的直接对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，从 ODE 平均化问题自然引出解决方案
- 价值: ⭐⭐⭐⭐⭐ 1-step 多模态策略对机器人实时控制有重大意义，思路可推广
