---
title: >-
  [论文解读] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts
description: >-
  [AAAI 2026][自动驾驶][端到端自动驾驶] 提出 ExpertAD，将混合专家（MoE）架构引入端到端自动驾驶系统的感知和预测模块——Perception Adapter 动态重加权 BEV 特征以放大任务关键语义，Mixture of Sparse Experts 通过路由器动态激活相关驾驶任务专家并用稀疏注意力降低计算量，在保持或提升规划效果的同时降低约 25% 推理延迟。
tags:
  - AAAI 2026
  - 自动驾驶
  - 端到端自动驾驶
  - 混合专家
  - 感知适配
  - 稀疏注意力
  - 推理效率
---

# ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts

**会议**: AAAI 2026  
**arXiv**: [2511.11740](https://arxiv.org/abs/2511.11740)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 端到端自动驾驶, 混合专家, 感知适配, 稀疏注意力, 推理效率

## 一句话总结

提出 ExpertAD，将混合专家（MoE）架构引入端到端自动驾驶系统的感知和预测模块——Perception Adapter 动态重加权 BEV 特征以放大任务关键语义，Mixture of Sparse Experts 通过路由器动态激活相关驾驶任务专家并用稀疏注意力降低计算量，在保持或提升规划效果的同时降低约 25% 推理延迟。

## 研究背景与动机

端到端自动驾驶系统（ADS）通过统一感知-预测-规划流水线取得了显著进展，但仍面临两大核心挑战：

**1. 语义模糊干扰决策**：BEV 特征包含各种语义信息（道路、车辆、交通标志等），但不同感知任务（跟踪 vs 建图）的关注点不同。直接传递全部特征可能让非关键维度掩盖关键信息。

**2. 多任务干扰与推理延迟**：预测模块包含本车状态估计、环境交互建模、导航执行等多种任务，全部激活会导致任务间干扰并增加计算开销。例如建图任务有助于弯道规划但对直行作用不大——不同场景需要不同任务组合。

现有 MoE 在自动驾驶中的应用多局限于单一模块（如规划中的轨迹选择），且面临动态场景中专家激活不稳定的问题。先前效率优化方法（DriveAdapter、PlanKD）以牺牲规划质量为代价换取速度。

## 方法详解

### 整体框架

ExpertAD 是一个即插即用的框架，可集成到现有 Transformer 基端到端 ADS（如 UniAD、VAD、VADv2）中。它替换原有的感知和预测模块：

1. **BEV Encoder**（保留）→ 生成 BEV 特征
2. **Perception Adapter (PA)**（新增）→ 动态选择和放大任务关键特征通道
3. **Mixture of Sparse Experts (MoSE)**（新增）→ 动态路由激活相关专家，稀疏注意力降低计算量
4. **Planning Module**（保留）→ 基于 MoSE 输出的 motion query 生成最终轨迹

### 关键设计

**1. Perception Adapter (PA)**

包含两个子组件：

**Learned Adapter**：为每个任务学习通道选择权重。先对 BEV 特征做时间维归一化和池化，然后用任务特定可学习参数 $w^{(t)}$ 计算各通道的重要性分数：

$$s = \frac{1}{H \times W}\sum_{i,j} \tilde{\text{BEV}}_{:,i,j} \odot w$$

通过约束优化求解软通道选择权重 $\lambda^{(t)} \in [0,1]^d$，保证聚焦 $\tau$ 个主导通道：

$$\max_\lambda \; s^\top \lambda + \epsilon\Omega(\lambda), \quad \text{s.t.} \; \mathbf{1}^\top\lambda = \tau, \; \lambda \in [0,1]^d$$

**Alignment Layer**：用选择权重重标定 BEV 特征：

$$F_{align} = \text{MLP}(\text{BEV} \odot \lambda) + \text{BEV}$$

MLP 引入非线性变换，残差连接保留原始空间信息并提供梯度捷径。对齐后的特征分别送入跟踪/建图 Transformer，输出 agent query 和 map query，拼接可学习 embedding 形成 ego query。

**2. Mixture of Sparse Experts (MoSE)**

将预测任务分为三组八个稀疏专家（Sparse Experts）：

| 专家类别 | 专家名称 | 稀疏注意力类型 | 功能 |
|---------|---------|-------------|------|
| Environmental | Tracking Expert, Mapping Expert | Block-wise（块大小 m） | 处理动态前景/地图拓扑 |
| Ego State | Velocity, Yaw, Acceleration Expert | Sliding Window（窗口 w） | 平滑车辆动力学建模 |
| Navigation | Reference Point, BEV, Command Expert | Global TopK | 长距离依赖和导航指令 |

每个专家通过各自的稀疏注意力机制融合 ego query 和专家特定嵌入：

$$\bar{\mathcal{F}}_{expert} = \text{MHCA}(\mathcal{F}_{ego}, \mathcal{F}_{expert}, \mathcal{F}_{expert})$$

**Router**：基于 MoE 门控机制，用可学习参数 $\mathbf{W}_{gate}$ 将 ego query 映射为专家 logits，训练时加入高斯噪声增加随机性，选择 Top-K 专家加权求和生成最终 motion query：

$$\mathcal{F}_{Motion} = \sum_{i=1}^k \mathcal{R}(\mathcal{F}_{ego})_i \cdot \bar{\mathcal{F}}_{expert_i}$$

### 损失函数 / 训练策略

总损失包含四项：

$$\mathcal{L}_{total} = \alpha_1\mathcal{L}_{perception} + \alpha_2\mathcal{L}_{prediction} + \alpha_3\mathcal{L}_{planning} + \alpha_4\mathcal{L}_{switch}$$

其中 Switch Loss 鼓励专家负载均衡：

$$\mathcal{L}_{switch} = N \cdot \sum_{i=1}^N f_i \cdot \mathcal{P}_i$$

惩罚实际负载 $f_i$ 和期望路由概率 $\mathcal{P}_i$ 不一致的专家。训练保持与基线相同的超参数，8× A100 GPU。

## 实验关键数据

### 主实验

**表1：整体性能（Open-loop + Closed-loop + 效率）**

| 方法 | Avg.Col↓ | Avg.L2↓ | DS↑ | SR↑ | RC↑ | Latency↓ |
|------|---------|---------|-----|-----|-----|---------|
| UniAD | 0.31 | 1.03 | 44.62 | 14.09 | 68.68 | 534ms |
| **Expert-UniAD** | **0.24** | **0.89** | **55.49** | **20.63** | **81.04** | **445ms** |
| VAD | 0.43 | 1.21 | 43.31 | 17.27 | 61.60 | 225ms |
| **Expert-VAD** | **0.34** | **1.10** | **52.53** | **19.53** | **76.73** | **157ms** |
| VADv2 | 0.12 | 0.33 | 75.90 | 55.01 | 90.08 | 330ms |
| **Expert-VADv2** | **0.10** | **0.28** | **78.18** | **58.34** | 89.32 | **258ms** |

**表2：稀有场景多技能能力（Bench2Drive220）**

| 方法 | Merge↑ | Overtake↑ | EmgBrake↑ | GiveWay↑ | Tsign↑ |
|------|--------|-----------|-----------|----------|--------|
| UniAD | 12.66 | 13.33 | 20.00 | 10.00 | 13.23 |
| Expert-UniAD | 27.38 | 23.67 | 51.67 | 20.00 | 40.93 |
| VADv2 | 36.25 | 48.33 | 74.28 | 50.00 | 60.14 |
| Expert-VADv2 | 40.44 | 48.33 | 78.42 | 40.00 | 65.78 |

三个基线平均：碰撞率降低约 20%，推理延迟降低约 25%，DS/SR/RC 分别提升 16%/22%/14%。

### 消融实验

- **PA 超参 τ**：τ=128 最优（DS=52.53, SR=18.41, RC=76.73）；τ 过大（256）引入冗余反而下降
- **MoSE Top-K**：Top-4 优于 Top-8 全激活——选择性激活有效减少任务干扰
- **PA 组件**：MLP + ADD 组合 AMOTA 0.404 > 仅 ADD 的 0.390 > baseline 0.388
- **MoSE 组件**：Router 降低 L2 和碰撞率；Sparse Attention 显著降低延迟（Expert-UniAD 降 178ms），两者互补

### 关键发现

1. MoE 在 ADS 中的价值不仅是效率——动态专家选择减少了多任务干扰，**同时提升规划效果与效率**
2. 紧急制动和交通标志场景提升最大（因为这些场景的感知信息最丰富），而超车和让行场需要复杂推理，MoE 贡献有限
3. 跨城市泛化实验（波士顿训练→新加坡测试）显示 ExpertAD 碰撞率从 0.66 降至 0.46（Expert-UniAD），泛化能力较强
4. 统计显著性检验：所有改进 p-value 平均 0.026（p<0.05），结果可靠

## 亮点与洞察

- **横跨感知和预测的端到端 MoE 设计**，不同于先前仅在规划模块使用 MoE 的方法
- PA 的通道选择是可微且有约束的优化问题，比硬剪枝或静态选择更优雅
- 三类专家配备不同稀疏注意力机制，体现了对各任务特性的深入理解：环境→局部块、车辆状态→滑动窗口、导航→全局 TopK
- 即插即用的设计使其可直接增强 UniAD、VAD、VADv2 等多种基线，通用性强

## 局限与展望

- 专家数量（8个）和Top-K 值需要手动设定，可探索自适应专家选择
- 参数量增加（UniAD 89M→125M），尽管 GFLOPs 和延迟下降，但部署时内存开销增加
- 超车和让行场景改善有限，暗示需要更高层次的推理能力，MoE 可能需要与 LLM/世界模型结合
- 目前仅验证了视觉输入的场景，对多模态（LiDAR+Camera）融合的适用性未探索

## 相关工作与启发

- **UniAD** 和 **VAD/VADv2** 是模块化端到端 ADS 的代表，ExpertAD 在其基础上无缝增强
- MoE 在 LLM 中的成功（GLaM, Mixtral）启发了本文在 ADS 中的应用
- Sparse Attention 的三种变体（block-wise, sliding window, global TopK）借鉴了 Longformer 等高效 Transformer 的设计

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 实用价值 | 5 |
| 总评 | 4.4 |

<!-- RELATED:START -->

## 相关论文

- [Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](../../CVPR2026/autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [VILTA: A VLM-in-the-Loop Adversary for Enhancing Driving Policy Robustness](vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)
- [GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts](../../ICCV2025/autonomous_driving/gm-moe_low-light_enhancement_with_gated-mechanism_mixture-of-experts.md)
- [AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)
- [SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](../../NeurIPS2025/autonomous_driving/sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)

<!-- RELATED:END -->
