---
title: >-
  [论文解读] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning
description: >-
  [CVPR 2026][机器人][连续隐运动] 提出 CoMo，通过早期时序差分（Td）和时序对比学习（Tcl）两个机制协同解决连续隐运动学习中的捷径学习问题，从互联网视频中提取精细的连续伪动作标签，使视频数据与机器人动作在统一连续分布下联合训练，显著提升策略性能。
tags:
  - CVPR 2026
  - 机器人
  - 连续隐运动
  - 伪动作标签
  - 逆动力学模型
  - 时序对比学习
  - 视频-机器人联合训练
---

# CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning

**会议**: CVPR 2026  
**arXiv**: [2505.17006](https://arxiv.org/abs/2505.17006)  
**代码**: [github.com/MCG-NJU/CoMo](https://github.com/MCG-NJU/CoMo) (有)  
**领域**: 机器人学习  
**关键词**: 连续隐运动, 伪动作标签, 逆动力学模型, 时序对比学习, 视频-机器人联合训练

## 一句话总结

提出 CoMo，通过早期时序差分（Td）和时序对比学习（Tcl）两个机制协同解决连续隐运动学习中的捷径学习问题，从互联网视频中提取精细的连续伪动作标签，使视频数据与机器人动作在统一连续分布下联合训练，显著提升策略性能。

## 研究背景与动机

从互联网视频中学习隐运动（latent motion）是扩展机器人学习的关键方向，但现有方法面临根本瓶颈：

**离散化的信息损失**：LAPA、UniVLA、Moto-GPT 等先驱工作采用 VQ-VAE 将隐运动离散化（codebook 大小仅 8-16），虽然有效抑制了捷径学习（shortcut learning），但离散化导致大量运动信息丢失，难以捕获复杂和精细的动态。

**捷径学习问题**：如果直接学习连续隐运动（去掉 VQ），逆动力学编码器（IDM）会倾向于从未来帧中提取大量静态背景信息（而非前景运动），因为解码器可以更容易地重建像素细节。模型退化为无效的未来帧预测器。

**离散 vs 连续的分布不匹配**：离散隐运动与连续机器人动作之间存在天然的分布鸿沟，阻碍了统一策略的联合学习，需要复杂的多阶段训练流程。

核心问题：**能否在不用 VQ 的情况下，从无动作标注的视频中学习连续隐运动？** 连续表示更适合精细的帧间变化建模，且与连续机器人动作的分布天然一致。

## 方法详解

### 整体框架

CoMo 采用经典的**逆动力学编码器-前向动力学解码器（IDM-FDM）**架构：
- **IDM**：给定当前帧 $O_t$ 和未来帧 $O_{t+n}$，提取连续隐运动 $Z_{t,t+n}$
- **FDM**：条件于 $Z_{t,t+n}$ 和 $O_t$，重建未来帧 $\hat{O}_{t+n}$

CoMo 在此基础上引入 Td 和 Tcl 两个机制替代 VQ，实现连续隐运动学习。

### 关键设计

1. **早期时序差分（Td, Early Temporal Difference）**：受视频理解中时序差分网络启发，核心改进有两点：

    - **特征差分增强运动线索**：用共享 MAE 预训练 ViT 提取当前帧和未来帧的 token 级特征 $F_t, F_{t+n}$，计算元素级差分 $D_t = F_{t+n} - F_t$，稀疏的特征差分显式放大了运动信号。
    - **移除未来帧特征**：将 $[F_t, D_t]$ 拼接（而非 $[F_t, F_{t+n}]$）送入 Motion Q-former，从根本上增加捷径学习难度——编码器无法直接获取未来帧的完整表示。

2. **时序对比学习（Tcl, Temporal Contrastive Learning）**：仅靠 Td 的稀疏差分可能包含无关信息，且单独的对比学习可能只捕获前景物体身份（"是什么"和"在哪里"）而忽略运动模式（"怎么动"）。CoMo 设计了特殊的正负样本对：

    - **正样本对**：$Z_{t,t+n+\delta}$ 和 $Z_{t,t+n}$——时序略有偏移的运动对（$\delta \in [-n/5, n/5]$）
    - **负样本对**：$Z_{t+n,t}$ 和 $Z_{t,t+n}$——时序方向反转的运动对
    - 损失函数为 InfoNCE：
    $\mathcal{L}_{\text{tcl}} = -\log\frac{e^{S_1}}{e^{S_1} + e^{S_2} + e^{S_3}}$

   Td 确保运动线索增强 + Tcl 确保前景聚焦，两者协同工作。

3. **联合统一策略学习**：关键优势——连续隐运动与连续机器人动作共享相同的连续分布，可在统一策略模型中直接联合训练（只需分配不同的轻量 head），无需多阶段预训练或显式的两阶段 motion-before-action 流程。

4. **评估指标设计**：提出两个无需策略评估的隐运动分析指标：

    - **MSE**：训练 MLP 从隐运动回归真实动作，评估动作相关信息编码能力
    - **S-PCFC**：过去到当前 vs 未来到当前运动的余弦相似度，诊断动作无关背景噪声

### 损失函数 / 训练策略

- CoMo 联合最小化加权 InfoNCE 损失、像素级重建损失和感知损失
- 使用 120K 互联网视频（SAM-V 40K + EgoVid 40K + Droid 40K）训练 IDM-FDM
- 策略训练支持 Diffusion（LIBERO）和自回归（CALVIN）两种架构

## 实验关键数据

### 主实验

LIBERO 基准（每任务仅 10 条机器人轨迹 + 视频伪标签）：

| 方法 | Spatial | Object | Goal | Long | 平均SR |
|------|---------|--------|------|------|--------|
| DP (无视频) | 72.3 | 82.3 | 70.3 | 56.7 | 70.4 |
| GR2-like（未来帧特征） | 76.0 | 92.0 | 73.3 | 53.7 | 73.8 |
| GR00T（离散隐运动） | 80.7 | 83.3 | 80.0 | 59.7 | 75.9 |
| Dynamo（协方差正则化） | 75.3 | 92.7 | 80.7 | 46.0 | 73.7 |
| **CoMo** | **80.3** | **97.0** | **81.0** | **62.0** | **80.1** |

CALVIN ABC→D 基准：

| 方法 | 1步 | 2步 | 3步 | 4步 | 5步 | 平均 |
|------|-----|-----|-----|-----|-----|------|
| 无运动 | 0.772 | 0.494 | 0.307 | 0.191 | 0.114 | 1.878 |
| 离散(Moto) | 0.801 | 0.575 | 0.409 | 0.283 | 0.187 | 2.255 |
| **CoMo** | **0.882** | **0.732** | **0.589** | **0.490** | **0.377** | **3.070** |
| CoMo ×4维度 | 0.891 | 0.758 | 0.646 | 0.529 | 0.423 | 3.247 |

### 消融实验

| 配置 | 平均SR | MSE↓ | S-PCFC↓ | 说明 |
|------|--------|------|---------|------|
| 未来帧特征 | 73.8 | 2.14 | 1.000 | 大量背景噪声 |
| 离散(Dis.) | 75.9 | 5.67 | 0.481 | 信息损失严重 |
| 连续基线(Con.) | 75.2 | 1.63 | 0.927 | 捷径学习问题 |
| Con.+Td | 76.9 | 1.52 | 0.889 | Td 降低 S-PCFC |
| Con.+Tcl | 77.9 | 1.31 | 0.623 | Tcl 大幅降低 S-PCFC |
| **Con.+Td+Tcl (CoMo)** | **80.1** | **1.26** | **0.550** | 两者协同最优 |

### 关键发现

- Td 和 Tcl 各自有效但互补：Td 主要降低 MSE（增强运动线索），Tcl 主要降低 S-PCFC（抑制背景噪声）
- 连续表示优于离散表示：CoMo 平均 SR 80.1% vs 离散 75.9%，尤其在精细操作任务（Object 97.0% vs 83.3%）
- 维度可扩展：CoMo 从 128→512 维持续提升（CALVIN 3.070→3.247），而仅 Td 会随维度增加引入更多噪声
- 真实世界实验同样验证连续隐运动优势，尤其在低容错精细操作（开抽屉、插面包）上

## 亮点与洞察

- **连续 > 离散**的论证充分且有说服力：离散化虽能抑制捷径学习但代价太高
- **Td + Tcl 的协同设计**非常精巧：Td 解决"从哪里学"（移除未来帧直接信息），Tcl 解决"学什么"（正负样本引导前景聚焦 + 运动方向性）
- **MSE + S-PCFC 评估体系**提供了无需昂贵策略评估的快速诊断工具
- 联合训练的简洁性是重要工程价值：一阶段统一训练 vs 多阶段复杂流程

## 局限与展望

- 仅在 LIBERO 和 CALVIN 上验证，泛化到更复杂的真实世界场景有待探索
- 隐运动维度扩展的上限和最优维度选择缺乏理论指导
- Tcl 的正负样本构建依赖时序偏移假设（$\delta$ 范围），在快速运动场景可能不理想
- 真实世界实验规模较小（每任务 25 条轨迹 + 25 条人手视频）

## 相关工作与启发

- 与 LAPA/GR00T 等离散隐运动方法的核心对比：CoMo 证明连续化是更好的方向
- 时序差分来自视频理解社区（TDN），跨领域迁移到机器人学习效果显著
- 对比学习的正负样本设计有创意：用时序反转作为负样本，天然适合运动方向区分

## 评分

- **新颖性**: ⭐⭐⭐⭐ Td+Tcl 替代 VQ 的思路清晰有新意，但各组件并非全新
- **实验充分度**: ⭐⭐⭐⭐⭐ 消融极其全面，MSE/S-PCFC 分析深入，跨架构验证充分
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，实验分析有深度，但部分表述略显冗余
- **价值**: ⭐⭐⭐⭐⭐ 为从互联网视频扩展机器人学习提供了实用且原理清晰的方案

<!-- RELATED:START -->

## 相关论文

- [Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](../../ICCV2025/robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [Chain of World: World Model Thinking in Latent Motion (CoWVLA)](chain_of_world_world_model_thinking_in_latent_motion.md)
- [DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)
- [AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](atomicvla_unlocking_the_potential_of_atomic_skill.md)
- [Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)

<!-- RELATED:END -->
