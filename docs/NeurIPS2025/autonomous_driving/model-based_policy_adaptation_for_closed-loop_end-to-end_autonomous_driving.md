---
title: >-
  [论文解读] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving
description: >-
  [NeurIPS 2025][自动驾驶][端到端自动驾驶] 提出 MPA 框架，通过 3DGS 仿真生成反事实轨迹数据，训练扩散策略适配器和多原则 Q 值模型，在推理时引导预训练 E2E 驾驶模型提升闭环场景下的安全性和泛化能力。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 端到端自动驾驶
  - 闭环评估
  - 反事实数据
  - 扩散策略
  - Q值引导
---

# Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving

**会议**: NeurIPS 2025  
**arXiv**: [2511.21584](https://arxiv.org/abs/2511.21584)  
**代码**: [项目主页](https://mpa-drive.github.io/) (有)  
**领域**: Autonomous Driving  
**关键词**: 端到端自动驾驶, 闭环评估, 反事实数据, 扩散策略, Q值引导

## 一句话总结

提出 MPA 框架，通过 3DGS 仿真生成反事实轨迹数据，训练扩散策略适配器和多原则 Q 值模型，在推理时引导预训练 E2E 驾驶模型提升闭环场景下的安全性和泛化能力。

## 研究背景与动机

**领域现状**: 端到端（E2E）自动驾驶模型在开环评估中表现良好，但在闭环部署中性能显著下降，出现级联错误和泛化不足。

**现有痛点**: 开环训练基于模仿学习最小化行为克隆损失，与闭环部署中最大化累积奖励的目标存在根本性错配；现有改进方案要么缺乏闭环评估，要么计算成本高（如在线 RL）。

**核心矛盾**: 存在两个根本性 mismatch — (1) **观测错配**: 训练时的传感器输入与部署时闭环观测的分布偏移；(2) **目标错配**: 离线模仿学习缺乏有意义的奖励反馈，限制了长期推理。

**本文目标**: 将预训练的开环 E2E 驾驶模型适配为安全可靠的闭环驾驶智能体。

**切入角度**: 利用 3DGS 仿真引擎生成反事实数据弥合分布偏移，同时训练策略适配器和价值模型。

**核心 idea**: 反事实数据 + 扩散残差策略 + Q 值推理时缩放的三位一体框架。

## 方法详解

### 整体框架

MPA 包含三个核心组件：(1) 基于世界模型的反事实数据生成——用 3DGS 仿真器生成多样化的驾驶轨迹数据；(2) 扩散策略适配器——学习残差轨迹修正预训练模型输出；(3) Q 值引导的推理时采样——基于多原则价值模型选择最优轨迹候选。

### 关键设计

1. **反事实数据生成**: 使用 3DGS 仿真器（HUGSIM）渲染逼真的驾驶场景。基于预训练 E2E 策略 $\hat{\pi}_{\text{ref}}$ 的输出进行随机增强（旋转 $[-10°, 10°]$、变形、高斯噪声），生成多样化的行为轨迹。采用 beam search 保留最高奖励的候选轨迹，超出距离阈值或低于最低奖励的轨迹被丢弃。生成的数据包含 (state, action, observation, reward) 元组。

2. **扩散策略适配器**: 预测残差轨迹 $\Delta a = a^* - a^{\text{base}}$，其中 $a^{\text{base}}$ 为冻结的预训练模型输出。使用 1D U-Net 作为去噪网络，以场景编码 $z = \phi_{\text{enc}}(o, \boldsymbol{s}_{\text{ego}})$、自车历史和基础预测轨迹为条件，支持多模态输出。训练损失:
    $\mathcal{L}_{\text{diff}} = \mathbb{E}_{\Delta a^{(0)}, k, \epsilon} \min_i \|f_\theta(\Delta a^{(k)}, k, z, \boldsymbol{s}_{\text{ego}}, a^{\text{base}})[i] - \Delta a^{(0)}\|_2^2$
   推理时用 DDIM 采样残差，恢复适配轨迹 $a^{\text{adapt}} = a^{\text{base}} + \Delta a^{(0)}$。

3. **多原则 Q 值模型**: 训练四个独立的 Q 函数评估长期回报:

    - $Q_{\text{route}}$: 路线跟随
    - $Q_{\text{dist}}$: 车道距离
    - $Q_{\text{collision}}$: 碰撞避免
    - $Q_{\text{speed}}$: 速度合规
   
   总 Q 值为加权求和: $Q = \sum_{i} w_i \times Q_i$。推理时从策略适配器采样多个残差动作，选择 Q 值最高的:
    $\Delta\hat{a}^* = \arg\max_{\Delta a} Q(o_t, \boldsymbol{s}_{\text{ego}}, a^{\text{base}} + \Delta a; T)$

### 损失函数 / 训练策略

- 策略适配器用扩散损失训练（预测去噪后的残差动作）
- Q 值模型用反事实数据的多步累积奖励做监督训练
- 推理时采样 20 个候选动作，用 Q 值模型选择最优

## 实验关键数据

### 主实验

**域内闭环评估:**

| 模型 | RC↑ | NC↑ | DAC↑ | TTC↑ | HDScore↑ |
|------|-----|-----|------|------|----------|
| UniAD | 39.4 | 56.9 | 75.1 | 52.1 | 19.4 |
| VAD | 50.1 | 68.4 | 87.2 | 66.1 | 31.9 |
| LTF | 65.2 | 71.3 | 92.1 | 67.6 | 46.7 |
| Diffusion | 71.8 | 67.4 | 88.1 | 64.5 | 45.1 |
| **MPA (UniAD)** | **93.6** | 76.4 | 92.8 | 72.8 | **66.4** |
| **MPA (VAD)** | **94.9** | 75.4 | 93.6 | 72.5 | **67.0** |

**安全关键场景评估:**

| 模型 | RC↑ | NC↑ | HDScore↑ |
|------|-----|-----|----------|
| UniAD | 11.4 | 76.2 | 4.5 |
| VAD | 25.4 | 77.0 | 16.0 |
| LTF | 35.1 | 80.9 | 24.2 |
| **MPA (UniAD)** | **95.1** | 76.8 | **70.4** |
| **MPA (VAD)** | **96.6** | 79.8 | **74.7** |

MPA 在安全关键场景中将 HDScore 从 16.0 提升到 74.7（VAD 基线），路线完成率从 25.4% 提升到 96.6%。

### 消融实验

| ID | $Q_{\text{route}}$ | $Q_{\text{dist}}$ | $Q_{\text{collision}}$ | $Q_{\text{speed}}$ | Adapter | HDScore(安全) |
|----|----|----|----|----|---------|------------|
| 1 | ✗ | ✓ | ✓ | ✓ | ✗ | 3.6 |
| 2 | ✓ | ✗ | ✓ | ✓ | ✗ | 39.5 |
| 3 | ✓ | ✓ | ✗ | ✓ | ✗ | 39.2 |
| 4 | ✓ | ✓ | ✓ | ✗ | ✗ | 50.1 |
| 5 | ✓ | ✓ | ✓ | ✓ | ✗ | 55.3 |
| **6** | ✓ | ✓ | ✓ | ✓ | ✓ | **70.4** |

### 关键发现

- **路线引导是核心**: 移除 $Q_{\text{route}}$ 后性能骤降至近乎零（3.6），说明路线信息是驾驶行为的基础。
- **适配器大幅提升安全性**: 加入扩散适配器后 HDScore 从 55.3 提升到 70.4（安全场景下约 +15 点），路线完成率提升约 20%。
- **反事实步数越多越好**: 更长的反事实 rollout 步数为 Q 值模型提供更丰富的监督信号，但过长可能偏离参考数据。
- **模态容量影响性能**: 更大的适配器模态数在安全关键场景中带来一致的性能提升。
- **泛化性强**: MPA 在未见场景中的 HDScore 与域内评估相当，验证了框架的泛化能力。

## 亮点与洞察

- **系统性地诊断闭环退化原因**: 清晰地将问题分解为观测错配和目标错配两个维度，针对性设计解决方案。
- **推理时缩放策略**: 首次在 E2E 驾驶中引入类似 LLM 推理时缩放的思路——多候选采样 + 价值模型选择，效果显著。
- **框架通用性**: MPA 可无缝应用于不同预训练 E2E 模型（UniAD、VAD、LTF），一致带来提升。
- **安全关键场景大幅改善**: 在对抗性安全场景中的提升尤其显著（HDScore 从 16→74.7），实用价值高。

## 局限与展望

- 假设 3DGS 在受限轨迹偏差下的渲染可靠性，大偏差时可能失真。
- 价值建模与策略优化目前解耦，未来可探索联合优化。
- 目前仅在 nuScenes 数据集上验证，期望扩展到更多样的驾驶数据集。
- 尚未应用于多模态基础模型（如 VLM），对更严重的分布偏移的处理有待探索。
- 反事实数据生成依赖高质量 3DGS 重建，对场景重建质量有要求。

## 相关工作与启发

- **E2E 自动驾驶**: UniAD、VAD、LTF 等统一感知-预测-规划的框架在开环表现优异但闭环退化严重。
- **反事实数据生成**: 此前工作主要关注行为场景生成，未结合视觉信息；MPA 首次在 E2E 仿真器中系统生成反事实数据。
- **推理时奖励引导**: LLM 领域的推理时缩放思路（如 reward model 引导采样）被首次有效应用于 E2E 驾驶。
- **启发**: 反事实数据 + 推理时 Q 值引导的范式可能适用于其他 sim-to-real 的闭环控制问题。

## 评分

- 新颖性: ⭐⭐⭐⭐ 反事实数据+扩散适配器+Q值引导的组合新颖，推理时缩放在驾驶领域首创
- 实验充分度: ⭐⭐⭐⭐⭐ 三种评估设置（域内/未见/安全关键），完整消融，多基线对比
- 写作质量: ⭐⭐⭐⭐ 问题分析到位，框架清晰，公式化表达严谨
- 价值: ⭐⭐⭐⭐ 对闭环E2E驾驶的实用价值显著，安全关键场景提升巨大

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[NeurIPS 2025\] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)
- [\[CVPR 2025\] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](../../CVPR2025/autonomous_driving/diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)

<!-- RELATED:END -->
