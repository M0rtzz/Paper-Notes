---
title: >-
  [论文解读] When would Vision-Proprioception Policies Fail in Robotic Manipulation?
description: >-
  [ICLR 2026][机器人][视觉-本体感觉策略] 揭示视觉-本体感觉操作策略在运动转换阶段（motion-transition phases）会失效的原因——本体感觉信号在优化中占主导导致视觉学习被抑制，并提出Gradient Adjustment with Phase-guidance (GAP)算法，通过自适应调低本体感觉梯度来恢复视觉模态的学习，在仿真和真实环境中均显著提升策略的泛化性。
tags:
  - ICLR 2026
  - 机器人
  - 视觉-本体感觉策略
  - 模态时序性
  - 梯度调整
  - 运动转换阶段
  - 机器人操作
---

# When would Vision-Proprioception Policies Fail in Robotic Manipulation?

**会议**: ICLR 2026  
**arXiv**: [2602.12032](https://arxiv.org/abs/2602.12032)  
**代码**: [Project Page](https://gewu-lab.github.io/GAP/)  
**领域**: Robotics / Multimodal Policy Learning  
**关键词**: 视觉-本体感觉策略, 模态时序性, 梯度调整, 运动转换阶段, 机器人操作

## 一句话总结
揭示视觉-本体感觉操作策略在运动转换阶段（motion-transition phases）会失效的原因——本体感觉信号在优化中占主导导致视觉学习被抑制，并提出Gradient Adjustment with Phase-guidance (GAP)算法，通过自适应调低本体感觉梯度来恢复视觉模态的学习，在仿真和真实环境中均显著提升策略的泛化性。

## 研究背景与动机

本体感觉（proprioception）信息对精确的伺服控制至关重要，它提供了机器人的实时关节状态。在学习型操作策略中，将本体感觉与视觉结合使用被普遍认为能增强策略在复杂任务中的表现。然而，现有研究报告了**矛盾的发现**：
- HPT证明视觉+本体感觉明显优于纯视觉
- Octo发现加入本体感觉反而导致性能变差

**核心矛盾**：本体感觉信息理论上应该是有益的补充，但在实践中视觉-本体感觉策略的泛化性常常不如纯视觉策略。**这一矛盾的根源是什么？何时会发生？**

**关键发现**：通过时序控制实验（temporally controlled experiments），作者发现问题出在**运动转换阶段**（motion-transition phases）。在操作任务中，机器人的运动可以分为"运动一致阶段"（如持续向前移动）和"运动转换阶段"（如需要定位新目标并改变运动方向）。

- **运动一致阶段**：本体感觉信号有效，策略表现正常
- **运动转换阶段**：视觉模态应发挥关键作用（定位目标），但视觉-本体感觉策略中的视觉模态学习被抑制

**原因分析**：从优化角度看，本体感觉信号简洁且低维，在训练中提供更快的损失下降，导致优化被本体感觉信号主导。视觉信号虽然包含目标定位等关键信息，但由于像素级的变化相比本体感觉信号更微妙，其学习被压制（模态竞争/模态惰性问题）。

## 方法详解

### 整体框架
GAP算法的pipeline如下：
1. **输入**：专家演示轨迹（含视觉+本体感觉信息）
2. **运动表示定义**：基于本体感觉信号定义机器人运动
3. **运动转换阶段估计**：使用变点检测(CPD) + LSTM预测每个时间步属于"运动转换阶段"的概率 $\rho$
4. **梯度调整策略学习**：在训练中，根据 $\rho$ 值降低本体感觉分支的梯度幅度
5. **输出**：泛化性更强的视觉-本体感觉策略

### 关键设计

1. **运动表示（Motion Representation）**：基于本体感觉信号定义机器人运动。时间步 $i$ 到 $j$ 之间的运动定义为 $m_{i:j} = \{p_{i:j}, \theta_{i:j}, g_{i:j}\}$，其中 $p_{i:j}$ 是夹爪位置变化，$\theta_{i:j}$ 是朝向变化，$g_{i:j}$ 是夹爪开合度变化。这三个维度完整描述了机器人臂的运动。

2. **变点检测（Change Point Detection, CPD）分割**：使用动态规划来识别轨迹中运动方向发生根本变化的时间点。定义了运动一致性距离：
   $$d(m_{t_1:t_2}, m_{i:i+1}) = -\cos(p_{t_1:t_2}, p_{i:i+1}) - \alpha\cos(\theta_{t_1:t_2}, \theta_{i:i+1}) - \beta(\text{sgn}(g_{t_1:t_2}) == \text{sgn}(g_{i:i+1}))$$
   其中 $\alpha=1$, $\beta=2\times10^{-3}$ 平衡了位置、朝向和夹爪开合三种运动的贡献。CPD最小化总代价将轨迹分割为运动一致的阶段。

3. **LSTM时序建模预测转换概率**：CPD输出的是离散的分割点，但运动转换是连续过程。因此使用LSTM网络对本体感觉的时间差分 $\Delta s_i = s_{i+1} - s_i$ 进行建模，预测每个时间步属于运动转换阶段的连续概率 $\rho_i \in [0,1]$。LSTM以CPD输出作为监督信号，并对转换点附近的时间步降低惩罚，以更好捕捉渐变的转换过程。这比直接使用离散CPD标签效果更好（消融实验验证）。

4. **梯度调整（Gradient Adjustment）**：核心技术——在训练的每个epoch中，对本体感觉特征提取器的参数 $\omega_s$ 进行调制：
   $$\omega_s^{j+1} = \omega_s^j - \lambda \cdot (1-\rho) \cdot \eta \nabla_{\omega_s^j} \mathcal{L}_{BC}(\omega_s^j)$$
   其中 $\lambda=0.3$ 控制调整程度，$\rho$ 是转换概率。当 $\rho$ 高时（运动转换阶段），本体感觉的梯度被大幅削弱，迫使网络更多地依赖视觉信号来学习这些阶段的行为。这是一种**细粒度的、与阶段相关的模态平衡策略**。

5. **仅在训练早期应用GAP**：梯度调整仅在前50个epoch（总100个epoch）中应用。这避免了过度调制导致的训练不稳定和策略崩溃。消融实验表明这一选择是鲁棒的。

### 损失函数 / 训练策略
- 行为克隆（Behavior Cloning）范式，使用MSE损失作为基准
- 视觉分支：ResNet-18 → 512维表示 → 4层temporal transformer
- 本体感觉分支：3层MLP
- 两个分支的特征通过拼接（concatenation）融合，送入3层MLP policy head输出动作序列（长度L=9）
- 训练：Adam优化器，学习率3e-4，batch size 128，单卡RTX 3090，100 epochs
- Meta-World用100条专家演示，RoboSuite用500条合成轨迹

## 实验关键数据

### 主实验（仿真+真实环境）

| 方法 | Meta-World (4任务平均) | RoboSuite (4任务平均) | 说明 |
|------|------------|------------|------|
| Vision-only | 80.5% | 64.6% | 纯视觉基线 |
| Concatenation | 71.8% | 53.8% | 视觉-本体感觉，性能**下降** |
| MS-Bot | 83.5% | 69.4% | 使用语义阶段信息 |
| Aux Loss | 83.5% | 56.3% | 辅助视觉预测损失 |
| Mask | 84.5% | 62.0% | 随机掩码本体感觉 |
| **GAP (Ours)** | **88.4%** | **73.0%** | **全面最优** |

真实世界实验（20次rollout）：

| 方法 | 单臂3任务 | 双臂3任务 |
|------|----------|----------|
| Vision-only | 41/60 | 35/60 |
| Concatenation | 28/60 | 24/60 |
| **GAP (Ours)** | **50/60** | **49/60** |

### OOD泛化实验

| 方法 | assembly | bin-picking | stack | threading | cube | handover |
|------|----------|-------------|-------|-----------|------|----------|
| Vision-only | 78% | 59% | 63% | 32% | 12/20 | 12/20 |
| Concatenation | 62% | 32% | 49% | 28% | 7/20 | 9/20 |
| **GAP** | **88%** | **67%** | **72%** | **49%** | **15/20** | **15/20** |

### VLA模型兼容性（Octo微调）

| 方法 | disassemble | push-wall | put hammer | threading |
|------|-------------|-----------|------------|-----------|
| Octo-V (视觉only) | 95% | 77% | 92% | 69% |
| Octo-VP (视觉+本体感觉) | 82% | 65% | 88% | 57% |
| Octo-VP + **GAP** | **100%** | **85%** | **97%** | **78%** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Human标注 vs CPD | GAP (CPD) 更优 | 自动检测比人工更全面 |
| HDBSCAN聚类 | 性能下降 | 破坏了轨迹时序结构 |
| CoTPC | 次于GAP | 简单余弦距离不够 |
| Fixed ρ vs LSTM | LSTM显著更优 | 连续概率比离散标签好 |
| λ=0.1/0.2/0.3/0.4/0.5 | 0.2-0.4均可 | 对λ不敏感 |
| 应用GAP的epoch数 | 50 epochs最优 | 太少不够，太多会过度抑制 |

### 关键发现
- 标准的视觉-本体感觉拼接策略比纯视觉**一致性地更差**（平均下降约15%），验证了OOD泛化问题的普遍性
- 介入实验精确定位了问题：**运动转换阶段**的视觉模态被抑制，导致策略无法有效定位目标
- GAP通过梯度调整使视觉分支恢复学习，Linear-probing实验直接证实了视觉特征质量的提升（如assembly任务从61%到74%）
- GAP兼容多种模态融合方式（Concatenation、Summation、FiLM）和策略架构（MLP head、Diffusion head、VLA/Octo）
- 在真实世界的单臂和双臂任务中均有效，证明了方法的实际部署价值

## 亮点与洞察
- **诊断精准**：通过时序控制介入实验，精确定位了"运动转换阶段"这一关键失败模式，而非笼统地说"多模态不好"
- **优化视角深刻**：从梯度竞争/模态惰性的角度解释了视觉学习被抑制的机制，连接了多模态学习的理论分析
- **方法简洁有效**：仅修改本体感觉分支的梯度幅度——不改架构、不加模块、不引入额外损失
- **适配性强**：兼容CNN/Transformer/Diffusion/VLA多种架构，单臂/双臂，仿真/真实环境
- **消融极其充分**：运动检测方法、LSTM vs替代方案、超参数α/β/λ/训练阶段数全部消融
- **反直觉发现**：本体感觉本应帮助机器人，但在深度学习优化中反而抑制了视觉学习——揭示了多模态学习中一个被忽视的陷阱

## 局限与展望
- 所有实验均在单一具身（single embodiment）上进行，未在跨具身数据集上验证
- CPD的运动一致性距离依赖手工设计的度量（余弦相似度+符号函数），可能不适用于所有类型的操作任务
- LSTM的训练依赖CPD提供的标签，如果CPD分割有系统性偏差，LSTM也会受到影响
- GAP仅在训练早期应用（前50/100 epochs），这个阈值在不同任务和数据规模下可能需要调整
- 未探索更复杂的模态融合策略（如attention-based fusion）与GAP的结合
- 梯度调整的幅度对所有时间步使用相同的λ，可以考虑自适应λ
- 仅关注了视觉和本体感觉两种模态，未考虑触觉（tactile）等额外模态

## 相关工作与启发
- **模态竞争/模态惰性**（Huang et al. 2022, Fan et al. 2023）：多模态学习中一个模态主导优化的普遍问题，GAP是在机器人操作领域的具体解法
- **HPT (Wang et al. 2024)**：展示了本体感觉的积极效果
- **Octo**：报告了本体感觉的负面效果，本文解释并解决了这一矛盾
- **Modality Temporality (Feng et al.)**：提出了模态重要性随时间变化的概念，本文将其特化为"运动转换阶段"
- **Diffusion Policy (Chi et al. 2023)**：GAP与diffusion head兼容且效果更好
- 启发：梯度调整的思路可推广到任何一种模态在优化中过度主导的多模态学习场景

## 评分
- 新颖性: ⭐⭐⭐⭐ （诊断视角新颖，方法虽简洁但有效）
- 实验充分度: ⭐⭐⭐⭐⭐ （14+任务、多架构、仿真+真实、全面消融、OOD、VLA兼容性）
- 写作质量: ⭐⭐⭐⭐ （问题驱动，逻辑清晰，图表丰富）
- 价值: ⭐⭐⭐⭐⭐ （解决了机器人操作中一个实际且普遍的问题）

<!-- RELATED:START -->

## 相关论文

- [MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](../../CVPR2025/robotics/roboground_robotic_manipulation_with_grounded_vision-language_priors.md)
- [When Agents Persuade: Propaganda Generation and Mitigation in LLMs](when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)
- [RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation](robointer_a_holistic_intermediate_representation_suite_towards_robotic_manipulat.md)
- [TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)

<!-- RELATED:END -->
