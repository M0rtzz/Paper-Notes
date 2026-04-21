---
title: >-
  [论文解读] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving
description: >-
  [CVPR 2026][自动驾驶][端到端规划] 提出MeanFuser端到端自动驾驶框架，用高斯混合噪声替代离散轨迹词汇表实现连续多模态轨迹建模，通过MeanFlow Identity实现一步采样消除ODE数值误差，并设计ARM模块隐式判断是选择现有proposal还是重构新轨迹，在NAVSIM上以仅RGB输入+ResNet-34骨干达到89.0 PDMS且59 FPS。
tags:
  - CVPR 2026
  - 自动驾驶
  - 端到端规划
  - MeanFlow
  - 高斯混合噪声
  - 一步采样
  - 自适应轨迹重建
---

# MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving

**会议**: CVPR 2026  
**arXiv**: [2602.20060](https://arxiv.org/abs/2602.20060)  
**代码**: https://github.com/wjl2244/MeanFuser (有)  
**领域**: 自动驾驶  
**关键词**: 端到端规划, MeanFlow, 高斯混合噪声, 一步采样, 自适应轨迹重建

## 一句话总结
提出MeanFuser端到端自动驾驶框架，用高斯混合噪声替代离散轨迹词汇表实现连续多模态轨迹建模，通过MeanFlow Identity实现一步采样消除ODE数值误差，并设计ARM模块隐式判断是选择现有proposal还是重构新轨迹，在NAVSIM上以仅RGB输入+ResNet-34骨干达到89.0 PDMS且59 FPS。

## 研究背景与动机

**领域现状**：端到端自动驾驶直接从传感器输入学习到规划轨迹。TransFuser、UniAD、VAD等学习单模态轨迹效果好但无法捕获驾驶行为的多模态本质。VADv2、Hydra-MDP引入轨迹词汇表预测概率分布，但固定词汇表在效率和鲁棒性间存在权衡。DiffusionDrive和GoalFlow将生成模型引入轨迹规划，但前者需要多步采样，后者依赖离散anchor。

**现有痛点**：(1) **离散锚点词汇表的固有限制**——词汇表必须足够大才能覆盖测试时的轨迹分布，但大词汇表拖慢推理速度。当测试场景超出预定义锚点分布时，所有proposal都偏离最优轨迹；(2) **多步采样的计算开销**——flow matching需要多次ODE solver步骤（如GoalFlow需5步）才能达到最优性能，且ODE solver引入数值误差导致采样路径弯曲；(3) **标准高斯噪声的模式坍塌**——vanilla方法从标准高斯采样导致轨迹多样性不足。

**核心矛盾**：如何在不依赖固定离散词汇表的前提下，有效建模多模态驾驶行为，同时保持高推理效率？

**本文目标** (1) 消除对离散轨迹词汇表的依赖；(2) 实现one-step高质量采样；(3) 处理所有采样proposal都不够好的情况。

**切入角度**：将MeanFlow Identity引入端到端规划——MeanFlow直接建模噪声分布和轨迹分布之间的平均速度场而非瞬时速度场，使得单步采样精确无误差。同时用高斯混合模型作为先验分布，每个高斯成分捕获一种驾驶模式。

**核心 idea**：用高斯混合噪声替代锚点、MeanFlow替代多步ODE、自适应重建模块替代评分选择，三管齐下实现快速鲁棒的多模态轨迹规划。

## 方法详解

### 整体框架
MeanFuser由三部分组成：(1) **场景上下文编码器**：图像编码器提取BEV特征 + 车辆状态编码器提取自车信息，辅以地图辅助解码监督；(2) **多模态轨迹采样**：从8分量高斯混合噪声采样，通过轻量MeanFlow网络一步生成多条轨迹proposal；(3) **自适应重建模块(ARM)**：将所有proposal与BEV特征通过交叉注意力融合，输出最终规划轨迹。训练时用标准流匹配损失+ARM重建损失+地图损失。

### 关键设计

1. **高斯混合噪声 (GMN)**:

    - 功能：以连续分布替代离散轨迹词汇表，每个高斯成分捕获一种驾驶模式
    - 核心思路：对训练集所有专家轨迹做归一化处理——计算逐步差分 $\Delta\tau_j$，按全局均值/最大值归一化，然后用K-means将所有归一化轨迹聚类为 $K=8$ 组。每组的均值和标准差参数化一个高斯成分：$p_0 = \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \sigma_k^2 \cdot I)$。推理时从每个成分各采一个噪声点，并行生成8条多模态轨迹。训练时选择距离ground truth最近的高斯成分计算loss
    - 设计动机：标准高斯采样导致模式坍塌，离散锚点无法覆盖连续空间。GMN兼具轨迹先验（聚类中心编码了典型驾驶模式）和连续性（每个高斯的方差允许模式内变化）。有趣的副产品是不同成分自然对应不同驾驶风格（保守3.45m/s→激进9.11m/s），可支持个性化驾驶

2. **MeanFlow Identity 适配端到端规划**:

    - 功能：实现精确的one-step采样，消除ODE solver的数值误差
    - 核心思路：传统flow matching学习瞬时速度场 $v_\theta(z_t, t)$，即使构造了线性概率路径，学到的速度场也不保证产生直线采样路径，需要多步ODE求解。MeanFlow直接学习时间区间内的**平均速度场** $u(z_t, r, t) = \frac{1}{t-r}\int_r^t v(z_\tau,\tau)d\tau$，训练目标通过MeanFlow Identity推导：$u_{\text{tgt}} = v(z_t,t) - (t-r)(v(z_t,t)\partial_z u_\theta + \partial_t u_\theta)$，使用stop-gradient。推理时直接一步完成：$x_1 = x_0 + 1 \cdot u_\theta(x_0, 0, 1)$。训练时使用torch.autograd.functional.jvp高效计算Jacobian-vector product
    - 设计动机：GoalFlow需5步采样才达到最优性能，DiffusionDrive的扩散过程也需迭代。MeanFlow的one-step采样使规划模块推理速度达到434 FPS（GoalFlow仅11 FPS，加速39.45×），且无数值误差

3. **自适应重建模块 (ARM)**:

    - 功能：当所有采样proposal都不够好时，隐式重构更优轨迹
    - 核心思路：将所有候选轨迹 $\{\hat{\tau}_k\}_{k=1}^K$ 编码后与BEV场景特征 $c_{\text{bev}}$ 通过交叉注意力融合，结果送入Projector输出最终轨迹 $\hat{\tau}$。注意力权重隐式学习"选择还是重构"——如果某个proposal足够好，注意力集中于它（相当于选择）；如果都不够好，注意力分散地综合多个proposal的优势重构新轨迹。训练仅用专家轨迹L1监督 $\mathcal{L}_\tau = \|\tau - \hat{\tau}\|_1$
    - 设计动机：Hydra-MDP和WoTE用评估子指标（如PDM Score的子项）打分选择候选轨迹，但这依赖benchmark规则且无法处理所有proposal都差的情况。ARM不依赖任何benchmark规则，仅用专家轨迹监督，且能重构而不仅是选择

### 损失函数 / 训练策略
$\mathcal{L} = \lambda_1 \mathcal{L}_\tau + \lambda_2 \mathcal{L}_{\text{flow}} + \lambda_3 \mathcal{L}_{\text{map}}$，其中flow loss使用L1损失，ARM重建loss也是L1，辅以地图解码语义监督加速收敛。使用AdamW优化器，weight decay 0.1，余弦退火学习率 $2\times10^{-4}$，3 epoch warmup。隐藏维度128（参数量仅54.6M），8个GMN成分各采样1条，共8条轨迹。

## 实验关键数据

### 主实验

| 方法 | 输入 | PDMS↑(v1) | EPDMS↑(v2) | Plan FPS↑ | FPS↑ |
|--------|------|------|----------|------|------|
| TransFuser | C&L | 84.0 | 76.7 | 3934 | 63 |
| GoalFlow | C&L | 85.7 | - | 11 | 10 |
| Hydra-MDP | C&L | 86.5 | 81.4 | 25 | 20 |
| DiffusionDrive | C&L | 88.1 | 88.3 | 75 | 39 |
| WoTE | C&L | 88.3 | - | - | - |
| **MeanFuser** | **C only** | **89.0** | **89.5** | **434** | **59** |

注：MeanFuser仅用RGB相机输入(无LiDAR)就超过所有多模态(C&L)方法。参数量54.6M在所有方法中最小。

### 消融实验

| 配置 | PDMS↑ | N_proposals | P_{L2>0.5}↓ | N_{DAC=0}↓ |
|------|---------|------|------|------|
| DiffusionDrive | 88.1 | 20 | 20.0% | 84 |
| TransFuser(base) | 84.0 | - | - | - |
| + vanilla MeanFlow(ℳ₀) | 87.3(+3.3) | 16 | 40.6% | 143 |
| + GMN(ℳ₁) | 88.2(+0.9) | 16 | 18.5% | 58 |
| + ARM(ℳ₂=MeanFuser) | **89.0(+0.8)** | 17 | 16.9% | 48 |
| + 简单平均(ℳ₃) | 71.2(-17.8) | 17 | 18.0% | 57 |

### 关键发现
- **MeanFlow本身贡献最大(+3.3 PDMS)**：将MLP替换为条件化MeanFlow解码器就有显著提升，验证了flow-based建模轨迹分布的有效性
- **GMN大幅减少DAC=0案例**：ℳ₀有143个场景所有proposal都离开可行驶区域，加GMN后降到58个（比DiffusionDrive的84还少），说明GMN的覆盖能力远超标准高斯和离散锚点
- **简单平均proposal导致灾难性下降(-17.8 PDMS)**：证明采样的轨迹确实捕获了不同模式而非坍塌为单一模式，ARM的"隐式选择/重构"非常必要
- **ARM进一步减少DAC=0从58到48**：说明ARM能在所有proposal都不好时重构出更优轨迹
- **纯视觉超越多模态**：无LiDAR的MeanFuser超过所有Camera+LiDAR方法，说明感知信息不是瓶颈，规划策略才是
- **不同高斯成分自然对应不同驾驶风格**：速度从3.45m/s到9.11m/s，从保守到激进，为个性化驾驶提供了零成本控制接口

## 亮点与洞察
- **GMN的设计非常精妙**——用K-means聚类训练集轨迹然后拟合高斯混合模型，既保留了锚点方法的"模式先验"优势（每个高斯中心是一种典型驾驶模式），又克服了离散锚点"覆盖不全"的致命缺陷（高斯的方差允许连续探索）。这个思路可迁移到机器人操控等任何需要多模态动作生成的场景
- **MeanFlow Identity在规划中的首次应用**消除了flow matching的两大痛点：多步采样慢和数值误差。Plan FPS从GoalFlow的11提升到434（39x加速），使flow-based方法首次在实时性上与MLP直接回归竞争
- **ARM的"不选则构"设计**解决了一个长期被忽视的问题：如果所有候选都不好怎么办？传统选择器只能选最不差的，ARM能综合所有proposal重构新轨迹。这个设计可迁移到任何多候选选择场景

## 局限与展望
- GMN的高斯成分数K=8和混合系数$\pi_k=1$是预定义的，自适应确定K和利用场景上下文预测自适应混合系数可能进一步提升
- ARM通过交叉注意力隐式完成选择/重构，缺乏可解释性——不知道模型是"选了哪个"还是"重构了什么"
- 仅在NAVSIM上评测（非反应式模拟），在更真实的反应式模拟（如nuPlan）和实车上的效果有待验证
- 轨迹规划仅4秒（8个路点），长时域规划场景的适用性需进一步探索
- MeanFlow训练需计算JVP，训练成本与标准flow matching的对比未详细讨论

## 相关工作与启发
- **vs GoalFlow**: GoalFlow用flow matching+目标点引导但需5步采样且依赖离散目标点预测；MeanFuser用MeanFlow一步采样+GMN连续先验，PDMS高3.3且Plan FPS快39.45x
- **vs DiffusionDrive**: DiffusionDrive用扩散+聚类轨迹原型迭代精炼；MeanFuser用MeanFlow+GMN一步生成，PDMS高0.9且快1.55x
- **vs Hydra-MDP**: Hydra-MDP用离散轨迹词汇表+概率预测；MeanFuser用连续GMN替代，性能更好(+2.5 PDMS)且更快(2.65x)
- **vs WoTE**: WoTE引入世界模型评估候选轨迹（依赖benchmark子指标）；MeanFuser用ARM替代评估，不依赖benchmark规则

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个将MeanFlow引入端到端规划+GMN连续先验+ARM重构，三个设计都有创新
- 实验充分度: ⭐⭐⭐⭐ NAVSIMv1/v2全面评估，消融充分，但缺少nuPlan等更复杂基准
- 写作质量: ⭐⭐⭐⭐ 技术细节清晰，预备知识充分，图示直观
- 价值: ⭐⭐⭐⭐⭐ 解决了flow-based规划的关键效率瓶颈，GMN和ARM设计可广泛迁移

<!-- RELATED:START -->

## 相关论文

- [Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](../../AAAI2026/autonomous_driving/drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)
- [CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention](causalvad_de-confounding_end-to-end_autonomous_driving_via_causal_intervention.md)
- [FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](../../AAAI2026/autonomous_driving/fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)
- [DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](../../AAAI2026/autonomous_driving/diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)

<!-- RELATED:END -->
