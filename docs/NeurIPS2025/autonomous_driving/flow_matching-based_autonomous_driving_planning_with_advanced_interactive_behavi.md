---
title: >-
  [论文解读] Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling
description: >-
  [NeurIPS 2025][自动驾驶][自动驾驶规划] 提出 Flow Planner——通过细粒度轨迹 token 化、交互增强时空融合架构和 flow matching + classifier-free guidance 三项协同创新，在 nuPlan Val14 上首次作为纯学习方法突破 90 分大关（90.43），在交互密集的 interPlan 基准上比 Diffusion Planner 高 8.92 分。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 自动驾驶规划
  - Flow Matching
  - 交互行为建模
  - Classifier-Free Guidance
  - 轨迹生成
---

# Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2510.11083](https://arxiv.org/abs/2510.11083)  
**代码**: [https://github.com/DiffusionAD/Flow-Planner](https://github.com/DiffusionAD/Flow-Planner)  
**领域**: autonomous_driving / 规划  
**关键词**: 自动驾驶规划, Flow Matching, 交互行为建模, Classifier-Free Guidance, 轨迹生成  

## 一句话总结
提出 Flow Planner——通过细粒度轨迹 token 化、交互增强时空融合架构和 flow matching + classifier-free guidance 三项协同创新，在 nuPlan Val14 上首次作为纯学习方法突破 90 分大关（90.43），在交互密集的 interPlan 基准上比 Diffusion Planner 高 8.92 分。

## 研究背景与动机

**领域现状**：自动驾驶规划方法分为规则方法（PDM-Closed 等）和学习方法（模仿学习+生成模型）。学习方法近年因 transformer 和扩散模型而快速发展，但在交互场景中仍显不足。

**现有痛点**：(a) 简单堆叠 transformer block 缺乏对异质信息（静态车道 + 动态邻车）的有效融合机制；(b) 训练数据中高质量交互场景稀缺，朴素行为克隆收敛到偏离真实交互行为的分布；(c) 辅助 loss（碰撞惩罚等）需逐案设计且损害训练稳定性；(d) Diffusion Planner 的交互限于最近几辆车，架构无专门融合设计。

**核心矛盾**：有效的交互行为建模需要三个条件同时满足——(i) 表达性强的轨迹表示、(ii) 高效的异质信息融合、(iii) 对条件信号的动态增强以弥补交互数据不足。现有方法最多满足其中一个。

**切入角度**：从数据建模（轨迹 token 化）、模型架构（时空融合）、学习范式（flow matching + CFG）三个维度协同设计。

**核心idea**：Flow Planner = 细粒度轨迹 token + scale-adaptive attention 时空融合 + flow matching CFG 动态增强邻车条件。

## 方法详解

### 整体框架
输入为矢量化场景信息（邻车历史、车道、静态物体、导航），经 MLP-Mixer 编码为 token。自车轨迹通过细粒度 token 化分解为重叠片段。所有 token 在统一潜空间中通过 scale-adaptive attention 做时空融合。训练使用 Bernoulli 条件 masking 的 flow matching loss，推理时用 classifier-free guidance 增强邻车交互条件。

### 关键设计

1. **Fine-grained Trajectory Tokenization**：

    - 功能：将整条轨迹分解为有重叠的片段 token，平衡表达性与一致性。
    - 核心思路：$L$ 个路点的轨迹 $\tau_t$ 分为 $K$ 段，每段 $L_{seg}$ 个点，相邻段重叠 $L_{overlap}$：$F_{ego}^k = \text{MLP}((x_{l^k}, \ldots, x_{r^k}))$，其中 $l^k = (k-1)(L_{seg} - L_{overlap})$。加正弦位置编码后拼接为 $F_{ego} = \text{Concat}(F_{ego}^1, \ldots, F_{ego}^K)$。
    - 重叠区域施加一致性 loss：$\mathcal{L}_{consist} = \frac{1}{K-1} \sum_{k=1}^{K-1} \|\hat{\tau}^{k:k+1} - \hat{\tau}^{k+1:k}\|^2$
    - 设计动机：单 token 表示整条轨迹压缩过度导致场景信息融合不充分；逐时步 token 误差累积严重。重叠片段在二者之间取得平衡。最优段数为 20（消融验证）。

2. **Interaction-enhanced Spatiotemporal Fusion**：

    - 功能：高效融合异质场景 token 以增强交互建模。
    - 核心思路：
        - 先通过分别的 adaptive LayerNorm (adaLN) 将异质特征（lane, neighbor, ego）投射到共享潜空间并注入时步/导航条件。
        - 拼接后用 **scale-adaptive self-attention** 做全局融合：$F_{global} = \text{Softmax}\left(\frac{F_{global}W^Q (F_{global}W^K)^T}{\sqrt{d}} - \lambda \cdot D\right) F_{global}W^V$
        - 其中 $D$ 是 token 间的欧氏距离矩阵，$\lambda$ 是由 token 本身经线性投影生成的可学习感受野缩放因子。距离远的 token 得到更小的 attention score。
        - 融合后分解回模态特定 token，各自经独立 adaLN + FFN 进一步减少模态间隙。
    - 设计动机：vanilla attention 无法有效处理异质信息融合；scale-adaptive attention 让模型根据空间距离自适应关注重要邻车。

3. **Flow Matching + Classifier-Free Guidance**：

    - 功能：通过条件增强的流匹配实现多模态交互行为生成。
    - 条件生成分布：$\tilde{q}(\tau_1|C) \propto q(\tau_1)^{1-\omega} q(\tau_1|C)^{\omega}$，引导速度场：$\tilde{v}_t(\tau_t, t|C) = (1-\omega) v_t(\tau_t, t) + \omega \cdot v_t(\tau_t, t|C)$
    - 训练用 Bernoulli 条件 masking：$\mathcal{L}_{flow} = \mathbb{E}_{t, b \sim \mathcal{B}} \|\tau_\theta(\tau_t, t|(1-b) \cdot C + b \cdot \emptyset) - \tau_1\|^2$
    - 实际只 mask 邻车信息（实验发现对交互建模最关键）。
    - 推理用最优传输路径 + 二阶中点 ODE solver。
    - 设计动机：CFG 让模型同时学到"无条件规划"和"有条件规划"，差异部分就是邻车交互引起的行为变化——inference 时可通过 $\omega$ 放大这个差异来增强交互感知。

### 损失函数
- 最终 loss：$\mathcal{L} = \mathcal{L}_{flow} + \alpha \cdot \mathcal{L}_{consist}$
- 数据增强：对自车当前帧状态随机扰动，用五次多项式插值生成新 GT。

## 实验关键数据

### 主实验——nuPlan 闭环评测

| 类型 | 方法 | Val14 NR | Val14 R | Test14-hard R |
|------|------|---------|--------|--------------|
| 规则 | PDM-Closed | 92.84 | 92.12 | 75.19 |
| 混合 | PDM-Hybrid | 92.77 | 92.11 | 76.07 |
| 学习 | PLUTO (w/o refine.) | 88.89 | 78.11 | 59.74 |
| 学习 | Diffusion Planner | 89.87 | 82.80 | 69.22 |
| **学习** | **Flow Planner** | **90.43** | **83.31** | **70.42** |
| 混合 | Flow Planner w/ refine. | 94.31 | 92.38 | 80.25 |

Val14 上 **90.43** 是首个纯学习方法突破 90 分。

### interPlan 交互基准

| 方法 | Overall | Nudge Around | High Traffic | Jaywalk |
|------|---------|-------------|-------------|---------|
| PlanTF | 47.70 | 49.40 | 58.85 | 33.94 |
| PLUTO | 58.47 | 71.56 | 67.25 | 25.48 |
| Diffusion Planner | 52.90 | 60.48 | 49.71 | 26.20 |
| **Flow Planner** | **61.82** | **72.96** | **67.21** | **43.57** |

比 Diffusion Planner 总分高 8.92，Jaywalk 场景高 17.37——行人横穿是最难预测的交互场景。

### 消融实验

| 配置 | nuPlan Val14 | interPlan |
|------|-------------|-----------|
| Base (vanilla self-attention) | 88.10 | 41.27 |
| + Trajectory Tokenization | 88.33 | 44.14 |
| + Scale-Adaptive Attention | 88.77 | 46.25 |
| + Separate adaLN & FFN | 89.54 | 58.22 |
| + Classifier-Free Guidance | **90.43** | **61.82** |

### CFG Scale 消融

| CFG $\omega$ | Val14 Score |
|-----|-----------|
| 1.65 | 89.64 |
| 1.75 | 90.14 |
| **1.80** | **90.43** |
| 1.85 | 90.00 |
| 1.90 | 89.63 |

### 关键发现
- **Separate adaLN + FFN** 贡献最大：interPlan 从 46.25→58.22（+11.97），说明异质特征融合是交互建模的关键瓶颈。
- **CFG** 在 interPlan 上再提升 3.6 分，验证了推理时动态增强条件信号对交互场景的重要性。
- **轨迹 segment 数为 20** 时最优（Table 5），太少（1）无法建模多模态行为，太多（80）导致 token 负担过重。
- Flow Planner 在无保护左转和行人等交互密集场景中显著优于 Diffusion Planner，case study 显示它能识别后方来车并放弃变道。

## 亮点与洞察
- **CFG 用于自动驾驶规划**是真正的洞察亮点——通过 mask 邻车信息训练 unconditional 分支，推理时 $\omega > 1$ 放大条件影响，等于隐式学到"邻车引起的行为变化"。这比显式碰撞 penalty 更优雅且更有效。
- **细粒度轨迹 token 化 + 重叠一致性 loss** 解决了单 token vs 逐步 token 的两难——既避免了过度压缩又防止了误差累积。
- **Scale-adaptive attention** 引入空间距离先验到 attention score——直觉上正确（远处车影响小），实现上简洁（只需一个可学习标量 + 距离矩阵偏移）。

## 局限与展望
- 依赖 nuPlan 的处理好的感知输入（矢量化），未端到端处理原始传感器数据。
- CFG scale $\omega$ 需要手动调节，缺乏自适应机制。
- 未建模行人/骑行者意图的不确定性——Jaywalk 场景虽然大幅提升但绝对分数仍不高（43.57）。
- Flow Planner w/ refine. 与 Diffusion Planner w/ refine. 性能接近，说明后处理可能掩盖了模型本身的差异。

## 相关工作与启发
- **vs Diffusion Planner**：Diffusion Planner 用 DDPM + joint ego-neighbor 生成，但交互限于 fixed 最近车且架构无专门融合设计；Flow Planner 用 flow matching（更快收敛）+ CFG（动态增强交互）+ scale-adaptive fusion。
- **vs PLUTO**：PLUTO 依赖参考线先验 + 对比 loss + 后处理；Flow Planner 无需任何先验知识即领先。
- **vs PDM-Closed**：规则方法在常规场景优势明显（92.12 R），但在 hard 场景（75.19）不如 Flow Planner（流式学习捕获了分布外交互）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 三项创新协同设计，CFG 用于规划的洞察新颖
- 实验充分度: ⭐⭐⭐⭐⭐ nuPlan 三基准 + interPlan + 详尽消融 + case study
- 写作质量: ⭐⭐⭐⭐ 结构清晰，case study 直观
- 价值: ⭐⭐⭐⭐⭐ 首个纯学习 90+ 的里程碑 + 交互建模新范式

<!-- RELATED:START -->

## 相关论文

- [Towards Foundational LiDAR World Models with Efficient Latent Flow Matching](towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)
- [Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)
- [UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning](unimotion_a_unified_motion_framework_for_simulation_prediction_and_planning.md)
- [DriveGPT: Scaling Autoregressive Behavior Models for Driving](../../ICML2025/autonomous_driving/drivegpt_scaling_autoregressive_behavior_models_for_driving.md)

<!-- RELATED:END -->
