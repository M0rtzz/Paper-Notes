---
title: >-
  [论文解读] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective
description: >-
  [AAAI 2026][视频理解][非马尔可夫决策] 提出 LIBERO-Mem 基准（10 个非马尔可夫机器人操控任务）和 Embodied-SlotSSM 框架（结合 Slot Attention 和状态空间模型的物体中心记忆 VLA），解决视觉运动策略在部分可观测、需要物体级历史推理的长期任务中的失败问题。
tags:
  - AAAI 2026
  - 视频理解
  - 非马尔可夫决策
  - 物体中心记忆
  - 状态空间模型
  - VLA
  - 机器人基准
---

# Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective

**会议**: AAAI 2026  
**arXiv**: [2511.11478](https://arxiv.org/abs/2511.11478)  
**代码**: [libero-mem.github.io](https://libero-mem.github.io)  
**领域**: 视频理解 / 机器人操作  
**关键词**: 非马尔可夫决策, 物体中心记忆, 状态空间模型, VLA, 机器人基准

## 一句话总结

提出 LIBERO-Mem 基准（10 个非马尔可夫机器人操控任务）和 Embodied-SlotSSM 框架（结合 Slot Attention 和状态空间模型的物体中心记忆 VLA），解决视觉运动策略在部分可观测、需要物体级历史推理的长期任务中的失败问题。

## 研究背景与动机

**领域现状**：人类能轻松回忆与特定物体的过去交互（如盐瓶放在哪里、是否已经往汤里撒过盐），从而精确执行多步骤长期任务。然而，当前机器人视觉运动策略（如 OpenVLA、Octo、RT-1/2）通常仅依赖最近的感官输入来决策，缺乏编码和回忆**物体级历史**的机制。

**现有痛点**：

**马尔可夫假设的局限**：大多数 VLA 模型假设当前观测足以预测最优动作，但在涉及重复步骤、视觉相似物体、长期时间依赖的任务中，这一假设失效——相同的视觉输入可能对应不同的语义状态（如碗在盘子上 vs 碗刚被放回盘子上）。

**基准不足**：现有基准（如 RLBench、LIBERO、RoboCasa）主要在马尔可夫假设下构建，MemoryBench 和 MIKASA-Robo 虽关注记忆，但缺乏物体级歧义和时间扩展的系统压力测试。

**Token 扩展难题**：OpenVLA 使用 256 个密集 token 编码视频序列，物体中心 VLA 用 16 个 slot token，但 token 数随 slot 和序列维度**线性增长**，在长期任务中（数百帧）变得不可行。

**核心矛盾**：当两个时间点 $t_1$ 和 $t_2$ 的视觉观测 $\mathbf{v}_{t_1} \approx \mathbf{v}_{t_2}$ 但需要不同动作时（$P(\mathbf{a}_{t_1}|\mathbf{v}_{1:t_1},l) \neq P(\mathbf{a}_{t_2}|\mathbf{v}_{1:t_2},l)$），纯反应式策略必然失败。此即**物体级部分可观测马尔可夫决策过程（POMDP）**。

**切入角度**：从物体中心学习和状态空间模型出发，设计结构化、持久化的记忆表示，在保持计算可行性的同时支持长期非马尔可夫推理。

## 方法详解

### 整体框架

Embodied-SlotSSM 包含三个核心组件：（1）**Slot Attention** 将密集视觉特征分解为离散物体中心 token；（2）**SlotSSM** 通过基于 slot 的状态空间模型追踪物体时间动态；（3）**Relation Encoder + LLM Action Decoder** 将物体记忆与当前场景对齐用于动作预测。

### 关键设计

#### 1. **LIBERO-Mem 基准**

设计了 10 个任务跨越四个物体中心记忆维度：

- **物体运动（OM）**：机器人需记住上一次动作（拿起还是放下）才能正确行动（T1, T2）
- **物体序列（OS）**：成功取决于记住已操作物体的次数，视觉线索不足以判断（T3-T6，重复 3/5/7 次拿起放下）
- **多物体序列（OR）**：需追踪物体交互的时间顺序和关系（T7-T8，交换碗的位置）
- **多物体遮挡（OO）**：被遮挡物体要求机器人依赖过去放置的记忆来识别目标（T9-T10）

**关键特征**（对比 MemoryBench/MIKASA-Robo）：
- 非马尔可夫观测 ✓
- 长期轨迹（200-700帧） ✓
- **子目标感知评估** ✓（独有，支持细粒度进度评估）
- **物体身份歧义** ✓（独有，视觉相同的碗/盘仅靠 asset ID 区分）
- **时间扩展压力测试** ✓（独有）

每个任务收集 120 条轨迹（100 训练 + 20 验证），通过键盘控制+多键追踪采集平滑演示。

#### 2. **Slot Attention 用于物体定位**

将密集视觉嵌入 $\mathbf{v}_t \in \mathbb{R}^{K \times D_{\text{enc}}}$ 分解为 $N$ 个物体中心 token $\mathbf{s}_t = \{\mathbf{s}_t^1, ..., \mathbf{s}_t^N\}$，$N=16$。通过注意力和 GRU 循环更新迭代绑定空间特征到固定数量的可学习物体查询。

**时间一致性初始化**：$t=0$ 时随机初始化 slot，$t>0$ 时用前一帧的最终 slot 输出初始化当前帧 slot，实现跨时间的 slot 身份传播和持续物体追踪。

$$\mathbf{s}_t^{(0)} = \begin{cases} \text{RandomInit}() & t=0 \\ \mathbf{s}_{t-1}^{(T)} & t>0 \end{cases}$$

**时间对比损失**：在固定时间窗口内，以同一 slot 在邻近帧的表示为正样本，以不同视频/不同位置的 slot 为负样本，通过对比学习增强时间一致性。

#### 3. **SlotSSM 瞬态记忆**

基于 Mamba 的状态空间模型，设计分解为 block-diagonal 的 $\overline{A}_t$、$\overline{B}_t$、$C_t$ 矩阵，每个块仅条件化于对应 slot 输入：

$$\mathbf{h}_t^k = \overline{A}(\mathbf{s}_t^k)\mathbf{h}_{t-1}^k + \overline{B}(\mathbf{s}_t^k)\mathbf{s}_t^k$$

**窗口预测**：SlotSSM 不仅预测下一步，而是预测以当前时间步为中心的 $P = p+q$ 步窗口内的静态潜在表示（过去 $p$ 步到未来 $q$ 步），同时学习前向动态和后向时间一致性。

**核心命题**：当 $k$ 个物体在时间 $t$ 视觉不可区分（$z_t^{(i)} \approx z_t^{(j)}$）时，策略 $\pi(a_t|h_t)$ 必须条件化于物体特定的历史 $\mu_t^{(j)}$ 才能完成个体化——这正是 SlotSSM 提供的能力。

#### 4. **Slot 条件化的动作解码**

**Slot Fusion 模块**：整合当前 slot $\mathbf{s}_t^{(j)}$、预测的下一 slot $\hat{\mathbf{s}}_{t+1}^{(j)}$ 和 oracle 子目标嵌入 $\mathbf{g}_t^{(j)}$，产生动态潜变量 $\mathbf{d}_t^{(j)}$。

**Relation Encoder**：在 slot 潜变量和原始视觉特征之间做交叉注意力，产生 16 个关系 token $\{\mathbf{r}_t^{(j)}\}_{j=1}^L$，实现上下文感知的物体状态和交互推理。

**动作预测**：
$$\hat{\mathbf{a}}_t \sim P_\theta(\mathbf{a}_t | \{\mathbf{r}_t^{(j)}\}, \{\mathbf{d}_t^{(j)}\}, l)$$

### 损失函数 / 训练策略

结合 Slot Attention 的重建损失、时间对比损失、SlotSSM 的窗口预测损失、以及 VLA head 的动作预测损失（交叉熵）。当前 Naive E-SlotSSM 版本使用 oracle 文本子目标嵌入（如 "bowl 1 on plate 3"）作为进度监控。

## 实验关键数据

### 主实验

**LIBERO-Goal（一般马尔可夫任务）成功率**：

| 方法 | Token数 | bowl in stove | bowl on plate | mid drawer | top drawer→bowl | **平均** |
|------|---------|---------------|---------------|------------|-----------------|---------|
| SlotVLA (h=1) | 16 | 45% | 0% | 5% | 0% | 32% |
| SlotVLA (h=8) | 128 | 95% | 90% | 25% | 65% | 75.5% |
| **Naive E-SlotSSM** | 32 | **100%** | **90%** | **45%** | **70%** | **83.0%** |

**LIBERO-Mem（非马尔可夫 POMDP 任务）子目标完成率**：

| 任务 | π₀ (h=1) | SlotVLA (h=1) | SlotVLA (h=8) | **Naive E-SlotSSM** |
|------|----------|---------------|---------------|---------------------|
| T1 (拿放1次) | 50.0% | 0% | 50.0% | **50.0%** |
| T3 (拿放3次) | 0% | 0% | 0% | **33.3%** |
| T5 (拿放5次) | 0% | 0% | 0% | **14.3%** |
| T9 (碗入篮+移篮) | 0% | 0% | 0% | **30%** |
| T10 (碗入篮+移空篮) | 0% | 0% | 0% | **20%** |
| **平均** | 5.0% | 0% | 5.0% | **14.8%** |

### 消融实验

论文的消融主要通过**对比不同方法在不同任务维度上的表现**来隐式展示各组件的贡献：

| 对比维度 | SlotVLA (h=8) | Naive E-SlotSSM | 说明 |
|---------|---------------|-----------------|------|
| 一般任务平均 | 75.5% | **83.0%** | SSM记忆 +7.5% |
| POMDP任务平均 | 5.0% | **14.8%** | 结构化记忆关键 |
| Token 效率 | 128 tokens | 32 tokens | 4x 压缩 |
| 长期重复任务(T3-T6) | 全部0% | 部分成功 | 持久记忆生效 |

### 关键发现

1. **密集 token 和简单上下文扩展在 POMDP 中失败**：π₀（256 token）和 SlotVLA (h=8, 128 token) 在 LIBERO-Mem 上平均仅 5.0%，说明单纯增加帧数不能解决非马尔可夫问题。
2. **物体中心记忆提供强归纳偏置**：Embodied-SlotSSM 通过结构化 slot 追踪物体身份和状态，在 POMDP 任务上提升约 3x（5%→14.8%）。
3. **Slot 可视化证实物体恒常性**：可视化显示模型在抓取和放置全过程中对目标物体（碗、夹爪）保持一致的注意力。
4. **Token 扩展问题**：拼接方法在相同视觉状态但不同方向（抬起/放下）时容易混淆，而 SlotSSM 方法能正确执行。
5. **绝对性能仍然较低**（14.8%），主要受限于 oracle 子目标依赖。

## 亮点与洞察

1. **问题定义清晰有力**：明确将非马尔可夫机器人操控形式化为物体级 POMDP，用公式 $P(\mathbf{a}_{t_1}|\mathbf{v}_{1:t_1},l) \neq P(\mathbf{a}_{t_2}|\mathbf{v}_{1:t_2},l)$ 且 $\mathbf{v}_{t_1} \approx \mathbf{v}_{t_2}$ 精确描述马尔可夫假设的失效条件。
2. **基准设计巧妙**：通过视觉相同的碗/盘制造物体身份歧义，通过重复拿放制造序列记忆需求，简单但直击痛点。
3. **SlotSSM 的窗口预测设计**：预测过去+未来的 slot 表示而非单步，既支持前向动态预测又通过后向重建增强时间一致性。
4. **子目标感知评估**：超越简单成功/失败的二元评估，支持细粒度进度评估（完成了几个子目标）。
5. **计算效率的理论分析**：SlotSSM 仅需 32 token vs OpenVLA 256 / SlotVLA 128，实现 4-8x 压缩。

## 局限与展望

1. **Oracle 子目标依赖**：Naive E-SlotSSM 依赖预言机文本子目标嵌入（如 "bowl 1 on plate 3"），无法自主发现子目标，这是最大局限。
2. **仍为仿真环境**：LIBERO-Mem 未扩展到真实物理环境，迁移性能未知。
3. **绝对性能较低**：POMDP 任务平均 14.8%，作为"弱基线"距离实用仍有较大差距。
4. **固定 slot 数量**：K=16 的固定设计可能无法适应物体数量变化较大的场景。
5. **任务设计相对单一**：主要是拿放和交换任务，未涉及更复杂的长期操作（如烹饪、组装）。

## 相关工作与启发

- 将认知科学中的"物体恒常性"理论引入机器人操作是有意义的研究方向，SlotSSM 为此提供了计算实现路径。
- 与 MemoryBench 和 MIKASA-Robo 相比，LIBERO-Mem 的独特贡献在于物体身份歧义和时间扩展压力测试，这是评估"真正记忆"vs "短期感知"的关键。
- SlotSSM（基于 Mamba）的 block-diagonal 设计将全局 SSM 分解为物体独立的局部 SSM，是一种优雅的结构化归纳偏置。

## 评分

- 新颖性: ⭐⭐⭐⭐ （LIBERO-Mem 基准+Embodied-SlotSSM 框架，问题定义和解决思路均新颖）
- 实验充分度: ⭐⭐⭐ （多基线对比充分，但缺少组件级消融和真实环境实验）
- 写作质量: ⭐⭐⭐⭐ （问题形式化清晰，框架描述详尽，可视化有说服力）
- 价值: ⭐⭐⭐⭐ （为非马尔可夫机器人操控开辟新方向，基准有长期价值）

<!-- RELATED:START -->

## 相关论文

- [Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](../../CVPR2026/video_understanding/out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)
- [Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning](../../CVPR2026/video_understanding/reconstruction-guided_slot_curriculum_addressing_object_over-fragmentation_in_vi.md)
- [SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](../../CVPR2026/video_understanding/slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)
- [KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals](kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)
- [MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](state-space_hierarchical_compression_with_gated_attention_an.md)

<!-- RELATED:END -->
