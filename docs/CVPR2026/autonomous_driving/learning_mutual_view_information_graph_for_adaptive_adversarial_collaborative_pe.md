---
title: >-
  [论文解读] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception
description: >-
  [CVPR 2026][自动驾驶][协作感知安全] 提出 MVIG 攻击框架，通过将不同防御型协作感知系统的脆弱性统一建模为互视图信息图(Mutual View Information Graph)，结合时序图学习与熵感知漏洞搜索，实现自适应的伪造攻击，使防御成功率最高下降 62%。
tags:
  - CVPR 2026
  - 自动驾驶
  - 协作感知安全
  - 对抗攻击
  - 图神经网络
  - 时序建模
---

# Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception

**会议**: CVPR 2026  
**arXiv**: [2602.19596](https://arxiv.org/abs/2602.19596)  
**代码**: [有](https://github.com/yihangtao/MVIG)  
**领域**: 自动驾驶  
**关键词**: 协作感知安全, 对抗攻击, 图神经网络, 时序建模, 自动驾驶

## 一句话总结

提出 MVIG 攻击框架，通过将不同防御型协作感知系统的脆弱性统一建模为互视图信息图(Mutual View Information Graph)，结合时序图学习与熵感知漏洞搜索，实现自适应的伪造攻击，使防御成功率最高下降 62%。

## 研究背景与动机

### 1. 领域现状
协作感知(Collaborative Perception, CP)允许联网自动驾驶车辆(CAV)之间共享感知数据（如特征图），扩展单车的视野范围并解决遮挡问题。现有防御系统如 ROBOSAC、CP-Guard、MADE、CAD 等通过共识验证来对抗恶意代理的攻击。

### 2. 痛点
现有防御型 CP 系统存在两个关键弱点：
- **缺乏对系统化时空优化攻击的鲁棒性**：现有攻击方法没有系统性地确定何时何地发起伪造攻击，但攻击者完全可以针对多车共识验证机制开发更有针对性的策略
- **防御过程中无意泄露脆弱性知识**：CAV 之间交换的协作数据（特征图、占用图等）天然嵌入了关于周围环境的隐式置信度信息，攻击者可利用这些信息识别集体不确定性区域

### 3. 核心矛盾
CP 系统的安全防御依赖于多车共识验证，但验证过程中交换的数据本身在暴露系统薄弱点，形成"防御即泄密"的悖论。

### 4. 要解决什么
设计一种自适应对抗攻击框架，能够从不同防御型 CP 系统中学习其脆弱性，自动优化攻击位置、时机和持续性，且对不同防御配置具有通用性。

### 5. 切入角度
将不同防御系统泄露的信息统一建模为图结构（MVIG），利用图的频谱特性和时序演化来发现脆弱区域。

### 6. 核心 idea
构建互视图信息图(MVIG)，以 CAV 为节点、互信息为边权重，编码多车感知的一致性与分歧；通过时序图学习预测未来脆弱区域，用熵感知搜索优化攻击策略。

## 方法详解

### 整体框架

MVIG 攻击框架包含四个阶段：
1. **MVIG 构建**：从历史协作数据中提取各 CAV 的占用信息，构建加权无向图
2. **MVIGNet 预测**：通过图卷积+GRU 时序建模，预测未来帧的脆弱性评分图
3. **熵感知漏洞搜索**：优化攻击位置、时机和持续性
4. **交替优化**：解耦掩码预测与扰动优化，PGD 生成特征扰动

### 关键设计

#### 设计1：互视图信息图 (MVIG) 构建
- **功能**：将多 CAV 的感知状态建模为统一的图表示
- **核心思路**：每个 CAV 作为节点，特征向量由三部分组成：(1) 基本占用分布 $\mathbf{h}_i^{\text{basic}}$（free/occupied/unknown 三态频率）；(2) 位姿信息 $\mathbf{h}_i^{\text{pos}}$；(3) 多尺度空间上下文特征 $\mathbf{h}_i^{\text{spatial}}$。边权重用互信息衡量两车感知的位置级一致性：$\mathbf{W}_{ij} = \mathbb{E}_{(x,y)}\left[\sum_{a,b} p_{ij}(a,b) \log \frac{p_{ij}(a,b)}{p_i(a) p_j(b)}\right]$
- **设计动机**：高互信息 → 感知一致区域（难以攻击）；低互信息 → 感知冲突/不确定区域（攻击窗口）。统一图表示使框架可自适应不同防御系统

#### 设计2：MVIGNet 时序图学习
- **功能**：从历史 MVIG 序列预测未来帧的脆弱区域
- **核心思路**：三阶段架构——(1) 图卷积层聚合邻居特征（边权重调制消息传递）；(2) GRU 提取时序模式 $\mathbf{z}^{t+m} = \Phi_{\text{GRU}}(\{\mathbf{f}^\tau\})$；(3) FCN 评分头生成脆弱性评分图 $\mathbf{S}_{t+m} \in [0,1]^{H \times W}$
- **设计动机**：攻击需要提前预判（m=2帧延迟），时序建模能捕捉车辆运动带来的覆盖盲区动态变化

#### 设计3：熵感知漏洞搜索与攻击持续性
- **功能**：优化多帧攻击的位置选择与持续/终止决策
- **核心思路**：用 fabrication risk map 对攻击位置进行概率采样，通过持续性阈值 $\eta$ 控制攻击延续：$\mathcal{C}_{t+m+j} = \mathbb{I}\left[\mathbb{E}_{(x,y) \in \mathcal{N}(x_c,y_c)}[\tilde{\mathbf{S}}_{t+m}(x,y)] \geq \eta\right]$
- **设计动机**：过长的持续攻击会暴露时序异常，智能终止可在低伪造概率区域主动放弃，大幅降低被检测率

#### 设计4：自适应防御信息利用
- **功能**：根据防御系统是否共享占用图，自适应获取 MVIG 输入
- **核心思路**：对 CAD 等共享占用图的系统直接利用；对无占用图共享的系统（ROBOSAC、CP-Guard 等），用盲区分割算法 (BRS) 从特征图估计近似占用网格
- **设计动机**：实现跨防御配置的通用攻击能力

### 损失函数 / 训练策略

**扰动优化（PGD）**：
- Spoofing：$\phi(\boldsymbol{\delta}, M_t^*) = \sum_{b \in B'} \text{IoU}(b, b_t) \cdot \log(b_\sigma)$，最大化伪造目标的置信度
- Removal：取负号，最小化真实目标的检测置信度
- 约束：$\|\delta\|_\infty \leq 1.0$，PGD 迭代 5 次，步长 0.01

**掩码优化（MVIGNet）**：三目标联合损失
1. 攻击有效性损失：最大化伪造置信度，惩罚远离受害者的位置
2. 框区分损失：最小化与已有检测结果的重叠
3. 防御规避损失：避开多车占用图中的矛盾区域

两阶段解耦训练：MVIGNet 使用 PGD 生成的检测框计算损失，但不反向传播梯度到 PGD。

## 实验关键数据

### 主实验

**数据集**：OPV2V（训练）、Adv-OPV2V（测试，300场景 × spoofing/removal）

| 攻击方法 | 无防御 ASR | ROBOSAC DSR | CP-Guard DSR | GCP DSR | CAD DSR |
|---------|-----------|-------------|-------------|---------|---------|
| Basic [31] | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| RC spoof [39] | 92.4 | 12.0 | 18.1 | 12.2 | 83.5 |
| BAC [29] | 99.2 | 23.0 | 37.0 | 28.0 | 90.5 |
| **MVIG spoof** | 94.3 | **14.8** | **17.2** | **13.0** | **32.0** (-62%) |

| 攻击方法 | ROBOSAC DSR | CP-Guard DSR | GCP DSR | CAD DSR |
|---------|-------------|-------------|---------|---------|
| RC remove [39] | 14.2 | 21.4 | 15.0 | 90.1 |
| **MVIG remove** | **12.2** (-14%) | 21.3 | **14.2** (-5%) | **78.2** (-13%) |

MVIG 对 CAD 防御效果最为突出：spoof 模式下 DSR 从 83.5% 降至 32.0%（↓62%）。

### 持续性攻击（3-frame）

| 攻击方法 | ROBOSAC DSR | CP-Guard DSR | GCP DSR | CAD DSR |
|---------|-------------|-------------|---------|---------|
| BAC [29] | 74.2 | 68.1 | 86.0 | 98.0 |
| RC spoof [39] | 38.1 | 24.0 | 42.1 | 100.0 |
| **MVIG spoof** | **37.0** | **26.2** | **42.0** | **63.2** (-37%) |

持续攻击下 BAC 和 RC 的 DSR 大幅上升（易被时序检测），而 MVIG 保持显著优势。

### 消融实验

- **影响区域大小**：spoof 攻击从 15m→50m，DSR 从 40.2% 降至 14.8%（更大搜索空间有利）；remove 攻击相对稳定（72.1%-84.5%），受限于必须瞄准已有物体
- **持续阈值 η**：无阈值时 DSR 85.3%/91.7%（spoof/remove）；阈值 0.45-0.50 时最优，平衡攻击机会与被检测风险
- **实时性能**：MVIG 达到 15.6-29.9 FPS，满足实时要求

### 关键发现

1. MVIG 在 CAD 防御上优势最大（-62%），因为 CAD 共享占用图暴露了更多可利用的脆弱性信息
2. 时序一致性是多帧攻击规避检测的关键——BAC 的空间攻击产生不一致时序模式（DSR>98%），而 MVIG 的时序优化保持攻击连贯性（DSR=63.2%）
3. ROC 分析显示 MVIG 从本质上降低了恶意/良性传输的可分离性（AUC 0.77 vs 0.85-0.91）
4. 攻击利用多车自然遮挡和视角差异带来的感知冲突区域，使伪造看起来像正常感知差异

## 亮点与洞察

1. **"防御即泄密"洞察**：首次系统揭示 CP 防御机制在验证过程中无意泄露系统脆弱性的安全悖论，这是一个深刻的安全性观察
2. **统一图表示**：MVIG 用互信息作为边权重，优雅地统一了不同防御系统的脆弱性建模，无需针对特定防御定制
3. **时空解耦攻击**：将掩码预测和扰动优化解耦，既保证攻击质量又满足实时约束（29.9 FPS）
4. **智能攻击终止**：熵感知的持续性控制是巧妙设计——主动放弃低概率窗口，反而大幅提升整体攻击隐蔽性

## 局限与展望

1. **Removal 攻击空间受限**：必须瞄准已有物体，优化空间天然有限，相比 spoof 攻击提升幅度较小
2. **受覆盖率约束**：当良性 CAV 的联合覆盖率很高时，攻击效果会下降（虽然可利用动态覆盖间隙）
3. **仅验证 LiDAR 场景**：未扩展到相机或多模态融合感知，Camera-based CP 是潜在方向
4. **防御对策缺失**：作为攻击论文，未提出有效的防御策略建议（仅附录讨论）
5. **仿真环境**：仅在 CARLA 仿真器数据上验证，真实物理世界场景的可行性未知

## 相关工作与启发

- **Tu et al. [31]** (Basic)：特征图扰动先驱，但修改过于明显，容易被异常检测发现
- **Tao et al. [29]** (BAC)：利用受害者视角盲区进行混淆攻击，但仅考虑单车知识，缺乏对集体验证的规避
- **Zhang et al. [39]** (RC + CAD)：引入 LiDAR 光线投射和占用图验证，但缺少时空系统优化
- **启发**：协作系统的"共享即暴露"问题在联邦学习、多智能体通信等领域普遍存在，MVIG 的图建模思路可迁移

## 评分

⭐⭐⭐⭐ 安全性视角新颖、统一框架设计优雅、实验全面，但 removal 攻击提升有限且缺少真实场景验证。

<!-- RELATED:START -->

## 相关论文

- [CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [SiMO: Single-Modality-Operable Multimodal Collaborative Perception](../../ICLR2026/autonomous_driving/simo_single-modality-operable_multimodal_collaborative_perceptio.md)
- [AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception](adaradar_rate_adaptive_spectral_compression_for_radar-based_perception.md)
- [RESBev: Making BEV Perception More Robust](resbev_making_bev_perception_more_robust.md)
- [SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)

<!-- RELATED:END -->
