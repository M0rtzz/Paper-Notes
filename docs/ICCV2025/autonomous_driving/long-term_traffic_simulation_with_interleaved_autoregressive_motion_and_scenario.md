---
title: >-
  [论文解读] Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation
description: >-
  [ICCV 2025][自动驾驶][交通仿真] 提出 InfGen，一个统一的自回归 next-token prediction 模型，通过交替进行闭环运动仿真和场景生成（智能体的动态插入与移除），首次实现稳定的长时程（30秒）交通仿真，在短时程任务上达到 SOTA 水平，在长时程任务上显著超越所有现有方法。
tags:
  - ICCV 2025
  - 自动驾驶
  - 交通仿真
  - 长时程仿真
  - 自回归生成
  - 场景生成
  - Next-Token Prediction
---

# Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation

**会议**: ICCV 2025  
**arXiv**: [2506.17213](https://arxiv.org/abs/2506.17213)  
**代码**: [orangesodahub.github.io/InfGen](https://orangesodahub.github.io/InfGen)  
**领域**: 自动驾驶  
**关键词**: 交通仿真, 长时程仿真, 自回归生成, 场景生成, Next-Token Prediction

## 一句话总结

提出 InfGen，一个统一的自回归 next-token prediction 模型，通过交替进行闭环运动仿真和场景生成（智能体的动态插入与移除），首次实现稳定的长时程（30秒）交通仿真，在短时程任务上达到 SOTA 水平，在长时程任务上显著超越所有现有方法。

## 研究背景与动机

### 问题定义

交通仿真旨在为自动驾驶系统创造逼真的驾驶体验。理想的仿真器应提供完整行程级（trip-level）的真实交通流，包括逼真的环境、自车动力学和所有非自车交通参与者。

### 已有方法的不足

现有方法的根本假设是**智能体集合在仿真周期内保持固定**，这在长时程仿真中完全不成立：

**智能体消失问题**：随着自车移动到新区域，初始日志中的智能体逐渐离开视野

**场景变空**：当自车驶入日志未覆盖的地图区域时，这些区域没有任何智能体

**不真实的空场景**：SMART 等 SOTA 模型在 30 秒仿真后，自车周围变得空无一人（如图 1）

三类现有工作的局限：
- **闭环运动仿真**（SMART、CatK）：只仿真已存在智能体的运动，不生成新智能体
- **场景生成**（SceneGen、TrafficGen）：只生成静态初始场景或短时开环场景
- **对抗场景生成**：聚焦近碰撞场景，不适用于通用长时程仿真

### 核心动机

长时程交通仿真需要同时解决两个问题：（1）已有智能体的闭环运动仿真；（2）新智能体的动态生成和旧智能体的退出。 InfGen 将这两个任务统一到一个交替进行的 next-token prediction 框架中。

## 方法详解

### 整体框架

InfGen 将长时程交通仿真建模为在"动态智能体矩阵"上的交替扩展：
- **时间轴扩展**（运动仿真）：为每个已有智能体预测下一时刻的运动 token
- **空间轴扩展**（场景生成）：插入新智能体行（pose token）或删除退出智能体行

形式化表述：
$$p(\mathcal{A}'_{t+1:T'} | \mathcal{M}, \mathcal{A}_{0:t_0}) = \prod_{t=t_0}^{T'-1} p_{\text{scene}}(\mathcal{A}'_{t+1} | \mathcal{M}, \mathcal{A}_{t+1}) \times p_{\text{motion}}(\mathcal{A}_{t+1} | \mathcal{M}, \mathcal{A}'_{0:t})$$

### 关键设计

#### 1. **统一 Token 化方案**

- **功能**：将地图、运动、位姿和模式控制信息全部转换为离散 token 序列。
- **核心思路**：

  四种 Tokenizer：
  - **Map Tokenizer**：将道路元素切分为固定长度向量，编码起止点、方向和道路类型
  - **Motion Tokenizer**：将 0.5 秒的连续轨迹段通过 k-disks 聚类为离散运动词汇 $\mathcal{V}_{\text{motion}}$，用最近邻索引编码
  - **Pose Tokenizer**：新智能体的初始位姿编码为位置 token（以自车为中心的网格索引）和朝向 token（360° 等分）
  - **Mode Control Tokenizer**：4 个特殊 token 控制任务切换
    - `<BEGIN MOTION>`：下一个 token 是运动 token
    - `<ADD AGENT>`：下一个 token 是位姿 token，插入新智能体
    - `<KEEP AGENT>`：当前智能体保留
    - `<REMOVE AGENT>`：当前智能体将被移除

- **设计动机**：将复杂的混合任务仿真问题转化为简单的序列预测问题。4 个控制 token 的设计使模型能学习**何时**切换任务和**如何**决定智能体增减。

#### 2. **交替 Next-Token Prediction**

- **功能**：在每个时间步交替执行运动仿真和场景生成。
- **核心思路**：

  **时间运动仿真**（蓝色流）：
  对每个活跃智能体 $i$，其运动 token $m_i^t$ 作为查询 $q_{m_i^t}$，经过三层注意力：
  1. **Temporal Attention**：自注意力，关注该智能体过去 $t_w$ 步的运动 token
  2. **Agent-Agent Attention**：交叉注意力，关注同一时刻范围 $r^{a \leftrightarrow a}$ 内的其他活跃智能体
  3. **Map-Agent Attention**：交叉注意力，关注范围 $r^{m \leftrightarrow a}$ 内的地图 token

  然后运动头和控制头各自输出 token 分布并采样。控制 token 限定为 `<KEEP AGENT>` 或 `<REMOVE AGENT>`。

  **空间场景生成**（绿色流）：
  使用可学习的 agent query $a_0$，经过三层注意力（Grid Attention 替代 Temporal Attention）。Grid Attention 关注占据栅格 token（由位置 token 构建的 $\{0,1\}$ 占据指示）：
  $$q'_{a_0} = \text{MHCA}^g(q_{a_0}, \Gamma(\{k_{g_j}\}), \Gamma(\{v_{g_j}\}))$$

  控制 token 限定为 `<ADD AGENT>` 或 `<BEGIN MOTION>`。`<ADD AGENT>` 时插入新行并赋予位姿 token；`<BEGIN MOTION>` 时结束场景生成，进入下一时间步的运动仿真。

- **设计动机**：
    - 交替执行使两个任务天然耦合，模型可以根据当前场景状态自动决定是否需要添加/移除智能体
    - 占据栅格编码让场景生成模块感知当前空间状态，避免在已有智能体处重复插入
    - Next-token prediction 的自回归特性使得 **在短日志上训练即可泛化到长时程仿真**（6倍延长）

#### 3. **Occupancy Grid Encoder**

- **功能**：将当前场景的智能体空间分布编码为占据栅格特征，供场景生成使用。
- **核心思路**：将位置 token 词汇表 $\mathcal{V}_\text{pos}$ 的每个位置标记为 0（空）或 1（被占据），通过 MLP 转换为特征后输入 Grid Attention。
- **设计动机**：让场景生成模块高效推理智能体的空间分布，决定在哪些区域插入新智能体。

### 损失函数 / 训练策略

总训练损失为多种 token 的标准 NTP 损失加权和：
$$\mathcal{L} = \lambda_1 \mathcal{L}_\text{motion} + \lambda_2 \mathcal{L}_\text{pos} + \lambda_3 \mathcal{L}_\text{head} + \lambda_4 \mathcal{L}_\text{control} + \lambda_5 \mathcal{L}_\text{shape} + \lambda_6 \mathcal{L}_\text{type}$$

其中 $\lambda_1 = \lambda_3 = 1$，$\lambda_2 = \lambda_4 = 10$，$\lambda_5 = 0.2$，$\lambda_6 = 5$。

训练 token 序列的构建：每个时间步按固定顺序排列 token：运动 token → 控制 token（REMOVE/KEEP）→ 位姿 token（ADD）→ BEGIN MOTION。同类 token 按智能体与自车距离从近到远排序。

训练配置：batch size 8，8 张 A5000，AdamW + cosine annealing，初始学习率 0.0005。

## 实验关键数据

### 主实验

短时程仿真（WOSAC 基准，9s）：

| 方法 | Composite↑ | Kinematic↑ | Interactive↑ | Map↑ |
|------|-----------|-----------|-------------|------|
| TrafficBots | 0.6976 | 0.3994 | 0.7103 | 0.8342 |
| GUMP | 0.7404 | 0.4773 | 0.7872 | 0.8339 |
| SMART-7M | 0.7521 | 0.4799 | 0.8048 | 0.8573 |
| CatK | 0.7603 | 0.4611 | 0.8103 | 0.8732 |
| **InfGen** | **0.7514** | **0.4754** | 0.7936 | 0.8502 |

长时程仿真（30s，扩展 WOSAC 指标）：

| 方法 | Composite↑ | Kinematic↑ | Interactive↑ | Map↑ | Placement $N_+$↑ | $N_-$↑ | $D_+$↑ | $D_-$↑ |
|------|-----------|-----------|-------------|------|-----|-----|-----|-----|
| SMART-7M | 0.6519 | 0.5839 | 0.7542 | 0.8102 | 0.4324 | 0.5713 | 0.4964 | 0.3371 |
| CatK | 0.6584 | 0.5850 | 0.7584 | 0.8186 | 0.4424 | 0.5842 | 0.5233 | 0.3371 |
| **InfGen** | **0.6606** | **0.5966** | **0.7619** | 0.8087 | **0.4542** | **0.6273** | **0.5635** | 0.3169 |

### 消融实验

智能体数量误差（ACE）指标：

| 方法 | Mean ACE↓ | ACE Slope↓ | 说明 |
|------|----------|-----------|------|
| SMART-7M | 12.0 | 0.31 | 场景逐渐变空 |
| CatK | 12.2 | 0.32 | 同上 |
| **InfGen** | **8.1** | **0.15** | 误差增长率仅为基线一半 |

仅运动仿真（禁用智能体增减，30s）：

| 方法 | Composite↑ | Kinematic↑ | Interactive↑ | Map↑ |
|------|-----------|-----------|-------------|------|
| SMART-7M | 0.7428 | 0.5413 | 0.7626 | 0.8349 |
| CatK | 0.7316 | 0.5216 | 0.7347 | 0.8495 |
| InfGen | 0.7432 | 0.5495 | 0.7685 | 0.8213 |

### 关键发现

1. **长时程仿真 InfGen 显著优于基线**：在放置指标（Placement）上的优势最为明显，证明动态场景生成的核心价值
2. **ACE Slope 仅 0.15 vs 基线 0.31-0.32**：InfGen 的场景密度误差增长率仅为基线的一半
3. **短时程同样有竞争力**：未经任何短时程特定调优，短时程性能接近 CatK
4. **仅运动仿真时三者相近**：证明长时程性能差异主要来自场景生成能力，而非运动预测能力
5. **训练短、推理长**：在 ~9s 日志上训练，可稳定仿真 30s（6 倍延长），展示了自回归框架的泛化潜力
6. **可视化清晰展示**：SMART 在 18s 后场景变空，InfGen 持续维持真实的交通密度

## 亮点与洞察

1. **问题定义的重要贡献**：明确指出"固定智能体集合"假设是长时程仿真的根本瓶颈，这是一个被忽视但关键的问题
2. **受 Chameleon 等多模态交替生成启发**：将视觉-语言的交替生成思路成功迁移到"时间运动-空间布局"的交替生成
3. **统一的 NTP 框架**：运动仿真和场景生成共享同一 transformer，4 个控制 token 实现优雅的任务切换
4. **评估体系贡献**：提出 ACE 指标和扩展 WOSAC 指标，为长时程仿真研究建立了评估标准
5. **Occupancy Grid Encoder 的巧妙设计**：让场景生成模块"看到"当前空间占据状态，避免不合理的重叠放置

## 局限与展望

1. **未达行程级仿真**：30 秒仍远短于真实行程（>5 分钟），主要受限于 WOMD 地图覆盖范围
2. **纯监督学习的局限**：可能过拟合训练数据的因果关系，未来计划引入交互式强化学习
3. **Map 指标略低**：新插入的智能体可能出现在非行车道或道路边界，导致地图合规性稍差
4. **智能体类型有限**：主要关注车辆，对行人和自行车的建模可能不够
5. **仅在 WOMD 上验证**：泛化到 nuPlan 或其他仿真环境未经测试

## 相关工作与启发

- 与 **SMART** 的区别：SMART 是纯运动仿真的 NTP 模型，InfGen 在其基础上加入场景生成能力
- 与 **SceneGen/TrafficGen** 的区别：后者生成静态初始场景，InfGen 在闭环仿真过程中动态生成
- 与 **Chameleon** 的类比：交替生成的思路从"文本+图像"迁移到"运动+场景布局"
- 与 **SLEDGE** 的区别：SLEDGE 用生成模型 + 规则仿真器，InfGen 完全端到端

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将交替生成引入交通仿真，统一解决运动+场景生成
- **实验充分度**: ⭐⭐⭐⭐ — 短时程和长时程全面评估，但缺少真正的行程级仿真
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题动机清晰，可视化对比有说服力
- **价值**: ⭐⭐⭐⭐⭐ — 为真实行程级交通仿真迈出了重要一步
