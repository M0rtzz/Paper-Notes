---
title: >-
  [论文解读] Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation
description: >-
  [AAAI 2026][强化学习][课程强化学习] 提出一个课程强化学习（CRL）框架用于自主且可持续的海洋船舶航行，集成了基于真实AIS数据的仿真环境、扩散模型增强的动态海上交通模拟、以及机器学习燃油消耗预测模块，通过多目标奖励函数同时优化航行安全性、排放减少、时效性和目标完成。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "课程强化学习"
  - "自主航行"
  - "海上可持续性"
  - "扩散模型"
  - "燃油消耗预测"
---

# Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation

**会议**: AAAI 2026  
**arXiv**: [2601.10911](https://arxiv.org/abs/2601.10911)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 课程强化学习, 自主航行, 海上可持续性, 扩散模型, 燃油消耗预测

## 一句话总结

提出一个课程强化学习（CRL）框架用于自主且可持续的海洋船舶航行，集成了基于真实AIS数据的仿真环境、扩散模型增强的动态海上交通模拟、以及机器学习燃油消耗预测模块，通过多目标奖励函数同时优化航行安全性、排放减少、时效性和目标完成。

## 研究背景与动机

海运承载全球约90%的贸易量，但面临日益严峻的碳减排压力。国际海事组织（IMO）制定了到2050年至少减排50%的战略目标。当前，在"照常营业"情景下，航运碳排放预计到2050年将比2008年水平增加10-30%。

**安全挑战**：海上安全是可持续海运的基本要素。SANCHI号油轮碰撞事故导致32名船员遇难，超过10万吨石油产品排放或燃烧——调查结论指出人为错误是主要原因。

**现有DRL方法的局限**：
1. 现有工作多聚焦单一目标（仅避碰或仅减排），缺乏多目标同时优化
2. 在高度动态的真实海上环境中可扩展性和泛化能力有限
3. 仿真环境与真实场景差距大，缺乏基于真实数据的高保真仿真
4. 缺乏精确的燃油消耗反馈，难以做出排放感知的导航决策

**本文贡献**：提出将课程学习（CL）集成到DRL中，在真实数据驱动的仿真环境中，实现安全、减排、时效和目标完成的多目标自主航行。

## 方法详解

### 整体框架

系统由三个核心模块组成：

1. **燃油消耗预测模块**：基于XGBoost的机器学习模型，利用真实运营数据预测燃油消耗率
2. **海上交通环境模块**：基于真实AIS数据 + 扩散模型增强的高保真仿真环境
3. **CRL策略学习模块**：基于PPO算法的课程强化学习，包含Actor-Critic网络和多目标奖励函数

### 关键设计

#### 1. 燃油消耗预测模块

使用数百艘国际远洋船舶两年的真实运营数据训练XGBoost模型。

**输入特征**（共86维）：
- 航行参数：行驶距离、经纬度、对地速度（SOG）
- 船舶特征：总长（LOA）、船宽、总吨位（GT）、船型
- 时间变量：月、日、小时（与海洋条件相关）
- 分类特征采用one-hot编码

**输出**：燃油消耗率（FCR），单位为公吨/小时，综合考虑主机、辅机、辅助机械和四种燃油类型（HFO、LSFO、DO、LSGO）。

模型公式化为：
$$\hat{y} = f_{\text{xgboost}}(\mathbf{x}) = \sum_{k=1}^{K} f_k(\mathbf{x}), \quad f_k \in \mathcal{F}$$

#### 2. 扩散模型增强的海上交通环境

利用扩散模型生成合成AIS轨迹，丰富仿真环境的真实性和多样性。船舶轨迹表示为位置序列：

$$\mathbf{x}_0 = ((\phi_{t,1}, \lambda_{t,1}, v_{t,1}), \cdots, (\phi_{t,T}, \lambda_{t,T}, v_{t,T}))$$

前向扩散过程逐步添加高斯噪声：
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

训练损失为去噪目标：
$$\mathcal{L}_{\text{DM}} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

通过反向过程采样新轨迹，生成符合真实运动模式的多样化船舶轨迹。

#### 3. 课程强化学习框架

**状态表示**由两部分组成：
- **自身船舶状态** $\mathbf{s}_1^t \in \mathbb{R}^9$：当前经纬度、目的地经纬度、航向角、洋流方向、当前速度（角度特征使用正余弦编码）
- **环境状态** $\mathbf{s}_2^t \in \mathbb{R}^{64 \times 64 \times 3}$：以自身为中心的三通道图像张量
    - Channel 0：占用指示（是否有船舶）
    - Channel 1：对地速度（SOG）
    - Channel 2：对地航向（COG）

**动作空间**为连续二维空间：
$$\mathbf{a}_t = [\Delta\psi_t, v_t], \quad -\hat{\psi} \leq \Delta\psi_t \leq \hat{\psi}, \quad v_l \leq v_t \leq v_u$$

**课程学习机制**：引入随训练episode递减的距离阈值参数：
$$\omega(e) = \omega_0 \cdot (1 - \frac{e}{N_e}) + \omega_f \cdot \min(\frac{e}{N_e}, 1)$$

初始阈值 $\omega_0 = 5$ 海里，逐步缩小，使得智能体从简单任务过渡到困难任务。

**多目标奖励函数**：

$$r_t = \begin{cases} 30 + 1.5g_t - f_t - s_t, & \text{if } d_{\text{cur}} < \omega(e) \\ 1.5g_t - f_t - s_t - 1.0, & \text{if } d_{\text{cur}} > d_{\text{pre}} \\ 1.5g_t - f_t - s_t - 0.1 \cdot d_{\text{cur}}, & \text{if late and } d_{\text{cur}} \geq \omega(e) \\ 1.5g_t - f_t - s_t, & \text{otherwise} \end{cases}$$

其中 $g_t = d_{\text{pre}} - d_{\text{cur}}$（目标接近奖励），$f_t = \alpha \cdot \frac{f_{\text{XGBoost}}(\mathbf{x}_t)}{GT}$（燃油惩罚），$s_t$ 为基于DCPA/TCPA的安全惩罚。

**安全距离定义**：
$$d_{\text{safe}} = \max\left(\tau \cdot \frac{L_s + B_s + L_t + B_t}{2 \times 1852}, 0.5\right)$$

缓冲乘数 $\tau = 4$，最大接近时间 $t_{\max} = 15$ 分钟。

### 损失函数 / 训练策略

- **Actor网络**：双分支架构——环境图像通过轻量卷积（2层可分离卷积）提取空间特征，自身状态通过全连接层处理，两分支合并后计算动作分布
- **Critic网络**：与Actor共享输入处理结构，输出状态价值标量
- **训练算法**：PPO（Proximal Policy Optimization）
- **月度变化的洋流建模**：基于历史分布模拟洋流方向和速度的动态扰动

## 实验关键数据

### 主实验

**燃油消耗预测模型比较**：

| 方法 | MAE | RMSE | R² (%) |
|------|-----|------|--------|
| SVR | 0.4529 | 0.7050 | 45.01 |
| MLP | 0.2440 | 0.4916 | 77.06 |
| ET | 0.2116 | 0.4033 | 84.56 |
| LightGBM | 0.2015 | 0.3895 | 85.60 |
| RF | 0.1752 | 0.3832 | 86.06 |
| **XGBoost** | **0.1802** | **0.3827** | **86.10** |

**导航控制模型比较**（Instance Case 1，21小时，印度洋）：

| 方法 | 累积奖励(AR)↑ | 累积燃油(AFC)↓ | 累积安全分(ASS)↓ |
|------|-------------|--------------|----------------|
| CL-ABDDQN | 155.527 | 20.509 | 1.466 |
| CL-A2C | -82.034 | 13.092 | 7.501 |
| DDPG | 150.424 | 33.533 | 1.840 |
| **CRL (本文)** | **154.018** | **18.015** | **0.888** |

**Instance Case 2**（46小时）：

| 方法 | AR↑ | AFC↓ | ASS↓ |
|------|-----|------|------|
| CL-ABDDQN | **298.026** | 36.229 | 5.543 |
| CL-A2C | 287.175 | 21.487 | 3.952 |
| DDPG | 279.439 | 20.974 | 5.063 |
| **CRL** | 294.148 | **19.963** | 4.754 |

**Instance Case 3**（21小时，不同起终点）：

| 方法 | AR↑ | AFC↓ | ASS↓ |
|------|-----|------|------|
| CL-ABDDQN | **155.035** | 20.901 | 2.332 |
| CL-A2C | -100.699 | 35.154 | 7.258 |
| DDPG | 148.615 | **9.002** | 2.710 |
| **CRL** | 145.015 | 28.618 | **2.143** |

### 消融实验

| 实例 | 方法 | AR↑ | AFC↓ | ASS↓ | 说明 |
|------|------|-----|------|------|------|
| Case 1 | w/o CL | 139.629 | 21.893 | 1.069 | CRL优于所有指标 |
| Case 1 | **CRL** | **154.018** | **18.015** | **0.888** | — |
| Case 2 | w/o CL | -708.765 | 48.951 | 3.274 | 无CL导致任务失败 |
| Case 2 | **CRL** | **294.148** | **19.963** | 4.754 | — |
| Case 3 | w/o CL | 149.950 | **19.557** | 5.901 | 燃油稍优但安全差 |
| Case 3 | **CRL** | 145.015 | 28.618 | **2.143** | — |

### 关键发现

1. **CRL实现最佳多目标平衡**：虽非所有单项指标最优，但在安全、燃油、奖励三者间取得最好的综合表现
2. **课程学习是必需的**：Instance Case 2中移除CL直接导致任务失败（AR从294降至-709）
3. **CL加速收敛、稳定训练**：训练奖励曲线显示CRL更快达到高奖励且波动更小
4. **CL-A2C在多数场景下失败**：两个实例中AR为负，说明A2C在该连续动作空间中不适用
5. **DDPG虽燃油效率高但安全性差**：在Case 3中AFC最低但ASS最高

## 亮点与洞察

- **数据驱动的全栈解决方案**：从燃油预测、环境仿真到策略学习，每个环节都基于真实数据构建
- **扩散模型首次用于AIS轨迹生成**：丰富了仿真环境的多样性，比简单重放历史数据更灵活
- **图像化环境编码**：将复杂的海上交通态势压缩为 $64 \times 64 \times 3$ 的视觉表示，简洁高效
- **课程学习设计合理**：渐缩距离阈值的设计直觉清晰——先学到达远处目标，再逐步精细化
- **安全距离公式融入船舶几何特征**：考虑了自身和目标船只的LOA和beam，更加物理合理

## 局限与展望

1. **实验规模有限**：仅在印度洋一个区域的3个实例上验证，缺乏更广泛地理条件的测试
2. **CRL在Case 3的燃油效率较差**：AFC为28.618，远高于DDPG的9.002，多目标权重可能需要调整
3. **动态避碰能力未充分验证**：虽有ASS指标，但未展示具体的多船避碰场景的行为分析
4. **缺少与COLREGs规则的显式集成**：国际海上避碰规则未被硬编码到策略中
5. **燃油预测模型精度**：R²=86.1%仍有约14%的误差空间，极端条件下可能影响决策质量
6. **数据可获取性问题**：使用的AIS和燃油数据是行业合作的私有数据，难以复现

## 相关工作与启发

- 与Wang et al. (2024a)的DDPG港口拖船优化、Moradi et al. (2022)的减排路径优化形成对比
- CL思想可推广到其他复杂环境中的RL任务，如无人机编队、自动驾驶等
- 扩散模型增强仿真环境的思路可延伸到自动驾驶场景生成

## 评分

- 新颖性: ⭐⭐⭐ — 各模块（CRL、扩散模型、XGBoost）单独看不新，但组合和应用场景有价值
- 实验充分度: ⭐⭐⭐ — 消融完整但场景偏少，仅3个实例且局限于单一海域
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式规范，图表丰富
- 价值: ⭐⭐⭐⭐ — 面向实际的海事减排和安全问题，具有明确的应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding](start_small_think_big_curriculum-based_relative_policy_optimization_for_visual_g.md)
- [\[ACL 2026\] NaviMaster: Learning a Unified Policy for GUI and Embodied Navigation Tasks](../../ACL2026/reinforcement_learning/navimaster_learning_a_unified_policy_for_gui_and_embodied_navigation_tasks.md)
- [\[ICML 2025\] Learning Progress Driven Multi-Agent Curriculum](../../ICML2025/reinforcement_learning/learning_progress_driven_multi-agent_curriculum.md)
- [\[ICCV 2025\] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](../../ICCV2025/reinforcement_learning/navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)
- [\[ICML 2026\] Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training](../../ICML2026/reinforcement_learning/provable_benefit_of_curriculum_in_transformer_tree-reasoning_post-training.md)

</div>

<!-- RELATED:END -->
