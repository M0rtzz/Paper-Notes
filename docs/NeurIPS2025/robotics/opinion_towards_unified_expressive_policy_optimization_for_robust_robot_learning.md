---
title: >-
  [论文解读] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning
description: >-
  [NeurIPS 2025][机器人][离线到在线 RL] 提出 UEPO 框架，通过多种子动力学感知扩散策略、动态分歧正则化和基于扩散的数据增强三大核心组件，解决离线到在线强化学习中多模态行为覆盖不足和分布偏移问题，在 D4RL 基准上超越 Uni-O4。
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "离线到在线 RL"
  - "扩散策略"
  - "策略多样性"
  - "动力学建模"
  - "D4RL"
---

# Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning

**会议**: NeurIPS 2025  
**arXiv**: [2511.10087](https://arxiv.org/abs/2511.10087)  
**代码**: 无  
**领域**: 强化学习 / 机器人学习  
**关键词**: 离线到在线 RL, 扩散策略, 策略多样性, 动力学建模, D4RL

## 一句话总结

提出 UEPO 框架，通过多种子动力学感知扩散策略、动态分歧正则化和基于扩散的数据增强三大核心组件，解决离线到在线强化学习中多模态行为覆盖不足和分布偏移问题，在 D4RL 基准上超越 Uni-O4。

## 研究背景与动机

离线到在线强化学习（O2O-RL）是机器人安全高效部署的关键范式：先用离线数据预训练基础策略以捕捉物理动力学，再通过环境交互微调策略以适应动态场景。

现有方法面临两大核心挑战：

**多模态行为覆盖不足**：传统行为克隆（BC）依赖大量专家数据，难以覆盖多模态动作分布；扩散策略虽然擅长离线建模，但固定的噪声调度和缺乏环境反馈导致在线微调时出现策略退化和分布偏移

**离线与在线之间的接口薄弱**：现有框架如 Uni-O4 在离线预训练、生成模型适配和可扩展性方面仍存在局限，计算成本高、物理执行层面多样性不足、数据效率低

UEPO 的设计灵感来源于大语言模型的"预训练+微调"策略，将这一思路迁移到机器人策略学习。

## 方法详解

### 整体框架

UEPO 包含三个核心创新，贯穿离线和在线学习阶段：

1. **离线阶段**：多种子扩散采样初始化 → 分歧正则化增强多样性 → 扩散数据增强训练动力学模型
2. **在线阶段**：从离线阶段选择合格策略作为初始化，进行在线微调

### 关键设计

#### 1. 多种子条件动作序列生成

采用状态条件扩散策略建模完整动作序列分布 $p(a_{1:T}|s_{1:T})$，捕捉长时域依赖和多模态行为。

**核心思路**：避免训练多个独立模型，而是从单一训练好的扩散模型出发，通过改变反向采样的初始噪声种子构建 $n$ 个子策略集合 $\{\pi_\theta^i\}_{i=1}^n$。每个子策略在相同状态序列上使用不同随机种子 $\epsilon_i \sim \mathcal{N}(0, \mathbf{I})$ 初始化反向过程，生成对应不同行为模态的动作序列。

**优势**：相比传统集成方法需要训练多个独立模型，大幅降低计算成本。

#### 2. 扩散采样引导的分歧正则化

仅靠多种子提供初始多样性不够，需要确保子策略在动态执行中具有分歧性。

**动态分歧度量**：衡量两个动作序列 $a_i$ 和 $a_j$ 之间的一阶（速度）和二阶（加速度）动力学差异：

$$\text{div}(a_i, a_j) = \frac{1}{T}\sum_{t=1}^{T}\left(\|\dot{a}_{i,t} - \dot{a}_{j,t}\|_2 + (1 - \cos(\ddot{a}_{i,t}, \ddot{a}_{j,t}))\right)$$

**自适应扰动**：当分歧度低于阈值 $\tau$ 时，向去噪估计注入自适应噪声：

$$a_t^i \leftarrow a_t^i + \delta, \quad \delta \sim \mathcal{N}(0, \sigma_{\text{div}}^2 \mathbf{I}), \quad \sigma_{\text{div}} = \eta \cdot \frac{\tau - \text{div}(a_i, a_j)}{\tau}$$

分歧越小，扰动越大，迫使子策略探索不同动态模式。

**与序列级 KL 正则化的协同**：保留 Uni-O4 的 KL 散度惩罚，但从单步动作分布重新定义为整个动作序列分布，与序列级扩散策略自然对齐。

#### 3. 基于扩散的动力学模型增强

利用扩散策略生成物理上合理的虚拟轨迹来增强动力学模型的训练数据：

- 从离线数据分布中采样初始状态
- 使用扩散策略生成多步动作序列
- 通过真实转移动力学生成轨迹
- **关键过滤**：计算真实动力学和初始模型之间的 KL 散度，仅保留 $D_{KL} < \epsilon$ 的轨迹
- 增强数据量控制在原始数据的 2-3 倍

### 损失函数 / 训练策略

子策略总目标函数结合似然最大化和序列级 KL 正则化：

$$J(\hat{\pi}^i) = \mathbb{E}_{(s,a)\sim\mathcal{D}}[\log p_\theta(a|s)] + \alpha \mathbb{E}_{(s,a)\sim\mathcal{D}}\left[\log\frac{p_\theta(a|s)}{\max_j p_\theta(a|s)}\right]$$

动力学模型的联合训练目标：

$$\mathcal{L}(\hat{T}) = -\mathbb{E}_{(s,a,s')\sim\mathcal{D}\cup\mathcal{D}_{\text{diff}}}[\log \hat{T}(s'|s,a)]$$

## 实验关键数据

### 主实验：D4RL Locomotion 任务

| 环境 | CQL | TD3+BC | IQL | BPPO | Uni-O4 | **UEPO** |
|------|-----|--------|-----|------|--------|----------|
| halfcheetah-medium-v2 | 44.0 | 48.3 | 47.4 | 44.0 | 52.6 | **57±0.8** |
| hopper-medium-v2 | 58.5 | 59.3 | 66.3 | 93.9 | 104.4 | **108±0.5** |
| walker2d-medium-v2 | 72.5 | 83.7 | 78.3 | 83.6 | 90.2 | **91±1.4** |
| halfcheetah-medium-replay | 45.5 | 44.6 | 44.2 | 41.0 | 44.3 | **58.2±0.7** |
| hopper-medium-replay | 95.0 | 60.9 | 97.7 | 92.5 | 103.2 | **112.0±2.3** |
| walker2d-medium-replay | 77.2 | 81.8 | 73.9 | 77.6 | 98.4 | **103.8±1.7** |
| halfcheetah-medium-expert | 91.6 | 90.7 | 89.7 | 92.6 | 93.8 | **94.3±0.6** |
| hopper-medium-expert | 105.4 | 98.0 | 91.7 | 112.8 | 111.4 | **118.6±0.2** |
| walker2d-medium-expert | 108.8 | 110.1 | 109.6 | 113.1 | 118.1 | **120.7±0.3** |
| **Locomotion 总分** | 698.5 | 677.4 | 692.4 | 751.0 | 816.4 | **864.6±8.5** |

### Adroit 灵巧操控 + Kitchen 任务

| 环境 | CQL | IQL | BPPO | Uni-O4 | **UEPO** |
|------|-----|-----|------|--------|----------|
| pen-human | 37.5 | 71.5 | 117.8 | 116.2 | **122.8±5.8** |
| hammer-human | 4.4 | 1.4 | 14.9 | 247.1 | 30.2±3.3 |
| door-human | 9.9 | 4.3 | 25.8 | 17.3 | **29.3±0.7** |
| pen-cloned | 39.2 | 37.3 | 110.8 | 101.4 | **118.4±12.4** |
| **Adroit 总分** | 93.6 | 118.1 | 291.4 | 288.6 | **324.4±26.5** |
| kitchen-complete | 43.8 | 62.5 | 91.5 | 93.6 | **102.6±3.6** |
| kitchen-partial | 49.8 | 46.3 | 57.0 | 58.3 | 57.6±2.8 |
| kitchen-mixed | 51.0 | 51.0 | 62.5 | 65.0 | **70.3±5.6** |
| **Kitchen 总分** | 144.6 | 159.8 | 211.0 | 216.9 | **230.5±12.0** |
| **总分** | 936.7 | 970.3 | 1253.4 | 1322.0 | **1419.5±47.0** |

### 关键发现

1. **全面超越 Uni-O4**：Locomotion 总分提升 +48.2（+5.9%），Adroit 提升 +35.8（+12.4%），总分提升 +97.5（+7.4%）
2. **在 medium-replay 类任务上提升最显著**：halfcheetah-medium-replay 从 44.3 提升到 58.2（+31.4%），说明在数据质量较差时优势更大
3. **Adroit 灵巧操控显示强扩展性**：在高维、需要精细控制的任务上提升明显
4. **部分任务低于 Uni-O4**：hammer-human（30.2 vs 247.1）和 relocate-human（2.9 vs 27.1），说明在某些特定任务上仍有局限
5. **标准差总体较小**：说明方法稳定性好，但 pen-cloned（±12.4）和 hammer-human（±3.3）波动较大

## 亮点与洞察

- **单模型多种子集成策略**：避免训练多个模型的计算开销，通过改变噪声种子从同一扩散模型生成多样化子策略，是优雅且高效的设计
- **动力学级别的分歧度量**：使用速度和加速度差异而非简单的分布距离来衡量策略多样性，确保差异在物理执行层面是有意义的
- **自适应扰动机制**：分歧不足时自动增大扰动，形成负反馈调节环路，防止子策略收敛到相似模式
- **扩散增强数据的 KL 过滤**：确保合成轨迹与真实动力学一致，避免引入错误的增强数据

## 局限与展望

- **仅在 D4RL 模拟基准上评估**：缺乏真实机器人实验验证
- **hammer-human 和 relocate-human 性能大幅落后 Uni-O4**：某些任务上的退化原因未分析
- **计算成本未详细讨论**：虽然声称比多模型集成更高效，但未提供具体的训练时间和资源对比
- **超参数敏感性未探讨**：分歧阈值 $\tau$、扰动强度 $\eta$、增强数据量比例等关键超参的影响未分析
- **论文引用了大量与方法关联不大的参考文献**（如相机标定、多模态推荐），影响了学术严谨性

## 相关工作与启发

- **Uni-O4**：本文直接改进的基线，通过联合优化离线和在线目标实现 O2O-RL，但在策略多样性和数据效率方面有局限
- **Diffusion Policy**：扩散模型在策略学习中的成功应用，但固定噪声调度限制了在线适配
- **BPPO**：行为近端策略优化，在 Adroit 上表现强劲
- **启发**：从 LLM 的"预训练+微调"范式迁移到 RL 策略学习是有潜力的方向；多种子采样是构建策略集合的低成本替代方案

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 三个技术组件设计合理，多种子策略集成和动力学分歧度量有创意
- **有效性**: ⭐⭐⭐⭐ — D4RL 上全面提升，但部分任务退化需要解释
- **可复现性**: ⭐⭐⭐ — 未提供代码，部分实现细节不够完整
- **影响力**: ⭐⭐⭐⭐ — 对 O2O-RL 和扩散策略社区有直接贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Human-assisted Robotic Policy Refinement via Action Preference Optimization](human-assisted_robotic_policy_refinement_via_action_preference_optimization.md)
- [\[NeurIPS 2025\] Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm](trust_region_reward_optimization_and_proximal_inverse_reward_optimization_algori.md)
- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)

</div>

<!-- RELATED:END -->
