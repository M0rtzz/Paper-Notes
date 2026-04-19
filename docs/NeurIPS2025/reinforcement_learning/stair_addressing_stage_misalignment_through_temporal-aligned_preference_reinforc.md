---
title: >-
  [论文解读] STAIR: Addressing Stage Misalignment through Temporal-Aligned Preference Reinforcement Learning
description: >-
  [NeurIPS 2025][偏好强化学习] 发现并形式化了偏好强化学习（PbRL）中的"阶段错位"问题——比较不同阶段的行为片段会产生无效反馈，提出STAIR方法通过对比学习获取时间距离来近似阶段差异，用四边形距离选择阶段对齐的查询，在多阶段任务中显著超越现有PbRL方法。
tags:
  - NeurIPS 2025
  - 偏好强化学习
  - 阶段对齐
  - 时间距离
  - 对比学习
  - 多阶段任务
---

# STAIR: Addressing Stage Misalignment through Temporal-Aligned Preference Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.23802](https://arxiv.org/abs/2509.23802)  
**代码**: [GitHub](https://github.com/iiiiii11/STAIR)  
**领域**: 强化学习  
**关键词**: 偏好强化学习, 阶段对齐, 时间距离, 对比学习, 多阶段任务

## 一句话总结

发现并形式化了偏好强化学习（PbRL）中的"阶段错位"问题——比较不同阶段的行为片段会产生无效反馈，提出STAIR方法通过对比学习获取时间距离来近似阶段差异，用四边形距离选择阶段对齐的查询，在多阶段任务中显著超越现有PbRL方法。

## 研究背景与动机

### 领域现状

偏好强化学习（PbRL）通过人类对行为片段的偏好来学习奖励函数，避免了复杂的奖励工程。核心流程是：采样两个行为片段 $(\sigma_0, \sigma_1)$ → 人类标注偏好 $y$ → 基于Bradley-Terry模型训练奖励函数 $\hat{r}_\psi$ → 用学到的奖励训练策略。现有方法（PEBBLE、SURF等）通过无监督预训练、奖励不确定性引导等方式提高反馈效率。

### 核心矛盾——阶段错位

很多现实任务具有**多阶段结构**，例如"取物-送回"任务包含：① 导航到物体 → ② 抓取 → ③ 搬运到目标位置。当PbRL方法随机采样片段对进行比较时，经常会将不同阶段的片段放在一起——比如比较"移动中的手臂"和"正在抓取的手臂"。根据认知科学的**事件分割理论**，人类会将动作序列划分为离散的事件边界，跨越这些边界的比较增加了认知负荷，产生**模糊反馈**。

### 理论分析

本文通过抽象MDP模型严格分析了阶段错位的影响：
- **Proposition 2**：在最坏情况下，传统PbRL比阶段对齐PbRL多需要 $\mathcal{O}(|\Omega||\Upsilon|\log(|\Omega||\Upsilon|))$ 个查询
- **Proposition 3**：当存在阶段奖励偏差（人类倾向偏好靠后阶段）时，传统方法需要 $\mathcal{O}(|\Omega|^2|\Upsilon|\log|\Upsilon|)$ 额外查询——从线性增长变为二次增长

人类实验直接验证了阶段奖励偏差的存在：在MetaWorld window-open任务中，人类标注清楚地显示偏好后期时间步的片段。

## 方法详解

### 整体框架

STAIR包含两个核心模块：
1. **学习时间距离**：通过对比学习训练一个能量函数，量化状态之间的时间距离
2. **阶段对齐查询选择**：利用时间距离定义片段间的阶段差异度量，优先选择阶段对齐的查询

### 关键设计

1. **时间距离学习（Successor Distance）**：采用折扣状态占据度量定义时间距离：

$$d_{SD}^\pi(x, y) = \log\left(\frac{p_\gamma^\pi(s_+=y|s_0=y)}{p_\gamma^\pi(s_+=y|s_0=x)}\right)$$

其中 $p_\gamma^\pi(s_+=y|s_0=x) = (1-\gamma)\sum_{k=0}^{\infty}\gamma^k p^\pi(s_k=y|s_0=x)$。这是一个拟度量（quasimetric），满足非负性和三角不等式。

通过对比学习优化能量函数 $f_\theta(x,y)$，使用对称InfoNCE损失：

$$\mathcal{L}_\theta = \sum_{i=1}^{B}\left[\log\frac{\exp(f_\theta(x_i, y_i))}{\sum_j \exp(f_\theta(x_i, y_j))} + \log\frac{\exp(f_\theta(x_i, y_i))}{\sum_j \exp(f_\theta(x_j, y_i))}\right]$$

参数化为 $f_\theta(x,y) = c_{\theta_c}(y) - d_{\theta_d}(x,y)$，最优解时 $d_{\theta_d^*}(x,y)$ 恰好就是successor distance。**关键优势**：时间距离在on-policy数据上学习，随策略演化自动更新。

2. **四边形距离（Quadrilateral Distance）**：将状态级时间距离扩展到片段级阶段差异度量：

$$d_{\text{stage}}(\sigma_0, \sigma_1) = \frac{1}{4}\left(d_{SD}^\pi(s_0^0, s_0^1) + d_{SD}^\pi(s_{H-1}^0, s_{H-1}^1) + d_{SD}^\pi(s_0^0, s_{H-1}^1) + d_{SD}^\pi(s_{H-1}^0, s_0^1)\right)$$

其中边项（起点-起点、终点-终点的距离）衡量起始和结束位置的对齐程度，对角项（起点-终点交叉距离）偏好时间跨度短的片段，使之集中在单一阶段内。Proposition 5证明了四边形距离随阶段偏移程度严格递增。

3. **阶段对齐查询选择**：综合阶段对齐度和奖励模型不确定性来评分每个候选查询：

$$I(\sigma_0, \sigma_1) = (c_{\text{stage}} - d_{\text{stage}}(\sigma_0, \sigma_1))(c_{\text{state}} + d_{\text{state}}(\sigma_0, \sigma_1))$$

第一项鼓励阶段对齐（小 $d_{\text{stage}}$），第二项鼓励高不确定性（大 $d_{\text{state}}$，基于奖励模型集成的方差）。选择得分最高的 $M$ 个查询送给人类标注。

### 损失函数 / 训练策略

- 奖励模型的交叉熵损失（Bradley-Terry模型）
- 时间距离的对称InfoNCE损失
- 策略使用SAC训练
- 时间距离以频率 $K_{SD}$ 更新（on-policy方式）

## 实验关键数据

### 多阶段任务（MetaWorld）

| 任务 | STAIR | PEBBLE | RUNE | MRN | RIME | QPA |
|------|-------|--------|------|-----|------|-----|
| door-open (5k) | **~100%** | ~86% | ~50% | ~60% | ~75% | ~80% |
| sweep-into (10k) | **~93%** | ~60% | ~30% | ~50% | ~55% | ~65% |
| window-open (2k) | **~95%** | ~70% | ~40% | ~55% | ~65% | ~70% |
| window-close (2k) | **~90%** | ~70% | ~50% | ~55% | ~65% | ~70% |
| door-unlock (5k) | **~100%** | ~85% | ~50% | ~75% | ~80% | ~85% |
| faucet-open (3k) | **~97%** | ~85% | ~65% | ~70% | ~75% | ~80% |

### 单阶段任务（DMControl）

| 任务 | STAIR | PEBBLE | RUNE | QPA |
|------|-------|--------|------|-----|
| walker-run | **最优** | 中等 | 较差 | 竞争性 |
| quadruped-walk | 竞争性 | 中等 | 最优 | 较好 |
| quadruped-run | **竞争性** | 中等 | 最优 | 较好 |

### 反馈效率实验（door-open）

| 查询总数 $N_{\text{total}}$ | STAIR | PEBBLE |
|---------------------------|-------|--------|
| 500 | 52.01±23.18 | 20.00±17.88 |
| 2000 | 77.77±11.67 | 28.79±17.02 |
| 5000 | **100.00±0.00** | 85.57±12.77 |
| 10000 | **99.93±0.06** | 92.53±6.53 |

### 消融实验

| 变体 | door-open | 说明 |
|------|-----------|------|
| STAIR（完整） | **100%** | 四边形距离+时间距离 |
| Timestep+ISR | ~90% | 用时间步代替时间距离 |
| STAIR(ISR) | ~95% | 时间距离+ISR（一维度量） |
| $K_{SD}=5$（低频更新） | ~95% | on-policy更新频率降低 |
| $K_{SD}=50$ | ~85% | 严重滞后于策略演化 |

### 人类实验

| 指标 | STAIR | PEBBLE |
|------|-------|--------|
| 人类判断为同阶段查询的比例 | **~70-80%** | ~30-40% |

### 关键发现

- STAIR在所有多阶段任务中一致超越基准，在多数任务中接近100%成功率
- 即使在单阶段任务中也有竞争性表现——可能归因于隐式的课程学习效应
- 四边形距离显著优于一维阶段度量（ISR），因为它在二维空间中捕捉更复杂的片段关系
- 时间距离的on-policy更新频率对性能至关重要
- 超参数 $c_{\text{stage}}, c_{\text{state}}$ 的鲁棒性良好

## 亮点与洞察

- **问题识别精准**：阶段错位是PbRL中被忽视但影响显著的问题，理论和人类实验都给出了有力支持
- **设计优雅**：时间距离→四边形距离→查询选择的链条自然流畅，每一步都有清晰的动机
- **无需任务先验**：不依赖预定义的阶段划分，通过时间距离自动发现阶段结构
- **人类认知一致性**：STAIR选出的查询被人类判断为"同阶段"的比例是PEBBLE的2倍，验证了方法与人类认知的对齐
- 在单阶段任务中的良好表现暗示了一种**隐式课程学习**机制

## 局限与展望

- 四边形距离仅评估成对片段差异，难以扩展到其他偏好格式（如排序、评分）
- 时间距离的学习依赖足够的on-policy数据，在训练初期可能不够准确
- 实验主要使用脚本化教师（oracle），真实人类标注的噪声容忍度有待更充分验证
- 阶段数量的自动确定未被显式讨论
- 未探索将阶段信息用于奖励模型结构本身的可能性

## 相关工作与启发

- **时间距离学习**：Successor distance (Myers等2024) 提供了拟度量保证，本文首次将其应用于PbRL的查询选择
- **事件分割理论**：认知科学中的human event segmentation为阶段错位问题提供了心理学基础
- **课程学习**：STAIR在单阶段任务中的成功暗示了自适应查询选择可以自然产生课程效应
- 启发：在其他需要比较的RL场景（如RLHF for LLM）中，是否也存在类似的"上下文错位"问题？

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 精准识别并解决了PbRL中一个重要但被忽视的问题
- 实验充分度: ⭐⭐⭐⭐⭐ 多域实验+消融+人类实验+噪声鲁棒性+反馈效率全面覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 问题驱动式写作，理论分析→人类验证→方法设计→实验验证环环相扣
- 价值: ⭐⭐⭐⭐⭐ 对PbRL领域的实际应用有直接指导意义，方法简洁有效

<!-- RELATED:START -->

## 相关论文

- [Temporal-Difference Variational Continual Learning](temporal-difference_variational_continual_learning.md)
- [Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering](enhancing_interpretability_in_deep_reinforcement_learning_through_semantic_clust.md)
- [Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [Incremental Sequence Classification with Temporal Consistency](incremental_sequence_classification_with_temporal_consistency.md)
- [Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm](modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)

<!-- RELATED:END -->
