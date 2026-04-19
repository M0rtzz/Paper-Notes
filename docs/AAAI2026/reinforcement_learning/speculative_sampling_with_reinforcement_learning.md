---
title: >-
  [论文解读] Speculative Sampling with Reinforcement Learning
description: >-
  [AAAI 2026][推测采样] 提出 Re-SpS，首个将推测采样（Speculative Sampling）的草稿树超参数优化建模为 MDP 并用强化学习求解的框架，通过特征复用和动作缓存两大设计，在不损失输出保真度的前提下，相比 EAGLE-3 实现最高 1.12× 的额外加速。
tags:
  - AAAI 2026
  - 推测采样
  - 大语言模型推理加速
  - 强化学习
  - 草稿树优化
  - PPO
---

# Speculative Sampling with Reinforcement Learning

**会议**: AAAI 2026  
**arXiv**: [2601.12212](https://arxiv.org/abs/2601.12212)  
**代码**: [github.com/wmd3i/ReSpS](https://github.com/wmd3i/ReSpS)  
**领域**: 强化学习  
**关键词**: 推测采样, 大语言模型推理加速, 强化学习, 草稿树优化, PPO

## 一句话总结

提出 Re-SpS，首个将推测采样（Speculative Sampling）的草稿树超参数优化建模为 MDP 并用强化学习求解的框架，通过特征复用和动作缓存两大设计，在不损失输出保真度的前提下，相比 EAGLE-3 实现最高 1.12× 的额外加速。

## 研究背景与动机

### 问题背景

大语言模型（LLM）的推理延迟是制约实际部署的核心瓶颈，根源在于自回归解码的逐 token 串行生成机制。推测采样（Speculative Sampling, SpS）通过"先用小模型草拟候选→再用大模型一次性验证"的范式来减少大模型前向次数，是当前最有效的无损加速方法之一。

### 现有方法的局限

EAGLE-2/EAGLE-3 等 SOTA 方法已引入树结构草稿（draft tree）来并行探索多个候选延续，并通过置信度排序实现动态剪枝。然而，这些方法存在一个关键缺陷：**控制草稿树整体结构的超参数（总 token 数 TT、深度 d、扩展因子 k）在整个解码过程中是静态且手动调优的**。不同上下文和任务需要不同的"推测激进度"——简单上下文可以大胆推测（大深度），复杂上下文则需要保守推测（小深度、大 top-k），静态配置无法适应这种变化。

### 为什么用 RL？

推测采样超参数的动态选择天然是一个序列决策问题：在每一步解码时，智能体观察当前生成上下文，选择一组超参数配置，获得"接受 token 数/耗时"的即时奖励。这完美匹配了 MDP 框架。

### 朴素 RL 的挑战

直接每步调用 RL 策略会引入两大开销：

**状态表示开销**：用 SentenceBERT 编码上下文的延迟（约 5-15ms/步）可能抵消加速收益

**策略推理开销**：每步策略网络前向传播的累积成本巨大（一个回答可能涉及 50-100+ 步解码）

这两者叠加后很可能使整体推理速度慢于未使用 RL 的基线。

## 方法详解

### 整体框架

Re-SpS 建立在 EAGLE-3 之上，将其原有的静态超参数替换为 RL 智能体的动态决策输出。整体流程：目标模型生成隐藏状态 → 聚合为状态向量 $s_t$ → RL 策略输出超参数 $(TT, d, k)$ → 草稿模型构建树结构候选 → 目标模型验证 → 计算奖励 → 更新策略。

### 关键设计

#### 1. **MDP 建模**：将超参数选择形式化为马尔可夫决策过程

- **状态空间 $\mathcal{S}$**：当前生成上下文的特征表示
- **动作空间 $\mathcal{A}$**：离散超参数组合 $\{(TT, d, k) | TT \in \mathcal{S}_{TT}, d \in \mathcal{S}_d, k \in \mathcal{S}_k\}$，每个维度取有限个预定义整数值
- **奖励函数 $R$**：即时生成速度 $r_t = \frac{\text{accepted tokens}}{\text{elapsed time (seconds)}}$，直接对齐延迟最小化目标
- **转移函数 $\xi$**：由草稿树构造和推测解码过程隐式确定，具有确定性

#### 2. **高效状态表示——特征复用机制**

核心思路：**不使用额外编码器**，而是直接复用 EAGLE-3 草稿模型中目标 LLM 已经计算好的隐藏状态。

$$s_t = [h_{LM}^{(h,m,l)}]$$

其中 $h_{LM}^{(h,m,l)}$ 是目标模型三个战略层（高层 h、中层 m、低层 l）隐藏状态的拼接，分别捕获句法、语义和任务特定信息。

**设计动机**：这些特征本身就是 EAGLE-3 架构的一部分（用于驱动草稿模型），无需额外计算开销。与 EAGLE-3 的区别在于：EAGLE-3 通过全连接层融合为单个向量，而 Re-SpS 直接拼接三层隐藏状态以避免全连接层的计算。

#### 3. **多步动作持久化——动作缓存机制**

核心思路：RL 策略选定的超参数配置 $(TT, d, k)$ 被**缓存并复用 $N$ 步**（训练时 $N=10$，推理时 $N=30$），避免每步调用策略网络。

奖励信号在缓存区间内取平均：

$$r_{avg} = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{accepted\_tokens}_i}{\text{elapsed\_time}_i}$$

**设计动机**：利用马尔可夫性质，平均奖励自然捕获了时序动态和性能影响，无需复杂的多步状态历史。这在适应性和效率之间取得了良好平衡。

### 损失函数 / 训练策略

采用 PPO（近端策略优化）作为骨干 RL 算法，并探索了最大熵变体：

**标准 PPO 目标**：

$$L^{PPO}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

**最大熵 PPO**（增加熵正则化鼓励探索）：

$$L^{MAX\text{-}ENT}(\theta) = L^{PPO}(\theta) + \beta_H \mathbb{E}_t[H(\pi_\theta(\cdot|s_t))]$$

训练细节：

- 策略网络：两层 MLP，128 隐藏单元（actor 和 critic 各一个）
- 训练数据：ShareGPT + UltraChat200K 的 4000 问题子集，覆盖多领域
- 熵系数 $\beta_H = 0.1$
- **无损保真性**：继承 EAGLE-3 的目标模型验证机制，输出与贪心解码逐字节一致

## 实验关键数据

### 主实验

| 骨干模型 | 方法 | MT-Bench | HumanEval | GSM8K | Alpaca | CNN/DM | 平均 |
|---------|------|----------|-----------|-------|--------|--------|------|
| LLaMA 3.1-8B | EAGLE-3 | 3.39× | 3.65× | 3.52× | 3.67× | **2.96×** | 3.44× |
| LLaMA 3.1-8B | **Re-SpS** | **3.43×** | **3.89×** | **3.62×** | **3.90×** | 2.87× | **3.54×** |
| Vicuna-13B | EAGLE-3 | 3.75× | 4.28× | 3.85× | 3.76× | **3.35×** | 3.80× |
| Vicuna-13B | **Re-SpS** | **3.76×** | **4.64×** | **3.99×** | **3.99×** | 3.24× | **3.92×** |
| LLaMA 3.3-70B | EAGLE-3 | 4.35× | 4.87× | 4.74× | 4.77× | **4.09×** | 4.46× |
| LLaMA 3.3-70B | **Re-SpS** | **4.47×** | **5.45×** | **5.13×** | **5.34×** | 4.03× | **4.88×** |

所有 p-value < 10⁻⁴（Wilcoxon 符号秩检验），差异高度统计显著。

### 消融实验

| 模型 | 策略配置 | 平均加速比 vs EAGLE-3 | 唯一动作数 |
|------|---------|---------------------|-----------|
| LLaMA 3.1-8B | Standard PPO + Text Embedding | 1.044× | 3 |
| LLaMA 3.1-8B | Standard PPO + Feature Vector | **1.049×** | 5 |
| LLaMA 3.1-8B | Max-Entropy PPO + Text Embedding | 1.017× | 8 |
| LLaMA 3.1-8B | Max-Entropy PPO + Feature Vector | 1.025× | **18** |
| Vicuna-13B | Standard PPO + Text Embedding | 1.006× | 8 |
| Vicuna-13B | Standard PPO + Feature Vector | 1.028× | 3 |
| Vicuna-13B | Max-Entropy PPO + Feature Vector | **1.033×** | **15** |

### 关键发现

1. **模型越大收益越大**：70B 模型平均 1.06× 加速（vs EAGLE-3），8B 仅 1.03×，表明大模型中动态超参数调整的潜力更大
2. **特征复用优于外部编码**：Feature Vector 在所有配置中一致优于 Text Embedding，验证了零额外成本的隐藏状态复用策略
3. **缓存区间长度**：从 1 增到 50 步，推理延迟显著降低、生成速度持续提升，最优点约在 30 步
4. **Max-Entropy PPO 促进动作多样性**：虽然加速比不一定最高，但产生的唯一动作数远多于标准 PPO（18 vs 5），策略更鲁棒、更具自适应性
5. **CNN/DM 上的轻微退化**（0.98×）：因需增大最大序列长度（2048→2200）避免 KV cache 溢出，额外开销导致

## 亮点与洞察

- **首次将 RL 引入推测采样超参数优化**，开辟了 SpS 领域的新研究方向
- **零额外计算的状态表示**是关键设计亮点：直接复用目标模型已有隐藏状态，完全消除编码器开销
- **动作缓存策略简洁有效**：一个简单的"缓存 N 步"就将 RL 开销降低了一个数量级，且不损失太多适应性
- 保持了**逐字节的输出无损保真性**，这是实际部署的硬性要求
- 在最大模型（70B）上取得了**高达 5.45× 的整体加速**（含 EAGLE-3 自身的加速）

## 局限与展望

- CNN/DailyMail 上因序列长度限制导致轻微退化，长文档场景有待优化
- 当前仅在贪心解码（temperature=0）下验证，随机采样下的效果未知
- 动作空间是预定义的离散网格，连续动作空间或更细粒度的超参数控制可能带来更大收益
- 仅在 EAGLE-3 架构上验证，其他推测采样方法（Medusa、C2T 等）的迁移性待验证
- 训练需要在目标硬件上进行，迁移到不同 GPU 配置可能需要重新训练

## 相关工作与启发

- 与 SpecDec++、DySpec、OPT-Tree 等启发式自适应方法不同，Re-SpS 是首个**数据驱动的学习方法**
- 为其他 LLM 推理加速技术（如 KV cache 管理、批处理调度等）提供了 RL 优化的新视角
- 动作缓存思想可推广到其他需要频繁决策但单步开销大的 RL 应用场景
- BanditSpec 和 MetaSD 使用多臂老虎机选择策略/模型，但不能动态调整树结构超参数

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个将 RL 用于 SpS 超参数优化，但核心技术（PPO、特征复用）较为标准
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5 个 benchmark、3 种模型规模、完整消融、统计检验
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、方法推导严谨，图表丰富
- **价值**: ⭐⭐⭐⭐ — 加速收益稳定但增量有限（1.03-1.06× over EAGLE-3），实际落地价值取决于部署规模

<!-- RELATED:START -->

## 相关论文

- [Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [Learning from Demonstrations via Capability-Aware Goal Sampling](../../NeurIPS2025/reinforcement_learning/learning_from_demonstrations_via_capability-aware_goal_sampling.md)
- [RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms](rlslm_a_hybrid_reinforcement_learning_framework_aligning_rule-based_social_locom.md)
- [Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)

<!-- RELATED:END -->
