---
title: >-
  [论文解读] How LLMs Learn to Reason: A Complex Network Perspective
description: >-
  [ICLR 2026][RLVR] 从复杂网络视角提出**稀疏概念网假说**（平均度 $\langle k\rangle\approx 2$），统一解释 RLVR 训练的四大谜题（两阶段学习曲线、V 型回复长度、灾难性遗忘、策略坍缩），并据此设计 Annealed-RLVR 算法在 Minerva 和 AIME 上超越标准 RLVR。
tags:
  - ICLR 2026
  - RLVR
  - 概念网络
  - 稀疏图
  - 灾难性遗忘
  - 模拟退火
---

# How LLMs Learn to Reason: A Complex Network Perspective

**会议**: ICLR 2026  
**arXiv**: [2509.23629](https://arxiv.org/abs/2509.23629)  
**代码**: [GitHub](https://anonymous.4open.science/r/CoNet-83A4)  
**领域**: LLM推理 / 理论分析  
**关键词**: RLVR, 概念网络, 稀疏图, 灾难性遗忘, 模拟退火

## 一句话总结

从复杂网络视角提出**稀疏概念网假说**（平均度 $\langle k\rangle\approx 2$），统一解释 RLVR 训练的四大谜题（两阶段学习曲线、V 型回复长度、灾难性遗忘、策略坍缩），并据此设计 Annealed-RLVR 算法在 Minerva 和 AIME 上超越标准 RLVR。

---

## 研究背景与动机

RLVR（Reinforcement Learning with Verifiable Rewards）训练 LLM 时，数学/代码基准上反复出现四个令人困惑的宏观现象：

| 现象 | 具体表现 | 现有碎片化解释 |
|------|---------|---------------|
| ① 两阶段学习曲线 | 准确率快速上升后进入漫长平台期 | 熵耗尽、奖励信号过度利用 |
| ② V 型回复长度 | 正确回复先变短后变长 | 奖励驱动裁剪 → 自反思/更难问题 |
| ③ 灾难性遗忘 | 后续 SFT 立即摧毁 RLVR 能力 | 目标不匹配 |
| ④ 策略坍缩 | 解的多样性随训练推进持续下降 | 过度利用固定奖励 |

**核心痛点**：这些解释互不关联，缺乏统一框架将四个现象归因于共同的结构/动力学机制。

**核心矛盾**：直接从 LLM 高维潜空间构造推理图极其困难，无法直接研究结构根源。

**切入角度**：借鉴复杂系统**重整化群（RG）**思想——宏观涌现行为独立于微观细节。在语义层面构建最小代理模型 **CoNet**（Concept Network Model）对推理图做粗粒化，然后利用代理模型的透明性做微观分析。

---

## 核心假说：稀疏概念网

> **Hypothesis 1**：概念网（coarse-grained reasoning graph）是一个平均度钉在 $\langle k\rangle\approx 2$ 的**稀疏网络**，主要为树状/链状结构——**高效但脆弱**。

平均度 ≈ 2 意味着：

- 几乎没有冗余路径，连接远距概念需要长链
- 依赖少量"主干边"维持全局连通性
- 少量稀疏环（如自我纠正）不改变全局平均度

---

## 方法框架

整体路线：**实验验证→代理建模→微观分析→理论统一→算法设计**

1. 用 1.5B LLM（DeepSeek-R1-Distill-Qwen-1.5B，DeepScaleR 协议）验证四大 RLVR 现象
2. 构建 CoNet 最小模型：将 LLM "语义状态"映射为固定随机图上的节点，"逻辑转移"映射为可学习的概率边
3. 确认 CoNet 定性复现所有四个宏观现象（**动力学同构**）
4. 利用 CoNet 透明性逐一剖析微观机制
5. 从稀疏网假说推导统一解释，进而设计 Annealed-RLVR

---

## 四大现象的统一解释

### 1. V 型轨迹 = 局部优化→全局整合的结构演化

**下降段（快速学习阶段）**：模型并行发现短的、独立的"技能岛"。初始随机环境下短路径本身就更易被采样，策略快速丢弃低效探索→回复变短。技能岛数量在此阶段末达到峰值。

**上升段（慢学习阶段）**：技能岛逐步融合为统一概念网。由于 $\langle k\rangle\approx 2$ 的树状结构没有捷径，连接远距概念必须走长路径→回复长度被拓扑结构**几何性地强制增长**。实验中路径长度分布随训练持续右移，直接验证此机制。

> V 型不是推理风格的变化，而是概念网持续生长的拓扑必然结果。

### 2. 灾难性遗忘 = 稀疏网关键桥的拓扑断裂

稀疏网依赖少量"主干边"维系大量下游子图的可达性。SFT 的梯度更新在高频分支节点覆盖策略权重→**精确切断**连接子树的关键桥→整个子图变为不可达。

**关键洞察**：损伤是**拓扑性而非知识性**的。断裂子图中的知识完好无损，只是从主干被"断开"。因此恢复 RLVR 仅需"重新焊接"几条桥→**恢复速度远快于从头训练**。

### 3. 策略坍缩 = 前沿叶节点处连续相变的累积

慢学习阶段，概念网前沿（叶节点）的每个技能经历**连续相变式学习**：从广泛探索锁定到单一最优路径。单个技能的坍缩是必要的（掌握该技能），但标准 RLVR 在所有叶节点累积这些局部熵减→**全局多样性不可逆丧失**（即策略坍缩）。

树状结构的 $\langle k\rangle\approx 2$ 统计隔离了前沿节点，使其与系统全局竞争解耦，因此每个技能的相变可以干净地发生——但正是这种"干净坍缩"的累积导致了宏观问题。

### 4. 最大挫败态 = 退火干预的最佳时机

在慢学习开始时，技能岛间**竞争达到峰值**（最大挫败态）：

- 各技能岛争夺 $\langle k\rangle\approx 2$ 约束下的有限连接
- 表现为单个问题准确率的剧烈振荡（frustration-induced forgetting）
- **悖论性地**，此时探索能力（best@k）也达到峰值

此后，策略坍缩逐步侵蚀探索能力→best@k 持续下降。

---

## Annealed-RLVR 算法

受模拟退火启发，利用"灾难性遗忘即拓扑加热"的双刃剑特性：

| 阶段 | 操作 | 物理类比 |
|------|------|---------|
| **触发时机检测** | 监控 V 型曲线底部 + 奖励曲线"膝点" | 识别最大挫败态 |
| **SFT 加热** | 对低准确率（<0.1）问题执行短暂 SFT | 高温扰动，打破过早承诺 |
| **RLVR 冷却** | 恢复标准 RLVR 训练 | 受控冷却，沉降到更优配置 |

**为什么此时 SFT 不会灾难性遗忘？** 因为技能岛尚未固化为单一概念网，系统对拓扑扰动具有最大鲁棒性。

**SFT 的作用机制**：为最难问题"焊接"初始路径，使其从准确率 ≈ 0 变为对后续 RLVR 探索"可见"。

---

## 实验结果

### CoNet 与 LLM 的动力学同构验证

| 宏观现象 | CoNet 复现 | LLM 观测 | 一致性 |
|---------|-----------|---------|--------|
| 两阶段学习曲线（快→慢） | ✓ | ✓ | 高度吻合 |
| V 型回复/路径长度 | ✓ | ✓ | 定性一致 |
| 灾难性遗忘 + 快速恢复 | ✓ | ✓ | 定性一致 |
| 策略坍缩（多样性下降） | ✓ | ✓ | 定性一致 |
| 概念网平均度 $\to 2$ | ✓ 直接度量 | — | 核心假说验证 |

### Annealed-RLVR vs 标准 RLVR（DeepSeek-R1-Distill-Qwen-1.5B）

| 评估集 | 标准 RLVR | Annealed-RLVR | 说明 |
|--------|----------|---------------|------|
| 训练集 512 题（best@k） | 基线 | **更优** | 分布内 |
| Minerva（best@k） | 基线 | **更优** | OOD |
| AIME 2024/2025（best@k） | 基线 | **更优** | OOD |

具体机制：SFT 加热使最难问题的成功率从 ≈ 0 提升，后续 RLVR 冷却将此优势放大→**未解题数系统性减少**，解题分布整体右移。

### 微观机制验证

- **挫败遗忘**：慢学习初期个体问题准确率剧烈振荡，部分技能被竞争性抑制
- **相变式学习**：CoNet 中表现为阶跃式跳变；LLM 中为陡峭但连续的梯度（有限尺寸效应导致的"圆化"）
- **SFT 断裂可视化**：统一概念网在 SFT 后瞬间碎裂为断开的子图；恢复 RLVR 50 步即重连

---

## 理论贡献与启发

**统一性**：一个简洁假说（$\langle k\rangle\approx 2$）同时解释四个看似无关的宏观现象，核心逻辑是稀疏图的拓扑约束决定了训练动力学。

**方法论创新**：CoNet 作为"计算显微镜"——无法直接分析 LLM 的推理图，但可用最小代理模型复现宏观行为后剖析微观结构。这是 RG 思想在 AI 中的具体实践。

**从描述到设计**：理论不仅解释现象，还直接指导算法设计——Annealed-RLVR 的每个设计选择（时机、对象、方式）都有明确的拓扑理论依据。

---

## 局限性与未来方向

- **模型规模**：仅在 1.5B 参数模型上验证，更大模型的稀疏网特性是否保持有待确认
- **CoNet 简化**：极简代理模型可能遗漏 LLM 的某些重要特性（如注意力机制的具体影响）
- **推理图提取**：从 LLM 直接提取完整微观推理图仍是开放问题，需要新的语义分块/聚类算法
- **退火时机自动化**：当前依赖手动识别 V 型底部和奖励膝点，自动化检测方法需进一步开发
- **理论边界**：稀疏网假说对非数学/代码领域（如自然语言理解）的适用性未经验证

## 相关工作与启发
- **vs 现有RLVR分析**: 碎片化分析各现象的起因，本文提供统一的拓扑结构解释
- **vs 课程学习**: Annealed-RLVR可以看作理论驱动的课程调度，比启发式方法有更强理论支撑

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 物理视角统一解释RLVR四大谜题，极具原创性
- 实验充分度: ⭐⭐⭐⭐ CoNet与LLM的对应验证充分，算法效果有实验支持
- 写作质量: ⭐⭐⭐⭐⭐ 思维清晰，可视化丰富，理论与直觉兼备
- 价值: ⭐⭐⭐⭐⭐ 对理解和改进RLVR训练有深远影响

<!-- RELATED:START -->

## 相关论文

- [Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning](../../CVPR2026/reinforcement_learning/reinforce_to_learn_elect_to_reason_a_dual_paradigm_for_video_reasoning.md)
- [How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)
- [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](on_the_generalization_of_sft_a_reinforcement_learning_perspective_with_reward_re.md)
- [The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)

<!-- RELATED:END -->
