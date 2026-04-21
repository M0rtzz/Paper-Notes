---
title: >-
  [论文解读] Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation
description: >-
  [ICLR 2026][LLM推理][GRPO] 揭示GRPO的优势函数（std归一化）导致更新幅度在中等难度题目处最大、对难题和易题均隐式抑制的问题，提出MathForge框架——DGPO（用MAD替换std实现难度均衡 + softmax难度加权）+ MQR（添加故事背景/抽象术语/嵌套子问题三方面改写增加难度但保留原答案），在Qwen2.5-Math-7B上在6个数学推理benchmark上平均超GRPO +4.56%。
tags:
  - ICLR 2026
  - LLM推理
  - GRPO
  - difficulty-aware
  - mathematical reasoning
  - RLVR
  - 数据增强
---

# Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation

**会议**: ICLR 2026  
**arXiv**: [2601.20614](https://arxiv.org/abs/2601.20614)  
**代码**: [GitHub](https://github.com/AMAP-ML/MathForge)  
**领域**: LLM推理 / 强化学习  
**关键词**: GRPO, difficulty-aware, mathematical reasoning, RLVR, data augmentation

## 一句话总结

揭示GRPO的优势函数（std归一化）导致更新幅度在中等难度题目处最大、对难题和易题均隐式抑制的问题，提出MathForge框架——DGPO（用MAD替换std实现难度均衡 + softmax难度加权）+ MQR（添加故事背景/抽象术语/嵌套子问题三方面改写增加难度但保留原答案），在Qwen2.5-Math-7B上在6个数学推理benchmark上平均超GRPO +4.56%。

## 研究背景与动机

**领域现状**：RLVR（验证奖励强化学习）已成为提升LLM数学推理能力的主流范式（DeepSeek-R1等），GRPO是其中最具代表性的算法——通过组内相对优势估计替代价值网络。

**现有痛点**：

1. **算法层面**：GRPO的优势函数 $\hat{A}_{GR,i} = \frac{r_i - \text{mean}}{\text{std}}$ 使用标准差归一化，导致更新幅度 $\sum|A|$ 与准确率 $p$ 的关系为 $2G\sqrt{p(1-p)}$——在 $p=0.5$ 时最大，而在 $p$ 接近0或1时衰减。这意味着更难的题目（$p$ 小但非零）的更新幅度小于中等难度题目

2. **数据层面**：现有RLVR数据增强（如Liang et al. 2025）主要做题目改述提升多样性，未系统性增加题目难度。缺乏挑战性的训练数据限制了模型推理能力的上界

**核心矛盾**：难但可解的题目是最理想的训练材料（暴露模型弱点且有正确答案可学），但GRPO恰恰在这类题目上更新幅度最小。

**本文切入角度**：在算法端和数据端同时解决"忽视难题"问题——DGPO修正GRPO的内在失衡并加权难题，MQR生成更难的训练题目。

## 方法详解

### 整体框架

原始训练数据 → MQR（三方面改写增加难度，保留原答案）→ 增强数据集（原始+改写）→ DGPO训练（MAD归一化 + 难度加权 + 有效token平均）→ 增强后的策略模型。MathForge形成协同循环：MQR扩展数据难度前沿，DGPO高效从增强数据中学习。

### 关键设计

1. **DGPO：难度感知群组策略优化**

    - **难度均衡优势估计 (DGAE)**：将GRPO的标准差归一化替换为均值绝对偏差（MAD）归一化：$\hat{A}_{DG,i} = \frac{r_i - \text{mean}(\{r_i\})}{\text{MAD}(\{r_i\})}$，其中 $\text{MAD} = \frac{1}{G}\sum|r_i - \text{mean}|$
    - **定理2证明**：DGAE下单题的总更新幅度 $\sum|\hat{A}_{DG,i}| = G$，为常数，不随难度变化——彻底消除了GRPO中 $2G\sqrt{p(1-p)}$ 的钟形偏差。且无需二值奖励假设
    - **难度感知问题级加权 (DQW)**：在均衡基础上进一步通过softmax加权优先更新难题：$\lambda_s = B_v \cdot \frac{\exp(D_s/T)}{\sum\exp(D_s/T)}$，其中 $D_s = -\text{mean}(\{r_{si}\})$ 为难度度量，$T=2.0$ 为温度
    - 有效token级平均：仅在有效查询（非全对或全错）上计算token级平均损失，防止梯度波动

2. **MQR：多方面问题改写**

    - 使用大推理模型（默认o3）对训练题目进行三种改写：
        - **添加故事背景**：嵌入叙事噪声，挑战模型从噪声中提取关键数学信息
        - **引入抽象术语**：抽象化具体概念，挑战模型理解抽象数学概念
        - **嵌套子问题**：增加推理步骤和跨领域知识要求
    - 关键约束：所有改写必须保留原始gold answer，免去答案重生成的开销
    - 设计动机：数学推理需要多种技能，系统性增加题目难度可推动模型性能边界

### 损失函数 / 训练策略

DGPO目标函数：

$$\mathcal{J}_{DGPO}(\theta) = \frac{1}{\sum_{s=1}^{B_v}\sum_{i=1}^{G}|o_{si}|}\sum_{s=1}^{B_v}\lambda_s\sum_{i=1}^{G}\sum_{t=1}^{|o_{si}|}\min[I_{sit}\hat{A}_{DG,si}, \text{clip}(I_{sit}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{DG,si}]$$

- 使用纯准确率奖励（$r \in \{0,1\}$），无KL散度
- 8×NVIDIA H20 GPU，基于Open-R1代码库
- DQW温度 $T=2.0$（保证batch内最大/最小权重比 $\leq e^{0.5} \approx 1.65$）

## 实验关键数据

### 主实验

Qwen2.5-Math-7B在MATH数据集训练，6个benchmark平均表现：

| 方法 | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | Avg. | $\Delta_{GRPO}$ |
|------|--------|--------|-------|---------|---------|----------|------|----------------|
| Base | 12.19 | 4.79 | 35.23 | 48.60 | 15.07 | 16.33 | 22.04 | - |
| GRPO | 20.94 | 8.44 | 58.98 | 72.20 | 27.76 | 37.33 | 37.61 | - |
| Dr.GRPO | 21.04 | 8.23 | 58.59 | 72.05 | 28.58 | 35.89 | 37.40 | -0.21 |
| DAPO | 21.25 | 8.75 | 58.20 | 72.70 | 29.50 | 37.22 | 37.94 | +0.33 |
| GRPO-AD | 21.56 | 9.48 | 59.06 | 73.25 | 29.14 | 37.07 | 38.26 | +0.65 |
| DGPO | 23.85 | 10.21 | 61.02 | 74.25 | 31.07 | 38.33 | 39.79 | **+2.18** |
| MQR | 25.00 | 11.77 | 59.38 | 77.85 | 31.43 | 40.81 | 41.04 | +3.43 |
| **MathForge** | 24.58 | 12.60 | 59.84 | 79.95 | 33.36 | 42.67 | **42.17** | **+4.56** |

### 消融实验

DGPO组件消融（Qwen2.5-Math-7B）：

| 设置 | Avg. | $\Delta_{GRPO}$ |
|------|------|----------------|
| GRPO | 37.61 | - |
| +有效token平均 | 37.71 | +0.10 |
| +DGAE | 38.65 | +1.04 |
| +DGAE+DQW (full DGPO) | 39.79 | +2.18 |

DQW温度敏感性：$T=1.0$ → 39.03, $T=2.0$ → **39.79**, $T=5.0$ → 39.53, $T=10.0$ → 39.27

跨模型泛化（均超GRPO）：Qwen2.5-Math-1.5B +4.45, Qwen2.5-3B +3.54, DeepSeek-Math-7B +2.86

### 关键发现

- DGAE和DQW分别贡献+0.94%和+1.14%，两者互补
- MathForge在所有测试模型（4种）上均一致性超过GRPO，证明模型无关性
- DGPO可与其他方法叠加：+GPG→+0.99, +DAPO→+1.97, +GSPO→+1.61
- DGPO训练的模型输出更简洁（Fig. 1b），说明学会了更高效的推理路径

## 亮点与洞察

- 理论贡献扎实：定理1/2严格证明了GRPO更新幅度的钟形偏差和DGAE的常数均衡，数学推导清晰
- "先均衡再加权"的两步设计（DGAE→DQW）比直接在GRPO上做难度加权（如GRPO-AD）更有效
- MQR的"保留答案"约束是关键设计：既增加难度又免去答案重生成，大幅降低数据增强成本
- DGPO+MQR的协同效应（42.17 > 39.79 + 41.04 - 37.61），而非简单加和

## 局限与展望

- MQR依赖大推理模型（o3）作为改写器，增加数据增强成本
- 仅在数学推理领域验证，未测试代码生成/逻辑推理等其他推理任务
- DQW的温度超参数需要调优（虽然$T=2.0$在所有实验中表现稳健）
- MAD归一化在奖励分布对称时等价于std归一化，理论优势在非二值奖励下更显著但未充分验证

## 相关工作与启发

- **vs GRPO**：GRPO的std归一化导致钟形更新偏差，DGPO用MAD实现常数更新幅度——这是一个简单但有效的修正
- **vs GRPO-AD (Zhang & Zuo 2025)**：GRPO-AD在GRPO基础上做难度加权但未修正底层失衡，效果有限（+0.65 vs DGPO +2.18）
- **vs DAPO/GPG**：这些方法关注采样和KL散度等方面，与DGPO正交且可叠加
- **数据增强启发**：MQR的"保留答案约束"是一个实用的设计原则——确保增强数据的数学等价性

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论洞察（定理1/2）深刻，MAD替换std的修正虽简单但有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 6个benchmark×4个模型×多组消融+动态分析+叠加实验
- 写作质量: ⭐⭐⭐⭐ 理论与实验结合紧密，消融全面
- 价值: ⭐⭐⭐⭐ 对RLVR训练的通用优化，DGPO可直接叠加到现有管线中

<!-- RELATED:START -->

## 相关论文

- [DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)
- [Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](../../ACL2025/llm_reasoning/chain_of_reasoning_unified_math.md)
- [Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](../../ACL2025/llm_reasoning/safe_math_reasoning.md)
- [AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)

<!-- RELATED:END -->
