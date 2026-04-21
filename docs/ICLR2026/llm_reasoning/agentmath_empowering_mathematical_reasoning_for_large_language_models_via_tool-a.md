---
title: >-
  [论文解读] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent
description: >-
  [ICLR 2026][LLM推理][数学推理] AgentMath提出一个工具增强的Agent框架，通过自动化数据合成、多轮交互式强化学习和高效异步训练系统，将LLM推理能力与代码解释器的计算精度无缝结合，在AIME24/25和HMMT25上以30B-A3B规模达到SOTA水平（90.6%/86.4%/73.8%），超越o3-mini和Claude-Opus-4.0-Thinking。
tags:
  - ICLR 2026
  - LLM推理
  - 数学推理
  - 工具增强
  - 强化学习
  - 代码解释器
  - Agent框架
---

# AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent

**会议**: ICLR 2026  
**arXiv**: [2512.20745](https://arxiv.org/abs/2512.20745)  
**代码**: 无  
**领域**: LLM Reasoning  
**关键词**: 数学推理, 工具增强, 强化学习, 代码解释器, Agent框架

## 一句话总结
AgentMath提出一个工具增强的Agent框架，通过自动化数据合成、多轮交互式强化学习和高效异步训练系统，将LLM推理能力与代码解释器的计算精度无缝结合，在AIME24/25和HMMT25上以30B-A3B规模达到SOTA水平（90.6%/86.4%/73.8%），超越o3-mini和Claude-Opus-4.0-Thinking。

## 研究背景与动机
大型推理模型（LRM）如o3和DeepSeek-R1在长链思维推理上取得了显著进展，但在处理需要精确数学运算的问题时仍存在计算效率低和准确性不足的问题——纯文本推理的固有局限导致频繁的计算错误和冗余校正。现有工具增强方法面临三大挑战：(1) 高质量工具使用数据极度稀缺，手动标注成本高且不可扩展；(2) Agent强化学习在工具使用策略优化方面的潜力尚未被充分探索；(3) 竞赛级数学问题涉及超长推理链（96k tokens、96次工具调用），传统批同步RL训练无法胜任。本文的核心idea是：构建一个端到端的Agent框架，通过自动化数据合成解决数据稀缺、通过Agentic RL学习最优工具使用策略、通过异步训练架构解决效率瓶颈。

## 方法详解

### 整体框架
AgentMath将工具增强数学推理建模为马尔可夫决策过程（MDP），LLM策略生成交替的推理片段和可执行代码块，通过沙箱环境交互。系统采用结构化标记协议：`<think>`标记自然语言推理，`<code>`标记可执行代码，`<interpreter>`封装执行反馈。整体流程分两阶段：(1) 在合成的工具增强轨迹上进行SFT建立初始工具使用能力；(2) 通过大规模RL驱动探索最优工具使用策略。

### 关键设计

1. **工具驱动数据合成（Tool-Driven Data Synthesis）**: 三阶段自动合成管线。**阶段1**：从AM-Thinking和Open-Thoughts等公开数据源聚合纯文本长CoT数据，经过N-gram过滤消除评测集重叠，获得346k高质量数据；然后用DeepSeek-V3作为教师模型，将计算密集步骤替换为可执行代码块，保留简单计算为文本形式以防止过度依赖工具。**阶段2**：多维度质量精炼——格式一致性校正、代码可执行性验证（沙箱执行）、环境反馈对齐（用Qwen3-32B判断一致性，将模拟输出替换为真实执行结果）、工具使用合理性评估（通过AST深度和行数约束排除不必要代码）。**阶段3**：自纠正能力注入——从执行失败的轨迹中采样，让教师模型生成"诊断错误→修复代码→重新执行→继续推理"的纠正轨迹。最终产出316k工具增强训练集，平均每样本8.3次工具调用、16.9k tokens。

2. **Agent专用强化学习（Agentic RL）**: 基于GRPO优化算法，引入三个系统创新。**(a) 交替代码执行的Agent轨迹**：在rollout过程中通过"生成-暂停-执行-恢复"循环构建混合轨迹，工具调用上限为T次。**(b) 选择性损失掩码**：优势信号仅应用于`<think>`和`<code>`段的token，`<interpreter>`段的环境反馈token在优化时被掩码，确保梯度更新只来自模型自身的决策。**(c) 自适应批构建**：过滤全部正确或全部错误的问题（学习信号有限），通过反填保持批大小一致性。

3. **复合奖励设计**: 奖励函数整合答案正确性和工具使用效率：$R_{total} = R_{acc} + \mathbb{I}(R_{acc}=1) \cdot R_{tool}$。其中$R_{acc}$为基于数学等价性的二值反馈，$R_{tool} = \min(R_{max}, \alpha + \beta \cdot N_{code})$在答案正确时激励高效的工具利用。

4. **可扩展Agent RL基础设施**: 三大技术解决超长序列+高频工具交互的训练瓶颈。**(a) 分布式代码执行沙箱集群**：将CPU密集的代码执行从训练循环中卸载，将工具调用延迟从175s降至1.2s。**(b) 请求级异步Rollout调度**：每个轨迹作为独立的长运行请求，推理引擎和Agent通过异步通信解耦；请求暂停等待工具调用时，引擎立即处理其他就绪请求，消除队头阻塞。**(c) Agent部分Rollout**：将长轨迹分解为预算受限的片段（$\tau = \tau^{(1)} \oplus \tau^{(2)} \oplus \ldots$），每段受最大生成长度$L_{seg}$和最大工具调用数$T_{seg}$约束，防止单条轨迹垄断资源，实现2.2-2.5x加速。**(d) 前缀感知加权负载均衡**：根据前缀长度分配动态权重$w_j = \lfloor L_j / L_{base} \rfloor + w_{base}$，配合LRU粘性会话最大化KV-cache复用。整体实现4-5x训练加速。

### 损失函数 / 训练策略
- **SFT阶段**：选择性反馈掩码的自回归损失，$\mathcal{L}_{SFT-masked} = -\sum_t \sum_k (1 - \mathbb{I}(z_{t,k})) \log \pi_\theta(z_{t,k} | \cdot)$，掩码`<interpreter>`段token
- **RL阶段**：多阶段自适应策略，当截断率超过10%时自动扩展：上下文长度从48k→72k→96k，工具调用上限从48→72→96，部分rollout数从2→3→4
- 使用Llama-Factory进行SFT（6 epochs，学习率6e-5），使用verl 0.5.0进行RL（学习率1e-6，batch size 64，每问题8个rollouts）

## 实验关键数据

### 主实验

| 数据集 | 指标 | AgentMath-8B | AgentMath-30B-A3B | AgentMath-235B-A22B-SFT | 之前SOTA (同规模) | 提升 |
|--------|------|------|------|------|----------|------|
| AIME24 | avg@32 | 89.8% | 90.6% | 93.4% | 86.0% (DS-0528-Qwen3-8B) | +3.8% |
| AIME25 | avg@32 | 84.7% | 86.4% | 90.8% | 76.3% (DS-0528-Qwen3-8B) | +8.4% |
| HMMT25 | avg@32 | 71.3% | 73.8% | 81.7% | 61.5% (DS-0528-Qwen3-8B) | +9.8% |

AgentMath-30B-A3B（仅3B激活参数）在AIME24/25上超越OpenAI-o3-mini (87.3%/86.3%)和Claude-Opus-4.0-Thinking (83.0%/72.0%)，逼近DeepSeek-R1-671B (91.4%/87.5%)。

### 消融实验

| 配置 | AIME24 | AIME25 | 说明 |
|------|---------|------|------|
| 未精炼合成数据 | 35.3% | 25.7% | 格式不一致和不可执行代码导致性能差 |
| + 格式一致性校正 | 47.4% | 40.1% | +12.1%/+14.4% |
| + 代码可执行性验证 | 52.8% | 44.8% | +5.4%/+4.7% |
| + 环境反馈对齐 | 56.3% | 48.3% | +3.5%/+3.5% |
| + 自纠正能力注入 | 58.6% | 50.8% | +2.3%/+2.5% |
| + SFT选择性掩码 | 60.5% | 53.3% | 最终SFT性能 |
| Text-Based-SFT vs AgentMath-SFT | 57.1% vs 60.5% | 49.2% vs 53.3% | 工具增强数据优势 |
| Text-Based-RL vs AgentMath-RL | 68.7% vs 76.2% | 57.5% vs 67.5% | RL阶段4x效率提升 |

### 训练效率

| 方法 | 每步时间 | 加速比 |
|------|---------|------|
| 静态批同步Rollout | 3600-4000s | - |
| + 请求级异步调度 | 2100-2500s | 1.5-1.8x |
| + Agent部分Rollout | 1100-1300s | 3.0-3.3x |
| + 前缀感知负载均衡 | 750-900s | 4.0-5.0x |

### 关键发现
- 工具增强模型在RL中仅需约400步即达到76.2%（AIME24），而纯文本模型需要约1600步才达到68.7%，效率提升4x
- 多阶段RL训练中出现了涌现的代码自纠正能力
- 推理序列长度减少约4k tokens（~14%），工具代码替代了冗长的手动计算
- 数据从2k扩展到300k时，AIME24从27.2%提升到78.4%，展现良好的scaling law

## 亮点与洞察
- **系统性解决三大瓶颈**：数据稀缺（自动合成管线）、策略优化（Agentic RL）、训练效率（异步基础设施），形成完整的技术闭环
- **涌现的代码自纠正能力**：RL训练中模型自主学会了诊断和修复代码错误的能力，这是未被显式训练的涌现行为
- **MoE模型的惊人效率**：30B-A3B模型仅用3B激活参数就接近671B参数模型的性能，说明工具增强策略可以大幅弥补参数量的不足
- **部分Rollout的精妙设计**：将超长轨迹分解为可管理的片段，既解决了长尾延迟问题，又不损害性能（accuracy~70%在不同N设置下保持一致）

## 局限与展望
- 235B规模模型由于算力限制仅进行了SFT，未做RL训练，可能还有更大提升空间
- 目前仅关注数学竞赛基准测试，未验证在科学推理、工程计算等更广泛场景的泛化性
- 复合奖励函数中的工具使用奖励设计相对简单，可能无法精细引导最优的工具调用时机
- 代码解释器目前限于Python/SymPy，未探索其他计算工具（如Mathematica、SageMath）的集成

## 相关工作与启发
- **与ToRL/ReTool的对比**：这些方法也探索了RL+工具使用，但在数据质量和训练效率上不及AgentMath，且改进幅度有限
- **与CoRT的对比**：CoRT依赖高质量人工标注，不可扩展；AgentMath的自动合成管线解决了这个问题
- **工程启发**：异步训练系统的设计思路（请求级调度+部分Rollout+前缀感知LB）具有很强的通用性，可迁移到其他Agent RL场景
- **关于Agent系统设计**：本文表明GRPO等简单的outcome-based奖励在Agent场景中就足够有效，无需复杂的process reward

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)
- [Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models](../../ACL2025/llm_reasoning/chain-of-reasoning_towards_unified_mathematical_reasoning_in_large_language_mode.md)
- [Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](../../ACL2025/llm_reasoning/chain_of_reasoning_unified_math.md)
- [Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](../../ACL2025/llm_reasoning/safe_math_reasoning.md)
- [Conflict-Aware Fusion: Resolving Logic Inertia in Large Language Models via Structured Cognitive Priors](conflict-aware_fusion_resolving_logic_inertia_in_large_language_models_via_struc.md)

<!-- RELATED:END -->
