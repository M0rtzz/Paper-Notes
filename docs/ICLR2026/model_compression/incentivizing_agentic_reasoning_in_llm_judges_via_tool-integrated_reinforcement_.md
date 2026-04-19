---
title: >-
  [论文解读] Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning
description: >-
  [ICLR 2026][模型压缩][LLM-as-a-Judge] 提出 TIR-Judge，一个端到端的 RL 框架，训练 LLM 评判模型在评估过程中交替使用推理和代码执行工具，在7个公开基准上以 8B 参数超越 32B 推理奖励模型，且无需蒸馏的 TIR-Judge-Zero 可自举提升。
tags:
  - ICLR 2026
  - 模型压缩
  - LLM-as-a-Judge
  - 工具集成推理
  - 强化学习
  - 代码执行
  - 评估
---

# Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2510.23038](https://arxiv.org/abs/2510.23038)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: LLM-as-a-Judge, 工具集成推理, 强化学习, 代码执行, 评估

## 一句话总结
提出 TIR-Judge，一个端到端的 RL 框架，训练 LLM 评判模型在评估过程中交替使用推理和代码执行工具，在7个公开基准上以 8B 参数超越 32B 推理奖励模型，且无需蒸馏的 TIR-Judge-Zero 可自举提升。

## 研究背景与动机
LLM评判模型（LLM-as-a-Judge）在LLM生态中日益关键——训练阶段提供偏好信号、推理阶段做 best-of-N 选择、评估阶段替代人工。但目前评判模型面临两大问题：

**纯文本推理的天花板**：现有推理增强的评判模型（如JudgeLRM、J1-Judge）仅依赖文本推理链，in需要精确计算或符号推理的场景下力不从心（如验证代码输出、检查指令约束）

**工具使用的局限**：少数尝试引入工具的方法存在(i)仅在推理时使用工具而非训练时优化，(ii)局限于特定任务/领域

核心idea：用强化学习端到端训练评判模型学会何时调用代码解释器、如何基于执行结果迭代精化推理，实现推理与工具使用的深度融合。

## 方法详解

### 整体框架
TIR-Judge 基于多轮工具集成推理(TIR)构建评判轨迹 $s_k = \{r_1,c_1,o_1,...,r_k,c_k,o_k\}$，其中 $r_i$ 是推理步骤、$c_i$ 是生成的代码、$o_i = \mathcal{I}(c_i)$ 是执行结果。使用DAPO（GRPO改进版）进行RL训练。支持Pointwise/Pairwise/Listwise三种评判格式。

### 关键设计

1. **多样化训练数据构建**:

    - 功能：平衡可验证域（数学、编程）和不可验证域（对话、安全、通用代码）的训练数据
    - 核心思路：从HelpSteer3、UltraInteract、CodeRM等收集真实偏好对；从Qwen3-8B/14B等多个模型采样生成合成偏好对并自动验证。共约26K偏好对，覆盖多域多格式
    - 设计动机：让模型学会何时调用工具有用（可验证场景）、何时纯推理即可（不可验证场景）

2. **三维度奖励设计**:

    - 功能：引导模型同时优化正确性、格式规范和工具使用质量
    - 核心思路：$R = R_c \times (0.1 + 0.9 \cdot \mathbb{I}[R_t = 1 \wedge R_f = 1])$
        - 正确性奖励 $R_c$：预测是否匹配ground truth偏好
        - 格式奖励 $R_f$：输出是否符合结构化格式（\<score\>标签、\<preference\>标签等），对安全/通用场景要求不使用工具才给正分
        - 工具奖励 $R_t$：代码无错误且不超过3次调用
    - 设计动机：仅在"三者兼得"时给满分，单独的正确性只给10%奖励，避免不规范但碰巧正确的行为

3. **迭代自举训练策略 (TIR-Judge-Zero)**:

    - 功能：无需教师蒸馏，纯RL自举提升
    - 核心思路：交替执行 RL→拒绝采样→SFT→RL 循环：$\mathcal{T}_{t+1} \leftarrow \text{RS}(\pi_{\theta_t}), \pi_{\theta_{t+1}} \leftarrow \text{SFT}(\pi_{\theta_0}, \mathcal{T}_{t+1}), \pi_{\theta_{t+1}} \leftarrow \text{RL}(\pi_{\theta_{t+1}})$。每个prompt只保留最短/工具调用最少的正确轨迹以提高效率
    - 设计动机：证明TIR评判模型可不依赖蒸馏自我进化，降低对强教师模型的依赖

### 其他训练细节
- 骨干模型Qwen3-8B和Qwen3-4B；错误信息截断到最后一行防止上下文过长；执行结果在loss计算中被mask防止过拟合
- 蒸馏版使用 Gemini-2.5-Flash 作教师，收集约10K高质量轨迹
- 8张H100 80G GPU训练

## 实验关键数据

### 主实验（Pointwise + Pairwise）
| 模型 | PPE Avg | IFBench | CJBench | RWBench | RMBench | JGBench |
|------|---------|---------|---------|---------|---------|---------|
| Qwen3-8B Pointwise | 60.6 | 56.2 | 16.6 | 76.5 | 66.9 | 50.8 |
| Qwen3-8B Pairwise | 65.5 | 61.3 | 60.8 | 87.0 | 77.9 | 67.5 |
| Gemini-2.5-Flash Pairwise | 74.8 | 69.3 | 66.5 | 93.4 | 81.9 | 75.4 |
| **TIR-Judge (下文推断)** | **~70+** | **~66+** | **~63+** | **~90+** | — | — |

### 消融：Zero vs Distill
| 配置 | 规模 | 说明 |
|------|------|------|
| TIR-Judge-Zero (4B) | 4B | 纯RL自举，比蒸馏版高1.2% |
| TIR-Judge-Distill (4B) | 4B | 蒸馏冷启动后RL |
| TIR-Judge-Zero (8B) | 8B | 超越32B推理奖励模型 |

### 关键发现
- TIR-Judge 在 Pointwise 上提升最高6.4%，Pairwise 上提升最高7.7%，超越纯推理评判基线
- 8B参数的 TIR-Judge 在 PPE 上超越 32B 推理奖励模型
- TIR-Judge-Zero 在 4B 规模上反超蒸馏版1.2%，说明纯RL自举是可行且更优的策略
- Listwise 设置中达到 Claude-Opus-4 96% 的性能

## 亮点与洞察
- 将RL+工具使用从数学推理迁移到评判任务是一个natural但很有效的方向扩展
- 三维度奖励设计（正确性×格式×工具质量）的乘法结构巧妙，避免了简单加权的调参困难
- TIR-Judge-Zero 不依赖蒸馏的纯自举训练挑战了"需要强教师冷启动"的常见假设

## 局限与展望
- 在安全/通用域强制不使用工具可能过于简单，某些安全评估场景也可能受益于工具
- 多轮工具调用上限设为3可能限制了复杂评估任务的能力
- 实验主要在推理相关benchmark上表现最佳，在开放式对话评判上的优势需更多验证

## 相关工作与启发
- **vs JudgeLRM/J1-Judge**: 这些方法仅增强文本推理链，TIR-Judge 额外引入代码执行实现精确验证
- **vs AgentRM**: AgentRM 在推理时使用工具但未在训练时优化，TIR-Judge 端到端联合训练

## 评分
- 新颖性: ⭐⭐⭐⭐ TIR在评判任务的首次系统应用
- 实验充分度: ⭐⭐⭐⭐⭐ 7个benchmark、3种评判格式、Zero/Distill消融
- 写作质量: ⭐⭐⭐⭐ 框架清晰，细节充分
- 价值: ⭐⭐⭐⭐⭐ 8B模型超越32B，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows](multi-view_encoders_for_performance_prediction_in_llm-based_agentic_workflows.md)
- [Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)
- [A State-Transition Framework for Efficient LLM Reasoning](a_state-transition_framework_for_efficient_llm_reasoning.md)

<!-- RELATED:END -->
