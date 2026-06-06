---
title: >-
  [论文解读] Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning
description: >-
  [ICML 2026][LLM推理][最优控制] 将 LLM 推理建模为隐空间上的最优控制问题（线性二次调节器 LQR），提出 Test-Time Control (TTC) 层在前向传播中执行有限时域规划并解码最优控制动作作为下一 token 表示，配合辛迭代 CUDA 高效求解器…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "最优控制"
  - "LQR"
  - "测试时规划"
  - "状态空间模型"
  - "数学推理"
---

# Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning

**会议**: ICML 2026  
**arXiv**: [2603.09221](https://arxiv.org/abs/2603.09221)  
**代码**: https://vita-group.github.io/TTC-Net (项目页)  
**领域**: LLM推理  
**关键词**: 最优控制, LQR, 测试时规划, 状态空间模型, 数学推理  

## 一句话总结

将 LLM 推理建模为隐空间上的最优控制问题（线性二次调节器 LQR），提出 Test-Time Control (TTC) 层在前向传播中执行有限时域规划并解码最优控制动作作为下一 token 表示，配合辛迭代 CUDA 高效求解器，作为适配器插入预训练 LLM 后在 MATH-500 上提升最多 +27.8%，AMC/AIME 上 Pass@8 提升 2-3 倍。

## 研究背景与动机

**领域现状**：当前主流序列模型（Transformer、SSM、线性 RNN）共享一个核心设计原则——基于联想记忆的预测。Attention 保留全部 KV 缓存并通过查询匹配检索，线性 RNN 则将历史上下文压缩到固定大小隐状态再解码，本质上都是 System 1 式的快速模式匹配。

**现有痛点**：纯记忆范式在需要发现、推理和求解的任务上能力受限。虽然强化学习（RL）可以让模型更具目标导向性，但 RL 仅作为外部训练/后训练过程，在前向推理时缺席。模型学会了"优化什么"，却没有在计算过程中学会"如何通过规划来推理"。

**核心矛盾**：记忆架构对应 System 1 思维，而 System 2 式的深思熟虑、多步规划和长程推理需要专门的架构支持。RL 训练无法突破记忆架构施加的推理天花板，规划能力依然外挂于模型之外。

**本文目标**：将规划直接内化到模型架构中，使 LLM 在前向传播中就能执行目标导向的推理，而非依赖外部训练程序。

**切入角度**：作者观察到 LQR（线性二次调节器）是可解析求解的 MDP 子类，且线性动力系统已被证明能表达广泛的 MDP 族。如果将每一层的下一 token 预测建模为一个可微分的有限时域 LQR 问题，就能在前向推理时原生地执行规划。

**核心 idea**：用最优控制中的 LQR 规划替代纯记忆检索，让模型在预测前先"思考未来轨迹"，实现 System 2 推理的架构化。

## 方法详解

### 整体框架

TTC-Net 是一个混合架构：在预训练 Transformer 的 Attention 和 MLP 之间每隔 8 层插入一个 TTC 层。输入 token 特征经线性投影得到初始隐状态 $\boldsymbol{h}_0$，TTC 层在该隐状态上构建一个有限时域 LQR 问题并求解最优第一步动作 $\boldsymbol{u}_1^*$ 作为输出，经归一化和线性投影后加回残差流。整个过程端到端可微，支持从零训练或在预训练模型上微调。

### 关键设计

1. **Test-Time Control (TTC) 层**:

    - 功能：在前向推理时执行有限时域最优控制规划，将记忆状态解码为最优决策
    - 核心思路：给定编码上下文的初始状态 $\boldsymbol{h}_0$，构建线性状态转移 $\boldsymbol{h}_t = \boldsymbol{A}_t \boldsymbol{h}_{t-1} + \boldsymbol{B}_t \boldsymbol{u}_t$ 和二次代价函数 $\sum_{t=1}^{T}(\boldsymbol{h}_t^\top \boldsymbol{Q}_t \boldsymbol{h}_t + \boldsymbol{u}_t^\top \boldsymbol{R}_t \boldsymbol{u}_t)$，通过 Riccati 迭代求解最优第一步动作 $\boldsymbol{u}_1^* = \boldsymbol{K}_1^* \boldsymbol{h}_0$。LQR 参数由上下文 $\boldsymbol{h}_0$ 通过线性层动态生成（上下文化），并通过时间调制系数 $\boldsymbol{\Gamma}_\Box^t$ 实现时间异质参数化，使动力学和代价函数随规划步变化。反向传播通过 KKT 系统求解对偶 LQR 实现完全可微
    - 设计动机：现有记忆层（Attention/SSM）只能从过去上下文回忆信息，TTC 层则优化未来轨迹、最小化长程代价，为每个序列建模块赋予内在的值函数 $V_t(\boldsymbol{h}_t) = -\frac{1}{2}\boldsymbol{h}_t^\top \boldsymbol{P}_t \boldsymbol{h}_t$

2. **辛迭代高效求解器 (Symplectic Iteration Solver)**:

    - 功能：将经典 Riccati 递推中的顺序矩阵求逆替换为可并行的矩阵乘积链，实现 10 倍以上吞吐量提升
    - 核心思路：利用 LQR 动力学的辛结构，将 Riccati 递推重新表述为辛矩阵 $\boldsymbol{\Sigma}_t$ 的累积矩阵乘积。各时间步的矩阵逆 $\boldsymbol{A}_t^{-1}$ 和 $\boldsymbol{R}_t^{-1}$ 相互独立可完全并行计算，剩余顺序计算仅为矩阵乘法（可充分利用 Tensor Core）。通过对角化 $\boldsymbol{A}_t$ 和 $\boldsymbol{R}_t$ 将稠密矩阵求逆从 $O(T)$ 减至 $O(1)$。进一步融合为 CUDA kernel，行级分块、流式加载参数到 SRAM，并通过行归一化保持数值稳定
    - 设计动机：经典 Riccati 求解器需要顺序反向迭代 $T$ 步，每步含矩阵求逆（$O(Td^3)$），与 GPU 加速器严重不匹配。辛迭代将计算瓶颈从求逆转为乘法，前向缓存 $\boldsymbol{Y}_1$ 的 LU 分解和部分中间结果可复用于反向传播，消除额外的辛迭代开销

3. **混合架构与测试时规模化**:

    - 功能：将 TTC 层作为轻量适配器集成到预训练 LLM，并支持推理时灵活调整规划时域以提升性能
    - 核心思路：每 8 层 Attention 后插入 1 层 TTC（8:1 交错比），采用多头结构（头大小 16）。训练时从截断 Poisson 对数正态分布采样规划时域（均值 $T_\mu=8$，最大 32），避免固定时域导致的分布偏移。测试时可任意增大规划时域 $T_{test}$，模型在训练最大时域 32 的基础上可泛化至 $T=64$ 且性能持续提升。微调时输出投影 $\boldsymbol{W}_{out}$ 零初始化，保证初始模型与原始骨干一致
    - 设计动机：TTC 层需要丰富的记忆状态作为输入，必须与 Attention 交错使用。混合时域训练策略使模型适应不同规划深度，暴露出一个架构原生的测试时计算缩放轴

### 训练策略

采用混合时域训练：每次迭代从 Poisson 对数正态分布采样规划时域 $T_{train}$，均值 $T_\mu = 8$，对数标准差 $T_\sigma = 0.1$，上限 32。在预训练模型上微调时使用 OpenThoughts2-114K 数据集加 800K 自收集推理样本进行 SFT，相当于模仿学习 + 逆强化学习。

## 实验关键数据

### 主实验 — 数学推理（基于 Llama-3-Instruct-7B 微调）

| 模型 | MATH-500 | AMC Acc@8 | AMC Pass@8 | AIME24 Acc@8 | AIME24 Pass@8 | AIME25 Pass@8 |
|------|----------|-----------|------------|-------------|---------------|---------------|
| Base model | 25.00 | 6.63 | 31.32 | 0.00 | 0.00 | 0.00 |
| Full Finetuning | 46.80 | 20.78 | 46.98 | 1.67 | 6.67 | 0.00 |
| + Attention | 47.00 | 20.48 | 44.58 | 0.42 | 3.33 | 6.67 |
| + Mamba | 44.80 | 22.29 | 44.58 | 0.83 | 3.33 | 3.33 |
| + GDN | 47.80 | 17.77 | 37.35 | 0.42 | 3.33 | 6.67 |
| + MesaNet | 47.40 | 12.65 | 27.71 | 1.25 | 10.00 | 0.00 |
| **TTC-Net** | **52.80** | **23.34** | **54.22** | **3.33** | **20.00** | **20.00** |

### 消融实验 — MATH-500

| 配置 | $T_{test}=8$ | $T_{test}=16$ | 说明 |
|------|------------|-------------|------|
| 时间齐次参数化 | 48.40 | 45.70 | 去除时间调制，增大时域反而掉点 |
| 固定训练时域 | 50.60 | 31.50 | 无法泛化到更大测试时域 |
| 均匀采样时域 | 50.80 | 51.00 | 效果接近但训练成本翻倍 |
| Attn:TTC = 4:1 | 53.00 | — | 更多 TTC 层可提升但计算开销大 |
| Attn:TTC = 16:1 | 47.20 | — | TTC 层过少性能下降 |
| **TTC-Net (PLN + 8:1)** | **52.80** | **53.60** | 最优平衡点，可泛化至 $T=64$ |

## 亮点与洞察

- **架构范式转换**：首次将推理从"记忆检索"重新定义为"最优控制"，为 LLM 推理提供了 System 2 认知的架构化实现
- **测试时规模化新轴**：规划时域 $T$ 提供了一个正交于 token 生成数量的计算缩放轴，增大 $T$ 可持续提升推理准确率且不需要重新训练
- **突破推理天花板**：TTC-Net 在 AIME 上实现了从 0% 到 20% 的 Pass@8 突破，说明控制目标提供了记忆层无法达到的归纳偏置
- **辛迭代求解器**：通过算法-硬件协同设计实现 10 倍以上吞吐量提升，使最优控制层在大规模 LLM 中实际可用

## 局限性 / 可改进方向

- 多层 TTC 的联合动力学行为缺乏理论理解，层间交互机制不明确
- 当前仅在 7B 模型上验证，更大规模模型和全阶段训练（预训练 + RL）的效果未知
- LQR 的线性动力学和二次代价仍有表达力限制，非线性 MDP 公式化可能进一步提升
- 参数上下文化的线性层较简单，更丰富的世界模型参数化值得探索

## 相关工作与启发

- 与 TTT（Test-Time Training）系列工作形成对比：TTT 是测试时记忆（自监督回归），TTC 是测试时决策（最优控制）
- 与 RL for LLM（如 DeepSeek-R1）互补：RL 提供训练时目标，TTC 将目标内化到架构前向传播中
- 辛迭代求解器的设计思路可推广到其他需要在神经网络中嵌入优化层的场景
- Titans、DeltaNet 等记忆架构可与 TTC 混合，探索更丰富的记忆-规划交互

## 评分

- 新颖性: 9/10 — 首次将最优控制作为架构组件嵌入 LLM，范式性创新
- 实验充分度: 7/10 — Sudoku + 数学推理验证充分，但仅限 7B 模型且缺少 NLP/代码任务
- 写作质量: 9/10 — 从认知科学到控制论的叙事连贯，数学推导严谨
- 价值: 8/10 — 开辟了 LLM 推理的新架构方向，但实际落地需更大规模验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[ICML 2026\] Conformal Thinking: Risk Control for Reasoning on a Compute Budget](conformal_thinking_risk_control_for_reasoning_on_a_compute_budget.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)
- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)

</div>

<!-- RELATED:END -->
