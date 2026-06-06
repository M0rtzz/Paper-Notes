---
title: >-
  [论文解读] Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation
description: >-
  [CVPR2026][多模态VLM][GUI自动化] 提出 CES（Coordinator-Executor-State Tracker）多智能体框架和分阶段执行反馈强化学习算法，将高层任务规划与低层执行解耦…
tags:
  - "CVPR2026"
  - "多模态VLM"
  - "GUI自动化"
  - "长时序任务"
  - "多智能体框架"
  - "强化学习"
  - "状态追踪"
  - "任务调度"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation

**会议**: CVPR2026  
**arXiv**: [2511.22235](https://arxiv.org/abs/2511.22235)  
**代码**: [hehehahi4/CES](https://github.com/hehehahi4/CES)  
**领域**: 多模态VLM  
**关键词**: GUI自动化, 长时序任务, 多智能体框架, 强化学习, 状态追踪, 任务调度

## 一句话总结

提出 CES（Coordinator-Executor-State Tracker）多智能体框架和分阶段执行反馈强化学习算法，将高层任务规划与低层执行解耦，通过专门训练的 Coordinator 和 State Tracker 显著提升 GUI Agent 在长时序任务上的规划和状态管理能力。

## 背景与动机

1. **单智能体能力冲突**：现有端到端 GUI Agent 试图在单一模型中耦合任务规划、多步推理、GUI元素定位和精确动作执行等异质能力，有限参数下难以同时掌握高层和低层能力，随着任务复杂度增加容易出现灾难性能力崩溃
2. **缺乏任务状态感知**：长时序任务中，Agent 主要依赖截图推断进度，但截图是不充分且不可靠的状态表示——重复出现的主页界面、OOD界面等都会导致进度判断困难
3. **SFT 范式局限**：监督微调严重依赖大规模高质量轨迹标注数据，成本高昂且泛化能力差
4. **单步 RL 不足**：现有 RL 方法虽在简单任务上取得不错效果，但仍训练单一策略网络，未解决能力耦合问题
5. **Multi-Agent 缺优化**：已有多智能体框架多用通用 VLM + prompt engineering 扮演角色，缺乏对各角色的深度专门优化
6. **时序验证实验**：论文设计了截图时序判断实验，发现随步间距增大准确率急剧下降，实证表明截图无法充分表征任务状态

## 方法详解

### 整体框架：CES 协作循环

类比操作系统设计，三个专门化 Agent 形成循环协作：

- **Coordinator（CPU/规划核心）**：融合用户高层指令 $q$、State Tracker 提供的状态摘要 $m^{t-1}$ 和当前截图 $s^t$，分解生成清晰的原子指令 $l^t = \pi_c(q, m^{t-1}, s^t)$
- **Executor（I/O设备/执行终端）**：冻结的预训练 GUI 模型，仅需根据原子指令 $l^t$ 和当前截图 $s^t$ 执行动作 $u^t = (th^t, a^t) = \pi_e(l^t, s^t)$，无需理解长期意图
- **State Tracker（动态内存）**：语言模型，不直接感知 GUI 环境，通过理解 Executor 输出 $u^t$、用户意图 $q$ 和上一步状态 $m^{t-1}$，生成新的高语义自然语言状态摘要 $m^t = \pi_s(q, m^{t-1}, u^t)$

### 分阶段执行反馈强化学习

**Warm-up SFT**：先用已有轨迹数据对 Coordinator 和 State Tracker 做监督微调，使其学会基本角色职责和输出格式。

**执行反馈奖励函数**：不直接评价中间输出质量，而是将输出传递至 Executor 执行，用规则化奖励函数客观评分：

$$R = \alpha_1 R_{format} + \alpha_2 R_{executor}, \quad R_{executor} = \gamma_1 R_{type} + \gamma_2 R_{param}$$

其中 $R_{format}$ 奖励格式正确性，$R_{type}$ 奖励动作类型正确，$R_{param}$ 奖励动作参数正确。

**Stage 1 — 训练 Coordinator**：冻结 Executor，使用 ground-truth 状态作为 $m^{t-1}$ 输入，基于 GRPO 算法和执行反馈奖励优化 Coordinator 的规划策略。

**Stage 2 — 训练 State Tracker**：冻结已训练的 Coordinator 和 Executor，State Tracker 生成的状态摘要经过完整 CES 循环，最终的执行反馈奖励回传用于优化 State Tracker，使其学会生成对 Coordinator 最有价值的状态信息。

### 训练细节

- Coordinator 基座：Qwen2.5-VL-7B；State Tracker 基座：Qwen3-4B
- SFT 阶段：LLaMA Factory，1 epoch，lr=5e-5
- RL 阶段：Verl 框架，Coordinator 10 epochs lr=1e-6，State Tracker 5 epochs
- 奖励系数：$\alpha_1=0.1, \alpha_2=0.9$；$\gamma_1=0.2, \gamma_2=0.8$
- 训练 Executor：GUI-R1-7B（冻结）；硬件：8×80G GPU

## 实验关键数据

### 主实验：长时序任务性能（Table 1）

在 AITZ（平均7.5步）、AMEX（平均12.8步）、GUI-Odyssey（平均15.3步）三个benchmark上：

| 模型 | 方法 | AITZ SR | AMEX SR | GUI-Odyssey SR |
|------|------|---------|---------|----------------|
| Qwen2.5-VL-7B | Zero Shot | 18.11 | 35.10 | 34.37 |
| GUI-R1-7B | RL | 30.59 | 43.69 | 38.79 |
| GUI-Owl-7B | RL | 32.70 | 40.48 | 35.82 |
| + GPT-5 | Multi-Agent | 40.55 | 35.80 | 42.47 |
| **+ CES (Ours)** | **Multi-Agent** | **43.05** | **48.48** | **53.69** |

CES 在 GUI-R1-7B 基线上平均 Type 准确率提升 10.38%，GUI-Odyssey SR 从 38.79% 提升至 53.69%（+14.9%）。

### 泛化性验证（Table 2）

CES 作为即插即用模块在不同规模 Executor 上均显著提升：

| Executor | 设置 | AMEX SR | GUI-Odyssey SR |
|----------|------|---------|----------------|
| UI-R1-3B | Baseline → CES | 35.81 → 43.38 (+7.57) | 32.49 → 38.04 (+5.55) |
| GUI-Owl-7B | Baseline → CES | 40.48 → 47.24 (+6.76) | 35.82 → 46.65 (+10.83) |
| GUI-Owl-32B | Baseline → CES | 43.16 → 52.05 (+8.89) | 39.60 → 56.75 (+17.15) |

### 消融实验（Table 3）

| 配置 | AMEX SR | GUI-Odyssey SR |
|------|---------|----------------|
| CES 完整 | 48.48 | 53.69 |
| w/o Coordinator | 33.27 (-15.21) | 39.15 (-14.54) |
| w/o State Tracker | 42.08 (-6.40) | 42.52 (-11.17) |
| w/o RL (SFT only) | 36.54 (-11.94) | 42.89 (-10.80) |

去掉任一组件或RL阶段均导致显著性能下降，验证了各组件和训练策略的必要性。

## 亮点

1. **类OS设计理念**：将 GUI 自动化类比为操作系统的 CPU-I/O-Memory 架构，优雅地解耦规划、执行和状态管理
2. **State Tracker 创新**：用纯语言模型做动态上下文压缩和状态摘要，将状态理解从高维视觉空间转移到低维语义空间，几乎完全消除了 State Loss 错误（14%→2%）
3. **执行反馈奖励**：巧妙解决抽象任务（规划/状态跟踪）难以直接评估的问题，用下游执行结果反向指导上游优化
4. **即插即用泛化性**：7B Coordinator + 4B State Tracker 的轻量组合即可让不同 Executor 大幅受益，甚至 7B+4B 组合可达到 32B 单模型的效果
5. **实证验证充分**：时序判断预实验、三个长时序benchmark、多规模Executor泛化、详细消融和失败案例分析

## 局限与展望

1. **Executor 仍是瓶颈**：失败案例分析显示性能瓶颈已转移至 Executor 的感知限制（Perception Error 和 Generalization Failure），CES 无法解决这部分问题
2. **分阶段训练非联合优化**：Coordinator 和 State Tracker 分开训练，未探索联合训练或协同进化的可能性（论文 Future Work 中提及）
3. **领域适用性**：仅在移动端 GUI 场景验证，未扩展至 Web、桌面等其他 GUI 环境
4. **状态摘要质量依赖**：Stage 1 依赖 ground-truth 状态标注，获取此类标注在实际场景中的可行性待验证
5. **计算开销**：三个模型串行推理（7B+冻结Executor+4B），推理延迟可能是实际部署的障碍

## 与相关工作的对比

- **vs GUI-R1**：同样用 RL 训练 GUI Agent，但 GUI-R1 训练单一端到端模型，CES 解耦高层/低层并专门优化高层调度，GUI-Odyssey SR 从 38.79% 提升至 53.69%
- **vs SWIRL**：同为多阶段工作流方法，SWIRL 在 GUI-Odyssey 上 SR=51.65%，CES 达到 53.69% 并在其他benchmark上同样领先
- **vs Mobile-Agent-v3 / MobiAgent**：同为多智能体框架，但这些方法用 prompt engineering 做角色分配缺乏专门优化，CES 通过执行反馈 RL 对各角色深度训练
- **vs GPT-5 Multi-Agent**：用 GPT-5 做 Coordinator 和 State Tracker 效果不稳定（部分指标下降），而 CES 的专门训练模型显著且稳定地优于 prompt 方案

## 评分

- 新颖性: ⭐⭐⭐⭐ — OS类比的三角色解耦设计 + 执行反馈RL训练范式有较好原创性
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个benchmark、多规模泛化、详细消融、失败案例分析、预实验验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，预实验设计巧妙
- 价值: ⭐⭐⭐⭐ — 即插即用的高层调度模块对 GUI Agent 社区有实用价值，分阶段训练思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](../../ICML2026/multimodal_vlm/learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_highfidelity_incontext_learning_for_multimo.md)

</div>

<!-- RELATED:END -->
