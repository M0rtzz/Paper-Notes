---
title: >-
  [论文解读] RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation
description: >-
  [ICLR 2026][机器人][中间表示] 提出 RoboInter 操作套件——统一的中间表示数据/基准/模型资源：RoboInter-Tool（半自动标注 GUI）+ RoboInter-Data（23 万 episode × 571 场景 × 10+ 类中间表示的密集逐帧标注）+ RoboInter-VQA（29 类具身 VQA 基准）+ RoboInter-VLA（支持模块化和端到端的 plan-then-execute 框架），为通过中间表示提升 VLA 泛化提供完整基础设施。
tags:
  - ICLR 2026
  - 机器人
  - 中间表示
  - VLA
  - 操作数据集
  - 具身VQA
  - plan-then-execute
---

# RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation

**会议**: ICLR 2026  
**arXiv**: [2602.09973](https://arxiv.org/abs/2602.09973)  
**代码**: [GitHub](https://github.com/RoboInter)  
**领域**: 机器人学习 / 数据集  
**关键词**: 中间表示, VLA, 操作数据集, 具身VQA, plan-then-execute

## 一句话总结

提出 RoboInter 操作套件——统一的中间表示数据/基准/模型资源：RoboInter-Tool（半自动标注 GUI）+ RoboInter-Data（23 万 episode × 571 场景 × 10+ 类中间表示的密集逐帧标注）+ RoboInter-VQA（29 类具身 VQA 基准）+ RoboInter-VLA（支持模块化和端到端的 plan-then-execute 框架），为通过中间表示提升 VLA 泛化提供完整基础设施。

## 研究背景与动机

**领域现状**：VLA（Vision-Language-Action）系统将大规模预训练 VLM 与机器人操作相结合，但现有操作数据集存在成本高、embodiment 特异、覆盖不足等问题。Plan-then-execute 范式（先生成高层规划再翻译为低层动作）已被验证是提升泛化的有效思路，但其核心依赖中间表示（subtask、trace、grounding 等）的监督信号。

**现有痛点**：

1. 现有数据集几乎不提供密集的中间表示标注 → 限制 plan-then-execute 方法的发展
2. 已有标注工作要么规模小（ShareRobot 仅 51k），要么标注类型单一（LLARVA 只有 trace），要么依赖自动标注质量不可控（ECoT）
3. 缺乏系统评估 VLM 在具身场景中空间/时序推理能力的基准
4. 模块化 vs 端到端 VLA 的对比缺乏统一框架和数据支撑

**核心矛盾**：plan-then-execute 范式的潜力已被验证，但缺乏大规模、高质量、多类型的中间表示标注数据来真正释放这一潜力。

**本文方案**：构建完整的中间表示生态系统——从标注工具（RoboInter-Tool）到数据（RoboInter-Data）到基准（RoboInter-VQA）到模型框架（RoboInter-VLA），一站式解决数据、评估和方法三大瓶颈。

## 方法详解

### 整体框架

RoboInter 操作套件由四大组件构成：

1. **RoboInter-Tool**：基于 PyQt5 的轻量级 GUI，支持半自动逐帧标注，集成 SAM2 进行物体分割跟踪
2. **RoboInter-Data**：23 万+ episode、571 场景、10+ 类中间表示的密集逐帧标注数据
3. **RoboInter-VQA**：9 类空间 + 20 类时序的具身 VQA 基准和数据集
4. **RoboInter-VLA**：VLM Planner + Executor 的 plan-then-execute 框架，支持三种变体

### 关键设计一：多层次中间表示标注体系

数据标注覆盖三个层次：

- **任务层**：任务分解为子任务 → 15 种预定义原始技能（Pick, Place, Push, Pull 等）→ 片段级和视频级语言标注
- **物体层**：通过 SAM2 + 人工审查 → 6100 万帧物体 grounding 标注、19 万 affordance box 和 placement proposal
- **执行层**：末端执行器 2D 轨迹（7000 万帧 trace 标注）、接触点、6D 抓取位姿、夹爪 bounding box

所有标注在时间上与动作、状态、第三视角和腕部视角观测严格对齐。

### 关键设计二：F-CoT 灵活思维链桥接规划与执行

引入 Flexible Chain-of-Thought（F-CoT），由多种中间表示组合而成：

- 作为 VQA 监督信号训练 Planner
- 作为动作对齐的指导信号输入 Executor
- 支持文本形式（Te-Modular）和视觉提示形式（Im-Modular）
- 用户可按任务需求灵活选择子集（subtask + trace、affordance + skill 等）

### 关键设计三：三种 Plan-then-Execute VLA 变体

| 变体 | 架构特点 | 中间表示使用方式 |
|------|---------|----------------|
| IC-E2E | VLM 直接作为 Executor 的特征提取器 | 隐式（通过预训练权重） |
| EC-E2E | VLM 同时生成中间表示和动作 | 显式（联合优化 CoT + action） |
| Modular | Planner 和 Executor 分离 | 显式（Planner 输出 → Executor 条件） |

Executor 采用 Qwen2.5-VL backbone + DiT action head + 信息聚合器（将所有 token 隐藏状态压缩为可控长度的条件特征）。

## 实验与结果

### 主实验：第三方基准上的 VLM 能力评估

| 模型 | Where2Place ↑ | RoboRefIt ↑ | RoboVQA ↑ | RefCOCOg ↑ |
|------|:---:|:---:|:---:|:---:|
| QwenVL2.5-7B | 18.9% | 75.8% | 38.4 | 87.2% |
| RoboBrain-2.0-7B | 63.6% | 8.8% | 31.6 | 62.9% |
| **RoboInter-Qwen-7B** | **65.8%** | **85.6%** | **74.4** | **88.4%** |
| RoboInter-LLaVAOV-7B | 66.3% | 89.3% | 74.5 | 87.3% |

RoboInter-VLM 在所有具身基准上大幅超越基线，同时保持通用能力稳定（TextVQA 83.0、MME 2281）。

### Open-Loop 执行器评估

| 方法 | OLS@0.1 | OLS@0.05 | OLS@0.01 | mOLS |
|------|:---:|:---:|:---:|:---:|
| Vanilla | 0.6793 | 0.3608 | 0.0189 | 0.3086 |
| IC-E2E | 0.6984 | 0.3810 | 0.0204 | 0.3218 |
| EC-E2E | 0.7049 | 0.3930 | 0.0314 | 0.3340 |
| **Te-Modular** | **0.7124** | **0.4133** | **0.0584** | **0.3543** |
| Oracle+Executor | 0.7511 | 0.4640 | 0.0587 | 0.3861 |

Te-Modular（文本 F-CoT + 模块化架构）取得学习方法中最佳结果，解耦规划和执行有利于各自专注优化。

### 消融实验：中间表示类型的贡献

| 中间表示组合 | mOLS |
|------------|:---:|
| Vanilla（无中间表示） | 0.3086 |
| + Subtask | 0.3146 |
| + Subtask + Primitive Skill | 0.3159 |
| + ... + Object Box | 0.3289 |
| + ... + Gripper Box | 0.3391 |
| + ... + Affordance | 0.3435 |
| + ... + **Trace** | **0.3861** |

粗粒度表示（Subtask、Skill）贡献边际，空间精确的信号（Object Box、Affordance）贡献显著，**Trace 提供了最大收益**（密集时序信息直接约束动作生成）。

### 真实世界闭环评估

| 模型 | ID 平均成功率 | OOD 平均成功率 | ID→OOD 下降 |
|------|:---:|:---:|:---:|
| OpenVLA | 45.0% | 23.3% | 21.7% |
| π₀ | 63.3% | 45.0% | 18.3% |
| Vanilla | 65.0% | 38.3% | 26.7% |
| IC-E2E | 77.3% | 58.3% | 19.0% |
| **EC-E2E** | 68.3% | **60.0%** | **8.3%** |

EC-E2E 在 OOD 上表现最优且下降最小（仅 8.3%），显式中间表示推理提供了更强的泛化鲁棒性。

## 论文评价

### 优点

1. **系统性极强**：从标注工具到数据到基准到模型框架，对中间表示的研究提供了"全家桶"式基础设施
2. **规模可观**：23 万 episode + 571 场景 + 6100 万帧 grounding 标注，远超现有工作
3. **实验完整**：open-loop / closed-loop / 跨平台 / SimplerEnv / 消融实验全覆盖
4. **洞察深刻**：系统验证了中间表示粒度、架构设计选择（模块化 vs E2E）对性能的影响

### 不足

1. VQA 数据来自模板 + 重组标注，多样性受限于标注模板的设计
2. 真实世界实验仅在 4 个任务上评估，泛化到更复杂长序列任务的效果未知
3. 模块化 Planner 推理延迟较高（~2.4s），实际部署需要异步推理等工程优化

### 评分

⭐⭐⭐⭐⭐

RoboInter 是机器人中间表示研究的里程碑式工作。它不仅提供了迄今为止最大规模的多类型中间表示数据集，还通过 VQA 基准和 VLA 框架的系统设计，为 plan-then-execute 范式的研究搭建了完整的实验平台。消融实验清晰揭示了 Trace > Affordance > Object Box > Subtask 的中间表示价值层级，这一洞察对未来的具身 AI 研究具有重要指导意义。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](../../CVPR2026/robotics/lada_robotic_manipulation.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](../../CVPR2026/robotics/language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[ICLR 2026\] When would Vision-Proprioception Policies Fail in Robotic Manipulation?](when_would_vision-proprioception_policies_fail_in_robotic_manipulation.md)
- [\[CVPR 2025\] LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](../../CVPR2025/robotics/language-grounded_decoupled_action_representation_for_robotic_manipulation.md)

<!-- RELATED:END -->
