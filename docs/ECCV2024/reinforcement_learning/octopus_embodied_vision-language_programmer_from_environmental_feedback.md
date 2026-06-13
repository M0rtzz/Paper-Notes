---
title: >-
  [论文解读] Octopus: Embodied Vision-Language Programmer from Environmental Feedback
description: >-
  [ECCV2024][强化学习][具身智能] 提出 Octopus，一个具身视觉-语言编程模型，通过生成可执行代码来连接高层规划与底层操控，并引入 Reinforcement Learning with Environmental Feedback (RLEF) 训练方案来提升决策质量。 大型视觉-语言模型 (VLM) 在多…
tags:
  - "ECCV2024"
  - "强化学习"
  - "具身智能"
  - "视觉语言"
  - "代码生成"
  - "environmental feedback"
---

# Octopus: Embodied Vision-Language Programmer from Environmental Feedback

**会议**: ECCV2024  
**arXiv**: [2310.08588](https://arxiv.org/abs/2310.08588)  
**代码**: [GitHub](https://github.com/dongyh20/Octopus)  
**领域**: 机器人  
**关键词**: 具身智能, vision-language model, 代码生成, 强化学习, environmental feedback

## 一句话总结

提出 Octopus，一个具身视觉-语言编程模型，通过生成可执行代码来连接高层规划与底层操控，并引入 Reinforcement Learning with Environmental Feedback (RLEF) 训练方案来提升决策质量。

## 背景与动机

大型视觉-语言模型 (VLM) 在多模态感知与推理上取得了显著进展，但将其应用于具身智能体时存在明显鸿沟：现有方法要么只输出底层操控动作序列，要么只提供抽象层面的高层规划，**高层规划与真实世界操控之间缺乏有效衔接**。

以往的编程范式（如 Voyager、VisProg）利用 LLM 生成程序调用 API，但它们**不涉及视觉感知**，直接从环境解析文本数据。少数尝试融合视觉的工作（如 TAPA、SayPlan）则过度依赖预训练视觉模型将图像转为语言描述，转换过程中可能丢失关键信息，且无法生成可执行代码。EmbodiedGPT 虽然结合了视觉-语言建模与策略映射，但具身 VLM 直接生成可执行程序的能力仍未被充分探索。

## 核心问题

1. 如何让具身视觉-语言模型**直接从视觉输入生成可执行代码**，而非仅输出抽象计划？
2. 如何构建**视觉依赖的**仿真环境和函数调用体系，确保智能体无法绕过视觉信息完成任务？
3. 如何利用**环境反馈**来进一步优化模型的规划与代码生成能力？

## 方法详解

### OctoVerse 环境套件

作者精心设计了三个仿真环境，统称为 OctoVerse，核心设计原则是所有函数调用必须**依赖视觉信息 (vision-dependent)**：

- **OctoGibson**：基于 OmniGibson 构建，支持 50 个场景、5000+ 标注物体、476 个任务（367 个常规任务 + 109 个推理任务）。集成 16 个可执行函数（如 `moveBot()`、`easyGrasp()`）。关键设计：限制 `moveBot(object)` 的参数只能是大型物体（桌子、柜子），强迫智能体通过视觉判断小物体位置。
- **OctoMC**：基于 Minecraft 构建，包含 6 个功能动作、40 个任务、10 种不同生物群系。取消了 Voyager 中不依赖视觉的 `exploreUntil()` 函数，替换为 `teleport(yaw, distance)`，要求智能体主动利用视觉线索导航。
- **OctoGTA**：基于 GTA-V 构建，包含 19 个函数、25 个任务、5 个任务组。用 `goForward(distance)` 和 `turnPlayer(degree)` 替代 `walkTo(location)`，确保操作依赖视觉感知。

### Octopus 模型架构

模型基于 Otter 架构，核心组件包括：

- **视觉编码器**：CLIP ViT-L/14
- **语言解码器**：MPT-7B
- **跨模态模块**：Perceiver Resampler + Cross-Gated Attention（来自 Flamingo 设计）
- **视觉输入**：8 张第一人称视角图像（每 45° 一张，覆盖 360°）+ 2 张鸟瞰图（OctoMC 仅用 4 张第一人称图像）

### 训练数据收集

1. **环境信息收集**：每个状态格式化为包含 Observed Objects、Observed Relations、Inventory、Task Goal 的环境消息
2. **GPT-4 自动采集**：精心设计 system prompt，使用 GPT-4 32K 模型控制探索型智能体，在仿真环境中执行任务并生成动作蓝图与可执行代码
3. **错误管理**：GPT-4 执行中的错误（语法错误、物理约束失败等）也被记录；主任务超过 10 步未完成则终止；所有无语法错误的数据对均保留用于训练
4. **环境反馈收集**：自动标注 step-level（单步执行后状态是否匹配目标）和 task-level（整体任务是否完成）的二值反馈

### 训练流程

**阶段一：SFT（监督微调）**

在收集的数据集 $\mathcal{D}_E = \{(\mathbf{X}_v, \mathbf{T}_i, \mathbf{T}_r)\}$ 上进行 token 级别的监督微调，训练目标为下一个 token 预测：

$$p(\mathbf{T}_r | \mathbf{T}_i, \mathbf{X}_v) = \prod_{l=1}^{L} p(t_l | \mathbf{X}_v, \mathbf{T}_i, \mathbf{T}_{r,<l})$$

**阶段二：RLEF（环境反馈强化学习）**

- **树状任务表示**：将任务执行过程建模为树结构，每个节点代表一个子任务，附带成功/失败的二值标注
- **奖励模型**：使用 CodeLLaMA-7B 加额外 value head 作为奖励模型 $r_\phi$，在偏好数据 $\mathcal{D}_R$ 上微调，评估状态转换的质量
- **策略优化**：以 SFT 模型为初始策略 $\pi^{\text{INIT}}$，复制一份 $\pi_\theta^{\text{RL}}$ 使用 PPO 算法优化，损失函数包含奖励项和 KL 散度惩罚项

## 实验关键数据

### OctoGibson 主要结果（任务完成率 / 规划分数）

| 模型 | 整体 | 可见环境 | 未见环境 | 常规任务 | 推理任务 |
|------|------|---------|---------|---------|---------|
| GPT-4 (blind) | 0.43/0.68 | 0.42/0.69 | 0.46/0.67 | 0.49/0.78 | 0.27/0.40 |
| GPT-4V | 0.45/0.63 | 0.40/0.62 | 0.60/0.67 | 0.42/0.67 | 0.53/0.53 |
| CodeLLaMA | 0.12/0.25 | 0.09/0.20 | 0.20/0.40 | 0.16/0.31 | 0.00/0.07 |
| TAPA (step) | 0.15/0.38 | 0.16/0.42 | 0.13/0.27 | 0.18/0.38 | 0.07/0.40 |
| EmbodiedGPT | 0.10/0.40 | 0.04/0.36 | 0.27/0.53 | 0.13/0.38 | 0.00/0.40 |
| **Octopus (SFT)** | 0.15/0.37 | 0.11/0.33 | 0.27/0.47 | 0.16/0.38 | 0.13/0.33 |
| **Octopus (SFT+RLEF)** | **0.18/0.42** | 0.13/0.38 | **0.33/0.53** | 0.18/0.40 | **0.20/0.53** |

### OctoMC & OctoGTA 结果

- OctoMC：Octopus (SFT+RLEF) 整体任务完成率 0.30/规划分数 0.65，优于 EmbodiedGPT (0.25/0.58)
- OctoGTA：Octopus (SFT+RLEF) 整体 0.20/0.56，优于 EmbodiedGPT (0.16/0.42)
- GPT-4V 在所有环境中仍是最强基线（OctoMC: 0.73/0.85, OctoGTA: 0.56/0.82）

### 消融实验

- **模型组件**：仅调 connector 只完成 4/60 任务；调 connector + language decoder 完成 5/60 任务；全参数微调效果最佳
- **模型规模**：3B 模型性能明显低于 7B，SFT 和 RLEF 模型均如此
- **视觉输入**：随机打乱视觉输入顺序导致性能显著下降，证明模型确实依赖结构化视觉信号
- **代码执行率**：CodeLLaMA 代码执行成功率 92%，LLaMA 仅 24%，Octopus MPT-7B 为 72%

## 亮点

1. **编程范式的创新定位**：用可执行代码作为高层规划与底层操控之间的桥梁，既保留了规划的灵活性，又具备直接执行的能力
2. **视觉依赖的环境设计**：三个环境中精心约束函数调用，确保任务完成必须依赖视觉信息，避免了简单的文本 shortcut
3. **RLEF 训练方案**：利用模拟器的环境反馈构建偏好数据，通过 PPO 优化策略模型，在推理任务和未见环境上提升尤为显著（推理任务从 0.13 提升到 0.20）
4. **完整的数据收集流水线**：GPT-4 驱动的自动数据收集 + 错误管理机制，可扩展性强

## 局限与展望

1. **绝对性能仍有差距**：最佳 Octopus 模型在 OctoGibson 上任务完成率仅 0.18，远低于 GPT-4V 的 0.45（且后者也远未饱和）
2. **空间推理能力不足**：在 OctoMC 和 OctoGTA 中，模型在精确空间推理（角度、距离）上常常失败，即使规划正确仍因动作不精确而导致任务失败
3. **复杂决策能力有限**：OctoGTA 中面对非直觉障碍（如墙壁阻隔需要绕路/翻墙），即使训练数据中有类似案例仍难以泛化
4. **GTA 数据依赖人工标注**：OctoGTA 因无法获取文本环境信息，训练数据完全手工创建，仅 160 条，限制了模型表现
5. **后处理依赖**：代码中生成的泛化对象名需要通过字符串相似度匹配替换为模拟器中的精确名称，这一步在真实场景中可能不可用
6. **CodeLLaMA 集成失败**：将 MPT 替换为 CodeLLaMA 会生成无意义输出，说明视觉-代码对齐数据可能不足

## 与相关工作的对比

| 维度 | Octopus | Voyager | EmbodiedGPT | TAPA | SayCan/RT-2 |
|------|---------|---------|-------------|------|-------------|
| 视觉输入 | 原生多模态 | 无（文本解析） | 有 | 外接 OVD | 有 |
| 代码生成 | ✓ | ✓ (GPT驱动) | ✗ | ✗ | ✗ |
| 环境反馈 | RLEF | 有（GPT自评） | 有 | ✗ | 有 |
| 模型训练 | ✓ (SFT+RLEF) | ✗ (in-context) | ✓ | ✓ | ✓ |
| 输出形式 | 规划+代码 | 代码 | 规划+策略映射 | 规划 | 动作 |

Octopus 的核心区分点在于：(1) 统一的视觉-语言模型同时输出规划和代码，无需外接视觉模块；(2) 引入了基于真实环境执行反馈的强化学习，而非 LLM 自评。

## 启发与关联

- **代码作为中间表示**的思路值得关注：相比直接输出动作或语言规划，代码天然具有组合性、可调试性和可执行性，可能成为具身智能体的通用输出格式
- **RLEF 范式**可推广到其他可获得环境反馈的场景（如代码执行、网页交互、工具调用），核心是将环境执行结果转化为偏好信号
- 视觉依赖环境设计的思路可用于评估其他具身模型是否真正利用了视觉信息
- 7B 规模模型在复杂具身任务上的能力上限仍然有限，更大规模模型或更好的视觉-代码对齐方案可能是突破方向

## 评分
- 新颖性: ⭐⭐⭐⭐ — 编程范式 + RLEF + 视觉依赖环境设计的组合有新意
- 实验充分度: ⭐⭐⭐ — 三个环境覆盖面广但绝对性能较低；消融实验充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，动机阐述明确
- 价值: ⭐⭐⭐⭐ — 开源环境+数据+模型，对具身视觉编程研究具有基础性贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](../../AAAI2026/reinforcement_learning/vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ICCV 2025\] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](../../ICCV2025/reinforcement_learning/navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)
- [\[ACL 2025\] Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback](../../ACL2025/reinforcement_learning/align-slm_textless_spoken_language_models_with_reinforcement_learning_from_ai_fe.md)
- [\[AAAI 2026\] STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision](../../AAAI2026/reinforcement_learning/stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)
- [\[ICLR 2026\] DVLA-RL: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning](../../ICLR2026/reinforcement_learning/dvla-rl_dual-level_vision-language_alignment_with_reinforcement_learning_gating_.md)

</div>

<!-- RELATED:END -->
