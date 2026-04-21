---
title: >-
  [论文解读] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles
description: >-
  [CVPR 2026][人体理解][GUI Agent] 提出 State-aware Reasoning (StaR)，通过教会多模态 Agent "感知当前状态→分析目标状态→决定是否操作"的三步推理链，将 GUI 开关控制准确率提升超 30%，同时不损害通用 Agent 任务性能。
tags:
  - CVPR 2026
  - 人体理解
  - GUI Agent
  - Toggle Control
  - 多模态推理
  - 状态感知
  - State-aware Reasoning
---

# See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles

**会议**: CVPR 2026  
**arXiv**: [2509.13615](https://arxiv.org/abs/2509.13615)  
**代码**: [有](https://github.com/ZrW00/StaR)  
**领域**: 人体理解  
**关键词**: GUI Agent, Toggle Control, 多模态推理, 状态感知, State-aware Reasoning  

## 一句话总结

提出 State-aware Reasoning (StaR)，通过教会多模态 Agent "感知当前状态→分析目标状态→决定是否操作"的三步推理链，将 GUI 开关控制准确率提升超 30%，同时不损害通用 Agent 任务性能。

## 研究背景与动机

### 1. 领域现状

多模态 Agent（如 AppAgent、UI-TARS、Mobile-Agent 等）利用多模态大语言模型（MLLM）直接感知 GUI 截图并执行类人操作，已在 GUI 交互领域取得显著进展。这些 Agent 无需依赖 API，可作为灵活可靠的人机交互助手。

### 2. 痛点

**Toggle 控件（开关/切换按钮/复选框）是 GUI 中无处不在的基本交互元素**，广泛存在于手机设置、车载系统、智能家居和工业控制等场景。然而，现有 Agent 在执行 toggle 指令时极不可靠——作者构建基准测试后发现，包括 GPT-5 在内的大多数 Agent 执行准确率低于 50%。

### 3. 核心矛盾

Agent 存在强烈的"点击偏向"（toggling bias）：
- **假阳性 (False Positive)**：当前状态已满足目标，Agent 仍去点击切换
- **假阴性 (False Negative)**：当前状态不满足目标，Agent 却未执行切换

本质原因是 Agent 缺乏对 toggle 当前状态的感知与推理能力，总是倾向于预测 CLICK，而不会先判断"是否需要操作"。

### 4. 要解决什么

提升多模态 Agent 的内在推理能力，使其能准确感知、推理和执行 toggle 控制指令。

### 5. 切入角度

分析现有两种直觉方案的不足：(a) Prompt engineering 无法从根本上增强推理；(b) 引入额外标注器（multi-agent 协作）存在悖论——若标注器自身不可靠则无意义，若可靠则不如直接用标注器。因此需要一种能提升 Agent 自身推理能力的方法。

### 6. 核心 Idea

模拟人类执行 toggle 指令的认知过程：先看当前状态 → 再理解指令要求的目标状态 → 最后比较决定是否操作。将这种"See-Think-Act"的三步推理通过训练内化到 Agent 中。

## 方法详解

### 整体框架

StaR (State-aware Reasoning) 是一种多模态推理方法，通过将状态感知显式集成到推理链中来改进 toggle 控件的执行。整体流程分为两个阶段：

1. **基准构建阶段**：从公开数据集出发，通过三步标注流水线构建包含 81,836 个样本的 state control benchmark
2. **训练阶段**：在 benchmark 训练集上训练 Agent 学习 StaR 推理过程，同时在 agentic benchmark 中对涉及 toggle 的 episode 替换推理链为 StaR 风格，保持其他 episode 不变

### 关键设计

#### 设计一：三步标注流水线 (State Control Benchmark 构建)

**功能**：从 6 个公开数据集（AMEX、RICOSCA、GUIAct-Mobile、AndroidWorld、AITW、OS-Atlas grounding）构建高质量的 toggle 控制基准。

**核心思路**：

- **Step 1 — Widget Parsing**：提取截图中的 widget bounding box，进一步使用 OminiParser 解析额外的可点击元素，合并得到统一的 bounding box 集合
- **Step 2 — Toggle Identification**：使用 Qwen-2-VL-72B 和 GLM-4V 作为独立标注器执行交叉验证（inter-annotator agreement），只有两者均认定为 toggle 的才保留
- **Step 3 — State-functionality Annotation**：同样由双标注器独立标注 toggle 的状态（on/off）和功能描述，再通过交叉一致性过滤

**设计动机**：公开数据集缺乏可靠的 XML 树来提取状态信息，因此需要自主标注。双标注器交叉验证消除单一模型的偏差，人工验证 200 样本显示状态标注准确率 91%、功能标注 92.5%。

**数据扩展**：每个样本 $\langle s, b, \sigma, f \rangle$ 扩展为正负两条指令。若 $\sigma=1$（开启），则生成"关闭 $f$"→CLICK 和"开启 $f$"→COMPLETED，最终得到 81,836 个平衡样本（73,652 训练 + 8,184 测试）。

#### 设计二：StaR 三步推理链

**功能**：在推理链中显式嵌入状态感知，替代原始的 Thought→Action 模式。

**核心思路**：

- **Perceiving（感知）**：引导 Agent 从截图中识别当前 toggle 状态 $\sigma$，将视觉特征与细粒度的 toggle 状态关联
- **Analyzing（分析）**：引导 Agent 从用户指令推断目标状态 $\sigma_u$。正向指令 $\sigma_u \neq \sigma$，负向指令 $\sigma_u = \sigma$
- **Deciding（决策）**：比较 $\sigma$ 与 $\sigma_u$，若 $\sigma \neq \sigma_u$ 则执行 CLICK，否则标记为 COMPLETED

**设计动机**：直接 prompting Agent "注意 toggle 状态"效果有限（实验已验证），必须通过训练将三步推理内化为 Agent 的固有能力。

#### 设计三：保持通用性的混合训练策略

**功能**：在 toggle benchmark 和 agentic benchmark 上联合训练，避免灾难性遗忘。

**核心思路**：对 agentic benchmark（AndroidControl、AITZ、GUI-Odyssey）中涉及 toggle 的 episode，将推理链替换为 StaR 风格；其他 episode 保留原始推理。Agent 学会在 toggle 场景自适应使用 StaR 推理，在其他场景保持原有推理模式。

**设计动机**：这些 benchmark 本身就是目标 Agent 原始训练集的一部分，通过替换推理链而非增加额外数据，实现精准注入 toggle 推理能力。

### 损失函数 / 训练策略

- 使用 LLaMA-Factory 框架进行微调
- 学习率 $5 \times 10^{-6}$，训练 3 个 epoch
- 使用 FlashAttention 加速
- Click 坐标归一化到 $[0, 1000]$
- 在 4 个不同架构的 Agent（OS-Atlas-7B、UI-TARS-7B、AgentCPM-GUI-8B、GUI-Owl-7B）上分别训练验证

## 实验关键数据

### 主实验一：State Control Benchmark 上的表现

| 模型 | 设置 | O-TMR↑ | O-AMR↑ | P-AMR↑ | N-AMR↑ | N-FPTR↓ | N-FPR↓ |
|------|------|--------|--------|--------|--------|---------|--------|
| OS-Atlas-7B | Zero-shot | 67.16 | 43.95 | 52.10 | 35.80 | 64.10 | 28.67 |
| OS-Atlas-7B | StaR Prompting | 73.52 | 50.07 | 49.88 | 50.27 | 49.62 | 22.21 |
| OS-Atlas-7B | **StaR Training** | **96.13** | **79.72** | **62.95** | **96.48** | **3.52** | **1.52** |
| UI-TARS-7B | Zero-shot | 67.14 | 47.45 | 54.94 | 39.96 | 48.29 | 17.62 |
| UI-TARS-7B | **StaR Training** | **95.82** | **77.86** | **59.19** | **96.53** | **3.45** | **1.34** |
| AgentCPM-GUI-8B | Zero-shot | 81.74 | 64.08 | 60.04 | 68.11 | 30.69 | 11.07 |
| AgentCPM-GUI-8B | **StaR Training** | **95.98** | **79.00** | **60.53** | **97.46** | **2.54** | **0.95** |
| GUI-Owl-7B | Zero-shot | 76.58 | 53.57 | 48.97 | 58.16 | 39.15 | 14.66 |
| GUI-Owl-7B | **StaR Training** | **95.99** | **77.60** | **58.87** | **96.33** | **3.67** | **1.56** |

**关键结论**：StaR 训练在 O-AMR 上分别提升 +35.77%（OS-Atlas）、+30.41%（UI-TARS）、+14.92%（AgentCPM）、+24.03%（GUI-Owl）。训练后的 7B 模型超越了 zero-shot 的 Qwen-2-VL-72B（O-AMR 66.42%），弥合了模型规模差距。

### 主实验二：动态环境评估

| 模型 | 无 StaR | 有 StaR |
|------|---------|---------|
| UI-TARS-7B | 35 (7/20) | 40 (8/20) |
| OS-Atlas-7B | 10 (2/20) | **55 (11/20)** |
| AgentCPM-GUI-8B | 20 (4/20) | 42.5 (8.5/20) |

**关键结论**：在 AndroidWorld 框架下的真实动态环境中，StaR 一致性提升任务成功率。OS-Atlas-7B 从 10% 跃升至 55%，印证了 StaR 对弱推理 Agent 改造效果最显著。

### 消融实验

- **StaR-style Prompting vs. StaR Training**：仅 prompting 效果有限（如 OS-Atlas O-AMR 仅提升 6.12%），而训练提升 35.77%，证明结构化推理必须通过训练学习
- **Prompt Engineering baseline**（Section 3.2）：简单提示关注 toggle 状态对 UI-TARS 和 GUI-Owl 略有改善，但对 AgentCPM 几乎无效
- **跨架构泛化**：四种不同架构和历史建模策略的 Agent 均受益，验证了 StaR 的模型无关性

### 关键发现

1. **所有现有 Agent 存在强烈的点击偏向**：低 P-FNR + 高 N-FPTR + 非零 N-FPR，表明 Agent 倾向于无条件预测 CLICK
2. **通用专有模型 grounding 能力差**：GPT-5/GPT-4o/Gemini 2.5 Pro 的 P-TMR 接近 100% 但 P-AMR 仅约 20%
3. **StaR 最大收益来自弱推理模型**：OS-Atlas-7B 起点最低但提升最大（O-AMR +35.77%，动态环境 10%→55%），说明 StaR 能有效重塑推理能力
4. **通用 Agent 任务不受损**：在 AndroidControl、AITZ、GUI-Odyssey 上一致保持或超越 baseline，复杂长链任务（GUI-Odyssey）TSR 提升近 10-20%
5. **StaR 推理链可进一步辅助决策**：在 AndroidControl-L 中，StaR 风格的推理链比原始低级指令更能促进准确决策

## 亮点与洞察

- **问题定义精准**：首次系统性地揭示了 toggle 控制这个被忽视但极其常见的 GUI Agent 瓶颈，构建了包含 81,836 样本的大规模基准
- **方法优雅简洁**：三步推理链（See-Think-Act）直觉清晰，模拟人类认知过程，无需额外标注器或多 Agent 协作
- **双标注器交叉验证**的数据构建流水线具有通用参考价值
- **实验全面**：静态 benchmark + 通用 agentic benchmark + 真实动态环境，三个层面逐步验证

## 局限与展望

- 仅聚焦于移动端 toggle（二值状态），未涉及 slider、dropdown 等连续/多值控件
- 动态评估仅 20 个任务，规模偏小
- StaR 需要对每个目标 Agent 单独微调，缺乏即插即用的 zero-shot 方案
- 未探索 StaR 推理与 RL-based reasoning（如 GUI-R1 的 GRPO）的结合
- Toggle 状态识别依赖视觉感知，对极细粒度或非标准 toggle 样式的鲁棒性未充分验证

## 相关工作与启发

- **多模态推理**：CoAT (AITZ) 的语义标注+推理链 → StaR 进一步将状态感知嵌入推理链
- **GUI Agent**：UI-TARS、AgentCPM-GUI 等已有较强基础能力，但在细粒度控件交互上仍有盲区
- **启发**：StaR 的"感知-分析-决策"模式可推广到其他需要状态比较的 GUI 交互场景（如判断列表项是否已选中、文本框是否已填写等）

## 评分

⭐⭐⭐⭐ 问题定义清晰、方法简洁有效、实验充分且涵盖真实场景，但核心贡献更偏向工程优化（通过训练注入推理模式），技术新颖性中规中矩。

<!-- RELATED:START -->

## 相关论文

- [Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_multimodal_task_generation_agents.md)
- [AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video](avatar_reinforcement_learning_to_see_hear_and_reason_over_video.md)
- [I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue](../../ACL2025/human_understanding/i_see_what_you_mean_co-speech_gestures_for_reference_resolution_in_multimodal_di.md)
- [GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)
- [GardenDesigner: Encoding Aesthetic Principles into Jiangnan Garden Construction via a Chain of Agents](gardendesigner_encoding_aesthetic_principles_into_jiangnan_garden_construction_v.md)

<!-- RELATED:END -->
