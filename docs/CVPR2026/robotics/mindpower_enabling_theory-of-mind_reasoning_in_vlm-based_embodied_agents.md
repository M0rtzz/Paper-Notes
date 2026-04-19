---
title: >-
  [论文解读] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents
description: >-
  [CVPR 2026][机器人][Theory of Mind] MindPower 提出以机器人为中心（Robot-Centric）的心智理论推理框架，将感知→信念→欲望→意图→决策→行动组织为三级六层推理层级（MindPower Reasoning Hierarchy），并用 Mind-Reward（基于 GRPO 强化学习）优化推理一致性，在决策和动作生成上分别超过 GPT-4o 12.77% 和 12.49%。
tags:
  - CVPR 2026
  - 机器人
  - Theory of Mind
  - BDI推理
  - 具身Agent
  - Mind-Reward
  - GRPO
  - Robot-Centric
---

# MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents

**会议**: CVPR 2026  
**arXiv**: [2511.23055](https://arxiv.org/abs/2511.23055)  
**代码**: [https://zhangdaxia22.github.io/MindPower/](https://zhangdaxia22.github.io/MindPower/) (Benchmark)  
**领域**: robotics  
**关键词**: Theory of Mind, BDI推理, 具身Agent, Mind-Reward, GRPO, Robot-Centric  

## 一句话总结

MindPower 提出以机器人为中心（Robot-Centric）的心智理论推理框架，将感知→信念→欲望→意图→决策→行动组织为三级六层推理层级（MindPower Reasoning Hierarchy），并用 Mind-Reward（基于 GRPO 强化学习）优化推理一致性，在决策和动作生成上分别超过 GPT-4o 12.77% 和 12.49%。

## 背景与动机

1. **具身 Agent 缺乏心智推理能力**：现有 VLM-based 具身 agent 只能执行显式指令，无法推断人类的信念、欲望和意图，更无法据此主动提供帮助
2. **现有 ToM Benchmark 只关注角色视角（Role-Centric）**：MuMA-ToM、MMToM-QA 等 benchmark 只推断视频中人物的心理状态，不涉及 agent 自身视角的推理
3. **缺少从推理到行动的闭环**：现有 benchmark 多为选择题形式，不要求 agent 基于心智推理生成可执行的决策和动作序列
4. **VLM 在高层推理上表现差**：实验表明 GPT-4o、Gemini 等闭源 VLM 虽然感知层尚可，但在信念推理和行动生成上远低于人类水平
5. **开源 VLM 更为薄弱**：InternVL3.5、LLaVA-OV 等在动作生成上 SR/AC 几乎为零，输出多为不可执行的模糊表述
6. **标准 CoT 推理效果有限**：通用的 `<think>` 推理链在心智推理任务上不如结构化的 BDI 推理层级

## 方法详解

### 整体框架

MindPower 包含三部分：(1) **MindPower Benchmark**——590 个家庭场景的数据集，含两个任务；(2) **MindPower Reasoning Hierarchy**——三级六层结构化推理；(3) **Mind-Reward**——基于 GRPO 的强化学习优化。基础模型为 Qwen2.5-VL-7B-Instruct。

### MindPower Reasoning Hierarchy（三级六层）

- **Level-1 感知** `<Perception>`：观察环境，理解当前发生了什么
- **Level-2 心智推理**：
    - `<Belief>`：推断自己和人类的信念（含二阶信念——"我认为 Alice 认为苹果在桌上"）
    - `<Desire>`：确定辅助目标（"Alice 想喝牛奶"）
    - `<Intention>`：形成具体行动意图（"我应该帮她从冰箱拿牛奶"）
- **Level-3 决策与行动**：
    - `<Decision>`：选择行动计划
    - `<Action>`：输出原子操作序列，如 `walk(fridge), open(fridge), pick(milk)`

### Robot-Centric 视角（核心创新）

区别于现有 Role-Centric 设计：agent 不仅推断他人心理状态，还显式建模自己的信念，形成完整的二阶推理闭环。例如："我知道苹果实际在冰箱里"+"我推断 Alice 认为苹果在桌上"→"她的信念是错的，我应该帮她纠正"。

### 两个核心任务

1. **错误信念纠正（False-Belief Correction）**：检测人类对环境的错误信念（如物体被移动但人不知道），agent 需识别矛盾并主动纠正
2. **隐式目标推断与完成（Implicit Goal Inference & Completion）**：从人类的搜索行为、重复失败等微妙线索推断隐含目标并提供帮助。覆盖特殊人群（轮椅用户、儿童）、物体属性推理、功能组合、对话推断四类场景

### Mind-Reward 训练策略

**两阶段训练**：(1) SFT 冷启动（5 epochs），建立基本推理对齐；(2) GRPO 强化优化（400 iterations，每次采样 8 个输出）。

**Mind-Reward 设计**：将每层推理输出由 Qwen3-Max 转换为原子动作序列，计算三种对齐指标：
- **原子准确度**（ROUGE-1）：正确匹配的原子动作比例，标注视角属性确保 Robot-Centric 对齐
- **局部一致性**（ROUGE-2）：相邻原子对的连贯性
- **全局一致性**（ROUGE-L）：整体推理序列的对齐度

$$R_{\text{Mind}} = \alpha_1 R_{\text{atomic}} + \alpha_2 R_{\text{local}} + \alpha_3 R_{\text{global}}$$

其中 $\alpha_1=0.2, \alpha_2=0.3, \alpha_3=0.5$，辅以格式奖励 $R_{\text{Format}}$（检查六层标签是否按序出现），总奖励 $R = R_{\text{Mind}} + R_{\text{Format}}$，用 GRPO 优化。

## 实验关键数据

### 主实验：与基线 VLM 对比

| 方法 | Decision (S↑) | Action SR↑ | Action AC↑ | BPC↑ |
|------|:---:|:---:|:---:|:---:|
| GPT-4o（图像输入） | 34.35 | 1.82 | 2.91 | 8.05 |
| Gemini-2.5 Pro（视频输入） | 33.87 | 2.08 | 2.54 | 8.56 |
| Video-R1-7B | 30.33 | 1.43 | 1.72 | 6.45 |
| Qwen2.5-VL-7B（base） | 26.56 | 0.29 | 0.22 | 6.07 |
| **Ours (SFT+Mind-Reward)** | **47.12** | **11.75** | **15.40** | **8.87** |
| Human Baseline | 56.66 | 19.37 | 26.26 | 8.19 |

相对 GPT-4o：决策 Sentence Transformer +12.77pp，动作准确率 AC +12.49pp。

### 消融实验

| 配置 | Decision (S↑) | Action AC↑ | BPC↑ |
|------|:---:|:---:|:---:|
| Qwen2.5-VL-7B（基线） | 26.56 | 0.22 | 6.07 |
| Mind-Reward only（无 SFT） | 24.68 | 0.40 | 6.63 |
| SFT only（无 RL） | 43.84 | 10.48 | 8.78 |
| **SFT + Mind-Reward** | **47.12** | **15.40** | **8.87** |

- 仅 SFT 已带来巨大提升（AC: 0.22→10.48），说明推理层级结构本身非常有效
- 仅 Mind-Reward 无 SFT 效果有限（AC: 0.40），证明需要 SFT 冷启动
- SFT+RL 组合最优，RL 在 SFT 基础上进一步提升约 5 个点
- MindPower Hierarchy vs 标准 CoT（GPT-4o 上测试）：结构化 BDI 推理比通用 `<think>` 推理在决策上好 4.89%

## 亮点

1. **认知科学 × 具身 AI 的系统化结合**：将 BDI 框架（信念-欲望-意图）系统化地引入 VLM agent，形成从感知到行动的可解释推理链
2. **Robot-Centric 视角创新**：首次要求 agent 同时建模自己和他人的信念，实现二阶心智推理，与现有 Role-Centric benchmark 形成本质区别
3. **Mind-Reward 奖励设计精巧**：将推理质量分解为原子-局部-全局三个粒度，比黑盒 LLM 评分更可控、可复现
4. **任务设计有洞察力**：错误信念纠正和隐式目标推断都是真实人机协作中的核心场景，且覆盖轮椅用户/儿童等特殊人群

## 局限与展望

1. **数据规模有限**：590 个样本偏少，且仅覆盖两个模拟器（VirtualHome + ThreeDWorld），泛化到真实物理环境存疑
2. **动作空间受限**：原子动作为高层操作（如 `pick(apple)`），未涉及底层运动控制（关节角度、力控），与真实机器人部署仍有鸿沟
3. **评估依赖 LLM**：BPC 评分由 GPT-4o 打分，存在评估偏差和不可复现性
4. **计算成本**：GRPO 训练需要 H800 GPU，对资源受限的研究者不友好
5. **未验证多轮交互**：所有场景为单轮推理，未测试 agent 在持续交互中维持信念一致性的能力

## 与相关工作的对比

### vs MuMA-ToM / MMToM-QA（现有 ToM Benchmark）

MuMA-ToM 和 MMToM-QA 只推断视频中角色的心理状态（Role-Centric），输出为选择题答案，不涉及决策和行动生成。MindPower 是 Robot-Centric 的，要求 agent 从自身视角推理并输出可执行的原子动作序列，同时评估从感知到行动的完整推理链。

### vs Smart-Help / RoboBench（具身协作 Agent）

Smart-Help 依赖预定义目标优化人机舒适度，RoboBench 将高层目标分解为子任务顺序执行。两者都不进行心智推理——不涉及一阶/二阶信念推理。MindPower 的关键区别在于 agent 需要推断"人类相信什么"以及"我自己知道什么"，然后基于心理状态差异做决策，而非仅执行给定目标。

### vs Visual-RFT / LLaVA-CoT（推理增强 VLM）

Visual-RFT 和 LLaVA-CoT 提供通用的视觉推理增强方法。MindPower 的 MindPower Reasoning Hierarchy 是专门为 ToM 设计的结构化推理（Perception→Belief→Desire→Intention→Decision→Action），实验证明比通用 CoT 在决策任务上高 4.89%。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — Robot-Centric ToM + 三级六层推理层级 + Mind-Reward 均为首创
- 实验充分度: ⭐⭐⭐⭐ — 对比 10+ 基线 VLM，有消融和定性分析，但数据规模偏小
- 写作质量: ⭐⭐⭐⭐ — 框架清晰，图示丰富，层次分明
- 价值: ⭐⭐⭐⭐ — 为具身 AI 的社会智能提供了重要基准和方法，但与真实机器人部署仍有距离

<!-- RELATED:START -->

## 相关论文

- [GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](../../NeurIPS2025/robotics/mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)
- [Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](../../AAAI2026/robotics/adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection](probabilistic_concept_graph_reasoning_for_multimodal_misinformation_detection.md)

<!-- RELATED:END -->
