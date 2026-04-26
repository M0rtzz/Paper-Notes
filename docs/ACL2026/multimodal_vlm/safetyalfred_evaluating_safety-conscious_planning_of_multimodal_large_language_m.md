---
title: >-
  [论文解读] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models
description: >-
  [ACL 2026][多模态][具身安全] 本文提出 SafetyALFRED 基准，在 ALFRED 具身任务中引入六类厨房安全隐患，揭示了多模态大语言模型在静态 QA 中能识别危险（最高 92%）但在具身规划中却难以主动缓解危险（<60%）的严重对齐差距，倡导从 QA 评估范式转向具身安全评估。
tags:
  - ACL 2026
  - 多模态
  - 具身安全
  - 危险缓解
  - 多模态评估
  - 安全规划
  - ALFRED
---

# SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.19638](https://arxiv.org/abs/2604.19638)  
**代码**: https://github.com/sled-group/SafetyALFRED  
**领域**: 多模态VLM  
**关键词**: 具身安全, 危险缓解, 多模态评估, 安全规划, ALFRED

## 一句话总结
本文提出 SafetyALFRED 基准，在 ALFRED 具身任务中引入六类厨房安全隐患，揭示了多模态大语言模型在静态 QA 中能识别危险（最高 92%）但在具身规划中却难以主动缓解危险（<60%）的严重对齐差距，倡导从 QA 评估范式转向具身安全评估。

## 研究背景与动机

**领域现状**：多模态大语言模型正被越来越多地作为具身环境中的自主代理使用，将高级自然语言指令转化为可执行计划。现有安全基准如 ASIMOV、Multimodal Situational Safety、MM-SafetyBench 主要通过基于静态图像/视频的问答任务评估危险识别能力。

**现有痛点**：现有评估存在根本性缺陷——它们只测试模型是否"认识"危险，不测试模型是否能在动态具身环境中生成缓解危险的计划。一个能识别"水槽中有手机"是危险的模型，在执行"洗刀"任务时可能完全忽略先将手机从水槽中取出。这种"知识-行动"的脱节从未被系统化量化。

**核心矛盾**：静态 QA 评估中的高准确率给人一种虚假的安全感——模型"知道"什么是危险的，但在需要同时执行任务和缓解危险时，它们系统性地优先完成任务而忽视安全。QA 性能是具身安全的糟糕代理。

**本文目标**：（1）构建一个将危险识别与主动缓解结合评估的具身基准；（2）量化 QA 识别与具身缓解之间的对齐差距；（3）探索多代理框架是否能改善这一差距。

**切入角度**：扩展 ALFRED 基准（基于 AI2-THOR 的具身指令跟随任务），在 30 个厨房环境中引入六类真实世界安全隐患。利用预渲染轨迹提供地面真相历史，隔离"安全推理能力"与"任务执行能力"。

**核心 idea**：在同一场景上同时运行 QA 评估（能否识别危险）和具身评估（能否在执行任务的同时缓解危险），通过对齐率量化两者之间的差距。

## 方法详解

### 整体框架
SafetyALFRED 将安全约束规划建模为元组 $\mathcal{P} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{G}, \mathcal{H}, \mathcal{R}_{\text{safe}} \rangle$，要求安全意识策略 $\pi^*$ 在存在危险时优先执行修正动作 $\mathcal{R}_{\text{safe}}(h_i, s_t)$，只有在无危险状态下才推进任务目标。评估管线包括：（1）环境扰动引入危险；（2）QA 任务中模型作为安全评判者识别危险；（3）具身任务中模型生成包含缓解的计划。

### 关键设计

1. **六类厨房安全隐患**:

    - 功能：覆盖真实世界厨房事故的主要类型
    - 核心思路：基于厨房事故统计定义六个类别：家电误用（微波炉中放金属/易燃物）、食品变质（冰箱门未关）、跌倒/绊倒（柜门未关）、火灾隐患（炉灶打开）、财产损害（水敏物品在水槽中）、不卫生（目标物在脏地板上）。每类定义了环境条件谓词和修正动作
    - 设计动机：这六类覆盖了从高频（跌倒/绊倒是最常见伤害源）到高破坏性（火灾是最具破坏性的事故类型）的完整风险谱

2. **双设置评估（QA + 具身）**:

    - 功能：量化抽象安全知识到具体行为的转化差距
    - 核心思路：同一模型在两个独立实例中评估同一场景——QA 实例作为外部安全评判者判断是否存在危险（通过结构+NLI 两阶段验证），具身实例在执行家务任务时逐帧生成下一步动作和子目标。对齐率 $\mathcal{A} = \frac{1}{K}\sum_{k=1}^{K}\mathbb{I}(v_{ik} = a_{ik})$ 衡量 QA 识别与具身缓解的一致性
    - 设计动机：这种设计直接暴露"知道但不做"的问题，是对现有纯 QA 评估的根本性补充

3. **多代理框架**:

    - 功能：尝试通过角色分离改善安全缓解
    - 核心思路：将危险识别与缓解解耦——专门的安全评判代理负责识别危险并将安全信息传递给具身代理。这测试了"如果模型被告知存在危险，它是否能缓解"的假设
    - 设计动机：如果单代理的失败源于任务干扰（执行任务分散了对安全的注意力），那么多代理分工应该改善性能

### 损失函数 / 训练策略
本文是评估性工作，不涉及模型训练。所有模型使用温度 0 和最大 512 token 的设置。

## 实验关键数据

### 主实验
11 个 MLLM 在 QA 识别和具身缓解上的表现对比。

| 模型 | QA 识别（有元数据）| 具身缓解（有元数据）| 差距 |
|------|------|------|------|
| Qwen 2.5 VL 72B | 60.8% | 12.3% | -48.5% |
| Qwen 3 VL 32B | 57.2% | 19.7% | -37.5% |
| Gemini 1.5 ER | 77.9% | 45.7% | -32.2% |
| Gemini 2.5 | 92.5% | 60.1% | -32.4% |

### 多代理改善

| 模型 | 单代理 | 多代理 | 提升 |
|------|--------|--------|------|
| Gemma 3 27b | 7.0% | 25.1% | +18.1% |
| Qwen 3 VL 32b | 19.7% | 32.5% | +12.8% |
| Qwen 2.5 VL 72b | 12.3% | 28.5% | +16.2% |

### 关键发现
- **对齐差距惊人**：即使是最强的 Gemini 2.5，QA 中 92.5% 的识别率在具身任务中仅转化为 60.1% 的缓解率
- 模型系统性地**优先完成任务而非缓解危险**：Qwen 3 VL-32B 在无危险帧的动作预测准确率为 80.7%，但危险缓解成功率仅 19.7%
- **火灾隐患**是唯一在两个设置中都表现良好的类别（炉灶开关状态容易感知和操作），其他类别的差距巨大
- 多代理框架有帮助但不完全解决问题：即使安全评判代理正确识别了危险，具身代理仍可能不执行缓解动作
- 模型在安全场景中**频繁幻觉危险**（>50% 假阳性率），表现出过度保守偏见
- 模型规模扩大通常**降低**安全对齐率——更大的模型在 QA 中识别更多但在具身中缓解不成比例

## 亮点与洞察
- **"知道但不做"的发现**极具影响力：它根本性地挑战了当前 MLLM 安全评估的有效性。大量工作用 QA/选择题评估安全性，但本文证明这是不够的
- 实验设计的**控制变量思路**值得学习：提供地面真相历史以隔离安全推理、使用视觉-only 和元数据增强两种模式分离感知和推理缺陷
- 多代理框架的结果揭示了一个更深层的问题：不仅是注意力分配的问题，模型在需要"打断"任务流程插入安全动作时存在根本性的规划困难
- 可迁移到自动驾驶等领域：安全约束下的规划能力评估是通用需求

## 局限与展望
- 使用预渲染轨迹而非实时交互，不完全代表真实机器人场景
- 仅评估三个模型家族（Qwen、Gemma、Gemini），结论的泛化性有限
- AI2-THOR 模拟器的厨房危险是简化的，不能完全捕捉真实世界的复杂性和不可预测性
- 使用 NLI 模型自动评估 QA 响应，可能引入偏差
- 未探索通过训练数据增强来提升模型具身安全能力的方法

## 相关工作与启发
- **vs ASIMOV/MM-SafetyBench**: 这些基准仅评估静态 QA 中的危险识别，SafetyALFRED 增加了具身缓解维度并量化了两者的差距
- **vs Son et al./Chen et al.**: 前者限于文本 PDDL 环境，后者限于静态 AI 生成图像。SafetyALFRED 在有导航的多模态模拟环境中评估
- **启发**：未来安全评估应要求模型"做"安全而非只是"说"安全；训练数据需要包含安全-任务平衡的示例

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统量化 QA 安全识别与具身安全缓解的对齐差距，问题定义新颖
- 实验充分度: ⭐⭐⭐⭐ 11个模型、6类危险、多种评估指标，但使用预渲染轨迹是简化
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，但论文较长且部分分析分散在附录中

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs](faithful-first_reasoning_planning_and_acting_for_multimodal_llms.md)
- [\[CVPR 2025\] Evaluating Vision-Language Models as Evaluators in Path Planning](../../CVPR2025/multimodal_vlm/evaluating_vision-language_models_as_evaluators_in_path_planning.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)

<!-- RELATED:END -->
