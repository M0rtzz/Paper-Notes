---
title: >-
  [论文解读] COOPERA: Continual Open-Ended Human-Robot Assistance
description: >-
  [NeurIPS 2025][机器人][人机协作] 提出 COOPERA 框架，首次实现持续、开放式的人机协作研究，通过LLM驱动具有心理特征和长期意图的模拟人类与机器人在3D环境中多天交互，机器人通过学习人类特征和上下文意图逐步提升个性化协作能力。
tags:
  - NeurIPS 2025
  - 机器人
  - 人机协作
  - 持续学习
  - 开放式任务
  - LLM驱动人类模拟
  - 个性化机器人助手
---

# COOPERA: Continual Open-Ended Human-Robot Assistance

**会议**: NeurIPS 2025  
**arXiv**: [2510.23495](https://arxiv.org/abs/2510.23495)  
**代码**: 无  
**领域**: 机器人  
**关键词**: 人机协作, 持续学习, 开放式任务, LLM驱动人类模拟, 个性化机器人助手

## 一句话总结
提出 COOPERA 框架，首次实现持续、开放式的人机协作研究，通过LLM驱动具有心理特征和长期意图的模拟人类与机器人在3D环境中多天交互，机器人通过学习人类特征和上下文意图逐步提升个性化协作能力。

## 研究背景与动机

**领域现状**：机器人辅助任务的研究主要集中在短暂的情节式设置中，机器人在预定义的短期任务上进行评估。

**现有痛点**：现实世界中人类具有偏好和长期目标，需要在不同时间获得不同类型的帮助。现有方法使用预定义的封闭式任务表示，无法处理开放式和个性化的人机交互。

**核心矛盾**：要做到真正的个性化辅助，机器人不仅需要理解当前环境，还需要跨越长时间推理人类的行为模式、偏好和习惯——这在现有框架中完全缺失。

**本文目标**：建立一个支持持续、开放式人机协作的框架，机器人能随时间学习并适应个体人类的特征。

**切入角度**：用LLM模拟具有心理特征驱动的长期行为人类，并设计反馈机制让机器人逐步个性化。

**核心 idea**：将人机协作从"单次情节式任务"扩展到"多天持续开放式交互"，让机器人通过日终反馈逐步理解个体的特征、习惯和时间依赖行为模式。

## 方法详解

### 整体框架
COOPERA 由三部分组成：(1) LLM驱动的模拟人类，具有性格特征和全天候行为；(2) 机器人辅助智能体，通过VLM和分类器推断人类意图并提供帮助；(3) 日终反馈机制，人类评价机器人表现并更新其对人类的认知画像。

### 关键设计

1. **特征驱动的人类模拟**:

    - 功能：生成具有心理特征和长期行为一致性的模拟人类
    - 核心思路：从合成对话数据推断大五人格特征，基于特征、环境信息和行为历史，由LLM在每个时间段生成意图和任务。使用Reflexion机制进行两轮自我纠错，通过记忆检索优化长上下文输入
    - 设计动机：需要人类行为在天内有时间依赖性（上午清洁→下午运动），在天间有多样性（周一和周二的9点活动不同）

2. **解耦式意图-任务推断**:

    - 功能：机器人推断人类当前意图并决定协作方式
    - 核心思路：将任务推断解耦为两阶段——先由VLM想象多个可能的意图，再由意图分类器筛选；对每个正分类意图推断具体任务，再由任务分类器筛选。分类器在每天结束时用反馈数据微调
    - 设计动机：人类行为具有天间多样性，VLM通过想象多种可能性+分类器筛选的方式捕捉正确的元意图/元任务集合

3. **人类画像持续更新**:

    - 功能：通过多天交互逐步理解人类的个性和习惯
    - 核心思路：每天结束后人类与机器人讨论，VLM从协作历史中推断并总结人类的特征、习惯和心理数据。这些画像信息被编入后续的VLM提示和分类器输入中
    - 设计动机：不同人类需要不同的协作策略，持续积累的画像使机器人能逐步从"通用助手"进化为"个性化助手"

### 损失函数 / 训练策略
VLM通过提示工程优化（无需训练），分类器基于 Mistral-7B 使用 LoRA 微调，输出二元 yes/no 判断。

## 实验关键数据

### 主实验

| 方法 | 天内提升 | 跨天提升 | 说明 |
|------|---------|---------|------|
| COOPERA（本文） | 最高 | 最高（仅次Oracle） | 全方位最优 |
| Oracle（给定意图） | 低 | 第一 | 已知意图但无法学习 |
| Direct Prompting | 几乎无 | 微弱 | 纯提示无法适应变化 |
| Direct Finetuning | 微弱 | 微弱 | 建立1-to-1映射，无法适应多样性 |

### 消融实验

| 实验 | 泛化精度 | 说明 |
|------|---------|------|
| 新场景泛化 | 0.465 vs 0.269基线 | 跨场景泛化相对容易 |
| 新人类泛化 | 0.343 vs 0.258基线 | 跨人类泛化更困难 |

### 关键发现
- 天内改进主要来自时间依赖学习：机器人学到"下午2点这个人通常在运动"
- 跨天改进来自人格画像积累：多天反馈让机器人更准确理解个体偏好
- 新人类泛化比新场景泛化难得多，因为人类行为的多样性远大于场景的差异

## 亮点与洞察
- 首次将人机协作扩展到多天持续交互设置，填补了一个重要空白
- 模拟人类通过心理特征驱动行为的设计非常巧妙，用户研究表明真实人类可以辨识出不同模拟人类的个性（准确率71.2%）
- VLM+分类器的解耦设计思路可迁移到其他需要推断不确定意图的场景

## 局限与展望
- 使用 Habitat 3.0 模拟环境，与真实世界存在差距
- 模拟人类行为基于LLM，可能存在与真实人类行为的偏差
- 仅考虑了pick-and-place和简单交互任务，未涉及更复杂的协作场景

## 相关工作与启发
- **vs Watch-and-Help**: WAH是单次封闭任务，COOPERA是多天开放式持续协作
- **vs Generative Agents**: 生成式智能体侧重语言层面社交模拟，COOPERA将心理特征驱动的行为与3D环境交互结合

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 持续开放式人机协作框架是全新设定
- 实验充分度: ⭐⭐⭐⭐ 多设定、用户研究、泛化实验全面
- 写作质量: ⭐⭐⭐⭐ 框架复杂但阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 为个性化机器人助手研究开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)
- [\[ICML 2025\] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](../../ICML2025/robotics/hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](../../ICML2025/robotics/founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[NeurIPS 2025\] Knolling Bot: Teaching Robots the Human Notion of Tidiness](knolling_bot_teaching_robots_the_human_notion_of_tidiness.md)

</div>

<!-- RELATED:END -->
