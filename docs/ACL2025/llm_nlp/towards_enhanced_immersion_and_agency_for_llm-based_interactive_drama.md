---
title: >-
  [论文解读] Towards Enhanced Immersion and Agency for LLM-based Interactive Drama
description: >-
  [ACL 2025][LLM/NLP][交互式戏剧] 提出沉浸感-代理感（Immersion-Agency）范式来理解LLM交互式戏剧，并设计Playwriting-guided Generation和Plot-based Reflection两种方法分别增强故事生成质量和玩家对故事的影响力。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "交互式戏剧"
  - "沉浸感"
  - "代理感"
  - "剧本生成"
  - "角色扮演Agent"
---

# Towards Enhanced Immersion and Agency for LLM-based Interactive Drama

**会议**: ACL 2025  
**arXiv**: [2502.17878](https://arxiv.org/abs/2502.17878)  
**代码**: [GitHub](https://github.com/gingasan/interactive-drama)  
**领域**: LLM NLP  
**关键词**: 交互式戏剧, 沉浸感, 代理感, 剧本生成, 角色扮演Agent

## 一句话总结

提出沉浸感-代理感（Immersion-Agency）范式来理解LLM交互式戏剧，并设计Playwriting-guided Generation和Plot-based Reflection两种方法分别增强故事生成质量和玩家对故事的影响力。

## 研究背景与动机

LLM交互式戏剧是一种新型的AI对话应用场景：玩家扮演故事中的角色，与LLM Agent扮演的其他角色对话互动，体验展开的故事。然而现有工作存在以下不足：

1. **缺乏理论框架**：之前的工作主要关注通用架构设计，未深入探讨交互体验的核心维度。本文引入经典交互叙事理论中的两个关键概念——沉浸感（Immersion，玩家融入故事的感觉）和代理感（Agency，玩家影响故事世界的能力）。
2. **故事生成质量不足**：LLM虽然在预训练阶段接触了大量文学作品，但微调过程缺乏对戏剧创作技巧的强调，导致生成的故事缺乏基本的戏剧结构和引人入胜的冲突。实验显示，GPT-4o和Qwen2.5-72b在无显式提示下几乎不使用任何叙事技巧。
3. **角色代理感被忽视**：之前的角色Agent构建很少考虑玩家行为如何有意义地影响角色反应和故事走向。

## 方法详解

### 整体框架

系统分为两大模块：(1) 剧本生成——通过 Playwriting-guided Generation 从玩家提供的前提段落生成高质量戏剧故事（含剧情结构、叙事技巧）；(2) 角色Agent——通过 Plot-based Reflection 让NPC根据玩家行为动态调整剧情链，增强代理感。

### 关键设计

1. **Playwriting-guided Generation（剧本引导生成）**：
    - 定义8种经典戏剧情境（如爱情、凤凰涅槃、灰姑娘、复仇等），基于亚里士多德三幕结构（设定、对抗、解决）描述。
    - 总结6种微观叙事技巧（悬念、反转、非线性叙事、多重叙事、讽刺、象征）。
    - 生成流程：采样1种戏剧情境+3种叙事技巧 → Writer LLM生成故事 → Critic LLM评审并给出改进建议 → Writer修改 → 重复3次选最佳 → 渐进式细节打磨。
    - 效果：叙事技巧使用率从基线的6-12%提升到28-74%（以GPT-4o为基础）。

2. **Plot-based Reflection（基于剧情的反思）**：
    - 角色Agent每隔k=5个交互步执行一次反思，分析玩家行为（情感、意图）的记忆，动态调整剧情链。
    - 每次反思限制为调整不超过1个未完成剧情或插入不超过1个新剧情，防止LLM过度修改导致剧情不连贯。
    - 使角色展现由玩家行为驱动的有意义的反应转变，如泄露秘密、提供陪伴、按特定方式推进剧情。

3. **混合Agent架构（Hybrid Architecture）**：
    - Director-Actor架构：导演Agent全局协调+独立Actor Agent各自扮演角色，适合强互动场景。
    - One-for-All架构：单一全局Agent扮演所有角色，效率更高，适合叙事为主的场景。
    - 混合方案根据场景特征动态切换两种架构，平衡性能与效率（推理加速1.49倍）。

### 损失函数 / 训练策略

本文不涉及模型训练，所有Agent基于GPT-4o的提示工程。关键策略包括：
- Sampling-Critic-Revise循环确保戏剧技巧正确应用
- 渐进式生成（从粗到细添加细节）
- 记忆系统保留所有对话历史在prompt中

## 实验关键数据

### 主实验

故事生成评估（50个前提段落，人类标注者评估胜率）：

| 方法 | 冲突（最佳↑/最差↓） | 悬念 | 情感张力 | 角色弧 | 技巧遵循率 |
|------|--------|------|---------|--------|-----------|
| Outline-First | 18%/34% | 10%/28% | 10%/50% | 18%/36% | - |
| Playwriting-Guided | **32%/24%** | **32%/22%** | **48%/16%** | **34%/20%** | 92% |
| w/o Critic & Revise | 24%/24% | 26%/34% | 18%/28% | 12%/32% | 66% |
| w/o Refinement | 26%/18% | 32%/26% | 24%/6% | 36%/12% | - |

角色Agent评估（5分制，手工剧本"车站七人"，10名人类+10名Agent玩家）：

| 架构 | 角色一致性 | 吸引力 | 叙事完整 | 进展 | 影响力 | 意图跟随 | 加速比 |
|------|----------|--------|---------|------|--------|---------|--------|
| Director-Actor | 3.9 | 4.2 | 3.8 | 3.6 | 4.2 | 3.9 | 1.00x |
| 混合架构 | 4.1 | 3.9 | 4.3 | **4.3** | 4.0 | 4.0 | 1.49x |
| w/o Reflection | 4.0 | 3.5 | 4.2 | 3.9 | 3.5 | 3.3 | 1.90x |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 去掉Critic & Revise | 技巧遵循率92%→66% | Critic LLM对确保技巧正确应用至关重要 |
| 去掉Refinement | 情感张力48%→24% | 渐进细化对情感细节贡献最大 |
| 去掉Plot-based Reflection | 影响力4.0→3.5, 意图跟随4.0→3.3 | 反思机制是代理感的核心 |
| 纯Director-Actor | 进展3.6 vs 混合4.3 | 多agent通信导致信息损失，影响叙事进展 |

### 关键发现

1. **渐进式细化对情感张力贡献最大**（10%→48%），因为情感通常来自文本中的细微细节。
2. **Plot-based Reflection不仅提升代理感，也增强角色吸引力**（3.5→3.9），推测是因为反思鼓励角色展现更强的共情。
3. **激进的Agent玩家反而在角色吸引力和影响力维度获得更高评分**，可能因为Agent更积极互动，高质量回应反过来给标注者留下深刻印象。
4. 算法能自动为不同主题匹配合适的戏剧情境（如浪漫→爱情、犯罪→救世主）。

## 亮点与洞察

- **理论贡献**：首次为LLM交互式戏剧建立沉浸感-代理感评价范式，从叙事学和心理学角度提供了分析框架。
- **戏剧技巧的系统化**：将经典戏剧理论（Polti的36种戏剧情境、亚里士多德三幕结构）与现代LLM prompt工程结合，是计算创造力的有趣实践。
- **人类评估为主**：拒绝使用LLM评估器，坚持由受过人文训练的标注者进行评估，因为文学作品的评价需要准确性和共情。
- **混合架构的实用性**：根据场景特征动态选择架构，在效率和质量间取得良好平衡。

## 局限与展望

1. **依赖GPT-4o**：所有Agent基于同一闭源模型，成本高且不可控。
2. **效率问题**：Playwriting-guided Generation比vanilla prompting慢10-12倍。
3. **反思边界控制**：LLM倾向于过度调整剧情，当前通过硬性限制解决，但这可能限制了更具创造性的适应。
4. **仅关注对话形式**：未探索场景生成（视觉、音乐等多模态元素）对沉浸感的增强。
5. **评估规模有限**：仅10名人类玩家和1个手工剧本，泛化性有待验证。

## 相关工作与启发

- **Mateas (2000)** 的Interactive Drama理论提供了Immersion和Agency的原始定义，本文将其操作化为可评估的维度。
- **Park等 (2023)** 的memory-based reflection关注记忆合成，而Plot-based Reflection关注剧情适应，两者是互补的并行技术。
- **Wu等 (2024)** 首先定义了LLM交互式戏剧的六要素和剧情链机制，本文在此基础上增加了反思和生成增强。
- 本文展示了将古典戏剧理论融入AI系统的可能性，对NPC AI、游戏叙事、教育模拟等领域有广泛启发。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 4 |
| 实验充分性 | 3 |
| 应用价值 | 4 |
| 写作质量 | 4 |
| 总分 | 3.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring](alignment_drift_in_cefr-prompted_llms_for_interactive_spanish_tutoring.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)

</div>

<!-- RELATED:END -->
