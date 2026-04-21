---
title: >-
  [论文解读] Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models
description: >-
  [ACL 2025][LLM/NLP][inverse scaling] 通过重新定义物理/数学常量和计量单位（如"令 π=500"），系统研究 LLM 在逆缩放任务中的表现，发现模型规模越大越倾向于锚定记忆中的原始值而拒绝遵循 prompt 的重新定义，且错误信心（拒绝弃权而给出错误答案）随规模上升。
tags:
  - ACL 2025
  - LLM/NLP
  - inverse scaling
  - redefinition
  - anchoring
  - memorization
  - reasoning flexibility
  - physical constants
---

# Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models

**会议**: ACL 2025  
**作者**: Elena Stringli, Maria Lymperaiou, Giorgos Filandrianos, Athanasios Voulodimos, Giorgos Stamou (NTUA)  
**arXiv**: [2502.12821](https://arxiv.org/abs/2502.12821)  
**代码**: —  
**领域**: LLM/NLP, 推理评估  
**关键词**: inverse scaling, redefinition, anchoring, memorization, reasoning flexibility, physical constants  

## 一句话总结

通过重新定义物理/数学常量和计量单位（如"令 π=500"），系统研究 LLM 在逆缩放任务中的表现，发现模型规模越大越倾向于锚定记忆中的原始值而拒绝遵循 prompt 的重新定义，且错误信心（拒绝弃权而给出错误答案）随规模上升。

## 研究背景与动机

- **核心问题:** 逆缩放任务（inverse scaling tasks）指模型性能随规模增大而下降的任务，能暴露 LLM 推理能力的潜在缺陷。重新定义任务（redefinition）属于"强先验"（strong priors）类逆缩放，人类可以轻松完成（100% 准确率），但 LLM 可能失败。
- **现有不足:** 逆缩放在文献中研究不足。Inverse Scaling Prize (McKenzie et al., 2024) 初步揭示了问题，但对重新定义任务的系统研究——包括不同难度级别、不同响应格式、不同提示策略的影响——仍属空白。
- **研究动机:** 随着 LLM 在高风险场景（科学计算、工程决策）中的应用增多，理解它们在面对与训练知识冲突的指令时能否灵活调整推理路径至关重要。

## 方法详解

### 整体框架

实验设计涵盖两类重新定义对象 × 多级难度 × 三级问题难度 × 两种响应格式：
1. **常量重新定义**：15 个物理/数学常量（π, e, φ, c, G, h 等），包含赋值（assignment）和交换（swap）两种方式
2. **单位重新定义**：15 种计量单位（分钟-秒、千克-克、米-厘米等）的换算关系修改

### 关键设计

1. **递增难度的重新定义级别**：赋值分三级——$R_{a1}$ 接近原值（π=4.5）、$R_{a2}$ 偏差数量级（π=500）、$R_{a3}$ 不合理值（π=-10）；交换分两级——$R_{s1}$ 接近值交换（π↔φ）、$R_{s2}$ 差异大交换（π↔h=6.626×10⁻³⁴）。
2. **递增难度的问题级别**：$Q_1$ 简单检索（"重新定义后 π 的第一个非零数字是什么？"）、$Q_2$ 简单运算（"π × 3 = ?"）、$Q_3$ 多步推理（"地球的表面积是多少？"——需用重新定义的 π 计算）。
3. **两种响应格式**：自由生成（FF）和多项选择（MC），后者设置了具有迷惑性的干扰项。

### 损失函数

本工作为评估研究，不涉及模型训练。

## 实验

### 主实验——不同模型族和规模的锚定率

| 模型族 | 规模趋势 | 锚定率变化 | 关键观察 |
|--------|---------|-----------|---------|
| GPT 系列 | 3.5→4→4o | **上升** | 更大模型更频繁使用原始值 |
| Llama 系列 | 8B→70B→405B | **上升** | 锚定行为在 405B 尤为严重 |
| Gemma 系列 | 2B→9B→27B | **上升** | 即使最小模型也有显著锚定 |
| 人类基线 | — | — | 100% 准确率（可轻松重新定义） |

### 消融实验——影响因素分析

| 因素 | 结果 |
|------|------|
| 问题难度 Q1→Q2→Q3 | 锚定率随难度上升，多步推理时最严重 |
| 重新定义难度 Ra1→Ra2→Ra3 | 偏离原值越大，锚定率越高 |
| 赋值 vs 交换 | 交换任务中锚定率更高（两个记忆值冲突） |
| 自由生成 vs 多选 | 多选格式降低锚定率，但不能消除 |
| Chain-of-Thought | CoT 有时反而增加锚定（模型在推理过程中"回忆"原始值） |
| 系统提示强调 | 强调"必须使用新值"的系统提示有帮助但效果有限 |

### 关键发现

- 更大模型不仅更频繁出错，还更少选择"弃权"——它们宁可自信地给出基于原始值的错误答案，也不承认不确定性
- 提示技术（CoT、系统提示、响应格式）可以影响锚定率，但无法从根本上消除先验知识的锚定效应
- 单位重新定义比常量重新定义更容易被遵循，可能因为单位转换比常量值在训练数据中出现频率更低
- 交换型重新定义最困难——模型需要同时覆盖两个记忆值并正确使用

## 亮点

- 实验设计系统全面：15 个常量 × 15 个单位 × 多级难度 × 多级问题 × 多种格式，覆盖面广
- 揭示了一个反直觉但重要的现象：LLM 的"知识"可能是推理灵活性的障碍
- 100% 人类准确率与 LLM 的逆缩放形成鲜明对比，突显了记忆与推理的本质差异

## 局限性

- 仅测试了英文 prompt，未验证多语言设置下的表现
- 未探索微调或 RLHF 对锚定行为的影响
- 实验中的常量重新定义较为简单（直接赋值），未测试更复杂的条件重新定义
- 部分模型的 API 版本可能影响结果的可复现性
- 未分析锚定行为是否与特定训练数据来源（如维基百科条目频率）相关

## 相关工作

- **逆缩放:** Inverse Scaling Prize (McKenzie et al., 2024) 揭示强先验、不良模仿等逆缩放原因
- **LLM 推理:** 记忆与推理的界限模糊（Wu et al., 2024; Mahowald et al., 2024），LLM 在修改表述下性能下降
- **知识冲突:** 上下文信息与存储知识冲突时 LLM 倾向于依赖训练数据（Xu et al., 2024）

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验完整度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style](investigating_context-faithfulness_in_large_language_models_the_roles_of_memory_.md)
- [TESS 2: A Large-Scale Generalist Diffusion Language Model](tess_2_a_large-scale_generalist_diffusion_language_model.md)
- [A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant](a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)
- [LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_vs_human_judges_study.md)
- [BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework](bipro_zero-shot_chinese_poem_generation_via_block_inverse_prompting_constrained_.md)

<!-- RELATED:END -->
