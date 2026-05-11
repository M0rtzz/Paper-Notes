---
title: >-
  [论文解读] Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models
description: >-
  [ACL 2025][Codenames] 将桌游Codenames实现为LLM评测基准，LLM同时扮演线索给出者(Spymaster)和猜测者(Field Operative)，在13种不同难度实验中与确定性对手对战，14个模型中最佳(o3-mini)胜率仅49%，揭示了LLM在词汇关联、策略选择和纠错能力上的显著局限。
tags:
  - ACL 2025
  - Codenames
  - 游戏评测
  - 概念生成
  - LLM评测
  - 合作博弈
---

# Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models

**会议**: ACL 2025  
**arXiv**: [2502.11707](https://arxiv.org/abs/2502.11707)  
**代码**: [GitHub](https://github.com/clembench/clembench) (codenames目录)  
**领域**: LLM评测  
**关键词**: Codenames, 游戏评测, 概念生成, 语用推理, 合作博弈

## 一句话总结

将桌游Codenames实现为LLM评测基准，LLM同时扮演线索给出者(Spymaster)和猜测者(Field Operative)，在13种不同难度实验中与确定性对手对战，14个模型中最佳(o3-mini)胜率仅49%，揭示了LLM在词汇关联、策略选择和纠错能力上的显著局限。

## 研究背景与动机

**领域现状**：LLM的评测面临"评测危机"——传统参考答案对比模式不适合聊天场景，且训练数据污染问题严重。游戏作为交互式评测环境正在兴起。

**现有痛点**：(a) 现有评测无法充分测试LLM的语用推理和合作能力；(b) Codenames需要即兴概念生成(ad-hoc concept forming)、心智理论(theory of mind)和共创能力，是极具挑战性的测试场景；(c) 先前的Codenames LLM研究让两个LLM对战，非确定性导致结果不可复现。

**切入角度**：使用确定性模拟对手(每轮翻开1个词)消除随机性，通过控制词表、频率、歧义性等变量设计13种实验。

## 方法详解

### 整体框架

基于clembench框架实现。LLM扮演蓝队的Spymaster和Field Operative，对抗程序化的mock对手。Spymaster输出"CLUE: 线索词 + TARGETS: 目标词列表"，Field Operative收到线索后输出"GUESS: 猜测列表"。GameMaster管理游戏流程和规则检查。

### 关键设计

1. **13种实验设置**:
    - 风险等级：1个vs 5个刺客词
    - 词汇关联：简单(同类词分配给同队)vs困难(同类词分散到不同阵营)
    - 对手速度：每轮翻0/1/2个词
    - 词频：高频vs低频
    - 歧义性：多义词vs单义词
    - 抽象性：具体词vs抽象词
    - 设计动机：控制变量分析LLM在不同语言认知维度上的表现

2. **评估指标体系**:
    - clemscore：综合分(played比例×quality score)
    - 敏感度(Sensitivity)：揭露己方词的比例
    - 效率(Efficiency)：$\min(1, \frac{1}{2} \cdot \frac{\text{team words revealed}}{\text{turns}})$，基准为每轮2词
    - 错误分类：目标幻觉、猜测幻觉、猜测数量错误、将线索作为猜测

3. **确定性对手设计**:
    - 功能：模拟一个每轮翻开n个词的固定策略对手
    - 设计动机：消除双方都使用LLM带来的非确定性，确保实验可复现

## 实验关键数据

### 主实验

14个模型的总体排名(130个游戏实例)：

| 排名 | 模型 | clemscore | Played% | Quality Score |
|------|------|-----------|---------|--------------|
| 1 | o3-mini | 49.2 | **100.0** | 49.2 |
| 2 | Claude-3.5 | 46.9 | 93.8 | 50.0 |
| 3 | GPT-4o | 45.4 | 93.8 | 48.4 |
| 4 | Deepseek-r1 | 45.4 | 85.4 | **53.2** |
| 5 | Gemini-2.0 | 37.7 | 96.2 | 39.2 |
| 14 | Llama-3.1-8B | 14.6 | 52.3 | 27.9 |

### 消融实验

特定实验难度对比(Quality Score)：

| 实验 | o3-mini | GPT-4o | Deepseek-r1 |
|------|---------|--------|-------------|
| 词汇关联-简单 | 100.0 | 100.0 | 100.0 |
| 词汇关联-困难 | 20.0 | 10.0 | 28.6 |
| 对手速度-无(不翻词) | 80.0 | 77.8 | 62.5 |
| 对手速度-困难(翻2词) | **0.0** | **0.0** | **22.2** |
| 具体词 | 80.0 | 50.0 | 88.9 |
| 抽象词 | 20.0 | 60.0 | 50.0 |

### 关键发现

1. **所有模型简单关联100%通过，困难模式<30%**：说明LLM能识别明显的词汇类别但缺乏跨类别创造性关联
2. **Deepseek-r1是唯一在"困难对手"模式中获胜的模型(22.2%)**：推理模型在策略规划上有明显优势
3. **低频词不比高频词更难**（与人类直觉相反）：LLM的词汇知识覆盖不受频率影响
4. **抽象词确实更难**（与人类一致）：但GPT-4o是个例外(抽象60%>具体50%)
5. **开源模型主要输在指令遵循**：目标幻觉、猜测幻觉、将线索当猜测等错误在开源模型中频率高5-10倍
6. **Deepseek-r1效率最高(每轮平均2.2词)**，但也因激进策略触发更多刺客词

## 亮点与洞察

- **游戏评测的优势**：易于生成无限新实例(避免数据污染)，交互式评估更接近实际使用
- **策略分析深入**：通过效率/敏感度/错误类型三维分析揭示模型间的策略差异
- **案例分析精彩**：展示o3-mini在抽象词游戏中目标9个词但暴露4个对手词导致失败的完整过程
- **推理模型的优势与代价**：Deepseek-r1在策略上最激进但延迟111秒/查询(vs GPT-4o的0.81秒)

## 局限与展望

- 仅英文，多语言扩展有待进行
- 未深入分析模型生成线索的内部推理过程
- 确定性对手过于简单，未测试真正的博弈对抗
- 游戏规则较复杂，指令遵循失败率偏高可能掩盖真实的推理能力差异

## 相关工作与启发

- **clembench框架**：本文所基于的LLM游戏评测框架
- **BigBench Codenames**：测试涌现能力，但缺乏交互性和控制实验
- 启发：游戏评测不仅能测试语言能力，更能揭示策略思维、风险管理和合作能力等高级认知

## 评分

- 新颖性: ⭐⭐⭐⭐ 将Codenames转化为系统性LLM评测框架，13种实验设置设计精心
- 实验充分度: ⭐⭐⭐⭐ 14个模型、多维分析、定性定量结合
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，案例分析生动
- 价值: ⭐⭐⭐⭐ 弥补了LLM在语用推理和合作能力评测上的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory](papersplease_a_benchmark_for_evaluating_motivational_values_of_large_language_mo.md)
- [\[ACL 2025\] Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](cov_eval_evaluating_llms_from_code_security_perspective.md)
- [\[ACL 2025\] CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](cov_eval_evaluating_llms_from_code_security_perspective.md)

</div>

<!-- RELATED:END -->
