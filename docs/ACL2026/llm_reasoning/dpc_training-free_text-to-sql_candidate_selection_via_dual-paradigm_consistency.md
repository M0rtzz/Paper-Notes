---
title: >-
  [论文解读] DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency
description: >-
  [ACL 2026][LLM推理][SQL选择] DPC 将 Text-to-SQL 的候选选择从"在隐藏数据上猜测"转化为"在可见数据上确定性验证"：构造最小区分数据库（MDD）使冲突 SQL 产生不同结果，再用 Python/Pandas 解作为参考锚点通过跨范式一致性选择正确候选，在 BIRD 和 Spider 上超越 Self-Consistency 最高 2.2%。
tags:
  - ACL 2026
  - LLM推理
  - SQL选择
  - 双范式一致性
  - 最小区分数据库
  - 训练免
  - 对抗环境合成
---

# DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency

**会议**: ACL 2026  
**arXiv**: [2604.15163](https://arxiv.org/abs/2604.15163)  
**代码**: [GitHub](https://github.com/HKUSTDial/DPC)  
**领域**: NLP应用 / Text-to-SQL  
**关键词**: SQL选择, 双范式一致性, 最小区分数据库, 训练免, 对抗环境合成

## 一句话总结
DPC 将 Text-to-SQL 的候选选择从"在隐藏数据上猜测"转化为"在可见数据上确定性验证"：构造最小区分数据库（MDD）使冲突 SQL 产生不同结果，再用 Python/Pandas 解作为参考锚点通过跨范式一致性选择正确候选，在 BIRD 和 Spider 上超越 Self-Consistency 最高 2.2%。

## 研究背景与动机

**领域现状**：Text-to-SQL 采用"生成再选择"范式——生成 K 个候选 SQL 再选最优。但存在显著的"生成-选择差距"：Pass@K 远高于 Pass@1（如 BIRD 上 58.8% vs ~50%），说明正确 SQL 已在候选中但选择失败。

**现有痛点**：（1）Self-Consistency（多数投票）在模型系统性偏差时失败——模型一致性地收敛于错误答案；（2）LLM-as-Judge 因"符号盲目性"无法心算复杂 SQL 的执行状态；（3）训练式验证器需昂贵标注且领域脆弱性差。

**核心矛盾**：SQL 验证的三重挑战——部分可观察性（真实数据库太大放不进上下文）、符号盲目性（LLM 无法内部模拟 SQL 执行）、固有确认偏差（模型偏向自己生成的候选）。

**本文目标**：设计一个训练免的 SQL 选择框架，将验证从概率性猜测转化为确定性判断。

**切入角度**：构造一个精心设计的小数据库使冲突 SQL 必然产生不同结果，再用 Python 代码作为独立推理路径来交叉验证。

**核心 idea**：对抗性环境合成（MDD）+ 双范式执行（SQL vs Python）+ 一致性投票。

## 方法详解

### 整体框架
DPC 四阶段流水线：（1）候选聚类与配对：按执行结果聚类，选冠军（最大类）和挑战者（第二大类）；（2）对抗环境合成：Slicer 精简 schema + Tester 生成对抗性数据填充 MDD，使冠军和挑战者产生不同结果；（3）双范式执行：在 MDD 上执行 SQL + Solver 生成 Python/Pandas 脚本执行；（4）一致性验证：用 BS-F1 指标比较 SQL 结果与 Python 参考的语义等价性。

### 关键设计

1. **最小区分数据库（MDD）构造**:

    - 功能：将 SQL 验证从部分可观察转为完全可观察
    - 核心思路：分两步构造。Slicer Agent 迭代精简 schema 到仅包含候选 SQL 所需的表和列，通过 Dry-Run 验证结构完整性。Tester Agent 对抗性地生成数据，通过判别反馈循环确保冠军和挑战者在 MDD 上产生不同执行结果。例如区分 INNER JOIN 和 LEFT JOIN 需要特定的无匹配键记录
    - 设计动机：随机数据采样不足以区分语义相似但逻辑不同的 SQL——需要对抗性地针对候选之间的具体差异来构造数据

2. **双范式一致性验证**:

    - 功能：通过独立推理路径打破 LLM 的确认偏差
    - 核心思路：利用 LLM 在命令式语言（Python）上的能力优于声明式语言（SQL）的"能力差异"——Python 在预训练语料中覆盖更广，且命令式代码迫使模型显式规划数据操作步骤。Solver Agent 在 MDD 上生成 Python/Pandas 解，其执行结果作为代理真值（proxy ground truth）来验证 SQL 候选
    - 设计动机：让同一个 LLM 用两种不同范式解同一问题——如果两种范式的答案一致则大概率正确，不一致则可以通过 Python（更可靠的范式）来仲裁

3. **二部图软 F1 指标（BS-F1）**:

    - 功能：鲁棒地量化 SQL 结果和 Python 结果之间的语义等价性
    - 核心思路：处理跨范式比较的两大挑战：类型不兼容（SQL DECIMAL vs Python float、SQL NULL vs Python NaN）和排序歧义（无 ORDER BY 的 SQL 结果是无序集）。先归一化类型，再用匈牙利算法做行级最优匹配，计算匹配行的列重叠率得到 Soft-F1
    - 设计动机：标准执行准确率（EX）要求严格相等，在跨范式场景中会大量误判

### 损失函数 / 训练策略
DPC 是纯推理时框架，不涉及训练。所有 agent 都通过提示工程实现，使用同一个 LLM 骨干。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 执行准确率 | vs Self-Consistency |
|--------|------|-----------|-------------------|
| BIRD | DPC | 最优 | +2.2% |
| Spider | DPC | 最优 | +1-2% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无 MDD（直接用样本数据） | 显著下降 | 对抗性数据构造是关键 |
| 无 Python 范式 | 下降 | 单范式验证不如双范式 |
| 无 BS-F1（用严格 EX） | 下降 | 跨范式比较需要软匹配 |

### 关键发现
- MDD 的判别反馈循环是核心——随机数据采样在大部分情况下无法区分冲突 SQL
- Python 范式作为验证锚点比 SQL 本身更可靠，印证了 LLM 在命令式语言上能力更强的假设
- BS-F1 比严格 EX 在跨范式验证中显著减少误判
- DPC 在多个 LLM 骨干上一致优于 Self-Consistency

## 亮点与洞察
- **将选择问题转化为构造性验证问题**的思路非常优雅——与其猜哪个 SQL 对，不如构造一个能揭示差异的"实验"
- **跨范式一致性**利用了 LLM 在不同编程语言上的能力差异——Python 作为"第二意见"来交叉验证 SQL，类似于科学中的独立复现
- MDD 的对抗性构造思想可以推广到任何需要区分相似候选的选择任务

## 局限与展望
- MDD 构造需要多轮 LLM 调用，增加了推理延迟和成本
- 仅聚焦冠军-挑战者二选一，可能错过排名更低的正确候选
- Python 解的质量依赖 LLM 的 Python 编程能力，并非总是可靠
- 对于非常复杂的 SQL（如多层嵌套子查询），MDD 构造的成功率可能下降

## 相关工作与启发
- **vs Self-Consistency**: SC 靠多数投票在系统性偏差下失效，DPC 用确定性执行证据替代概率性投票
- **vs LLM-as-Judge**: Judge 模式受符号盲目性限制无法模拟执行，DPC 通过实际执行获取确定性证据
- **vs 训练式验证器**: 训练需要标注且领域脆弱，DPC 完全训练免

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 对抗性环境合成+双范式一致性的组合是全新思路
- 实验充分度: ⭐⭐⭐⭐ BIRD+Spider 两个标准 benchmark，多 LLM 骨干
- 写作质量: ⭐⭐⭐⭐⭐ 问题形式化清晰，pipeline 描述逻辑流畅

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction](../../NeurIPS2025/llm_reasoning/sql-of-thought_multi-agentic_text-to-sql_with_guided_error_correction.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](../../NeurIPS2025/llm_reasoning/sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[AAAI 2026\] A Reasoning Paradigm for Named Entity Recognition](../../AAAI2026/llm_reasoning/a_reasoning_paradigm_for_named_entity_recognition.md)
- [\[AAAI 2026\] Text-to-Scene with Large Reasoning Models](../../AAAI2026/llm_reasoning/text-to-scene_with_large_reasoning_models.md)

<!-- RELATED:END -->
