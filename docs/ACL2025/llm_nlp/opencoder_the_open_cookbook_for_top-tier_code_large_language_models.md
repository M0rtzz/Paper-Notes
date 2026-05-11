---
title: >-
  [论文解读] OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models
description: >-
  [ACL2025][LLM/NLP][Code LLM] 提出OpenCoder，一个完全开源的代码大语言模型（含1.5B和8B版本），不仅性能达到顶级水平，更作为"open cookbook"开放了可复现的数据处理流水线、预训练数据集、消融实验和训练协议，为代码智能研究提供基础设施。
tags:
  - ACL2025
  - LLM/NLP
  - Code LLM
  - Data Curation
  - Pretraining Pipeline
  - Reproducible
  - RefineCode
---

# OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models

**会议**: ACL2025  
**arXiv**: [2411.04905](https://arxiv.org/abs/2411.04905)  
**代码**: [OpenCoder](https://opencoder-llm.github.io)  
**领域**: 代码大语言模型 / LLM预训练  
**关键词**: Code LLM, Data Curation, Pretraining Pipeline, Reproducible, RefineCode  

## 一句话总结

提出OpenCoder，一个完全开源的代码大语言模型（含1.5B和8B版本），不仅性能达到顶级水平，更作为"open cookbook"开放了可复现的数据处理流水线、预训练数据集、消融实验和训练协议，为代码智能研究提供基础设施。

## 研究背景与动机

- **开源代码LLM的差距**：虽然Copilot、Cursor等商业工具改变了开发者工作流，但开源代码LLM在性能上仍落后于闭源SOTA模型
- **透明度缺失**：领先的代码LLM普遍不公开训练数据和处理流水线，限制了社区的基线建设和深入研究
- **可复现性稀缺**：现有开源代码LLM很少同时开放训练数据、数据处理流程、SFT语料和中间检查点
- **三大目标**：(1)为机制可解释性研究提供透明强基线；(2)深入研究预训练和指令数据的策划流程；(3)解锁基于透明代码LLM的多样化定制方案

## 方法详解

### 整体框架

OpenCoder的训练分为三个阶段：预训练 → 退火(Annealing) → 两阶段指令微调(SFT)

### 预训练数据：RefineCode

构建了约960B token的高质量代码预训练数据集RefineCode，覆盖607种编程语言。主要包含原始代码和代码相关网页数据两部分。

**原始代码处理流水线**：

1. **预处理**：排除>8MB文件，按文件扩展名筛选607种编程语言
2. **去重（优先执行）**：
    - 精确去重：SHA256哈希匹配，保留星数最高+最新提交的文件（移除约75%完全重复文件）
    - 模糊去重：5-gram MinHash + LSH（16 bands, 128 rows），移除约6%文件
    - 关键发现：文件级去重优于仓库级去重
3. **转换**：
    - 版权声明移除：超过15%的代码文件包含重复性版权信息
    - PII消减：正则替换密码、邮箱、IP等为占位符
4. **过滤（三类启发式规则）**：
    - 自然语言过滤规则：文件大小、行数等通用文本特征
    - 通用代码过滤规则：变量数、平均函数长度等代码特征
    - **语言特定过滤规则（首创）**：为Python/C/C++/C#/Java/JavaScript/Go/HTML共8种语言设计专用规则
5. **数据采样**：降采样Java(449GB→200GB)和HTML(474GB→64GB)，最终约730B token

**代码相关网页数据**（约330GB）：
- 用FastText分类器从CommonCrawl三轮迭代召回
- 代码相关域发现 + URL人工标注
- 从FineWeb、SkyPile、AutoMathText等额外召回
- 训练分类器从GitHub文本中识别代码相关内容（额外178GB）

### 退火阶段数据

退火是预训练到SFT的桥梁，共约100B token：
- **原始分布数据（84%）**：保持与预训练阶段一致的数据分布，防止灾难性遗忘
- **算法语料**：从原始数据中采样含"leetcode"、"def solution"等关键词的文件
- **合成数据**：
    - 高质量代码片段：以算法语料为种子，LLM生成自包含函数+测试用例，保留通过测试的数据
    - 代码教科书：基于hqcode数据集用Qwen2-72B生成交互式代码分析

### 模型架构

| 参数 | 1.5B | 8B |
|------|------|------|
| 层数 | 24 | 32 |
| 隐藏维度 | 2240 | 4096 |
| 注意力头 | 14 | 32 |
| KV头 | 14 | 8 |
| 激活函数 | SwiGLU | SwiGLU |
| 词表大小 | 96,640 | 96,640 |
| 上下文长度 | 4096 | 8192 |

### 两阶段指令微调

**数据来源**：
- 开源数据：Evol-Instruct、Infinity-Instruct（LLM过滤代码相关）、McEval、WildChat用户查询
- Educational-Instruct：高质量种子数据 → 评分筛选 → LLM生成QA + 测试用例验证
- Package-Instruct：用最新库文档生成准确的工具调用指令（解决过时API问题）
- Large-scale Diverse-Instruct：清洗网页 → 任务规范定义 → LLM生成问答 → 代码执行验证

**Stage 1（理论知识）**：RealUser 0.7M + Diverse 2.3M + Infinity 1.0M，batch=4096, LR=2e-5
**Stage 2（实践编码）**：McEval 36K + Evol 111K + Educational 110K + Package 110K，batch=512, LR=5e-5

### 去污染

严格执行SFT数据去重：移除含HumanEval/MBPP测试集entry points的数据，并进行10-gram重叠去除。

## 实验

### Base Model评估

| 模型 | 大小 | HE | HE+ | MBPP | MBPP+ | BigCodeBench |
|------|------|------|------|------|------|------|
| DS-Coder-6.7B | 6.7B | 47.6 | 39.6 | 70.2 | 56.6 | 41.1 |
| Yi-Coder-9B | 9B | 53.7 | 46.3 | 48.4 | 40.7 | 42.9 |
| Qwen2.5-Coder-7B | 7B | 61.6 | 53.0 | 76.9 | 62.9 | 45.8 |
| StarCoder2-15B | 15B | 46.3 | 37.8 | 66.2 | 53.1 | 38.4 |
| **OpenCoder-8B** | **8B** | **66.5** | **63.4** | **79.9** | **70.4** | 40.5 |
| **OpenCoder-1.5B** | **1.5B** | **54.3** | **49.4** | **70.6** | **58.7** | 24.5 |

OpenCoder-8B在HumanEval和MBPP上超越所有同级别模型，HE+达63.4远超次优的Qwen2.5-Coder-7B的53.0。

### Instruct Model评估

| 模型 | HE | HE+ | MBPP | LiveCodeBench |
|------|------|------|------|------|
| DS-Coder-V2-Lite-Instruct | 81.1 | 75.0 | 82.3 | 24.3 |
| Yi-Coder-9B-Chat | 85.4 | 79.9 | - | 20.5 |
| Qwen2.5-Coder-7B-Instruct | 88.4 | 84.1 | - | 26.8 |
| **OpenCoder-8B-Instruct** | **83.5** | **78.7** | **80.2** | 21.0 |

OpenCoder-8B-Instruct的HumanEval Pass@1达83.5，超越所有完全开源（含可复现数据集）的模型。

### 关键消融实验发现

1. **去重策略**：文件级去重优于仓库级去重，通过保持数据多样性提升下游性能
2. **GitHub Stars过滤**：按星数过滤可能减少数据多样性，导致次优结果
3. **代码相关网页数据**：对模型能力有显著提升
4. **退火数据质量**：高质量数据在退火阶段的重要性远高于数量
5. **两阶段SFT**：分阶段策略让模型先获得广泛能力再精进代码任务，优于一阶段策略

## 亮点与洞察

1. **透明度标杆**：首个同时开放数据处理Pipeline、可复现预训练数据集、大规模SFT语料和中间检查点的顶级代码LLM
2. **语言特定过滤规则首创**：考虑不同编程语言的独特特性设计过滤规则，如Python的pass语句频率、C的goto使用
3. **PCA可视化验证**：用CodeBERT嵌入可视化证明RefineCode相比Stack v2分布更紧凑、离群值更少
4. **Package-Instruct解决实际痛点**：用最新文档生成SFT数据，解决LLM使用过时API版本的问题
5. **训练效率**：用RefineCode训练的1.5B模型在600B token处就超越Stack v2训练的同规模模型
6. **严格去污染**：10-gram粒度去重确保评估公正性

## 局限性

1. **整体性能不敌最强闭源**：BigCodeBench等复合任务上仍落后于Qwen2.5-Coder等使用更大训练token的模型
2. **计算预算有限**：1.5B模型的退火数据混合比例可能非最优
3. **非英语覆盖不足**：中文代码相关数据处理较少
4. **LiveCodeBench表现一般**：在复杂算法推理任务上与顶级方法有差距
5. **合成数据质量天花板**：依赖教师模型的质量限制了合成数据上限

## 相关工作

- **代码LLM**：CodeLlama、StarCoder系列、DeepSeek-Coder、Qwen-Coder等
- **预训练数据**：The Stack v1/v2、FineWeb等开源代码数据集
- **数据质量**：数据清洗、去重、质量过滤等方面的研究
- **指令微调**：Code-Alpaca、Evol-Instruct等代码SFT数据构建方法
- **评估基准**：HumanEval、MBPP、BigCodeBench、LiveCodeBench等

## 评分 ⭐⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐ 方法本身非全新，但开放cookbook理念具有范式意义
- **实验充分性**：⭐⭐⭐⭐⭐ 大量消融实验验证每个设计决策
- **实用价值**：⭐⭐⭐⭐⭐ 完全开源的顶级代码LLM+可复现数据流水线
- **写作质量**：⭐⭐⭐⭐⭐ 每一步决策的动机和效果都有清晰论述

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)
- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] Open-Set Living Need Prediction with Large Language Models](open-set_living_need_prediction_with_large_language_models.md)

</div>

<!-- RELATED:END -->
