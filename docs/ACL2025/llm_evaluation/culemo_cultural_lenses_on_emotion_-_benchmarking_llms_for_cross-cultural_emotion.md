---
title: >-
  [论文解读] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding
description: >-
  [LLM评测] 提出 CuLEmo，首个评估文化感知情感预测的多语言基准数据集，涵盖 6 种语言/文化（阿姆哈拉语、阿拉伯语、英语、德语、印地语、西班牙语），通过 400 个文化相关场景评估 LLM 的跨文化情感理解能力，发现情感表达在不同文化间存在显著差异且 LLM 表现参差不齐。 问题定义 情感是语言和文化依赖的…
tags:
  - "LLM评测"
---

# CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding

## 基本信息

- **会议**: ACL2025
- **arXiv**: [2503.10688](https://arxiv.org/abs/2503.10688)
- **代码**: [https://github.com/llm-for-emotion/culemo](https://github.com/llm-for-emotion/culemo)
- **领域**: LLM评测
- **关键词**: 跨文化情感理解, 多语言LLM评估, 文化感知, 情感预测, 情感分析

## 一句话总结

提出 CuLEmo，首个评估文化感知情感预测的多语言基准数据集，涵盖 6 种语言/文化（阿姆哈拉语、阿拉伯语、英语、德语、印地语、西班牙语），通过 400 个文化相关场景评估 LLM 的跨文化情感理解能力，发现情感表达在不同文化间存在显著差异且 LLM 表现参差不齐。

## 研究背景与动机

### 问题定义
情感是语言和文化依赖的。同一事件在不同文化中可能激发截然不同的情感反应。例如，在餐厅不给小费：在美国会引发内疚，但在中国这很正常，在日本给小费甚至可能被视为冒犯。当前 LLM 是否具备文化感知的情感理解能力？

### 现有不足

**关键词依赖**：现有情感基准大多依赖关键词情感识别，忽略了深层文化维度

**翻译偏差**：许多跨语言情感数据集通过翻译英语标注数据获得，可能引入不完整或误导性的洞察

**文化覆盖不足**：此前研究仅限于有限的文化/语言组合（如仅美国 vs 日本）和有限的情感类别（如仅骄傲/羞耻）

**标注不公平**：翻译后的数据标签来自英语文化立场，无法反映本地文化视角

### 研究问题
- RQ1: LLM 能否提供文化感知的情感回应？
- RQ2: 哪些文化在 LLM 中更好地被表征？
- RQ3: LLM 能否仅通过 prompt 语言识别国家文化？
- RQ4: prompt 语言如何影响 LLM 的文化感知情感理解？

## 方法详解

### 整体框架

CuLEmo 包含 400 个文化相关事件/场景，每个场景以"How would you feel when..."的形式呈现，覆盖 6 种语言和文化。

### 数据集构建

**事件收集**：
- 手工创建场景 + 网络搜索 + LLM prompt 收集
- 涵盖各目标国家的传统、事件、规范和行为
- 不包含显式情感关键词（区别于传统情感数据集）
- 参考 ISEAR 数据格式："When I … situations that cause a specific emotion"

**十大类别**：

| 类别 | 题数 |
|---|---|
| 家庭关系 | 45 |
| 社交礼仪和互动 | 65 |
| 个人外貌和着装规范 | 32 |
| 文化和宗教实践 | 62 |
| 性和亲密关系 | 38 |
| 职业场景 | 28 |
| 饮食礼仪 | 35 |
| 个人隐私 | 25 |
| 情感和心理情境 | 40 |
| 公共行为和规范 | 30 |

**翻译流程**：
- Google Translate 初翻为 5 种目标语言（阿拉伯语、阿姆哈拉语、德语、印地语、西班牙语）
- 母语者审核和修正（翻译在标注之前完成，避免标签偏差）

**标注**：
- Amazon Mechanical Turk + 自定义 POTATO 标注工具
- 每个实例至少 5 名来自目标国家的母语标注者
- 无多数投票时追加 2 名标注者
- 6 个情感类别：joy, fear, sadness, anger, guilt, neutral
- 薪酬 $12/小时

### 关键设计

**语言与文化的覆盖考量**：
- 类型学多样性：5 种语言、4 种文字系统
- 地理多样性：东方 vs 西方
- 资源可用性：低资源 vs 高资源语言
- 6 个目标国家：阿联酋、美国、德国、埃塞俄比亚、印度、墨西哥

**Prompt 构建**：
- 英语 prompt：所有指令、输入、预期输出均为英语
- In-language prompt：全部为目标语言
- 带/不带国家上下文："You live in <<country name>>,"

**评估任务**：
1. 情感预测（6分类：joy, fear, sadness, anger, guilt, neutral）
2. 情感分析（3分类：positive, negative, neutral）

## 实验

### 模型选择
- **开源**：LLaMA-3 (3.2-3B, 3.1-8B)、Gemma (2B, 9B)、Aya (expanse-8b, 101-13B)、Ministral (3B, 8B)
- **闭源**：GPT (3.5, 4)、Gemini-1.5、Claude (3.5-sonnet, 3-opus)

### 主实验结果（情感预测，带国家上下文）

| LLM | USA(EN) | UAE(EN) | UAE(AR) | Germany(EN) | Germany(DE) | India(EN) | India(HI) | Mexico(ES) |
|---|---|---|---|---|---|---|---|---|
| GPT-4 | 0.60 | 0.55 | 0.48 | 0.51 | 0.50 | 0.40 | 0.40 | 0.65 |
| Ministral-8B | 0.65 | 0.61 | 0.06 | 0.58 | **0.72** | 0.39 | 0.19 | 0.32 |
| Gemini1.5-flash | 0.56 | 0.56 | 0.56 | 0.46 | 0.48 | 0.41 | 0.41 | 0.64 |
| Claude-3.5-sonnet | 0.57 | 0.48 | 0.54 | 0.46 | 0.42 | 0.40 | 0.36 | 0.61 |
| Gemma-2-2B | 0.62 | 0.59 | 0.45 | **0.64** | 0.54 | 0.38 | 0.29 | 0.56 |

### 标注者间文化差异发现

**情感标签分布**：
- 德语标注中性（无情感）比例最高（87%）
- 阿姆哈拉语（29.5%）和阿拉伯语（22%）愤怒情感最多
- 印度标注与其他国家分歧最大（与德国仅 29% 一致）

**典型案例**：
- "不给服务员小费"：美国=guilt，阿联酋=neutral，埃塞俄比亚=guilt
- "穿黑色衣服参加婚礼"：美国=sadness，埃塞俄比亚=anger，其余=neutral
- "女性在街上穿短裤"：埃塞俄比亚=anger，其余=neutral

### 文化表征偏差

不带国家上下文时，LLM 对美国、墨西哥、德国的情感更准确，对阿联酋、埃塞俄比亚、印度明显偏低，说明训练数据中某些文化更普遍。

### Prompt 语言影响

- 去除国家上下文后准确率显著下降
- 加入 "You live in <<country name>>" 后一致提升（GPT-4 印地语 +21%，阿姆哈拉语 Claude-3.5 +9%）
- **语言本身不能可靠传达文化上下文**
- 对低资源语言，英语 prompt + 国家上下文往往优于 in-language prompt

### 情感分析（3分类）

- 粗粒度分类表现好于细粒度情感预测
- GPT-4 印地语提升 +22%
- 墨西哥文化（西班牙语）上 Claude-3-opus 和 GPT-4 均达 75%

### 关键发现
1. 情感确实是文化依赖的，同一事件跨文化引发截然不同的情感
2. LLM 对高资源文化（美国、西欧）表现更好，对低资源文化（印度、埃塞俄比亚）明显退化
3. 显式提供国家上下文对文化感知情感理解至关重要
4. 更大的模型不一定更好：Gemma-2-2B 和 Ministral-8B 在某些文化上可竞争甚至超越闭源模型

## 亮点与洞察

1. **首个文化感知情感基准**：不依赖翻译标注，而是相同场景跨文化独立标注
2. **"语言≠文化"的关键发现**：仅通过 prompt 语言无法传达文化上下文，需显式指定国家
3. **低资源文化的性能缺口**：揭示了 LLM 训练数据中的文化偏见
4. **实用建议**：对低资源语言，用英语 prompt + 指定目标国家往往是最优策略
5. **标注设计巧妙**：事件无显式情感词，要求真正的文化推理而非关键词匹配

## 局限性

1. **事件数量有限**：每种语言仅 400 个问题，难以全面覆盖文化差异
2. **情感类别有限**：仅 6 类，缺少惊讶、厌恶等细粒度类别
3. **多数投票的缺陷**：排除了少数视角，可能丢失文化多元性
4. **模型规模受限**：仅评估了小/中型开源模型，大模型表现未知
5. **仅覆盖 6 种语言/国家**：泛化性有限

## 相关工作

- **文化感知NLP**：CulturalBench (Liu et al., 2024)、NORMSAGE (Fung et al., 2023)
- **跨语言情感**：XLM-EMO (Bianchi et al., 2022)、De Bruyne (2023)
- **文化情感分析**：Havaldar et al. (2023) — 美/日骄傲/羞耻、Ahmad et al. (2024) — 豪萨语
- **多语言LLM评估**：MEGA (Ahuja et al., 2023)

## 评分 ⭐⭐⭐⭐

- 创新性：⭐⭐⭐⭐ — 首个文化感知情感基准，问题定义清晰
- 实用性：⭐⭐⭐⭐ — 对构建文化敏感的 AI 系统有直接指导意义
- 方法新颖度：⭐⭐⭐ — 基准构建+评估为主，无新模型/方法
- 实验充分度：⭐⭐⭐⭐ — 11 个模型、多 prompt 配置、情感+情感分析双任务、全面对比

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](androidlab_autonomous_agent.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)
- [\[ACL 2025\] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)

</div>

<!-- RELATED:END -->
