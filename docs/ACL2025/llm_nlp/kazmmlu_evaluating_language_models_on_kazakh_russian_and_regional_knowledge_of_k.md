---
title: >-
  [论文解读] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan
description: >-
  [LLM/NLP] 提出 KazMMLU，首个专为哈萨克斯坦设计的 MMLU 风格双语（哈萨克语+俄语）评估基准，包含 23,000 道来自真实教育材料的多选题，覆盖 STEM、人文、社会科学等多学科多教育层次，评估了 27 个多语言 LLM，揭示了当前模型在哈萨克语上的显著不足。
tags:
  - LLM/NLP
---

# KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan

## 基本信息

- **会议**: ACL2025
- **arXiv**: [2502.12829](https://arxiv.org/abs/2502.12829)
- **代码**: [https://huggingface.co/datasets/MBZUAI/KazMMLU](https://huggingface.co/datasets/MBZUAI/KazMMLU)
- **领域**: LLM NLP
- **关键词**: 哈萨克语, MMLU, 多语言评估, 低资源语言, 双语基准

## 一句话总结

提出 KazMMLU，首个专为哈萨克斯坦设计的 MMLU 风格双语（哈萨克语+俄语）评估基准，包含 23,000 道来自真实教育材料的多选题，覆盖 STEM、人文、社会科学等多学科多教育层次，评估了 27 个多语言 LLM，揭示了当前模型在哈萨克语上的显著不足。

## 研究背景与动机

### 问题定义
尽管哈萨克斯坦拥有超过两千万人口，但其语言和文化在 NLP 领域仍然严重欠缺代表性。哈萨克语作为突厥语族语言，有超过 1400 万人使用，但专门的 LLM 模型和评估基准极为稀缺。

### 现有不足

**现有多语言基准缺乏哈萨克斯坦特定内容**：GlobalMMLU、XCOPA、XGLUE 等跨语言基准不包含哈萨克语或缺少当地文化内容

**翻译基准的局限**：大多数现有资源严重依赖从英语翻译，缺乏本地文化丰富性

**已有哈萨克语 NLP 数据集任务单一**：KazNERD（命名实体识别）、KazQAD（问答）、KazSANDRA（情感分析）聚焦窄任务，不评估推理和领域知识

**TUMLU 缺乏哈萨克斯坦语境**：虽然覆盖突厥语言，但无国家特定内容

### 动机
构建一个反映哈萨克斯坦双语教育体系的大规模评估基准，真实评估 LLM 在哈萨克语和俄语上的知识与推理能力。

## 方法详解

### 整体框架

KazMMLU 遵循 MMLU 框架设计，包含单选题，每题 4-5 个选项，覆盖不同学科和教育层次。

### 数据集构建

**数据来源**：
- 国家考试（iTest.kz、ymnik.kz、oltest.kz）
- 教科书（Book - Shyn Kitap）
- 专业认证材料

**三种采集策略**：
1. 自动在线爬取（占 85%）
2. 扫描书籍人工转录
3. 在线资源人工提取

**语言分布**：
- 哈萨克语：10,969 题（48%）
- 俄语：12,031 题（52%）

**教育层次**：
- 高中：哈萨克语 + 俄语题目
- 大学/专业：仅俄语题目（反映哈萨克斯坦高教以俄语为主的现实）

**学科覆盖**：

| 组别 | 学科示例 |
|---|---|
| STEM | 生物、化学、信息学、数学、物理、医学 |
| 人文 | 哈萨克历史、哈萨克文学、世界历史、哲学心理学 |
| 社会科学 | 经济学、法学、管理营销、地理、社会学 |
| 语言 | 哈萨克语、俄语、阅读素养 |

### 质量控制

- 两名专业标注员（至少学士学位，精通哈/俄双语）人工审查所有题目
- 验证正确性和完整性，丢弃有错误或缺失组件的题目
- 编写脚本检测重复、验证元数据、消除格式问题
- 所有题目均为西里尔文字（哈萨克斯坦正式教育标准）

### 关键设计

**Prompt 配置**：
1. 哈萨克语 prompt + 英语字母输出
2. 英语 prompt + 英语字母输出

**答案选择方法**：
- 开源模型：next-token prediction，对各选项概率做 softmax 选最大
- 闭源模型：free-text generation + 字符串匹配提取答案

## 实验

### 主实验结果

在英语 prompt 下，27 个模型的表现（Average Accuracy）：

| 模型 | STEM | Social Sci. | Humanities | Language | Other | Average |
|---|---|---|---|---|---|---|
| GPT-4o | 70.0 | 81.9 | 83.3 | 73.4 | 62.1 | **76.6** |
| DeepSeek V3 | 77.6 | 81.3 | 78.9 | 61.2 | 65.1 | **76.9** |
| Gemma-2-27B-IT | 57.3 | 60.5 | 63.2 | 39.1 | 48.3 | 57.4 |
| Llama3.1-70B | 58.0 | 59.1 | 57.4 | 41.8 | 49.3 | 56.2 |
| YandexGPT | 54.8 | 70.6 | 63.7 | 42.6 | 57.0 | 60.2 |
| Sherkala-Chat-8B | 43.1 | 49.8 | 50.0 | 34.9 | 38.3 | 45.6 |
| KazakhLLM-8B | 40.3 | 45.6 | 44.0 | 31.3 | 38.6 | 41.7 |
| BLOOMZ-7B | 23.8 | 24.1 | 23.4 | 23.9 | 22.5 | 23.8 |

- GPT-4o 和 DeepSeek V3 表现最佳（~76-77%），远超其他模型
- **Language 类别始终最难**，所有模型得分最低
- 指令微调影响因模型而异：Llama3.1-70B-instruct 比 base 差 7.9%，但 8B 版本 instruct 提升 4.9%

### 哈萨克语 vs 俄语表现
- 所有模型在俄语上表现略优于哈萨克语
- GPT-4o 在哈萨克语最佳（76.90%），DeepSeek V3 在俄语最佳（81.4%）
- 差距可能源于训练数据可用性、语言复杂度或分词差异

### Few-shot 分析
- 所有模型随 shot 数增加一致提升
- Qwen-2.5-7B 和 Mistral-7B-v0.3 受益最大
- 最大精度跳跃在 0-shot → 1-shot 之间
- 英语 prompt 在 1/2/3-shot 下一致优于哈萨克语 prompt

### 否定敏感性分析
- 筛选含否定短语（жоқ, емес, не 等）的 2,554 道题
- DeepSeek V3 对否定最鲁棒，Llama3.1-70B 在阅读素养上降幅最大（57.1% → 50.0%）

### 模型置信度分析
- 模型置信度与准确率强相关（r > 0.9）
- 问题长度对置信度影响微弱

### 关键发现
1. 即使最佳模型在 KazMMLU 上也仅达 ~77%，远低于英语 MMLU 表现
2. 专门为哈萨克语微调的模型（KazakhLLM-8B, Sherkala-Chat-8B）表现不如通用大模型
3. 英语 prompt 一致优于哈萨克语 prompt，暗示模型的哈萨克语理解能力不足
4. 教育层次对性能有影响：开源模型在大学级别题目上表现明显下降

## 亮点与洞察

1. **第一个本地化 MMLU**：专为哈萨克斯坦设计，包含当地历史、传统和语言学内容，非简单翻译
2. **反映真实双语教育体系**：高中哈/俄双语 + 大学以俄语为主的结构设计恰当
3. **大规模可靠**：23,000 题全部经过人工验证，数据来源权威（国家考试等）
4. **全面的模型评估**：27 个模型横跨开源/闭源、不同尺寸、不同语系优化
5. **否定敏感性和模型置信度分析**：提供了超越准确率的多维评估视角

## 局限性

1. **仅支持文本模态**：未涵盖涉及图像、音频等多模态问题
2. **缺乏显式推理评估**：选择题形式无法评估模型的推理过程和开放式问答能力
3. **静态评估**：预定义题目无法完全捕获模型在动态真实场景中的泛化能力
4. **全部为西里尔文字**：未涵盖哈萨克斯坦正在推进的拉丁字母转写

## 相关工作

- **哈萨克语 LLM**：KazakhLLM (ISSAI, 2024)、Sherkala (Koto et al., 2025)
- **多语言基准**：MMLU (Hendrycks et al., 2021)、ArabicMMLU (Koto et al., 2024)、CMMLU (Li et al., 2024)、IndoMMLU (Koto et al., 2023)
- **突厥语评估**：TUMLU (Isbarov et al., 2025)、SIGTURK (Maxutov et al., 2024)
- **哈萨克语 NLP**：KazNERD (Yeshpanov et al., 2022)、KazQAD (Yeshpanov et al., 2024)

## 评分 ⭐⭐⭐⭐

- 创新性：⭐⭐⭐⭐ — 首个哈萨克斯坦本地化 MMLU，具有开创意义
- 实用性：⭐⭐⭐⭐⭐ — 对低资源语言 LLM 评估有重要推动作用
- 方法新颖度：⭐⭐⭐ — 基准构建方法成熟但非技术创新
- 实验充分度：⭐⭐⭐⭐⭐ — 27 模型 + 多维度分析（few-shot/否定/置信度/跨语言）非常全面

<!-- RELATED:START -->

## 相关论文

- [Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)

<!-- RELATED:END -->
