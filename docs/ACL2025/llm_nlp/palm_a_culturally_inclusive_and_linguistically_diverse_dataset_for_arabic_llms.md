---
title: >-
  [论文解读] Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs
description: >-
  [ACL 2025][LLM 其他][Arabic NLP] 由 44 名阿拉伯世界研究者历时一年社区驱动构建的 Palm 数据集，涵盖全部 22 个阿拉伯国家、20 个文化主题、10 种方言，共 17,411 条人工创建的指令对，用于评估和提升 LLM 的阿拉伯文化和方言能力。 核心问题： LLM 训练数据以英语为主导…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "Arabic NLP"
  - "Cultural Awareness"
  - "Dialectal Arabic"
  - "Instruction Dataset"
  - "LLM Evaluation"
  - "Community-Driven"
---

# Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs

**会议**: ACL 2025  
**arXiv**: [2503.00151](https://arxiv.org/abs/2503.00151)  
**代码**: [github.com/UBC-NLP/palm](https://github.com/UBC-NLP/palm)  
**领域**: LLM NLP / 多语言与文化对齐  
**关键词**: Arabic NLP, Cultural Awareness, Dialectal Arabic, Instruction Dataset, LLM Evaluation, Community-Driven

## 一句话总结

由 44 名阿拉伯世界研究者历时一年社区驱动构建的 Palm 数据集，涵盖全部 22 个阿拉伯国家、20 个文化主题、10 种方言，共 17,411 条人工创建的指令对，用于评估和提升 LLM 的阿拉伯文化和方言能力。

## 研究背景与动机

**核心问题**: LLM 训练数据以英语为主导，翻译数据引入西方/英美中心偏见。例如，有阿拉伯语 LLM 在用户祈祷后建议喝啤酒——完全违背阿拉伯文化价值观、宗教习俗和社会规范。

**阿拉伯世界的复杂性**:
   - 覆盖非洲和亚洲的 22 个国家，人口超 4.5 亿
   - 多样化的本地文化、习俗、传统、政治体系和社会实践
   - 语言多样性：经典阿拉伯语、现代标准阿拉伯语 (MSA)、各地方言 (DA)，方言间差异可达国家层级

**现有数据集的不足**:
   - **AYA**: 204K 多语言指令中仅 5K 阿拉伯语，文化覆盖有限
   - **AraDiCE**: 基于翻译和数据重定向，非原创；仅覆盖 6 种方言
   - **CIDAR**: 基于本地化，非从头创建
   - **BLEnD**: 仅覆盖 1 个阿拉伯国家
   - 无一数据集同时满足：覆盖全部 22 国 + 多种方言 + 全人工创建 + 开放式指令

**现有阿拉伯 LLM 的局限**: JAIS、AceGPT 等模型的指令数据主要是机器生成/翻译，缺乏对具体国家文化的评估

## 方法详解

### 团队结构

- **44 名研究者**: 全部为论文共同作者，来自 15 个阿拉伯国家的本地母语者
- **覆盖策略**: 15 个国家有本地研究者；其余 7 个（巴林、科摩罗、吉布提、伊拉克、利比亚、卡塔尔、索马里）由邻国研究者补充
- **多元性保障**: 每国至少 2 名标注者，来自不同地区以确保文化和方言覆盖

### 标注指南

- **迭代开发**: 约 3 个月迭代完善，4 名资深成员主导，最终版本达 100 页
- **两大类指令**:
  1. **通用类**: MSA 编写的通用知识（科学、技术等）
  2. **国家特定类**: 反映国家文化的内容，可用 MSA 或当地方言——覆盖庆典、习俗、地理、历史、谚语、食物等 20 个主题

### 数据构建流程

1. **收集**: 从可靠来源（官方信息、学术文献等）收集内容
2. **标注**: 使用 Label Studio 平台，按国家分组
3. **质量保证**: 每周会议 + Slack 频道实时协作
4. **交叉审核**: 完成后团队成员交叉审核，确保每个样本至少被 2 名审核者检查

### 数据集分析

- **总规模**: 17,411 条指令对
- **高资源国家** (>500 条): 埃及、约旦、毛里塔尼亚、摩洛哥、巴勒斯坦、沙特、苏丹、叙利亚、突尼斯、阿联酋、也门 + 通用类 (1,109 条) = 16,066 条 (92%)
- **低资源国家** (~100 条): 阿尔及利亚、巴林等 11 国 = 1,345 条 (8%)
- **方言数据**: 10 个高资源国家各约 380 条方言示例，共 4,211 条方言指令
- **指令类型 6 大类**: 总结/解释、指导/程序、事实/信息查询、创意/建构、分析/评估、叙事/描述
- **数据划分**: 训练 13,559 + 公开测试 1,926 + 私有测试 1,926

### 评估设置

#### Surface-Level 评估
- **重复率**: 测量生成回答中的序列重复
- **语言一致性**: 阿拉伯语指令是否得到阿拉伯语回答
- **方言一致性**: 方言指令是否得到方言回答

#### LLM-as-Judge 评估
- **Judge 模型**: GPT-4o、CMDR+、QWEN 2.5-72B
- **评分**: 1-10 分的正确性评分，基于指令和 ground truth
- **评估者一致性**: ICC = 0.68（良好一致性）

#### 人类评估
- **规模**: 92 条 MSA + 4 国各 20 条方言 = 172 条
- **评估者**: 来自数据创建团队的同一批研究者
- **与自动评估一致性**: Pearson r = 0.76 (p < 0.05)，ICC = 0.78

## 实验

### Surface-Level 结果（Table 2）

| 模型 | 重复率↓ | 语言一致性↑ | 方言一致性↑ |
|------|---------|------------|------------|
| GPT-4o | 0.00 | 91.07 | 8.33 |
| Claude-3.5-Sonnet | 0.00 | 91.02 | 9.44 |
| Qwen2.5-72B | 0.00 | 91.33 | 8.33 |
| gemma-2-9b | **42.47** | 90.76 | 6.11 |
| Llama-3.1-8B | 8.20 | 91.28 | 10.56 |

**关键发现**:
- 所有模型方言一致性极低（<11%），即使用方言 prompt 也约 90% 用 MSA 回答
- Gemma-2-9b 有严重的重复生成问题（42%）
- 语言一致性整体良好（~91%）

### LLM-as-Judge 正确性评分

**总体排名** (Figure 4a): GPT-4o 和 Claude-3.5-Sonnet 最佳（中位数>6.0）> CMDR+/Qwen2.5-72B (~5.8-6.0) > AceGPT/Gemma/Llama-70B (~4.5-5.0) > Jais-13b/Llama-8B (~3.0-4.0)

**国家差异** (Figure 4b):
- GPT-4o: 叙利亚 7.5、吉布提 7.3（最高）；部分国家较低
- Jais-13b: 沙特 2.2（最低）
- Claude: 也门 7.0 但黎巴嫩仅 5.8，国家间差异显著

**方言差异** (Figure 5):
- 埃及和突尼斯方言表现较好（GPT-4o: 8.1/8.4）
- 摩洛哥和巴勒斯坦方言对多数模型较难
- 小模型在方言上普遍<4.0

### 人类评估结果（Table 3 - 方言）

| 国家 | AceGPT-32B | Llama-8B | Qwen-72B | Claude-3.5 | Jais-13b |
|------|-----------|---------|---------|-----------|---------|
| 埃及 | **6.47** | 4.26 | 4.71 | 4.15 | 4.08 |
| 摩洛哥 | 4.55 | 2.87 | 4.44 | **6.23** | 3.10 |
| 叙利亚 | 3.27 | 3.40 | 4.03 | **4.65** | 2.27 |
| 也门 | 2.13 | 1.85 | 2.58 | **4.28** | 2.90 |

各模型在不同国家方言上优劣互异，没有单一模型全面领先——AceGPT 擅长埃及方言，Claude 在摩洛哥/叙利亚/也门表现最佳。

### 模型规模与性能

模型规模与性能正相关（>70B 模型通常优于小模型），但即使是最大的模型也存在文化盲点——某些国家始终表现较差。

## 亮点与洞察

1. **社区驱动的数据构建范式**: 44 位本地研究者全员参与标注且全部署名为作者，在 NLP 数据集构建中树立了包容性和归属感的标杆
2. **覆盖全面性无出其右**: 首个覆盖全部 22 个阿拉伯国家 × 20 个文化主题 × 10 种方言的数据集，填补了重要空白
3. **方言处理能力的揭露**: 所有模型方言一致性<11%，暴露了当前 LLM 在非标准语言形式上的严重不足
4. **文化公平性差异**: 某些国家（如埃及、阿联酋）在现有模型中被较好表示，而其他国家（如伊拉克、毛里塔尼亚、也门）严重欠缺
5. **评估方法的验证**: 人类评估与 LLM-as-Judge 的高度一致性 (r=0.76, ICC=0.78) 为自动评估方法在文化领域的应用提供了信心

## 局限性

1. 低资源国家的内容由邻国标注者补充，可能缺乏本地深度
2. 许多阿拉伯国家内部有多种地区方言，当前覆盖尚不完全
3. LLM 自动评估在处理方言和微妙文化元素时可能存在偏差
4. 部分国家数据量较少（~100 条），可能不足以代表该国文化全貌
5. 未评估 Fanar、Allam 等新近发布的阿拉伯语 LLM

## 相关工作

- **阿拉伯语 LLM**: JAIS、AceGPT、Jasmine、NileChat、Fanar、Allam 等
- **阿拉伯语编码器**: AraBERT、CAMeLBERT、MARBERT 等
- **文化评估**: World Value Survey 对齐评估 (AlKhamissi et al.)、CALText 偏见检测 (Naous et al.)
- **多语言指令数据**: AYA (204K多语言)、BLEnD (文化盲测)

## 评分 ⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐ 数据集构建方法本身不新，但覆盖规模和社区驱动模式是独特贡献
- **数据质量**: ⭐⭐⭐⭐⭐ 全人工创建 + 100 页标注指南 + 交叉审核 + 人类评估验证
- **分析深度**: ⭐⭐⭐⭐ 国家级/方言级/主题级的多维度分析非常丰富
- **社会影响**: ⭐⭐⭐⭐⭐ 为 4.5 亿阿拉伯语人口的文化代表性迈出重要一步

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Geo-Culturally Grounded LLM Generations](geocultural_grounded_llm.md)
- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)
- [\[ACL 2025\] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge](enhancing_transformation_from_natural_language_to_signal_temporal_logic_using_ll.md)
- [\[ACL 2025\] A Modular Dataset to Demonstrate LLM Abstraction Capability](a_modular_dataset_to_demonstrate_llm_abstraction_capability.md)
- [\[ACL 2025\] NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews](newsinterview_a_dataset_and_a_playground_to_evaluate_llms_grounding_gap_via_info.md)

</div>

<!-- RELATED:END -->
