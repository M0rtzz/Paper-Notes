---
title: >-
  [论文解读] Content Fuzzing for Escaping Information Cocoons on Social Media
description: >-
  [ACL 2026][社会计算][信息茧房] 提出 ContentFuzz，一个从内容创作者视角出发的置信度引导模糊测试框架，通过 LLM 改写帖子使其在保持人类解读含义不变的前提下改变机器推断的立场标签，从而突破社交媒体信息茧房。
tags:
  - "ACL 2026"
  - "社会计算"
  - "信息茧房"
  - "立场检测"
  - "模糊测试"
  - "内容改写"
  - "推荐系统"
---

# Content Fuzzing for Escaping Information Cocoons on Social Media

**会议**: ACL 2026  
**arXiv**: [2604.05461](https://arxiv.org/abs/2604.05461)  
**代码**: 无  
**领域**: 社交计算 / 对抗学习  
**关键词**: 信息茧房, 立场检测, 模糊测试, 内容改写, 推荐系统

## 一句话总结
提出 ContentFuzz，一个从内容创作者视角出发的置信度引导模糊测试框架，通过 LLM 改写帖子使其在保持人类解读含义不变的前提下改变机器推断的立场标签，从而突破社交媒体信息茧房。

## 研究背景与动机

**领域现状**：社交媒体平台使用立场检测作为推荐和排序管道中的重要信号，将帖子主要路由给持相同观点的受众，减少了跨立场曝光。这限制了不同意见的传播范围，阻碍了建设性讨论。

**现有痛点**：现有打破信息茧房的方法主要是平台侧的算法干预（如多样性重排序），但这些方法由平台控制，个人用户和内容创作者无法修改推荐算法，也看不到帖子如何被过滤、排序和分发。创作者缺乏主动扩展内容触达范围的工具。

**核心矛盾**：用户和创作者有扩大跨群体曝光的需求，但缺乏可操作的技术手段——唯一能控制的是内容本身。

**本文目标**：从创作者角度，探索如何通过内容改写突破信息茧房——找到保持人类解读立场但改变机器分类立场的语义保持改写。

**切入角度**：借鉴软件测试中的模糊测试（fuzzing）方法论，将立场检测模型视为"被测系统"，迭代发现使其分类结果翻转的输入变体。

**核心 idea**：用立场检测模型的置信度反馈引导 LLM 生成语义保持改写——置信度下降说明改写在探索分类器决策边界附近，反复迭代直到标签翻转或耗尽预算。

## 方法详解

### 整体框架
ContentFuzz 从原始帖子出发，迭代执行：选择种子→LLM 变异生成候选改写→运行立场检测器获取置信度→保留降低置信度的候选作为未来种子→直到某候选改变了预测立场或迭代耗尽。

### 关键设计

1. **置信度引导反馈**:

    - 功能：指导 LLM 生成朝"正确"方向（接近决策边界）演化的改写
    - 核心思路：每次变异后运行立场分析器获取预测立场和置信度。如果新候选的置信度低于种子，说明它在推动模型远离当前决策，将其加入种子池。如果立场标签翻转则立即返回成功
    - 设计动机：盲目改写效率低，置信度反馈提供了"温度"信号——温度越低越接近决策边界

2. **种子调度策略**:

    - 功能：优先选择最有潜力的种子进行下一轮变异
    - 核心思路：维护种子池，按置信度排序——置信度越低的种子越接近决策边界，越值得进一步变异。同时考虑种子已被变异的次数，避免过度利用单一种子
    - 设计动机：当计算资源有限时，聚焦最有希望的搜索方向至关重要

3. **语义保持变异**:

    - 功能：生成保持原意但可能改变机器判断的改写
    - 核心思路：用 LLM（如 GPT-4）生成改写，通过精心设计的提示指令要求保留核心观点和态度，但允许修改措辞、句式、修辞手法等表面特征。同时生成多个候选以增加覆盖面
    - 设计动机：与对抗攻击不同，ContentFuzz 要求改写对人类读者而言含义完全不变——这是"逃离茧房"而非"欺骗分类器"

### 损失函数 / 训练策略
ContentFuzz 是推理时框架，无需训练。优化目标是最小化立场检测器对原始标签的置信度直到标签翻转。

## 实验关键数据

### 主实验

| 设置 | 立场模型 | 成功率 | 语义保持 | 流畅度 |
|------|---------|-------|---------|-------|
| 英文数据集 | BERT-based | 高 | 强 | 高 |
| 英文数据集 | LLM-based | 高 | 强 | 高 |
| 中文数据集 | BERT-based | 高 | 强 | 高 |
| 跨主题迁移 | 多模型 | 稳定 | 稳定 | 稳定 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无置信度反馈（随机变异） | 低成功率 | 无方向性探索效率极低 |
| 无种子调度（均匀选择） | 降低 | 浪费资源在低潜力种子上 |
| 完整 ContentFuzz | **最优** | 反馈+调度协同作用 |

### 关键发现
- ContentFuzz 在 3 个数据集、2 种语言、4 个立场检测模型上均有效
- 改写在保持语义完整性的同时成功翻转机器立场标签
- 微小的措辞变化就能显著影响立场检测器的输出，揭示了这些模型的脆弱性

## 亮点与洞察
- **视角转换**是最大亮点：从"平台如何打破茧房"转为"创作者如何突围"，这是一个被忽视但实际可操作的方向
- **fuzzing 方法论的跨域迁移**很巧妙——将软件测试的核心理念（迭代变异+反馈引导+种子调度）无缝应用到 NLP 场景
- **揭示了立场检测模型的脆弱性**——语义不变的改写就能翻转预测，这对推荐系统的可靠性提出了质疑

## 局限与展望
- 依赖黑盒/灰盒访问立场检测模型——完全黑盒的推荐系统可能无法获取置信度
- 成功改写是否真能改变推荐算法的分发决策未在真实平台上验证
- 可能被滥用于操纵舆论——需要考虑伦理边界

## 相关工作与启发
- **vs 对抗攻击**：对抗攻击追求最小扰动翻转标签，ContentFuzz 追求语义保持的自然改写
- **vs 平台侧干预**：互补关系——平台控制算法，创作者控制内容

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个内容侧信息茧房突破框架，视角独特
- 实验充分度: ⭐⭐⭐⭐ 多语言多模型验证全面
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法类比恰当
- 价值: ⭐⭐⭐⭐ 对信息多样性和推荐系统鲁棒性有双重价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ACL 2026\] DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects](dia-harm_dialectal_disparities_in_harmful_content_detection_across_50_english_di.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](bayesian_social_deduction_with_graph-informed_language_models.md)
- [\[ACL 2026\] The Proxy Presumption: From Semantic Embeddings to Valid Social Measures](the_proxy_presumption_from_semantic_embeddings_to_valid_social_measures.md)
- [\[NeurIPS 2025\] Precise Information Control in Long-Form Text Generation](../../NeurIPS2025/social_computing/precise_information_control_in_long-form_text_generation.md)

</div>

<!-- RELATED:END -->
