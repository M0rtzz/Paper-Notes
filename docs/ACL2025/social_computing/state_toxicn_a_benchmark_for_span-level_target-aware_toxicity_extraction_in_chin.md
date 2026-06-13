---
title: >-
  [论文解读] STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection
description: >-
  [ACL 2025][社会计算][仇恨言论] 构建了首个中文 span 级仇恨言论检测数据集 STATE ToxiCN（8029 条帖子、9533 个四元组标注），提出 Target-Argument-Hateful-Group 四元组标注体系，并首次建立了中文仇恨俚语标注词典（830 条），系统评估了多种 LLM 在 span 级中文仇恨言论检测上的能力。
tags:
  - "ACL 2025"
  - "社会计算"
  - "仇恨言论"
  - "细粒度标注"
  - "中文"
  - "目标-论点提取"
  - "仇恨俚语"
---

# STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection

**会议**: ACL 2025  
**arXiv**: [2501.15451](https://arxiv.org/abs/2501.15451)  
**代码**: [有](https://github.com/shenmeyemeifashengguo/STATE-ToxiCN)  
**领域**: NLP / 仇恨言论检测 / 中文NLP  
**关键词**: 仇恨言论, 细粒度标注, 中文, 目标-论点提取, 仇恨俚语

## 一句话总结

构建了首个中文 span 级仇恨言论检测数据集 STATE ToxiCN（8029 条帖子、9533 个四元组标注），提出 Target-Argument-Hateful-Group 四元组标注体系，并首次建立了中文仇恨俚语标注词典（830 条），系统评估了多种 LLM 在 span 级中文仇恨言论检测上的能力。

## 研究背景与动机

仇恨言论检测领域存在两个关键空白：

**中文span级检测资源缺失**：现有中文仇恨言论数据集（COLD、SWSR、ToxiCN 等）均停留在帖子级别，缺乏 span 级细粒度标注。但仇恨的强度和方向性与其目标 (Target) 和论点 (Argument) 密切相关

**中文仇恨俚语研究空白**：中文作为表意文字，同义词和近义词丰富，使得仇恨俚语形式多样——包括谐音替换、拆字合字、历史典故等手法——这些对模型检测构成严峻挑战

中文 span 级检测的特殊挑战：
- 灵活的语序（如倒装句改变了主谓宾结构）
- 缺乏空格的词分隔符
- 仇恨俚语的隐蔽性

## 方法详解

### 整体框架

本文贡献包括两部分：
1. STATE ToxiCN 数据集：span 级四元组标注
2. 中文仇恨俚语标注词典

### 关键设计

1. **四元组标注体系 (Target-Argument-Hateful-Group)**

   | 组件 | 含义 | 示例 |
   |------|------|------|
   | Target | 仇恨言论的对象/目标 | "男同" (Gay people) |
   | Argument | 支撑仇恨的论据/论点 | "艾滋高发群体" |
   | Hateful | 是否构成仇恨 | hate / non-hate |
   | Group | 目标群体分类 | LGBTQ, others |

   设计动机：相比 TBO 数据集的三元组，增加了 Group 维度以支持群体级分析

2. **数据来源与过滤**

    - 基于 ToxiCN 帖子级数据集进行 span 级标注
    - 过滤低质量内容：移除广告、随机字符、过短（<5 字）或过长（>500 字）的文本
    - 排除缺乏明确 Target-Argument 结构的帖子

3. **标注流程与质量控制**

    - 多样化标注团队：不同性别、年龄、民族、地区、教育背景
    - 独立标注：每条文本至少两人独立标注
    - 定期交叉验证：20% 样本由其他标注者重新标注
    - 专家仲裁：对分歧样本由领域专家团队裁定
    - IAA：Target span κ=0.65, Argument span κ=0.61, Hateful κ=0.68, Group κ=0.75

4. **中文仇恨俚语词典构建**

    - 从真实在线论坛（知乎、贴吧）收集常用仇恨俚语
    - 每条俚语标注：频繁目标群体 (Group) + 解释 (Explanation)
    - 解释内容涵盖：字面意义、引申含义、对目标群体仇恨的原因、常见使用模式
    - 共 830 条标注俚语
    - 利用共享在线文档进行动态维护

5. **仇恨俚语的隐蔽手法类型**

    - 谐音替换
    - 拆字合字：如"默"（黑犬）→ 种族歧视
    - 历史典故引用
    - 这些手法使传统关键词方法难以检测

### 损失函数 / 训练策略

评估实验中对开源模型进行微调：
- 使用基础 prompt（含任务定义、输出格式和预测要求）
- 闭源模型通过 API 调用并额外提供 2 个示例（1 仇恨 + 1 非仇恨）
- 采用 Macro-F1 作为主要评估指标

## 实验关键数据

### 数据集统计

| 指标 | 数值 |
|------|------|
| 总帖子数 | 8,029 |
| 含仇恨内容帖子 | 4,942 (61.55%) |
| 总四元组数 | 9,533 |
| 仇恨四元组 | 6,063 (63.60%) |
| 性别歧视 | 1,663 (17.44%) |
| 种族歧视 | 1,232 (12.92%) |
| 地域歧视 | 1,323 (13.88%) |
| LGBTQ | 628 (6.59%) |
| 多群体 | 866 (9.08%) |
| 仇恨俚语词典 | 830 条 |

### 主实验（Span级检测综合评估，Macro-F1）

| 模型 | Target (Hard/Soft) | Argument (Hard/Soft) | 四元组 (Hard/Soft) |
|------|-------------------|---------------------|-------------------|
| mT5-base | 59.15/70.55 | 28.63/67.03 | 16.60/38.61 |
| LLaMA3-8B | 64.07/73.74 | 36.72/70.82 | 24.27/46.08 |
| Qwen2.5-7B | 63.96/74.64 | 35.42/70.36 | 23.70/47.03 |
| ShieldGemma-9B | 63.40/74.31 | 34.40/71.11 | 23.49/47.14 |
| GPT-4o | — | — | — |
| Claude-3.5-Sonnet | — | — | — |

### 闭源 LLM 表现（API, 2-shot）

| 模型 | Target (Hard) | 四元组 (Hard) |
|------|--------------|---------------|
| LLaMA3-70B | 30.54 | 3.69 |
| Qwen2.5-72B | 40.94 | 8.74 |
| Gemini-1.5-Pro | 29.80 | — |

### 关键发现

1. **Hard matching 远低于 Soft matching**：中文 span 边界的模糊性使精确匹配极为困难
2. **微调后的 7B 模型显著优于 70B+ 闭源 LLM**：说明 span 级提取需要任务特定训练
3. **Argument 提取难度远高于 Target 提取**：因为论点表述更加灵活多变
4. **完整四元组提取 F1 极低（Hard < 25%）**：表明 span 级中文仇恨言论检测仍极具挑战性
5. **安全领域模型（ShieldLM、ShieldGemma）并未表现出优势**：它们在帖子级设计，不直接适用于 span 级

## 亮点与洞察

1. **填补重要空白**：首个中文 span 级仇恨言论数据集，同时也是首个有可解释标注的中文仇恨俚语词典
2. **四元组设计务实**：Target-Argument-Hateful-Group 结构既捕捉了仇恨的方向性，又支持群体级分析
3. **标注质量保证充分**：多样化标注者、交叉验证、专家仲裁的组合确保了数据质量
4. **仇恨俚语分析有价值**：揭示了中文仇恨言论中独特的语言现象（谐音、拆字等），对检测系统设计有指导意义

## 局限与展望

1. 数据规模相对较小（8K 帖子），可能不足以覆盖中文仇恨言论的所有形式
2. 仅来自知乎和贴吧两个平台，可能存在平台特异性偏差
3. 未探索多模态仇恨言论（图文结合）
4. 当前评估主要使用基础 prompt，可能未充分发挥 LLM 潜力
5. 仇恨俚语词典需要持续更新以覆盖新兴表达

## 相关工作与启发

- TBO (Zampieri et al., 2023)：英文 span 级仇恨言论数据集，首创 Target-Argument-Harmful 三元组
- ToxiCN (Lu et al., 2023)：中文帖子级仇恨言论数据集，STATE ToxiCN 在其基础上扩展到 span 级
- HateXplain (Mathew et al., 2021)：英文 span 级数据集，但无 Argument 标注
- ToxiCloakCN (Xiao et al., 2024)：评估 LLM 对伪装仇恨的鲁棒性

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个中文 span 级数据集 + 首个标注的中文仇恨俚语词典
- **实验充分度**: ⭐⭐⭐⭐ 12 个模型评估、多粒度指标、Hard/Soft 双重匹配
- **写作质量**: ⭐⭐⭐⭐ 数据集构建过程描述详尽，标注体系清晰
- **价值**: ⭐⭐⭐⭐⭐ 为中文仇恨言论检测领域提供了重要基础资源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](implihatevid_video_hate.md)
- [\[ACL 2025\] Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings](exploring_multimodal_challenges_in_toxic_chinese_detection_taxonomy_benchmark_an.md)
- [\[ACL 2025\] HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](hateday_global_hate_speech.md)
- [\[ACL 2025\] Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch](silencing_empowerment_allowing_bigotry_auditing_the_moderation_of_hate_speech_on.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](../../ACL2026/social_computing/toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)

</div>

<!-- RELATED:END -->
