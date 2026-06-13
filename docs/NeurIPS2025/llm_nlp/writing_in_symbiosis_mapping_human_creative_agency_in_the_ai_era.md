---
title: >-
  [论文解读] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era
description: >-
  [NeurIPS 2025][LLM 其他][human-AI coevolution] 通过对 5 万+文档的纵向语料分析，提出"双轨演化"假说——LLM 时代人类写作在主题上趋同、风格上结构性分化，并发现三种作者适应策略原型（Adopters/Resistors/Pragmatists）。 问题背景 大型语言模型的普及引…
tags:
  - "NeurIPS 2025"
  - "LLM 其他"
  - "human-AI coevolution"
  - "creative writing"
  - "stylometric analysis"
  - "authorial archetypes"
  - "LLM influence detection"
---

# Writing in Symbiosis: Mapping Human Creative Agency in the AI Era

**会议**: NeurIPS 2025  
**arXiv**: [2512.13697](https://arxiv.org/abs/2512.13697)  
**代码**: 暂未开源（论文称 code available on request）  
**领域**: LLM/NLP  
**关键词**: human-AI coevolution, creative writing, stylometric analysis, authorial archetypes, LLM influence detection

## 一句话总结

通过对 5 万+文档的纵向语料分析，提出"双轨演化"假说——LLM 时代人类写作在主题上趋同、风格上结构性分化，并发现三种作者适应策略原型（Adopters/Resistors/Pragmatists）。

## 研究背景与动机

### 问题背景
大型语言模型的普及引发了一个根本问题：当"创作"变成人与机器的协作行为时，人类创作的独特性是否正在消失？

### 现有研究的局限

**同质化叙事太简单**：现有工作主要关注 LLM 对互联网/学术文本的"风格同质化"，将其框定为 AI 对人类的单向影响

**缺少个体分辨率**：大多研究只看总体趋势，忽略了作者个体层面的差异化适应策略

**主题与风格混淆**：已有研究常把主题变化和风格变化混为一谈，缺乏有效分离

**缺乏纵向控制**：没有对同一作者在 LLM 出现前后的系统对比

### 核心假说

**双轨演化假说（Dual-Track Evolution）**：
- **轨道一**：AI 相关主题的普遍趋同（thematic convergence）
- **轨道二**：在风格层面出现结构性分化（stylistic differentiation），而非简单趋同

## 方法详解

### 整体框架

研究设计分为三个阶段：语料构建 → 特征工程 → 多视角分析。

**时间分界线**：2022 年 11 月 30 日（ChatGPT 发布日）将时间分为 pre-LLM 和 post-LLM 两个时期。

### 语料构建

- **规模**：50,000+ 文档（源自 823k+ 消息和论文）
- **时间跨度**：2021 年 1 月 - 2024 年 12 月
- **两种体裁**：
    - **正式文本**：arXiv 计算机科学预印本（permissive license）
    - **非正式文本**：Discord Unveiled 数据集（CC BY 4.0），公开服务器匿名通信
- **AI 参考语料**：ShareGPT-90k (Apache-2.0) + Dolly-15k (CC BY-SA 3.0)
- **抽样策略**：
    - Discord：主题控制的分层抽样，平衡 pre/post LLM 各类别配额
    - arXiv：按月抽样保证时间连续性

### 关键设计：Perplexity-Gap 分析

核心创新是用**困惑度差异（Perplexity Gap）**来量化风格的时间演变：

$$\Delta_{ppl} = \frac{-\ln(p_{GPT2}(x))}{|x|_{chars}} - \frac{-\ln(p_{Llama}(x))}{|x|_{chars}}$$

- **Pre-LLM Judge**：GPT-2 Medium (355M)，仅在 2022 年前数据上训练（847M tokens，A100 约 41.7 小时）
- **Current Baseline**：Llama-3-8B-base（冻结的现代模型）
- **含义**：对当前模型"容易"但对旧模型"难"的文本，具有 LLM 时代的语言特征

**AI-likeness 指数**（作者内 z-score 标准化）：

$$AI_{likeness} = \frac{\Delta_{ppl} - \mu_{author}}{\sigma_{author}}$$

### Δ-特征向量

每位作者的风格变化用 7 维标准化向量表示：

| 特征 | 含义 |
|------|------|
| $\Delta_{ppl}$ | 困惑度差异 |
| $\Delta_{TTR}$ | 词汇多样性（type-token ratio） |
| $\Delta_{FKGL}$ | 可读性等级（Flesch-Kincaid） |
| $\Delta_{passive\%}$ | 被动语态比例 |
| $\Delta_{1p\%}$ | 第一人称代词频率 |
| $\Delta_{punct}$ | 标点密度 |
| $\Delta_{sent\_len}$ | 平均句长 |

所有特征均在作者内部、跨时间边界做 z-score 标准化。

### 统计控制

固定效应模型：

$$y_{it} = \beta \cdot PostLLM_t + \gamma \cdot len_{it} + \delta_c + \alpha_i + \epsilon_{it}$$

其中 $\alpha_i$ 为作者固定效应，$\delta_c$ 为服务器类别固定效应，使用 HC3 稳健标准误 + Holm-Bonferroni 校正。

### 聚类方法

对 Δ-特征向量使用 **HDBSCAN**（min_cluster_size=15, min_samples=5, metric=euclidean），选择 HDBSCAN 的理由：
- 能识别不同形状和密度的簇
- 自动处理噪声点（不强制分类）
- 让真实的作者原型从数据中自然涌现

## 实验关键数据

### 主实验：三种作者原型

对 2,100 位社交语料作者的聚类分析揭示三种行为模式：

| 原型 | 人数 | 占比 | 核心特征 |
|------|------|------|----------|
| **Resistors（抵抗者）** | 442 | 21% | 低/负困惑度差异，维持 pre-LLM 语言复杂度 |
| **Adopters（采纳者）** | 370 | 18% | 最高困惑度差异，写作向 LLM 风格靠近 |
| **Pragmatists（实用主义者）** | 866 | 41% | 中等风格变化 + 高 AI 主题参与度 |

**聚类质量指标**：
- Silhouette: 0.426 (95% CI: 0.419-0.433)
- ARI 鲁棒性: 0.891 (95% CI: 0.884-0.898)
- Bootstrap 一致性: 89%
- 预测 AUC: 0.813

### 宏观趋势验证

| 发现 | 数据 |
|------|------|
| AI 主题趋同 | 两种体裁均在 Nov 2022 后显著增加 AI 相关内容 |
| 主题结构断点 | Q1 2023（PELT 算法检测） |
| 风格复杂度断点 | Q2 2023（滞后于主题变化） |
| 社交语料困惑度差异增长 | +23%（2023 初） |
| 正式语料困惑度差异增长 | +15%（2023 初） |

### 消融实验：风格适应的动态弧线

最重要的发现是**两阶段模式**：

| 阶段 | 时间 | 社交语料 | 正式语料 |
|------|------|----------|----------|
| 趋同阶段 | 2023 初 | 困惑度差异 ↑23% | ↑15% |
| 回避阶段 | 2023 底-2024 | 困惑度差异 ↓18%（从峰值） | ↓12% |

这表明当 AI 风格特征被"污名化"后，作者会主动回避，特别是在正式场合。

### 关键发现

- 交叉验证准确率: 89.3% (95% CI: 87.1-91.5)
- Held-out arXiv 数据: 89.1% (86.8-91.4)
- 空模型对比: silhouette 0.31 vs 0.43（p<0.001）
- 时间边界鲁棒性: 84% 原型分配一致性（极端变化 91%）
- AI-likeness 在控制 FKGL/TTR/句长后仍显著: 偏相关 r=0.34, p<0.001

## 亮点与洞察

1. **双轨演化框架**是一个优雅的理论贡献——将看似矛盾的"趋同"和"分化"统一在一个模型中
2. **Perplexity-Gap 方法**巧妙利用了不同时代语言模型的能力差异来量化风格变化，避免了循环论证
3. **两阶段动态弧线**（先趋同后回避）揭示了社会压力对写作风格的调节作用——AI 检测意识和学术审稿压力导致风格回撤
4. **对 AI 检测领域的重要启示**：三种原型意味着简单的"人类 vs 机器"二分检测框架不足——Adopter 的文本统计上比 Resistor 更接近 AI 输出
5. 多数作者（Resistors + Pragmatists，合计 62%）维持非 AI 风格签名，说明人类独特表达仍是被珍视且主动保护的

## 局限与展望

1. **观察性而非因果性**：无法确证是 AI 工具使用导致风格变化（可能有其他混杂因素）
2. **仅限英语**：可能偏差于英语写作社区的适应模式，不同语言/文化可能有不同反应
3. **缺少直接人类验证**：原型是统计构建，未通过参与者研究验证（如问卷调查作者实际 AI 使用行为）
4. **社交语料偏差**：Discord 用户不能代表所有互联网用户
5. **可能被滥用**：原型框架可能被用于作者身份监控或不公平的写作风格歧视
6. **改进方向**：可扩展到多语言、加入因果推断设计、结合用户调研

## 相关工作与启发

- **与 Geng & Trotta (2025) 的关系**：后者关注学术写作中的人-LLM 共演化，本文扩展到社交语料并提出个体级分析
- **与 AI 检测研究的张力**：检测器假设人/机二分，但本文说明这一假设在 coevolution 情境下已不适用
- **Scaffolded collaboration (Dhillon et al. 2024)**：支架式协作的不同策略与本文发现的原型一致
- **语言简化趋势 (Di Marco et al. 2024)**：社交媒体上更广泛的语言简化趋势可能混淆了 AI 影响的判断

**启发**：在设计 AI 写作工具时，应考虑支持不同原型用户的需求——Resistors 需要保持独特性的工具，Adopters 需要深度协作工具，Pragmatists 需要内容探索但风格保护的工具。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "双轨演化"假说和个体级原型框架在该领域较新颖，Perplexity-Gap 方法有创意
- **实验充分度**: ⭐⭐⭐⭐ — 大规模语料、多种统计控制、聚类鲁棒性验证充分；但缺少人类验证和多语言实验
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，叙事逻辑通顺，从宏观到微观层层递进
- **价值**: ⭐⭐⭐⭐ — 对 AI 时代创作研究、AI 检测、人机交互均有重要启示；但实际应用路径尚需探索

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era](towards_implicit_aggregation_robust_image_representation_for_place_recognition_i.md)
- [\[ACL 2025\] Towards Enhanced Immersion and Agency for LLM-based Interactive Drama](../../ACL2025/llm_nlp/towards_enhanced_immersion_and_agency_for_llm-based_interactive_drama.md)
- [\[ACL 2025\] EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents](../../ACL2025/llm_nlp/escapebench_creative_agent.md)
- [\[ACL 2025\] Educators' Perceptions of Large Language Models as Tutors: Comparing Human and AI Tutors in a Blind Text-only Setting](../../ACL2025/llm_nlp/educators_perceptions_of_large_language_models_as_tutors_comparing_human_and_ai_.md)
- [\[ACL 2025\] Mapping 1,000+ Language Models via the Log-Likelihood Vector](../../ACL2025/llm_nlp/mapping_1000_models_loglikelihood.md)

</div>

<!-- RELATED:END -->
