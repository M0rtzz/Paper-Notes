---
title: >-
  [论文解读] SlideTailor: Personalized Presentation Slide Generation for Scientific Papers
description: >-
  [AAAI 2026][推荐系统][个性化幻灯片生成] 定义了偏好引导的论文到幻灯片生成新任务，提出 SlideTailor 框架：从用户提供的论文-幻灯片样例对中蒸馏内容偏好、从 .pptx 模板蒸馏美学偏好，通过 chain-of-speech 机制将幻灯片内容与预期口述叙事对齐，在自建 PSP 基准上以 75.8% 的综合得分和 81.63% 的人评胜率显著超越现有方法。
tags:
  - "AAAI 2026"
  - "推荐系统"
  - "个性化幻灯片生成"
  - "偏好蒸馏"
  - "chain-of-speech"
  - "Agent框架"
  - "学术论文演示"
---

# SlideTailor: Personalized Presentation Slide Generation for Scientific Papers

**会议**: AAAI 2026  
**arXiv**: [2512.20292](https://arxiv.org/abs/2512.20292)  
**代码**: [SlideTailor](https://github.com/nusnlp/SlideTailor)  
**领域**: 文档到演示文稿生成  
**关键词**: 个性化幻灯片生成, 偏好蒸馏, chain-of-speech, Agent框架, 学术论文演示

## 一句话总结

定义了偏好引导的论文到幻灯片生成新任务，提出 SlideTailor 框架：从用户提供的论文-幻灯片样例对中蒸馏内容偏好、从 .pptx 模板蒸馏美学偏好，通过 chain-of-speech 机制将幻灯片内容与预期口述叙事对齐，在自建 PSP 基准上以 75.8% 的综合得分和 81.63% 的人评胜率显著超越现有方法。

## 研究背景与动机

**领域现状**：自动演示文稿生成是活跃的研究方向。已有方法（DOC2PPT、PPTAgent）开始整合文本和视觉元素进行多模态演示生成，取得一定进展。

**现有痛点**：
(1) **忽视用户主观性**——现有方法将幻灯片生成视为直接的文档到幻灯片转换，无法满足不同用户在叙事结构、重点选择、视觉风格等方面的个性化需求；
(2) **偏好表达困难**——要求用户用详细文字描述偏好不自然且负担重；Persona-Aware-D2S 仅支持 4 种固定偏好类别，无法覆盖真实世界的多样化需求；
(3) **内容与演讲脱节**——已有方法未考虑幻灯片内容与口头叙述之间的对齐，导致生成的幻灯片难以直接用于演讲。

**本文切入角度**：用户只需提供一个论文-幻灯片样例对（编码内容偏好）和一个 .pptx 模板（编码美学偏好），系统自动蒸馏隐式偏好并生成个性化幻灯片。

## 方法详解

### 整体框架

SlideTailor 采用模仿人类制作幻灯片流程的三阶段 Agent 框架：
(1) **隐式偏好蒸馏**：从样例对和模板中提取结构化偏好 $P = P_C \cup P_A$
(2) **偏好引导的幻灯片规划**：包含条件论文重组、chain-of-speech 大纲设计和模板选择
(3) **幻灯片实现**：通过布局感知编辑和代码执行生成可编辑 .pptx 文件

### 关键设计

1. **双分支隐式偏好蒸馏**

    - **功能**：从未标注的用户输入中提取结构化、可解释的偏好表示
    - **内容偏好蒸馏**：将 $f_{content}: D_{ref} \to S_{ref}$ 建模为隐函数，利用 LLM（GPT-4.1）推断内容的选择、强调、省略和重排方式，输出结构化偏好 $P_C$（叙事流、章节级别的详细度/重点/格式偏好）
    - **美学偏好蒸馏**：用 VLM 推断模板中各幻灯片级和元素级组件的功能角色，结合 .pptx 原始元数据（位置/尺寸），输出布局 schema $P_A$
    - **设计动机**：将两类偏好解耦使系统具备模块化灵活性——任意美学模板可与任意内容偏好组合

2. **Chain-of-Speech 机制**

    - **功能**：在设计幻灯片大纲时同步起草演讲稿，使幻灯片内容与预期口述叙事对齐
    - **核心思路**：受人类演讲者在制作幻灯片时同步排练演讲的启发，系统在规划每页幻灯片时会模拟口述叙事。这既让幻灯片内容更清晰连贯，又自然产出了可用于下游视频演示的演讲脚本
    - **设计动机**：好的幻灯片不仅是信息展示，更是演讲的视觉辅助。演讲脚本的同步生成确保了内容选择服务于口头传达的需要

3. **模板感知布局选择与编辑**

    - **功能**：根据幻灯片语义内容为每页匹配最合适的模板布局，并通过代码 Agent 生成可编辑 .pptx
    - **核心思路**：基于美学偏好 schema $P_A$ 进行逐页匹配（如内容密集页选文本布局、图表页选混合布局），然后由布局感知 Agent 映射内容到具体元素（标题框、文本框、图片占位符），最后代码 Agent 生成 Python 脚本直接编辑 .pptx
    - **设计动机**：保留原始模板的布局和主题，同时生成完全可编辑的标准格式文件，便于用户后续修改

### 下游应用

结合 chain-of-speech 产出的演讲脚本，可通过零样本 TTS 系统（MegaTTS 3 等）合成用户声音的个性化旁白，结合视觉幻灯片自动生成演示视频，甚至可整合音频驱动数字人头像进一步增强沉浸感。

## 实验关键数据

### 主实验——PSP 基准对比

| 方法 | Coverage↑ | Flow↑ | Content Structure↑ | Aesthetic↑ | Overall↑ |
|------|-----------|-------|-------------------|-----------|---------|
| ChatGPT | 62.62 | 56.84 | 61.60 | 80.80 | 62.86 |
| AutoPresent (GPT-4.1) | 72.84 | 59.58 | 49.60 | 22.40 | 48.78 |
| PPTAgent (GPT-4.1) | 64.41 | 54.24 | 57.60 | 97.20 | 67.30 |
| SlideTailor (Qwen2.5) | 70.19 | 62.16 | 68.41 | 92.80 | 69.21 |
| **SlideTailor (GPT-4.1)** | **74.47** | **66.65** | **72.80** | **98.00** | **75.80** |

### 消融实验

| 配置 | Coverage | Flow | Content Structure | Content | Overall |
|------|----------|------|-------------------|---------|---------|
| 去除内容偏好 | 65.80 (-9.0) | 56.83 (-11.6) | 54.67 (-11.3) | 65.73 | 68.61 |
| 去除 chain-of-speech | 73.60 | 63.99 | 66.00 | 47.33 (-19.1) | 69.91 |
| **完整系统** | **74.82** | **68.38** | **66.00** | **66.40** | **74.31** |

### 人类评估

4 名研究生评估 30 个案例（每案 2 人），SlideTailor vs PPTAgent 的胜率为 **81.63%**。人评与 MLLM 评估的 Pearson 相关系数平均 0.64。

### 关键发现

- 没有方法能达到 80% 以上综合分，说明偏好引导的幻灯片生成仍是开放难题
- 去除内容偏好蒸馏后 Coverage/Flow/Content Structure 均下降约 10%，验证了偏好建模的核心价值
- 去除 chain-of-speech 后内容质量暴降 19.1%（66.4%→47.3%），说明演讲叙事对齐对内容质量至关重要
- 开源 Qwen2.5 版本综合 69.21% 也超越所有基线，展示了框架的跨模型适应性
- 单次 10 页生成成本：GPT 版 $0.665，Qwen 版仅 $0.016

## 亮点与洞察

1. **任务定义有价值**：偏好引导的幻灯片生成抓住了演示文稿制作的核心痛点——主观性
2. **隐式偏好蒸馏思路巧妙**：用户无需写描述，只需提供自然的样例对，系统自动挖掘偏好
3. **Chain-of-speech 一举两得**：既提升幻灯片质量又自动产出演讲脚本，开启视频演示等下游应用
4. **PSP 基准设计周到**：200 论文 × 50 样例对 × 10 模板 = 10 万种组合，覆盖 AI/医学/化学等多领域

## 局限与展望

1. 基准仅限学术论文，未涵盖商务报告、教育材料等领域
2. 纯 zero-shot 框架，端到端多模态训练可能进一步提升效果
3. MLLM 评分器存在自偏差（如 GPT 评 GPT 生成的结果偏高），评估可靠性有待改进
4. 模板匹配为启发式，复杂布局（多图多表）场景下可能不够精确

## 相关工作与启发

- 隐式偏好蒸馏的范式可推广到其他个性化内容生成场景（如个性化摘要、自适应报告生成）
- Chain-of-speech 的思想可迁移到教育领域（如自动生成带讲解的教学课件）
- Agent 驱动的多阶段生成范式（蒸馏→规划→实现）为复杂文档处理提供了通用模板

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐⭐：新任务定义，偏好蒸馏+chain-of-speech 双重创新
- **实验充分度** ⭐⭐⭐⭐：定量+消融+人评齐全，但案例数相对有限
- **写作质量** ⭐⭐⭐⭐⭐：问题动机清晰，系统设计层层递进
- **价值** ⭐⭐⭐⭐：为个性化幻灯片生成开辟新方向，数据集和评估体系对后续研究有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation](../../ACL2026/recommender/clusterrag_cluster-based_collaborative_filtering_for_personalized_retrieval-augm.md)
- [\[AAAI 2026\] AutoPP: Towards Automated Product Poster Generation and Optimization](autopp_towards_automated_product_poster_generation_and_optimization.md)
- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[NeurIPS 2025\] NeurIPS Should Lead Scientific Consensus on AI Policy](../../NeurIPS2025/recommender/neurips_should_lead_scientific_consensus_on_ai_policy.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](../../ACL2026/recommender/intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)

</div>

<!-- RELATED:END -->
