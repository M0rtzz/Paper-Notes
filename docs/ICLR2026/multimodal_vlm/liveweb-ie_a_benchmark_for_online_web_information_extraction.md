---
title: >-
  [论文解读] LiveWeb-IE: A Benchmark For Online Web Information Extraction
description: >-
  [ICLR 2026][多模态][网页信息抽取] 提出首个面向在线网页的信息抽取（WIE）基准LiveWeb-IE，覆盖文本/图片/超链接等多类数据抽取，并设计Visual Grounding Scraper（VGS）框架，通过模拟人类认知过程——视觉扫描定位区域→精确定位元素→生成XPath——在动态网页上实现鲁棒的信息抽取。
tags:
  - ICLR 2026
  - 多模态
  - 网页信息抽取
  - 在线评估
  - 视觉定位
  - XPath生成
  - 多模态Agent
---

# LiveWeb-IE: A Benchmark For Online Web Information Extraction

**会议**: ICLR 2026  
**arXiv**: [2603.13773](https://arxiv.org/abs/2603.13773)  
**代码**: [GitHub](https://github.com/sbY99/LiveWeb-IE)  
**领域**: 多模态VLM  
**关键词**: 网页信息抽取, 在线评估, 视觉定位, XPath生成, 多模态Agent

## 一句话总结

提出首个面向在线网页的信息抽取（WIE）基准LiveWeb-IE，覆盖文本/图片/超链接等多类数据抽取，并设计Visual Grounding Scraper（VGS）框架，通过模拟人类认知过程——视觉扫描定位区域→精确定位元素→生成XPath——在动态网页上实现鲁棒的信息抽取。

## 研究背景与动机

网页信息抽取（WIE）是从网页中自动提取结构化数据的任务。现有WIE基准（如SWDE、WEIR、PLAtE等）全部基于**静态HTML快照**构建，存在根本性缺陷：

**时效失配**：网页布局和结构随时间不断变化，静态快照无法反映当前网页状态

**性能不可靠**：LLM based wrapper方法在结构演变后的同一网站上F1平均下降超过15%

**数据类型单一**：现有基准仅关注文本抽取，忽略了图片和超链接抽取需求

**复杂度维度缺失**：没有系统化的任务复杂度分层

此外，现有WIE方法过度依赖HTML解析。随着网页结构日趋复杂，HTML的冗余性使得从中准确定位信息越来越困难。

## 方法详解

### 整体框架

本文包含两个贡献：（1）**LiveWeb-IE基准**：面向在线网页的WIE评估基准；（2）**VGS方法**：Visual Grounding Scraper，模拟人类认知过程的多阶段Agent框架。

### 关键设计

#### 1. LiveWeb-IE基准设计

**四大特性**：
- **在线评估**：要求WIE系统在评估时直接访问目标URL，处理当前时刻的网页DOM结构
- **多样可靠的网站**：15个获得授权的网站，横跨8个领域，经过robots.txt检查、使用条款审核和管理员直接授权
- **多类数据抽取**：涵盖文本、图片、超链接三类数据
- **多维任务复杂度**：基于属性数量和值的基数定义4个级别

**四种任务类型**：
- **Type I**：单属性、单值（如"这个教授的邮箱是什么"）
- **Type II**：多属性、单值（如"这个球员的身高和体重"）
- **Type III**：单属性、列表值（如"这个页面上所有论文标题"）
- **Type IV**：多属性、列表值（如"所有产品的名称和价格"）

**数据构建流程**：网站选择 → 页面分组（按布局聚类） → 数据标注 → 人工交叉验证。最终包含342个查询、97个唯一属性、46个页面组。

**内容稳定性设计**：查询针对事实性信息（如2022年世界杯决赛比分），即使网页布局变化答案本身不变，保证基准长期有效。

#### 2. VGS（Visual Grounding Scraper）框架

VGS模拟人类在网页上查找信息的认知过程，通过四个阶段逐步缩小观察空间：

**阶段1：属性识别**。用LLM将自然语言查询分解为结构化的目标属性集合：
$$\hat{\mathcal{A}} = \text{LLM}(I_a, Q)$$

**阶段2：视觉定位**。将网页渲染为垂直区域序列（固定宽高的截图）。对每个属性，用VLM在区域序列中定位相关区域：
$$r'_i = \text{VLM}(I_g, \mathcal{R}, \hat{a}_i)$$
关键价值在于**大幅缩减观察空间**。

**阶段3：元素精确定位**。在定位到的区域内精确找到目标值位置。采用两步策略：
1. 生成候选边界框（文本属性通过VLM扫描，非文本属性通过HTML标签定位）
2. 使用Set-of-Mark Prompting覆盖带编号标记，让VLM选择正确元素子集：
$$\mathcal{B}_i^* = \text{VLM}(I_p, r_i^*, \hat{a}_i)$$

**阶段4：XPath合成**。基于精确定位的边界框找到对应DOM元素，提取局部HTML片段（邻近距离 $d$ 内），VLM结合视觉和结构信息生成可复用XPath：
$$x_i = \text{VLM}(I_x, \mathcal{H}_i, \hat{r}_i, \hat{a}_i)$$
最终XPath集合构成可复用的wrapper。

### 损失函数 / 训练策略

VGS是**无需训练**的Agent框架，完全基于预训练LLM/VLM的推理能力。评估指标采用Precision、Recall和F1。

## 实验关键数据

### 主实验

**LiveWeb-IE上的Overall F1对比**：

| 骨干模型 | 方法 | Type I F1 | Type II F1 | Type III F1 | Type IV F1 | Overall F1 |
|----------|------|-----------|------------|-------------|------------|------------|
| GPT-4o | COT | 47.54 | 40.84 | 8.15 | 7.24 | 24.60 |
| GPT-4o | AutoScraper | 55.22 | 42.65 | 9.10 | 6.92 | 26.76 |
| GPT-4o | **VGS** | **65.87** | **46.35** | **45.38** | **41.50** | **48.58** |
| Gemini-2.5-Flash | **VGS** | **49.02** | **44.82** | **42.92** | **38.13** | **43.44** |

**开源模型对比（Overall F1）**：

| 骨干模型 | COT | AutoScraper | VGS |
|----------|-----|-------------|-----|
| Qwen-2.5-7B | 11.67 | 16.04 | **21.74** |
| Qwen-2.5-32B | 17.74 | 21.61 | **35.05** |
| Gemma-3-27B | 16.65 | 19.04 | **30.79** |

### 消融实验

VGS各阶段的贡献：
1. **去除视觉定位**：不先定位区域直接精确定位元素，性能显著下降
2. **去除元素精确定位**：跳过Set-of-Mark步骤，复杂类型退化明显
3. **使用HTML替代视觉信息**：Type III和Type IV的F1大幅下降

### 关键发现

1. **静态→在线的性能鸿沟**：LLM方法在结构演变后F1平均下降超过15%，证实在线评估必要性
2. **复杂度差距巨大**：VGS的最大优势在复杂类型——GPT-4o+VGS的Type III F1达45.38%，而COT仅8.15%
3. **视觉信息的关键作用**：纯HTML方法在复杂网页上失败，VGS通过视觉定位绕过HTML噪声
4. **开源vs闭源差距**：即便使用VGS，Qwen-2.5-32B (35.05%) 与GPT-4o (48.58%) 仍有显著差距
5. **Wrapper可复用性**：VGS生成的XPath具有跨同类页面的泛化能力

## 亮点与洞察

- **问题定义创新**：首次将WIE评估从离线搬到在线，通过内容稳定性设计解决标注持久性问题
- **认知启发的设计**：VGS四阶段流程完美模拟人类在网页上找信息的过程
- **视觉+结构双通道**：XPath生成巧妙结合视觉定位结果和局部HTML
- **多类数据覆盖**：将图片和超链接纳入WIE评估贴合实际需求

## 局限与展望

1. 基准规模有限：仅15个网站342个查询，更大规模扩展有价值
2. 内容稳定性假设：部分网站可能改版导致无法访问，需定期维护
3. VLM调用成本高：4个阶段每个都需VLM推理，大规模抽取效率待优化
4. XPath脆性：生成的XPath仍依赖DOM结构，网页大幅改版后可能失效
5. 动态内容处理不足：JavaScript动态渲染内容的处理未充分讨论

## 相关工作与启发

LiveWeb-IE与WebArena等网页Agent基准目标不同——后者关注多步任务完成，LiveWeb-IE关注单页面精确信息抽取。VGS的视觉定位思想与Set-of-Mark Prompting结合，展示了VLM在网页理解中的潜力，可拓展到网页自动化测试等应用。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 在线WIE基准是新颖且有实际价值的贡献
- **技术质量**: ⭐⭐⭐⭐ — VGS设计合理但技术创新点偏工程化
- **实验充分度**: ⭐⭐⭐⭐ — 多骨干模型对比充分，但消融可更系统
- **实用性**: ⭐⭐⭐⭐⭐ — 直接面向真实网页数据采集场景
- **写作质量**: ⭐⭐⭐⭐ — 基准设计动机和方法流程论述清晰
- **综合**: ⭐⭐⭐⭐ (8.0/10)

<!-- RELATED:START -->

## 相关论文

- [WebDS: An End-to-End Benchmark for Web-based Data Science](webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [Exploring LLMs for Scientific Information Extraction using the SciEx Framework](../../AAAI2026/multimodal_vlm/exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)
- [Relation-Rich Visual Document Generator for Visual Information Extraction](../../CVPR2025/multimodal_vlm/relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [LAION-C: An Out-of-Distribution Benchmark for Web-Scale Vision Models](../../ICML2025/multimodal_vlm/laion-c_an_out-of-distribution_benchmark_for_web-scale_vision_models.md)
- [Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation](small_drafts_big_verdict_information-intensive_visual_reasoning_via_speculation.md)

<!-- RELATED:END -->
