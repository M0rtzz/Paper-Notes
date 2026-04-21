---
title: >-
  [论文解读] AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations
description: >-
  [ICLR 2026][人体理解][科学插图生成] 提出AutoFigure——第一个基于"推理渲染"范式的Agent框架，通过解耦结构布局规划和美学渲染两阶段自动从长科学文本生成达到出版质量的科学插图，配合首个大规模基准FigureBench（3,300对）进行系统评估，66.7%的生成结果被原作者认为可用于camera-ready版本。
tags:
  - ICLR 2026
  - 人体理解
  - 科学插图生成
  - 多智能体框架
  - 长文本理解
  - FigureBench
  - VLM评估
---

# AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations

**会议**: ICLR 2026  
**arXiv**: [2602.03828](https://arxiv.org/abs/2602.03828)  
**代码**: [https://github.com/ResearAI/AutoFigure](https://github.com/ResearAI/AutoFigure)  
**领域**: Human Understanding (AI for Science / Scientific Communication)  
**关键词**: 科学插图生成, 多智能体框架, 长文本理解, FigureBench, VLM评估

## 一句话总结
提出AutoFigure——第一个基于"推理渲染"范式的Agent框架，通过解耦结构布局规划和美学渲染两阶段自动从长科学文本生成达到出版质量的科学插图，配合首个大规模基准FigureBench（3,300对）进行系统评估，66.7%的生成结果被原作者认为可用于camera-ready版本。

## 研究背景与动机
高质量科学插图对传达复杂科学概念至关重要，读者可以在几分钟内快速理解论文核心思想。然而，手工创建通常需要数天时间，要求创作者同时具备领域知识和专业设计技能。

**现有工作的两大局限**：

**基准层面**：Paper2Fig100k、ACL-Fig、SciCap+等现有数据集主要关注从图片标题或短文本片段重建图形，而非从长文本（平均>10k token）的方法论中提炼核心结构。缺少真正面向"长上下文科学插图设计"任务的基准。

**方法层面**：
   - PosterAgent、PPTAgent等系统仅擅长"理解、提取和重组"已有的多模态内容，不具备从原始文本**生成**视觉内容的能力
   - AutoTikZ等基于代码的方法侧重结构和几何正确性，但美学表现力差
   - DALL-E / GPT-Image等端到端T2I模型能生成美观图片，但无法保持**结构忠实度**——长科学文本中的逻辑关系和层次结构经常丢失

**核心矛盾**：结构准确性 vs 视觉美感的trade-off。代码方法结构好但不美观，生成模型美观但结构混乱。

**AutoFigure的切入角度**：解耦这两个需求——先用LLM做结构推理和布局规划，再用生成模型做美学渲染。

## 方法详解

### 整体框架
AutoFigure采用"推理渲染"（Reasoned Rendering）范式，分为两个阶段：
- 输入：长科学文本$T$（论文/综述/博客/教材）
- Stage I：语义解析 + 布局规划 → 结构化符号布局$(S_{\text{final}}, A_{\text{final}})$
- Stage II：美学渲染 + 文本后处理 → 出版质量插图$I_{\text{final}}$

### 关键设计

1. **Stage I - 概念提取与符号构建**：

    - **概念提取Agent**：从输入文本$T$中提取方法论摘要$T_{\text{method}}$以及实体和关系集合
    - 将结构序列化为标记语言（SVG/HTML）的符号布局$S_0$和风格描述$A_0$
    - $S_0$编码一个有向图$G_0 = (V_0, E_0)$，表示概念之间的逻辑关系

2. **Stage I - 批评-细化循环**（核心"思考"过程）：

    - 模拟AI"设计师"与AI"评论家"之间的对话
    - 每轮迭代中：评论家$\Phi_{\text{critic}}$评估当前最佳布局，生成反馈$F^{(i)}_{\text{best}}$
    - 生成器$\Phi_{\text{gen}}$根据反馈重新解读方法论文本，生成候选布局
    - 与当前最佳比较（通过分数$q$），若更优则替换
    - 循环至收敛或达最大迭代次数（实验中约5轮），输出最终布局
    - 本质上是一种**test-time compute scaling**：更多迭代带来更好的布局质量

3. **Stage II - 风格引导美学渲染**：

    - 转换函数$\Phi_{\text{prompt}}$将$(S_{\text{final}}, A_{\text{final}})$转化为exhaustive的text-to-image prompt
    - 配合从$S_{\text{final}}$导出的结构图，输入多模态生成模型（如GPT-Image / Nano-Banana）渲染高保真图像$I_{\text{polished}}$

4. **Stage II - "擦除-纠正"文本精确化**：

    - 问题：T2I模型渲染的文本常模糊不清或拼写错误
    - 解决方案：
      1. 非LLM擦除器$\Phi_{\text{erase}}$：移除所有文字像素→干净背景$I_{\text{erased}}$
      2. OCR引擎$\Phi_{\text{ocr}}$：提取初步字符串和边界框
      3. 多模态验证器$\Phi_{\text{verify}}$：将OCR结果与$S_{\text{final}}$中的ground-truth标签对齐校正
      4. 在$I_{\text{erased}}$上叠加矢量文字层→最终图$I_{\text{final}}$

### FigureBench基准
- **规模**：3,300个高质量科学文本-插图对
- **来源**：论文(3,200) + 综述(40) + 博客(20) + 教材(40)
- **测试集**：300个（200篇从Research-14K随机抽样经GPT-5筛选和双人标注，Cohen's $\kappa = 0.91$；100个从综述/博客/教材中手动策）
- **开发集**：3,000个（用微调后的VLM自动筛选器从Research-14K中构建）
- **评估协议**：VLM-as-a-judge（参考评分 + 盲测对比），涵盖视觉设计、沟通效果、内容忠实度三大维度八个子指标

## 实验关键数据

### 主实验（自动评估，Paper类别）

| 方法 | Overall | Win-Rate | 美学 | 准确性 |
|------|---------|----------|------|--------|
| AutoFigure | **7.03** | **53.0%** | **7.28** | 6.96 |
| HTML-Code | 6.35 | 11.0% | 5.90 | **6.99** |
| SVG-Code | 5.49 | 31.0% | 5.00 | 6.15 |
| GPT-Image | 3.47 | 7.0% | 4.24 | 4.77 |
| Diagram Agent | 2.12 | 0.0% | 2.25 | 2.11 |

### 人类专家评估（10位一作评审自己论文的生成结果）

| 指标 | 数值 | 说明 |
|------|------|------|
| Win-Rate（vs其他AI） | 83.3% | 仅次于人类原图96.8% |
| 出版意愿率 | **66.7%** | 愿意在camera-ready中使用 |
| 准确性评分 | ~3.5/5 | 在合理范围内 |
| 美学评分 | ~4/5 | 接近人类水平 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 迭代轮数（0→5） | Overall从6.28→7.14 | 批评-细化循环的test-time scaling效果明显 |
| 推理模型选择 | Claude-4.1-Opus > GPT-5 > Gemini-2.5-Pro | 更强推理模型 → 更优布局 |
| 中间格式 | SVG(8.98) > HTML(8.85) >> PPT(6.12) | SVG/HTML可一次性生成完整文件 |
| 文本细化模块 | +0.04 Overall（+0.10美学） | 对出版质量至关重要 |
| 开源模型 | Qwen3-VL-235B达到Overall 7.08 | 超越多个商业模型，接近GPT-5 |

### 关键发现
- AutoFigure在Blog(7.60)、Survey(6.99)、Textbook(**8.00**)、Paper(7.03)四类文档上全面领先
- Textbook类别Win-Rate达97.5%，说明教学性质的标准化图表最容易自动化
- Paper类别Win-Rate相对较低(53.0%)，因为论文插图通常需要定制化设计，无先验视觉模板
- TikZ代码方法Overall<1.5，说明端到端代码生成范式的根本局限——LLM在序列化高维结构时认知负荷过大
- **人机相关性验证**：VLM与人类评分的Pearson相关系数 r=0.659，Spearman $\rho=0.593$，排名误差<1

## 亮点与洞察
- **"推理渲染"解耦范式**：将科学插图生成分解为"结构推理"+"美学渲染"的思路非常精妙，各模块可独立优化
- **批评-细化循环 = test-time scaling**：更多迭代显著提升质量，这与LLM推理中的scaling规律一致
- **"擦除-纠正"策略**：巧妙地解决了T2I模型文本渲染差的痛点，通过OCR+矢量叠加保证文字准确性
- **实践价值极高**：66.7%的出版意愿率意味着AutoFigure已经接近实用阈值
- **开源模型潜力**：Qwen3-VL-235B达到超越多数商业模型的水平，降低部署门槛

## 局限与展望
- **文本渲染精度仍有瓶颈**：小字号/密集布局/复杂背景下仍有字符级错误（如"ravity"缺"g"）
- **Paper类别表现相对较弱**：论文插图的层次复杂度高（宏观流程+微观子步骤+细节实体），且需要定制化设计
- **"具象化"倾向**：当源文本描述不充分时，系统可能生成视觉上合理但内容不精确的结构
- 仅面向CS领域，未验证在生物学、化学等具有独特视觉规范的学科中的效果
- 端到端延迟约9-17分钟，对于实时交互场景仍偏长

## 相关工作与启发
- **PosterAgent / PPTAgent**：海报/PPT生成系统，但仅擅长重排现有内容，不能从文本从头生成
- **AutoTikZ / TikZero**：基于LaTeX TikZ的代码生成方法，结构准确但美学差
- **AI Scientist / Zochi**：AI自主科学发现系统，视觉表达能力是其关键瓶颈
- **Research-14K / CycleResearcher**：为FigureBench提供源数据的科学论文数据集
- 启发：随着AI Scientist的崛起，"让AI表达自己的发现"成为关键需求。AutoFigure补上了从"理解科学"到"展示科学"的最后一环。未来方向可能包括动态/交互式科学图表生成

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （开创性任务定义 + 首个大规模基准 + 新颖范式）
- 实验充分度: ⭐⭐⭐⭐⭐ （自动评估 + 人类专家评估 + 丰富消融 + 开源模型验证）
- 写作质量: ⭐⭐⭐⭐⭐ （图表精美，叙事完整，附录极其详尽）
- 价值: ⭐⭐⭐⭐⭐ （直击实际痛点，实用价值极高，对AI for Science方向影响深远）

<!-- RELATED:START -->

## 相关论文

- [Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation](think-while-generating_on-the-fly_reasoning_for_personalized_long-form_generatio.md)
- [Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals](inverse_virtual_try-on_generating_multi-category_product-style_images_from_cloth.md)
- [MagicArticulate: Make Your 3D Models Articulation-Ready](../../CVPR2025/human_understanding/magicarticulate_make_your_3d_models_articulation-ready.md)
- [Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining](../../ICCV2025/human_understanding/iris_breaking_gui_complexity_with_adaptive_focus_and_self-refining.md)

<!-- RELATED:END -->
