---
title: >-
  [论文解读] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language
description: >-
  [ACL 2026][LLM评测][抽象话] 本文将中文互联网亚文化语言"抽象话"引入 NLP 社区，构建首个评估基准 Mouse（含翻译、表征分类、意图识别、毒性检测、含义选择、完形填空六个任务），发现 SOTA LLM 在上下文语义理解上表现尚可但在其他任务上存在明显局限。
tags:
  - "ACL 2026"
  - "LLM评测"
  - "抽象话"
  - "互联网亚文化语言"
  - "LLM基准测试"
  - "中文网络用语"
  - "跨文化理解"
---

# Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language

**会议**: ACL 2026  
**arXiv**: [2604.15841](https://arxiv.org/abs/2604.15841)  
**代码**: [GitHub](https://github.com/csdq777/Mouse)  
**领域**: LLM评测  
**关键词**: 抽象话, 互联网亚文化语言, LLM基准测试, 中文网络用语, 跨文化理解

## 一句话总结
本文将中文互联网亚文化语言"抽象话"引入 NLP 社区，构建首个评估基准 Mouse（含翻译、表征分类、意图识别、毒性检测、含义选择、完形填空六个任务），发现 SOTA LLM 在上下文语义理解上表现尚可但在其他任务上存在明显局限。

## 研究背景与动机

**领域现状**：LLM 在标准 NLP 任务上表现出色，但对非标准语言变体（特别是非西方亚文化语言）的处理能力几乎未被探索。现有中文互联网语言研究主要局限于毒性检测和扰动语言检测。

**现有痛点**：（1）现有 LLM 和基准存在显著的西方中心偏差，中文亚文化语言被严重忽视；（2）抽象话作为中国青年最广泛使用的网络亚文化语言，融合了谐音替换、视觉符号类比、语义映射等复杂机制，远超表情包和网络流行语的复杂度；（3）此前研究将此类语言现象局限于负面维度（毒性），忽视了其进化后的中性甚至正面功能。

**核心矛盾**：抽象话通过多层映射（谐音→字形分解→语义隐喻→方言借用）创造了偏离标准中文的非标准语义空间，但 LLM 的预训练主要基于标准文本，导致在理解和处理这种亚文化编码时存在系统性能力缺陷。

**本文目标**：形式化定义抽象话的分类体系，构建评估基准 Mouse，系统评估 SOTA LLM 在抽象话上的能力边界。

**切入角度**：从符号表征学视角将抽象话分为谐音、视觉和语义三个核心维度，结合意图分类构建双维度分类体系。

**核心 idea**：首个抽象话 NLP 基准，通过六个任务全面评估 LLM 对亚文化语言的理解、翻译和使用能力。

## 方法详解

### 整体框架
构建包含 1,099 个抽象话评估实例（CXEI）的 Mouse 基准，每个实例包含原文、标准中文参考翻译、表征分类标签、意图标签和毒性标签。设计六个任务：翻译（TR）、表征分类（RC）、意图识别（IR）、毒性检测（TD）、含义选择（MS）和完形填空（CC）。

### 关键设计

1. **抽象话的三维表征分类体系**:

    - 功能：系统化地描述抽象话的语言学构成
    - 核心思路：谐音维度利用中文的语音冗余进行同音替换（如"主包"→"主播"）；视觉维度利用汉字的象形性和emoji的图像属性（如"彳亍口巴"→"行吧"通过部首分解）；语义维度通过直接符号字面化或方言借用进行含义映射（如"踩到皮"→"踩到香蕉皮"）。一条句子可同时含多个维度
    - 设计动机：此前分类基于起源（符号/谐音/方言/表情包）存在特征重叠，从符号表征学视角分类更系统且可操作

2. **双维度评估实例（CXEI）**:

    - 功能：提供全面的评估单元
    - 核心思路：每个 CXEI 包含五个属性——原文（混合emoji/汉字/拉丁字母的抽象话文本）、参考翻译（纯标准中文）、表征维度（谐音/视觉/语义）、意图（评论/情感表达/陈述/性暗示/玩梗/催促/群体认同等）、毒性（二元标签）
    - 设计动机：抽象话不仅是语言现象也是社会交互工具，需要同时评估语言解码能力和语用理解能力

3. **六任务评估设计**:

    - 功能：从不同维度全面评估 LLM 对抽象话的掌握程度
    - 核心思路：翻译（解码能力）→ 表征分类（语言学分析能力）→ 意图识别（语用理解）→ 毒性检测（安全识别）→ 含义选择（核心语义理解）→ 完形填空（抽象话使用能力）。从被动理解到主动使用形成能力梯度
    - 设计动机：单一任务无法全面评估——LLM 可能能理解上下文但无法翻译，能检测毒性但无法识别意图

### 损失函数 / 训练策略
本文是纯评估工作，不涉及模型训练。评估使用 LLM-as-Judge 做翻译质量评分，其他任务使用准确率/平衡准确率。

## 实验关键数据

### 主实验

| 任务 | SOTA LLM 表现 | 说明 |
|------|-------------|------|
| 含义选择 (MS) | 较好 | 上下文语义理解尚可 |
| 完形填空 (CC) | 较好 | 上下文适应性尚可 |
| 翻译 (TR) | 差 | 无法解码复杂谐音/视觉映射 |
| 表征分类 (RC) | 差 | 无法识别语言学构成 |
| 意图识别 (IR) | 中等 | 部分意图可从上下文推断 |
| 毒性检测 (TD) | 差 | 抽象话的毒性绕过传统过滤 |

### 消融实验

| 分析维度 | 发现 |
|---------|------|
| 谐音 vs 视觉 vs 语义 | 视觉维度最难处理 |
| LLM-as-Judge 与人类一致性 | 翻译评估中一致性中等 |
| 影响翻译的关键因素 | 谐音链的复杂度是主要因素 |

### 关键发现
- SOTA LLM 在涉及上下文语义理解的任务（MS、CC）上表现尚可，但在需要解码抽象话特有编码机制的任务（TR、RC）上表现差
- 这说明 LLM 理解了"在说什么"但不理解"怎么说的"——对抽象话的形式层面（如字形分解、多级谐音链）缺乏认知
- 视觉维度（字形分解、emoji 隐喻）是 LLM 最大的盲区
- 毒性检测表现差说明抽象话可以有效绕过现有安全过滤

## 亮点与洞察
- **将亚文化语言正式引入 NLP 评估**有重要的社会语言学价值——互联网语言是活的、不断进化的，NLP 系统需要跟上
- 六任务设计从"理解"到"使用"形成完整的能力评估谱——类似外语教学中的分级测试
- 发现 LLM 对"形式"和"内容"的理解存在明显不对称——能理解含义但不理解编码机制，这对 LLM 的语言能力本质有启发

## 局限与展望
- 数据集规模有限（1,099 个实例），可能不足以覆盖抽象话的全部变体
- 仅关注中文抽象话，其他语言的类似亚文化现象（如日语网络用语、韩语缩略语）待探索
- LLM-as-Judge 在翻译评估中的可靠性有限
- 毒性标注可能受标注者主观偏见影响

## 相关工作与启发
- **vs 中文表情包研究 (Xie et al., 2025)**: 表情包是抽象话的子集，抽象话的语义结构更复杂
- **vs 扰动语言检测 (Xiao et al., 2024)**: 他们关注毒性绕过，本文将抽象话视为中性亚文化现象
- **vs 跨文化 LLM 评估 (Cao et al., 2023)**: 他们关注价值观偏差，本文关注语言编码能力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个抽象话 NLP 基准，独特的研究视角
- 实验充分度: ⭐⭐⭐⭐ 六任务全面评估、多模型对比
- 写作质量: ⭐⭐⭐⭐ 语言学分析细致，示例丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)

</div>

<!-- RELATED:END -->
