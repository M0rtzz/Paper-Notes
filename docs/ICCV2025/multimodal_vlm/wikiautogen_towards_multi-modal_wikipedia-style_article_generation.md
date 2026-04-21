---
title: >-
  [论文解读] WikiAutoGen: Towards Multi-Modal Wikipedia-Style Article Generation
description: >-
  [ICCV2025][多模态][多模态文章生成] 提出 WikiAutoGen 多智能体框架，通过整合文本和图像的多模态检索与多视角自反思机制，自动生成高质量的多模态 Wikipedia 风格文章，在自建基准 WikiSeek 上相比已有方法提升 8%–29%。
tags:
  - ICCV2025
  - 多模态
  - 多模态文章生成
  - 多智能体框架
  - 自反思机制
  - Wikipedia
  - 知识检索
---

# WikiAutoGen: Towards Multi-Modal Wikipedia-Style Article Generation

**会议**: ICCV2025  
**arXiv**: [2503.19065](https://arxiv.org/abs/2503.19065)  
**代码**: [wikiautogen.github.io](https://wikiautogen.github.io/)  
**领域**: multimodal_vlm  
**关键词**: 多模态文章生成, 多智能体框架, 自反思机制, Wikipedia, 知识检索

## 一句话总结

提出 WikiAutoGen 多智能体框架，通过整合文本和图像的多模态检索与多视角自反思机制，自动生成高质量的多模态 Wikipedia 风格文章，在自建基准 WikiSeek 上相比已有方法提升 8%–29%。

## 研究背景与动机

知识发现和内容生成是信息组织与传播的关键环节，但传统方式依赖大量人工进行收集、结构化和验证。随着 LLM 的发展，自动化 Wikipedia 风格文章生成逐渐兴起，代表性方法如 Storm 和 Co-Storm。然而，现有方法存在两个核心限制：

**仅支持纯文本生成**：无法检索和整合图像等多模态内容，降低了文章的信息丰富度和可读性

**生成内容缺乏广度、深度和可靠性**：文章的信息量和可信度不足，难以满足高质量知识生成的需求

此外，现有评估基准（如 Surfer100、FreshWiki）也主要聚焦于纯文本生成，缺少针对多模态知识生成的评估标准，且多覆盖简单主题，未充分考验模型在冷门/困难主题上的能力。

## 方法详解

### WikiAutoGen 整体框架

WikiAutoGen 是一个多智能体协作框架，包含四个核心模块，按流水线协同工作：

#### 1. 大纲提议模块（Outline Proposal Module）

- 对**文本主题**：LLM 分析输入，识别子主题，生成结构化大纲
- 对**图像主题**：利用 Google Vision Search 获取元数据，通过 NER 提取 Top-10 高频实体作为查询关键词，结合原始主题生成大纲
- 对**图文混合主题**：融合两种模态提取的子主题，由 LLM 整合生成统一大纲

#### 2. 文本文章撰写模块（Textual Article Writing Module）

这一模块包含三个子组件：

- **角色生成器（Persona Generator）**：根据大纲草稿，LLM 生成 n 个与主题相关的不同角色，每个角色作为独立智能体，拥有外部搜索工具的访问权限
- **多智能体知识探索（Multi-agent Knowledge Exploration）**：一个固定的 "提问者" 智能体遍历大纲提出问题，n 个角色智能体搜索互联网获取信息，共享发现并讨论。在此过程中与多视角自反思模块交互，从"写作者"视角获取可靠性、一致性等方面的反馈
- **文章生成（Article Generation）**：由 LLM 写作智能体汇总收集的知识并生成文章初稿，然后将每个段落送入自反思模块迭代改进

#### 3. 多视角自反思模块（Multi-Perspective Self-Reflection Module）

这是本文的核心创新之一。从**七个关键维度**（可靠性、参与度、信息量、连贯性、可读性、一致性、有用性）评估文章质量，并设计了**四个评估视角**：

- **督导视角（Supervisor）**：评估内容是否完整回应了提问，审视文章的深度、广度和连贯性，以及多智能体讨论的有效性
- **写作者视角（Writer）**：关注知识探索和文章生成阶段，评估连贯性、参与度、事实准确性和逻辑一致性，提出具体改进建议（如重排句子、添加过渡词、简化复杂概念）
- **读者视角（Reader）**：评估图像放置位置和描述的可读性、参与度和有用性，确保视觉内容有效融入阅读体验
- **编辑视角（Editor）**：审查图像与文本描述之间的一致性和对齐度，提供调整图像说明、位置或文本解释的建议

#### 4. 多模态文章撰写模块（Multimodal Article Writing Module）

在文本生成完成后，通过以下步骤整合视觉内容：

- **图像定位提议**：LLM 智能体为文章中适当位置提出图像放置方案和对应描述，经自反思模块从读者视角评估
- **图像检索**：基于多来源（通用搜索引擎、Wikipedia、参考文献中的网站）进行图像搜索
- **图像选择**：先用 CLIP 模型按语义相似度排序检索图像、选取 Top-3 候选，再用多模态模型进一步评估选出最合适的图像
- **文章润色**：将图像整合进文章后，利用多模态模型修订全文以增强跨模态连贯性，并通过编辑视角的自反思反馈进一步精修

### WikiSeek 基准

WikiSeek 是本文提出的多模态评估基准，具有以下特点：

- 从 WikiWeb2M 数据集（约 200 万英文 Wikipedia 文章）中选取主题
- 聚焦 Wikipedia 上覆盖不足的冷门主题，按主文内容字符数分为三个难度等级：hard（300–500字符）、very hard（100–300字符）、extremely hard（<100字符）
- 共 300 个主题，三个难度各 100 个，支持纯文本、纯图像、图文混合三种输入格式
- 经过严格的人工筛选，去除过于笼统（如"1997 in Japan"）或语义不明确的主题

### 实现细节

- 使用 DSPy 框架 + GPT-4o / GPT-4o-mini / GPT-o3-mini
- 自反思模块用 GPT-o3-mini（推理能力强），多模态知识探索用 GPT-4o，其他任务用 GPT-4o-mini
- 通过 Serper API 进行实时网页检索，每个查询返回最多 5 个网页
- 温度 1.0，top_p 0.9

## 实验关键数据

### 文本质量评估（9维评估）

| 方法 | 文本主题 | 图像主题 | 图文主题 |
|------|---------|---------|---------|
| oRAG | 60.71 | 48.38 | 58.76 |
| Storm | 65.26 | 43.57 | 61.12 |
| Co-Storm | 69.96 | 44.39 | 63.77 |
| OmniThink | 63.98 | 41.39 | 58.67 |
| **WikiAutoGen** | **78.73** | **77.49** | **78.82** |

- 文本主题提升 +8.8（vs Co-Storm）
- 图像主题提升 +29.1（vs oRAG），优势最为显著
- 图文主题提升 +15.1（vs Co-Storm）

### 图像质量评估（4维评估）

| 方法 | 文本主题 | 图像主题 | 图文主题 |
|------|---------|---------|---------|
| oRAG | 57.28 | 54.32 | 58.85 |
| WikiAutoGen | **68.99** | **68.78** | **70.98** |

图像质量在各输入模态上均提升 11%–14%。

### 消融实验（图像主题输入）

| 配置 | 平均分 |
|------|-------|
| 无任何模块 | 48.78 |
| + 多智能体 | 66.56（+17.78）|
| + 大纲提议 | 72.88（+24.10）|
| + 自反思 | 71.60（+22.82）|
| 全部模块 | **77.49** |

大纲提议对内容结构化的贡献最大（+24.10），自反思对质量精修至关重要（+22.82），三者互补。

### 与商业 Deep Research 对比

| 方法 | 平均分 | 耗时 |
|------|-------|------|
| OpenAI | 93.06 | ~30 min |
| Grok | 85.06 | ~10 min |
| Google | 81.91 | ~12 min |
| **WikiAutoGen** | **89.01** | **~8 min** |

WikiAutoGen 接近 OpenAI 水平，超过 Grok 和 Google，且速度最快（比 OpenAI 快 3.75×）。

### 人类评估

在 100 个文本主题上通过 AMT 进行两两比较：
- 97.7% 的参与者认为添加图像能提高对主题的理解
- 在可读性、参与度、信息量和整体偏好上，WikiAutoGen 均一致优于 Storm 和 OmniThink

## 亮点与洞察

1. **首个多模态 Wikipedia 文章生成系统**：突破了已有方法仅限纯文本的局限，真正实现了"图文并茂"的自动化知识生成
2. **多视角自反思设计精妙**：四种角色（督导/写作者/读者/编辑）×七个评估维度的矩阵式反思，模拟了人类协作写作的真实流程，每个角色在不同阶段介入
3. **难度分层基准的设计思路值得借鉴**：WikiSeek 按 Wikipedia 覆盖程度定义难度，以"冷门主题"作为真正的挑战，避免模型简单复制已有高质量 Wikipedia 内容
4. **图像主题场景的巨大优势**（+29.1 分）说明多模态检索在纯图像输入时的价值极高——基线方法几乎不知如何从图像出发生成文章
5. **实际效率出色**：8 分钟生成一篇文章，比 OpenAI Deep Research 快 3.75×，同时质量接近

## 局限与展望

1. **对闭源模型的依赖**：核心组件均使用 GPT-4o/o3-mini，成本高且不利于复现；未探索开源 LLM 的表现
2. **评估以 GPT-4o 为主**：LLM-as-judge 的偏差问题未被充分讨论，虽附录补充了 Gemini 和 Prometheus2 评估，但人类评估规模有限
3. **图像选择流程的局限**：CLIP → 多模态模型的两阶段筛选可能遗漏语义更相关但视觉风格不匹配的图像；图像质量（分辨率、版权）未被充分考虑
4. **可扩展性存疑**：多智能体 + 多轮自反思的流程复杂度高，每篇文章需要大量 API 调用，批量生成成本可观
5. **未涉及事实性验证**：自反思关注文本质量维度较多，缺少对事实准确性的外部验证机制（如与知识图谱的交叉检验）
6. **WikiSeek 基准规模较小**（300 个主题），且仅覆盖英文 Wikipedia

## 相关工作与启发

- **Storm / Co-Storm**（Shao et al., 2024）：Wikipedia 文章自动生成的先驱，WikiAutoGen 在其基础上引入多模态能力和更精细的自反思
- **OmniThink**（Xi et al., 2025）：通过迭代扩展和反思增强文章质量，模拟人类慢思考
- **Self-RAG**（Asai et al., 2023）：自检索增强生成的代表，启发了 oRAG 基线
- **DSPy**（Khattab et al., 2024）：模块化编程框架，WikiAutoGen 以此构建多智能体流水线
- 多视角自反思的设计思路可推广到其他长文本生成任务（如调研报告、综述写作、技术文档生成）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个多模态 Wikipedia 文章生成系统，多视角自反思有创新
- 实验充分度: ⭐⭐⭐⭐ — 文本/图像/消融/人类评估/商业对比，维度全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，模块划分明确，图示丰富
- 价值: ⭐⭐⭐⭐ — 多模态知识生成是重要方向，框架设计有较好的工程参考价值

<!-- RELATED:START -->

## 相关论文

- [Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](../../CVPR2025/multimodal_vlm/marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](../../CVPR2026/multimodal_vlm/wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)
- [FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)
- [Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)

<!-- RELATED:END -->
