---
title: >-
  [论文解读] Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy
description: >-
  [CVPR 2025][LLM 其他][组合图像检索] 提出IP-CIR方法，通过大语言模型生成"想象中的目标图像描述"作为代理，将组合图像检索(CIR)转化为标准图像检索问题，在CIRR和FashionIQ等基准上达到零样本SOTA。 领域现状：组合图像检索(CIR)的任务是给定参考图像+修改文本，检索符合修改要求的目标图…
tags:
  - "CVPR 2025"
  - "LLM 其他"
  - "组合图像检索"
  - "虚拟代理图像"
  - "CLIP"
  - "零样本泛化"
  - "文本-图像对齐"
---

# Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy

**会议**: CVPR 2025  
**arXiv**: [2411.16752](https://arxiv.org/abs/2411.16752)  
**代码**: 待确认  
**领域**: 图像检索 / 多模态学习  
**关键词**: 组合图像检索、虚拟代理图像、CLIP、零样本泛化、文本-图像对齐

## 一句话总结
提出IP-CIR方法，通过大语言模型生成"想象中的目标图像描述"作为代理，将组合图像检索(CIR)转化为标准图像检索问题，在CIRR和FashionIQ等基准上达到零样本SOTA。

## 研究背景与动机

**领域现状**：组合图像检索(CIR)的任务是给定参考图像+修改文本，检索符合修改要求的目标图像。现有方法需要大量三元组标注（参考图-文本-目标图），标注成本极高。

**现有痛点**：
   - 有监督CIR方法需昂贵的三元组数据，泛化性有限
   - 零样本CIR方法（如Pic2Word、SEARLE）虽无需三元组，但性能显著低于有监督方法
   - 现有方法难以有效融合图像内容与文本修改意图

**核心洞察**：如果能想象出修改后的目标图像长什么样（即生成一个"虚拟代理"描述），就可以用标准的文本-图像检索代替复杂的组合检索。LLM可以完成这种"想象"——根据参考图像描述和修改文本，推理出目标图像的描述。

## 方法详解

### 整体框架
参考图像 → BLIP2生成图像描述 → LLM(GPT-4)结合修改文本生成目标描述 → CLIP文本编码器编码 → 与数据库图像特征匹配检索。

### 关键设计

1. **想象代理生成（Imagined Proxy）**

    - 用BLIP2将参考图像转为文本描述
    - 将图像描述+修改文本输入LLM，生成"想象中的目标图像描述"
    - 例：参考图描述"红色裙子" + 修改文本"换成蓝色" → 想象描述"蓝色裙子"
    - 设计动机：利用LLM的推理能力完成视觉想象，避免复杂的多模态融合

2. **特征融合与匹配**

    - 用CLIP文本编码器编码想象描述得到代理特征
    - 结合原始参考图像的CLIP视觉特征进行加权融合
    - 与数据库中所有图像的CLIP视觉特征计算相似度
    - 设计动机：保留参考图像的视觉细节，同时注入文本修改意图

3. **训练策略**

    - 零样本设置：无需任何CIR三元组训练数据
    - 仅在推理时使用LLM生成代理描述
    - 可选fine-tune：在有标注数据时可进一步微调对齐模块

### 损失函数 / 训练策略
零样本设置无需训练。有监督设置使用对比学习损失对齐代理特征与目标图像特征。

## 实验关键数据

### 主实验：零样本CIR

| 数据集 | 指标 | IP-CIR | Pic2Word | SEARLE |
|--------|------|--------|----------|--------|
| CIRR | Recall@10 | **70.07** | 58.2 | 62.1 |
| CIRR | Recall@50 | **87.3** | 79.6 | 82.5 |
| FashionIQ (Dress) | Recall@10 | **32.4** | 26.8 | 28.9 |

### 消融实验

| 配置 | Recall@10 | 说明 |
|------|----------|------|
| 仅文本修改 | 55.3 | 忽略参考图像 |
| 仅图像特征 | 48.7 | 无文本修改 |
| 想象代理(文本) | 65.2 | LLM生成描述 |
| 想象代理+图像融合 | **70.07** | 完整方法 |

### 关键发现
- 想象代理显著优于直接文本修改（70.07 vs 55.3），证明LLM推理对CIR有价值
- 图像特征融合额外贡献约5个点的Recall，说明保留视觉细节重要
- 零样本性能接近甚至超越部分有监督方法

## 亮点与洞察
- **范式转换**：将组合检索问题转化为"先想象后检索"，绕过了三元组标注瓶颈
- **LLM作为视觉推理器**：利用LLM的常识推理能力完成视觉想象
- **即插即用**：方法与具体CLIP模型无关，可适配任何多模态基础模型

## 局限与展望
- 依赖LLM的推理质量，复杂修改（如空间关系变换）可能想象不准确
- BLIP2的图像描述可能遗漏关键视觉细节
- LLM推理增加推理延迟
- 对细粒度属性修改（如纹理变化）效果有待验证

## 相关工作与启发
- **vs Pic2Word**：Pic2Word将图像投影到文本空间，但缺乏推理；本文用LLM显式推理
- **vs SEARLE**：SEARLE用检索增强，本文用生成增强

## 评分
- 新颖性: ⭐⭐⭐⭐ 想象代理的思路直观且有效
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证，消融清晰
- 写作质量: ⭐⭐⭐⭐ 动机清晰
- 价值: ⭐⭐⭐⭐ 零样本CIR的实用解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[ACL 2025\] Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](../../ACL2025/llm_nlp/improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)
- [\[CVPR 2025\] Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)
- [\[ACL 2025\] Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting](../../ACL2025/llm_nlp/dolphin_document_image_parsing_via_heterogeneous_anchor_prompting.md)
- [\[NeurIPS 2025\] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era](../../NeurIPS2025/llm_nlp/towards_implicit_aggregation_robust_image_representation_for_place_recognition_i.md)

</div>

<!-- RELATED:END -->
