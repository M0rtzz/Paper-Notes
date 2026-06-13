---
title: >-
  [论文解读] A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation
description: >-
  [CVPR 2025][图像生成][文本编码器] 系统研究了使用decoder-only LLM作为文本到图像扩散模型文本编码器的效果，发现直接使用最后一层embedding效果差于T5，但通过层归一化平均（layer-normalized averaging）聚合所有层的embedding可显著超越T5基线。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "文本编码器"
  - "decoder-only LLM"
  - "文本到图像生成"
  - "层归一化平均"
  - "视觉语言推理"
---

# A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation

**会议**: CVPR 2025  
**arXiv**: [2506.08210](https://arxiv.org/abs/2506.08210)  
**代码**: 无  
**领域**: 图像生成 / 多模态  
**关键词**: 文本编码器, decoder-only LLM, 文本到图像生成, 层归一化平均, 视觉语言推理

## 一句话总结

系统研究了使用decoder-only LLM作为文本到图像扩散模型文本编码器的效果，发现直接使用最后一层embedding效果差于T5，但通过层归一化平均（layer-normalized averaging）聚合所有层的embedding可显著超越T5基线。

## 研究背景与动机

**领域现状**：当前文本到图像生成模型（如Stable Diffusion、DALL-E 3）普遍使用T5或CLIP作为文本编码器。然而T5是encoder-decoder架构的旧模型，CLIP模型规模小（354M）、token长度仅77，表达能力有限。

**现有痛点**：decoder-only LLM在NLP领域已全面超越encoder-decoder架构，但在文本到图像生成中的潜力未被系统研究。少数使用LLM的工作（如Lumina、Sana）直接用最后一层输出，且训练配置各异，无法公平对比。

**核心矛盾**：decoder-only LLM使用causal attention mask，信息只能从左向右流动，最后一层可能不是最佳的embedding表示；而encoder-decoder模型（T5）使用bidirectional attention，最后一层信息更完整。

**本文要解**：(1) decoder-only LLM能否替代T5用于文本到图像？(2) 如何最优地从LLM中提取embedding？(3) LLM微调的embedding模型是否更好？(4) 模型规模增大是否有用？

**核心idea**：使用layer-normalized averaging聚合LLM所有层的embedding，让不同层捕获的不同语言信息互补，构建更丰富的文本表示。

## 方法详解

### 整体框架

基于Stable Diffusion v2的U-Net架构，只替换文本编码器（冻结），通过cross-attention将文本embedding注入U-Net。加入一个线性投影层（输出1024维）适配不同文本编码器的embedding维度。在46M text-image pairs上训练800K iterations，256×256分辨率，32×A100。

### 关键设计

1. **Embedding提取策略对比**：

    - 功能：对比last-layer、single intermediate layer、average、layer-normalized average四种提取方式
    - 核心发现：对decoder-only LLM，last-layer embedding效果最差（VQAScore 0.675 for Mistral-7B），远不如T5的0.741；中间层（如15层）略好（0.725）；简单平均（avg）提升至0.731；**层归一化平均（norm avg）效果最佳（0.769）**，因为不同层的embedding范数差异巨大，归一化后再平均才能公平融合各层信息
    - 设计动机：LLM每一层捕获不同的语言特征——底层捕获词法/语法信息，中间层捕获语义，顶层压缩为next-token预测目标。平均所有层可综合利用这些互补信息

2. **LLM微调embedding模型评估**：

    - 功能：评估MTEB排行榜top的微调embedding模型（bge-Gemma2, sfr-Mistral, gte-Qwen2）
    - 核心发现：bge-Gemma2（基于Gemma2-9B微调）在norm avg下达到最佳性能（VQAScore 0.789），全面超越T5（0.741）。但gte-Qwen2表现极差（0.482），可能因为其微调目标过于偏向句子级语义，丢失了token级别的细粒度信息
    - 设计动机：embedding模型通过对比学习微调提升语义理解能力，理论上应更好地捕获文本到图像对齐所需的语义信息

3. **模型规模效应**：

    - 功能：对比Gemma2-2B vs 9B、Qwen2-1.5B vs 7B
    - 核心发现：增大模型规模持续提升性能（Gemma2: 0.757→0.789，Qwen2: 0.740→0.769），但并非所有方面均匀提升——Counting和Comparison技能提升最大，而Scene和Negation提升有限
    - 设计动机：验证LLM的scaling law是否迁移到文本到图像生成

### 训练策略

- 使用VFC（VisualFactChecker）做caption upsampling增强训练文本多样性
- 推理时使用Gemma2-9B做prompt upsampling匹配训练分布
- CFG固定为7.0进行公平对比
- 评估使用VQAScore（GPT-4o实现），比CLIPScore/FID更准确反映组合文本到图像对齐

## 实验关键数据

### 主实验：Last-Layer Embedding对比（VQAScore on GenAI-Bench）

| 模型 | 参数量 | 平均 | Counting | Comparison | Negation |
|------|--------|------|----------|------------|----------|
| CLIP-ViT-H/14 | 354M | 0.622 | 0.529 | 0.522 | 0.480 |
| T5-XXL | 4.7B | 0.741 | 0.677 | 0.717 | 0.599 |
| Mistral-7B | 7B | 0.675 | 0.576 | 0.556 | 0.524 |
| Gemma2-9B | 9B | 0.710 | 0.642 | 0.659 | 0.544 |
| bge-Gemma2 | 9B | 0.737 | 0.662 | 0.654 | 0.623 |

### 消融实验：不同Embedding策略（VQAScore）

| 模型 | 策略 | 平均 | Counting | Comparison | Negation |
|------|------|------|----------|------------|----------|
| T5-XXL | last layer | 0.741 | 0.677 | 0.717 | 0.599 |
| T5-XXL | norm avg | 0.747 | 0.687 | 0.736 | 0.617 |
| Mistral-7B | last layer | 0.675 | 0.576 | 0.556 | 0.524 |
| Mistral-7B | norm avg | **0.769** | 0.699 | 0.716 | 0.630 |
| bge-Gemma2 | last layer | 0.737 | 0.662 | 0.654 | 0.623 |
| bge-Gemma2 | norm avg | **0.789** | **0.745** | **0.776** | **0.712** |

### 关键发现

- **Last-layer是陷阱**：所有decoder-only LLM的last-layer embedding都弱于T5，但norm avg后反超。这是因为LLM最后一层被next-token prediction目标"污染"，信息被压缩
- **Norm avg是关键**：Mistral-7B从0.675→0.769（+13.9%），bge-Gemma2从0.737→0.789（+7.1%）。归一化解决了不同层embedding范数差异巨大的问题
- **最佳模型**：bge-Gemma2 + norm avg达到0.789，全面超越T5-XXL的0.741，在所有10个技能维度上均领先
- **Negation能力显著提升**：这是CLIP/T5最弱的技能（需理解"not"等否定语义），LLM在此有天然优势

## 亮点与洞察

- **反直觉发现**：直接用LLM最后一层做text-to-image比T5差，但换个提取策略就能大幅超越。这说明"怎么用"比"用什么"更重要，对整个社区使用LLM作为文本编码器有重要指导意义
- **层归一化平均的优雅性**：不需要任何训练，仅改变embedding提取方式就能获得巨大提升。这个trick可直接应用于任何使用LLM做文本编码器的系统
- **系统性benchmark设计**：27个模型×统一训练配置×10维技能评估，控制变量严谨，结论可信度高

## 局限与展望

- 仅在256×256分辨率的U-Net架构上验证，未测试DiT架构和更高分辨率
- 计算成本高：每个模型训练7天×32 A100，27个模型的系统研究耗费巨大
- 未探索更高效的层融合策略（如learned layer weighting）
- 未验证在autoregressive图像生成模型中是否同样成立

## 相关工作与启发

- **vs Playground-v3**：PGv3也使用Llama3做文本编码器，但通过adapter和不同DiT块使用不同中间层。本文的norm avg更简单且效果更好
- **vs Lumina/Sana**：它们直接用Gemma2最后一层输出，根据本文发现是次优选择
- **vs T5基线**：T5作为bidirectional encoder-decoder模型，最后一层自然包含完整双向信息，但LLM通过多层聚合可以弥补causal attention的信息损失

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性研究视角新颖，layer-normalized average虽简单但有效，核心发现对社区有重要价值
- 实验充分度: ⭐⭐⭐⭐⭐ 27个模型、统一训练配置、10维技能分解评估，控制变量极其严谨
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，但部分表格数据密度过高
- 价值: ⭐⭐⭐⭐⭐ 为整个文本到图像社区使用LLM文本编码器提供了明确的最佳实践指南

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] RandAR: Decoder-only Autoregressive Visual Generation in Random Orders](randar_decoder-only_autoregressive_visual_generation_in_random_orders.md)
- [\[CVPR 2025\] VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness](verbdiff_text-only_diffusion_models_with_enhanced_interaction_awareness.md)
- [\[CVPR 2025\] Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)
- [\[CVPR 2025\] Make It Count: Text-to-Image Generation with an Accurate Number of Objects](make_it_count_text-to-image_generation_with_an_accurate_number_of_objects.md)
- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
