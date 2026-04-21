---
title: >-
  [论文解读] X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs
description: >-
  [ECCV 2024][多模态][多模态大语言模型] 提出X-Former，一个轻量级Transformer模块，通过双交叉注意力机制融合CLIP-ViT（对比学习）和MAE-ViT（掩码图像建模）的互补视觉特征，在仅使用1/10数据量的情况下显著超越BLIP-2在细粒度视觉理解任务上的表现。
tags:
  - ECCV 2024
  - 多模态
  - 多模态大语言模型
  - 对比学习
  - 掩码图像建模
  - 视觉表征
  - Q-Former
---

# X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs

**会议**: ECCV 2024  
**arXiv**: [2407.13851](https://arxiv.org/abs/2407.13851)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 多模态大语言模型, 对比学习, 掩码图像建模, 视觉表征, Q-Former

## 一句话总结

提出X-Former，一个轻量级Transformer模块，通过双交叉注意力机制融合CLIP-ViT（对比学习）和MAE-ViT（掩码图像建模）的互补视觉特征，在仅使用1/10数据量的情况下显著超越BLIP-2在细粒度视觉理解任务上的表现。

## 研究背景与动机

当前多模态大语言模型（MLLMs）普遍采用CLIP-ViT作为视觉编码器，但CLIP基于对比学习训练，主要关注低频信号和全局模式，在捕捉细粒度视觉细节方面存在明显不足，如物体朝向、结构细节、空间关系和多实例识别等。MAE-ViT通过掩码图像建模训练，擅长理解局部和高频视觉特征，但两者的融合并非简单拼接就能生效。

作者通过实验验证了两个关键发现：

**简单拼接CLIP和MAE特征效果与BLIP-2持平**，说明两种编码器提供的信息差异大，模型难以同时学习全局和局部信息

**早期交叉注意力虽略有提升但增加了大量参数（75M）**，且在GQA数据集上反而下降

核心矛盾在于：如何在不大幅增加参数的前提下，有效融合CL和MIM两种视觉表征，使LLM能同时理解全局语义和局部细节。X-Former的切入角度是设计一种双交叉注意力交互机制，通过重建损失引导MAE特征与全局语义对齐。

## 方法详解

### 整体框架

X-Former采用两阶段训练：
- **Stage 1（预训练）**：从两个冻结的视觉编码器（CLIP-ViT和MAE-ViT）中学习全局+局部视觉表征
- **Stage 2（LLM对齐）**：将X-Former的输出与冻结的LLM对齐

框架由四部分组成：冻结的CLIP-ViT编码器、冻结的MAE-ViT编码器、冻结的MAE解码器、以及可训练的X-Former模块。

### 关键设计

1. **Q-Former基础模块**:

    - 功能：使用可学习查询向量从CLIP-ViT提取全局语义视觉特征
    - 核心思路：查询向量通过自注意力层相互交互，通过交叉注意力层与冻结图像特征交互
    - 设计动机：继承BLIP-2的成功架构，但仅能捕捉全局表征

2. **双交叉注意力模块（X-Former核心）**:

    - 功能：在Q-Former输出基础上融入MAE的局部细节特征
    - 核心思路：分两步交叉注意力
        - **第一步**：MAE特征M作为Query，Q-Former输出Zq作为Key/Value → 生成语义增强的MAE特征M'（将全局语义信息注入MAE局部特征）
        - **第二步**：增强后的MAE特征M'作为Key/Value，Zq作为Query → 生成最终增强查询Z'（将局部细节信息注入全局查询）
    - 设计动机：先对齐再融合。直接融合两种差异巨大的表征效果不好，通过中间桥接步骤逐步对齐两种表征空间

3. **MAE掩码与重建**:

    - 功能：对输入图像进行随机掩码（50%比例），增强后的MAE特征M'送入冻结MAE解码器重建掩码区域
    - 核心思路：重建损失迫使网络从MAE中提取有意义的局部信息，而非走捷径
    - 设计动机：没有重建目标时，网络无法有效利用MAE特征（消融实验验证性能大幅下降）

### 损失函数 / 训练策略

**Stage 1 预训练（4个损失函数）**：

| 损失函数 | 作用 | 注意力掩码 |
|---------|------|-----------|
| ITC（图像-文本对比） | 最大化正样本对的图像-文本相似度 | 单模态自注意力掩码，防止查询和文本交互 |
| ITM（图像-文本匹配） | 二分类判断图像-文本对是否匹配 | 双向自注意力掩码，允许查询和文本互相关注 |
| ITG（图像条件文本生成） | 以图像为条件生成对应文本 | 多模态因果自注意力掩码 |
| Reconstruction（重建） | 重建MAE掩码的图像区域 | 作用于增强MAE特征M' |

**Stage 2 LLM对齐**：
- 将X-Former输出Z'通过一个全连接层映射到LLM嵌入空间
- 仅使用语言建模损失训练，冻结所有视觉编码器和LLM
- 不使用重建损失（消融表明Stage 2加重建损失反而有害）

**训练细节**：
- Stage 1训练9个epoch，Stage 2训练1个epoch
- CLIP-ViT使用EVA-CLIP的ViT-G，MAE使用ViT-H
- LLM使用OPT模型（2.7B和6.7B两个规模）
- 训练数据仅14M图文对（BLIP-2用129M），约为BLIP-2的1/10
- 训练时间增加约10%，GPU显存增加约4.7%

## 实验关键数据

### 主实验：零样本VQA

| 数据集 | 指标 | X-Former (OPT 6.7B) | BLIP-2 (OPT 6.7B) | 提升 |
|--------|------|---------------------|-------------------|------|
| VQAv2 | Overall Acc | **55.0** | 52.4 | +2.6% |
| VQAv2 | Number Acc | **37.8** | 30.8 | +7.0% |
| GQA | Acc | **34.9** | 33.1 | +1.8% |
| OKVQA | Acc | **34.2** | 31.5 | +2.7% |

### 细粒度视觉感知

| 任务 | 数据集 | X-Former | BLIP-2* (129M) | BLIP-2 (14M) | 说明 |
|------|--------|---------|---------------|-------------|------|
| 物体计数OC | COCO | **39.64** | 34.3 | 25.88 | 超越129M数据训练的BLIP-2 |
| 物体计数OC | VCR | **27.24** | 18.9 | 21.12 | 大幅超越 |
| 多类识别MCI | COCO | **69.44** | 69.44 | 61.5 | 持平 |
| 多类识别MCI | VCR | 69.28 | **74.16** | 65.3 | 略低 |

### 消融实验

| 配置 | VQAv2 | GQA | OKVQA | 说明 |
|------|-------|-----|-------|------|
| X-Former (完整) | **55.0** | **34.9** | **34.2** | 最佳 |
| 无重建损失（Stage 1&2都无） | 33.1 | 25.4 | 12.1 | 性能灾难性下降 |
| Stage 1有+Stage 2也有重建 | 52.4 | 32.2 | 29.2 | Stage 2不需要重建 |
| 用CLIP L26层替代MAE | 53.7 | 32.6 | 31.2 | MAE优于CLIP中间层 |
| 简单拼接 (110M params) | 52.3 | 32.1 | 31.9 | 无效 |
| 早期交叉注意力 (183M params) | 53.8 | 32.7 | 31.5 | 参数多但效果差 |

### 关键发现
- 物体计数任务提升最为显著（COCO +13%，VCR +6.1%），证明局部细节理解的增强
- 仅用1/10数据量即超越BLIP-2官方checkpoint（129M数据训练）
- 重建损失是核心关键——没有它网络无法有效利用MAE特征
- 计算开销增加很小：训练时间+10%，显存+4.7%，推理~890ms vs ~680ms

## 亮点与洞察
- **互补性融合的优雅设计**：双交叉注意力"先对齐再融合"的策略比简单拼接或早期交互更有效，且参数效率更高
- **重建损失的双重作用**：既作为MAE特征的对齐信号，又防止网络走捷径忽略局部信息
- **数据效率惊人**：14M数据超越129M数据训练的BLIP-2，说明更好的视觉表征比更多数据更重要
- **即插即用**：X-Former可以替代Q-Former应用于其他MLLM框架

## 局限与展望
- 仅在OPT上验证，未覆盖更强的LLM（如LLaMA、Vicuna等）
- 推理时间增加约30%（890ms vs 680ms），因引入额外MAE编码器
- MCI任务在VCR上略低于BLIP-2，全局理解略有牺牲
- 未与LLaVA等instruction-tuning方法公平对比
- 可尝试更多MIM变体（如BEiT、SimMIM）替代MAE

## 相关工作与启发
- **BLIP-2**：直接的baseline和框架基础，X-Former在其Q-Former上扩展
- **MMVP**：同样利用自监督编码器，但依赖instruction tuning
- **CL+MIM联合训练**：之前仅在视觉预训练中探索，未应用于VL理解
- 启发：不同预训练目标的视觉编码器确实编码了互补信息，关键在于融合机制的设计

## 评分
- 新颖性: ⭐⭐⭐⭐ 双交叉注意力融合CL和MIM的思路清晰有效，但整体框架仍在BLIP-2基础上改进
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证+详尽消融，但缺少与更多MLLM方法的对比
- 写作质量: ⭐⭐⭐⭐ 动机-实验-分析逻辑链完整，循序渐进，从失败尝试引出最终方案
- 价值: ⭐⭐⭐⭐ 数据高效且即插即用，对MLLM视觉表征改进有实际参考价值

<!-- RELATED:START -->

## 相关论文

- [CLAP: Isolating Content from Style Through Contrastive Learning with Augmented Prompts](clap_isolating_content_from_style_through_contrastive_learni.md)
- [Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)
- [BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_visionlanguage_model.md)
- [CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios](cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)
- [Merlin: Empowering Multimodal LLMs with Foresight Minds](merlin_empowering_multimodal_llms_with_foresight_minds.md)

<!-- RELATED:END -->
