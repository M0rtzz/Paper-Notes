---
title: >-
  [论文解读] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models
description: >-
  [ECCV 2024][多模态][视觉词汇表] 提出 Vary 方法，通过生成并融合新的视觉词汇表来扩展 LVLM 的视觉感知能力，使模型在保持通用能力的同时获得文档 OCR、图表理解等细粒度视觉感知能力。
tags:
  - ECCV 2024
  - 多模态
  - 视觉词汇表
  - LVLM
  - OCR
  - 文档理解
  - 视觉编码器
---

# Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models

**会议**: ECCV 2024  
**arXiv**: [2312.06109](https://arxiv.org/abs/2312.06109)  
**代码**: https://varybase.github.io/  
**领域**: 多模态视觉语言模型  
**关键词**: 视觉词汇表, LVLM, OCR, 文档理解, 视觉编码器

## 一句话总结

提出 Vary 方法，通过生成并融合新的视觉词汇表来扩展 LVLM 的视觉感知能力，使模型在保持通用能力的同时获得文档 OCR、图表理解等细粒度视觉感知能力。

## 研究背景与动机

当前主流的大型视觉语言模型（LVLM）如 BLIP-2、MiniGPT-4、LLaVA 等，几乎都依赖于同一个视觉词汇表——CLIP-ViT。CLIP-ViT 作为通用视觉词汇表，通过对比学习在 4 亿+ 图文对上训练，能够覆盖大多数自然图像和常见视觉任务。

然而，**CLIP-ViT 在某些特殊视觉任务上存在明显短板**：
- **高分辨率感知**：CLIP 将图像编码为固定数量（通常256个）的 token，难以有效编码密集文本信息
- **非英文 OCR**：CLIP 对中文等非英语文本的编码效率低下
- **文档/图表理解**：对于文档级别的密集内容，CLIP 可能将其视为"外语"，导致词汇表外（OOV）问题

现有的解决方案（如 mPlug-Owl 和 Qwen-VL 通过解冻 CLIP 来微调）存在三个问题：(1) 可能覆盖原有词汇表的知识；(2) 在大 LLM（7B）上更新视觉词汇表的训练效率低；(3) 由于 LLM 的强记忆能力，无法多轮训练视觉数据。

**核心 idea**：类比文本 LLM 中词汇表扩展的做法——将英文 LLM 迁移到中文时需要扩展文本词汇表，对于视觉分支中的"外语"图像（文档、图表），也需要扩展视觉词汇表。

## 方法详解

### 整体框架

Vary 的流程分为两个阶段：**视觉词汇表的生成**和**视觉词汇表的融合**。

- **Vary-tiny**：由新词汇表网络 + OPT-125M 小型 decoder-only transformer 组成，通过自回归方式训练生成新的视觉词汇表
- **Vary-base**：将新词汇表网络与原始 CLIP-ViT 并行集成，构建最终的 LVLM

### 关键设计

1. **新视觉词汇表网络**：

    - 使用 SAM 预训练的 ViTDet（base 规模）图像编码器作为骨干
    - 输入分辨率为 1024×1024，输出步幅为 16，最后一层特征形状为 64×64×256
    - 在最后一层后添加两个卷积层进行 token 合并：第一个 conv（kernel=3）将特征转换为 32×32×512，第二个 conv 进一步转换为 16×16×1024
    - 展平后得到 256×1024 的特征，与 CLIP-ViT 的输出形状对齐
    - **设计动机**：基于自回归的词汇表生成方式比基于对比学习的 CLIP 更适合密集感知任务，因为 next-token 方式可以压缩更长的文本，且数据格式更多样

2. **数据引擎（生成阶段）**：

    - **正样本 - 文档数据**：从 arXiv 和 CC-MAIN-2021-31 收集英文文档，从互联网电子书收集中文文档，共构建 100 万中文 + 100 万英文文档图文对
    - **正样本 - 图表数据**：使用 matplotlib 和 pyecharts 渲染，中英文各 25 万（matplotlib）+ 50 万（pyecharts），文本标注转换为 python-dict 格式
    - **负样本 - 自然图像**：从 COCO 提取 12 万图像，配以简单的自然图像描述文本，确保新词汇表不会对自然图像引入噪声

3. **词汇表融合（Vary-base）**：

    - 新旧视觉词汇表并行处理，各自拥有独立的输入嵌入层（线性层：1024→2048）
    - 拼接后通道维度为 4096，与 LLM（Qwen-7B 或 Vicuna-7B）的输入对齐
    - **关键策略**：冻结新旧视觉词汇表网络的权重，仅训练输入嵌入层和 LLM，避免视觉知识被覆盖
    - **设计动机**：冻结词汇表权重就像在纯文本 LLM 预训练中冻结 tokenizer 一样，确保已编码的视觉知识不被篡改

4. **数据引擎（融合阶段）**：

    - **LaTeX 渲染文档**：从 arXiv 收集 .tex 源码，提取表格、公式和纯文本，用 10+ 模板重新渲染，获得 50 万英文 + 40 万中文页面，标注转为 Mathpix Markdown 格式
    - **语义关联图表**：使用 GPT-4 生成具有语义关联的图表数据 20 万条（解决随机生成文本语义无关联的问题）
    - **通用数据**：从 LAION-COCO 随机提取 400 万自然图文对用于预训练；SFT 阶段使用 LLaVA-80k/LLaVA-CC665k + DocVQA 训练集 + ChartQA 训练集

### 损失函数 / 训练策略

- **Vary-tiny 训练**：全参数优化，batch size=512，训练 3 epochs，AdamW 优化器 + cosine annealing，学习率 5e-5，自回归损失
- **Vary-base 训练**：冻结新旧视觉词汇表网络，优化输入嵌入层和 LLM。预训练学习率 5e-5，SFT 学习率 1e-5，batch size=256，各训练 1 epoch

## 实验关键数据

### 主实验：细粒度文档感知

| 方法 | 中文 OCR (Edit Dist↓) | 英文 OCR (Edit Dist↓) | 公式转换 (Edit Dist↓) | 表格转换 (Edit Dist↓) | 平均 F1↑ |
|------|----------------------|----------------------|----------------------|----------------------|---------|
| Nougat | — | 0.126 | 0.154 | 0.335 | 79.97 |
| Vary-tiny | 0.266 | 0.197 | — | — | — |
| Vary-base | **0.174** | **0.106** | **0.082** | 0.280 | **81.10** |

### 下游任务性能

| 方法 | DocVQA (val) | DocVQA (test) | ChartQA (Avg) |
|------|-------------|--------------|--------------|
| Donut | — | 67.5 | 41.8 |
| Pix2Struct | — | 72.1 | 56.0 |
| Qwen-VL | — | 65.1 | 65.7 |
| Vary-base (80k) | **78.2** | **76.3** | 65.3 |
| Vary-base (665k) | 78.1 | 76.3 | **66.1** |

### 消融实验：通用性能（MMVet）

| 方法 | Rec | OCR | Know | Gen | Spat | Math | Total |
|------|-----|-----|------|-----|------|------|-------|
| LLaVA-7B | 28.0 | 17.1 | 16.3 | 18.9 | 21.2 | 11.5 | 23.8 |
| LLaVA-13B | 39.2 | 22.7 | 26.5 | 29.3 | 29.6 | 7.7 | 32.9 |
| LLaVA1.5-7B | — | — | — | — | — | — | 30.5 |
| Vary-base (Vicuna, 665k) | 38.7 | 22.0 | 23.6 | 24.1 | 29.6 | 7.7 | 32.9 |
| Vary-base (Qwen, 80k) | 38.9 | **30.1** | 22.4 | 21.7 | **34.3** | 7.7 | **36.2** |

### 关键发现

- Vary-base 在英文纯文本文档上与专用文档解析模型 Nougat 性能持平，在 Markdown 转换上甚至更优（81.10% vs 79.97% F1），得益于 7B LLM 强大的文本纠错能力
- 在 DocVQA 上达到 78.2% ANLS，显著优于 Qwen-VL（65.1%），证明视觉词汇表扩展方法对下游任务同样有效
- 使用相同 LLM（Vicuna-7B）和 SFT 数据时，Vary 在 MMVet 上比 LLaVA-1.5 提升 2.4%（32.9% vs 30.5%），说明数据和训练策略不损害模型通用能力

## 亮点与洞察

- **类比文本词汇表扩展的创新思路**：将视觉词汇表扩展类比为文本 LLM 中的词汇表扩展，思路自然且有效，从 tokenizer 视角重新审视 LVLM 设计
- **冻结策略的巧妙设计**：在融合阶段冻结新旧视觉词汇表网络、仅训练嵌入层和 LLM，避免了视觉知识被覆盖的问题，同时降低了训练成本
- **完整的数据引擎**：为两个阶段分别设计了针对性的数据构建策略，特别是负样本设计（自然图像配简单描述）确保新词汇表不干扰原有能力
- **通用方法论**：Vary 是一种通用策略，可应用于任何 CLIP 不擅长的下游视觉任务

## 局限与展望

- 目前仅在文档和图表场景验证了词汇表扩展的有效性，对于其他细粒度任务（如遥感、医学影像）尚待探索
- 两阶段训练流程较为复杂，新词汇表的生成需要额外的 Vary-tiny 训练
- 合成数据的质量和多样性可能限制模型在真实场景中的泛化能力
- 固定 256 个 token 的视觉表示可能仍然不够，可以探索动态 token 数量的设计

## 相关工作与启发

- **vs CLIP-ViT 微调（mPlug-Owl, Qwen-VL）**：直接微调可能覆盖原有知识且训练效率低，Vary 通过生成独立词汇表并冻结融合来避免这些问题
- **vs Nougat**：Nougat 是专用文档解析模型，Vary 作为通用 LVLM 在文档解析上已达到可比甚至更优的性能
- **vs LLaVA**：LLaVA 使用简单线性层对齐视觉特征，Vary 在此基础上引入了并行的双词汇表结构，更好地处理细粒度视觉信息

## 评分

- 新颖性: ⭐⭐⭐⭐ 从 tokenizer 角度思考 LVLM 视觉能力的提升，类比文本词汇表扩展的 idea 新颖且自然
- 实验充分度: ⭐⭐⭐⭐ 涵盖文档 OCR、下游 VQA、通用 benchmark 等多维度评测，但缺少更多细粒度任务的验证
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，方法阐述自然流畅，类比恰当
- 价值: ⭐⭐⭐⭐⭐ 提供了一种通用的视觉词汇表扩展范式，对后续 LVLM 研究具有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_vision-language_models.md)
- [IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)
- [MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](marvelovd_marrying_object_recognition_and_visionlanguage_mod.md)
- [NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models](navgpt-2_unleashing_navigational_reasoning_capability_for_large_vision-language_.md)
- [Robust Calibration of Large Vision-Language Adapters](robust_calibration_of_large_vision-language_adapters.md)

<!-- RELATED:END -->
