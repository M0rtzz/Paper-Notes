---
title: >-
  [论文解读] Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion
description: >-
  [CVPR 2025][多模态][生成式视觉编码器] 用生成式视觉基础模型 Florence-2 替换 CLIP 作为 VLM 视觉编码器，通过"深度-广度融合"（DBFusion）整合底层 DaViT 特征和三种任务提示（描述/OCR/定位）的高层特征，以单编码器 576 token 实现超越多编码器方案的性能。
tags:
  - CVPR 2025
  - 多模态
  - 生成式视觉编码器
  - Florence-2
  - 深度-广度融合
  - 多任务特征提取
  - OCR
---

# Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion

**会议**: CVPR 2025  
**arXiv**: [2412.04424](https://arxiv.org/abs/2412.04424)  
**代码**: https://github.com/JiuhaiChen/Florence-VL  
**领域**: 多模态VLM  
**关键词**: 生成式视觉编码器、Florence-2、深度-广度融合、多任务特征提取、OCR增强

## 一句话总结
用生成式视觉基础模型 Florence-2 替换 CLIP 作为 VLM 视觉编码器，通过"深度-广度融合"（DBFusion）整合底层 DaViT 特征和三种任务提示（描述/OCR/定位）的高层特征，以单编码器 576 token 实现超越多编码器方案的性能。

## 研究背景与动机

**领域现状**：当前主流 VLM（LLaVA、Cambrian 等）使用 CLIP 或 SigLIP 等对比学习预训练的视觉编码器。部分方法（Cambrian）为弥补单一编码器的不足，采用多编码器融合策略。

**现有痛点**：CLIP 通过对比学习产生单一的全局特征，缺乏针对不同视觉任务（OCR、定位、场景描述）的专项能力。多编码器方案虽然弥补了这一缺陷，但增加了参数量和推理开销。

**核心矛盾**：VLM 的视觉编码器需要同时具备多种感知能力（文字识别、空间定位、全局理解），但传统对比学习编码器只学到一种统一的视觉-语言对齐表征。

**本文目标** 用单个生成式视觉模型替代 CLIP，同时提供多种任务级别的视觉特征，实现"一个编码器做多个编码器的事"。

**切入角度**：Florence-2 是一个生成式视觉基础模型，可以通过不同文本提示提取不同任务的视觉特征（如 OCR、详细描述、区域定位），这种多任务能力天然适合作为 VLM 的视觉前端。

**核心 idea**：利用 Florence-2 的多提示特征提取能力，通过通道维度拼接底层和多任务高层特征（DBFusion），以 576 token 实现高效且全面的视觉编码。

## 方法详解

### 整体框架
输入图像 → Florence-2 的 DaViT 骨干提取底层特征 $V$ → 三种任务提示分别送入 Florence-2 编码器提取高层特征 $V'_{t1}$（详细描述）、$V'_{t2}$（OCR）、$V'_{t3}$（密集区域描述）→ 沿通道维度拼接 $[V, V'_{t1}, V'_{t2}, V'_{t3}]$ → MLP 投影到 LLM 输入空间 → 576 个视觉 token 送入 Phi 3.5 或 LLaMA 3。

### 关键设计

1. **广度维度：三种任务提示提取**:

    - 功能：从同一图像提取不同感知侧重的视觉特征
    - 核心思路：利用 Florence-2 的生成式架构（可接受文本提示控制输出），设计三种提示——"describe what is shown in the image with a paragraph"（全局语义）、"provide the text shown in the image"（文字信息）、"locate the objects in the image, with their descriptions"（空间关系）。每种提示通过 Florence-2 编码器产生不同的 576 token 特征图
    - 设计动机：消融实验显示 OCR 特征去除后影响最大（avg 57.3 vs 完整 58.3），验证了多任务特征的互补性。这也是 Florence-VL 在文档任务上大幅领先的原因

2. **深度维度：底层-高层特征融合**:

    - 功能：保留底层细节信息弥补高层语义特征的不足
    - 核心思路：DaViT 骨干网络的底层特征 $V$ 包含纹理、边缘等低级信息，与编码器输出的高层语义特征互补。直接拼接保留了两个层级的信息
    - 设计动机：消融实验显示仅用底层特征 $[V]$，OCRBench 只有 31.2，加入三种高层特征后跳到 41.4（+32.7%），DocVQA 从 27.9 到 44.5（+59.5%）

3. **通道维度拼接融合策略**:

    - 功能：在保持 token 数量不变的前提下融合多种特征
    - 核心思路：对比了三种融合策略——Token 拼接（沿 token 维度拼接，1728 token，慢）、平均池化（576 token，信息损失）、通道拼接（沿通道维度拼接，576 token，保留所有信息）。通道拼接在同等 token 数下性能最优
    - 设计动机：Token 拼接虽能保留所有信息但 3× token 数带来巨大推理开销；平均池化丢失了任务特异性信息。通道拼接通过线性投影层让 LLM 自行学习如何整合不同通道的信息

### 损失函数 / 训练策略
两阶段训练：预训练阶段在 16.9M 图文描述数据上全参数训练（Florence-2 + 投影 + LLM），使用 64×H100 GPU；指令微调阶段在 10M 指令数据上只训练投影层 + LLM。关键区别：不像 LLaVA 1.5 在预训练时冻结视觉编码器，Florence-VL 全程微调视觉编码器。

## 实验关键数据

### 主实验

| 方法 | 编码器 | Tokens | MMBench | POPE | MM-Vet | TextVQA | OCRBench | DocVQA |
|------|--------|--------|---------|------|--------|---------|----------|--------|
| LLaVA-Next 8B | CLIP | 2880 | 72.2 | 86.6 | 41.7 | - | - | - |
| Cambrian 8B | 多编码器 | 576 | 75.9 | 87.4 | 48.0 | 71.7 | 62.4 | 77.8 |
| **Florence-VL 8B** | Florence-2 | 576 | **76.2** | **89.9** | **56.3** | **74.2** | **63.4** | **84.9** |

### 消融实验

| 配置 | MMBench | POPE | OCRBench | DocVQA | 说明 |
|------|---------|------|----------|--------|------|
| 仅底层 [V] | 64.3 | 86.1 | 31.2 | 27.9 | 底层特征不够 |
| 完整 DBFusion | **66.1** | **89.4** | **41.4** | **44.5** | 深度+广度 |
| 去掉 OCR 特征 | 65.6 | 88.8 | 35.2* | 42.1* | OCR 影响最大 |
| 去掉描述特征 | 64.9 | 89.3 | - | - | MM-Vet 掉 3.4 |
| Token 拼接 (1728) | 66.6 | 88.7 | 40.8 | 44.6 | 3x token 无明显优势 |

### 关键发现
- **Florence-2 与 LLM 的对齐性最好**：量化分析显示 Florence-2 的 alignment loss 低于 CLIP、SigLIP、DINOv2、Stable Diffusion 编码器
- **单编码器超越多编码器**：Florence-VL 8B（单 Florence-2）全面超越 Cambrian 8B（多编码器融合），说明高质量生成式编码器可以替代多编码器组合
- **OCR 特征是文档任务的关键**：在 LLaVA 1.5 设置下，Florence-VL 的 DocVQA 从 28.1 提升到 44.5（+58%），TextVQA 从 58.2 到 62.8（+8%），几乎完全归功于 OCR 提示特征
- **知识类 benchmark 依赖 LLM 而非视觉**：消融显示不同视觉特征组合对 MMMU 等知识类指标影响极小，说明这些任务的瓶颈在 LLM 知识

## 亮点与洞察
- **"生成式编码器替代对比学习编码器"的范式转变**：Florence-2 的多提示特征提取能力是对比学习模型无法实现的，这暗示 VLM 的视觉编码器可能应该从对比学习转向生成式预训练
- **通道拼接融合简洁高效**：和 Cambrian 的复杂 SVA 融合相比，简单的通道拼接 + 线性投影就能达到更好效果，证明融合策略不需要过度设计
- **全参数预训练的重要性**：对比 LLaVA 1.5 冻结编码器的做法，Florence-VL 全程微调编码器带来了显著提升

## 局限与展望
- Florence-2 是 0.23B 的编码器，比 CLIP ViT-L（0.3B）更小，但加上三次前向提取高层特征，实际推理开销约 4× 单次编码
- 三种提示的选择是手动设计的，更多或不同的提示是否能进一步提升未探索
- 仅在 Phi 3.5 和 LLaMA 3 上验证，与其他 LLM 骨干的兼容性未知
- 与最新的高分辨率 VLM（如 InternVL2、Qwen2-VL）在相同分辨率设置下的对比缺失

## 相关工作与启发
- **vs Cambrian**：Cambrian 用 CLIP + DINOv2 + SigLIP 等多编码器融合才达到的性能，Florence-VL 用单个 Florence-2 就超越了，且 token 数相同（576）
- **vs LLaVA-Next**：LLaVA-Next 用 2880 token 才在部分指标上接近 Florence-VL 的 576 token，5× token 差距说明编码器质量比 token 数量更重要
- **vs InternVL2**：InternVL2 使用 InternViT（6B），远大于 Florence-2（0.23B），但两者在很多指标上接近，说明预训练方式（生成式 vs 对比式）可能比模型大小更关键

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统性地将生成式视觉基础模型用于 VLM 编码器，DBFusion 设计直观有效
- 实验充分度: ⭐⭐⭐⭐⭐ 15+ 个 benchmark、详细的融合策略/深度/广度消融、与多编码器方案的公平对比
- 写作质量: ⭐⭐⭐⭐ 方法动机清楚，消融实验组织合理
- 价值: ⭐⭐⭐⭐⭐ 为 VLM 视觉编码器的选择提供了新范式，Florence-2 + DBFusion 可直接应用于其他 VLM 框架

<!-- RELATED:START -->

## 相关论文

- [Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder](../../ACL2025/multimodal_vlm/exploring_how_generative_mllms_perceive_more.md)
- [SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models](spa-vl_a_comprehensive_safety_preference_alignment_dataset_for_vision_language_m.md)
- [VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge](vila-m3_enhancing_vision-language_models_with_medical_expert_knowledge.md)
- [EVEv2: Improved Baselines for Encoder-Free Vision-Language Models](../../ICCV2025/multimodal_vlm/evev2_improved_baselines_for_encoderfree_visionlanguage_mode.md)
- [SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](../../NeurIPS2025/multimodal_vlm/sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)

<!-- RELATED:END -->
