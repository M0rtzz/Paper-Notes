---
title: >-
  [论文解读] GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models
description: >-
  [ICLR 2026][图像生成][地理多样性] 提出 GeoDiv 框架，利用 LLM 和 VLM 的世界知识，从社会经济视觉指数（SEVI）和视觉多样性指数（VDI）两个维度系统评估 T2I 模型的地理多样性，揭示了模型对印度、尼日利亚等国家存在系统性贫困化偏见。
tags:
  - ICLR 2026
  - 图像生成
  - 地理多样性
  - 文本到图像模型
  - 社会经济偏见
  - VLM评估
  - 可解释指标
---

# GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models

**会议**: ICLR 2026  
**arXiv**: [2602.22120](https://arxiv.org/abs/2602.22120)  
**代码**: [GitHub](https://github.com/moha23/geodiv)  
**领域**: 文本到图像生成 / 公平性评估  
**关键词**: 地理多样性, 文本到图像模型, 社会经济偏见, VLM评估, 可解释指标

## 一句话总结
提出 GeoDiv 框架，利用 LLM 和 VLM 的世界知识，从社会经济视觉指数（SEVI）和视觉多样性指数（VDI）两个维度系统评估 T2I 模型的地理多样性，揭示了模型对印度、尼日利亚等国家存在系统性贫困化偏见。

## 研究背景与动机
- **领域现状**: T2I 模型（如 Stable Diffusion、FLUX.1）在商业中广泛应用，但其生成结果常缺乏地理多样性，对不同地区的描绘存在刻板印象
- **现有痛点**: 现有多样性指标要么依赖标注数据集（如 GeoDE），要么仅关注低层视觉相似性（如 Vendi-Score），无法解释性地捕获地理多样性的多维度特征
- **核心矛盾**: 地理多样性涵盖经济、环境、文化等多维度变化，单一指标无法全面衡量，且现有方法在国家级别的精细偏见检测上能力有限
- **切入角度**: 利用 LLM/VLM 的隐含世界知识，设计可解释的自动化评估框架
- **核心 idea**: 将地理多样性分解为 SEVI（富裕度+维护状态）和 VDI（实体外观+背景外观）四个可解释维度，用 Hill Number 量化多样性

## 方法详解

### 整体框架
GeoDiv 流水线：给定实体 $e$ 和国家 $c$，LLM 生成与 $e$ 相关的属性问题和固定的背景问题；VQA 模型对图像集预测答案分布；基于分布计算 VDI（标准化 Hill Number）；同时 VQA 模型对每张图像评分得到 SEVI。

### 关键设计

1. **视觉多样性指数 (VDI)**:

    - 功能：评估图像在实体外观和背景外观两个轴上的视觉变化
    - 核心思路：用多个 LLM 集合生成属性问题-答案对，VQA 模型预测答案分布，用标准化 Hill Number 量化
    - 多样性评分：$\text{Diversity-Score} = \frac{\exp(H(\hat{P_k})) - 1}{|\hat{\mathcal{A}_k}| - 1}$，其中 $H(\cdot)$ 为 Shannon 熵
    - 设计动机：不同问题答案数量不同，标准化后可公平比较

2. **社会经济视觉指数 (SEVI)**:

    - 功能：捕获图像的 Affluence（富裕度，1-5 分）和 Maintenance（维护状态，1-5 分）
    - 核心思路：VLM 直接评分，分布上再用 Hill Number 计算多样性
    - 设计动机：将社会经济指标与视觉分析结合，确保主观概念的一致性

3. **可靠性保障机制**:

    - Visibility Step：过滤掉属性不可见的图像，减少 VQA 幻觉
    - Multi-Select：允许多选答案，避免强制单选的分布扭曲
    - NOTA 选项：附加"以上都不是"选项（仅 2.6% 被选），减少猜测
    - 大规模人工验证：14 个国家的本地标注者验证 SEVI 与人工判断的一致性

### 训练/评估策略
- 使用 Gemini-2.5-flash 作为 VQA/VLM（最优精度86%，人工相关系数 ρ=0.76/0.69）
- 评估了 4 个 T2I 模型、10 个实体、16 个国家，共 160,000 张合成图像

## 实验关键数据

### 主实验

| VQA 模型 | VDI 实体精度 | VDI 背景精度 | SEVI-Affluence ρ | SEVI-Maintenance ρ |
|----------|-------------|-------------|-------------------|---------------------|
| Gemini-2.5-flash | 0.87 | 0.85 | 0.76 | 0.69 |
| gpt-4o | 0.85 | 0.81 | 0.76 | 0.76 |
| Qwen2.5-VL | 0.85 | 0.77 | 0.69 | 0.71 |
| LLaVA-v1.6 | 0.70 | 0.66 | 0.65 | 0.68 |

### 关键发现：国家级偏见

| 国家组 | 平均 Affluence | 平均 Maintenance | 多样性分数 |
|--------|---------------|------------------|-----------|
| 印度/尼日利亚/哥伦比亚 | 2.31 | 3.34 | 低 |
| 日本/阿联酋/英国 | 3.53 | 4.30 | 低 |
| FLUX.1 全局 | 3.82 | 4.73 | 极低(0.15) |

### 关键发现
- FLUX.1 生成最精致的图像但多样性最低，揭示了"精致"与"多样"间的权衡
- 新模型版本的整体地理多样性反而在下降
- 背景多样性（0.31）远低于实体多样性（0.44），山脉仅在 12% 图像中出现
- 与 Vendi-Score 对比：仅实体多样性有中等相关（ρ=0.56），其余维度低

## 亮点与洞察
- 首个系统化、可解释的 T2I 地理多样性评估框架，支持任意实体和国家扩展
- 发现 FLUX.1 的"高质量低多样性"权衡，对模型开发有直接指导意义
- 开源了全部数据、标注和代码

## 局限与展望
- 仅覆盖 16 国 10 实体，扩展到更多地区可能揭示新偏见模式
- 依赖 LLM/VLM 的世界知识，其本身可能带有偏见
- 文化表征方面仍有局限，注释者与 VQA 模型在某些国家不一致

## 相关工作与启发
- **vs Vendi-Score**: VendiScore 仅衡量视觉变化，无法捕获社会经济维度
- **vs GRADE**: GRADE 仅评估日常物品多样性，未涉及地理维度的复杂性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将地理多样性分解为 SEVI+VDI 四维度评估
- 实验充分度: ⭐⭐⭐⭐⭐ 160K 图像、大规模人工验证、多模型多国家对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现有洞察力
- 价值: ⭐⭐⭐⭐ 对 T2I 公平性评估有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [The Intricate Dance of Prompt Complexity, Quality, Diversity, and Consistency in T2I Models](the_intricate_dance_of_prompt_complexity_quality_diversity_and_consistency_in_t2.md)
- [How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions](../../AAAI2026/image_generation/how_bias_binds_measuring_hidden_associations_for_bias_control_in_text-to-image_c.md)
- [AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models](../../CVPR2026/image_generation/autodebias_automated_framework_for_debiasing_text-to-image_models.md)
- [Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)
- [Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)

<!-- RELATED:END -->
