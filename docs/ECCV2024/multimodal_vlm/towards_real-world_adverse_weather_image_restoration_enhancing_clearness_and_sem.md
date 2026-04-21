---
title: >-
  [论文解读] Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models
description: >-
  [ECCV 2024][多模态][恶劣天气图像复原] 本文提出WResVLM半监督学习框架，利用视觉-语言模型（VLM）为真实恶劣天气图像提供清晰度评估和语义描述监督信号，通过VLM图像评估+天气提示学习增强清晰度、描述辅助的语义正则化增强语义，在真实去雨/去雾/去雪任务上全面超越现有方法。
tags:
  - ECCV 2024
  - 多模态
  - 恶劣天气图像复原
  - 视觉-语言模型
  - 半监督学习
  - 伪标签
  - 天气提示学习
---

# Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models

**会议**: ECCV 2024  
**arXiv**: [2409.02101](https://arxiv.org/abs/2409.02101)  
**代码**: [GitHub](https://github.com/jiaqixuac/WResVLM) (有)  
**领域**: 多模态VLM / 图像复原  
**关键词**: 恶劣天气图像复原, 视觉-语言模型, 半监督学习, 伪标签, 天气提示学习

## 一句话总结
本文提出WResVLM半监督学习框架，利用视觉-语言模型（VLM）为真实恶劣天气图像提供清晰度评估和语义描述监督信号，通过VLM图像评估+天气提示学习增强清晰度、描述辅助的语义正则化增强语义，在真实去雨/去雾/去雪任务上全面超越现有方法。

## 研究背景与动机

恶劣天气（雨、雾、雪）严重影响户外视觉系统的性能。现有图像复原方法主要针对单一天气类型设计，近年来All-in-One方法（TransWeather、WeatherDiff等）试图用单一模型处理多种天气条件。

**现有痛点**：

**域差距问题**：现有方法主要在合成数据集上训练，应用到真实场景时泛化能力严重不足。合成数据与真实天气退化之间存在显著的分布差异。

**语义缺失问题**：现有方法主要关注视觉清晰度的恢复，忽略了场景的语义上下文。这导致复原结果对下游高级视觉任务（如检测、分割）帮助有限。

**核心矛盾**：真实恶劣天气图像没有对应的干净图像作为ground truth，无法直接进行监督学习。如何利用无标注的真实数据来训练复原模型？

**切入角度**：借助大规模视觉-语言模型（VLM）的能力——它们在大量天气相关图像上训练，具备评估图像清晰度和理解场景语义的能力。因此可以用VLM为真实数据提供伪监督信号，构建半监督学习框架。

## 方法详解

### 整体框架
WResVLM是一个半监督学习框架，同时使用有标注的合成数据和无标注的真实数据训练All-in-One图像复原模型。框架核心包括两个VLM增强模块：(1) **清晰度增强**——通过VLM图像评估选择伪标签 + 天气提示学习调制复原过程；(2) **语义增强**——通过VLM生成的描述进行语义正则化。骨干网络采用MSBDN。

### 关键设计

1. **VLM图像评估与伪标签生成**:

    - 功能：利用VLM的零样本能力评估复原图像的天气可见度质量，选择最优伪标签
    - 核心思路：设计天气相关的图像质量问答模板，提示VLM对图像进行评分。使用五级MOS评分（excellent-good-fair-poor-bad），通过softmax将VLM对五个评级token的logits转换为数值分数 $r^{vlm} = \sum_{i=1}^{5} i \times p_i$
    - 设计动机：现有图像质量评估方法（NIMA、MUSIQ等）主要针对技术性失真（噪声、模糊），而非天气相关的可见度退化。VLM因见过大量天气图像，能更准确地判断天气退化程度
    - 伪标签更新：训练过程中持续比较模型预测与当前伪标签的VLM评分，若预测更优则更新伪标签数据库

2. **天气提示学习（Weather Prompt Learning）**:

    - 功能：学习CLIP空间中的天气状态表示，引导复原模型生成更清晰的图像
    - 核心思路：分两阶段——(a) 提示嵌入学习：冻结CLIP参数，学习四个天气提示向量 $\{t_c, t_r, t_h, t_s\}$（晴/雨/雾/雪），通过交叉熵分类损失优化；(b) 复原模型优化：最大化复原图像嵌入与"晴天"提示的余弦相似度
    - 损失函数：$\mathcal{L}_{wpl} = \frac{e^{cos(\mathcal{E}_I(\hat{y}), \mathcal{E}_T(t_c))}}{\sum_{t} e^{cos(\mathcal{E}_I(\hat{y}), \mathcal{E}_T(t))}}$
    - 补充正则化：仅用天气提示损失会产生噪声伪影，额外引入特征相似度损失 $\mathcal{L}_{feat}$（使用Depth Anything提取特征），对齐预测与伪标签/输入的特征

3. **描述辅助的语义增强**:

    - 功能：利用VLM生成天气图像的场景描述，通过改变天气描述但保持语义内容不变来提供语义正则化
    - 核心思路：使用LLaVA为输入图像生成包含天气描述的negative caption（如"A person is walking in heavy rain..."），再用LLM将其转换为positive caption（"The weather looks good. A person is walking..."）。两个描述场景内容相同但天气状态不同
    - 损失函数：$\mathcal{L}_{sem} = \frac{e^{cos(\mathcal{E}_I(\hat{y}), \mathcal{E}_T(d_{pos}))}}{\sum_{d} e^{cos(\mathcal{E}_I(\hat{y}), \mathcal{E}_T(d))}}$，鼓励复原图像与"好天气"描述对齐
    - 设计动机：与天气提示不同，这些描述是针对每张具体图像的，提供了图像级别的语义监督

4. **训练策略**:

    - 功能：伪标签初始化 + 迭代更新
    - 伪标签初始化：收集多个现有天气专用和All-in-One复原方法的输出，使用多个VLM专家投票选出最佳伪标签
    - 迭代更新：分多轮训练（经验设4轮），每轮使用一个VLM在线评估，轮间用全部VLM重新评估并更新伪标签、天气提示和描述

### 损失函数 / 训练策略
总损失为五项加权组合：
$$\mathcal{L} = \mathcal{L}_{sup} + w_1 \mathcal{L}_{ps} + w_2 \mathcal{L}_{wpl} + w_3 \mathcal{L}_{sem} + w_4 \mathcal{L}_{feat}$$
权重设置：$w_1=0.5, w_2=0.2, w_3=0.05, w_4=0.2$

其中 $\mathcal{L}_{sup}$ 是合成数据上的有监督外观损失，$\mathcal{L}_{ps}$ 是伪标签损失（L1），$\mathcal{L}_{wpl}$ 是天气提示学习损失，$\mathcal{L}_{sem}$ 是语义正则化损失，$\mathcal{L}_{feat}$ 是特征相似度损失。

## 实验关键数据

### 主实验（真实天气图像，无参考指标）

| 方法 | NIMA↑ | MUSIQ↑ | CLIP-IQA↑ | 整体排名 |
|------|-------|--------|-----------|---------|
| Restormer | 4.992 | 56.38 | 0.438 | 第3 |
| TransWeather | 4.904 | 52.24 | 0.355 | 第7 |
| PromptIR | 5.009 | 56.07 | 0.443 | 第2 |
| DA-CLIP | 5.010 | 55.59 | 0.412 | 第4 |
| **WResVLM (Ours)** | **5.084** | **59.34** | **0.456** | **第1** |

| 方法 | LIQE↑ | Q-Align↑ | VLM-Vis↑ | 整体排名 |
|------|-------|----------|----------|---------|
| Restormer | 2.456 | 3.503 | 0.343 | 第3 |
| PromptIR | 2.437 | 3.491 | 0.343 | 第2 |
| DA-CLIP | 2.438 | 3.480 | 0.346 | 第4 |
| **WResVLM (Ours)** | **2.640** | **3.574** | **0.387** | **第1** |

### 消融实验

| 配置 | MUSIQ↑ | CLIP-IQA↑ | VLM-Vis↑ | 说明 |
|------|--------|-----------|----------|------|
| $\mathcal{L}_{sup}$ 仅合成监督 | 53.41 | 0.388 | 0.343 | 基线 |
| + $\mathcal{L}_{ps}$ 伪标签 | 54.08 | 0.396 | 0.354 | 简单半监督 |
| + $r^{vlm}$ VLM评估 | 56.68 | 0.429 | 0.366 | VLM选伪标签 |
| + init 伪标签初始化 | 57.34 | 0.425 | 0.370 | 多方法初始化 |
| + $\mathcal{L}_{wpl}$ 天气提示 | 58.13 | 0.437 | 0.376 | CLIP引导清晰度 |
| + $\mathcal{L}_{sem}$ 语义正则 | 58.91 | 0.445 | 0.381 | 描述辅助语义 |
| + iter 迭代更新 | **59.34** | **0.456** | **0.387** | 完整框架 |

### 关键发现
- VLM图像评估（$r^{vlm}$）是最大的提升来源，将MUSIQ从54.08提升到56.68（+2.60）
- 每个组件都有正向贡献，且各组件之间存在协同效应
- 在用户研究中，WResVLM在可见度和图像质量两个维度上均优于所有对比方法
- 使用VLM评估选择伪标签优于使用NIMA/MUSIQ/CLIP-IQA等传统质量评估指标
- 语义正则化能帮助模型处理微妙的天气语境差异（如将"foggy"变为"overcast"）

## 亮点与洞察
- 首次系统性地将VLM引入恶劣天气图像复原领域，既用于清晰度评估也用于语义引导
- 天气提示学习的设计巧妙——利用CLIP已有的天气概念知识，仅学习轻量级提示向量
- 将VLM的高级语义能力"蒸馏"到底层图像复原任务中，是一个有趣的跨层级能力迁移
- 多VLM专家投票机制缓解了单一VLM的偏见问题
- 负-正描述转换的思路（保持场景语义、改变天气状态）设计巧妙

## 局限与展望
- 使用大型VLM进行在线评估的计算开销仍然较高
- 真实数据收集规模有限（每种天气约2000-2400张），可能影响方法的泛化性
- 骨干网络MSBDN相对较老，换用更强backbone可能获得更大提升
- 仅处理雨、雾、雪三种天气，未涵盖沙尘暴、低光照等其他恶劣条件
- 评估完全依赖无参考指标，缺少真实场景下游任务（检测、分割）的端到端验证

## 相关工作与启发
- 与PromptIR（退化条件提示）和DA-CLIP（退化信息对比学习）形成互补——前者在模型内部加提示，本文在训练信号层面利用VLM
- 与CLIP-IQA/Q-Bench等低级视觉VLM应用相关，但本文将VLM应用拓展到了训练阶段的伪监督生成
- 启发：VLM可以作为通用的"质量裁判"和"语义桥梁"，帮助解决缺少标注数据的底层视觉任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 将VLM多维度引入天气复原的半监督框架具有创新性
- 实验充分度: ⭐⭐⭐⭐ 消融实验系统完整，但缺少下游任务验证
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，图表配合良好
- 价值: ⭐⭐⭐⭐ 为真实场景图像复原提供了有前景的VLM辅助训练范式

<!-- RELATED:START -->

## 相关论文

- [AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)
- [MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](marvelovd_marrying_object_recognition_and_visionlanguage_mod.md)
- [Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_vision-language_models.md)
- [FlexAttention for Efficient High-Resolution Vision-Language Models](flexattention_for_efficient_highresolution_visionlanguage_mo.md)
- [CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios](cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)

<!-- RELATED:END -->
