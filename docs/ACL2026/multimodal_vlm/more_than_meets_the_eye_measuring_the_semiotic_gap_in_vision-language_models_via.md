---
title: >-
  [论文解读] More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage
description: >-
  [ACL 2026][多模态VLM][视觉语言模型] 本文从认知符号学角度揭示 VLM 的"字面优越偏差"——模型在高保真图像上倾向于字面解读而非隐喻/习语理解，通过引入 DIVA 基准（图标化简化图像）和 Semantic Alignment Gap 指标…
tags:
  - "ACL 2026"
  - "多模态VLM"
  - "视觉语言模型"
  - "符号学鸿沟"
  - "字面偏差"
  - "图标化抽象"
  - "名词复合词"
---

# More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage

**会议**: ACL 2026  
**arXiv**: [2604.17354](https://arxiv.org/abs/2604.17354)  
**代码**: [GitHub](https://github.com/risehnhew/More-than-meets-the-eye)  
**领域**: 多模态VLM / 符号理解  
**关键词**: 视觉语言模型, 符号学鸿沟, 字面偏差, 图标化抽象, 名词复合词

## 一句话总结

本文从认知符号学角度揭示 VLM 的"字面优越偏差"——模型在高保真图像上倾向于字面解读而非隐喻/习语理解，通过引入 DIVA 基准（图标化简化图像）和 Semantic Alignment Gap 指标，证明降低视觉保真度能显著缩小字面与习语解读之间的鸿沟。

## 研究背景与动机

**领域现状**：文本到图像模型已能生成高度逼真的图像，VLM 也能很好地解码图像的字面内容。但在理解抽象含义（如习语、比喻）方面，VLM 仍然存在根本性的认知鸿沟。

**现有痛点**：(1) 现有 VL 基准主要关注字面意义的视觉-文本对齐（物体检测、属性绑定等），对比喻意义评估不足；(2) 名词复合词（如 "Eye Candy"）的视觉表示需要从字面图标性转向习语象征性，但模型常常被高保真视觉细节所误导；(3) 缺乏跨架构一致的评估指标——判别式模型用余弦相似度、生成式模型用 token 概率、闭源模型只能用行为探测。

**核心矛盾**：VLM 的预训练目标过度优化了物理重建和视觉仿真（Iconicity），导致模型在面对需要抽象/象征理解的任务时，高保真视觉细节反而成为"认知干扰"——模型看到"Eye"就只想到眼睛，而非 "Eye Candy" 的隐喻含义。

**本文目标**：(1) 量化 VLM 的字面偏差程度；(2) 验证"降低视觉保真度能否提升象征理解"的假说；(3) 提供跨架构的统一评估框架。

**切入角度**：从符号学理论出发——图标（icon）通过相似性传递意义，符号（symbol）通过约定传递意义。文本天然是符号性的，但图像通常是图标性的。当图像的图标性（高保真细节）太强时，模型会被锁定在字面解读上。

**核心 idea**：通过"图标化抽象"（Iconographic Abstraction）——系统性降低图像的视觉保真度（去纹理、去光影、简化构图），将图像从"现实仿真"转变为"意义符号"，从而释放模型的象征理解潜能。

## 方法详解

### 整体框架

DIVA 基准的构建流程：(1) 从 SemEval-2025 AdMIRe 任务获取 100 个英语名词复合词的字面和习语高保真图像；(2) 使用 Gemini 生成对应的图标化（低保真、示意图式）图像，每个复合词生成 5 种对比图像（高习语、高字面、弱习语、弱字面、干扰项）；(3) 3 名标注员进行人工验证。评估时，对每个 NC 计算模型在字面图和习语图上的语义匹配分数差异。

### 关键设计

1. **Semantic Alignment Gap (Δ) 和 Signed Literal Bias (b)**:

    - 功能：统一量化不同架构 VLM 的字面-习语鸿沟大小和方向
    - 核心思路：对每个名词复合词 $t$，计算模型对字面图 $v_{lit}$ 和习语图 $v_{id}$ 的语义匹配分数差异。$b(t) = \mathcal{S}(v_{lit}, t) - \mathcal{S}(v_{id}, t)$ 衡量方向（正为字面偏好），$\Delta(t) = |b(t)|$ 衡量强度。$\mathcal{S}$ 根据模型架构有三种实现
    - 设计动机：现有评估要么限于特定架构，要么不区分方向和强度。Δ 作为模型内部的相对度量，允许在同一架构族内进行有意义的趋势分析

2. **三模态评分函数 (Tri-fold Scoring)**:

    - 功能：使 Δ 指标适用于判别式、开源生成式和闭源模型
    - 核心思路：(i) 判别式模型（CLIP/SigLIP）用嵌入空间的余弦相似度；(ii) 开源生成式模型（LLaVA/InternVL）用强制 Yes/No 回答的 token 概率（LID）；(iii) 闭源模型（GPT-5/Claude）用自报告置信度分数 $\gamma \in [0,100]$，并用 10 次重复强制选择的行为频率作为验证
    - 设计动机：不同架构的"置信度信号"获取方式完全不同，统一指标需要适应这种异质性。三种实现允许在各自范式内进行一致的趋势分析

3. **Iconographic Abstraction Pipeline**:

    - 功能：将高保真图像转换为低保真图标化图像，保持语义核心同时去除视觉噪声
    - 核心思路：使用 Gemini 进行两阶段处理——语义蒸馏（保留意图含义，去除偶然场景细节）和几何重建（约束为平面、低细节的图标风格）。生成后由人工标注员验证语义保持和风格约束
    - 设计动机：基于"语义锚定"理论——当视觉信号变得更"数字化"（非模拟），模型更不容易默认字面解读，更倾向采取符号立场

### 损失函数 / 训练策略

本文不涉及模型训练，是纯评估性工作。DIVA 包含 1,000 张图标化图像（100 个 NC × 5 种对比 × 2 个语义方向）。

## 实验关键数据

### 主实验

| 模型 | Δ (AdMIRe/高保真) | Δ (DIVA/图标化) | Δ 降低 |
|------|------------------|----------------|--------|
| SigLIP 2 | 0.245 | 0.178 | -27% |
| EVA-CLIP-18B | 0.262 | 0.191 | -27% |
| InternVL3-78B | 0.138 | 0.089 | -36% |
| Qwen2.5-VL-32B | 0.145 | 0.095 | -34% |
| LLaVA-OV-7B | 0.176 | 0.122 | -31% |
| GPT-5 | 0.065 | 0.021 | -68% |
| Claude 4.5 Sonnet | 0.072 | 0.028 | -61% |

### 消融实验

| 分析维度 | 结果 |
|----------|------|
| 判别式 vs 生成式 | 判别式模型 Δ 最大（~0.25），生成式显著更小（~0.14），闭源最小（~0.07） |
| 模型规模效应 | 同架构内，更大模型不一定更小 Δ——规模无法自动解决字面偏差 |
| 5-way 选择准确率 | 图标化图像在所有模型族上均提升准确率（判别式 42.3→58.7%，闭源 78.5→91.3%） |

### 关键发现

- 所有模型在所有条件下均表现出正 $b(t)$（字面偏好），且在高保真图像上更为严重
- 图标化抽象在所有架构族内一致降低 Δ——GPT-5 从 0.065 降到 0.021，接近零偏差
- 判别式模型受"认知干扰"最严重——CLIP 类模型过度依赖纹理和表面特征
- Spearman 相关分析显示人类评估与 Δ 指标高度一致（ρ=0.64-0.73）

## 亮点与洞察

- 从符号学理论切入 VLM 评估是一个非常新颖的角度——将"为什么模型不懂比喻"转化为可量化的"图标-符号连续体"上的位置测量
- "高保真度是认知干扰"这一反直觉发现极有启发性——更逼真的图像不一定有利于理解，这挑战了"图像越清晰越好"的隐含假设
- Δ 指标的三模态设计巧妙地解决了跨架构评估的可比性问题

## 局限与展望

- 仅限于英语名词复合词，未涉及跨文化隐喻（如中文"铁饭碗"）
- 图标化图像的特定风格（扁平设计）可能引入风格偏差——模型可能因熟悉特定风格而表现更好
- 闭源模型的自报告置信度可能反映指令跟随倾向而非真实语义判断
- 仅作为诊断工具，未提出如何改进模型的方法

## 相关工作与启发

- **vs T2I-CompBench/GenEval**: 这些基准关注物理组合性（红方块旁蓝球），本文关注语义组合性——名词组合产生超越字面的抽象意义
- **vs AdMIRe (SemEval-2025)**: AdMIRe 评估模型能否对齐习语图像，但使用高保真图像可能引入混淆因素；DIVA 通过图标化控制了视觉复杂度
- **vs IconQA**: IconQA 使用图标式图表进行推理，但不涉及比喻理解

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 符号学视角 + 图标化抽象假说 + 跨架构统一指标，高度原创
- 实验充分度: ⭐⭐⭐⭐ 8个模型、三种架构范式、人类评估验证，但仅限英语名词复合词
- 写作质量: ⭐⭐⭐⭐⭐ 理论框架优雅，方法论严谨，论述清晰
- 价值: ⭐⭐⭐⭐ 深刻揭示了 VLM 的字面偏差问题，但缺少改进方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] More than the Sum: Panorama-Language Models for Adverse Omni-Scenes](../../CVPR2026/multimodal_vlm/more_than_the_sum_panorama-language_models_for_adverse_omni-scenes.md)
- [\[AAAI 2026\] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis](../../AAAI2026/multimodal_vlm/patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision_language_models.md)
- [\[ACL 2026\] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
