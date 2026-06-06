---
title: >-
  [论文解读] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow
description: >-
  [CVPR 2026][多模态VLM][视觉语言模型] 本文发现 VLM 中文本 token 对无关视觉 token 的过度注意力是"看到但感知错误"的根本原因，提出基于 token 动态熵的自适应信息流调控方法（AIF），通过推理时修改因果掩码来阻断无关视觉-文本连接，免训练提升多种 VLM 的感知能力。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "视觉语言模型"
  - "信息流调控"
  - "token动态"
  - "因果掩码"
  - "免训练"
---

# Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow

**会议**: CVPR 2026  
**arXiv**: [2604.15809](https://arxiv.org/abs/2604.15809)  
**代码**: [https://cxliu0.github.io/AIF/](https://cxliu0.github.io/AIF/)  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 信息流调控, token动态, 因果掩码, 免训练

## 一句话总结

本文发现 VLM 中文本 token 对无关视觉 token 的过度注意力是"看到但感知错误"的根本原因，提出基于 token 动态熵的自适应信息流调控方法（AIF），通过推理时修改因果掩码来阻断无关视觉-文本连接，免训练提升多种 VLM 的感知能力。

## 研究背景与动机

**领域现状**：视觉语言模型（VLM）如 LLaVA、Qwen2.5-VL 在视觉问答、OCR、目标定位等广泛任务上展现了强大能力。

**现有痛点**：近期研究发现 VLM 存在"看到但感知不对"的问题——模型能正确捕获与问题相关的图像区域，但最终输出错误答案。已有的改进方法要么需要重训练（计算资源大），要么依赖视觉裁剪（推理时间大幅增加且对计数/关系推理无效）。

**核心矛盾**：VLM 解码过程中的信息流不够优化：文本 token 的交叉注意力分散在大量无关背景视觉 token 上，形成空间弥散的注意力模式，引入了噪声视觉信息干扰了正确推理。

**本文目标**：通过推理时信息流调控来改善 VLM 的感知能力，无需任何训练。

**切入角度**：观察到与目标区域对应的视觉 token 在 LLM 特定层中展现出独特的激活模式（高活跃度），而无关区域的 token 激活模式不规则。利用这种"token 动态"差异来识别重要 token。

**核心 idea**：文本 token 只需要与重要视觉 token 交互，通过修改因果掩码阻断无关视觉 token 到文本 token 的信息流，同时保留视觉 token 之间的信息流以确保图像信息不丢失。

## 方法详解

### 整体框架

AIF 方法分三步：（1）一步 LLM 解码收集每个视觉 token 的跨层激活统计（token 动态）；（2）基于 token 动态计算每个视觉 token 的重要性熵，生成 token 熵图；（3）自适应选择掩码阈值，修改因果掩码阻断无关 token 到文本 token 的连接。整个过程仅需一次额外解码步骤，后续推理过程不变。

### 关键设计

1. **Token 动态分析与重要性度量**:

    - 功能：量化每个视觉 token 对文本推理的重要性
    - 核心思路：定义 token 动态 $\mathcal{D}_{v_i} = \{d_{v_i}^l\}_{l=1}^L$，其中 $d_{v_i}^l = \max_j a_{i,j}^l$ 表示第 $i$ 个视觉 token 在第 $l$ 层对所有文本 token 的最大注意力值。计算平均值 $\mu_{v_i}$ 和基于动态的熵 $\text{Ent}_{v_i} = \sum -\frac{d_{v_i}^l}{L \cdot \mu_{v_i}} \log(\frac{d_{v_i}^l}{L \cdot \mu_{v_i}})$。高熵意味着激活模式随机（不重要），低熵意味着在特定层集中激活（重要）
    - 设计动机：目标区域的视觉 token 在关键层强烈激活（低熵），无关区域在各层随机激活（高熵），熵自然区分了两者

2. **自适应信息流调控（Adaptive Information Flow Modulation）**:

    - 功能：自动确定最优掩码比例并修改因果掩码
    - 核心思路：按熵值排序视觉 token，尝试不同掩码比例（0.1 到 0.9），计算保留 token 的注意力分布熵 $S$。选择使 $S$ 与原始分布熵 $S_0$ 差距最大的掩码比例——这意味着注意力分布从弥散变为集中。被掩码的视觉 token 与文本 token 之间的连接在因果掩码中被阻断（设为 $-\infty$）
    - 设计动机：不同图像和问题需要不同的掩码比例，自适应选择避免了超参数调优。最大化注意力集中度变化确保模型聚焦在真正相关的区域

3. **保留视觉-视觉信息流**:

    - 功能：确保掩码过程不丢失任何图像信息
    - 核心思路：只阻断被掩码视觉 token 到文本 token 的连接，保留被掩码 token 之间以及与其他视觉 token 之间的注意力。这与视觉 token 剪枝（完全移除 token）有本质区别
    - 设计动机：如果完全移除视觉 token，可能丢失上下文信息。保留视觉-视觉信息流让其他 token 仍然能利用被掩码 token 的信息

### 损失函数 / 训练策略

完全免训练的推理时方法。仅需一次额外解码步骤获取 token 动态并生成掩码，后续推理过程与标准方法完全一致。

## 实验关键数据

### 主实验

| 方法 | V* | RealWorldQA | MMStar | TextVQA | CountBench |
|------|-----|------------|--------|---------|------------|
| LLaVA-1.5-7B | 42.4 | 55.6 | 33.1 | 47.8 | 47.0 |
| **+ AIF** | **50.3 (+7.9)** | **60.5 (+4.9)** | **39.5 (+6.4)** | **49.9 (+2.1)** | **50.1 (+3.1)** |
| Qwen2.5-VL-7B | 78.5 | 68.5 | 63.9 | 84.9 | 87.1 |
| **+ AIF** | **84.8 (+6.3)** | **74.5 (+6.0)** | **70.9 (+7.0)** | **86.0 (+1.1)** | **89.5 (+2.4)** |

### 视觉定位实验

| 方法 | RefCOCO 平均 | RefCOCO+ 平均 | RefCOCOg 平均 |
|------|-------------|-------------|--------------|
| Qwen2.5-VL-7B | 89.3 | 80.1 | 87.2 |
| **+ AIF** | **91.4 (+2.1)** | **82.7 (+2.6)** | **89.5 (+2.3)** |

### 关键发现

- 90% 的低 $\mu_{v_i}$ token 被掩码后性能几乎不变，但仅掩码 10% 的高 $\mu_{v_i}$ token 就导致 50%+ 的性能下降，证实了只有少量视觉 token 对输出有显著影响
- AIF 在 Qwen2.5-VL-7B 上的 MMStar 提升 7.0，使其超过 GPT-4o 和 Qwen2.5-VL-72B 在部分指标上的表现
- 视觉定位任务上 AIF 甚至超越了专门的定位模型 Grounding-DINO-L
- Oracle 实验表明信息流调控的潜在提升上限更高（RealWorldQA: 55.6→61.6）

## 亮点与洞察

- **信息流作为控制信号**：之前的工作主要将注意力分析用于诊断和解释，本文首次将其作为控制信号来主动改善模型性能。这个视角转变极具启发性
- **免训练+即插即用**：仅通过修改因果掩码就能持续提升多种 VLM 和多种任务的性能，体现了信息流调控的普适价值
- **保留视觉-视觉流的设计哲学**：与暴力 token 剪枝的关键区别，确保了信息完整性

## 局限与展望

- 需要一次额外的解码步骤获取 token 动态，虽然开销较小但非零
- 自适应阈值选择需要尝试多个候选比例，可能不是最优的选择策略
- 对于需要全局理解的任务（如场景描述），过度掩码可能反而有害
- 仅在 LLaVA-1.5 和 Qwen2.5-VL 上验证，更多模型的泛化性有待确认

## 相关工作与启发

- **vs ViCrop**: ViCrop 通过裁剪放大相关区域来增强细粒度感知，但推理时间大幅增加且对关系推理无效；AIF 通过掩码修改实现，开销小且对多种任务有效
- **vs 视觉 token 剪枝**: 剪枝完全移除 token 导致信息丢失，AIF 只阻断到文本的连接保留了完整的视觉信息
- **vs Pei et al. / Wang et al.**: 这些工作从架构或训练层面修改注意力模式，需要重训；AIF 仅推理时修改

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 信息流调控作为免训练改进 VLM 的新范式，视角独特
- 实验充分度: ⭐⭐⭐⭐⭐ VQA、OCR、定位、计数、幻觉等多任务全覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 从发现到分析到方法逻辑链完整，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ 免训练显著提升 VLM 性能，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](modix_positional_index_scaling.md)
- [\[NeurIPS 2025\] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](../../NeurIPS2025/multimodal_vlm/flowcut_rethinking_redundancy_via_information_flow_for_effic.md)
- [\[ACL 2026\] Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding](../../ACL2026/multimodal_vlm/revisit_what_you_see_revealing_visual_semantics_in_vision_tokens_to_guide_lvlm_d.md)
- [\[ICML 2026\] VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](../../ICML2026/multimodal_vlm/interpreting_and_enhancing_emotional_circuits_in_large_vision-language_models_vi.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](../../ACL2026/multimodal_vlm/i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)

</div>

<!-- RELATED:END -->
