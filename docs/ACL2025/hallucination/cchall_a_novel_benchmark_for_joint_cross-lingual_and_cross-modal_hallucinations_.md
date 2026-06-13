---
title: >-
  [论文解读] CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models
description: >-
  [ACL2025][幻觉检测][跨语言] 提出首个**联合跨语言与跨模态**幻觉检测基准 CCHall，覆盖 9 种语言和 4 类多模态数据集，系统评估 6 款主流 MLLM 在联合场景下的幻觉表现，揭示当前模型在该联合场景中 F1 比单独跨模态低 10.9、比单独跨语言低 3.4，且提出多语提示和外部工具辅助两条缓解路径。
tags:
  - "ACL2025"
  - "幻觉检测"
  - "跨语言"
  - "跨模态"
  - "benchmark"
  - "多模态大模型"
---

# CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models

**会议**: ACL2025  
**arXiv**: [2505.19108](https://arxiv.org/abs/2505.19108)  
**代码**: [GitHub](https://github.com/BRZ911/CCHall)  
**领域**: 幻觉检测  
**关键词**: 幻觉检测, 跨语言, 跨模态, benchmark, 多模态大模型

## 一句话总结

提出首个**联合跨语言与跨模态**幻觉检测基准 CCHall，覆盖 9 种语言和 4 类多模态数据集，系统评估 6 款主流 MLLM 在联合场景下的幻觉表现，揭示当前模型在该联合场景中 F1 比单独跨模态低 10.9、比单独跨语言低 3.4，且提出多语提示和外部工具辅助两条缓解路径。

## 研究背景与动机

**幻觉是 LLM 部署的核心障碍**：大语言模型在医学诊断、图像描述和语音转文本等高风险应用中产生幻觉，严重阻碍大规模落地。

**跨语言幻觉已有研究但孤立**：mFACT、HalOmi、MM-Eval 等基准仅评估多语场景下的翻译或摘要幻觉，未涉及视觉模态。

**跨模态幻觉研究同样孤立**：CHAIR、POPE、MHaluBench、HallusionBench 等仅在英文单语环境下评估视觉-文本对齐，忽略语言差异带来的额外挑战。

**联合场景更贴近真实世界**：实际应用中常需同时处理多语翻译和多模态对齐（如国际医疗影像报告），但目前没有基准覆盖这一联合场景。

**联合场景更具挑战性**：语言差异和模态差异叠加会放大幻觉风险，模型既要对齐图文又要对齐多语查询，难度远超单一场景。

**缺乏系统性评估与缓解策略分析**：现有工作未系统比较 CoT、SRO、VDGD、HalluciMAD 等缓解策略在联合跨语言跨模态场景下的有效性。

## 方法详解

### 整体框架：四阶段基准构建流程

- **功能**：构建一个同时覆盖跨语言和跨模态幻觉的综合检测基准。
- **为什么**：填补联合跨语言×跨模态幻觉评估的空白，为 MLLM 提供更真实的评估环境。
- **怎么做**：按四个阶段依次执行——(1) 原始多模态数据集选择；(2) 跨模态幻觉数据构造；(3) 跨语言幻觉数据构造；(4) 联合数据集组装。

### 关键设计 1：跨模态幻觉数据构造

- **功能**：在图像描述中注入语义相似但不存在于图中的实体，构造具有迷惑性的幻觉样本。
- **为什么**：简单不相关的虚假实体容易被检测，语义相近的替换才能真正考验模型的细粒度视觉推理能力。
- **怎么做**：从 GQA 和 AMBER 选取物体存在性问题（VQA），从 XM3600 和 xFlickr&Co 选取图像描述（IC），限制每个物体最多出现两次减少冗余；然后用 Gemini-1.5-Pro 对照真实答案和图像，嵌入图中不存在但语义相近的名词生成自然幻觉；每个子集随机采样 900 条，总计 3600 条。

### 关键设计 2：跨语言幻觉数据构造

- **功能**：将英文数据翻译为 9 种不同资源水平的语言，并通过人工校验保证质量。
- **为什么**：不同资源水平的语言在翻译质量和模型理解能力上差异显著，需要全面覆盖才能揭示模型在语言迁移中的弱点。
- **怎么做**：按高/中/低资源分组，每组选翻译错误率最低的 3 种语言——高资源(fr/es/pt)、中资源(cs/nl/sv)、低资源(hr/cy/sw)；用 Google Translate 翻译后打乱排序消除顺序偏差；以英文为锚语言配对随机目标语言；最后组织人工复审，检查幻觉数据是否符合标准、翻译是否准确保留原意。

### 关键设计 3：四类幻觉组合定义

- **功能**：定义四种样本类型——无幻觉、仅跨模态幻觉、仅跨语言幻觉、联合跨语言×跨模态幻觉。
- **为什么**：细粒度分类使得评估可以分离不同因素的影响，便于对比联合 vs 单独场景的难度差。
- **怎么做**：每个样本包含图像、问题和两个语言的回答，按幻觉有无和类型标注为四分类任务，模型需在 AMBER/GQA/xFlickr&Co/XM3600 四个子集上检测幻觉类型。

## 实验关键数据

### 表1：主实验——6款 MLLM 在 CCHall 上的 Acc/Macro-F1 (%)

| 模型 | 方法 | AMBER Acc | GQA Acc | xFlickr Acc | XM3600 Acc | AVG Acc | AVG F1 |
|------|------|-----------|---------|-------------|------------|---------|--------|
| InternVL2-8B | Direct | 29.1 | 29.9 | 38.3 | 38.8 | 34.0 | 42.9 |
| Llama-3.2-11B | CoT | 32.0 | 34.3 | 43.6 | 46.4 | 39.1 | 46.8 |
| Qwen2-VL-7B | CoT | 38.6 | 33.9 | 48.3 | 48.4 | 42.3 | 46.7 |
| Pixtral-12B | HalluciMAD | 46.3 | 45.2 | 57.1 | 58.7 | 51.8 | 56.4 |
| Gemini-1.5-Flash | HalluciMAD | 52.2 | 59.0 | 61.6 | 63.7 | 59.1 | 61.0 |
| **GPT-4o** | **HalluciMAD** | **70.9** | **68.6** | **84.1** | **86.4** | **77.5** | **78.8** |

**关键发现**：
- CCHall 极具挑战性，最弱模型(InternVL2-8B Direct)仅 34.0% 准确率，最强组合(GPT-4o+HalluciMAD)也只达 77.5%。
- 闭源 GPT-4o 和 Gemini-1.5-Flash 明显优于开源模型；开源中 Qwen2-VL-7B 优于更大的 Llama-3.2-11B，说明训练策略比参数量更重要。
- 基础策略(CoT/SRO)更适合小模型(<12B)，高级策略(VDGD/HalluciMAD)在强模型上更有效。

### 表2：分析实验——资源水平、分辨率与响应长度的影响

| 分析维度 | 关键发现 |
|----------|----------|
| 语言资源水平 | 高资源语言检测准确率最高，低资源(hr/cy/sw)显著下降 |
| 图像分辨率 | 高分辨率 > 低分辨率 > 无图像，视觉信息对减少幻觉至关重要 |
| 模型规模 | InternVL: 8B > 4B > 2B，参数越多性能越好 |
| 响应长度 | 超过 120 词后幻觉率急剧上升，长输出更不可靠 |
| 多语提示 | 在 Gemini-1.5-Flash 上，En+SL 比仅 En 在 GQA 上提升 4.8% |
| 外部工具(UniHD) | 平均比 HalluciMAD 再提升 2.7%，外部验证有效 |

## 亮点

1. **首创联合评估维度**：首个同时覆盖跨语言和跨模态幻觉的基准，填补了重要空白。
2. **精心的数据构造流程**：用 Gemini-1.5-Pro 生成语义相近的幻觉实体，比简单替换更具挑战性；三级语言资源覆盖确保评估全面性。
3. **丰富的分析维度**：不仅评估模型表现，还系统探索了语言资源、图像分辨率、响应长度、多语提示和外部工具等影响因素，为缓解幻觉提供了实用指导。

## 局限与展望

1. **仅覆盖文本+图像模态**：未涉及音频/语音等模态，随着多模态模型扩展到更多模态，基准需要同步升级。
2. **幻觉数据由模型生成**：虽经人工校验，但 Gemini-1.5-Pro 生成的幻觉样本可能存在遗留错误或分布偏差。
3. **翻译质量依赖 Google Translate**：低资源语言翻译质量本身较差，可能引入非幻觉的翻译噪声。
4. **四分类任务设置较简单**：实际幻觉可能是连续程度而非离散类别，未来可引入细粒度幻觉严重程度标注。
5. **缺乏自动化幻觉归因**：知道有幻觉但不知道是哪部分产生幻觉，未来可加入 span-level 标注。

## 与相关工作的对比

| 维度 | CCHall | HallusionBench (Guan et al. 2024) | POPE (Li et al. 2023) |
|------|--------|----------------------------------|----------------------|
| 跨语言 | ✔ 9种语言 | ✘ 仅英文 | ✘ 仅英文 |
| 跨模态 | ✔ VQA+IC | ✔ VQA | ✔ 物体存在性 |
| 联合场景 | ✔ | ✘ | ✘ |
| 数据规模 | 大 (3600×多语言) | 346图+1129问 | 较大 |
| 幻觉类型 | 4类组合 | 视觉推理 | 二分类 |

- **vs MM-Eval (Son et al. 2024)**：MM-Eval 覆盖 18 种语言但仅评估文本幻觉，不涉及跨模态场景。CCHall 同时评估视觉和语言幻觉的交互。
- **vs XTRUST (Li et al. 2024)**：XTRUST 覆盖 10 种语言和多个信任维度（幻觉、错误信息、公平性等），但不包含跨模态场景。CCHall 聚焦于跨模态×跨语言的联合幻觉检测。

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](../../CVPR2026/hallucination/mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)
- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/hallucination/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[ICML 2026\] Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation](../../ICML2026/hallucination/capturing_gaze_shifts_for_guidance_cross-modal_fusion_enhancement_for_vlm_halluc.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)

</div>

<!-- RELATED:END -->
