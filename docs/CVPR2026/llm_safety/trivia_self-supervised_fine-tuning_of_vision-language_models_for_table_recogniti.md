---
title: >-
  [论文解读] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition
description: >-
  [CVPR 2026][多模态VLM][表格识别] 提出 TRivia 自监督微调框架，通过表格问答（QA）驱动的 GRPO 强化学习，让 VLM 直接从无标注表格图像中学习表格识别能力，3B 参数的 TRivia-3B 在多个基准上超越 Gemini 2.5 Pro 和 GPT-5 等私有模型。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "表格识别"
  - "自监督微调"
  - "GRPO"
  - "视觉语言模型"
  - "强化学习"
---

# TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition

**会议**: CVPR 2026  
**arXiv**: [2512.01248](https://arxiv.org/abs/2512.01248)  
**代码**: [https://github.com/HKU-TASR/TRivia](https://github.com/HKU-TASR/TRivia)  
**领域**: 多模态VLM  
**关键词**: 表格识别, 自监督微调, GRPO, 视觉语言模型, 强化学习

## 一句话总结
提出 TRivia 自监督微调框架，通过表格问答（QA）驱动的 GRPO 强化学习，让 VLM 直接从无标注表格图像中学习表格识别能力，3B 参数的 TRivia-3B 在多个基准上超越 Gemini 2.5 Pro 和 GPT-5 等私有模型。

## 研究背景与动机

**领域现状**：表格识别（Table Recognition, TR）旨在将表格图像转换为 HTML 或 Markdown 等半结构化表示。近年 VLM 的发展让 TR 性能大幅提升，私有模型如 Gemini 2.5 Pro 已展现出强大的 TR 能力。开源 VLM 则受限于标注数据规模，仍明显落后。

**现有痛点**：TR 数据获取面临三难困境——(1) 合成数据可扩展但缺乏真实视觉多样性；(2) 真实数据标注昂贵耗时；(3) 从私有模型蒸馏伪标签不仅成本高，还受限于教师模型的性能天花板且可能违反服务协议。MinerU2.5 即使用了百万级样本、人工标注加 Gemini 蒸馏，仍无法超越教师模型。

**核心矛盾**：开源 TR 模型标注数据受限且天花板由教师模型决定——标注数据 vs 性能的瓶颈。而海量无标注表格图像唾手可得却无法直接利用。

**本文目标** (1) 如何从无标注表格图像中提取有效监督信号；(2) 如何筛选最具训练价值的样本；(3) 如何生成多样且可验证的 QA 对作为奖励信号。

**切入角度**：QA 是 TR 的下游任务——如果模型能正确回答关于表格的问题，就隐含表明它对表格结构和内容的识别是准确的。这比直接预测 HTML 标注容易得多，且 QA 对的正确性可通过交叉验证核实。

**核心 idea**：用"能否正确回答表格问题"作为 proxy reward，通过 GRPO 让 VLM 从无标注表格图像中自监督学习表格识别。

## 方法详解

### 整体框架
TRivia 分为两个阶段：(1) 数据准备阶段——从无标注表格图像中筛选最有信息量的样本，并为每张图自动生成多样化的 QA 对；(2) 训练阶段——用 GRPO 强化学习框架，以 QA 正确率作为奖励函数微调 VLM。整体训练分三个 stage：OTSL 暖身（700K 合成数据）→ 监督微调（50K 真实数据）→ TRivia 自监督 RL（50K 无标注数据）。

### 关键设计

1. **QA 驱动的 GRPO 自监督微调**:

    - 功能：将 TR 任务转化为可用无标注数据优化的 RL 问题
    - 核心思路：对每张表格图像，TR 模型（policy）生成 $R$ 个识别结果 $\{o_j\}$，每个结果送入 LLM（Qwen3-8B）回答预生成的 QA 对。奖励 $\text{Reward}(o_j) = \frac{1}{|QA|}\sum_{(q,a)} F1(M_{LLM}(q;o_j), a)$，即回答的 F1 分数均值。GRPO 基于组内奖励的相对差异优化 policy，找到能产生最准确表格识别结果的策略。额外引入 illegal-sample filtering，剔除无效/重复输出（奖励为 0 的样本），避免奖励分布被压缩导致训练不稳定。
    - 设计动机：QA 比直接预测 HTML 标注简单得多——不需要推断 colspan/rowspan 等复杂结构，只需理解特定区域内容。且 QA 的正确性可自动验证，无需人工标注。

2. **Response-Consistency Sampling（响应一致性采样）**:

    - 功能：从无标注数据池自动识别最有训练价值的样本
    - 核心思路：对每张图像让 TR 模型生成 $K$ 个识别结果，计算所有结果对之间的 TEDS（Tree Edit Distance-based Similarity）相似度均值作为一致性得分：$\text{Consistency}(I) = \frac{2}{K^2-K}\sum_{i<j} \text{TEDS}(o_i, o_j)$。一致性越低说明模型对该样本越不确定，因此该样本对 GRPO 训练越有价值（因为 GRPO 受益于多样化的响应）。实际操作中过滤掉一致性低于 0.4 的噪声样本，在 0.4-1.0 范围内均匀采样。
    - 设计动机：不是所有无标注样本都有同等贡献——聚类方法只能衡量多样性但不能评估对特定模型的训练价值，人工筛选不可扩展。Response-consistency 直接与 GRPO 的训练机制对齐。

3. **Attention-Guided Diverse QA Generation（注意力引导的 QA 生成）**:

    - 功能：为每张表格图像生成覆盖不同区域、多样化的 QA 对
    - 核心思路：利用 VLM 注意力机制的视觉 grounding 特性——每个答案 token 关注的 visual token 构成该 QA 对的"视觉来源"：$VS((q,a)) = \{v | \mathcal{A}(v|a) > \tau_\mathcal{A}\}$。三步流程：(1) 用 Qwen2.5-VL-72B 多次采样生成候选 QA 池；(2) 用 InternVL3-78B 交叉验证每个 QA 的正确性和视觉依赖性（有图能答、无图不能答）；(3) 贪心选择视觉来源 IOU 最小的 QA 子集，确保覆盖表格不同区域。每张图最终保留约 30 个多样 QA 对。
    - 设计动机：单次 QA 生成只覆盖表格部分区域，多次采样又倾向于生成同义替换。注意力引导可精确量化每个 QA 的信息来源，从而显式最大化覆盖范围。

### 损失函数 / 训练策略
三阶段训练：Stage 1 用 700K 合成+公开数据做 OTSL 格式暖身（冻结视觉编码器）；Stage 2 用 50K 真实表格做全参数监督微调；Stage 3 用 TRivia 框架在 50K 无标注数据上做 GRPO RL 微调。

## 实验关键数据

### 主实验

| 模型 | OmniDocBench TEDS | CC-OCR TEDS | OCRBench TEDS | Overall TEDS |
|------|-------------------|-------------|---------------|--------------|
| UniTable | 82.76 | 57.84 | 67.73 | 70.86 |
| Qwen2.5-VL-72B | 87.85 | 81.22 | 81.33 | 83.52 |
| Gemini 2.5 Pro | 90.90 | **85.56** | 88.94 | 88.93 |
| GPT-5 | 84.91 | 63.25 | 79.91 | 78.30 |
| MinerU2.5 | 90.85 | 79.76 | 87.13 | 86.82 |
| PaddleOCR-VL | 91.12 | 79.62 | 79.29 | 83.36 |
| **TRivia-3B** | **91.60** | 84.90 | **90.76** | **89.88** |

### 消融实验

| 配置 | OmniDocBench | CC-OCR | OCRBench | Overall | 说明 |
|------|-------------|--------|----------|---------|------|
| Stage-2 (SFT baseline) | 90.08 | 82.48 | 90.08 | 88.57 | 有监督微调天花板 |
| + 72B 伪标签 SFT | 84.41 | 70.54 | 80.87 | 80.02 | 伪标签质量差，性能暴跌 -8.55 |
| + 72B 伪标签 GRPO | 86.19 | 78.12 | 84.16 | 83.65 | GRPO 缓解但仍跌 -4.92 |
| TRivia-3B | 91.60 | 84.90 | 90.76 | 89.88 | QA reward 突破监督天花板 +1.31 |
| w/o Attention-guided QA | - | - | - | 显著下降 | 复杂表格尤其脆弱 |
| w/o Response-consistency | - | - | - | TEDS 52→63.5 | 随机采样收敛慢 |
| w/o Illegal filtering | - | - | - | 训练不稳定 | 收敛步数增加 25%，最终性能 -3 TEDS |

### 关键发现
- **QA proxy reward 突破了监督学习天花板**：TRivia-3B（89.88 TEDS）超越 Stage-2 监督极限（88.57），提升 1.31 个 TEDS。而用同一教师模型（72B）直接生成伪标签反而暴跌 8+ TEDS
- **3B 参数碾压 72B+ 私有模型**：TRivia-3B 以 3B 参数超越 Gemini 2.5 Pro（>千亿参数）和 GPT-5，证明自监督 RL 可弥补参数规模差距
- **Response-consistency sampling 加速收敛**：相比随机采样，TEDS 从 52 提升到 63.5（同等训练步数），关键是选到了对当前模型最有挑战性的样本
- **Illegal-sample filtering 对训练稳定性至关重要**：不过滤非法输出导致训练后期严重震荡，过滤后收敛步数减少 25%
- 作为数据标注器：TRivia-3B 生成的伪标签用于 SFT 蒸馏，可获得 89.99 TEDS，几乎等于 TRivia-3B 本身

## 亮点与洞察
- **QA 作为 proxy supervision 的精妙设计**：避开了直接预测难以验证的 HTML 标注，转而用下游任务（QA）的正确性作为间接监督。这个思路可推广到其他结构化输出任务——只要能设计出下游验证任务，就能实现自监督 RL。
- **注意力分布的创造性利用**：用 VLM 生成答案时的注意力分布来定位 QA 的视觉来源，实现了无需额外标注的空间 grounding，解决了 QA 多样性问题。
- **突破教师模型天花板**：传统蒸馏受限于教师模型质量，而 TRivia 通过 RL 绕过了这个限制——不直接使用教师的输出作为标签，而是仅让教师生成 QA 对作为验证工具，学生模型可超越教师。

## 局限与展望
- 当前仅针对**表格识别**验证，扩展到其他文档解析任务（图表、公式、布局）需要重新设计 QA proxy
- Response-consistency sampling 在离线阶段执行，训练过程中模型能力变化后可能采样分布不再最优——在线更新可能进一步提升
- 依赖多个外部模型（Qwen2.5-VL-72B 生成 QA、InternVL3-78B 验证、Qwen3-8B 答题），部署复杂度高
- 仅在 OTSL 格式上验证，对 Markdown/HTML 等更通用格式的适用性未验证
- PubTabNet 上的 S-TEDS 略低于专用 expert 模型，说明对特定领域数据的过拟合仍有价值

## 相关工作与启发
- **vs MinerU2.5**：大规模人工标注+Gemini 蒸馏，性能受限于教师模型天花板（86.82 TEDS）。TRivia 用 RL 突破天花板达 89.88，且无需人工标注。
- **vs UniTable**：传统 image-to-markup 方法，受限于分辨率和上下文窗口（448×448, 512 tokens），复杂表格上性能差。TRivia 基于 Qwen2.5-VL 架构支持更高分辨率。
- **vs DeepSeek-R1 的 GRPO 应用**：DeepSeek-R1 将 GRPO 用于 LLM 推理增强，TRivia 将其迁移到视觉文档理解领域，验证了 GRPO 在视觉任务的有效性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 自监督 RL 突破标注数据天花板的范式非常新颖，QA proxy reward 设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 四个基准、12 个 baseline、全面消融，还验证了作为标注器的泛化能力
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但整体篇幅较长，部分内容可精简
- 价值: ⭐⭐⭐⭐⭐ 3B 模型超越 Gemini 2.5 Pro，为开源 TR 指明自监督 RL 方向，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2026\] Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)
- [\[ACL 2026\] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](../../ACL2026/llm_safety/why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)
- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)

</div>

<!-- RELATED:END -->
