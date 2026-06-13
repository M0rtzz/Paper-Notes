---
title: >-
  [论文解读] NegVQA: Can Vision Language Models Understand Negation?
description: >-
  [ACL 2025][多模态VLM][否定理解] 提出 NegVQA 基准（7,379 道二选一 VQA 题），系统评估 20 个 VLM 对否定句的理解能力，发现所有模型在否定问题上性能大幅下降（平均 29.7%），并揭示"U 型"缩放趋势。 - 否定是语言的基本现象：一个"not"可以完全反转句子含义…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "否定理解"
  - "VQA benchmark"
  - "视觉语言模型"
  - "缩放趋势"
  - "诊断评估"
---

# NegVQA: Can Vision Language Models Understand Negation?

**会议**: ACL 2025  
**arXiv**: 2505.22946  
**代码**: [https://yuhui-zh15.github.io/NegVQA/](https://yuhui-zh15.github.io/NegVQA/)  
**领域**: 多模态VLM  
**关键词**: 否定理解, VQA benchmark, 视觉语言模型, 缩放趋势, 诊断评估

## 一句话总结

提出 NegVQA 基准（7,379 道二选一 VQA 题），系统评估 20 个 VLM 对否定句的理解能力，发现所有模型在否定问题上性能大幅下降（平均 29.7%），并揭示"U 型"缩放趋势。

## 背景与动机

- **否定是语言的基本现象**：一个"not"可以完全反转句子含义，如"谁写了这本书？"→"谁**没有**写这本书？"
- **VLM 部署于高风险场景**：机器人、具身 AI 等系统中，如果模型无法正确理解否定指令（如"不要执行某操作"），可能导致与用户意图相反的危险行为
- **已有 VQA 数据集几乎全是肯定句**：现有基准未系统考察否定理解能力，缺乏针对性的诊断工具
- **文本侧已有研究但视觉侧空白**：先前工作主要评估纯语言模型和 CLIP 的否定能力，针对生成式 VLM 的否定理解尚无系统评估

## 方法详解

### 1. 数据构建流程

NegVQA 基于 VMCBench（多选 VQA 基准）构建，包含两个核心步骤：

**步骤一：否定问题生成**

利用 GPT-4o 将原始问题转换为否定版本，同时保持句法结构和语义。对每个原始问题，LLM 判断其是否可否定（如"求 x 的值"无法有意义地否定），最终从 9,018 题中筛选出 7,379 题。人工验证 100 个样本的准确率达 97%。

**步骤二：答案选项调整**

将原始四选一格式缩减为二选一：保留正确答案和随机选取的一个错误答案，然后**反转正确性**。形式化表示：

$$
\text{NegVQA}(q, a^+, a^-) = (\text{Negate}(q), a^-, a^+)
$$

其中 $a^+$ 为原始正确答案，$a^-$ 为原始错误答案。否定后 $a^-$ 变为正确答案，$a^+$ 变为错误答案。这确保模型必须真正理解否定才能作答。

### 2. 数据集覆盖范围

NegVQA 涵盖 20 个 VQA 数据集的四大领域：

- **通用 VQA**：VQAv2、OKVQA、MMVet、VizWiz、A-OKVQA、MMStar、SEEDBench
- **推理任务**：MathVision、GQA、MMMU、RealWorldQA、MathVista、ScienceQA
- **OCR**：OCRVQA、TextVQA
- **文档/图表**：DocVQA、InfoVQA、ChartQA、TableVQABench、AI2D

否定形式覆盖对象缺失、属性否定（颜色/大小/位置）、动作否定（未发生的事件）及需深层推理的复杂否定。

### 3. U 型缩放趋势的理论解释

作者将 NegVQA 上的表现建模为两种能力的复合：

$$
\text{Perf}_{\text{NegVQA}}(s) = f_{\text{QA}}(s) \cdot g_{\text{neg}}(s)
$$

其中 $s$ 为模型规模，$f_{\text{QA}}(s)$ 是原始问答能力（随规模单调递增），$g_{\text{neg}}(s)$ 是否定理解能力（类似 tanh 曲线，先平坦后骤升）。

- **小模型**：推理能力弱，对否定和非否定问题都随机猜测，表现接近
- **中等模型**：原始问答能力提升，但仍忽略否定词，把否定题当肯定题做，性能反而下降
- **大模型**：规模足够时否定理解能力激活，性能恢复并提升

## 实验结果

### 表1：各模型家族在 NegVQA 上的平均表现

| 模型 | PosVQA (%) | NegVQA (%) | 下降幅度 |
|------|-----------|-----------|---------|
| Qwen2-VL-72B | 92.2 | 72.7 | -19.5 |
| Molmo-72B | 87.5 | 74.5 | -13.0 |
| Qwen2-VL-7B | 88.8 | 57.2 | -31.6 |
| Cambrian-34B | 87.4 | 59.9 | -27.5 |
| VILA1.5-40B | 85.7 | 56.6 | -29.1 |
| VILA1.5-8B | 78.5 | 56.2 | -22.3 |
| Molmo-7B-D | 83.0 | 55.3 | -27.7 |
| Cambrian-8B | 83.8 | 55.7 | -28.1 |
| Qwen2-VL-2B | 85.4 | 53.4 | -32.0 |
| LLaVA-1.5-7B | 73.3 | 47.9 | -25.4 |
| LLaVA-1.5-13B | 74.3 | 40.3 | -34.0 |
| DeepSeek-VL-7B | 79.8 | 41.9 | -37.9 |
| DeepSeek-VL-1.3B | 75.0 | 37.2 | -37.8 |
| InstructBLIP-7B | 55.3 | 28.9 | -26.4 |
| InstructBLIP-13B | 67.0 | 35.2 | -31.8 |
| 人类基线 | — | 89.0 | — |

### 表2：按领域分类的 NegVQA 表现（最佳模型 Qwen2-VL-72B）

| 领域 | PosVQA (%) | NegVQA (%) | 下降幅度 |
|------|-----------|-----------|---------|
| 通用 (General) | 93.6 | 71.7 | -21.9 |
| 推理 (Reasoning) | 83.4 | 64.1 | -19.3 |
| OCR | 99.0 | 91.8 | -7.2 |
| 文档/图表 (Doc&Chart) | 94.8 | 72.4 | -22.4 |
| 平均 | 92.2 | 72.7 | -19.5 |

## 关键发现

1. **所有 VLM 都难以理解否定**：平均性能下降 29.7 个百分点，最佳模型 Qwen2-VL-72B 仍比人类（89%）低 16.3%
2. **U 型缩放趋势**：模型从小到大，NegVQA 性能先降后升，在 Cambrian 和 VILA 系列中最为显著
3. **训练数据缺乏否定样本**：LLaVA 微调数据中仅 1.1% 的对话包含"not"，这可能是核心原因
4. **OCR 任务受影响最小**：可能因 OCR 问题更聚焦于文本匹配而非语义推理
5. **推理和文档任务 U 型效应更明显**：这些任务需要更深层的语义理解，否定误解影响更大
6. **模型规模并非万能**：LLaVA-1.5-13B（40.3%）反而比 LLaVA-1.5-7B（47.9%）更差，体现 U 型中的下降阶段

## 亮点

- **简洁有效的基准设计**：通过 LLM 自动否定 + 二选一反转，以低成本构建高质量诊断数据集，97% 转换准确率
- **U 型缩放发现**：揭示否定理解可能是一种涌现能力，为 VLM 缩放研究提供新视角
- **覆盖面广**：20 个子数据集、4 大领域、7 个模型家族 20 个模型，结论具有普遍性
- **强烈的安全警示**：指出否定理解失败在具身 AI 场景中的严重隐患

## 局限性

- **仅限多选格式**：二选一选项可能无法完全反映开放式场景中的否定理解能力
- **仅零样本评估**：未探索 few-shot 提示是否能改善否定理解
- **未提出改进方案**：论文诊断了问题但未给出训练时的具体解决方法（仅建议数据增强）
- **否定类型分析不够细致**：未对不同否定形式（词汇否定 vs 句法否定 vs 隐含否定）进行分类对比
- **GPT-4o 转换偏差**：3% 的错误率可能在大规模数据上引入系统性噪声

## 相关工作

- **VQA 基准**：VQAv2、OKVQA、MMMU、VMCBench 等提供原始评估数据
- **否定理解（NLP 侧）**：Kassner & Schütze (2020) 评估 PLM 否定；Zhang et al. (2023) 发现语言模型的 U 型否定缩放
- **否定理解（视觉侧）**：Alhamoud et al. (2025) 和 Singh et al. (2024) 评估 CLIP 否定理解，但缺乏生成式 VLM 评估
- **缩放规律**：Kaplan et al. (2020) 语言模型缩放律；McKenzie et al. (2023) 逆缩放；Wei et al. (2022) U 型缩放

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 技术深度 | ⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 表达清晰度 | ⭐⭐⭐⭐ |
| 影响力 | ⭐⭐⭐⭐ |

**综合**: ⭐⭐⭐⭐ — 诊断性基准论文，实验全面、发现有洞察力（U 型缩放），对 VLM 安全部署具有重要警示意义。技术深度相对有限，但选题精准、结论扎实。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](../../CVPR2025/multimodal_vlm/vision-language_models_do_not_understand_negation.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?](cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)
- [\[ACL 2025\] Can MLLMs Understand the Deep Implication Behind Chinese Images?](can_mllms_understand_the_deep_implication_behind_chinese_images.md)

</div>

<!-- RELATED:END -->
