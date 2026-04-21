---
title: >-
  [论文解读] PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset
description: >-
  [CVPR 2025][图像生成][视觉幻觉] 本文提出 PhD，一个 ChatGPT 辅助构建的大规模视觉幻觉评估数据集，包含 14K+ 日常图片、750 张反常识图片和 102K VQA 三元组，通过 4 种评估模式×5 种视觉任务系统化评估多模态大语言模型的幻觉问题，在规模和挑战性上远超现有基准。
tags:
  - CVPR 2025
  - 图像生成
  - 视觉幻觉
  - 多模态大语言模型
  - 评估基准
  - ChatGPT辅助构建
  - 二值VQA
---

# PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset

**会议**: CVPR 2025  
**arXiv**: [2403.11116](https://arxiv.org/abs/2403.11116)  
**代码**: https://github.com/jiazhen-code/PhD  
**领域**: 多模态VLM / 幻觉评估  
**关键词**: 视觉幻觉, 多模态大语言模型, 评估基准, ChatGPT辅助构建, 二值VQA

## 一句话总结

本文提出 PhD，一个 ChatGPT 辅助构建的大规模视觉幻觉评估数据集，包含 14K+ 日常图片、750 张反常识图片和 102K VQA 三元组，通过 4 种评估模式×5 种视觉任务系统化评估多模态大语言模型的幻觉问题，在规模和挑战性上远超现有基准。

## 研究背景与动机

1. **领域现状**：多模态大语言模型（MLLM）如 LLaVA、Qwen-VL 等在多种视觉任务上表现出色，但普遍存在视觉幻觉问题——生成与视觉内容不一致的描述。视觉幻觉评估（VHE）是一个新兴且重要的研究方向。
2. **现有痛点**：(a) POPE 等先驱数据集规模小（仅 3K 个 VQA 三元组）、任务单一（仅物体识别）、hallucinatory item（hitem）选择不够图像特异化；(b) AMBER 依赖全手工标注，成本高、词汇受限、难以扩展；(c) 现有数据集性能快速饱和（图 1(d)），无法区分不断进化的模型。
3. **核心矛盾**：有效的幻觉评估需要选择图像特异化且具有挑战性的 hallucinatory item（hitem），但 hitem 选择在现有工作中很少被系统研究——POPE/ROME 基于标签共现（非图像特异化），NOPE/CIEM 完全跳过了 hitem 选择。
4. **本文目标**：构建一个大规模、多模式、多任务的 VHE 数据集，明确关联 MLLM 视觉幻觉的三大原因（视觉歧义、多模态输入不一致、反常识内容），支持细粒度的模型分析。
5. **切入角度**：利用 ChatGPT 辅助实现半自动化的数据集构建管线，通过 CLIP 对 hitem 进行视觉相关性排序，使生成的 VQA 样本具有图像特异化的挑战性。
6. **核心 idea**：围绕幻觉三大原因设计四种评估模式（PhD-base、PhD-sec、PhD-icc、PhD-ccs），覆盖五种视觉任务（物体/属性/情感/位置识别+计数），用 ChatGPT 辅助生成 hitem、问题和上下文。

## 方法详解

### 整体框架

PhD 基于 TDIUC 数据集的标注进行改造，通过四个核心模块构建：(1) 任务特异化的 hitem 选择→(2) hitem 嵌入问题生成→(3) 似是而非/错误上下文生成→(4) 反常识（CCS）图像生成。最终数据集支持 4 种模式×5 种任务=20 种评估组合，总计 102K VQA 三元组。选择 TDIUC 是因为其图像来自 MS-COCO（可能被 MLLM 训练时见过），因此模型的错误响应更可能归因于幻觉而非能力不足。

### 关键设计

1. **ChatGPT+CLIP 辅助的 hitem 选择**:
    - 功能：为每张图像的每个视觉任务选择具有图像特异化挑战性的 hallucinatory item
    - 核心思路：以颜色属性识别为例——(a) 用 ChatGPT 扩展颜色词汇表（手动几个→自动扩展到 35 个颜色）；(b) 从 TDIUC 问答对中用 ChatGPT 提取主语和属性（如"motorcycle"和"black"）；(c) 排除 ground truth 及同义词得到候选 hitem；(d) 用 CLIP 计算每个候选 hitem+主语组合与图像的余弦相似度，选择视觉上最具迷惑性的作为 hitem。最后进行人工抽检。共选出 1,452 个多样化且有挑战性的 hitem。
    - 设计动机：CLIP 排序使得 hitem 与图像"看起来合理但实际不对"，比随机选择或共现选择更能有效诱发幻觉

2. **四种评估模式设计**:
    - 功能：分别评估 MLLM 在三种幻觉原因下的表现
    - 核心思路：(a) **PhD-base**（无上下文的日常图像问答）——测试视觉歧义引发的幻觉（原因 I）；(b) **PhD-sec**（附加似是而非上下文）和 **PhD-icc**（附加错误上下文）——测试多模态输入不一致引发的幻觉（原因 II）；(c) **PhD-ccs**（反常识图像问答）——测试内部知识与视觉内容冲突时的幻觉（原因 III）。似是而非上下文由 ChatGPT 生成，要求与图像有关但可能不反映当前状态；错误上下文则直接矛盾。CCS 图像由 Doubao 和 DALL-E3 生成（如"方形轮胎的汽车"）。
    - 设计动机：现有数据集缺乏对不同幻觉原因的显式评估，PhD 的模式-任务结构使得可以精确定位模型的薄弱环节

3. **PhD Index 评估指标**:
    - 功能：提供平衡的幻觉评估分数
    - 核心思路：分别计算 Yes 问题和 No 问题的 recall，取二者的调和平均值作为 PhD Index。一个只回答 Yes（或 No）的模型得分为 0，随机猜测得分为 0.5。这确保了对肯定/否定偏好的平衡评估。
    - 设计动机：避免模型通过一律回答 "Yes" 或 "No" 来获得高分，真正测试其视觉理解能力

### 损失函数 / 训练策略

- 本文是评估数据集，不涉及模型训练
- 数据集构建管线中 ChatGPT 用于生成 hitem、问题、上下文，CLIP 用于 hitem 排序，AIGC 工具用于 CCS 图像生成
- 人工参与主要集中在抽检验证，整体是半自动化流程

## 实验关键数据

### 主实验

**开源模型整体 VHE（PhD Index）：**

| 模型 | ViT | LLM | POPE | AMBER | PhD Index |
|------|-----|-----|------|-------|-----------|
| LLaVA-OneVision | SoViT-400m/14 | Qwen2-72B | 0.84 | 0.90 | 0.698 |
| Molmo | -L/14 | Qwen2-72B | 0.84 | 0.85 | 0.690 |
| InternVL-1.5 | InternViT | InternLM2-20B | - | - | ~0.65 |
| LLaVA-1.5 | CLIP-L/14 | Vicuna-7B | ~0.80 | ~0.82 | 0.265 |
| LLaVA-1.5-L | CLIP-L/14 | Vicuna-13B | ~0.80 | ~0.82 | 0.270 |

### 消融实验

| 评估模式 | 代表性发现 | 说明 |
|---------|-----------|------|
| PhD-base | 模型表现最好 | 无干扰，仅测视觉歧义 |
| PhD-sec（似是而非上下文） | 显著下降 | 模型容易被合理但不准确的文本误导 |
| PhD-icc（错误上下文） | 下降更大 | 模型强烈偏向文本信息 |
| PhD-ccs（反常识图像） | 最具挑战性 | 模型依赖内部知识而非视觉内容 |

### 关键发现

- **PhD 远比 POPE/AMBER 更具挑战性**：LLaVA 系列在 POPE 上达 0.84、AMBER 上达 0.90，但在 PhD 上仅 0.265-0.698，有力区分模型能力
- **更大的 LLM 未必更好**：LLaVA-1.5-L（13B）和 LLaVA-1.5（7B）在 PhD 上几乎无差异（0.270 vs 0.265），表明参数量不是决定性因素
- **模型普遍存在文本偏好**：在 PhD-sec 和 PhD-icc 模式下性能大幅下降，说明 MLLM 的 LLM 内核倾向于信任文本输入而非视觉输入
- **反常识场景是最大短板**：PhD-ccs 暴露了模型过度依赖训练时学到的常识知识，无法根据实际视觉内容做出判断
- **VCD 和 Woodpecker 等幻觉缓解方法效果有限**：说明当前的缓解策略还远不足以解决幻觉问题

## 亮点与洞察

- **ChatGPT+CLIP 的半自动 hitem 选择管线**是最大创新：ChatGPT 负责词汇扩展和文本生成（零成本扩展），CLIP 负责视觉相关性排序（确保图像特异化），人工仅做验证，实现了规模与质量的平衡。这套管线可迁移到任何需要构建对抗性评估集的场景。
- **四模式×五任务的结构化评估框架**非常有价值：它不仅给出一个整体分数，更能精确定位模型在哪种幻觉原因和哪种视觉任务上最弱，为模型开发者提供改进方向。
- **故意使用 MS-COCO 图像（可能已被训练见过）**的设计很巧妙：如果模型在一张"见过"的图像上还产生幻觉，更能说明问题出在幻觉而非能力不足。

## 局限与展望

- **依赖 TDIUC 标注**：数据集的图像和初始标注来自 TDIUC/MS-COCO，可能存在标注错误影响质量
- **仅支持二值 VQA**：Yes/No 问答虽然便于大规模评估，但无法捕捉更复杂的幻觉模式（如部分正确的描述）
- **CCS 图像质量不稳定**：AIGC 工具生成的反常识图像质量参差，部分可能不够逼真
- **英语为主**：TDIUC 数据和 ChatGPT 生成的文本均为英语，对多语言 MLLM 的评估覆盖不足
- 改进方向：扩展到开放式问答评估；增加更多高级视觉任务（如视觉推理）；构建针对特定领域（如医学）的幻觉评估集

## 相关工作与启发

- **vs POPE**: POPE 仅 3K 样本、仅物体识别、基于共现选 hitem（非图像特异化）；PhD 规模大 34 倍（102K）、5 种任务、图像特异化 hitem 选择
- **vs AMBER**: AMBER 依赖全手工标注（687 hitem）、14K VQA 样本；PhD 半自动化构建（1,452 hitem）、102K VQA 样本，扩展性更好
- **vs HallusionBench**: HallusionBench 聚焦高级视觉推理的主观评估；PhD 聚焦低中级视觉识别的客观评估，二者互补
- **vs MMMU**: MMMU 评估高级学科知识理解能力而非幻觉；PhD 专注于幻觉评估，即使模型具备能力也可能因幻觉出错

## 评分

- 新颖性: ⭐⭐⭐⭐ 四模式评估框架设计和 ChatGPT+CLIP 半自动构建管线均有创新，与幻觉三原因的显式关联是独特贡献
- 实验充分度: ⭐⭐⭐⭐ 评估了 15 个开源模型 + 3 个商业模型 + 2 个缓解方法，模式/任务/模型多维度分析详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集构建管线描述详细，表格和图示信息量大
- 价值: ⭐⭐⭐⭐ 作为目前最大且最具挑战性的 VHE 基准，对 MLLM 社区有重要的评估参考价值

<!-- RELATED:START -->

## 相关论文

- [Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation](../../NeurIPS2025/image_generation/hallucination_as_an_upper_bound_a_new_perspective_on_text-to-image_evaluation.md)
- [ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](../../ICCV2025/image_generation/adiee_automatic_dataset_creation_and_scorer_for_instruction-guided_image_editing.md)
- [ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)
- [lbGen: Low-Biased General Annotated Dataset Generation](low-biased_general_annotated_dataset_generation.md)
- [ViUniT: Visual Unit Tests for More Robust Visual Programming](viunit_visual_unit_tests_for_more_robust_visual_programming.md)

<!-- RELATED:END -->
