---
title: >-
  [论文解读] Distraction is All You Need for Multimodal Large Language Model Jailbreaking
description: >-
  [CVPR 2025][多模态VLM][MLLM安全] 提出"分散假说"——通过构造高对比度多子图复合输入增加视觉复杂度来制造 OOD 效果，配合查询分解和精心设计的无害指令，实现对 GPT-4o 等闭源 MLLM 高达 42-64% 攻击成功率的黑盒越狱。 领域现状：MLLM 的安全对齐机制（safety alignmen…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "MLLM安全"
  - "越狱攻击"
  - "注意力分散"
  - "对比子图"
  - "黑盒攻击"
---

# Distraction is All You Need for Multimodal Large Language Model Jailbreaking

**会议**: CVPR 2025  
**arXiv**: [2502.10794](https://arxiv.org/abs/2502.10794)  
**代码**: [https://github.com/TeamPigeonLab/CS-DJ](https://github.com/TeamPigeonLab/CS-DJ)  
**领域**: 多模态VLM  
**关键词**: MLLM安全、越狱攻击、注意力分散、对比子图、黑盒攻击

## 一句话总结
提出"分散假说"——通过构造高对比度多子图复合输入增加视觉复杂度来制造 OOD 效果，配合查询分解和精心设计的无害指令，实现对 GPT-4o 等闭源 MLLM 高达 42-64% 攻击成功率的黑盒越狱。

## 研究背景与动机

**领域现状**：MLLM 的安全对齐机制（safety alignment）在面对精心设计的越狱攻击时仍然脆弱。已有攻击方法如 FigStep（将有害文本转为图片）、Hades（有害图文混合）的成功率在闭源模型上普遍较低（<20% ASR）。

**现有痛点**：现有视觉越狱方法主要依赖"内容层面"的对抗——用有害图像内容绕过安全检测。但 GPT-4o 等模型对有害内容的检测越来越强，基于内容的攻击效果持续下降。更根本的问题是：为什么某些视觉输入能绕过安全机制？缺乏机制层面的理解。

**核心矛盾**：MLLM 的安全机制依赖于对输入的整体理解来判断有害性，但当输入的视觉复杂度超过模型的处理能力时，安全检测机制可能失效——这一假说尚未被系统验证和利用。

**本文目标** 验证"视觉复杂度（而非内容）才是越狱的关键"这一假说，并据此设计高效的黑盒越狱框架。

**切入角度**：构造多个高对比度（CLIP 空间中距离最大化）的子图组成复合输入，配合问题分解为多个子查询图像，形成多层次的注意力分散，使模型无法有效聚焦于有害内容的检测。

**核心 idea**：用最大化 CLIP 语义差异的对比子图制造视觉复杂度，通过分散模型注意力而非隐藏有害内容来绕过安全机制。

## 方法详解

### 整体框架
三步流程：(1) 查询分解：将有害查询 $Q$ 用小模型分解为 $m$ 个子查询并转为图像；(2) 对比子图检索：从图像库中贪心检索 $k$ 个与查询和彼此 CLIP 相似度最低的图像；(3) 越狱执行：将子查询图像和对比子图拼成网格复合图，配合三段式无害指令送入 MLLM。

### 关键设计

1. **查询分解（Structured Distraction）**:

    - 功能：将有害意图碎片化，降低单句触发安全检测的概率
    - 核心思路：用 Qwen2.5-3B-Instruct 将有害查询拆分为 $m=3$ 个子查询，然后用排版变换将每个子查询转为图像（Super Moods 字体、红色、50号字）。子查询图像嵌入复合图中
    - 设计动机：直接文本查询几乎 100% 被拦截，但分解后每个子查询单独看是模糊或无害的。3 个子查询是最优甜蜜点——6 个更好但 9 个反而下降（过度分解可能暴露原始意图）

2. **对比子图检索（Visual-Enhanced Distraction）**:

    - 功能：通过语义高度不相关的图像制造视觉复杂度
    - 核心思路：用 CLIP-ViT-L/14 编码查询，从 LLaVA 预训练数据集（10K 子集）中贪心检索余弦相似度最低的图像。选择策略最大化所有节点（查询+子图）之间的成对 L2 距离：$DL = \sum_{i}\sum_{j \neq i} \|v_i - v_j\|_2$。默认检索 9 张对比子图
    - 设计动机：对比距离与攻击成功率强正相关——单一相似图 ×9 仅 24.66% ASR，9 张最相似图 32.53% ASR，9 张最对比图 43.73% ASR。关键发现：随机噪声（20% ASR）几乎无效，因为 MLLM 能轻松识别低信息量输入

3. **三段式无害指令设计**:

    - 功能：引导模型处理复合图中的所有内容而非聚焦安全检测
    - 核心思路：指令分三部分——角色引导（设置场景）、任务引导（要求同时处理多任务）、视觉引导（暗示子图与任务相关）。即使只用任务引导（32% ASR）也远超 Hades（5.51%），三部分组合达 43.73%
    - 设计动机：指令的核心作用是让模型相信复合图中所有子图都与任务相关，从而分散注意力到多个无关图像上，降低对子查询图像中有害内容的检测能力

### 损失函数 / 训练策略
完全无训练的黑盒方法——不需要模型梯度或参数访问，只使用 CLIP 做检索和一个小模型做查询分解。

## 实验关键数据

### 主实验

| 模型 | Hades ASR | CS-DJ ASR | Hades EASR | CS-DJ EASR |
|------|-----------|-----------|------------|------------|
| GPT-4o-mini | 6.08% | **57.80%** | 12.00% | **77.20%** |
| GPT-4o | 5.51% | **42.24%** | 10.13% | **65.86%** |
| GPT-4V | 20.33% | **45.44%** | 45.06% | **71.86%** |
| Gemini-1.5-Flash | 9.20% | **64.11%** | 14.79% | **81.46%** |

### 消融实验

| 配置 | GPT-4o ASR | 说明 |
|------|-----------|------|
| 原始查询图像 (RQ) | 3.20% | 基线 |
| 3 子查询 (3SQ) | 18.80% | 分解有效 +15.6 |
| 3SQ + 9 单一相似图 | 24.66% | 低对比度低效 |
| 3SQ + 9 最相似图 | 32.53% | 多样性帮助 |
| 3SQ + 9 随机噪声 | 20.00% | 噪声几乎无效 |
| 3SQ + 9 最对比图 | **43.73%** | 对比度是关键 |

### 关键发现
- **复杂度而非内容驱动越狱**：9 张随机噪声几乎不增加 ASR（20% vs 18.8% 无子图），但 9 张语义丰富的对比图将 ASR 翻倍以上（43.73%），支持分散假说
- **对比距离与 ASR 强相关**：最对比组的 DL=1878.83 对应 43.73% ASR，最相似组 DL=1378.77 对应 32.53%，单一图 DL=399.08 对应 24.66%
- **黑盒高效性**：在不访问模型参数的情况下，对 GPT-4o 的 ASR 从 Hades 的 5.51% 提升到 42.24%，提升 7.7 倍
- **注意力可视化验证**：Hades 下模型注意力集中在有害图像上（因此能检测），CS-DJ 下注意力被分散到多个子图

## 亮点与洞察
- **"分散假说"提供了越狱攻击的机制解释**：不是"隐藏有害内容"而是"淹没安全机制的处理能力"。这暗示当前安全对齐可能依赖于"足够简单的输入才能有效检测"
- **对比子图检索策略简洁有效**：利用 CLIP 空间的语义距离最大化构造分散材料，无需对抗训练或梯度优化。可以作为安全评估的标准红队工具
- **指令设计的作用不可忽略**：仅靠视觉分散（无特殊指令）效果有限，三段式指令与对比子图的协同作用是关键

## 局限与展望
- 对比距离度量（DL）基于 CLIP 空间，而 MLLM 使用不同的视觉编码器，CLIP 距离与实际分散效果可能不完全对应
- 没有讨论防御对策——如果 MLLM 增加输入复杂度检测的预处理步骤，该方法可能失效
- 方法的伦理争议：虽然是安全研究，但公布了完整的攻击框架，可能被恶意使用
- 对开源模型的效果提升较小（LLaVA-OneVision +7.07%），暗示开源模型的安全对齐可能本就较弱

## 相关工作与启发
- **vs FigStep / MM-SafetyBench**：这些方法将有害文本转为图像绕过检测，但 ASR 很低（3.73%/5.86%），因为模型仍然能"读"出图中文字。CS-DJ 不依赖内容伪装，而是通过复杂度分散
- **vs Hades**：Hades 使用有害图文混合，但仍是单图输入，模型可以集中注意力检测。CS-DJ 的多子图策略从根本上分散了注意力
- **安全启示**：当前 MLLM 的安全机制可能假设了"输入是相对简单的单图+文本"，面对复合多图输入时防御能力大幅下降

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "分散假说"是全新视角，对比子图检索策略巧妙，从机制层面解释越狱而非暴力攻击
- 实验充分度: ⭐⭐⭐⭐ 消融详尽（分解数、对比度、指令成分），但缺少防御评估和更多模型验证
- 写作质量: ⭐⭐⭐⭐ 假说-验证的论证结构清晰，对比距离分析有说服力
- 价值: ⭐⭐⭐⭐ 对 MLLM 安全研究有重要启示，揭示了安全对齐的一个基本弱点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/multimodal_vlm/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[CVPR 2025\] Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](playing_the_fool_jailbreaking_llms_and_multimodal_llms_with_out-of-distribution_.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[CVPR 2025\] Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model](period-llm_extending_the_periodic_capability_of_multimodal_large_language_model.md)
- [\[CVPR 2025\] SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model](seqafford_sequential_3d_affordance_reasoning_via_multimodal_large_language_model.md)

</div>

<!-- RELATED:END -->
