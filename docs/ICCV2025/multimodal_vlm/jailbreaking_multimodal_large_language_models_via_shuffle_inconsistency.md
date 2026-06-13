---
title: >-
  [论文解读] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency
description: >-
  [ICCV 2025][多模态VLM][多模态大语言模型安全] 发现多模态大语言模型(MLLMs)在理解能力和安全能力之间存在**打乱不一致性(Shuffle Inconsistency)**——模型能理解打乱后的有害指令，但安全机制却无法防御；据此提出基于查询的黑盒越狱攻击方法 SI-Attack，在开源和闭源商用模型上均显著提升攻击成功率。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "多模态大语言模型安全"
  - "越狱攻击"
  - "Shuffle Inconsistency"
  - "黑盒优化"
  - "安全对齐"
---

# Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency

**会议**: ICCV 2025  
**arXiv**: [2501.04931](https://arxiv.org/abs/2501.04931)  
**作者**: Shiji Zhao, Ranjie Duan, Fengxiang Wang, Chi Chen, Caixin Kang, Jialing Tao, YueFeng Chen, Hui Xue, Xingxing Wei (北京航空航天大学)
**领域**: 多模态VLM  
**关键词**: 多模态大语言模型安全, 越狱攻击, Shuffle Inconsistency, 黑盒优化, 安全对齐

## 一句话总结

发现多模态大语言模型(MLLMs)在理解能力和安全能力之间存在**打乱不一致性(Shuffle Inconsistency)**——模型能理解打乱后的有害指令，但安全机制却无法防御；据此提出基于查询的黑盒越狱攻击方法 SI-Attack，在开源和闭源商用模型上均显著提升攻击成功率。

## 研究背景与动机

多模态大语言模型(如 GPT-4o、Claude-3.5-Sonnet)在商业应用中取得了显著进展，但仍面临安全漏洞风险。越狱攻击(Jailbreak Attack)作为红队测试手段，旨在绕过模型的安全机制、发现潜在风险。

现有越狱方法的主要问题：

**复杂度高**：多数方法依赖复杂的对抗优化（如对抗扰动注入）或精心设计的图文提示

**闭源模型攻击效果差**：商用闭源模型通常有额外的外部安全护栏(outer safety guardrails)，能检测有害意图并拦截攻击指令，导致现有方法的攻击成功率有限

**缺乏对模型能力间隙的利用**：先前工作较少关注模型理解能力与安全能力之间的差异

本文的核心发现：先前研究表明 MLLMs 对打乱顺序的文本和图像仍能保持较好的理解能力（如文本/图像检索任务）。作者由此提出两个关键问题：

- **理解能力维度**：MLLMs 是否能理解打乱后的有害文本和图像？
- **安全能力维度**：MLLMs 的防御机制是否能抵御打乱后的有害指令？

实验显示答案令人惊讶：**MLLMs 能够理解打乱后的有害指令，但其安全防御机制却无法有效防御这些打乱后的指令**。这种理解能力与安全能力之间的不一致即为"Shuffle Inconsistency"。

## 方法详解

### 整体框架

SI-Attack 的核心思路是：利用 Shuffle Inconsistency 漏洞，通过打乱有害指令的文本和图像来绕过安全机制，并通过基于查询的黑盒优化来克服打乱操作的随机不稳定性，选出最有效的打乱组合。

整体流程：
1. 将输入的有害文本按词级别(word-wise)拆分并随机打乱
2. 将输入的有害图像按块级别(patch-wise)拆分并随机打乱
3. 将打乱后的文图输入目标 MLLM 获取响应
4. 用毒性评判模型评估响应的有害程度
5. 若有害分数达到阈值则攻击成功；否则重复打乱直至达到最大迭代次数

### 关键设计

**1. 文本打乱(Text Shuffle)**

将有害文本 $T = [w_1, w_2, \ldots, w_n]$ 按词级别随机打乱：

$$T' = \text{Shuffle}_w(T)$$

实验比较了多种打乱策略：不打乱、仅打乱名词和形容词、三元组打乱、三元组内打乱、基于 BPE 的 token 级打乱、全词打乱。结果显示**全词打乱效果最佳**（ASR 80.41%），因为它最大程度地破坏了安全机制的模式匹配能力，同时模型仍能理解语义。

**2. 图像打乱(Image Shuffle)**

将有害图像划分为 $m$ 个 patch 块并随机打乱：

$$I' = \text{Shuffle}_p(I), \quad I = [p_1, p_2, \ldots, p_m]$$

实验比较了不同 patch 数目：1、4、9、16、25、64。结果显示 **4 个 patch 块效果最佳**（ASR 80.41%）。patch 数过多会导致模型理解困难，数过少则无法有效绕过安全机制。

**3. 基于查询的黑盒优化**

单纯随机打乱具有不稳定性，并非所有打乱结果都能绕过防御。因此引入迭代优化机制：

- 使用 ChatGPT-3.5 作为毒性评判模型 $\mathcal{J}$
- 毒性评分范围 1-5（1=安全，5=高风险有害）
- 攻击成功阈值 $S_\tau = 4$
- 最大查询迭代次数设为 10
- 每次迭代重新随机打乱文图，评估毒性，若达到阈值则停止

$$\text{ASR} = \frac{\text{sum}\{\mathcal{J}(I, y) \geq S_\tau\}}{N_{\text{total}}}$$

### 损失函数/优化目标

本方法不涉及传统的梯度优化损失函数，而是基于**查询反馈的黑盒优化**。优化目标是最大化目标模型响应的毒性评分，以反馈驱动的方式筛选最有效的打乱组合。算法在每次迭代中：若当前打乱输入的响应毒性评分 $\geq S_\tau$，则攻击成功；否则继续下一轮随机打乱。

## 实验关键数据

### 主实验：三大基准测试

评估模型覆盖 4 个开源模型(LLaVA-NEXT, MiniGPT-4, InternVL-2, VLGuard)和 4 个闭源模型(GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, Qwen-VL-Max)。

**MM-safetybench (with typography) 攻击成功率(ASR%)**：

| 模型 | 原始攻击 | SI-Attack | 提升 |
|------|---------|-----------|------|
| LLaVA-NEXT | 43.99% | 62.68% | +18.69% |
| MiniGPT-4 | 27.20% | 62.44% | +35.24% |
| InternVL-2 | 40.30% | 71.01% | +30.71% |
| VLGuard | 9.52% | 40.77% | +31.25% |
| GPT-4o | 20.77% | 68.57% | +47.80% |
| Claude-3.5-Sonnet | 7.50% | 47.20% | +39.70% |
| Gemini-1.5-Pro | 21.07% | 71.25% | +50.18% |
| Qwen-VL-Max | 33.04% | 68.63% | +35.59% |

**SafeBench (Figstep) 攻击成功率(ASR%)**：

| 模型 | Figstep | SI-Attack | 提升 |
|------|---------|-----------|------|
| LLaVA-NEXT | 44.40% | 74.00% | +29.60% |
| InternVL-2 | 38.60% | 82.60% | +44.00% |
| GPT-4o | 11.80% | 59.20% | +47.40% |
| Claude-3.5-Sonnet | 29.40% | 48.60% | +19.20% |
| Gemini-1.5-Pro | 50.60% | 80.20% | +29.60% |

### 消融实验

**图像与文本打乱组件贡献**（GPT-4o, MM-safetybench 01-IA 子集）：

| 设置 | Toxic Score | ASR(%) |
|------|------------|--------|
| 原始图文 | 1.64 | 13.40% |
| 仅打乱图像 | 2.51 | 35.05% |
| 仅打乱文本 | 3.69 | 67.01% |
| 图文均打乱 | 3.96 | 80.41% |

**查询优化的必要性**：

| 设置 | Toxic Score | ASR(%) |
|------|------------|--------|
| 原始输入 | 1.64 | 13.40% |
| 随机打乱(无优化) | 2.65 | 28.87% |
| 优化后打乱 | 3.96 | 80.41% |

**最大迭代次数影响**：

| 迭代次数 | Toxic Score | ASR(%) |
|---------|------------|--------|
| 1 | 2.65 | 28.87% |
| 5 | 3.75 | 69.07% |
| 10 | 3.96 | 80.41% |
| 20 | 4.01 | 81.44% |

### 关键发现

1. **文本打乱比图像打乱更有效**：仅文本打乱的 ASR(67.01%) 远高于仅图像打乱(35.05%)，表明 MLLMs 在文本侧的安全漏洞更严重
2. **查询优化至关重要**：优化后 ASR 从 28.87% 提升到 80.41%，说明并非所有随机打乱都有效，需要筛选
3. **10 次迭代基本收敛**：20 次迭代仅微幅提升(81.44% vs 80.41%)
4. **对不同规模模型均有效**：InternVL-2 的 4B/8B/26B 版本 ASR 均约 70%，说明方法不受模型规模影响
5. **对抗 PPL 检测器仍有效**：面对困惑度检测防御，SI-Attack 仍保持 71.13% 的 ASR

## 亮点与洞察

1. **发现了一个深刻的安全漏洞**：Shuffle Inconsistency 揭示了 MLLMs 理解能力和安全能力之间的根本性脱节——安全对齐训练未能覆盖打乱后的有害指令空间，这一发现对 AI 安全研究具有重要启示
2. **方法极其简洁**：不需要对抗扰动优化、不需要白盒模型访问、不需要精心设计的 prompt，仅通过随机打乱+查询优化即可实现高效攻击
3. **对闭源商用模型效果显著**：GPT-4o 上 ASR 从 20.77% 提升到 68.57%，Gemini-1.5-Pro 上从 21.07% 提升到 71.25%，表明商用模型的外部安全护栏同样存在该漏洞
4. **PCA 可视化提供了机理解释**：通过可视化开源模型的隐藏状态，清晰展示了模型对原始和打乱输入有不同的内部表示，说明安全对齐训练确实未覆盖此分布
5. **"优势即弱点"的哲学洞察**：当安全能力无法匹配优秀的理解能力时，理解能力反而成为可被利用的弱点

## 局限性

1. **方法相对简单，防御难度不高**：一旦该漏洞被发现，防御方应能通过在安全对齐训练中加入打乱样本来弥补
2. **依赖外部毒性评判模型**：需要 ChatGPT-3.5 做毒性评判，增加了成本和对外部 API 的依赖
3. **打乱策略较为固定**：仅探索了词级和 patch 级打乱，更灵活的打乱粒度（句子级、字符级、语义级）未充分探索
4. **评估局限**：毒性评分由 ChatGPT-3.5 自动评判，可能与人类判断存在偏差
5. **缺乏对最新防御方法的对抗评估**：如对抗训练、输入预处理等新型防御机制下的表现未知
6. **伦理风险**：论文发布了完整的攻击方法，可能被恶意利用

## 相关工作与启发

**越狱攻击方向**：
- **FigStep** [Gong et al., 2023]：将有害文本嵌入排版图像利用 OCR 能力越狱
- **MM-safetybench** [Liu et al., 2023]：生成查询相关图像配合排版进行攻击
- **HADES** [Li et al., 2024]：通过精心制作的图像隐藏和放大有害意图
- 相比上述方法，SI-Attack 更简洁且对闭源模型更有效

**防御方向**：
- **LLama Guard**：微调 LLaMA 检测有害意图
- **VLGuard**：构建视觉-语言安全指令数据集微调 MLLM
- 本文发现 VLGuard 的安全微调也无法抵御 SI-Attack

**启发**：
1. 安全对齐训练需要覆盖更广的输入分布，包括各种变体和扰动形式
2. 理解能力和安全能力应联合训练，确保两者一致
3. 未来防御可考虑在推理时对输入进行规范化(canonicalization)，将打乱输入恢复为标准形式再做安全检查

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | ⭐⭐⭐⭐ | 发现了 Shuffle Inconsistency 这一新颖安全漏洞，视角独到 |
| 技术深度 | ⭐⭐⭐ | 方法本身简洁，但分析充分（PCA 可视化、多维度消融） |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 三大基准、8 个模型、详细消融、自适应攻击实验，非常全面 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，动机阐述有力，图表丰富 |
| 实用价值 | ⭐⭐⭐⭐ | 对 AI 安全研究有重要警示意义，方法简单易复现 |
| **综合** | **⭐⭐⭐⭐** | 出色的安全分析与红队测试工作，发现有深度，实验扎实 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[CVPR 2025\] Distraction is All You Need for Multimodal Large Language Model Jailbreaking](../../CVPR2025/multimodal_vlm/distraction_is_all_you_need_for_multimodal_large_language_model_jailbreaking.md)
- [\[ICLR 2026\] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](../../ICLR2026/multimodal_vlm/shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [\[CVPR 2025\] Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](../../CVPR2025/multimodal_vlm/playing_the_fool_jailbreaking_llms_and_multimodal_llms_with_out-of-distribution_.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
