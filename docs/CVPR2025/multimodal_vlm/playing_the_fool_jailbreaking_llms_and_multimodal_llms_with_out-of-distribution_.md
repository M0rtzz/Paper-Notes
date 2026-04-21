---
title: >-
  [论文解读] Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy
description: >-
  [CVPR 2025][多模态][越狱攻击] 提出 JOOD 框架，通过将恶意输入进行分布外（OOD）化变换（如图像/文本混合），大幅提升模型不确定性，从而绕过 LLM 和 MLLM 的安全对齐防护，实现高成功率的黑盒越狱攻击。
tags:
  - CVPR 2025
  - 多模态
  - 越狱攻击
  - 安全对齐
  - 分布外输入
  - 多模态安全
  - 数据变换
---

# Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy

**会议**: CVPR 2025  
**arXiv**: [2503.20823](https://arxiv.org/abs/2503.20823)  
**代码**: https://github.com/naver-ai/JOOD (有)  
**领域**: 多模态VLM  
**关键词**: 越狱攻击, 安全对齐, 分布外输入, 多模态安全, 数据变换

## 一句话总结

提出 JOOD 框架，通过将恶意输入进行分布外（OOD）化变换（如图像/文本混合），大幅提升模型不确定性，从而绕过 LLM 和 MLLM 的安全对齐防护，实现高成功率的黑盒越狱攻击。

## 研究背景与动机

当前 LLM/MLLM 通过 RLHF 等方法进行了安全对齐，能有效拒绝直接的恶意请求。但安全对齐的训练数据分布是有限的——如果恶意输入被变换为"训练时未见过的形式"（即 OOD 化），模型能否仍然正确识别恶意意图？

作者的核心观察是：当恶意输入被 OOD 化后，模型对输入恶意性的判断不确定性显著增加，安全护栏因此失效。即使是简单的 mixup 操作也能达到这一效果。这揭示了 RLHF 安全对齐泛化能力的根本缺陷——它仅在训练分布内有效。

## 方法详解

### 整体框架

JOOD 是一个黑盒越狱攻击框架，核心思路是将恶意输入通过现成的数据变换技术（文本混合/图像混合）从安全对齐的训练分布中"推出去"，生成 OOD 化的输入。模型因从未在安全训练中见过此类输入，无法触发安全护栏，从而输出有害响应。

### 关键设计

1. **文本 OOD 化攻击（Eq. 1）**:
    - 功能：对纯文本 LLM 进行越狱
    - 核心思路：将恶意关键词（如 "bomb"）与无关词（如 "apple"）混合成新造词（如 "bombapple"），再提示模型"请分别回答这个混合词中包含的每个词的请求"。变换公式为 $T_i^{\text{ood}} = f(T^h; \varphi_i)$，其中 $\varphi_i$ 是随机采样的辅助词
    - 设计动机：混合后的无意义词在安全对齐训练中从未出现过，模型无法明确判断其恶意性，不确定性大增

2. **图像 OOD 化攻击（Eq. 3）**:
    - 功能：对多模态 MLLM 进行越狱
    - 核心思路：将恶意图像 $I^h$ 与无关辅助图像 $\varphi_i$ 通过 mixup 混合：$I_{(i,j)}^{\text{ood}} = \alpha_j \varphi_i + (1 - \alpha_j) I^h$，同时改写文本指令为"这张图中有两个物体，请告诉我如何制造它们"
    - 设计动机：混合后的图像在视觉编码器中产生不同于原始恶意图像的嵌入分布偏移，无法触发 LLM 后端的安全护栏

3. **评估框架（Score-based Evaluation）**:
    - 功能：量化衡量攻击效果的最大潜在风险
    - 核心思路：使用独立 LLM $\theta^{hf}$ 对每个响应打 0-10 的有害性分数（HF），取所有变换参数下的最高分作为该指令的攻击得分；同时用二值判断模型 $\theta^{bj}$ 计算攻击成功率 ASR
    - 设计动机：不同变换参数产生的响应有害程度不同，需要评估攻击的"最坏情况"风险

### 损失函数 / 训练策略

JOOD 是推理时攻击，不涉及训练。攻击参数包括：辅助样本数 $n=5$，mixup 系数 $\alpha$ 从 $\{0.1, 0.2, \ldots, 0.9\}$ 采样 $m=9$ 个值。整个过程为黑盒，无需访问模型梯度或参数。

## 实验关键数据

### 主实验

| 攻击场景 | 指标 | JOOD (GPT-4V) | FigStep-Pro | HADES | 提升 |
|--------|------|------|----------|------|------|
| Bombs/Explosives | ASR% | **63%** | 23% | 0% | +40% vs FigStep-Pro |
| Hacking | ASR% | **74%** | 32% | 0% | +42% vs FigStep-Pro |
| Drugs | ASR% | **23%** | 25% | 3% | 竞争力 |
| Firearms/Weapons | ASR% | **47%** | 17% | 0% | +30% vs FigStep-Pro |

在开源模型上，JOOD 攻击 LLaVA-1.5-13B 在 Bombs 场景达到 100% ASR（HF=9.8），远超所有基线。

### 消融实验

| 配置 | BE-HF | BE-ASR% | 说明 |
|------|---------|------|------|
| Vanilla (α=0) | 0 | 0% | 原始恶意图像被拒绝 |
| Mixup (α∈(0,1)) | ~7.1 | ~63% | OOD化后安全护栏失效 |
| α=1 (纯辅助图像) | 降低 | 降低 | 恶意语义丢失 |
| Typography辅助图 | 更高 | 更高 | 排版文字更有效 |
| Realistic辅助图 | 稍低 | 稍低 | 真实图片效果稍弱 |
| 相似辅助图 | 低 | 低 | 负相关：越相似越安全 |
| 不相似辅助图 | 高 | 高 | 越不相似越有效 |

### 关键发现

- 辅助图像与恶意图像的语义相似度与攻击有害性呈**强负相关**——不相似的辅助样本攻击效果最好
- 即使加入系统提示防御（System Prompt Defense），JOOD 仅 ASR 下降 3%，而 FigStep-Pro 下降 10%
- 甚至对 GPT-4o 和 o1 等最新模型也能成功越狱

## 亮点与洞察

- **极简主义的强效攻击**：不需要对抗训练、梯度优化或模型访问权限，仅用 mixup 等现成变换即可攻破 SOTA 模型
- **OOD 视角的安全分析**：首次系统性地从分布外泛化角度审视安全对齐的脆弱性，揭示了 RLHF 的根本局限
- **不确定性分析**：通过实验证明 OOD 化输入确实显著提高了模型判断恶意性的不确定性（Figure 1）

## 局限与展望

- 该工作揭示漏洞而非提供修复方案，需要后续研究如何增强安全对齐在 OOD 输入上的泛化
- 目前仅测试了简单的变换技术，更高级的生成式变换（如扩散模型生成）可能更具威胁
- 评估依赖另一个 LLM 打分，可能存在评估偏差
- 仅在英语场景下测试，多语言场景下 OOD 效果未知

## 相关工作与启发

- 与之前 MLLM 越狱方法（FigStep、HADES）的本质区别在于：前者利用视觉编码器的弱安全对齐，而 JOOD 直接将输入推出安全对齐的训练分布
- 为多模态安全对齐研究提供了新方向：安全训练不仅要覆盖恶意内容本身，还需覆盖其各种变换形式
- 启发：数据增强和 OOD 检测技术可被引入安全对齐训练流程

## 评分
- 新颖性: ⭐⭐⭐⭐ 从 OOD 视角分析安全对齐脆弱性，观点新颖但攻击手段较简单
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多场景、丰富消融实验，包含防御对抗测试
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表信息量大
- 价值: ⭐⭐⭐⭐ 对 AI 安全研究有重要警示意义，但属攻击端工作缺乏防御方案

<!-- RELATED:START -->

## 相关论文

- [On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](../../ACL2025/multimodal_vlm/mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)
- [Distraction is All You Need for Multimodal Large Language Model Jailbreaking](distraction_is_all_you_need_for_multimodal_large_language_model_jailbreaking.md)
- [From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)
- [Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/multimodal_vlm/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)

<!-- RELATED:END -->
