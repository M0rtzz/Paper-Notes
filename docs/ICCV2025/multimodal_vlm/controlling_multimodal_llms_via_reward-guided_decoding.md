---
title: >-
  [论文解读] Controlling Multimodal LLMs via Reward-guided Decoding
description: >-
  [ICCV 2025][多模态][多模态大语言模型] 提出多模态奖励引导解码 (MRGD)，通过构建两个奖励模型分别控制物体精度和召回率，在推理时实现对 MLLM 输出的细粒度可控性，同时显著降低物体幻觉。
tags:
  - ICCV 2025
  - 多模态
  - 多模态大语言模型
  - 奖励引导解码
  - 幻觉缓解
  - 视觉定位
  - 推理时对齐
---

# Controlling Multimodal LLMs via Reward-guided Decoding

**会议**: ICCV 2025  
**arXiv**: [2508.11616](https://arxiv.org/abs/2508.11616)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 多模态大语言模型, 奖励引导解码, 幻觉缓解, 视觉定位, 推理时对齐

## 一句话总结

提出多模态奖励引导解码 (MRGD)，通过构建两个奖励模型分别控制物体精度和召回率，在推理时实现对 MLLM 输出的细粒度可控性，同时显著降低物体幻觉。

## 研究背景与动机

随着多模态大语言模型 (MLLM) 被广泛应用，用户对控制其行为的需求日益增长，核心在两个维度：(a) 控制输出的精度与全面性（如物体召回率），(b) 控制推理所用的计算量。例如，视觉障碍用户希望获得高精度输出以避免幻觉，而用合成描述训练下游模型的用户则更看重多样性和全面性。

现有方法存在明显不足：
- **提示工程 (Prompting)**：控制粗糙，依赖人工设计
- **监督微调 (SFT) / RLHF 微调**：推理时完全不可控，微调学到的原则不一定被遵守
- **特定解码策略 (VCD, CGD)**：效果有限或缺乏多维度控制

对于纯文本 LLM，奖励引导解码已被证明有效，但多模态场景面临独特挑战：奖励模型需同时处理视觉和文本信息，理解生成文本与图像输入之间的交互关系，这种交互会导致特定类型的幻觉。本文首次将奖励引导解码扩展到多模态场景。

## 方法详解

### 整体框架

MRGD 在每个推理步骤中执行如下搜索过程：
1. 从 MLLM 采样 k 个候选续写（以句子为单位，以句号为分隔符）
2. 用两个奖励模型的线性组合对每个候选打分
3. 选择得分最高的候选添加到已生成上下文
4. 重复直到生成 EOS token

最终得分为两个奖励的加权组合：s = w · r_hal + (1-w) · r_rec，其中 w 在 [0,1] 之间，是推理时可调节的引导强度超参数。w=1 时完全关注幻觉抑制，w=0 时完全关注召回提升。

### 关键设计

**1. 幻觉奖励模型 r_hal（学习型）**

- 骨干网络为 PaliGemma (3B)，增加线性回归头将最后一个 token 嵌入映射为标量
- 在多个公开偏好数据集上训练：LLaVA-RLHF (9.4k) + RLHF-V (5.7k) + POVID (17k) + SugarCrepe (7.5k)
- 采用 Bradley-Terry 模型训练偏好分类器，同时用 MSE 正则约束输出范围到 [0,1]
- LoRA 微调骨干，batch size 256，余弦学习率调度，仅训练 1 个 epoch
- 验证集准确率 82.05%，VLFeedback 跨域准确率 67.68%

**2. 召回奖励模型 r_rec（规则型，无需训练）**

由三个现成模块组合而成：
- **OWLv2** (开放词汇检测器)：提取图像中的参考物体
- **NLTK POS 标注器**：从生成文本中提取预测物体
- **Sentence-BERT**：计算参考和预测物体的语义相似度，阈值 tau=0.5

本质上是估计物体召回率：匹配的预测物体数 / 参考物体总数。

**3. 句子级评估策略**

为避免对不完整文本评估不稳定，每 T 个句子调用一次奖励模型。当 T 足够大时退化为 best-of-k 拒绝采样。实验发现更频繁评估（T=1）能提升采样效率。

### 损失函数 / 训练策略

幻觉奖励模型的训练损失结合 Bradley-Terry 偏好损失与 MSE 正则化。MSE 项鼓励正样本得分接近 1、负样本接近 0，便于与召回奖励线性组合，同时避免梯度饱和。

## 实验关键数据

### 主实验

在 COCO (CHAIR) 和 AMBER 两个幻觉基准上评估 LLaVA-1.5 7B：

| 方法 | Ci (↓) | Cs (↓) | Rec. (↑) | CHAIR (↓) | Hal. (↓) | Cov. (↑) |
|------|--------|--------|----------|-----------|----------|----------|
| Greedy | 15.05 | 48.94 | 81.30 | 7.6 | 31.8 | 49.3 |
| VCD | 15.76 | 54.18 | 81.66 | 9.7 | 42.8 | 51.6 |
| CGD | 9.48 | 37.48 | 80.11 | 5.1 | 24.0 | 48.3 |
| MRGD w=1.0 | **4.53** | **18.19** | 76.04 | **3.4** | **15.9** | 52.4 |
| MRGD w=0.5 | 5.34 | 22.54 | 78.63 | 4.4 | 25.4 | **60.8** |

MRGD 也在 Llama-3.2-Vision (11B) 和 SmolVLM-2 (2.2B) 上验证有效，且奖励模型无需重新训练。

### 消融实验

不同权重 w 对精度-召回权衡的影响（LLaVA-1.5, COCO）：

| w | Ci (↓) | Rec. (↑) | Len. |
|---|--------|----------|------|
| 1.0 | 4.53 | 76.04 | 95.90 |
| 0.75 | 4.76 | 76.84 | 96.17 |
| 0.5 | 5.34 | 78.63 | 97.96 |
| 0.25 | 7.67 | 81.56 | 105.34 |
| 0.0 | 24.20 | 85.23 | 108.92 |

w 提供了从低幻觉/低召回到高召回/高幻觉的平滑过渡。MRGD 还可叠加在已经过 RLHF 微调的模型上进一步提升。

### 关键发现

- MRGD w=1.0 将 CHAIRi 从 15.05% 降低约 70% 至 4.53%，召回只下降约 6.5%
- 最优 w 因数据集而异：COCO 约 0.25，AMBER 约 1.0（COCO 图像平均 21.4 个物体 vs AMBER 9.9 个）
- VCD 和 LLaVA-RLHF 在生成式幻觉基准上反而增加了幻觉率
- PaliGemma-2 作为骨干效果略差于 PaliGemma，说明更大的骨干不一定更好

## 亮点与洞察

1. **推理时可控性**：通过单一参数 w 即可动态调节精度-召回权衡，无需重新训练
2. **模型无关性**：训练好的奖励模型可直接应用于不同 MLLM（LLaVA、Llama-3.2、SmolVLM-2）
3. **双奖励的精巧设计**：一个学习型（偏好数据训练），一个规则型（现成模块组合），兼顾效果与成本
4. **揭示生成式 vs 判别式幻觉基准的差异**：VCD 等方法在 VQA 基准有效但在描述生成基准反而变差

## 局限与展望

- 推理成本随采样数 k 线性增长，每步需多次前向传播和奖励评估
- 召回奖励模型依赖检测器精度（63%）和 POS 标注器精度（67%）
- 未在视频理解、多轮对话等更复杂场景验证
- 奖励模型仅关注物体级幻觉，未覆盖属性幻觉、关系幻觉等

## 相关工作与启发

- 与 CGD 相比，训练专用多模态奖励模型效果显著优于使用 CLIP 引导
- 与 RLHF 微调互补，可在已对齐模型上再次提升
- 为多模态推理时计算扩展 (test-time compute scaling) 提供了新方向
- 双奖励混合的思路可推广到控制其他属性（如详细程度、风格等）

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [CaptionSmiths: Flexibly Controlling Language Pattern in Image Captioning](captionsmiths_flexibly_controlling_language_pattern_in_image_captioning.md)
- [G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)
- [CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi-holmes_towards_explainable_and_generalizable_ai-generated_image_detection_v.md)
- [Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)

<!-- RELATED:END -->
