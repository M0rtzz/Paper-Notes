---
title: >-
  [论文解读] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs
description: >-
  [多模态] 提出了 Mixture of Decoding (MoD)，通过 JS 散度衡量模型对图像 token 注意力的正确性，在注意力正确时采用互补解码放大关键信息，注意力错误时采用对比解码抑制误导信息，从而自适应地缓解多模态大模型的幻觉问题。
tags:
  - 多模态
---

# Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs

- **会议**: ACL 2025
- **arXiv**: [2505.17061](https://arxiv.org/abs/2505.17061)
- **代码**: [xlchen0205/MoD](https://github.com/xlchen0205/MoD)
- **领域**: multimodal_vlm
- **关键词**: 多模态大模型, 幻觉缓解, 对比解码, 注意力机制, 自适应解码

## 一句话总结

提出了 Mixture of Decoding (MoD)，通过 JS 散度衡量模型对图像 token 注意力的正确性，在注意力正确时采用互补解码放大关键信息，注意力错误时采用对比解码抑制误导信息，从而自适应地缓解多模态大模型的幻觉问题。

## 研究背景与动机

大型视觉语言模型 (LVLMs) 在各种视觉任务中表现出色，但"幻觉"问题——模型生成的文本与视觉信息不一致——严重制约了其可靠性。现有的对比解码方法存在明显不足：

**VCD** 和 **M3ID** 主要将幻觉归因于语言先验偏差，VCD 通过给图像加高斯噪声、M3ID 通过移除图像输入来获得幻觉 logits 进行对比，但它们忽视了视觉输入本身（如虚假关联）对幻觉的影响。

**AvisC** 虽然考虑了注意力分布，认为注意力权重过高的图像 token 会触发幻觉，但它一律削弱高注意力 token，没有区分注意力是否正确。当模型已经正确聚焦到相关信息时，AvisC 反而会削弱有用信号，导致不可靠的对比结果。

核心洞察在于：**模型的注意力分布可能是正确的，也可能是错误的**。关键在于判断注意力的正确性，然后动态调整解码策略——这正是 MoD 的出发点。作者发现，用 JS 散度衡量原始输出与基于注意力 token 生成的输出之间的一致性，能够有效区分幻觉输出和非幻觉输出（在 POPE 上体现为非幻觉样本集中在低 JS 散度区域，在 CHAIR 上 JS 散度与 CHAIR_i 的 Pearson 相关系数高达 0.85）。

## 方法详解

### 整体框架

MoD 包含三个核心步骤：

1. **提取注意力图像 token**：利用最后一个输入 token 在所有层和注意力头上的平均注意力权重，选取 top-λ 比例的高注意力图像 token，将其余 token 置零，得到 $v_{att}$。
2. **生成双路 logits**：分别基于原始图像 token $v$ 和注意力图像 token $v_{att}$ 进行前向传播，得到两组输出概率分布。
3. **JS 散度判别 + 自适应解码**：计算两组分布的 JS 散度，低于阈值 $\gamma$ 时采用互补策略，高于阈值时采用对比策略。

### 关键设计一：注意力图像 token 提取

模型利用自回归特性，取输入序列最后一个 token 对所有图像 token 的注意力权重，在所有层和头上取平均：

$$A^I = \frac{1}{L \cdot H} \sum_{l=1}^{L} \sum_{h=1}^{H} A_l[\cdot, h, -1, IDX^I]$$

然后选取注意力权重最高的 top-λ 比例图像 token 的索引 $IDX^I_{att}$，将其余图像 token 置零，得到 $v_{att}$。默认 $\lambda = 0.2$，即保留 20% 的图像 token。

这种设计的优势在于：不依赖特定层或特定头的注意力，而是综合所有层和头的信息，获得模型对图像 token 的全局理解。

### 关键设计二：基于 JS 散度的自适应解码策略

计算两组输出分布的 JS 散度来判断注意力正确性：

$$d(v, v_{att}) = D_{JS}[p_\theta(y_t | v, x, y_{<t}) \| p_\theta(y_t | v_{att}, x, y_{<t})]$$

根据 $d(v, v_{att})$ 与阈值 $\gamma$ 的关系，选择不同的解码策略：

- **注意力正确（$d \leq \gamma$，一致性高）**：互补解码，将两组 logits 相加放大关键信息：

$$y_t \sim \text{softmax}[\text{logit}_\theta(y_t|v,x,y_{<t}) + \alpha_1 \cdot \text{logit}_\theta(y_t|v_{att},x,y_{<t})]$$

- **注意力错误（$d > \gamma$，一致性低）**：对比解码，用原始 logits 减去注意力 logits 以抑制误导信息：

$$y_t \sim \text{softmax}[(1+\alpha_2) \cdot \text{logit}_\theta(y_t|v,x,y_{<t}) - \alpha_2 \cdot \text{logit}_\theta(y_t|v_{att},x,y_{<t})]$$

默认超参数为 $\alpha_1=4$、$\alpha_2=1$、$\gamma=0.05$，在所有任务和模型上共用一套参数，无需针对特定场景调参。

### 关键设计三：一致性作为幻觉指示器

JS 散度能有效区分幻觉的直觉在于：当模型正确关注了相关图像区域时，仅保留这些高注意力 token 所生成的输出应与原始输出高度一致（低 JS 散度）；当模型错误关注了无关区域时，仅保留这些 token 会导致与原始输出的显著偏离（高 JS 散度）。实验验证了这一点——在 POPE 上非幻觉输出集中在低 JS 散度区域，在 CHAIR 上 JS 散度与幻觉率呈强正相关（Pearson r=0.85, p<0.01）。

## 实验关键数据

### 表1：POPE 基准测试（MS-COCO，Random 设置）

| 方法 | LLaVA-1.5 Acc | LLaVA-1.5 F1 | Qwen-VL Acc | Qwen-VL F1 | LLaVA-NEXT Acc | LLaVA-NEXT F1 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Sampling | 83.8 | 84.2 | 84.9 | 82.9 | 84.4 | 82.3 |
| VCD | 85.0 | 84.2 | 85.5 | 83.6 | 86.0 | 84.3 |
| M3ID | 86.1 | 85.0 | 85.3 | 83.4 | 85.5 | 83.6 |
| AvisC | 82.3 | 83.5 | 82.9 | 80.0 | 85.2 | 82.8 |
| **MoD** | **89.2** | **89.1** | **86.0** | **84.1** | **86.6** | **84.8** |

MoD 在所有三种 POPE 设置（random/popular/adversarial）都取得最佳表现，在 LLaVA-1.5 上 Accuracy 超第二名 3.1 点、F1 超 4.1 点。

### 表2：CHAIR 基准测试（生成式描述任务）

| 方法 | LLaVA-1.5 CHAIR_s↓ | CHAIR_i↓ | Recall↑ | LLaVA-NEXT CHAIR_s↓ | CHAIR_i↓ | Recall↑ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Sampling | 52.8 | 15.9 | 77.3 | 35.8 | 12.0 | 59.5 |
| VCD | 51.0 | 14.9 | 77.2 | 40.2 | 10.7 | 62.1 |
| AvisC | 44.0 | 13.7 | 72.9 | 40.4 | 12.4 | 60.0 |
| **MoD** | **42.6** | **12.4** | **78.9** | **33.6** | **9.6** | **61.4** |

MoD 在降低幻觉率的同时保持甚至提升了 Recall，说明不仅减少了错误内容，描述的完整性也没有损失。注意 VCD 和 AvisC 在 LLaVA-NEXT 上反而加重了幻觉。

### 表3：MME 基准测试（MME Score）

| 模型 | Sampling | VCD | M3ID | AvisC | **MoD** |
|------|:---:|:---:|:---:|:---:|:---:|
| LLaVA-1.5 | 510.0 | 531.7 | 553.3 | 596.7 | **638.3** |
| Qwen-VL | 581.7 | 593.3 | 586.7 | 578.3 | **613.3** |
| LLaVA-NEXT | 595.0 | 611.7 | 608.3 | 613.3 | **653.3** |

MoD 在三个模型上分别超过第二名 41.6、20.0 和 40.0 分。

## 关键发现

1. **JS 散度是有效的幻觉指示器**：原始输出与注意力 token 输出的一致性能准确区分幻觉与非幻觉，Pearson 相关系数达 0.85。
2. **自适应优于单一策略**：消融实验表明 MoD 比单独使用互补解码或对比解码分别高出 23.3 和 20.0 分（MME），说明动态切换策略的必要性。
3. **超参数鲁棒性强**：$\gamma$ 在 0.02-0.08 范围内 MoD 均稳定优于单一方法，且所有任务和模型共用同一组超参数，无需逐场景调参。
4. **模型无关性**：在 LLaVA-1.5、Qwen-VL、LLaVA-NEXT 三种不同架构上均取得一致提升，且在某些方法（如 VCD、AvisC）反而加重幻觉的情形下，MoD 仍然稳定有效。
5. **AMBER 综合评分**：MoD 在三个模型上的 AMBER Score 分别比第二名高 2.2、0.7 和 2.6 分，在判别和生成任务上均表现最优。

## 亮点与洞察

- **问题拆解精准**：将"注意力是否正确"作为解码策略的切换条件，抓住了现有方法忽略的核心维度——注意力分布的正确性具有不确定性，不应一刀切地对待。
- **设计简洁优雅**：不需要额外训练、不需要外部知识、不需要反复采样，仅用一个 JS 散度阈值即可实现自适应切换，实现了复杂度与效果的良好平衡。
- **互补解码的创新性**：大多数对比解码工作只关注"减去什么"，MoD 首次引入互补思路——当注意力正确时"加上什么"来放大关键信息，跳出了对比解码的固有范式。
- **Precision 显著提升**：在 POPE 上 Precision 最高超出其他方法 6.8 点，说明 MoD 有效抑制了 LVLM 倾向回答 "Yes" 的偏好，使模型更加审慎。

## 局限性

1. **推理开销翻倍**：与其他对比解码方法一样，MoD 需要两次前向传播，推理延迟近似翻倍。
2. **掩码策略粗糙**：当前直接置零低注意力 token，可能损失位置信息（在 MME 的 position 子集上有轻微下降），更精细的策略（如池化保留部分信息）可能进一步提升效果。
3. **阈值全局固定**：$\gamma=0.05$ 对所有 token 位置一视同仁，但幻觉可能在特定生成阶段更易发生，动态阈值可能更优。
4. **未处理训练数据偏差**：MoD 作为推理时方法，不能解决训练数据中的固有偏见问题。

## 相关工作与启发

本文属于 LVLM 幻觉缓解中的**推理时对比解码**方向。与 VCD（图像加噪）、M3ID（移除图像）、AvisC（高注意力 token 对比）不同，MoD 的核心创新在于**不预设注意力的好坏**，而是用一致性度量来动态判断。同时期的 DeGF 也采用了类似的自适应思路（通过生成图像来判断一致性），但 MoD 的方案更轻量——直接利用模型内部的注意力信息而无需额外的图像生成步骤。

启发：这种"先判别再决策"的二阶段思路可以推广到其他场景，例如在 RAG 中判断检索文档与查询的相关性后再决定是否使用、在多轮对话中判断上下文注意力是否正确后调整生成策略等。

## 评分

⭐⭐⭐⭐ — 方法思路清晰、实验全面、设计简洁有效，通过一致性度量实现自适应解码是一个优雅的解决方案。不足在于推理翻倍的开销和较为粗糙的掩码策略。

<!-- RELATED:START -->

## 相关论文

- [ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference](madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)
- [Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](../../ICCV2025/multimodal_vlm/hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](../../CVPR2025/multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)

<!-- RELATED:END -->
