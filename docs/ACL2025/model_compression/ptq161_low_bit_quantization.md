---
title: >-
  [论文解读] PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models
description: >-
  [ACL 2025][模型压缩][量化] 提出 PTQ1.61，首个将 LLM 权重有效压缩到真正 sub-2-bit（1.61-bit）的后训练量化方法，通过一维结构化掩码（仅增加 0.0002-bit 开销）、分块缩放因子优化和量化预处理三项技术实现 SOTA 性能。 - 现有问题：已有极低比特 PTQ 方法（如 PB-…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "量化"
  - "low-bit"
  - "LLM compression"
  - "binarization"
  - "structured mask"
---

# PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2502.13179](https://arxiv.org/abs/2502.13179)  
**代码**: [zjq0455/PTQ1.61](https://github.com/zjq0455/PTQ1.61)  
**领域**: Model Compression  
**关键词**: post-training quantization, low-bit, LLM compression, binarization, structured mask  

## 一句话总结

提出 PTQ1.61，首个将 LLM 权重有效压缩到真正 sub-2-bit（1.61-bit）的后训练量化方法，通过一维结构化掩码（仅增加 0.0002-bit 开销）、分块缩放因子优化和量化预处理三项技术实现 SOTA 性能。

## 研究背景与动机

- **现有问题**：已有极低比特 PTQ 方法（如 PB-LLM、BiLLM）虽声称 sub-2-bit，但使用的非结构化细粒度掩码无法压缩，实际等效位宽分别为 2.7-bit 和 2.1-bit，均超过 2-bit。
- **掩码开销**：PB-LLM 和 BiLLM 使用逐元素的非结构化掩码来区分显著权重，每个权重额外引入 1-bit 以上的存储开销，导致真实压缩率远低于标称值。
- **缩放因子局限**：先前方法独立解析推导每行的缩放因子，忽略了权重矩阵中行间的隐式依赖关系和角度偏差。
- **本文方案**：提出三项关键技术——(1) 基于输入激活的一维结构化掩码将显著通道保留为 4-bit；(2) 分块缩放因子优化考虑行间依赖和角度偏差；(3) 量化预处理将散乱的显著权重变换为行式集中分布。

## 方法详解

### 整体框架

PTQ1.61 的量化流程：(1) 使用量化预处理通过恢复性 LoRA 将预训练模型权重变换为行式集中分布；(2) 基于输入激活通道幅度的一维掩码选择显著权重通道保留在 4-bit；(3) 对非显著通道进行二值化，使用分块缩放因子优化框架学习最优缩放参数。

### 关键设计

- **一维结构化掩码**：通过分析量化误差上界 $\mathcal{E} \leq \sum_{i=1}^{m}(|x_i| \sum_{j=1}^{n}|w_{i,j}^q - w_{i,j}|)$，发现输入激活幅度比权重大约 1000 倍（尤其是 top-20% 通道），因此提出按通道级别保留显著权重行到 4-bit。这种一维掩码只需要 0.0002-bit 的额外存储（相比 PB-LLM 的 1-bit 和 BiLLM 的 1.1-bit）。
- **分块缩放因子优化**：将 MSE 损失和余弦相似度负对数损失联合优化：$\mathbb{E}(f_1, f_2) = \|f_1 - f_2\|_2 + \mathcal{D}_{NLC}(f_1, f_2)$，同时考虑量化误差传播和输出差异两条分支。
- **量化预处理**：发现预训练模型的显著权重分布是散乱的，不适合逐通道量化。通过在预训练数据集上进行轻量级恢复性 LoRA 微调，将显著权重变换为行式集中分布，使模型更适合后续的通道级量化。

### 损失函数

分块优化的目标函数为：

$$\arg\min_{\alpha_s^*, \alpha_r^*} \big(\mathbb{E}(\mathcal{F}(X,W), \mathcal{F}(X_q, W_q')) + \mathbb{E}(\mathcal{F}(X_q, W), \mathcal{F}(X_q, W_q'))\big)$$

其中第一项减少量化误差传播，第二项量化同输入下的输出差异。

## 实验

### 主实验结果（WikiText2 Perplexity，越低越好）

| 方法 | Bits | LLaMA-7B | LLaMA-13B | LLaMA-30B | LLaMA-65B | LLaMA2-7B | LLaMA2-13B |
|------|------|----------|-----------|-----------|-----------|-----------|------------|
| FP | 16 | 5.68 | 5.09 | 4.10 | 3.53 | 5.47 | 4.88 |
| GPTQ | 2 | 2.1e3 | 5.5e3 | 1.9e3 | 55.91 | 7.7e3 | 2.1e3 |
| OmniQuant | 2 | 15.47 | 13.21 | 8.81 | 7.58 | 37.37 | 17.21 |
| PB-LLM | 1.7(+1) | 102.19 | 48.11 | 26.37 | 12.91 | 66.30 | 462.84 |
| BiLLM | 1(+1.1) | 35.04 | 15.14 | 10.52 | 8.51 | 32.48 | 21.77 |
| **PTQ1.61** | **1.61** | **12.50** | **9.67** | **7.95** | **7.02** | **12.70** | **9.74** |

### 消融实验（下游任务平均准确率，LLaMA-7B）

| 方法 | Bits | PIQA | ARC-e | HellaS | WinoG | ARC-c | LAMB-o | 平均 |
|------|------|------|-------|--------|-------|-------|--------|------|
| BiLLM | 1(+1.1) | 61.10 | 40.99 | 31.80 | 53.67 | 20.64 | 23.15 | 36.00 |
| PTQ1.61 | 1.61 | 63.71 | 49.62 | 35.73 | 56.75 | 25.26 | 38.93 | 41.14 |
| FP | 16 | 78.67 | 75.29 | 56.99 | 70.01 | 41.81 | 73.57 | 63.06 |

### 关键发现

1. PTQ1.61 在所有 LLaMA 变体上显著优于 2-bit 的 GPTQ 和 OmniQuant，在真正更低的位宽下实现了更好的性能。
2. 相比标称 1-bit 但实际 2.1-bit 的 BiLLM，PTQ1.61 在真实 1.61-bit 下 Perplexity 低约 2-6 倍（LLaMA-7B：12.50 vs 35.04）。
3. 量化预处理策略可通用地应用于其他极低比特 PTQ 方法并带来显著提升，验证了"预训练模型不一定是量化最优起点"这一洞察。
4. 模型规模越大，PTQ1.61 与全精度的性能差距越小（LLaMA-65B 仅 PPL 7.02 vs 3.53），暗示更大模型更具量化友好性。

## 亮点

- 首个将 LLM 权重真正压缩到 sub-2-bit（1.61-bit）的 PTQ 方法，掩码开销仅 0.0002-bit/weight。
- 提出"量化预处理"新范式，通过恢复性 LoRA 将显著权重变换为行式集中分布，与已有 QLoRA 等后量化微调方法在目标和方法上有本质区别。
- 从数学角度分析量化误差上界的结构化因子，找到输入激活通道作为关键影响因素。

## 局限性

- 量化预处理需要在预训练数据集（如 RedPajama）上进行 LoRA 微调，引入额外计算开销。
- 1.61-bit 压缩下仍有显著性能损失（LLaMA-7B PPL 12.50 vs FP 5.68），在精度敏感的应用中可能不可接受。
- 实验主要集中在 LLaMA 和 OPT 系列，对其他架构（如 Mistral、Qwen）的适用性有待验证。
- 4-bit 显著通道比例固定，未探索自适应比例的影响。

## 相关工作

- **LLM PTQ**：GPTQ（Hessian 矩阵列式量化）、AWQ（激活感知保留 1% 显著权重）、SmoothQuant（平滑通道异常值）、OmniQuant（联合优化平滑和量化参数）。
- **极低比特量化**：BNN/XNOR-Net（经典二值化）、PB-LLM（10% 8-bit + 非结构化掩码）、BiLLM（多组二值化 + 细粒度掩码）。
- **后量化微调**：QLoRA、QA-LoRA 等在量化后进行任务微调，与本文"在量化前做预处理"的思路互补。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 5 |
| 实用性 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 总评 | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)
- [\[ACL 2025\] Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)
- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](../../AAAI2026/model_compression/quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)

</div>

<!-- RELATED:END -->
