---
title: >-
  [论文解读] Robust Multi-bit Text Watermark with LLM-based Paraphrasers
description: >-
  [ICML 2025][LLM安全][text watermark] 提出基于LLM释义器（paraphraser）的多比特文本水印方法，通过共训练一对行为差异化的释义器和一个解码分类器，利用PPO强化学习优化编码-解码对，在1.1B小模型上实现>99.99% AUC的检测精度，同时保持文本语义不变。
tags:
  - "ICML 2025"
  - "LLM安全"
  - "text watermark"
  - "multi-bit"
  - "paraphrasing"
  - "PPO"
  - "co-training"
---

# Robust Multi-bit Text Watermark with LLM-based Paraphrasers

**会议**: ICML 2025  
**arXiv**: [2412.03123](https://arxiv.org/abs/2412.03123)  
**代码**: [github.com/xiaojunxu/multi-bit-text-watermark](https://github.com/xiaojunxu/multi-bit-text-watermark)  
**领域**: AI安全 / 文本水印  
**关键词**: text watermark, multi-bit, paraphrasing, PPO, co-training

## 一句话总结

提出基于LLM释义器（paraphraser）的多比特文本水印方法，通过共训练一对行为差异化的释义器和一个解码分类器，利用PPO强化学习优化编码-解码对，在1.1B小模型上实现>99.99% AUC的检测精度，同时保持文本语义不变。

## 研究背景与动机

- 文本水印需要在文本中嵌入不可感知的信号，应用于版权保护和LLM生成文本追踪。
- 现有方法存在局限：
    - **同义词替换方法**（如 NLW）：操作空间有限，鲁棒性差。
    - **LLM输出水印**（如 KGW、KTH）：仅适用于LLM生成的文本，不能水印任意文本。
    - **释义方法**（如 RemarkLLM、Waterfall）：多比特准确率低、检测AUC不足。
- 本文目标：设计一个通用的、高精度、高鲁棒性的多比特文本水印管线。

## 方法详解

### 整体框架

管线分编码和解码两个阶段：
- **编码**：用一对释义器 $(\theta_0, \theta_1)$ 交替释义输入文本的每个句子，根据水印码的当前比特选择释义器。
- **解码**：将文本分段（按句子），对每段用文本分类器 $\theta_d$ 判断属于 class-0 还是 class-1，拼接得到解码水印。

### 关键设计

**双释义器编码**：两个LLM释义器 $\theta_0, \theta_1$ 从同一基座微调而来，但训练过程中通过JS散度损失鼓励差异：

$$\ell_{init}(\theta_0, \theta_1) = \ell_{SFT}(\theta_0) + \ell_{SFT}(\theta_1) - \lambda_{JS} \cdot \text{JS}(\pi_{\theta_0}, \pi_{\theta_1})$$

**解码器训练（交叉熵）**：

$$\ell_D(\theta_d; x^w, M) = \sum_{i} \text{CE}(g_s(\tilde{x}_i^w; \theta_d), M[i])$$

**编码器训练（PPO + 奖励模型）**：

将解码器作为奖励模型，水印奖励为成功解码的比特数：

$$r_w(x^w, M) = \sum_{i} \mathbb{1}\{D(x^w)[i] = M[i]\}$$

总奖励加入语义相似度正则：

$$r(x^w, x^o, M) = \lambda_w \cdot r_w + \lambda_s \cdot r_s(x^w, x^o)$$

PPO损失对两个释义器分别在各自生成的token上更新，同时包含KL散度正则防止偏离参考模型。

### 共训练框架

编码器和解码器交替更新：每步随机采样水印码 $M$，生成水印文本，计算advantage，分别用 $\ell_D$ 更新解码器、用 $\ell_E$ 更新编码器。

## 实验关键数据

### 主实验（C4 RealNewsLike 数据集，128 token）

| 方法 | 比特准确率 | 比特数 | AUC | TPR@FPR=1% | TPR@FPR=0.01% | 相似度 |
|------|:-------:|:----:|:---:|:--------:|:-----------:|:----:|
| RemarkLLM (4bit) | 0.766 | 4.0 | 0.786 | 0.0% | 0.0% | 0.810 |
| KGW (multi-bit) | 0.638 | 4.46 | 0.833 | 22.9% | 6.3% | 0.812 |
| KTH (zero-bit) | - | - | 0.892 | 61.4% | 46.6% | 0.820 |
| Waterfall (κ=1) | - | - | 0.939 | 62.4% | 35.5% | 0.842 |
| **Ours** | **0.956** | **5.57** | **0.998** | **98.0%** | **78.0%** | **0.874** |

### 抗攻击鲁棒性

**词语替换（20%替换率）**：

| 方法 | 比特准确率 | AUC | TPR@1% |
|------|:-------:|:---:|:-----:|
| KTH (zero-bit) | - | 0.813 | 41.5% |
| Waterfall (κ=1) | - | 0.856 | 25.6% |
| Ours | 0.861 | 0.947 | 51.6% |
| Ours (AdvT) | **0.914** | **0.985** | **78.7%** |

**句子释义攻击（LlamaPara）**：

| 方法 | 比特准确率 | AUC | TPR@1% |
|------|:-------:|:---:|:-----:|
| Ours | 0.714 | 0.865 | 43.9% |
| Ours (AdvT) | **0.849** | **0.924** | **36.8%** |

### OOD泛化

在C4上训练的模型，在HH-RLHF上AUC=0.999、PKU SafeRLHF上AUC=0.996，展示良好泛化性。

### 多次运行提升

重复5次释义取最优：比特准确率 > 0.99，检测AUC > 0.9999。

### 隐蔽性验证

GPT进行 in-context learning 分类准确率仅57%，接近随机猜测（50%），证明水印不可被人或AI检测。

## 亮点与洞察

1. **PPO+共训练的巧妙设计**：将解码器作为编码器的奖励模型，形成对抗性共进化，比端到端训练更稳定。
2. **信息密度高**：每23个token编码1比特（128/5.57），优于所有基线。
3. **小模型大能力**：仅用1.1B参数的TinyLlama即可达到极高检测性能，可并行运行5次进一步提升。
4. **对抗训练提升鲁棒性**：在训练中引入扰动（词替换或释义），显著改善抗攻击能力。

## 局限性

- 句子级释义攻击仍会造成较大性能下降，这是所有文本水印方法的固有限制——攻击者可将水印文本释义回原始语义。
- 需要训练专门的编码器-解码器对，部署成本高于零训练方法（如 KGW）。
- 目前基于句子级分段，对于超短文本（1-2句）信息容量有限。
- 超参数 $\lambda_s$ 和 $\lambda_k$ 需要调整以平衡检测力和保真度。

## 相关工作

- **文本水印**：同义词替换 (Topkara et al., 2006)、LSTM释义 (Abdelnabi & Fritz, 2021)、Gumbel softmax (RemarkLLM, Zhang et al., 2024b)、不变特征 (Yoo et al., 2023)。
- **LLM输出水印**：KGW (Kirchenbauer et al., 2023)、KTH (Kuditipudi et al., 2023)、语义水印 (Liu et al., 2023)、Waterfall (Lau et al., 2024)。
- **释义编码器+分类器共训练**：Xu et al. (2024) 提出的框架，本文在此基础上扩展到多比特场景。

## 评分

⭐⭐⭐⭐ — 方法设计优雅（PPO共训练 + 双释义器），实验全面（鲁棒性、OOD、隐蔽性、消融），性能大幅领先基线。但句子释义攻击下的脆弱性和部署成本是实际应用的瓶颈。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](../../ACL2026/llm_safety/xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[ICML 2025\] The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)
- [\[ICML 2025\] TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems](tamas_benchmarking_adversarial_risks_in_multi-agent_llm_systems.md)
- [\[ICML 2025\] Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks](revealing_weaknesses_in_text_watermarking_through_self-information_rewrite_attac.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)

</div>

<!-- RELATED:END -->
