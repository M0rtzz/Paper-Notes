---
title: >-
  [论文解读] PLA: Prompt Learning Attack against Text-to-Image Generative Models
description: >-
  [ICCV 2025][图像生成][对抗攻击] 本文提出 PLA（Prompt Learning Attack），一种针对黑盒 T2I 模型的梯度驱动对抗攻击框架，通过敏感知识编码和多模态相似度损失来学习对抗性 prompt，从而绕过 prompt 过滤器和后置安全检查器，平均 ASR-4 达 90%+，远超现有方法。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "对抗攻击"
  - "T2I安全"
  - "黑盒攻击"
  - "提示学习"
  - "NSFW内容检测"
---

# PLA: Prompt Learning Attack against Text-to-Image Generative Models

**会议**: ICCV 2025  
**arXiv**: [2508.03696](https://arxiv.org/abs/2508.03696)  
**代码**: 无  
**领域**: 扩散模型 / AI安全  
**关键词**: 对抗攻击, T2I安全, 黑盒攻击, Prompt Learning, NSFW内容检测

## 一句话总结

本文提出 PLA（Prompt Learning Attack），一种针对黑盒 T2I 模型的梯度驱动对抗攻击框架，通过敏感知识编码和多模态相似度损失来学习对抗性 prompt，从而绕过 prompt 过滤器和后置安全检查器，平均 ASR-4 达 90%+，远超现有方法。

## 研究背景与动机

**领域现状**：T2I 模型（如 Stable Diffusion、DALL·E 3）已广泛应用于艺术创作和内容生成，但同时面临被滥用生成 NSFW（Not-Safe-For-Work）内容的风险。为此，开发者部署了 prompt 过滤器（基于敏感词列表拦截输入）和后置安全检查器（检测生成图像中的不当内容）两类安全机制。

**现有痛点**：现有黑盒攻击方法（如 SneakyPrompt）大多依赖词替换策略，在有限的搜索空间中寻找替代词来绕过 prompt 过滤器，但搜索空间有限导致攻击成功率不高。相比之下，梯度驱动的训练方法具有更强的优化能力，但在黑盒设置下（无法访问模型内部参数）难以直接应用。

**核心矛盾**：黑盒 T2I 模型不仅隐藏了内部架构和参数，而且安全机制会在检测到 NSFW 内容时中断前向传播并返回黑图，导致传统的基于模型输出的梯度估计方法失效（黑图使得梯度为零）。

**本文目标** (a) 如何在黑盒设置下实现有效的梯度驱动对抗 prompt 学习？(b) 如何解决安全机制返回黑图导致的梯度消失问题？

**切入角度**：利用目标 prompt 中蕴含的敏感信息作为语义引导，结合一个无安全机制的辅助模型生成目标图像，通过多模态（文本-图像、图像-图像）相似度构建可计算梯度的训练目标。

**核心 idea**：通过敏感知识编码保留目标 prompt 的语义意图，结合基于CLIP相似度的多模态损失和改进的零阶梯度优化，在黑盒设置下训练 prompt encoder 生成能绕过双重安全机制的对抗性 prompt。

## 方法详解

### 整体框架

PLA 包含三个核心组件：(1) 敏感知识引导编码（Sensitive Knowledge Guided Encoding），将目标 prompt 编码为包含敏感语义的可学习嵌入；(2) 攻击安全机制的 pipeline，利用预训练语言模型生成对抗 prompt 并尝试绕过黑盒 T2I 模型的安全机制；(3) 多模态损失（Multimodal Loss），通过文本-图像和图像-图像相似度指导梯度优化。

输入是一个包含敏感词的目标 prompt $p_{tar}$，输出是一个不含敏感词但能诱导生成与目标 prompt 语义一致的 NSFW 图像的对抗 prompt $p_{adv}$。

### 关键设计

1. **敏感知识提取模块（SKE）**:

    - 功能：从目标 prompt 中提取敏感语义信息，生成敏感嵌入 $e_{sen}$
    - 核心思路：使用预训练文本编码器 $\mathcal{T}_\theta$ 将 $p_{tar}$ 编码为文本嵌入 $e_{tar} \in \mathbb{R}^d$，然后通过两层投影（低维投影 $W_l \in \mathbb{R}^{d \times d_l}$ + 高维投影 $W_h \in \mathbb{R}^{d_l \times d_s}$）映射为敏感嵌入 $e_{sen} \in \mathbb{R}^{M \times d_s}$
    - 设计动机：利用文本嵌入的高维特征保留目标 prompt 的敏感语义意图，使生成的对抗 prompt 能隐式携带敏感信息而不触发关键词过滤

2. **Prompt 编码器**:

    - 功能：将敏感嵌入融入随机 prompt 的编码过程，生成可学习嵌入 $e_{pe}$
    - 核心思路：给定随机 prompt $p_{ran}$，在编码器的第 $l$ 层注入敏感嵌入：$\hat{e}_l = e_l + \omega \cdot e_{sen}$，其中 $\omega$ 控制敏感信息的融合程度。最终将 $e_{pe}$ 与 $p_{tar}$ 拼接后输入 PLM（如 BERT 或 T5）生成对抗 prompt：$p_{adv} = \mathcal{PLM}([e_{pe}; p_{tar}])$
    - 设计动机：通过中间层注入而非简单拼接，使敏感信息与随机文本特征深度融合，增强对抗 prompt 的隐蔽性

3. **辅助模型生成目标图像**:

    - 功能：使用一个无安全机制的辅助 T2I 模型（如 SDv1.4）生成目标图像 $I_{tar} = \mathcal{M}_s(p_{tar})$
    - 核心思路：由于黑盒模型的安全机制会返回黑图，无法直接获得目标图像，因此借助辅助模型来提供图像层面的监督信号
    - 设计动机：解决黑盒设置下缺乏目标图像参考的问题，为多模态损失提供 image-image 对比信号

### 损失函数 / 训练策略

**多模态损失** $\mathcal{L}_{MS}$ 由两部分组成：

- 文本-图像相似度损失：$\mathcal{L}_a = 1 - \cos(\mathcal{T}_{en}(p_{tar}), \mathcal{V}_{en}(I_{gen}))$，利用 CLIP 的文本/图像编码器度量目标 prompt 和生成图像的语义一致性
- 图像-图像相似度损失：$\mathcal{L}_b = 1 - \cos(\mathcal{V}_{en}(I_{tar}), \mathcal{V}_{en}(I_{gen}))$，度量辅助模型生成的目标图像与黑盒模型实际生成图像的一致性

**梯度优化**：由于黑盒设置无法直接反传梯度，采用改进的零阶优化（ZOO）。传统 ZOO 通过有限差分估计梯度 $g_1(\varsigma) = \frac{\mathcal{L}_{MS}(\varsigma + c \cdot \Delta) - \mathcal{L}_{MS}(\varsigma - c \cdot \Delta)}{2c \cdot \Delta}$，但当两个扰动均生成黑图时梯度为零。改进方法引入历史梯度动量：$g_2(\varsigma) = \beta \hat{g}_2 + (1 - \beta) \eta \cdot g_1(\varsigma + \hat{g}_2)$，当当前梯度为零时沿历史方向继续更新。此外提出"重启"策略：首步即遇到黑图时，用高斯噪声替代黑图参与梯度计算。

## 实验关键数据

### 主实验

在 I2P 数据集上评估，使用 100 条 nudity prompts 和 30 条 violence prompts，攻击 SDv1.5、SDXLv1.0、SLD 三个黑盒模型，结合 SC、Q16、MHSC 三种后置安全检查器。

| 模型 | 方法 | AVG ASR-4 (Nudity) | AVG ASR-1 (Nudity) | AVG ASR-4 (Violence) | AVG ASR-1 (Violence) |
|------|------|---------------------|---------------------|----------------------|----------------------|
| SDv1.5 | MMA-Diffusion | 77.76 | 58.38 | 78.26 | 61.04 |
| SDv1.5 | **PLA-BERT** | **91.45** | **68.69** | **88.62** | **69.51** |
| SDXLv1.0 | MMA-Diffusion | 73.30 | 45.24 | 75.53 | 50.61 |
| SDXLv1.0 | **PLA-BERT** | **90.57** | **71.43** | **86.95** | **66.61** |
| SLD | MMA-Diffusion | 76.48 | 53.00 | 76.45 | 56.95 |
| SLD | **PLA-BERT** | **90.82** | **69.30** | **89.20** | **72.03** |

在线服务攻击（Stability.ai 和 DALL·E 3）上 PLA-T5 的 ASR-4 分别达到 69.70% 和 51.98%（violence），均大幅超越其他方法。

### 消融实验

| 配置 | ASR-4 (Violence) | ASR-1 (Violence) | ASR-4 (Nudity) | ASR-1 (Nudity) |
|------|-------------------|-------------------|-----------------|-----------------|
| $\mathcal{L}_a + \mathcal{L}_b$ (Full) | 93.34 | 79.62 | 93.41 | 75.60 |
| w/o $\mathcal{L}_a$ | 81.02 | 54.57 | 82.99 | 51.07 |
| w/o $\mathcal{L}_b$ | 79.34 | 47.88 | 74.66 | 44.87 |
| $G_{PLA}$ (完整梯度) | 91.69 | 70.23 | 95.37 | 76.20 |
| $G_{ZOO}$ (传统零阶) | 52.89 | 46.73 | 58.44 | 41.27 |
| $G_{RE}$ (无重启策略) | 70.12 | 58.24 | 78.33 | 53.90 |

### 关键发现

- 图像-图像相似度损失 $\mathcal{L}_b$ 的贡献比文本-图像损失 $\mathcal{L}_a$ 更大，去掉后 ASR 下降更严重。这说明目标图像中包含更丰富的敏感信息。
- 改进的梯度优化方法 $G_{PLA}$ 显著优于传统零阶优化 $G_{ZOO}$（ASR-4 差距约 35%），验证了历史梯度动量对于解决黑图导致的梯度消失问题的有效性。
- "重启"策略对于首步梯度消失的情况至关重要，缺失后 ASR-4 下降约 17-20 个百分点。
- PLA-BERT 和 PLA-T5 在不同数据集上各有优势，说明不同 PLM 对不同类型敏感内容有不同的"偏好"。

## 亮点与洞察

- **多模态损失设计巧妙**：在无法访问黑盒模型参数的情况下，通过辅助模型+CLIP 相似度构建了有效的梯度信号。这种"借助辅助模型桥接黑盒"的思路可以迁移到其他黑盒优化任务。
- **梯度消失的解决方案实用**：安全机制返回黑图导致的梯度消失是黑盒攻击中的独特挑战，通过历史梯度动量和高斯噪声重启策略巧妙解决，该思路可推广到其他存在梯度消失的零阶优化场景。
- **系统性的安全评估**：覆盖了三种黑盒模型、三种安全检查器、两种在线服务，提供了 T2I 安全性的全面评估视角。

## 局限与展望

- 论文侧重攻击但对防御改进的建议较少，未讨论如何根据 PLA 的攻击模式设计更鲁棒的防御机制
- 辅助模型（SDv1.4）和黑盒模型可能共享类似的架构偏好，对完全不同架构的 T2I 模型（如 autoregressive 模型）的迁移性未验证
- 评估仅限于 nudity 和 violence 两个概念，对其他敏感类别（如仇恨、自残等）的效果未探讨
- PLM 生成的对抗 prompt 的可读性和自然性未做量化评估

## 相关工作与启发

- **vs MMA-Diffusion**: MMA-Diffusion 是白盒攻击，需要访问模型内部参数；PLA 在黑盒设置下仍能大幅超越其白盒性能，说明多模态相似度损失提供了足够的优化信号
- **vs SneakyPrompt**: 采用强化学习进行词替换搜索，受限于有限搜索空间；PLA 通过连续优化 prompt encoder 参数绕过了离散搜索的瓶颈
- 该工作揭示了当前 T2I 安全机制的脆弱性，对红队测试和安全防御研究有重要参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 黑盒梯度攻击设计有创意，但核心思路（CLIP相似度+零阶优化）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多模型、多检查器、在线服务，消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 对 T2I 安全研究有重要意义，但需注意潜在滥用风险

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](../../CVPR2025/image_generation/implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[ICML 2025\] PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs](../../ICML2025/image_generation/pak-ucb_contextual_bandit_an_online_learning_approach_to_prompt-aware_selection_.md)
- [\[ICCV 2025\] FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning](feddifrc_unlocking_the_potential_of_text-to-image_diffusion_models_in_heterogene.md)
- [\[CVPR 2025\] Minority-Focused Text-to-Image Generation via Prompt Optimization](../../CVPR2025/image_generation/minority-focused_text-to-image_generation_via_prompt_optimization.md)

</div>

<!-- RELATED:END -->
