---
title: >-
  [论文解读] Latent Guard: a Safety Framework for Text-to-Image Generation
description: >-
  [ECCV 2024][图像生成] 提出 Latent Guard 框架，通过在文本编码器的潜在空间中学习黑名单概念与输入提示词的嵌入映射，实现高效、灵活且可抵御对抗攻击的文本到图像生成安全检测。
tags:
  - ECCV 2024
  - 图像生成
---

# Latent Guard: a Safety Framework for Text-to-Image Generation

**会议**: ECCV 2024  
**arXiv**: [2404.08031](https://arxiv.org/abs/2404.08031)  
**领域**: 图像生成

## 一句话总结

提出 Latent Guard 框架，通过在文本编码器的潜在空间中学习黑名单概念与输入提示词的嵌入映射，实现高效、灵活且可抵御对抗攻击的文本到图像生成安全检测。

## 研究背景与动机

文本到图像（T2I）模型如 DALL-E 3 和 Stable Diffusion 可能被滥用生成不当内容。现有安全措施主要有两类：

**文本黑名单**：通过关键词匹配拦截，但容易被改写绕过

**有害内容分类器**：依赖大规模数据训练，计算成本高，灵活性差

两者均无法有效应对同义词替换和对抗性文本攻击。本文提出一种新思路：不直接分类安全/不安全，而是在潜在空间中检测黑名单概念是否存在于输入提示词中，从而兼顾效率、灵活性和鲁棒性。

## 方法详解

### 整体框架

Latent Guard 由三个阶段组成：
1. **训练数据生成**：利用 LLM（Mixtral 8x7B）生成以黑名单概念为中心的不安全提示词 $\mathcal{U}$，并通过 LLM 移除有害概念生成对应的安全提示词 $\mathcal{S}$
2. **嵌入映射层（Embedding Mapping Layer）**：在预训练文本编码器（CLIP）之上添加可训练的交叉注意力模块，将概念 $c$ 和提示词 $p_{T2I}$ 映射到联合潜在空间
3. **对比学习训练**：通过监督对比损失拉近不安全提示词与对应概念的嵌入，推远安全提示词的嵌入

### 关键设计

**嵌入映射层架构**：采用多头交叉注意力机制（最优配置 $I=16, d=128$），以概念嵌入 $z_c$ 作为 Query、提示词嵌入 $z_p$ 作为 Key 和 Value，通过注意力权重自动学习哪些 token 与黑名单概念相关。例如在"a man gets murdered"中，自动聚焦于"murdered"这个与"murder"概念相关的 token。

**推理机制**：对输入提示词，计算其与所有黑名单概念嵌入的余弦距离，若存在任意概念的距离低于阈值 $\gamma$，则判定为不安全并阻止生成。概念嵌入可预计算存储，部署开销极低。

**测试时灵活性**：黑名单 $\mathcal{C}_{check}$ 可在测试时任意修改（增删概念），无需重新训练。

### 损失函数

采用监督对比损失 $\mathcal{L}_{cont}$：
- **锚点**：概念嵌入 $h_c^b$
- **正样本**：包含该概念的不安全提示词嵌入 $h_{u_c}^b$
- **负样本**：其他不安全提示词 $h_{u_c}^{\bar{b}}$ + 对应安全提示词 $h_{s_c}^b$ + 其他安全提示词 $h_{s_c}^{\bar{b}}$

安全提示词作为负样本的作用尤为关键——帮助交叉注意力将概念相关特征与无关特征解耦。

## 实验关键数据

### 主实验

在自建 CoPro 数据集（723 有害概念、226,104 条提示词）和两个外部数据集上评估：

| 方法 | Explicit_ID (Acc) | Synonym_ID (Acc) | Adversarial_ID (Acc) | Explicit_OOD (Acc) | Synonym_OOD (Acc) | Adversarial_OOD (Acc) |
|---|---|---|---|---|---|---|
| Text Blacklist | 0.805 | 0.549 | 0.587 | 0.895 | 0.482 | 0.494 |
| CLIPScore | 0.628 | 0.557 | 0.504 | 0.672 | 0.572 | 0.533 |
| BERTScore | 0.632 | 0.549 | 0.509 | 0.739 | 0.594 | 0.512 |
| LLM (7B) | 0.747 | 0.764 | 0.867 | 0.746 | 0.757 | 0.862 |
| **Latent Guard** | **0.868** | **0.828** | **0.829** | **0.867** | **0.824** | **0.819** |

外部数据集泛化测试（AUC）：

| 方法 | Unsafe Diffusion | I2P++ |
|---|---|---|
| CLIPScore | 0.641 | 0.299 |
| BERTScore | 0.749 | 0.697 |
| **Latent Guard** | **0.873** | **0.749** |

### 消融实验

| 变体 | Exp_ID AUC | Syn_ID AUC | Adv_ID AUC | Exp_OOD AUC | Syn_OOD AUC | Adv_OOD AUC |
|---|---|---|---|---|---|---|
| **Latent Guard (完整)** | **0.985** | **0.914** | **0.908** | **0.944** | **0.913** | **0.915** |
| 去掉交叉注意力 | 0.975 | 0.908 | 0.818 | 0.947 | 0.896 | 0.866 |
| 去掉安全提示词 | 0.922 | 0.607 | 0.587 | 0.813 | 0.611 | 0.617 |

### 关键发现

- 去掉安全提示词后，Synonym 和 Adversarial 场景性能暴跌（0.914→0.607, 0.908→0.587），证明安全提示词对概念解耦至关重要
- 推理延迟仅约 1ms（578 概念），GPU 内存占用 13MB，可无缝集成到现有 T2I 管线
- 训练仅需 30 分钟（单张 3090 GPU，1000 iterations）
- t-SNE 可视化显示潜在空间自动涌现了安全/不安全区域的清晰分离

## 亮点与洞察

1. **问题重构**：将安全检测从二分类问题转化为概念检测问题，实现测试时黑名单动态更新
2. **极低计算开销**：嵌入映射层仅 130 万参数，相对于 CLIP 文本编码器的 6300 万参数可忽略不计
3. **对抗鲁棒性**：由于在文本编码器的潜在空间上操作，对以文本编码器为目标的对抗攻击具有固有鲁棒性
4. **LLM 数据合成管线**：构建了首个包含概念-提示词配对的安全评估数据集 CoPro

## 局限性

- 基于黑名单的方法本质上受限于黑名单的完备性——减少 $\mathcal{C}_{check}$ 至 10% 时性能显著下降
- 对完全未见过类型的有害内容（不在训练概念分布内）可能效果有限
- 仅在文本层面检测，无法处理视觉层面的安全问题（如安全文本生成不安全图像的情况）
- CoPro 数据集由 LLM 合成，可能无法完全覆盖真实世界中恶意用户的攻击模式
- 当前仅在 CLIP 文本编码器上验证，对其他文本编码器（如 T5）的适用性未知

## 评分

- **创新性**: ⭐⭐⭐⭐ — 概念检测范式新颖，潜在空间安全检测思路独特
- **实用性**: ⭐⭐⭐⭐⭐ — 极低开销、即插即用、黑名单可动态更新
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、四个基线、丰富的消融实验
- **论文质量**: ⭐⭐⭐⭐ — 写作清晰，动机明确，方法与实验衔接紧密

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] LCM-Lookahead for Encoder-based Text-to-Image Personalization](lcm-lookahead_for_encoder-based_text-to-image_personalization.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)

<!-- RELATED:END -->
