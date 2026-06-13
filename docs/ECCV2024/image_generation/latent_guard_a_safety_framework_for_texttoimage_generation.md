---
title: >-
  [论文解读] Latent Guard: A Safety Framework for Text-to-Image Generation
description: >-
  [ECCV 2024][图像生成][text-to-image safety] 提出Latent Guard框架，在T2I模型文本编码器之上学习一个潜在空间，通过对比学习将黑名单概念与包含该概念的输入prompt映射到相近位置，实现高效的不安全prompt检测（ID Explicit AUC 0.985），支持黑名单测试时灵活更新且无需重训练。
tags:
  - "ECCV 2024"
  - "图像生成"
  - "text-to-image safety"
  - "blacklist"
  - "对比学习"
  - "latent space"
  - "adversarial robustness"
---

# Latent Guard: A Safety Framework for Text-to-Image Generation

**会议**: ECCV 2024  
**arXiv**: [2404.08031](https://arxiv.org/abs/2404.08031)  
**代码**: [项目页面](https://latentguard.github.io/)  
**领域**: T2I安全 / 内容审核  
**关键词**: text-to-image safety, blacklist, contrastive learning, latent space, adversarial robustness

## 一句话总结

提出Latent Guard框架，在T2I模型文本编码器之上学习一个潜在空间，通过对比学习将黑名单概念与包含该概念的输入prompt映射到相近位置，实现高效的不安全prompt检测（ID Explicit AUC 0.985），支持黑名单测试时灵活更新且无需重训练。

## 研究背景与动机

**领域现状**：T2I模型（DALL·E 3、Stable Diffusion等）可生成高质量图像，但也可被用于生成deepfake、暴力、歧视等不安全内容。现有安全措施包括文本黑名单、LLM审查、NSFW图像分类器等。

**现有痛点**：(1) **文本黑名单**容易被同义词改写或对抗攻击绕过；(2) **LLM审查**计算代价高且可被对抗prompt绕过；(3) **图像分类器**需要先生成图像再判别，浪费计算资源；(4) **概念遗忘**方法需要昂贵的微调且黑名单不可灵活更新。

**核心矛盾**：需要一个既能检测改写/对抗攻击、又能灵活更新黑名单、且计算高效的安全框架。

**本文要解决什么？** 在T2I文本编码器的潜在空间中高效检测黑名单概念的存在，实现灵活且鲁棒的安全防护。

**切入角度**：不做安全/不安全的二分类，而是检测prompt中是否包含特定黑名单概念——在潜在空间中度量概念与prompt的距离。

**核心idea一句话**：通过对比学习训练Embedding Mapping Layer，使包含黑名单概念的prompt嵌入与该概念嵌入在潜在空间中距离接近，检测时只需计算余弦距离即可判断。

## 方法详解

### 整体框架

定义黑名单 $\mathcal{C}$ → LLM生成unsafe/safe prompt对 → CLIP文本编码器提取特征 → Embedding Mapping Layer（cross-attention + MLP）映射到潜在空间 → 对比学习训练 → 推理时计算概念-prompt余弦距离判断安全性。

### 关键设计

1. **LLM驱动的训练数据生成**

    - 从黑名单概念 $c$（如"murder"）出发，用LLM生成unsafe prompt $u_c$（如"a man gets murdered"）
    - LLM将unsafe prompt中的概念替换为无害词生成safe prompt $s_c$（如"a man gets kissed"）
    - 三元组：{概念 $c$，unsafe prompt $u_c$，safe prompt $s_c$}
    - 设计动机：safe prompt与unsafe prompt结构相似但概念不同，迫使模型精准定位概念相关的token

2. **Embedding Mapping Layer**

    - 概念嵌入 $z_c$ 作为query、prompt嵌入 $z_p$ 作为key/value进行多头cross-attention
    - $h_p = \text{MLP}_p({}^1h_p \| \cdots \| {}^I h_p)$，概念嵌入 $h_c = \text{MLP}_c(z_c)$
    - 注意力矩阵 $A \in \mathbb{R}^{C \times P}$ 自动学习哪些prompt token与概念相关
    - 设计动机：cross-attention自动加权prompt中与概念相关的token（如"murdered"），过滤无关token（如"a man"）

3. **对比学习训练策略**

    - 锚点 $a$：概念嵌入 $h_c^b$
    - 正样本 $p$：对应unsafe prompt嵌入 $h_{u_c}^b$
    - 负样本 $n$：其他batch内unsafe prompt $h_{u_c}^{\bar{b}}$ + 对应safe prompt $h_{s_c}^b$ + 其他safe prompt $h_{s_c}^{\bar{b}}$
    - $\mathcal{L}_{\text{cont}} = \sum_b \mathcal{L}_{\text{supcon}}(h_c^b, h_{u_c}^b, h_{u_c}^{\bar{b}} \| h_{s_c}^b \| h_{s_c}^{\bar{b}})$
    - 设计动机：safe prompt作负样本帮助模型区分概念present/absent，同batch其他概念防止混淆

### 损失函数 / 训练策略

仅训练Embedding Mapping Layer，CLIP编码器冻结。AdamW，lr=1e-3，weight decay=1e-2，batch=64，1000迭代收敛。单卡3090仅需约**30分钟**训练。推理时概念嵌入可预计算缓存。

## 实验关键数据

### 主实验

在CoPro数据集上（723概念，226K prompt）的分类准确率：

| 方法 | ID Explicit↑ | ID Synonym↑ | ID Adversarial↑ | OOD Explicit↑ | OOD Synonym↑ | OOD Adversarial↑ |
|------|-------------|-------------|-----------------|---------------|-------------|-----------------|
| Text Blacklist | 0.805 | 0.549 | 0.587 | 0.895 | 0.482 | 0.494 |
| CLIPScore | 0.628 | 0.557 | 0.504 | 0.672 | 0.572 | 0.533 |
| BERTScore | 0.632 | 0.549 | 0.509 | 0.739 | 0.594 | 0.512 |
| LLM | 0.747 | 0.764 | **0.867** | 0.746 | 0.757 | **0.862** |
| **Latent Guard** | **0.868** | **0.828** | 0.829 | **0.867** | **0.824** | 0.819 |

AUC对比（threshold-based方法）：

| 方法 | ID Explicit↑ | ID Synonym↑ | ID Adversarial↑ | OOD Explicit↑ | OOD Synonym↑ | OOD Adversarial↑ |
|------|-------------|-------------|-----------------|---------------|-------------|-----------------|
| CLIPScore | 0.697 | 0.587 | 0.504 | 0.733 | 0.596 | 0.560 |
| BERTScore | 0.783 | 0.591 | 0.481 | 0.832 | 0.622 | 0.556 |
| **Latent Guard** | **0.985** | **0.914** | **0.908** | **0.944** | **0.913** | **0.915** |

### 消融实验

| 设计选择 | 效果影响 |
|---------|---------|
| 无safe prompt负样本 | 准确率显著下降，难以区分概念present/absent |
| 无cross-attention | 退化为简单投影，准确率下降 |
| 嵌入维度 $d$ 变化 | 适中维度最优 |

### 关键发现

- ID Explicit AUC达到**0.985**，远超CLIPScore(0.697)和BERTScore(0.783)
- 在同义词场景下仍保持0.828准确率（vs Text Blacklist 0.549），证明潜在空间检测的鲁棒性
- 对抗攻击场景AUC 0.908，显著优于CLIP/BERT（~0.5），证明对encoder-level攻击的防御能力
- OOD概念（训练未见）的准确率与ID接近（0.867 vs 0.868），泛化能力强
- 训练仅需30分钟（单卡3090），推理无需生成图像

## 亮点与洞察

- **概念检测而非安全分类**的问题定义允许黑名单在测试时灵活更新，无需重训练
- 对比学习+safe prompt作负样本的设计精准解耦了概念与句子上下文
- 30分钟训练+预计算概念嵌入的推理效率极高，适合工业部署
- 对encoder-level对抗攻击（SneakyPrompt等）也展现出鲁棒性

## 局限性 / 可改进方向

- 对抗场景下准确率（0.829）低于LLM方法（0.867），因为LLM有更强的语义理解
- 依赖CLIP文本编码器，换用其他T2I模型的编码器需要重新训练
- 概念黑名单需人工定义或LLM辅助生成，覆盖范围有限
- 仅处理文本端安全，无法检测通过图像后处理产生的不安全内容

## 相关工作与启发

- **vs Text Blacklist**：黑名单仅做字符串匹配，Latent Guard在潜在空间检测，对改写鲁棒
- **vs LLM审查**：LLM计算代价高且不可灵活更新，Latent Guard轻量且黑名单可动态调整
- **vs Safe Latent Diffusion**：SLD操纵扩散过程仍需生成，Latent Guard在文本端直接拦截
- 启发：对比学习在T2I安全审核中的应用值得进一步探索

## 评分

- 新颖性: ⭐⭐⭐⭐ 概念检测+潜在空间+对比学习的组合新颖实用
- 实验充分度: ⭐⭐⭐⭐ 自建CoPro数据集、6种测试场景、4个基线、AUC+准确率双指标
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法动机充分
- 价值: ⭐⭐⭐⭐ 对T2I安全部署有直接实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[CVPR 2026\] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](../../CVPR2026/image_generation/when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections](garmentaligner_text-to-garment_generation_via_retrieval-augmented_multi-level_co.md)

</div>

<!-- RELATED:END -->
