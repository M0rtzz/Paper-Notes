---
title: >-
  [论文解读] Rethinking VLMs for Image Forgery Detection and Localization
description: >-
  [CVPR 2026][AI安全][图像伪造检测] 提出 IFDL-VLM 框架，发现 VLM 固有的语义合理性偏向（而非真实性）会阻碍伪造检测性能，因此将检测/定位与语言解释解耦为两阶段优化，并利用定位掩码作为 VLM 的辅助输入增强可解释性，在 9 个基准上全面达到 SOTA。
tags:
  - CVPR 2026
  - AI安全
  - 图像伪造检测
  - 视觉语言模型
  - 伪造定位
  - 可解释性
  - AIGC安全
---

# Rethinking VLMs for Image Forgery Detection and Localization

**会议**: CVPR 2026  
**arXiv**: [2603.12930](https://arxiv.org/abs/2603.12930)  
**代码**: [sha0fengGuo/IFDL-VLM](https://github.com/sha0fengGuo/IFDL-VLM)  
**领域**: AI安全  
**关键词**: 图像伪造检测, 视觉语言模型, 伪造定位, 可解释性, AIGC安全

## 一句话总结

提出 IFDL-VLM 框架，发现 VLM 固有的语义合理性偏向（而非真实性）会阻碍伪造检测性能，因此将检测/定位与语言解释解耦为两阶段优化，并利用定位掩码作为 VLM 的辅助输入增强可解释性，在 9 个基准上全面达到 SOTA。

## 研究背景与动机

随着 AIGC 技术（扩散模型、GAN、自回归 Transformer）的发展，图像篡改变得极为便利，图像伪造检测与定位（IFDL）面临严峻挑战。现有方法尝试将 VLM（如 CLIP + LLM + SAM）引入 IFDL 以增强可解释性，但作者发现：

**语义合理性 vs 真实性**：CLIP 等 VLM 在预训练时优化的是高层语义与语言的对齐，导致篡改图像即使物体被替换/添加，其视觉 token 表征仍与原图高度相似（余弦相似度高达 96-98%），VLM 无法区分真伪。

**现有流水线的耦合问题**：如 SIDA、FakeShield 等方法将检测、定位和语言解释在同一 VLM 中联合优化，但 VLM 缺乏伪造相关概念的先验，反而拖累检测/定位性能。

核心洞察：定位掩码本身就显式编码了伪造概念，可以反过来作为 VLM 的额外先验，简化其训练优化。

## 方法详解

### 整体框架

IFDL-VLM 将 IFDL 任务解耦为两个阶段：

- **Stage-1**：训练 ViT + SAM 专家模型进行伪造检测与定位
- **Stage-2**：利用 Stage-1 的检测/定位结果作为辅助输入，微调 VLM 生成语言解释

### 关键设计

1. **解耦优化（Decoupled Optimization）**：与现有方法不同，IFDL-VLM 不在单一 VLM 中联合优化所有子任务。Stage-1 使用可训练 ViT（CLIP-ViT-L/14 初始化）提取 $\langle\text{SEG}\rangle$ token 送入冻结的 SAM-H 生成定位掩码，同时利用 $\langle\text{CLS}\rangle$ token 做三分类（真实/全合成/篡改）。设计动机：VLM 的语义合理性偏向会干扰低层伪造痕迹检测，解耦后检测/定位模块不受 VLM 偏向影响。

2. **区域感知视觉特征增强（Region-Aware Visual Feature Enhancement）**：Stage-2 的核心创新。将 Stage-1 得到的定位掩码 $M$ 与原图 $x$ 做 element-wise 相乘提取伪造区域，分别送入 CLIP 编码后加权融合：

$$T_{vis} = \alpha \cdot \text{CLIP}(x) + (1 - \alpha) \cdot \text{CLIP}(x \odot M)$$

   其中 $\alpha = 0.5$。这样做的好处是：(a) 用低层伪造区域线索丰富视觉特征，提高真伪图像表征的可分性；(b) 定位掩码显式编码了伪造概念，LLM 无需从数据中隐式学习，简化训练优化。推理时用 Stage-1 预测的 $\hat{M}$ 替代真值 $M$。

3. **多头注意力特征融合**：ViT 输出的 patch-level 特征通过投影和多头注意力融合生成 $\langle\text{SEG}\rangle$ token，作为 SAM 的 prompt embedding。全局 $\langle\text{CLS}\rangle$ token 用于线性分类。这种设计让 ViT 同时承担像素级定位和图像级检测任务。

### 损失函数 / 训练策略

**Stage-1 损失**：

$$\mathcal{L}_{st\text{-}1} = \mathcal{L}_{det} + \mathcal{L}_{loc} = \lambda_{det}\mathcal{L}_{ce}(\hat{D}, D) + \lambda_{bce}\mathcal{L}_{bce}(\hat{M}, M) + \lambda_{dice}\mathcal{L}_{dice}(\hat{M}, M)$$

其中 $\lambda_{bce} = \lambda_{dice} = \lambda_{det} = 1.0$。

**Stage-2 损失**：

$$\mathcal{L}_{st\text{-}2} = \mathcal{L}_{ce}(\hat{y}_{des}, y_{des})$$

即 LLM 输出语言解释的自回归交叉熵损失。LLM 骨干采用 Vicuna-13B。

训练细节：AdamW 优化器，学习率 1e-5，线性 warmup-decay，batch size 4 + 梯度累积 10，FP16/BF16 混合精度。

## 实验关键数据

### 主实验

**SID-Set 检测性能**：

| 方法 | 整体 Acc | 整体 F1 | 说明 |
|------|---------|---------|------|
| SIDA-13B | 0.94 | 0.94 | 之前 SOTA |
| UnivFD | 0.65 | 0.80 | 传统方法 |
| **IFDL-VLM** | **0.997** | **0.998** | 近乎完美 |

**SID-Set 定位性能**：

| 方法 | AUC | F1 | IoU | 提升 |
|------|-----|----|----|------|
| SIDA-7B | 0.87 | 0.74 | 0.44 | - |
| **IFDL-VLM** | **0.99** | **0.87** | **0.65** | +21% IoU |

**跨数据集泛化（8 数据集平均）**：

| 方法 | 平均 IoU | 平均 F1 | 提升 |
|------|---------|---------|------|
| FakeShield | 0.39 | 0.45 | - |
| SIDA-13B* | 0.38 | 0.45 | - |
| **IFDL-VLM** | **0.47** | **0.58** | +13% IoU, +19% F1 |

### 消融实验

**可解释性评估（GPT-5 自动评分，0-5 分）**：

| 维度 | SIDA-13B | IFDL-VLM | 说明 |
|------|----------|----------|------|
| Mask | 1.22 | **2.28** | 定位掩码质量 |
| Tampered Content | 1.14 | **1.98** | 篡改内容描述 |
| 综合 Overall | 1.44 | **2.36** | 提升 63.9% |

**CSS 语义相似度评估**：

| 维度 | SIDA-13B | IFDL-VLM | 说明 |
|------|----------|----------|------|
| Areas | 0.61 | **0.67** | 篡改区域 |
| Tampered Content | 0.44 | **0.49** | 篡改内容 |
| CSS(weighted) | 0.57 | **0.62** | 加权提升 8.8% |

### 关键发现

- **VLM 先验对检测/定位无益**：CLIP 视觉特征在真伪图像间余弦相似度 96-98%，几乎无法区分。解耦后检测/定位性能大幅提升
- **定位掩码辅助 VLM**：将掩码作为显式伪造概念输入 VLM，可解释性显著提升（GPT-5 评分 +63.9%，CSS +8.8%）
- **人类评估**：50 名评估者中 65.2% 偏好 IFDL-VLM 的解释，仅 11.3% 偏好 SIDA-13B
- **跨数据集泛化**：在 8 个跨域数据集中 7 个上取得最佳性能，验证了框架的泛化能力

## 亮点与洞察

- 深刻揭示了 VLM "语义合理性偏向"对伪造检测的负面影响——这是一个反直觉但极有价值的发现
- "解耦+反哺"的设计哲学优雅：先训练专家模型做好检测/定位，再用结果辅助 VLM 做解释，而非让 VLM 同时承担所有任务
- 方法简洁但效果显著——Stage-1 仅加入 ViT + SAM 冻结解码器，Stage-2 仅修改视觉输入，没有复杂架构设计

## 局限与展望

- Stage-2 依赖 Stage-1 定位掩码质量，若定位失败则解释也会受影响（级联误差）
- 仅在 Vicuna-13B 上验证，未探索更强 LLM（如更大规模模型）是否能进一步提升可解释性
- 跨数据集泛化实验中在 IMD2020 上未超越 FakeShield，特定数据集仍有提升空间
- 未讨论计算效率——两阶段流水线在推理时的延迟开销

## 相关工作与启发

- **SIDA / FakeShield**：当前 IFDL + VLM 的代表方法，将 CLIP + LLM + SAM 耦合训练，本文证明解耦更优
- **MVSS-Net / CAT-Net**：传统 IFDL 方法，依赖手工先验（BayarConv、DCT）检测低层异常
- **SAM**：本文冻结 SAM-H 图像编码器，仅微调掩码解码器，有效利用其分割能力
- 启发：对于多模态辅助任务，先让专家模型做好基础判断，再将结果反哺给大模型做高层理解，可能是更好的范式

## 评分

- 新颖性: ⭐⭐⭐⭐ (VLM 偏向分析+解耦反哺设计，洞察深刻)
- 实验充分度: ⭐⭐⭐⭐⭐ (9 基准 + 检测/定位/可解释性三维评估 + 人类评估)
- 写作质量: ⭐⭐⭐⭐⭐ (动机清晰，从观察到方案的推导逻辑严谨)
- 价值: ⭐⭐⭐⭐⭐ (对 IFDL 领域有范式级贡献)

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)
- [\[CVPR 2026\] Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction](towards_highly_transferable_vision-language_attack_via_semantic-augmented_dynami.md)
- [\[CVPR 2026\] PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing](pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im.md)
- [\[ECCV 2024\] Noise-Assisted Prompt Learning for Image Forgery Detection and Localization](../../ECCV2024/ai_safety/noise-assisted_prompt_learning_for_image_forgery_detection_and_localization.md)
- [\[CVPR 2026\] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)

<!-- RELATED:END -->
