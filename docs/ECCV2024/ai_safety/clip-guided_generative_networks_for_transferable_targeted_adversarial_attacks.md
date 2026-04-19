---
title: >-
  [论文解读] CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks
description: >-
  [ECCV 2024][AI安全][adversarial attack] 提出 CGNC，利用 CLIP 文本编码器为条件生成网络注入目标类别语义信息，结合交叉注意力模块和 masked fine-tuning，大幅提升多目标/单目标定向对抗攻击的黑盒迁移成功率。
tags:
  - ECCV 2024
  - AI安全
  - adversarial attack
  - targeted transferability
  - CLIP
  - generative model
  - 注意力机制
  - multi-target attack
---

# CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks

**会议**: ECCV 2024  
**arXiv**: [2407.10179](https://arxiv.org/abs/2407.10179)  
**代码**: [ffhibnese/CGNC_Targeted_Adversarial_Attacks](https://github.com/ffhibnese/CGNC_Targeted_Adversarial_Attacks)  
**领域**: ai_safety  
**关键词**: adversarial attack, targeted transferability, CLIP, generative model, cross-attention, multi-target attack

## 一句话总结

提出 CGNC，利用 CLIP 文本编码器为条件生成网络注入目标类别语义信息，结合交叉注意力模块和 masked fine-tuning，大幅提升多目标/单目标定向对抗攻击的黑盒迁移成功率。

## 研究背景与动机

**定向迁移攻击难度高**：相比非定向攻击，定向对抗攻击要求黑盒模型输出指定类别，迁移成功率远低于非定向攻击，是对抗安全研究的核心难题。

**单目标生成式攻击计算开销大**：TTP、DGTA-PI 等方法为每个目标类别训练一个独立生成器，当目标类别达到数百/千级别时训练成本不可接受。

**现有多目标方法条件信息贫乏**：MAN 和 C-GSP 仅使用类别索引或 one-hot 向量作为条件输入，无法利用目标类别的丰富语义信息，导致黑盒迁移率有限。

**分类边界过拟合**：基于梯度的迭代方法（MIM、DIM 等）严重过拟合代理模型的分类边界，在黑盒模型上迁移效果极差（通常 < 5%）。

**视觉-语言模型的语义先验未被利用**：CLIP 等 VLM 在大规模图文对上预训练，包含丰富的类别语义知识，但此前未被用于指导对抗扰动生成。

**跨域场景下单目标方法不可行**：单目标攻击需要目标类别图像计算损失，在训练集不含目标类别的跨域场景下完全失效，亟需不依赖目标类数据的方案。

## 方法详解

### 整体框架

CGNC（CLIP-guided Generative Network with Cross-attention）是一个条件生成网络，输入为干净图像 $\bm{x}_s$ 和目标类别文本描述 $\bm{t}_c$（如 "a photo of a sea lion"），输出为 $\ell_\infty$ 约束下的对抗扰动 $\bm{\delta}$。网络由三个核心模块组成：VL-Purifier、F-Encoder、CA-Decoder。训练目标为最小化代理模型对对抗样本在目标类别上的交叉熵损失：

$$w^* \leftarrow \arg\min_w \mathcal{L}\big(f_\theta(\bm{x}_s + G_w(\bm{x}_s, \Phi(\bm{t}_c))), c\big)$$

### 关键设计 1：Vision-Language Feature Purifier

将目标类别文本输入 CLIP 文本编码器 $\Phi$ 得到 512 维嵌入 $\bm{e}_t$，再通过由全连接层 + 谱归一化层组成的 VL-Purifier 压缩为 16 维任务特定表示 $\bm{e}_t^*$。这一步将 CLIP 通用语义空间的嵌入适配到对抗扰动生成任务，避免直接使用高维通用表示带来的噪声。

### 关键设计 2：Feature Fusion Encoder

F-Encoder 将净化后的文本嵌入与图像视觉特征进行多级融合：先将 $\bm{e}_t^*$ 在空间维度扩展后与图像编码特征 $\bm{h}_s$ 在通道维度拼接；随后经下采样后再次与文本嵌入拼接，重复多次。这种多级拼接机制同时利用了实例级视觉信息和类别级文本信息，使生成扰动包含更强的语义模式。

### 关键设计 3：Cross-Attention Decoder

在解码器中引入交叉注意力层，以 CLIP 原始 512 维嵌入 $\bm{e}_t$ 作为 Key/Value、解码器中间特征 $\bm{z}_t$ 作为 Query：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

最终通过 $\bm{\delta} = \epsilon \cdot \tanh(\bm{o})$ 平滑投影确保扰动满足 $\ell_\infty \leq \epsilon$ 约束。交叉注意力使解码器能动态聚焦于 CLIP 语义空间中与目标类别最相关的特征维度。

### 关键设计 4：Masked Fine-Tuning (MFT)

针对单目标场景，固定条件输入为特定目标类别文本，对已训练的多目标生成器进行微调。为缓解微调过程中扰动集中于特定区域导致的过拟合问题，在扰动输出上施加 patch-wise 随机掩码（mask ratio=0.2），迫使生成器学习更分散的扰动模式。仅需额外 5 个 epoch 即可完成。

## 损失函数与训练策略

- **损失函数**：交叉熵损失，引导代理模型将对抗样本分类为目标类别
- **数据增强**：训练时对输入图像进行数据增强，增强扰动泛化性
- **代理模型**：Inc-v3 和 Res-152
- **扰动预算**：$\epsilon = 16/255$
- **训练**：学习率 2e-4，训练 10 个 epoch；MFT 阶段 5 个 epoch

## 实验

### 多目标攻击：常规模型（代理模型 Res-152 → 黑盒）

| 方法 | VGG-16 | GoogleNet | Inc-v3 | DN-121 | Inc-v4 | Inc-Res-v2 |
|------|--------|-----------|--------|--------|--------|------------|
| MIM | 0.20 | 0.30 | 0.50 | 0.30 | 0.40 | 0.60 |
| Logit | 9.20 | 3.70 | 10.10 | 12.70 | 10.70 | 12.80 |
| C-GSP | 45.90 | 41.70 | 37.70 | 64.20 | 33.33 | 20.28 |
| **CGNC** | **63.36** | **62.23** | **53.39** | **85.66** | **51.53** | **34.24** |

CGNC 在所有黑盒模型上均大幅超越 C-GSP，平均提升 17.88%；在 DN-121 上提升 21.46%，验证了 CLIP 语义先验的有效性。

### 鲁棒模型攻击（代理模型 Res-152 → 鲁棒训练模型）

| 方法 | Inc-v3_adv | IR-v2_ens | Res50_SIN | Res50_IN | Res50_fine | Res50_Aug |
|------|-----------|-----------|-----------|----------|------------|-----------|
| C-GSP | 14.60 | 16.01 | 16.84 | 60.30 | 65.51 | 42.88 |
| **CGNC** | **22.21** | **26.71** | **29.83** | **79.80** | **84.05** | **63.75** |

在鲁棒训练模型上 CGNC 同样大幅领先，Res50_Aug 上提升 20.87%，证明方法在防御场景下仍具优势。

### 单目标攻击对比（8 类平均，代理 Inc-v3）

| 方法 | Inc-v4 | Inc-Res-v2 | Res-152 | DN-121 | GoogleNet | VGG-16 |
|------|--------|------------|---------|--------|-----------|--------|
| TTP | 46.04 | 39.37 | 16.40 | 33.47 | 25.80 | 25.73 |
| DGTA-PI | 67.95 | 55.03 | 50.50 | 47.38 | 47.67 | 48.11 |
| **CGNC†** | **74.76** | **64.48** | **62.00** | **78.94** | **69.06** | **70.74** |

MFT 单目标变体 CGNC† 黑盒平均提升 15.36%，且仅需 1 个多目标生成器 + 8 次微调，计算量远低于训练 8 个独立生成器。

### 消融实验（代理 Res-152）

| 变体 | VGG-16 | GoogleNet | Inc-v3 | DN-201 |
|------|--------|-----------|--------|--------|
| CGNC-CA-t（one-hot 条件） | 56.55 | 51.09 | 47.44 | 74.65 |
| CGNC-CA（CLIP 文本，无交叉注意力） | 56.64 | 54.29 | 49.73 | 75.99 |
| **CGNC（完整）** | **63.36** | **62.23** | **53.39** | **82.69** |

CLIP 文本嵌入和交叉注意力模块各自贡献显著，完整 CGNC 在 DN-201 上比 one-hot 基线高 8.04%。

## 亮点

- **语义驱动的条件机制**：首次将 CLIP 文本编码作为多目标对抗生成器的条件输入，从根本上突破了 one-hot 条件的语义贫乏瓶颈
- **多目标大幅提升**：在 8 类和 200 类目标设定下均大幅超越 C-GSP，且随类别数增加性能下降更平缓
- **高效单目标适配**：MFT 仅需 5 epoch 微调即可超越从头训练的单目标 SOTA，攻击 8 类可节省超 100 epoch
- **跨域泛化**：在不含目标类别图像的 MS-COCO 和 Comics 数据集上仍有不错表现，是单目标方法无法实现的
- **可视化证据充分**：生成扰动呈现清晰的目标类别语义模式（如海狮形态），且随文本条件变化

## 局限性

- 仍依赖 CLIP 预训练模型，性能受限于 CLIP 对特定类别的表征质量
- 扰动预算 $\epsilon=16/255$ 相对较大，在更严格约束（如 $8/255$）下效果未充分验证
- 多目标生成器随类别数超过 200 后性能开始明显下降，距离真正的 1000 类实用仍有距离
- MFT 的 mask ratio 需要手动调参（论文固定 0.2），缺乏自适应机制
- 实验仅在 ImageNet 分类器上验证，未扩展到目标检测等更复杂视觉任务

## 相关工作

- **多目标生成攻击**：MAN (ICCV 2019) 首提多目标框架但降质严重；C-GSP (CVPR 2023) 引入层次划分改进但条件表达受限
- **单目标生成攻击**：TTP (ICCV 2021) 目标分布匹配；DGTA-PI (CVPR 2023) 动态网络 + pattern injection；本文 MFT 以更低成本超越
- **CLIP 在对抗领域的应用**：此前 CLIP 主要用于零样本分类和多模态任务，本文首次将其文本编码用于指导对抗扰动生成
- **交叉注意力融合**：借鉴 Stable Diffusion 等文本条件生成模型的 cross-attention 设计，适配到对抗扰动场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — CLIP 文本语义指导对抗生成是新颖且直觉合理的思路
- 实验充分度: ⭐⭐⭐⭐ — 覆盖常规/鲁棒/防御/跨域/多类别数多种场景，消融完整
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，图表丰富，扰动可视化直观有说服力
- 价值: ⭐⭐⭐⭐ — 为对抗迁移攻击引入 VLM 先验开辟了新路线，对模型鲁棒性评估有实际意义

<!-- RELATED:START -->

## 相关论文

- [X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP](../../ICML2025/ai_safety/x-transfer_attacks_towards_super_transferable_adversarial_attacks_on_clip.md)
- [Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [Any Target Can Be Offense: Adversarial Example Generation via Generalized Latent Infection](any_target_can_be_offense_adversarial_example_generation_via_generalized_latent_.md)
- [Resilience of Entropy Model in Distributed Neural Networks](resilience_of_entropy_model_in_distributed_neural_networks.md)
- [Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)

<!-- RELATED:END -->
