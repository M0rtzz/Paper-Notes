---
title: >-
  [论文解读] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation
description: >-
  [模型压缩] 提出通用统一 tokenizer UniFlow，通过层级自适应自蒸馏保留语义理解能力 + 轻量 patch-wise 像素流解码器实现高保真重建，在 13 个基准上实现理解与生成的双赢，7B UniFlow-XL 用 40% 更少数据超越 14B TokenFlow-XL 6.05%。
tags:
  - "模型压缩"
---

# UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2510.10575](https://arxiv.org/abs/2510.10575)
- **代码**: 未公开
- **领域**: 视觉理解与生成 / 统一 Tokenizer
- **关键词**: 统一 tokenizer, 视觉编码器, flow matching, 自蒸馏, 图像重建与生成

## 一句话总结

提出通用统一 tokenizer UniFlow，通过层级自适应自蒸馏保留语义理解能力 + 轻量 patch-wise 像素流解码器实现高保真重建，在 13 个基准上实现理解与生成的双赢，7B UniFlow-XL 用 40% 更少数据超越 14B TokenFlow-XL 6.05%。

## 研究背景与动机

视觉理解和生成是计算机视觉的两大核心任务，当前面临**统一 tokenizer 的困境**：

**双编码器方案**（如 TokenFlow）：语义编码器 + 像素编码器，模型冗余且训练低效

**冻结 VFM + 潜在扩散解码器**（如 EMU2、BLIP3-o）：继承了理解能力，但语义编码器无法建模细粒度细节，且受限于预训练 VAE 天花板

**统一编码器微调方案**（如 VILA-U、UniTok）：初始化后在像素解码器上微调，但高层语义与低层重建目标冲突，导致理解能力退化

**核心矛盾**：高级语义抽象与低级像素重建之间的固有冲突。如何在单一 tokenizer 中同时实现强语义理解和高保真重建？

## 方法详解

### 整体框架

UniFlow 把一个预训练视觉编码器改造成既能理解又能重建的统一 tokenizer：统一编码器 $\mathcal{E}_U$ 在自蒸馏约束下编码出语义 token，再交给一个轻量的 patch-wise 像素流解码器 $\mathcal{D}_{\text{flow}}$ 直接在像素空间还原图像。它不引入第二个编码器、也不依赖预训练 VAE，整套适配只需在 ImageNet 上训练 30 个 epoch，且可嫁接到任意 VFM 或 MLLM 视觉骨干。

### 关键设计

**1. 层级自适应自蒸馏：让重建目标不再侵蚀语义理解。** 统一编码器既要保留教师 VFM 的语义，又要为下游重建腾出建模细节的空间，二者天然冲突——若全层都强行对齐教师，细节无从学起；若放松约束，语义又会退化。UniFlow 的观察是深层负责语义消歧、浅层负责细粒度细节，因此蒸馏强度应该分层而非一刀切：深层施加更强的保留约束，浅层给更大灵活性。它用冻结教师 $\mathcal{E}_T$ 监督学生 $\mathcal{E}_U$，并按层自适应分配权重 $w_l = \frac{w_l^{\text{base}} \cdot \exp(\beta \cdot \alpha_l)}{\sum_{k=1}^{L} w_k^{\text{base}} \cdot \exp(\beta \cdot \alpha_k)}$，其中层级先验 $w_l^{\text{base}} = l/L$ 让深层基础权重更高，对齐惩罚 $\alpha_l$（第 $l$ 层学生与教师 token 的平均余弦距离）让对齐越差的层临时获得更多关注，$\beta$ 控制这种自适应的强度。最终蒸馏损失是加权的逐层余弦距离 $\mathcal{L}_{\text{dist}} = \sum_{l=1}^{L} w_l \cdot \left(1 - \frac{1}{S} \sum_{i,j} \frac{\langle \mathbf{H}_U^{(l,i,j)}, \mathbf{H}_T^{(l,i,j)} \rangle}{\|\mathbf{H}_U^{(l,i,j)}\| \|\mathbf{H}_T^{(l,i,j)}\|}\right)$，这样深层守住语义、浅层放手补细节，避免了 VILA-U/UniTok 那种微调即退化的问题。

**2. Patch-wise 像素流解码器：绕过 VAE 天花板直接在像素空间重建。** 现有统一方案大多在预训练 VAE 的潜空间上接扩散/流解码器，重建上限被 VAE 锁死。UniFlow 改为直接在像素空间学习速度场：基于 Rectified Flow 定义干净图像与噪声之间的线性插值路径 $\mathbf{x}_t^{(i,j)} = (1-t)\mathbf{x}^{(i,j)} + t \cdot \epsilon^{(i,j)},\ t \in [0,1]$，再用一个轻量 MLP 逐 patch 预测速度场 $v_\theta(\mathbf{x}_t^{(i,j)}, t, \mathbf{c}^{(i,j)})$。逐 patch 解码把整张图的复杂分布拆成局部子分布，简化了学习目标、提升了训练效率，但代价是各 patch 各自为政会留下网格状接缝伪影。为此在条件注入前加一个全局 Transformer 块，让上采样后的 token 先经自注意力互通信息再分发条件 $\mathbf{C} = \mathcal{GTB}(\mathcal{P}_{\text{up}}(\mathbf{z}) + \mathbf{PE})$，使每个 patch 的解码都带上全局上下文，从而消除网格伪影。

### 损失函数 / 训练策略

总目标只是蒸馏与流匹配两项的加权和 $\mathcal{L}_{\text{total}} = \lambda_d \mathcal{L}_{\text{dist}} + \lambda_f \mathcal{L}_{\text{flow}}$，其中流匹配损失为速度场回归 $\mathcal{L}_{\text{flow}} = \mathbb{E}\left[\|v_\theta(\mathbf{x}_t^{(i,j)}, t, \mathbf{c}^{(i,j)}) - (\epsilon^{(i,j)} - \mathbf{x}^{(i,j)})\|_2^2\right]$。相比 VQ-GAN 系一贯需要 GAN+L1+L2+LPIPS 的多损失拼盘，UniFlow 仅靠单一流匹配损失就完成重建监督，训练更稳、调参更省，也正是它能在 30 epoch 内完成通用适配的原因之一。

## 实验

### 图像重建质量（256×256 ImageNet-1K）

| 方法 | 类型 | 下采样比 | rFID ↓ |
|------|------|---------|--------|
| SD-VAE 3 | 生成专用 | 8 | 0.20 |
| FLUX-VAE | 生成专用 | 8 | 0.18 |
| UniTok | 统一 | 16 | 0.41 |
| TokenFlow | 统一 | 16 | 1.37 |
| **UniFlow(InternViT)** | **统一** | 14 | **0.26** |
| **UniFlow(DINOv2)** | **统一** | 14 | 0.54 |

UniFlow(InternViT) 在统一 tokenizer 中 SOTA，rFID 仅 0.26（UniTok 0.41，↓0.15），甚至接近生成专用 tokenizer。

### 多模态理解（LLaVA-v1.5 设置）

| 方法 | LLM | POPE | GQA | TQA | MMB | MME-P | Avg |
|------|-----|------|-----|-----|-----|-------|-----|
| LLaVA-1.5 | Vicuna-7B | 85.9 | 62.0 | 46.1 | 64.3 | 1510.7 | - |
| Janus | DeepSeek-1.3B | 87.0 | 59.1 | - | 69.4 | 1338.0 | - |
| **UniFlow-LV** | Vicuna-7B | **高** | **高** | **高** | **高** | **高** | **SOTA** |

7B UniFlow-XL 在整体平均理解基准上超越 14B TokenFlow-XL 6.05%，且训练数据少 40%。

### 图像生成

gFID（无 guidance）比 UniTok 好 0.09，验证了生成质量的竞争力。

### 关键发现

1. **理解与生成双赢**：UniFlow 同时提升两个方向的性能，打破了传统的 trade-off
2. **编码器通用性**：CLIP、SigLIP2、DINOv2、InternViT 四种编码器均有效，InternViT 最优
3. **训练效率高**：仅用 ImageNet 训练 30 epoch，40% 更少数据超越 TokenFlow
4. **无 VAE 限制**：直接在像素空间建模，重建上限更高
5. **Patch-wise 策略有效**：简化数据分布，提升训练效率，配合全局 Transformer 消除网格伪影

## 亮点

- 层级自适应自蒸馏设计优雅，动态平衡语义保留与细节适应
- Patch-wise 像素流解码器概念新颖，直接绕过 VAE 天花板
- 全局 Transformer 块有效消除 patch-wise 解码的网格伪影
- 通用适配范式，可嫁接到任何预训练编码器
- 实验覆盖 13 个基准、7 个任务，充分验证了多任务能力

## 局限性

- 流解码器的推理步数可能影响重建速度，trade-off 未详细讨论
- 下采样比为 14（CLIP/DINOv2/InternViT），大于常见的 8 或 16，对比公平性需注意
- 生成质量虽优于 UniTok，但与最佳生成专用 tokenizer（FLUX-VAE rFID 0.18）仍有差距
- 未在视频生成任务上验证
- patch-wise 策略在更高分辨率下的扩展性和全局一致性有待探索

## 相关工作

- **生成专用 Tokenizer**：VQ-GAN、SD-VAE 系列、FlowMo、SelfTok — 重建质量好但语义弱
- **统一 Tokenizer**：VILA-U、UniTok、QLIP（微调冲突）、TokenFlow（双编码器冗余）、DualToken
- **扩散/流解码器**：l-DeTok、FlowMo、SelfTok — 受限于预训练 VAE 潜空间
- **自蒸馏**：DualToken、TokLIP — 仅最后层蒸馏或需大规模对比学习

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — Patch-wise 像素流解码 + 层级自适应自蒸馏的组合创新
- **技术深度**: ⭐⭐⭐⭐⭐ — 方法设计有深度，损失设计简洁有效
- **实验充分度**: ⭐⭐⭐⭐⭐ — 13 个基准、7 个任务、4 种编码器、多种对比
- **实用价值**: ⭐⭐⭐⭐⭐ — 通用适配范式，30 epoch 即可完成，实际可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[AAAI 2026\] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)

</div>

<!-- RELATED:END -->
