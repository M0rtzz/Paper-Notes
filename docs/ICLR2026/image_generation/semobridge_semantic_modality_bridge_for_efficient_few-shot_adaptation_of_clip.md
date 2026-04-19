---
title: >-
  [论文解读] SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP
description: >-
  [ICLR 2026][图像生成][CLIP 适配] 提出 SeMoBridge，一种轻量级语义模态桥，通过将图像嵌入映射到文本模态，将不可靠的模态内（图像-图像）比较转换为可靠的模态间（文本-图像）比较，以极低训练开销在少样本分类中超越现有方法。
tags:
  - ICLR 2026
  - 图像生成
  - CLIP 适配
  - 模态间隙
  - 模态内不对齐
  - 少样本分类
  - 伪 EOS 令牌
---

# SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP

**会议**: ICLR 2026  
**arXiv**: [2509.26036](https://arxiv.org/abs/2509.26036)  
**代码**: [https://github.com/christti98/semobridge](https://github.com/christti98/semobridge)  
**领域**: 少样本学习 / 视觉语言模型  
**关键词**: CLIP 适配, 模态间隙, 模态内不对齐, 少样本分类, 伪 EOS 令牌

## 一句话总结

提出 SeMoBridge，一种轻量级语义模态桥，通过将图像嵌入映射到文本模态，将不可靠的模态内（图像-图像）比较转换为可靠的模态间（文本-图像）比较，以极低训练开销在少样本分类中超越现有方法。

## 研究背景与动机

CLIP 通过对比学习将图像和文本对齐到共享嵌入空间，在零样本任务上表现优秀。但在少样本分类中存在**模态内不对齐**问题：

- CLIP 存在固有的**模态间隙 (modality gap)**——图像和文本嵌入之间的系统性分离
- CLIP 的对比训练目标仅关注跨模态对齐，未校准同模态内部的语义结构
- 导致查询图像可能被错误地放置在距离错误类别的少样本质心更近的位置

现有方法的局限：
- Tip-X、APE 等在逻辑分数层面操作，无法充分利用 CLIP 的模态间语义先验
- Cross the Gap 通过逐样本优化解决，但计算开销巨大

## 方法详解

### 核心思想

将图像嵌入映射到文本模态，保持语义内容不变，从而将不可靠的图像-图像比较转换为可靠的图像-文本模态间比较。

### 关键设计一：伪 EOS 令牌推导

利用 CLIP 训练目标保证的方向对齐：

$$\frac{\mathbf{f}_{\text{img}}}{\|\mathbf{f}_{\text{img}}\|} \approx \frac{\hat{\mathbf{f}}_{\text{txt}}}{\|\hat{\mathbf{f}}_{\text{txt}}\|}$$

通过文本投影矩阵的 Moore-Penrose 伪逆反投影并重新缩放：

$$\hat{\mathbf{f}}_{\text{eos}} \approx \frac{\|\mathbf{T}_{\text{eos}}\|}{\|\mathbf{W}_{\text{txt}}^+ \mathbf{f}_{\text{img}}\|} \mathbf{W}_{\text{txt}}^+ \mathbf{f}_{\text{img}}$$

最终桥接嵌入：

$$\hat{\mathbf{f}}_{\text{txt}} = \mathbf{W}_{\text{txt}} \hat{\mathbf{f}}_{\text{eos}} \approx \frac{\|\mathbf{T}_{\text{eos}}\|}{\|\mathbf{W}_{\text{txt}}^+ \mathbf{f}_{\text{img}}\|} \mathbf{f}_{\text{img}}$$

由于 $\mathbf{W}_{\text{txt}}\mathbf{W}_{\text{txt}}^+$ 近似为单位矩阵，变换简化为对原始图像嵌入的缩放。

### 关键设计二：三重逻辑分数融合

$$\mathbf{z}_q = \lambda_1 \mathbf{z}_1 + \lambda_2 \mathbf{z}_2 + \lambda_3 \mathbf{z}_3$$

- $\mathbf{z}_1$：零样本先验（查询图像 vs 类文本提示）
- $\mathbf{z}_2$：原始少样本 vs 桥接查询（桥接查询在文本空间与少样本图像比较）
- $\mathbf{z}_3$：原始查询 vs 桥接少样本（反置信号增强鲁棒性）

### 关键设计三：多模态监督训练 (SeMoBridge-T)

添加类特定偏置 (CSB) $\hat{\boldsymbol{\tau}} \in \mathbb{R}^{C \times d_t}$：

$$\hat{\mathbf{F}}_{\text{eos}}^c \approx \frac{\|\mathbf{T}_{\text{eos}}\|}{\|\hat{\mathbf{W}}_{\text{txt}}^+ \mathbf{F}_{\text{img}}^c + \hat{\boldsymbol{\tau}}^c\|} (\hat{\mathbf{W}}_{\text{txt}}^+ \mathbf{F}_{\text{img}}^c + \hat{\boldsymbol{\tau}}^c)$$

多模态损失：

$$\mathcal{L} = \lambda_{\text{it}} \mathcal{L}_{\text{img}} + (1-\lambda_{\text{it}})\frac{\mathcal{L}_{\text{txte}} + \mathcal{L}_{\text{txtp}}}{2} + \lambda_c \mathcal{L}_{\text{cons}} + \lambda_b \mathcal{L}_{\text{bias}}$$

- $\mathcal{L}_{\text{img}}$：桥接嵌入与原始图像嵌入对齐
- $\mathcal{L}_{\text{txte}}, \mathcal{L}_{\text{txtp}}$：与类描述 EOS 令牌和投影对齐
- $\mathcal{L}_{\text{cons}}$：同类少样本一致性
- $\mathcal{L}_{\text{bias}}$：CSB 正则化

训练时仅更新桥的参数，CLIP 完全冻结。

## 实验

### 训练效率比较

| 方法 | 参数量 | 平均训练时间 | 平均准确率 |
|------|--------|-------------|-----------|
| CoOp | 0.01M | 10h 00min | 63.90% |
| PromptSRC | 0.05M | 1h 42min | 77.90% |
| APE-T | 0.51M | 3min 30s | 77.18% |
| LDC | 69M | 2min | 77.17% |
| **SeMoBridge-T** | **0.77M** | **27s** | **78.15%** |

SeMoBridge-T 仅需 27 秒训练，准确率最高。

### 少样本分类结果

- **无训练 SeMoBridge** 在 11 个数据集中 7 个超越 APE
- **SeMoBridge-T** 在低样本场景（1/2/4-shot）中整体表现最佳
- 在 OxfordPets 等类别相似的数据集上改进最为显著

### 分布外泛化（16-shot ImageNet）

| 方法 | ImageNet | ImageNet-V2 | ImageNet-Sketch |
|------|----------|-------------|-----------------|
| APE | 71.81 | 64.81 | 49.95 |
| SeMoBridge | 71.86 | 64.90 | 49.55 |
| APE-T | 74.13 | 66.21 | 49.73 |
| SeMoBridge-T | 与 APE-T 竞争 | — | — |

### 关键发现

- 模态内不对齐是 CLIP 少样本失败的主要原因
- 简单的模态桥（缩放 + 投影）即可有效解决该问题
- CSB 对 ImageNet 等大类别数量的数据集有帮助，但对小数据集可有可无
- 多模态损失中图像和文本的平衡权重固定为 1:1 已足够

## 亮点

- 方法极其简洁优雅——核心计算是一个伪逆矩阵乘法
- 训练时间极短（27 秒），比次优方法快一个数量级
- 无训练版本已具竞争力，训练版进一步提升
- 理论动机清晰：从 CLIP 训练目标出发推导桥接
- 在低样本场景（1/2/4-shot）中优势最明显

## 局限性

- 假设 $\mathbf{W}_{\text{txt}}\mathbf{W}_{\text{txt}}^+$ 近似单位矩阵，这在某些 CLIP 变体中可能不成立
- CSB 在训练时使用但推理时不对查询使用，可能导致训练-推理分布不匹配
- 主要在 ViT-B/16 上验证，对更大模型的效果未充分探索
- 三重逻辑融合的权重需要验证集调优

## 相关工作

- **CLIP 适配**：CoOp、Tip-Adapter、APE 等提示/适配器方法
- **模态间隙研究**：Liang et al. 发现的 CLIP 嵌入空间模态间隙
- **模态反转**：OTI/OVI（逐样本优化）、SD-IPC（闭合形式投影）

## 评分

- 新颖性：⭐⭐⭐⭐ — 模态桥的思路及从 SD-IPC 的迁移巧妙
- 简洁性：⭐⭐⭐⭐⭐ — 方法优雅，易于理解和复现
- 实验：⭐⭐⭐⭐ — 11 个数据集 + 训练效率对比 + OOD 泛化
- 实用性：⭐⭐⭐⭐⭐ — 27 秒训练，极低门槛

<!-- RELATED:START -->

## 相关论文

- [Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](../../CVPR2026/image_generation/uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](../../CVPR2026/image_generation/v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)
- [Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [Branched Schrödinger Bridge Matching](branched_schrödinger_bridge_matching.md)

<!-- RELATED:END -->
