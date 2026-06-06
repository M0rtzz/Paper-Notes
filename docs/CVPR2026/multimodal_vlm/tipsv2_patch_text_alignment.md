---
title: >-
  [论文解读] TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment
description: >-
  [CVPR 2026][多模态VLM][视觉-语言预训练] 提出 TIPSv2，通过发现蒸馏能显著提升 patch-text 对齐能力，并将该洞察转化为新的预训练目标 iBOT++（可见 token 也参与损失计算），结合头部EMA和多粒度文本增强，在 9 个任务 20 个数据集上达到 SOTA。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "视觉-语言预训练"
  - "patch-text对齐"
  - "iBOT++"
  - "蒸馏"
  - "零样本分割"
---

# TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment

**会议**: CVPR 2026  
**arXiv**: [2604.12012](https://arxiv.org/abs/2604.12012)  
**代码**: [https://gdm-tipsv2.github.io/](https://gdm-tipsv2.github.io/)  
**领域**: 多模态VLM  
**关键词**: 视觉-语言预训练, patch-text对齐, iBOT++, 蒸馏, 零样本分割

## 一句话总结

提出 TIPSv2，通过发现蒸馏能显著提升 patch-text 对齐能力，并将该洞察转化为新的预训练目标 iBOT++（可见 token 也参与损失计算），结合头部EMA和多粒度文本增强，在 9 个任务 20 个数据集上达到 SOTA。

## 研究背景与动机

**领域现状**：视觉-语言预训练有两大方向：对比/sigmoid 方法（CLIP、SigLIP、PE）提供图文对齐和零样本能力；自监督方法（DINO、iBOT）擅长密集任务的空间理解。

**现有痛点**：实现同时在全局（图像级）和密集（patch级）理解上都出色的统一表示仍是重大挑战。TIPS、SigLIP2 等统一方法仍难以维持精确的 patch 级文本对齐。一个令人惊讶的趋势是：最大的旗舰模型在 patch-text 对齐上反而不如小模型。

**核心矛盾**：最终 Transformer 层往往作为全局对比"解码器"工作，而非保留局部语义，导致 patch 级对齐退化。

**本文目标**：在预训练阶段直接解决 patch-text 对齐问题。

**切入角度**：发现蒸馏过程通过对所有 patch token 施加有效监督，显著提升了空间对齐——蒸馏学生模型的 patch-text 对齐大幅超过教师模型。

**核心 idea**：将蒸馏的洞察转化为预训练目标 iBOT++，使可见 token 也直接参与 MIM 损失。

## 方法详解

### 整体框架

TIPSv2 在 TIPS 基础上整合了三个改进：(1) iBOT++ 目标，对可见和掩码 token 都施加自监督损失；(2) Head-only EMA 策略，仅更新投影层而非整个模型的 EMA；(3) 多粒度文本增强，结合 PaliGemma 和 Gemini 的合成标题。总损失为 $\mathcal{L} = \mathcal{L}_{CLIP} + \mathcal{L}_{DINO} + \mathcal{L}_{iBOT++}$。

### 关键设计

1. **iBOT++ (增强的掩码图像建模)**:

    - 功能：在预训练中直接增强 patch-text 对齐
    - 核心思路：标准 iBOT 仅对掩码 token 计算损失（$\mathcal{L}_{iBOT} = -\sum m_i \cdot h_t(f_t(I)_i)^T \log h_s(f_s(I_{mask})_i)$）。iBOT++ 移除掩码条件，让可见 token 也参与损失计算。这相当于对所有 patch token 施加表示一致性约束
    - 设计动机：蒸馏实验表明，去除掩码并对所有 token 施加监督是提升 patch-text 对齐的关键因素。TIPS ViT-L（蒸馏学生）在零样本分割上大幅超过 TIPS ViT-g（教师），mIoU 差距超过 20 个点

2. **Head-only EMA**:

    - 功能：大幅减少训练内存和参数量
    - 核心思路：标准 SSL 需要完整模型的 EMA 教师来防止模型坍塌。但 TIPSv2 有额外的图文对比损失提供稳定信号，因此只需对投影头进行 EMA 更新，共享单一视觉编码器
    - 设计动机：将可训练参数减少近一半，使得大规模训练更高效

3. **多粒度标题采样**:

    - 功能：增强文本监督的多样性和鲁棒性
    - 核心思路：结合原始网页 alt-text 标题、PaliGemma 的空间关系标题和 Gemini 的详细描述标题，在训练中随机采样不同粒度的标题
    - 设计动机：不同粒度的标题捕获不同的视觉信息，提升模型在多种下游任务上的适应性

### 损失函数 / 训练策略

三部分损失：CLIP 对比损失（双 CLS token 分别用于对象和空间标题）、DINO 全局自蒸馏损失、iBOT++ patch 级损失（对所有 token 而非仅掩码 token）。Head-only EMA 更新投影层。

## 实验关键数据

### 主实验

| 模型 | PC59 mIoU | PC60 mIoU | VOC21 mIoU | ADE150 mIoU |
|------|-----------|-----------|------------|-------------|
| TIPS ViT-g (教师) | 11.4 | 10.8 | 19.7 | 2.6 |
| TIPS ViT-L (蒸馏学生) | 33.5 | 30.4 | 30.5 | 20.8 |
| TIPSv2 ViT-L | **42.1** | **38.2** | **45.3** | **28.7** |
| DINOv2 ViT-L | 35.8 | 32.1 | 38.6 | 23.4 |
| PE-core ViT-L | 28.3 | 25.6 | 32.1 | 18.2 |

### 消融实验

| 配置 | 掩码率 | PC59 | VOC21 | ADE150 |
|------|--------|------|-------|--------|
| 标准预训练 | 0.75 | 6.9 | 6.7 | 0.3 |
| 蒸馏 | 0.75 | 16.0 | 22.5 | 5.9 |
| 蒸馏 | 0.5 | 15.5 | 24.0 | 7.0 |
| 蒸馏 (无掩码=iBOT++) | 0.0 | **31.4** | **30.8** | **20.0** |

### 关键发现

- 去除掩码是关键：蒸馏时掩码率从 0.75 降到 0.0，PC59 mIoU 从 16.0 跃升到 31.4
- Head-only EMA 在有文本监督时与全模型 EMA 性能相当，但内存减少约一半
- 蒸馏学生大幅超过教师的现象说明 patch-text 对齐是可以后天获得的

## 亮点与洞察

- "蒸馏学生超越教师"的发现极具启发性：说明在大模型中 patch 语义被全局对比损失"稀释"，而蒸馏通过对所有 token 施加密集监督恢复了局部语义
- iBOT++ 的改动极其简单（一行代码级别：去掉掩码条件），效果却是颠覆性的，这种 minimal 修改产生大效果的工作非常有价值
- Head-only EMA 的观察指出文本监督可以替代 EMA 防止坍塌的角色

## 局限与展望

- 主要在 Google 内部规模的数据上验证，社区复现难度较高
- iBOT++ 的理论解释仍不够深入
- 未评估在视频理解和 3D 任务上的表现
- 可探索 iBOT++ 在更大模型上的效果

## 相关工作与启发

- **vs TIPS**: TIPSv2 的 iBOT++ 直接在预训练中增强 patch-text 对齐，不需要蒸馏步骤
- **vs DINOv2**: DINOv2 缺乏文本对齐，TIPSv2 同时实现空间理解和文本对齐
- **vs PE**: PE 优化全局对比但牺牲密集任务，TIPSv2 通过 iBOT++ 在两者间取得平衡

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 蒸馏学生超越教师的发现和 iBOT++ 的提出都很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 9 任务 20 数据集的全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 从发现到方法的逻辑链非常清晰
- 价值: ⭐⭐⭐⭐⭐ 对视觉基础模型预训练有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models](mostly_text_smart_visuals_asymmetric_text-visual_pruning_for_large_vision-langua.md)
- [\[CVPR 2026\] VISion On Request: Enhanced VLLM Efficiency with Sparse, Dynamically Selected, Vision-Language Interactions](vision_on_request_enhanced_vllm_efficiency_with_sparse_dynamically_selected_visi.md)
- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)
- [\[ICML 2026\] TRAP: 用对抗 patch 劫持 VLA 的 CoT 推理实现目标行为攻击](../../ICML2026/multimodal_vlm/trap_hijacking_vla_cot-reasoning_via_adversarial_patches.md)
- [\[CVPR 2026\] PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)

</div>

<!-- RELATED:END -->
