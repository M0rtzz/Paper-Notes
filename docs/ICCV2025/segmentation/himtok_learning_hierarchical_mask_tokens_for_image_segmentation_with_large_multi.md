---
title: >-
  [论文解读] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model
description: >-
  [ICCV 2025][语义分割][分层掩码Token化] 提出HiMTok（分层掩码Token化器），将分割掩码表示为最多32个由粗到细的离散token，使LMM像生成文本一样直接生成分割结果，无需额外的图像条件掩码解码器，在多个分割基准上达到SOTA。 现有LMM驱动的分割方法主要有三种范式，各有不足： 边界点序列：（如…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "分层掩码Token化"
  - "大型多模态模型"
  - "分割"
  - "向量量化"
  - "视觉定位"
---

# HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model

**会议**: ICCV 2025  
**arXiv**: [2503.13026](https://arxiv.org/abs/2503.13026)  
**代码**: [github.com/yayafengzi/LMM-HiMTok](https://github.com/yayafengzi/LMM-HiMTok)  
**领域**: 图像分割 / 多模态大模型  
**关键词**: 分层掩码Token化, 大型多模态模型, 分割, 向量量化, 视觉定位

## 一句话总结

提出HiMTok（分层掩码Token化器），将分割掩码表示为最多32个由粗到细的离散token，使LMM像生成文本一样直接生成分割结果，无需额外的图像条件掩码解码器，在多个分割基准上达到SOTA。

## 研究背景与动机

现有LMM驱动的分割方法主要有三种范式，各有不足：

**边界点序列**（如PolyFormer、VistaLLM）：将掩码表示为多边形顶点序列，但有限的顶点数难以表示复杂形状和多区域

**隐藏状态+掩码解码器**（如LISA、PixelLM、PSALM）：LMM输出特殊token的隐藏状态，再由额外的SAM/Mask2Former解码。存在三个局限：
   - LLM对精确空间定位学习不充分
   - 掩码在输入输出间表示不一致（特殊token仅作标识，丢失了对应的hidden state信息）
   - 架构复杂，掩码解码器需再次使用原图

**图像生成式**（VQ-GAN量化为2D token）：过于冗余、性能不够竞争力

核心问题：能否让LMM原生获得分割能力，像生成文本一样生成掩码，且无需外部分割模型？

## 方法详解

### 整体框架

HiMTok系统包含三部分：
- **掩码Token化器(MT)**：将分割掩码编码为若干1D潜在token
- **向量量化层(VQ)**：离散化潜在token
- **掩码去Token化器(MD)**：从离散token重建分割掩码

配合3阶段训练方案将分割能力渐进式注入LMM：
- Stage 1：训练HiMTok（单模态掩码重建）
- Stage 2：LMM + HiMTok联合训练（低分辨率图像，对齐视觉-语言与掩码token）
- Stage 3：仅训练LMM（高分辨率图像，精调）

### 关键设计

1. **分层掩码Token化**：受TiTok启发，将掩码图像压缩为32个1D离散token。关键创新是引入**因果注意力机制**：每个潜在token条件于输入掩码patch和前序token，确保由粗到细的层级关系：

$$p(m_1,...,m_K|\mathcal{M}) = \prod_{k=1}^K p(m_k|\mathcal{M}, m_1,...,m_{k-1})$$

前面的token主要对应粗糙位置和原型，后面的token聚焦局部细粒度细节。这种设计与LLM的自回归原则天然一致。

2. **分层掩码损失(Hierarchical Mask Loss, HML)**：在不同层级进行显式监督确保分层特性。第 $l$ 层级使用前 $l$ 个mask token由MD独立重建 $\hat{M}^{(l)}$，用不同大小高斯核模糊的掩码标签 $M^{(l)}$ 进行监督：

$$\mathcal{L}_{mask} = \sum_l \mathcal{L}_{mask}^{(l)}(\hat{M}^{(l)}, M^{(l)})$$

每级损失包含BCE Loss + Dice Loss。少token对应粗略高斯分布，多token对应精细边界。训练时按逆幂律分布采样部分层级以提高效率。

3. **双向信息流**：在训练数据中同时包含 box→mask 和 mask→box 两个方向的转换，使LMM学习检测和分割之间的固有关联。边界框由LMM直接生成而非从掩码解析。有趣发现：先输出mask token再输出box（视觉思维链）可以提升视觉定位精度。

### 损失函数 / 训练策略

- Stage 1：掩码重建任务，HiMTok全量训练，256×256分辨率，codebook大小1024，32个latent token
- Stage 2：交叉熵损失 + 分层掩码损失联合优化，LMM（InternVL 2.5基础）+ HiMTok部分参与训练，低分辨率448×448输入，7.1M数据
- Stage 3：仅交叉熵损失，仅训练LMM，高分辨率输入，5.0M数据（分割数据比例降至0.24）
- GPU小时：A800共2,752 GPU-hours（192+1920+640）

## 实验关键数据

### 主实验

**Referring Expression Segmentation (RefCOCO/+/g, cIoU)**：

| 方法 | w/ SFM | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|------|--------|-------------|--------------|--------------|
| LISA-7B(ft) | ✓ | 74.9 | 65.1 | 67.9 |
| PixelLM-7B | ✓ | 73.0 | 66.3 | 69.3 |
| PSALM | ✓ | 83.6 | 72.9 | 73.8 |
| u-LLaVA | ✓ | 83.0 | 77.1 | 77.1 |
| **LMM_HiMTok-8B** | **✗** | 81.1 | 77.1 | 75.8 |
| **LMM_HiMTok-8B(ft)** | **✗** | **85.0** | **79.7** | **80.0** |
| LMM_HiMTok-8B(ft)+SAM | ✓ | 85.9 | 80.5 | 80.1 |

不依赖任何分割基础模型(SFM)的情况下达到SOTA，大幅超越之前的SFM-free和SFM-based方法。

**开放词汇分割 (mIoU)**：

| 方法 | ADE20K (A-150) | PASCAL Context | PASCAL VOC |
|------|----------------|----------------|------------|
| PSALM | 18.2 | 48.5 | 81.3 |
| LaSagnA | 14.3 | 46.1 | 69.8 |
| **LMM_HiMTok-8B** | **25.0** | 43.9 | **82.0** |

### 消融实验

**分层掩码损失(HML)的效果**（RefCOCO val/RefCOCO+ val/RefCOCOg val）：

| HML | RefCOCO | RefCOCO+ | RefCOCOg |
|-----|---------|----------|----------|
| ✗ | 79.2 | 64.7 | 63.9 |
| ✓ | **81.1** | **77.1** | **75.8** |

没有HML时RefCOCO+/g大幅下降（-12.4/-11.9），且必须使用全长32 token才能工作；有HML则支持灵活token长度。

**Mask Token长度对REC(Visual Grounding)的影响**：

| Token数 → box | Acc@0.5 | Acc@0.9 |
|---------------|---------|---------|
| 0 (直接预测box) | ~90.3 | ~57 |
| 16 → box | ~92 | ~73 |
| 32 → box | **~93** | **~78** |

mask token作为视觉思维链，显著提升了高精度定位(Acc@0.9)。

### 关键发现

- 16个mask token已能达到82.8% cIoU，32个再提升2.5%
- 双向信息流中，mask→box方向更有价值（因前序mask token比直接预测box更容易生成）
- 模型在ReasonSeg上val和test得分几乎相同（60.7 vs 60.8 gIoU），说明文本理解能力强
- 通用图像理解能力基本保持（MME各维度与InternVL2.5-8B可比）
- 小物体分割仍是主要挑战（cIoU显著低于整体）

## 亮点与洞察

- **范式突破**：首次实现了不依赖外部分割模型的高质量LMM分割，掩码token如同新语言被LLM学习
- **输入输出一致性**：掩码的token化和去token化在LLM输入输出中一致，这是此前hidden state方案做不到的
- **分层设计与LLM自回归天然契合**：由粗到细的token层次完美匹配next-token-prediction
- **双向信息流的视觉CoT效果**：先分割再定位的思路新颖，为LMM的视觉推理提供了新视角
- **架构简洁**：去token化器是轻量级transformer，推理时不需要原图

## 局限与展望

- mask token长度需预定义，无法根据物体形状复杂度自适应
- 当前模型较被动，需要用户指定referring expression，不能主动分割所有感兴趣物体
- 缺少多尺度特征设计，细粒度区域分割表现受限
- 小物体分割性能与整体差距较大
- Stage 2训练需1,920 GPU-hours（A800），训练成本不低

## 相关工作与启发

- TiTok证明自然图像可被压缩为少量1D token，启发了掩码的紧凑表示
- VAR的coarse-to-fine next-scale prediction思路被应用到掩码层级
- InternVL 2.5作为基础LMM提供了强大的视觉语言基础
- LISA/PSALM等hidden state方案的局限性被清晰分析，为新范式提供了动机

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 全新的掩码表示范式，分层设计优雅，双向信息流有洞察
- **实验充分度**: ⭐⭐⭐⭐ 覆盖RES/GRES/ReasonSeg/OVS/REC/通用理解，消融详尽
- **写作质量**: ⭐⭐⭐⭐ 三种范式的对比图清晰，方法描述完整
- **价值**: ⭐⭐⭐⭐⭐ 为LMM分割开辟了新方向，代码开源，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] F-LMM: Grounding Frozen Large Multimodal Models](../../CVPR2025/segmentation/f-lmm_grounding_frozen_large_multimodal_models.md)
- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[ICCV 2025\] Refer to Any Segmentation Mask Group With Vision-Language Prompts](refer_to_any_segmentation_mask_group_with_vision-language_prompts.md)
- [\[ICCV 2025\] Advancing Visual Large Language Model for Multi-granular Versatile Perception](advancing_visual_large_language_model_for_multi-granular_versatile_perception.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)

</div>

<!-- RELATED:END -->
