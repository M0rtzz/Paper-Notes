---
title: >-
  [论文解读] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation
description: >-
  [ICCV 2025][人体理解][人体运动生成] 提出 GenM3 框架，通过 Multi-Expert VQ-VAE (MEVQ-VAE) 学习统一的离散运动表示，以及 Multi-path Motion Transformer (MMT) 处理模态内变异和跨模态对齐，整合 11 个运动数据集（约 220 小时），在 HumanML3D 上达到 SOTA FID 0.035。
tags:
  - ICCV 2025
  - 人体理解
  - 人体运动生成
  - VQ-VAE
  - Transformer
  - 大规模数据集
  - 文本条件生成
---

# GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation

**会议**: ICCV 2025  
**arXiv**: [2503.14919](https://arxiv.org/abs/2503.14919)  
**代码**: 无  
**领域**: Human Understanding  
**关键词**: 人体运动生成, VQ-VAE, 多路径Transformer, 大规模数据集, 文本条件生成

## 一句话总结

提出 GenM3 框架，通过 Multi-Expert VQ-VAE (MEVQ-VAE) 学习统一的离散运动表示，以及 Multi-path Motion Transformer (MMT) 处理模态内变异和跨模态对齐，整合 11 个运动数据集（约 220 小时），在 HumanML3D 上达到 SOTA FID 0.035。

## 研究背景与动机

基于文本描述生成多样且准确的人体运动是计算机视觉的关键研究方向。尽管预训练大模型在文本和图像领域取得巨大成功，运动生成领域仍面临三大挑战：

**运动数据分布异质性**：不同数据集在运动类型、录制设备上差异显著，联合训练可能导致在一个数据集上优化反而损害另一个数据集的效果

**缺乏专用运动预训练骨干网络**：现有方法将预训练语言模型适配到运动任务（如 MotionGPT），但运动和文本之间存在本质的结构和上下文差异

**缺乏高质量大规模统一运动数据集**：动作捕捉数据获取成本高，现有数据集各自局限于特定运动类型

典型案例：MotionGPT 使用 HumanML3D 训练后，面对简单的文本描述"a person places their hands on their hips"也无法生成正确的运动。

## 方法详解

### 整体框架

GenM3 包含两个核心组件和三阶段训练流程：
- **MEVQ-VAE**：将连续运动序列离散化为 token
- **MMT（Multi-path Motion Transformer）**：处理文本和运动的跨模态建模
- 训练流程：Stage 1 训练 MEVQ-VAE → Stage 2 预训练 MMT（仅运动） → Stage 3 文本条件训练

### 关键设计

1. **Multi-Expert VQ-VAE (MEVQ-VAE)**: 

    - 在标准 VQ-VAE 的编码器和解码器中引入多专家 1D 卷积层
    - **每个 block 包含 3 个标准 1D 卷积层 + 1 个多专家卷积层**
    - 所有 $e_q$ 个专家同时激活，输出为加权组合：$y = \sum_{i=1}^{e_q} w_i \cdot \text{Conv}_i(x)$
    - $w_i$ 为可学习权重，自适应调节各专家贡献
    - 使用共享 codebook（8192 个 32 维码字），下采样率为 4
    - 损失函数：$\mathcal{L}_q = \mathcal{L}_{rec} + \beta \mathcal{L}_{commit}$

2. **Motion Descriptor（运动描述器）**: 

    - CLIP 编码器生成文本嵌入 $\mathbf{E}_t$ 和全局文本特征 $\mathbf{e}_t$
    - 运动 token 通过运动嵌入器得到 $\mathbf{E}_m$
    - 通过文本引导的注意力聚合生成上下文摘要：$\mathbf{E}_{ctx} = \text{mean}(\text{softmax}(\mathbf{E}_m \mathbf{E}_t) \mathbf{E}_t)$
    - 上下文 token 提供高层运动语义，增强跨模态对齐

3. **Multi-path Motion Transformer (MMT)**: 

    - 前半部分（9 层）：标准 Transformer 自注意力 + FFN
    - 后半部分（9 层）：多路径 Transformer，在 FFN 层引入三条并行路径：
        - **运动路径**：处理运动 token，每条路径内含多个密集激活专家
        - **文本路径**：处理文本 token
        - **跨模态共享路径**：同时处理运动和文本 token，促进跨模态对齐
    - 每条路径中的专家输出通过门控函数加权：$\mathbb{E}_p(x) = \sum_i g_{p,i}(x) \mathbb{E}_{p,i}(x)$
    - 三条路径输出拼接后通过投影层：$\text{Output} = \mathbf{W}_{proj}([\mathbb{E}_{motion}; \mathbb{E}_{text}; \mathbb{E}_{cross-modal}]) + b_{proj}$

### 损失函数 / 训练策略

- **Stage 1（MEVQ-VAE）**：重建损失 + commitment 损失，使用 moving average 更新 codebook
- **Stage 2（预训练）**：仅使用运动数据，masked modeling 方式，所有路径均以运动 token 为输入。损失：$\mathcal{L} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\setminus \mathcal{M}})$
- **Stage 3（文本条件训练）**：使用运动+文本数据对，三条路径全部激活。仍用 masked modeling 框架，但依赖文本和可见运动 token
- 优化器：AdamW，学习率 2×10⁻⁴，warmup + cosine annealing
- Batch size 160，训练 120K iterations
- 推理时采用并行解码：所有 token 初始化为 [Mask]，逐步替换低置信 token

## 实验关键数据

### 主实验

**HumanML3D 基准（30FPS 评估器）**:

| 方法 | FID↓ | R-Precision Top3↑ | MMDist↓ | Diversity↑ |
|------|------|-------------------|---------|-----------|
| Real | 0.002 | 0.785 | 2.982 | 9.458 |
| T2M-GPT | 0.160 | 0.770 | 3.083 | 9.653 |
| MMM | 0.110 | 0.784 | 2.951 | 9.484 |
| **GenM3** | **0.046** | **0.804** | **2.852** | 9.675 |

**HumanML3D 基准（20FPS 评估器，与更多方法对比）**:

| 方法 | FID↓ | R-Precision Top3↑ | Diversity↑ |
|------|------|-------------------|-----------|
| MoMask | 0.045 | 0.807 | - |
| MotionGPT | 0.232 | 0.778 | 9.528 |
| OMG | 0.381 | 0.784 | 9.657 |
| **GenM3** | **0.035** | 0.795 | 9.341 |

**IDEA400 零样本泛化**:

| 方法 | FID↓ | R-Precision Top3↑ | MMDist↓ |
|------|------|-------------------|---------|
| T2M-GPT | 7.947 | 0.301 | 5.488 |
| MMM | 6.001 | 0.307 | 4.980 |
| GenM3 | 4.430 | 0.335 | 4.732 |
| **GenM3*** | **4.232** | **0.338** | **4.520** |

### 消融实验

**VQ 方法对比**:

| 方法 | FID↓ |
|------|------|
| 标准 VQ | 0.098 |
| **MEVQ-VAE** | **0.048** |
| RVQ | 0.043 |
| MERVQ（RVQ + Multi-Expert） | **0.032** |
| G-RVQ | 0.045 |
| FSQ | 0.057 |

**多路径 Transformer 消融**:

| 运动路径 | 文本路径 | 跨模态路径 | FID↓ | Diversity↑ |
|---------|---------|-----------|------|-----------|
| ✓ | - | - | 0.058 | 9.400 |
| ✓ | ✓ | - | 0.045 | 9.282 |
| ✓ | - | ✓ | 0.044 | 9.344 |
| ✓ | ✓ | ✓ | **0.035** | 9.341 |

**Dense MoE vs Sparse MoE**:

| 类型 | FID↓ | R-Precision Top3↑ |
|------|------|-------------------|
| Sparse MoE | 0.058 | 0.799 |
| **Dense MoE** | **0.046** | **0.804** |

### 关键发现

- **GenM3 的 FID 0.035 大幅领先**：相比第二名 MoMask (0.045) 提升 22%
- **预训练在大规模混合数据集上对 GenM3 提升最大（35.21%）**，显著高于对 MMM 和 T2M-GPT 的提升，归功于多专家设计对异质数据的适应
- Multi-Expert 设计可推广到 RVQ：MERVQ 的 FID 从 0.043 降至 0.032
- Dense MoE（全激活）优于 Sparse MoE（部分激活），因为全激活更好地捕获数据集间的共性
- 三条路径同时使用时效果最佳，跨模态路径比文本路径贡献更大（0.044 vs 0.045）
- GenM3* 在 IDEA400 零样本上表现更好，说明更多文本-运动数据对增强泛化能力

## 亮点与洞察

- **多专家密集激活的设计哲学**：与主流 sparse MoE 不同，GenM3 选择全专家激活+权重学习，更好地处理数据异质性
- **三阶段训练策略的递进逻辑**：先学好运动表示 → 再学通用运动模式 → 最后对齐文本条件
- **Motion Descriptor 的简洁设计**：通过文本引导的注意力将运动序列压缩为上下文 token，计算高效
- **11 个数据集的统一整合**：覆盖单人运动、双人交互、人物交互，约 220 小时，超越了现有最大的运动数据集

## 局限与展望

- Diversity 指标略低于某些方法（如 MoMamba 的 9.871），可能是多专家设计倾向于生成更稳定的结果
- 11 个数据集的统一需要大量手工标注和预处理（如 BABEL 使用 ChatGLM 生成描述、IMHD 手动标注）
- 未与最新的扩散式运动生成方法（如 OMG 的不同规模变体）在相同条件下对比
- 仅评估了 SMPL 参数的 22 关节骨架，未扩展到全身（包括手部和面部）
- 推理速度和内存占用未报告

## 相关工作与启发

- MoMask/MMM：masked completion 范式的运动生成
- MotionGPT：将运动 token 作为新语言与 LLM 联合建模
- OMG：大规模扩散模型进行运动生成
- VLMo：多路径 Transformer 的灵感来源（视觉-语言领域）
- 启发：运动生成领域的 "scaling law" 已开始显现，专用运动骨干网络 + 大规模数据的组合潜力巨大

## 评分

- **新颖性**: ⭐⭐⭐⭐ Multi-Expert VQ-VAE 和 Multi-path Transformer 的双重多专家设计新颖
- **实验充分度**: ⭐⭐⭐⭐ HumanML3D 和 IDEA400 评估、VQ 方法对比、组件消融
- **写作质量**: ⭐⭐⭐⭐ 框架图清晰，训练阶段阐述有条理
- **价值**: ⭐⭐⭐⭐⭐ FID 0.035 大幅刷新 SOTA，11 个统一数据集对社区有重要贡献

<!-- RELATED:START -->

## 相关论文

- [KinMo: Kinematic-Aware Human Motion Understanding and Generation](kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [GENMO: A GENeralist Model for Human MOtion](genmo_a_generalist_model_for_human_motion.md)
- [Large Motion Model for Unified Multi-Modal Motion Generation](../../ECCV2024/human_understanding/large_motion_model_for_unified_multimodal_motion_generation.md)
- [SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](../../CVPR2025/human_understanding/simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text](rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)

<!-- RELATED:END -->
