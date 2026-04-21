---
title: >-
  [论文解读] EditInfinity: Image Editing with Binary-Quantized Generative Models
description: >-
  [NeurIPS 2025][图像生成][二值量化生成模型] 提出 EditInfinity，首次将经典"图像反演-图像编辑"范式应用于二值量化自回归生成模型 Infinity，利用量化表示可获取精确中间监督的优势实现高精度图像反演，配合分段线性平滑核实现高保真编辑效果，在 PIE-Bench 上全面超越扩散模型基线。
tags:
  - NeurIPS 2025
  - 图像生成
  - 二值量化生成模型
  - Infinity
  - 图像反演
  - 分段线性平滑
  - 多尺度自回归编辑
---

# EditInfinity: Image Editing with Binary-Quantized Generative Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.20217](https://arxiv.org/abs/2510.20217)  
**代码**: [有](https://github.com/yx-chen-ust/EditInfinity)  
**领域**: 图像生成 / 图像编辑 / 自回归模型  
**关键词**: 二值量化生成模型, Infinity, 图像反演, 分段线性平滑, 多尺度自回归编辑

## 一句话总结

提出 EditInfinity，首次将经典"图像反演-图像编辑"范式应用于二值量化自回归生成模型 Infinity，利用量化表示可获取精确中间监督的优势实现高精度图像反演，配合分段线性平滑核实现高保真编辑效果，在 PIE-Bench 上全面超越扩散模型基线。

## 研究背景与动机

文本驱动图像编辑的经典范式包含两步：（1）图像反演——逆推生成轨迹，（2）沿轨迹在目标文本引导下编辑。该范式的核心瓶颈在于：

- **扩散模型中图像反演不精确**：无法获取源图像在生成轨迹上的精确中间表示，只能近似
- 近似误差会传播到编辑阶段，导致背景保持和语义对齐的 trade-off

关键洞察：**二值量化生成模型**（如 Infinity）将图像量化到离散潜空间建模，其天然特性是——**任意图像的精确多尺度量化表示可直接获得**。这意味着可以用精确中间结果作为监督信号进行图像反演优化，从根本上解决扩散模型的近似误差问题。

## 方法详解

### 整体框架

EditInfinity 基于 Infinity-2B（二值量化 T2I 模型）实现：
1. **图像反演**：优化可学习文本嵌入 + LoRA 微调，以精确量化 token 为监督
2. **图像编辑**：多尺度自回归 token 替换 + 分段线性平滑核确保无缝过渡

#### Infinity 预训练模型概要

- 图像 → 编码器 → 特征 F → 多尺度残差量化 {R_k}_{k=1}^K
- 每个尺度 k：残差 = F - F_{k-1}，下采样到 (h_k, w_k)，BSQ 二值量化
- 自回归建模：p(R_{1:K} | Ψ(t)) = Π_k p(R_k | R_{<k}, Ψ(t))
- Infinite-Vocabulary Classifier 将预测分解为 d 个独立二值分类器

### 关键设计

#### 图像反演（Image Inversion with Exact Supervision）

核心优势：量化 token R_{1...K}^sou 可作为精确监督信号。

**Textual Prompting Rectification（文本提示修正）：**
- 源文本 prompt t_sou 通常与源图像不精确匹配
- 附加 20 个可学习 prompt token t_l + 指令 prompt t_ins
- 冻结 Infinity 所有参数，仅优化 t_l
- 交叉熵损失：L_inv = -1/K Σ_k (R_k^sou · log p(R_k^inv | R_{<k}^sou, Ψ(t)))
- 精确量化 token 作为 ground truth，而非近似值

**Image Style Preservation（图像风格保留）：**
- 可学习 prompt 优化改善语义对齐，但可能改变全局风格
- 应用 LoRA 到 FFN 层，利用 low-rank 偏置的平滑全局修改特性
- 仅训练 20 步后停止 LoRA（防止过拟合导致忽略编辑意图）
- 训练好的 ΔW 在编辑阶段保留，保持源图像风格一致性

#### 图像编辑（Holistic Smoothing Strategy）

**分段线性平滑核 G：**
- 基于 Manhattan 距离 d^{i,j} = min_{(x,y)∈M} (|i-x| + |j-y|) 计算权重
- 三段式设计：
    - d ≤ τ₁: G=0（编辑区域，完全使用目标内容）
    - τ₁ < d < τ₂: G 线性插值（平滑过渡带）
    - d ≥ τ₂: G=1（未编辑区域，完全保留源内容）
- 默认 τ₁=1, τ₂=4，有效抑制边界拼接伪影

**Multi-scale Autoregressive Editing（多尺度编辑）：**
- 量化源图像得 R_{1:K}^sou
- 在每个尺度 k：
    - Infinity 生成 R_k^tar（条件于 [Ψ(t_tar, t_ins), t_l]）
    - 上采样 R_k^tar 和 R_k^sou 到最大分辨率
    - 在 G 引导下混合：E_k^tar = R_k^tar ⊙ (1-G) + R_k^sou ⊙ G
    - 若 k < K，下采样 E_k^tar 为 R̂_k^tar 供下一尺度使用
- 混合后的语义和结构跨尺度传播
- 最终 E_{1:K}^tar 解码为编辑图像

### 损失函数 / 训练策略

- 反演阶段：交叉熵损失优化可学习 prompt（精确量化 token 作 GT）
- LoRA 仅在 FFN 层，训练 20 步后冻结
- 编辑阶段无需训练，纯推理执行
- 硬件：2× NVIDIA L20 (反演), 1× NVIDIA L20 (编辑)

## 实验关键数据

### 主实验

**PIE-Bench 全量化对比**

| 方法 | 基座 | PSNR↑ | LPIPS×10³↓ | SSIM×10²↑ | Whole CLIP↑ | Edited CLIP↑ | IR×10↑ |
|------|------|-------|-----------|-----------|------------|-------------|--------|
| NTI (CVPR'23) | U-Net | 27.03 | 60.67 | 84.11 | 24.75 | 21.86 | 2.77 |
| PnP-Inv (ICLR'24) | U-Net | 22.46 | 106.06 | 79.68 | 25.41 | 22.62 | 4.17 |
| RF-Edit (ICML'25) | DiT | 23.22 | 131.18 | 81.44 | 25.22 | 22.40 | 5.18 |
| Gemini 2.0 | 商用 | 23.22 | 105.17 | 81.10 | 25.28 | 22.28 | 5.30 |
| **EditInfinity** | **AR** | **27.95** | **33.08** | **92.12** | **26.41** | **23.47** | **5.88** |

EditInfinity 在**背景保持**和**文本对齐**两个维度上全面领先。LPIPS 从 60.67 降至 33.08（最佳 U-Net 对比），PSNR 27.95（最高），IR 5.88（编辑成功率最高）。

**基座模型公平性验证（GenEval Benchmark）**

| 基座 | Overall |
|------|---------|
| SD v1.4 | 0.42 |
| FLUX.1-dev | 0.66 |
| Infinity-2B | 0.66 |

Infinity 与 FLUX 生成能力相当（0.66 vs 0.66），但 EditInfinity 大幅超越 FLUX-based 方法，证明方法优势而非基座优势。

### 消融实验

**平滑核消融（PIE-Bench random class）**

| G 配置 | PSNR↑ | LPIPS×10³↓ | IR×10↑ |
|--------|-------|-----------|--------|
| 无 G | 31.12 | 24.47 | 2.85 |
| Gaussian 核 | 28.15 | 32.91 | 4.61 |
| **Linear 核** | **28.50** | **31.58** | **5.39** |

无 G 时 IR 最低（编辑效果差），Linear 核在编辑质量和背景保持间取得最佳平衡。

**可学习 Prompt + LoRA 消融**：
- 去除两者：严重结构不一致
- 仅可学习 Prompt：文本对齐改善但全局风格偏移
- +LoRA (20步)：风格恢复一致性
- +LoRA (过多步)：过拟合，忽略编辑意图

### 关键发现

- **精确中间监督**是 EditInfinity 成功的核心——量化模型的天然优势
- 编辑速度极快（3.64s/次），反演成本前置（107s）但支持多次编辑
- 分段线性核优于高斯核，平滑过渡效果更好
- LoRA 训练步数需严格控制（20步），过多导致过拟合
- 用户研究中 43.2% 偏好率最高

**运行时间对比**

| 方法 | 反演 | 单次编辑 |
|------|------|---------|
| NTI | 95.5s | 10.3s |
| RF-Edit | 55.5s | 54.1s |
| **EditInfinity** | 107.1s | **3.6s** |

反演开销中等偏高，但单次编辑仅 3.6 秒——**比平均快 7 倍**，非常适合迭代编辑工作流。

## 亮点与洞察

1. **开辟新赛道**：首次将自回归量化模型用于图像编辑，利用精确量化表示的天然优势
2. **精确监督解决核心痛点**：量化 token 作为 GT 训练反演，从根本上优于扩散模型的近似
3. **极速迭代编辑**：3.6s/次编辑速度非常实用，前置反演成本可被多次编辑摊薄
4. **简洁的平滑核设计**：Manhattan 距离 + 线性插值，无需学习参数，效果优于高斯核
5. **公平评估设计完善**：GenEval 验证基座模型能力相当，排除优势来源于更强基座的质疑

## 局限与展望

- 反演阶段 107 秒较慢（虽然单次编辑快），可探索更快的文本优化方案
- 依赖用户提供编辑掩码（标准设定但限制了自动化）
- LoRA 训练步数（20 步）需要手动调优，不同图像可能需要不同设定
- 当前仅在 Infinity-2B 上验证，可扩展到更大模型或其他量化架构
- 量化模型的生成多样性可能不如扩散模型

## 相关工作与启发

- Infinity (Han et al., 2024)：BSQ 二值量化 + 多尺度残差预测，为 AR T2I 新基准
- I2SB 启发了"精确中间监督"思路——但 EditInfinity 的精确性是量化模型的固有特性
- 与 RF-Edit/StableFlow 等 DiT-based 方法形成扩散 vs 自回归的对比
- LoRA 微调作为风格保持手段，low-rank 偏置自然倾向全局平滑修改

## 评分

- 新颖性：⭐⭐⭐⭐⭐（首次将量化模型用于编辑，精确监督思路原创）
- 技术深度：⭐⭐⭐⭐（反演+编辑两阶段设计完整，各组件设计合理）
- 实验充分性：⭐⭐⭐⭐⭐（9 类编辑任务、8+ 对比方法、用户研究、运行时间、消融）
- 实用性：⭐⭐⭐⭐⭐（3.6s 单次编辑极具实用价值，代码开源）
- 表达清晰度：⭐⭐⭐⭐⭐（算法伪代码清晰，图示丰富，对比全面）

<!-- RELATED:START -->

## 相关论文

- [BitMark: Watermarking Bitwise Autoregressive Image Generative Models](bitmark_watermarking_bitwise_autoregressive_image_generative_models.md)
- [QVGen: Pushing the Limit of Quantized Video Generative Models](../../ICLR2026/image_generation/qvgen_pushing_the_limit_of_quantized_video_generative_models.md)
- [ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation](thermalgen_style-disentangled_flow-based_generative_models_for_rgb-to-thermal_im.md)
- [SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)
- [CAMILA: Context-Aware Masking for Image Editing with Language Alignment](camila_contextaware_masking_for_image_editing_with_language.md)

<!-- RELATED:END -->
