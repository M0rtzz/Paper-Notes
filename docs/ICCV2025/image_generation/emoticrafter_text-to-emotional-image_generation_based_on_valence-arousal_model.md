---
title: >-
  [论文解读] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model
description: >-
  [图像生成] 提出 EmotiCrafter，首个基于连续 Valence-Arousal (V-A) 模型的情感图像生成方法，通过情感嵌入映射网络将 V-A 值融合到文本特征中，注入 Stable Diffusion XL 实现精确的内容+情感双重控制，生成图像在情感连续性和可控性上显著优于现有方法。
tags:
  - 图像生成
---

# EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model

## 论文信息

- **会议**: ICCV 2025
- **arXiv**: 2501.05710
- **代码**: [https://github.com/idvxlab/EmotiCrafter](https://github.com/idvxlab/EmotiCrafter)
- **领域**: 图像生成 / 情感计算
- **关键词**: emotional image generation, valence-arousal, SDXL, continuous emotion, cross-attention injection

## 一句话总结

提出 EmotiCrafter，首个基于连续 Valence-Arousal (V-A) 模型的情感图像生成方法，通过情感嵌入映射网络将 V-A 值融合到文本特征中，注入 Stable Diffusion XL 实现精确的内容+情感双重控制，生成图像在情感连续性和可控性上显著优于现有方法。

## 研究背景与动机

情感在信息传达中至关重要，然而生成包含精确情感的图像内容仍是未解难题。现有工作的局限：

**EmoGen**（情感图像生成开创工作）：仅支持离散情感标签（happy/sad），无法控制具体图像内容，无法捕获微妙的情感差异

**离散 vs 连续情感**：心理学家对情感类别的划分尚无共识，离散标签的表达范围有限

**情感编辑方法**（IET）：依赖特定视觉元素（颜色、纹理），情感表达深度不足

EmotiCrafter 提出**连续情感图像生成 (C-EICG)** 任务：给定自由文本提示 + V-A 连续值，生成同时满足内容描述和情感表达的图像。V-A 模型用二维笛卡尔空间表示情感：Valence（愉悦度，-3 到 3）和 Arousal（激活度，-3 到 3），可以自然表示从"无聊"到"疲倦"的微妙情感渐变。

## 方法详解

### 整体框架

EmotiCrafter = V-A 编码器 + 情感注入 Transformer (EIT) + SDXL 生成器

输入：文本提示 → 编码器 $\mathcal{E}$ → 中性特征 $f_n$ → 情感嵌入网络 $\mathcal{M}$ → 情感特征 $\hat{f}_e$ → 注入 SDXL → 情感图像

$$\hat{f}_e = \mathcal{M}(f_n | (v, a))$$

### 关键设计 1：V-A 编码器

两个独立的 MLP 分别处理 Valence 值和 Arousal 值：
- $e_v = \text{MLP}_V(v)$：V-feature
- $e_a = \text{MLP}_A(a)$：A-feature

### 关键设计 2：情感注入 Transformer (EIT)

基于修改版 GPT-2 架构，包含 12 个情感注入块 (EIB)。每个 EIB 的处理流程：

1. **输入投影**：将中性特征投影到 Transformer 空间
$$h_0 = P_{\text{in}}(f_n) + \text{PE}$$

2. **情感注入**（每个 EIB）：
$$h'_i = \text{self-attn}(\text{LN}(h_{i-1})) + h_{i-1}$$
$$h^{(v)}_i = \text{cross-attn}(\text{LN}(h'_i), e_v) + h'_i$$
$$h^{(v,a)}_i = \text{cross-attn}(\text{LN}(h^{(v)}_i), e_a) + h^{(v,a)}_i$$
$$h_i = \text{FFN}(\text{LN}(h^{(v,a)}_i)) + h^{(v,a)}_i$$

3. **残差预测**：网络预测中性和情感特征间的**残差**（而非直接预测情感特征）：
$$\hat{f}_r = P_{\text{out}}(\text{LN}(h_{12}))$$
$$\hat{f}_e = \hat{f}_r + f_n$$

关键：移除了原始 GPT-2 中的因果掩码（causal mask），因为不需要自回归生成。

### 关键设计 3：增强情感表达的损失函数

$$\mathcal{L} = \frac{1}{n} \mathbb{E}\left(\frac{1}{d(v,a)} \|\hat{f}_e - f^t_e\|^2\right)$$

**缩放残差学习**：放大目标残差以增强情感变化：
$$f^t_e = f_n + \alpha(f_e - f_n)$$

其中 $\alpha = 1.5$（通过消融确定），使生成图像的情感变化更明显。

**V-A 密度加权**：使用核密度估计 (KDE) 计算训练样本在 V-A 空间的分布密度 $d(v,a)$，对稀疏区域赋予更高权重以缓解数据不平衡。

### 训练数据构建

- 基于 39,843 张人工标注 V-A 值的图像（来自 OASIS、EMOTIC、FindingEmo 数据集）
- 使用 GPT-4 为每张图像生成配对的中性提示和情感提示（共享核心含义但情感表达不同）
- 众包验证所有 LLM 生成的提示，投票机制解决分歧
- 训练：AdamW 优化器，2 块 A800 GPU，200 epochs，batch size 768，约 7-8 小时

## 实验关键数据

### 主实验：定量对比

| 方法 | A-Error ↓ | V-Error ↓ | CLIPScore ↑ | CLIP-IQA ↑ |
|------|-----------|-----------|-------------|------------|
| Cross Attention | 1.923 | 2.080 | 26.266 | 0.949 |
| Time Embedding | 1.941 | 2.031 | **26.566** | 0.786 |
| Textual Inversion | 1.958 | 1.923 | 22.346 | 0.370 |
| GPT-4+SDXL | 1.860 | 1.517 | 25.907 | 0.906 |
| **Ours** | **1.828** | **1.510** | 23.067 | 0.881 |

- 情感准确度（V/A-Error）最佳
- 连续性对比（LPIPS-Continuous）：Ours 0.220 vs GPT-4+SDXL 0.361，情感过渡更平滑

### 消融实验

| 配置 | A-Error ↓ | V-Error ↓ | CLIPScore ↑ |
|------|-----------|-----------|-------------|
| α=1.0 | ~1.95 | ~1.65 | ~24.5 |
| **α=1.5 (Ours)** | **1.828** | **1.510** | **23.067** |
| α=2.0 | ~1.75 | ~1.40 | ~21.5 |
| w/o d(v,a) | 1.829 | 1.546 | 21.977 |

**关键发现**：
- α 控制内容-情感权衡：α 增大 → 情感更准确但语义偏移更大
- α=1.5 是最优平衡点，实际效果优于线性回归预测
- 密度加权 d(v,a) 提升 CLIPScore 约 1 分，V-Error 降低 0.04

### 用户研究

| 指标 | Ours | GPT-4+SDXL |
|------|------|-----------|
| A-Ranking Consistency ↑ | **0.759** | 0.165 |
| V-Ranking Consistency ↑ | **0.887** | 0.584 |
| A-Error ↓ | **1.327** | 2.029 |
| V-Error ↓ | **0.692** | 1.229 |
| Emotion Consistency ↑ | **4.215** | 3.525 |
| Emotion Smoothness ↑ | **4.240** | 3.195 |

所有指标均通过 Wilcoxon 检验显著优于基线（p<0.05)。

## 亮点与洞察

1. **新任务定义 (C-EICG)**：首次将连续情感控制引入图像生成，比离散标签更符合心理学建模
2. **残差学习 + 缩放因子**：不直接预测情感特征，而是预测中性-情感残差并放大，巧妙增强情感表达
3. **情感-内容解耦**：V-A 值可覆盖提示中的语义情感（如"儿童游乐场"配悲伤 V-A 值）
4. **空prompt 生成**：不输入文本仅给 V-A 值也能生成情感一致的图像，验证了情感嵌入的有效性
5. **细粒度控制**：V-A 增量 0.2 即可观察到可感知的图像变化

## 局限性

- Arousal 控制难度大于 Valence（与情感分析领域的一致发现：标注者对 Arousal 的一致性更低）
- 即使提示中未提及人物，模型也经常生成人类活动场景（训练数据中非人场景不足）
- 情感调整会轻微改变语义内容（CLIPScore 下降），需要语义保持损失项
- 仅基于 SDXL，未验证在 DiT/VAR 等新架构上的效果

## 相关工作与启发

- 与 EmoGen 的对比：C-EICG 是 EICG 的重要升级，从离散到连续、从无内容控制到自由文本
- 情感嵌入网络的设计可推广到其他连续条件控制任务（如图像风格强度、光照参数等）
- IP-Adapter 的交叉注意力注入策略在 Cross Attention baseline 中效果不佳，说明情感是更高层的语义概念，需要在特征层面融合而非在 UNet 层面注入

## 评分

⭐⭐⭐⭐ — 首次定义 C-EICG 任务有开创性意义，情感嵌入网络设计合理，用户研究充分。但实际应用价值受限于 Arousal 控制的固有困难和语义偏移问题。对情感计算和创意生成领域有启发价值。

<!-- RELATED:START -->

## 相关论文

- [DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](../../CVPR2025/image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [FlowTok: Flowing Seamlessly Across Text and Image Tokens](flowtok_flowing_seamlessly_across_text_and_image_tokens.md)
- [MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)

<!-- RELATED:END -->
