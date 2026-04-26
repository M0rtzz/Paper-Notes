---
title: >-
  [论文解读] MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration
description: >-
  [ICCV 2025][图像恢复][高光谱图像复原] 提出 MP-HSIR 框架，通过整合光谱提示（通用低秩光谱模式）、文本提示和视觉提示三种模态的引导信息，构建了统一的高光谱图像复原模型，在包含去噪、去模糊、超分辨率、修复、去雾、波段补全等 9 个 HSI 复原任务上全面超越现有 all-in-one 方法和多个任务专用方法。
tags:
  - ICCV 2025
  - 图像恢复
  - 高光谱图像复原
  - 多提示学习
  - 全能复原
  - 光谱提示
  - 文本-视觉协同
---

# MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration

**会议**: ICCV 2025  
**arXiv**: [2503.09131](https://arxiv.org/abs/2503.09131)  
**代码**: [GitHub](https://github.com/ZhehuiWu/MP-HSIR)  
**领域**: 图像复原  
**关键词**: 高光谱图像复原, 多提示学习, 全能复原, 光谱提示, 文本-视觉协同

## 一句话总结

提出 MP-HSIR 框架，通过整合光谱提示（通用低秩光谱模式）、文本提示和视觉提示三种模态的引导信息，构建了统一的高光谱图像复原模型，在包含去噪、去模糊、超分辨率、修复、去雾、波段补全等 9 个 HSI 复原任务上全面超越现有 all-in-one 方法和多个任务专用方法。

## 研究背景与动机

高光谱图像（HSI）相比 RGB 图像提供了更高的光谱分辨率，在城市规划、农业生产和环境监测中至关重要。但 HSI 在成像过程中常受到多种退化影响，且不同退化对光谱特征的影响截然不同：

- **噪声**增加光谱波动
- **压缩**降低某些波段的光谱反射率
- **雾霾**全局偏移光谱曲线

现有方法的三大局限：

**任务专用方法缺乏通用性**：针对特定退化设计的方法（如去噪、超分辨率、去雾）无法泛化到其他退化场景，实际应用中需要维护多个专用模型。

**现有 all-in-one 方法不适用于 HSI**：
   - **视觉提示**方法（如 PromptIR）可解释性差
   - **文本提示**方法（如 InstructIR）存在文本-图像语义鸿沟
   - **双模态提示**方法（如 DACLIP-IR）的预训练 VLM 在 HSI 退化建模上能力有限
   - 以上方法全部**忽略了 HSI 的光谱特性**，常导致光谱畸变

**HSI 专用 all-in-one 方法不足**：DDS2M 和 HIR-Diff 等扩散模型方法推理慢、需复杂超参调优；PromptHSI 仅使用文本提示，缺乏光谱引导。

核心动机：设计一个能同时利用**光谱先验、退化语义和精细视觉特征**的统一框架，在不同退化类型和强度下都能有效复原 HSI。

## 方法详解

### 整体框架

MP-HSIR 采用三层级的编码器-解码器架构：
- 退化 HSI $\mathcal{Y} \in \mathbb{R}^{H \times W \times B}$ 经单层卷积提取浅层特征
- 三级层次编码器逐步下采样，每级包含多个 Prompt-Guided Spatial-Spectral Transformer Block (PGSSTB)
- 编码器-解码器跳跃连接处嵌入 Text-Visual Synergistic Prompt (TVSP) 模块
- 最终通过 $3 \times 3$ 卷积和图像级残差连接输出复原结果

### 关键设计

1. **提示引导的空间-光谱 Transformer (PGSSTB)**:

    - **功能**：作为核心构建块，融合空间自注意力 (SSA) 和提示引导的双分支光谱自注意力 (PGSSA)。
    - **核心思路**：SSA 在 $P \times P$ 滑动窗口内计算空间非局部相似性。PGSSA 包含两个互补分支：
        - **全局光谱自注意力**：标准的通道维度注意力，捕捉全局波段间依赖：
       $$A^g = \text{Softmax}\left(\frac{Q^g \cdot K^g}{\epsilon}\right), \quad \text{Attention}(Q^g, K^g, V^g) = W^P(A^g V^g)$$
        - **提示引导的局部光谱自注意力**：将输入分为 $\frac{HW}{P^2}$ 个不重叠 patch，每个 patch 通过 GAP 提取光谱特征 $M_j^s \in \mathbb{R}^{1 \times C}$
      两个分支的输出通过 GMLP 融合，实现全局与局部光谱建模的互补。
    - **设计动机**：HSI 的光谱维度包含丰富的任务相关信息。全局分支建模整体波段关系，局部分支关注空间区域内的光谱重建。结合 GMLP 的选择性融合，网络可以灵活平衡两者。

2. **光谱提示 (Spectral Prompt)**:

    - **功能**：为局部光谱自注意力提供通用低秩光谱模式作为先验知识。
    - **核心思路**：引入可学习的光谱提示 $P_S \in \mathbb{R}^{L \times D}$，其中 $L$ 为通用低秩光谱模式数量，$D$ 为维度。对局部光谱特征 $M_j^s$ 进行以下操作：
    $Q^l = \text{Softmax}(M_j^s W_1^l) P_S W_3^l$
    $[K^l, V^l] = M_j^s W_2^l W_3^l$
      即 query 通过对光谱提示的加权组合生成，而 key 和 value 直接从局部特征投射。这使得任何局部光谱特征都可以表示为通用低秩模式的线性组合。
    - **设计动机**：不同退化对光谱的影响不同，但**光谱的基本低秩结构是共享的**。光谱提示在训练中学习这些通用模式，提供跨退化类型的光谱重建引导。实验表明，即使在不同退化下，同一区域的光谱提示激活模式高度一致。

3. **文本-视觉协同提示 (TVSP)**:

    - **功能**：融合文本和视觉两种模态的提示信息，为复原过程提供退化特定的引导。
    - **核心思路**：
        - 退化预测器 $\Phi$ 对输入分类退化类型
        - 冻结的 CLIP 模型将预定义文本描述编码为文本提示 $P_T$：
       $$P_T = \Phi(\mathcal{X}) \cdot \text{Clip}(T_{text})$$
        - 引入可学习视觉提示 $P_V$，通过交叉注意力与文本提示融合
        - 融合结果拼接到编码器特征上，通过跳跃连接传递到解码器：
       $$F_l^{out} = \text{Concat}(F_l^e, \text{Attention}(P_T, P_V))$$
    - **设计动机**：文本提示提供全局退化类型信息但缺乏像素级精度；视觉提示提供精细的局部特征信息。交叉注意力融合两者，使退化信息在全局语义和局部细节层面同时发挥作用。相比直接使用 VLM 提取特征，TVSP 更灵活且可跨域适应。

### 损失函数 / 训练策略

- 使用 L1 损失
- AdamW 优化器（$\beta_1=0.9$, $\beta_2=0.999$）
- 自然场景 HSI：batch size 32，100 epochs，初始 lr $2 \times 10^{-4}$，cosine 退火到 $1 \times 10^{-6}$
- 遥感 HSI：batch size 32，300 epochs，初始 lr $1 \times 10^{-4}$，宽度因子 1.5
- 训练 patch 尺寸：$64 \times 64$，通道数：自然场景 31，遥感 100
- 两种数据分开训练（自然场景 vs 遥感）

## 实验关键数据

### 主实验（All-in-one 设置，9 个任务平均）

| 任务 | MP-HSIR | PromptIR (最强all-in-one) | 最强任务专用 | 说明 |
|------|---------|--------------------------|-------------|------|
| 高斯去噪 (ICVL) | **41.62/0.964** | 40.25/0.953 | LDERT: 41.92/0.969 | 逼近专用方法 |
| 复杂去噪 (ICVL) | **42.29/0.971** | 41.29/0.965 | LDERT: 43.42/0.977 | 逼近专用方法 |
| 高斯去模糊 (ICVL) | **48.07/0.990** | 47.67/0.990 | MLWNet: 47.66/0.990 | 超越专用方法 |
| 超分辨率 (ARAD) | 38.25/0.924 | 37.37/0.918 | **PIP: 38.36/0.926** | 接近专用方法 |
| 修复 (ICVL) | **51.53/0.996** | 46.38/0.990 | Restormer: 45.79/0.990 | 大幅超越全部 |
| 去雾 (PaviaU) | **39.59/0.986** | 37.41/0.982 | SCANet: 36.59/0.978 | 大幅超越全部 |
| 波段补全 (ARAD) | **56.48/0.999** | 46.60/0.994 | InstructIR: 51.31/0.997 | +5.17 dB |

### 消融实验

| 配置 | PSNR | SSIM | 参数量(M) | 说明 |
|------|------|------|----------|------|
| Baseline (仅空间 SA) | 39.24 | 0.963 | 20.93 | 基线 |
| + 文本提示 $P_T$ | 39.62 | 0.964 | 21.51 | +0.38 dB |
| + 视觉提示 $P_V$ | 39.57 | 0.964 | 23.68 | +0.33 dB |
| + $P_T$ + $P_V$ | 39.90 | 0.964 | 24.26 | 协同效果 |
| + 全局光谱 SA + $P_T$ + $P_V$ | 40.63 | 0.969 | 30.07 | 光谱 SA 关键 |
| + 局部光谱 SA + $P_T$ + $P_V$ + $P_S$ | 41.05 | 0.971 | 25.10 | 光谱提示最大贡献 |
| **Full Model** | **41.98** | **0.974** | 30.91 | 完整模型 |

### 关键发现

- **光谱提示的跨退化鲁棒性**：可视化显示，即使面对不同退化类型（噪声 vs 模糊），同一区域的光谱提示激活模式高度一致，说明光谱提示确实捕获了通用的光谱结构
- **泛化能力优异**：在运动去模糊（few-shot）和泊松去噪（zero-shot）两个未见过的任务上，仅用 5% 数据微调即超越所有 all-in-one 方法，在泊松去噪上甚至超越多个任务专用方法
- 在修复、去雾、波段补全三个任务上取得了**压倒性优势**，最大领先幅度达 5+ dB
- 光谱误差分析显示 MP-HSIR 在所有任务上实现了**最精确的光谱重建**

## 亮点与洞察

- **三层提示设计**：光谱提示（low-level 物理先验）+ 文本提示（high-level 语义）+ 视觉提示（mid-level 细节），覆盖了复原任务所需的不同层次信息
- **低秩光谱模式**：将光谱结构表示为可学习低秩模式的线性组合，思路简洁有效，类似字典学习的端到端可微分版本
- **任务覆盖广度**：一个模型处理 9 种退化（去噪×2, 去模糊×2, 超分辨率, 修复, 去雾, 波段补全, 泊松去噪），远超现有 all-in-one 方法
- **光谱提示激活一致性**的可视化验证令人信服
- TVSP 模块的设计比直接使用 VLM 更轻量、更灵活

## 局限与展望

- 自然场景和遥感 HSI 需要分别训练，尚未实现真正的跨域统一
- 退化预测器 $\Phi$ 需要预训练且依赖分类准确性
- CLIP 模型冻结使用，可能无法充分捕捉 HSI 特有的语义信息
- 光谱提示的低秩模式数量 $L$ 需要人为设定
- 训练 patch 尺寸 $64 \times 64$ 较小，可能限制对大尺度退化的建模能力
- 推理速度未做详细对比

## 相关工作与启发

- 光谱提示的设计可推广到其他具有先验结构知识的复原问题（如 MRI 的频率先验、遥感 SAR 的散射先验）
- TVSP 的文本-视觉融合策略可以启发多模态引导的通用图像复原
- 将 all-in-one 复原从 RGB 扩展到 HSI 的挑战揭示了光谱维度建模的重要性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 三模态提示框架设计新颖，光谱提示作为低秩先验的引入独到
- 实验充分度: ⭐⭐⭐⭐⭐ 9 个任务、13 个数据集、3 种设置（all-in-one/泛化/真实场景），极为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，但部分公式需要对照图表才能理解
- 价值: ⭐⭐⭐⭐⭐ 在 HSI 复原领域开创了多提示 all-in-one 的范式，性能提升显著

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] UniRes: Universal Image Restoration for Complex Degradations](unires_universal_image_restoration_for_complex_degradations.md)
- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](../../ECCV2024/image_restoration/seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](../../AAAI2026/image_restoration/large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)

<!-- RELATED:END -->
