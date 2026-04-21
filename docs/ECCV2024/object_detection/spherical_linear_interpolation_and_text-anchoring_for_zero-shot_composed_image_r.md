---
title: >-
  [论文解读] Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval
description: >-
  [ECCV 2024][目标检测][zero-shot composed image retrieval] 提出 Slerp-based ZS-CIR 方法，通过球面线性插值（Slerp）直接融合 VLP 模型的图像和文本嵌入构造组合查询表示，配合 Text-Anchored-Tuning (TAT) 用 LoRA 微调图像编码器缩小模态间隙，在 CIRR/CIRCO/FashionIQ 上达到 SOTA。
tags:
  - ECCV 2024
  - 目标检测
  - zero-shot composed image retrieval
  - spherical linear interpolation
  - text-anchored tuning
  - LoRA
  - CLIP/BLIP
---

# Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval

**会议**: ECCV 2024  
**arXiv**: [2405.00571](https://arxiv.org/abs/2405.00571)  
**代码**: 未提供  
**领域**: 视觉-语言 / 图像检索  
**关键词**: zero-shot composed image retrieval, spherical linear interpolation, text-anchored tuning, LoRA, CLIP/BLIP

## 一句话总结
提出 Slerp-based ZS-CIR 方法，通过球面线性插值（Slerp）直接融合 VLP 模型的图像和文本嵌入构造组合查询表示，配合 Text-Anchored-Tuning (TAT) 用 LoRA 微调图像编码器缩小模态间隙，在 CIRR/CIRCO/FashionIQ 上达到 SOTA。

## 研究背景与动机

**领域现状**：组合图像检索（CIR）是一种使用参考图像 + 修改文本来检索目标图像的任务。监督式 CIR 方法依赖昂贵的三元组标注〈参考图, 文本意图, 目标图〉，限制了可扩展性。零样本 CIR（ZS-CIR）方法使用通用的图像-文本对进行训练，具有更好的泛化能力。

**现有痛点**：当前主流 ZS-CIR 方法（Pic2Word, SEARLE, LinCIR）采用"伪词 token"策略——用投影模块将图像映射为文本词 token，拼接到文本中再过文本编码器生成组合表示。这存在两个问题：(1) 投影模块会扭曲原始图像表示；(2) 组合嵌入被限制在文本编码器输出空间中，无法充分捕获图像-文本联合信息。

**核心矛盾**：VLP 模型的图像和文本嵌入分布在同一超球面上（因为用余弦相似度训练），但两个模态之间存在显著的"模态间隙"（modality gap），直接在它们之间做插值效果受限。

**本文目标**：设计一种简单有效的图像-文本组合方案，避免伪词 token 的缺陷，并缩小模态间隙以提升组合检索性能。

**切入角度**：既然 VLP 模型的嵌入在超球面上，那么球面线性插值（Slerp）是最自然的融合方式；通过冻结文本编码器+LoRA 微调图像编码器来缩小模态间隙。

**核心 idea**：Slerp 直接插值图像和文本嵌入 + 文本锚定微调缩小模态间隙，简单有效地实现零样本 CIR。

## 方法详解

### 整体框架
方法分为两个阶段：(1) **Slerp-based ZS-CIR**——在推理时用球面线性插值融合查询图像嵌入和文本嵌入得到组合表示；(2) **Text-Anchored-Tuning (TAT)**——训练阶段冻结文本编码器，仅用 LoRA 微调图像编码器使图像嵌入靠近对应文本嵌入。两者结合产生最终的高性能 ZS-CIR 模型。

### 关键设计

1. **Spherical Linear Interpolation (Slerp) 检索**

    - 功能：在 VLP 嵌入超球面上找到图像和文本嵌入的中间表示作为组合查询
    - 核心思路：给定图像嵌入 $\mathbf{v}$ 和文本嵌入 $\mathbf{w}$（均为 l2 归一化），通过 Slerp 构造组合嵌入：
    $\mathbf{c} = \text{Slerp}(\mathbf{v}, \mathbf{w}; \alpha) = \frac{\sin((1-\alpha)\theta)}{\sin(\theta)}\mathbf{v} + \frac{\sin(\alpha\theta)}{\sin(\theta)}\mathbf{w}$
      其中 $\theta = \cos^{-1}(\mathbf{v} \cdot \mathbf{w})$ 是两个嵌入之间的角度，$\alpha \in [0,1]$ 控制图像和文本的贡献比例
    - 设计动机：VLP 模型使用缩放余弦相似度训练，图像和文本嵌入自然分布在超球面上。Slerp 沿球面弧线路径做插值（而非简单线性插值），保持嵌入在超球面上的分布特性。关键观察：text-only 检索通常优于 image-only，因此取 $\alpha \geq 0.8$（偏重文本）效果最佳
    - **不需要训练**：纯推理时操作，无需额外的投影模块或训练

2. **Text-Anchored-Tuning (TAT)**

    - 功能：缩小 VLP 模型中图像和文本嵌入之间的模态间隙
    - 核心思路：冻结文本编码器 $E_T$（保持文本表示的强表达力作为"锚点"），仅用 LoRA 在图像编码器 $E_I$ 上添加少量可训练参数 $\mathcal{P}_{lora}$（LoRA_α=16, rank=16, dropout=0.1），使图像嵌入重新对齐到对应的文本嵌入位置。训练目标是标准的 batch 对比损失：
    $\mathcal{L}_{cont.} = \mathcal{L}_{I2T} + \mathcal{L}_{T2I}$
      其中温度 $\tau$ 固定为 $1/0.07$ 确保训练稳定
    - 设计动机：(1) 文本在 CIR 中起主导作用（text-only 甚至超过部分方法），因此保留文本编码器原始能力至关重要；(2) LoRA 保留图像编码器原有知识的同时允许微调对齐；(3) 仅训练 <0.5% 参数即可完成对齐，单个 epoch 即收敛
    - **训练效率**：C-B32 + LLaVA-Align 数据集训练不到 0.5 小时

3. **$\alpha$ 参数的数据集适应性调节**

    - 功能：根据数据集特性调节 Slerp 中文本和图像的权重
    - 核心思路：CIRR 数据集中文本意图更关键，设 $\alpha=0.9$（更偏文本）；CIRCO 和 FashionIQ 中图像也很重要，设 $\alpha=0.8$
    - 设计动机：不同检索域对图像和文本的依赖程度不同，$\alpha$ 提供了用户可调的控制旋钮。实验表明 $\alpha=1.0$（纯文本）会显著下降，说明图像信息不可或缺

4. **推理流程**

    - 功能：完成组合图像检索
    - 核心思路：(1) 将 gallery 图像通过 $E_I$ 编码得到 $\mathbf{v}_g$；(2) 查询图像和查询文本分别通过 $E_I$ 和 $E_T$ 得到 $\mathbf{v}_q$ 和 $\mathbf{w}_q$；(3) Slerp 融合得到 $\mathbf{c}_q$；(4) 计算 $\mathbf{c}_q$ 与所有 $\mathbf{v}_g$ 的余弦相似度排序
    - 设计动机：不需要特定文本模板（如 "a photo of [$]"），查询文本直接使用，避免了模板选择成为性能瓶颈

### 损失函数 / 训练策略
- **损失函数**：标准 batch-wise 对比损失（与 VLP 预训练相同）
  $$\mathcal{L}_{I2T} = -\frac{1}{N_B}\sum_{i=1}^{N_B}\log\frac{\exp(\mathbf{v}_i^T \cdot \mathbf{w}_i / \tau)}{\sum_{j=1}^{N_B}\exp(\mathbf{v}_i^T \cdot \mathbf{w}_j / \tau)}$$
- **训练数据**：Laion-2M（默认）/ LLaVA-Align (585K) / CC3M (2.3M)
- **训练配置**：8×A100-80GB，batch size 1024，AdamW lr=1e-4，单 epoch
- **可训练参数**：< 0.5%（仅 LoRA 参数）

## 实验关键数据

### CIRR 主实验

| Backbone | 方法 | R@1 | R@5 | R@10 | Rs@1 | Rs@2 | Rs@3 |
|----------|------|-----|------|------|------|------|------|
| CLIP-L/14 | Pic2Word | 23.90 | 51.70 | 65.30 | - | - | - |
| CLIP-L/14 | SEARLE | 24.22 | 52.41 | 66.29 | 53.71 | 74.63 | 87.61 |
| CLIP-L/14 | LinCIR | 25.04 | 53.25 | 66.68 | 57.11 | 77.37 | 88.89 |
| CLIP-L/14 | Slerp (无训练) | 24.43 | 49.93 | 62.29 | 57.71 | 77.59 | 88.80 |
| CLIP-L/14 | **Slerp+TAT** | **30.94** | **59.40** | **70.94** | **64.70** | **82.92** | **92.31** |
| BLIP-L/16 | Slerp | 28.60 | 55.37 | 65.66 | 65.16 | 83.90 | 92.05 |
| BLIP-L/16 | **Slerp+TAT** | **33.98** | **61.74** | **72.70** | **68.55** | **85.11** | **93.21** |

### CIRCO 主实验

| Backbone | 方法 | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|----------|------|-------|--------|--------|--------|
| CLIP-L/14 | SEARLE | 11.68 | 12.73 | 14.33 | 15.12 |
| CLIP-L/14 | LinCIR | 12.59 | 13.58 | 15.00 | 15.85 |
| CLIP-L/14 | **Slerp+TAT** | **18.46** | **19.41** | **21.43** | **22.41** |
| BLIP-L/16 | **Slerp+TAT** | **17.84** | **18.44** | **20.24** | **21.07** |

### TAT 消融实验

| 配置 | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|------|-------|--------|--------|--------|
| (a) LLaVA-Align (0.58M) | 17.05 | 18.23 | 20.11 | 21.05 |
| (b) CC3M (2.3M) | 16.98 | 17.82 | 19.62 | 20.58 |
| (c) Laion-2M (默认) | **18.46** | **19.41** | **21.43** | **22.41** |
| (e) None-anchoring (双塔均微调) | 8.26 | 8.90 | 10.07 | 10.71 |
| (f) Image-anchoring (冻图调文) | 7.54 | 7.73 | 8.79 | 9.30 |
| (g) Pic2Word-Laion-2M | 8.93 | 9.96 | 11.50 | 12.02 |

### 关键发现
- **仅 Slerp（无训练）** 已可匹敌或超越需要训练的伪词 token 方法（如 SEARLE），证明球面插值的合理性
- **TAT 仅需 0.58M 数据和单个 epoch**，即可大幅超越使用数百万数据训练数十 epoch 的方法
- **文本锚定是关键**：None-anchoring（8.26 mAP@5）和 Image-anchoring（7.54）远不如 Text-anchoring（18.46），证明保持文本编码器不变的重要性
- **$\alpha=1.0$（纯文本）性能显著下降**，说明图像嵌入在 Slerp 中仍起重要作用
- TAT 训练的模型还可作为监督式 CIR 方法的更优初始化 checkpoint

## 亮点与洞察
- **回归本质的简洁设计**：在所有人都在设计复杂的投影模块和伪词 token 时，本文指出 VLP 嵌入在超球面上的几何性质可以直接利用——Slerp 就是超球面上最自然的插值方式。这种"看到本质后化繁为简"的思路非常优雅
- **不对称微调的洞察**：文本在 CIR 中比图像更重要→保持文本编码器不变作为锚点→仅调图像编码器向文本靠拢。这一设计逻辑链条清晰，且通过消融实验（text-anchoring >> none-anchoring >> image-anchoring）得到了强有力验证

## 局限与展望
- $\alpha$ 需要人工为每个数据集设定，未实现自适应调节
- TAT 使用固定温度 $\tau$，未探索温度动态调节对模态间隙收敛的影响
- 在 FashionIQ 的部分设置下 Slerp+TAT 并非一致最优（C-B32 R@10 上 Slerp 更好），可能因为时尚域图像信息权重与文本不同
- 仅验证了 CLIP 和 BLIP 两种 VLP 模型，对更新的模型（如 SigLIP、EVA-CLIP）的适用性未验证

## 相关工作与启发
- **vs Pic2Word [Saito et al.]**：Pic2Word 用投影模块将图像转为伪词 token 再交给文本编码器，Slerp 直接在嵌入空间插值，避免了信息损失
- **vs SEARLE [Baldrati et al.]**：SEARLE 需要 3M 图像-文本对 + 5.5M 文本样本训练，TAT 仅需 0.58M 数据和单 epoch 即超越
- **vs LinCIR [Gu et al.]**：LinCIR 在嵌入空间做线性组合，Slerp 做球面线性插值——后者在超球面几何上更合理
- **vs LoRA [Hu et al.]**：TAT 基于 LoRA 实现参数高效微调，但关键创新在于"文本锚定"策略而非 LoRA 本身（消融实验证明仅加 LoRA 到 Pic2Word 效果甚微）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Slerp 直接插值的思路极其简洁且有几何洞见支撑，Text-Anchored-Tuning 设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准(CIRR/CIRCO/FashionIQ)、三种backbone、详细消融（数据集/锚定策略/α/监督CIR初始化）
- 写作质量: ⭐⭐⭐⭐ 方法阐述清晰，动机和数学推导连贯
- 价值: ⭐⭐⭐⭐⭐ 训练效率极高(<0.5h)、性能大幅领先，实用价值突出

<!-- RELATED:START -->

## 相关论文

- [Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](../../CVPR2026/object_detection/beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)
- [Zero-Shot Detection of AI-Generated Images](zero-shot_detection_of_ai-generated_images.md)
- [OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)
- [Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](be_yourself_bounded_attention_for_multisubject_texttoimage_g.md)
- [Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)

<!-- RELATED:END -->
