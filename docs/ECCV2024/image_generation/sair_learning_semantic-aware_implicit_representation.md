---
title: >-
  [论文解读] SAIR: Learning Semantic-aware Implicit Representation
description: >-
  [ECCV 2024][图像生成][隐式神经表示] 本文提出语义感知隐式表示（SAIR），通过构建语义隐式表示（SIR）和外观隐式表示（AIR）两个模块，将CLIP提取的文本对齐语义嵌入融入隐式神经函数，使其在大面积缺失区域的图像修复任务中远超仅依赖外观信息的方法，在CelebAHQ上 PSNR 提升1.65-2.69dB。
tags:
  - ECCV 2024
  - 图像生成
  - 隐式神经表示
  - 语义感知
  - 图像修复
  - CLIP
  - 图像重建
---

# SAIR: Learning Semantic-aware Implicit Representation

**会议**: ECCV 2024  
**arXiv**: [2310.09285](https://arxiv.org/abs/2310.09285)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 隐式神经表示, 语义感知, 图像修复, CLIP, 图像重建

## 一句话总结
本文提出语义感知隐式表示（SAIR），通过构建语义隐式表示（SIR）和外观隐式表示（AIR）两个模块，将CLIP提取的文本对齐语义嵌入融入隐式神经函数，使其在大面积缺失区域的图像修复任务中远超仅依赖外观信息的方法，在CelebAHQ上 PSNR 提升1.65-2.69dB。

## 研究背景与动机

**领域现状**：隐式神经表示（如LIIF）通过将连续坐标映射到颜色值实现图像重建，在超分辨率等任务中表现出色。这类方法使用编码器提取外观特征，再用MLP将坐标+邻域特征映射为RGB颜色。

**现有痛点**：现有隐式表示方法**只建立连续的外观映射**，完全忽略像素背后的语义信息。当输入图像严重退化（如大面积区域缺失）时，邻域外观特征不可用，重建结果出现明显伪影和语义不一致——例如眼睛被遮挡时，仅靠周围"脸部皮肤"的外观特征无法推断出应该重建一只"眼睛"。

**核心矛盾**：外观特征是局部的、低级的，无法提供"这个位置应该属于什么物体"的全局语义理解。缺失区域的外观特征趋于零，但语义信息可以从上下文推断。

**本文目标** (a) 构建能在缺失区域推断语义的隐式表示；(b) 在图像修复中利用语义信息补偿缺失的外观信息。

**切入角度**：利用预训练CLIP模型的文本对齐嵌入作为语义信息。CLIP的视觉编码器输出的特征天然与语义类别对齐（可直接用文本做zero-shot分类），但其分辨率低且无法处理缺失区域。

**核心 idea**：先用隐式函数在语义嵌入空间完成缺失区域的语义补全（SIR），再将补全后的语义信息注入外观隐式表示（AIR）指导颜色重建。

## 方法详解

### 整体框架
SAIR包含两个串联模块：
- **输入**：带缺失区域的图像 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ 和掩膜 $\mathbf{M}$
- **SIR**：利用修改版CLIP提取低分辨率语义嵌入 $\mathbf{Z}^{sem} \in \mathbb{R}^{h \times w \times c}$，再用隐式函数上采样并补全缺失区域的语义
- **AIR**：融合外观特征（CNN提取）和SIR输出的语义特征，用另一个隐式函数预测任意坐标的颜色
- **输出**：完整重建图像

### 关键设计

1. **修改版 CLIP 编码器（MaskCLIP风格）**:

    - 功能：从图像提取空间感知的文本对齐语义嵌入
    - 核心思路：去除CLIP ViT最后一个注意力层的query和key投影，将value投影和输出线性层改为两个 $1 \times 1$ 卷积。这样不引入额外参数也不改变特征空间，但让CLIP输出从全局单向量变为 $h \times w$ 的空间特征图
    - 设计动机：原始CLIP输出只有全局CLS token，无空间位置信息。MaskCLIP的修改让每个位置都有独立的语义嵌入

2. **语义隐式表示（SIR）**:

    - 功能：将低分辨率、可能有缺失的语义嵌入上采样为全分辨率的完整语义特征
    - 核心思路：$\mathbf{z}_{\mathbf{p}}^{sem} = \sum_{\mathbf{q} \in \mathcal{N}_{\mathbf{p}}} \omega_{\mathbf{q}} f_\theta([\mathbf{z}_{\mathbf{q}}^{sem}, \mathbf{M}[\mathbf{q}]], \text{dist}(\mathbf{p}, \mathbf{q}))$，即将掩膜信息拼接到语义特征中，让MLP感知哪些邻域是缺失的
    - 设计动机：简单的resize上采样无法补全缺失区域的语义。隐式函数通过邻域加权聚合+MLP变换可以从已有语义推断缺失位置的语义（如从"脸颊"推断被遮挡位置是"眼睛"）
    - 三个关键性质：(a) 语义嵌入与文本类别对齐；(b) 非整数坐标可插值；(c) 缺失区域可补全

3. **外观隐式表示（AIR）**:

    - 功能：融合外观和语义信息，预测任意坐标的颜色
    - 核心思路：$c_{\mathbf{p}} = \sum_{\mathbf{q} \in \mathcal{N}_{\mathbf{p}}} \omega_{\mathbf{q}} f_\beta([\mathbf{z}_{\mathbf{q}}^{app}, \text{SIR}(\mathbf{I}, \mathbf{q})], \text{dist}(\mathbf{p}, \mathbf{q}))$
    - 输入三部分：CNN提取的外观嵌入 $\mathbf{z}^{app}$（与图像同分辨率）、SIR输出的增强语义嵌入、坐标距离
    - 设计动机：当像素缺失时，$\mathbf{z}_{\mathbf{p}}^{app}$ 趋于零，但SIR补全的语义信息告诉网络"这里应该是什么"

### 损失函数
仅使用 $L_1$ 重建损失：$\mathcal{L} = \|c_{pred} - c_{gt}\|_1$。MLP $f_\theta$（SIR）和 $f_\beta$（AIR）均为4层ReLU网络，隐藏维度256。Adam优化器，学习率0.0001，200个epoch。

## 实验关键数据

### 主实验（CelebAHQ数据集）

| 方法 | 0-20%掩膜PSNR↑ | 20-40%PSNR↑ | 40-60%PSNR↑ | 20-40%SSIM↑ | 40-60%LPIPS↓ |
|------|----------------|-------------|-------------|-------------|-------------|
| EdgeConnect | 34.53 | 27.30 | 22.32 | 0.889 | 0.195 |
| RFRNet | 34.93 | 27.50 | 22.77 | 0.890 | 0.185 |
| LAMA | 36.04 | 29.14 | 22.94 | 0.932 | 0.152 |
| MISF | 36.32 | 29.85 | 23.91 | 0.932 | 0.133 |
| LIIF | 35.27 | 28.80 | 23.30 | 0.923 | 0.136 |
| **SAIR** | **37.97** | **31.49** | **24.87** | **0.944** | **0.124** |

SAIR在所有掩膜比例和所有指标上均达到最优，20-40%掩膜比例下PSNR比MISF高1.64、比LIIF高2.69。

### ADE20K数据集对比

| 方法 | 20-40% PSNR↑ | 40-60% PSNR↑ | 40-60% LPIPS↓ |
|------|-------------|-------------|--------------|
| MISF | 24.97 | 20.59 | 0.233 |
| LIIF | 24.57 | 19.79 | 0.274 |
| **SAIR** | **26.44** | **21.88** | **0.193** |

在场景级数据集ADE20K上同样取得最优，证明方法不仅适用于人脸也适用于复杂场景。

### 关键发现
- **语义信息在大面积缺失时尤为关键**：掩膜比例越大（40-60%），SAIR相对LIIF的提升越大（1.57 vs 2.69 PSNR），说明外观信息越少，语义的补偿作用越重要
- **SIR有效补全缺失语义**：可视化显示CLIP编码器对被掩膜覆盖区域输出的特征质量很差，SIR模块成功恢复了这些区域的语义嵌入
- **语义信息加速收敛**：SAIR和LIIF的训练损失在相似step收敛，但SAIR的PSNR从训练一开始就高于LIIF
- 纯外观隐式表示（LIIF）在大面积缺失时生成模糊、有明显边界的结果；SAIR生成的结果语义连贯且视觉自然

## 亮点与洞察
- **语义+外观双层隐式表示架构**：SIR→AIR的串联设计优雅地解决了"先理解再重建"的问题。先在低维语义空间完成补全，再用补全后的语义指导高维像素重建。这个两阶段思路可推广到其他需要理解缺失区域内容的任务
- **MaskCLIP修改零开销获取空间语义**：不增加参数、不改变特征空间就让CLIP输出空间感知嵌入的trick非常实用
- **掩膜信息融入SIR**：将 $\mathbf{M}[\mathbf{q}]$ 拼接到MLP输入，让网络显式知道哪些邻域是缺失的，简单但有效

## 局限性
- 只在图像修复任务上验证，论文声称可用于生成、编辑、分割等任务但未实验
- CLIP语义嵌入分辨率很低（ViT-B/16下约为 $\frac{H}{16} \times \frac{W}{16}$），对需要精细语义边界的场景可能不足
- SIR的语义补全是通过局部邻域MLP实现的，对超大面积缺失（>60%）的全局理解能力有限
- 训练需要两块V100 GPU，200个epoch，相对于一些修复方法训练成本较高
- 没有与扩散模型修复方法（如 Repaint）进行对比

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将语义层面的隐式表示与外观隐式表示结合，思路清晰新颖
- 实验充分度: ⭐⭐⭐⭐ 两个数据集三种掩膜比例的全面评估，但缺少与扩散模型方法对比
- 写作质量: ⭐⭐⭐⭐ 框架图和数学符号清晰，但公式较多影响可读性
- 价值: ⭐⭐⭐⭐ 语义+外观双层隐式表示的思路有推广价值，对修复领域有启发意义

<!-- RELATED:START -->

## 相关论文

- [Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)
- [Implicit Style-Content Separation using B-LoRA](implicit_style-content_separation_using_b-lora.md)
- [Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)

<!-- RELATED:END -->
