---
title: >-
  [论文解读] Single Domain Generalization for Few-Shot Counting via Universal Representation Matching
description: >-
  [CVPR 2025][多模态VLM][少样本计数] 提出首个面向少样本计数的单域泛化模型URM，通过将CLIP的通用视觉-语言表征蒸馏到可学习原型中参与相关性构建，在不损失域内性能的前提下大幅提升跨域泛化能力（MAE降低27.5%）。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "少样本计数"
  - "域泛化"
  - "CLIP知识蒸馏"
  - "通用表征匹配"
  - "视觉-语言原型"
---

# Single Domain Generalization for Few-Shot Counting via Universal Representation Matching

**会议**: CVPR 2025  
**arXiv**: [2505.16778](https://arxiv.org/abs/2505.16778)  
**代码**: [https://github.com/jbr97/URM](https://github.com/jbr97/URM)  
**领域**: 多模态VLM  
**关键词**: 少样本计数、域泛化、CLIP知识蒸馏、通用表征匹配、视觉-语言原型

## 一句话总结

提出首个面向少样本计数的单域泛化模型URM，通过将CLIP的通用视觉-语言表征蒸馏到可学习原型中参与相关性构建，在不损失域内性能的前提下大幅提升跨域泛化能力（MAE降低27.5%）。

## 研究背景与动机

少样本计数（FSC）根据少量标注样例估计图像中目标物体数量。现有方法（如FamNet、LOCA、DAVE等）遵循"提取-匹配"流水线：先从样例中提取原型，再与图像特征做相关性匹配，最后回归密度图。然而，这些方法在跨域场景下性能急剧下降——例如FSC-147中物体清晰可见，但FSCD-LVIS中遮挡和质量退化更严重。

本文指出核心问题：**从窄分布源域学到的原型本身分布也窄**，在分布差异大的目标域中匹配效果差。通过t-SNE可视化验证了这一假设——传统方法在跨域设定下特征边界模糊。解决方案是引入CLIP预训练的通用视觉-语言表征，其在海量多样数据上训练，具备域不变的特性。

## 方法详解

### 整体框架

URM整体框架包含：ImageNet预训练ResNet50骨干网络提取多尺度特征→Prompt Encoder编码样例框信息→通用视觉和语言原型通过交叉注意力与编码后的图像特征交互→蒸馏冻结CLIP的通用表征（仅训练时）→拼接后的原型与图像特征做相关性匹配→回归密度图。推理时不需要CLIP，效率与无外部知识的方法一致。

### 关键设计

1. **Universal Vision Representation Learning（通用视觉表征学习）**:
    - 功能：获取具有域不变特性的局部物体级视觉表征
    - 核心思路：利用MaskCLIP改造CLIP视觉编码器，去除最后两层QK，改造V层和线性层为卷积层，生成类别相关的密集分割图 $\mathcal{M}$。然后通过Mask Pooling从CLIP视觉token中提取局部物体表征：$r_v = \text{MaskPool}(\mathcal{V}(\mathbf{I}), \mathcal{M})$。可学习的视觉原型 $p_v$ 通过 $N_1$ 层cross-attention与编码图像特征 $f^E$ 交互，并用feature mimicry loss蒸馏CLIP表征：$\mathcal{L}_{V\text{-}KD} = \frac{1}{|\mathcal{B}|}\sum \|r_{v_k} - p_{v_k}\|$
    - 设计动机：CLIP的CLS token关注全局属性，image token局部辨别力弱，直接用不适合局部物体匹配。MaskCLIP+Mask Pooling能精确提取物体区域的CLIP表征，同时去除非相关区域噪声

2. **Universal Language Representation Learning（通用语言表征学习）**:
    - 功能：获取包含丰富类别判别特征的文本表征
    - 核心思路：不仅用手写模板"A photo of {}"，还跟随CuPL方法用GPT-4生成包含关键判别特征的定制化描述（如物体形状、纹理、典型场景等）。多个prompt经CLIP文本编码器后平均得到通用文本表征 $r_t$。可学习的语言原型 $p_l$ 同样通过cross-attention更新，并用 $\mathcal{L}_{L\text{-}KD}$ 蒸馏
    - 设计动机：简单模板"A photo of {}"无法充分描述细粒度类别特征（如ImageNet的"A toy {}"模板在FSC-147上效果差），LLM生成的多样化描述能更好表征目标类别

3. **Universal Representation Matching（通用表征匹配）**:
    - 功能：利用蒸馏后的V-L原型构建相关性图进行计数
    - 核心思路：拼接视觉和语言原型 $\text{concat}(p_v, p_l)$，通过 $N_2$ 层cross-attention与编码图像特征 $f^E$ 做匹配（$q=f^E, k=v=\text{concat}(p_v,p_l)$），输出相关性图后由回归头预测密度图
    - 设计动机：传统方法只用视觉信息构建相关性，加入经CLIP蒸馏的V-L双通道原型能在域不变的特征空间中进行匹配，天然具备跨域泛化能力

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{\text{density}} + \alpha \mathcal{L}_{V\text{-}KD} + (1-\alpha)\mathcal{L}_{L\text{-}KD}$，其中density loss为标准化的 $\ell_2$ 损失：$\mathcal{L}_{\text{density}} = \frac{1}{2\mathcal{B}}\sum \frac{1}{N_k}\|\mathbf{G}_k - \mathbf{R}_k\|_2^2$。蒸馏权重 $\alpha = 0.9$，说明语言表征贡献更大（已对齐到视觉空间的CLIP文本编码更具信息量）。知识蒸馏仅在训练阶段进行，推理时CLIP模型移除，保持与其他方法一致的计算效率。ResNet50 backbone冻结，其余参数训练150 epochs，AdamW lr=1e-4，单V100 GPU。

## 实验关键数据

### 主实验（跨域Few-shot计数）

| 方向 | 方法 | MAE↓ | RMSE↓ |
|------|------|------|-------|
| FSC→FSCD-LVIS | MPCount (CVPR24 SOTA) | 25.11 | 41.32 |
| FSC→FSCD-LVIS | **URM** | **21.87** | **38.42** |
| FSCD-LVIS→FSC | MPCount | 22.07 | 80.17 |
| FSCD-LVIS→FSC | **URM** | **21.17** | **73.42** |
| FSC→FSCD-LVIS (zero-shot) | MPCount | 27.68 | 44.58 |
| FSC→FSCD-LVIS (zero-shot) | **URM** | **23.54** | **39.93** |

### 消融实验

| 配置 | MAE↓ | RMSE↓ | 说明 |
|------|------|-------|------|
| Baseline (无蒸馏) | 30.15 | 48.04 | 仅改架构无泛化增益 |
| + Language (Naive Template) | 25.44 | 42.59 | 简单模板即有显著提升 |
| + Language (Prompt Generator) | 23.83 | 41.03 | LLM生成prompt进一步提升 |
| + Vision ([CLS] Token) | 23.53 | 41.12 | 全局token不如局部表征 |
| + Vision (Global Pooling) | 22.94 | 39.44 | 全局池化有效但次优 |
| + Vision (Mask Pooling) | **21.87** | **38.42** | 局部物体表征最优，总提升27.5% |

| 蒸馏权重 $\alpha$ | 0 | 0.25 | 0.5 | 0.75 | **0.9** | 1 |
|-------------------|---|------|-----|------|---------|---|
| MAE↓ | 26.54 | 23.44 | 23.30 | 22.37 | **21.87** | 23.83 |

### 关键发现

- 仅改变架构（用可学习原型匹配）对泛化毫无帮助（MAE 30.15 vs. baseline），必须有CLIP知识蒸馏
- 语言知识贡献大于视觉（$\alpha=0.9$ 最优），因为CLIP文本编码已对齐到视觉空间，本身就是跨模态的桥梁
- V-L表征互补：仅用其中一种（$\alpha=0$ 或 $\alpha=1$）都劣于两者结合
- 推理时无需CLIP，效率与传统方法一致——蒸馏是"一次性投资"

## 亮点与洞察

- **核心发现极简而深刻**：域泛化的关键在于原型的分布宽度，而非架构复杂度。CLIP的通用表征天然解决了这个问题
- **训练-推理解耦**：蒸馏仅在训练时进行，推理时移除CLIP，实现零额外推理开销
- **t-SNE可视化直观验证假设**：清晰展示了窄分布原型在跨域时边界模糊，CLIP蒸馏后边界清晰

## 局限与展望

- 仅验证了两个数据集之间的跨域泛化，更多样化的域（如卫星图、医学图像）未涉及
- 语言prompt依赖GPT-4生成，有一定成本
- CLIP蒸馏假设目标类别有对应的文本描述，对极端长尾类别可能效果受限
- 未探索更强的视觉基础模型（如DINOv2）作为蒸馏教师的效果

## 相关工作与启发

- CuPL利用LLM生成定制化prompt提升CLIP零样本分类，本文将此思路引入计数任务的原型蒸馏
- MaskCLIP改造CLIP编码器获取像素级预测，本文用于提取局部物体表征
- CLIP-KD中feature mimicry被证明是最有效的CLIP知识蒸馏方式，本文直接采用
- 该方法的"蒸馏通用表征到task-specific原型"范式可推广到其他需要域泛化的视觉任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个单域泛化少样本计数，核心发现简洁有力
- 实验充分度: ⭐⭐⭐⭐ 跨域/域内/零样本多设定+详细消融，但数据集只有两个
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析→假设验证→方法设计→实验验证，逻辑流畅
- 价值: ⭐⭐⭐⭐ 蒸馏CLIP到task原型的思路具有广泛适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Towards Multimodal Domain Generalization with Few Labels](../../CVPR2026/multimodal_vlm/towards_multimodal_domain_generalization_with_few_labels.md)
- [\[CVPR 2025\] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] Data Distributional Properties as Inductive Bias for Systematic Generalization](data_distributional_properties_as_inductive_bias_for_systematic_generalization.md)

</div>

<!-- RELATED:END -->
