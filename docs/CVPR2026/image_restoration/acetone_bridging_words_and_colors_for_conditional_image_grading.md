---
title: >-
  [论文解读] AceTone: Bridging Words and Colors for Conditional Image Grading
description: >-
  [CVPR 2026][图像恢复][色彩调色] 提出AceTone，首个支持文本和参考图像多模态条件色彩调色的统一框架，通过VQ-VAE将3D-LUT压缩为64个离散token，训练VLM预测LUT token序列，再用GRPO强化学习对齐色彩相似度和美学偏好，在风格迁移和指令调色上LPIPS改善50%。
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "色彩调色"
  - "3D-LUT"
  - "VQ-VAE tokenizer"
  - "VLM"
  - "GRPO强化学习"
---

# AceTone: Bridging Words and Colors for Conditional Image Grading

**会议**: CVPR 2026  
**arXiv**: [2604.00530](https://arxiv.org/abs/2604.00530)  
**代码**: [https://github.com/martian422/AceTone](https://github.com/martian422/AceTone)  
**领域**: 图像修复  
**关键词**: 色彩调色, 3D-LUT, VQ-VAE tokenizer, VLM, GRPO强化学习

## 一句话总结
提出AceTone，首个支持文本和参考图像多模态条件色彩调色的统一框架，通过VQ-VAE将3D-LUT压缩为64个离散token，训练VLM预测LUT token序列，再用GRPO强化学习对齐色彩相似度和美学偏好，在风格迁移和指令调色上LPIPS改善50%。

## 研究背景与动机

**领域现状**：色彩调色（toning/grading）对图像风格和情感至关重要。现有方法要么依赖预定义滤镜库的权重组合，要么用CNN逐patch重着色。参考图风格迁移和文本指令调色两个任务使用不兼容的模型。

**现有痛点**：(1) 现有方法表达能力或效率不足；(2) 对抗损失（GAN）训练不稳定、模式崩溃；(3) 缺乏与人类审美偏好的对齐机制；(4) 参考迁移和文本调色需要独立模型。

**核心矛盾**：色彩调色既需要精确的色彩控制（LUT的优势），又需要理解复杂的语义指令（VLM的优势），但两者未被有效结合。

**切入角度**：将LUT作为色彩变换的原子操作token化，让VLM来生成这些token。

**核心idea**：(1) VQ-VAE tokenizer把 $3 \times 32^3$ 的LUT压缩为64个离散token；(2) VLM预测LUT token序列；(3) GRPO用色彩相似度+美学评分做奖励对齐。

## 方法详解

### 整体框架
三阶段训练：(1) LUT Tokenizer训练（VQ-VAE）→(2) 生成预训练（VLM学习LUT token预测）→(3) 后训练（SFT适配任务+GRPO对齐偏好）。推理时：查询图+文本/参考图→VLM预测LUT tokens→解码为3D-LUT→应用到图像。

### 关键设计

1. **3D LUT Tokenizer (VQ-VAE)**:

    - 功能：将连续的 $3 \times 32 \times 32 \times 32$ LUT压缩为64个离散token
    - 核心思路：3D卷积编码器逐步下采样到 $4 \times 4 \times 4 \times D$→向量量化层(K=256个码字)→3D卷积解码器。损失 $\mathcal{L} = \mathcal{L}_{rec} + \beta \mathcal{L}_{commit}$
    - 保真度：$\Delta E < 2$（人眼几乎不可感知的色差）
    - 设计动机：LUT本质是3D颜色映射体积，VQ-VAE能有效压缩并保持高保真

2. **VLM的LUT Token预测**:

    - 功能：让VLM学会从视觉-文本输入自回归预测LUT token序列
    - **生成预训练**：大量(图像, LUT, prompt)三元组，$\mathcal{L}_{gen} = -\sum \log p_\theta(z_t | z_{<t}, I, L(I), c)$
    - **SFT**：分别为风格迁移(PST)和指令调色(IGG)设计训练数据。PST提供参考图和查询图；IGG用Qwen2.5-VL-32B为(图像, LUT)对生成编辑指令
    - 设计动机：将色彩变换形式化为token序列生成问题，统一了参考和文本两种条件

3. **GRPO强化学习对齐**:

    - 功能：让模型输出的色彩调色对齐人类美学偏好
    - 两个奖励函数：
        - $r_{color}$：色彩相似度，$\frac{1}{\max(2, \Delta E) - 1}$（$\Delta E < 2$ 时最大奖励）
        - $r_{aes}$：美学评分，用预训练DeQA模型评估视觉愉悦度
    - 标准GRPO训练：采样G个候选LUT→计算奖励→组内归一化优势→策略更新
    - 设计动机：避免GAN的不稳定性，先建立稳定的似然生成模型，再用RL对齐偏好

4. **AceTone-800K数据集**:

    - ~10K许可LUT滤镜库 + PPR-10K专家调色 + PCA聚类后的8192个核心LUT
    - 800K自动标注的(图像, LUT, 指令)元组
    - 两个基准：AceTone-Bench Transfer（1024 样本）和 AceTone-Bench Instruct（128 样本）

### 损失函数 / 训练策略
Tokenizer：MSE+commitment loss。预训练/SFT：交叉熵。RL：GRPO目标+KL正则。

## 实验关键数据

### 主实验（风格迁移PST-50）

| 方法 | Aes.↑ | PSNR↑ | LPIPS↓ | ΔE↓ |
|------|-------|-------|--------|-----|
| Neural Preset | 3.03 | 21.24 | 0.15 | 9.57 |
| SA-LUT | 3.07 | 21.64 | 0.16 | 9.01 |
| ModFlow | 3.08 | 20.13 | 0.16 | 10.62 |
| **AceTone** | **3.29** | **24.26** | **0.09** | **7.26** |

AceTone-Bench[Transfer]上LPIPS从0.22(SA-LUT)降至**0.11**（改善50%）。

### 消融实验

| 配置 | Aes.↑ | LPIPS↓ | 说明 |
|------|-------|--------|------|
| 仅预训练 | 基线 | 基线 | 基础LUT预测能力 |
| + SFT | +提升 | +提升 | 任务适配 |
| + GRPO | **最优** | **最优** | 美学对齐关键 |
| 无美学奖励 | 下降 | 不变 | 美学分对感知质量贡献大 |
| 无色彩奖励 | 不变 | 下降 | 色彩准确度需要色彩奖励保证 |

### 关键发现
- GRPO阶段的贡献主要体现在美学分提升和色彩一致性优化
- LUT tokenizer的保真度（ΔE<2）是整个pipeline精度的基础
- 数据多样性对GRPO训练至关重要——用全训练集vs子集的效果差异显著
- 首次证明VLM可以有效预测3D色彩变换的离散表示

## 亮点与洞察
- **LUT Token化的创新**：将3D-LUT从连续体积巧妙压缩为64个离散token，使色彩变换成为VLM的"语言"，打通了语言模型和色彩操作的边界
- **分阶段学习范式**：先似然预训练建立稳定基础，再RL对齐偏好的范式避免了GAN训练的不稳定性，为色彩调色的可扩展训练提供了新路径
- **统一多模态条件**：同一个模型同时支持参考图和文本指令两种调色模式

## 局限与展望
- LUT是全局色彩变换，无法处理局部调色（如仅调整天空颜色）
- 32^3分辨率的LUT精度有限（极端色彩变换可能出现量化伪影）
- GRPO训练需要大量采样和奖励计算，训练成本较高
- 美学评估模型(DeQA)自身的偏好可能会被"学习"到模型中

## 相关工作与启发
- **vs Neural Preset/SA-LUT**: 预定义LUT库的组合，表达能力有限。AceTone从零生成LUT
- **vs 扩散模型编辑**: 扩散模型可以重着色但延迟高且可能破坏结构。LUT应用是无损的
- **vs CLIP指导方法**: CLIP将文本映射到色彩操作但输入受限于几个词。VLM理解复杂指令

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ LUT token化+VLM生成+GRPO对齐的完整创新链
- 实验充分度: ⭐⭐⭐⭐ 定量+用户研究，但数据集尚未公开
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数据集构建细节完整
- 价值: ⭐⭐⭐⭐⭐ 开创了语言驱动色彩调色的新方向，对影视后期等行业有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[ICML 2026\] One-shot Conditional Sampling: MMD meets Nearest Neighbors](../../ICML2026/image_restoration/one-shot_conditional_sampling_mmd_meets_nearest_neighbors.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] Learning to Translate Noise for Robust Image Denoising](learning_to_translate_noise_for_robust_image_denoising.md)
- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)

</div>

<!-- RELATED:END -->
