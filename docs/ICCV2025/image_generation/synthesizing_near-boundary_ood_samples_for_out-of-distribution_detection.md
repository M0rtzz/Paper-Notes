---
title: >-
  [论文解读] Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection
description: >-
  [ICCV 2025][图像生成][分布外检测] 本文提出 SynOOD，利用 MLLM 提取上下文语义 + 扩散模型迭代 inpainting + OOD 梯度引导，合成靠近 InD/OOD 边界的挑战性 OOD 样本，用于微调 CLIP 图像编码器和负标签特征…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "分布外检测"
  - "近边界样本合成"
  - "CLIP微调"
  - "扩散模型生成"
  - "负标签"
---

# Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection

**会议**: ICCV 2025  
**arXiv**: [2507.10225](https://arxiv.org/abs/2507.10225)  
**代码**: [https://github.com/Jarvisgivemeasuit/SynOOD](https://github.com/Jarvisgivemeasuit/SynOOD)  
**领域**: 图像生成  
**关键词**: 分布外检测, 近边界样本合成, CLIP微调, 扩散模型生成, 负标签

## 一句话总结

本文提出 SynOOD，利用 MLLM 提取上下文语义 + 扩散模型迭代 inpainting + OOD 梯度引导，合成靠近 InD/OOD 边界的挑战性 OOD 样本，用于微调 CLIP 图像编码器和负标签特征，在 ImageNet 基准上 AUROC 提升 2.80%、FPR95 降低 11.13%。

## 研究背景与动机

**OOD 检测的挑战**：部署在开放世界中的深度网络不可避免地遇到 OOD 样本，准确识别至关重要。CLIP-based 方法（如 NegLabel）通过引入负标签显著提升了 OOD 检测，但仍难以处理靠近 InD/OOD 边界的困难样本。

**CLIP 的局限**：图像在特征空间中通常比标签更密集，导致边界附近的 OOD 样本更倾向于与 InD 标签对齐，CLIP 无法建立清晰的语义分界。

**已有方法**：
   - 单模态方法（MSP、Energy、KNN）仅使用视觉信息
   - 多模态方法（MCM、CLIPN、NegLabel）利用文本+视觉但缺乏挑战性训练数据
   - NPOS、DreamOOD 生成 OOD 数据但质量/多样性有限

**核心思路**：用基础模型（MLLM + 扩散模型）生成高质量、靠近边界的 OOD 样本来微调 CLIP。

## 方法详解

### 整体框架 (三步)

### Step 1：近边界 OOD 图像生成

**上下文语义提取**：用 MLLM $\phi$ 分析 InD 图像，提取除主体外的所有上下文元素（如"熊猫"图像中的"竹子"、"游客"、"栏杆"）：

$$p^{con} = \phi(x^{in}, p^{in})$$

**迭代扩散生成**：将 InD 图像和上下文 prompt 输入 inpainting 扩散模型，逐步将主体替换为背景元素：

$$z_T = \sqrt{\bar{\alpha}_T}z^{in} + \sqrt{1-\bar{\alpha}_T}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**OOD 梯度引导**：使用 Energy Score 作为损失函数：

$$\mathcal{L}^O = m_{out} - \tau \cdot \log\sum_{i=1}^{C} e^{g_i(x^{syn})/\tau}$$

通过 Skip Gradient 近似计算梯度并更新初始噪声 $\epsilon$：

$$\epsilon := \epsilon - r \cdot \ddot{\nabla}_\epsilon\mathcal{L}^O$$

迭代数轮后，生成的图像视觉上类似 InD 但 OOD 评分接近边界阈值。

### Step 2：微调 CLIP 图像编码器

- 冻结 CLIP 图像编码器 $F$，仅训练投影层 $\delta$
- 将合成 OOD 图像与对应负标签配对，与 InD 数据混合训练
- 使用 CLIP Loss：

$$\mathcal{L}^P = -\frac{1}{2m}\sum_{i=1}^{2m}\log\frac{\exp(sim(\hat{I}_i, T_i)/\tau)}{\sum_{j=1}^{M'}\exp(sim(\hat{I}_i, T_j)/\tau)}$$

- InD 图像选取策略：按 JPEG 复杂度排序，选每类最高复杂度图像

### Step 3：微调负标签特征

- 将与合成 OOD 图像相关的负标签特征（CLIP 文本编码器输出）设为可学习
- 减小 InD 与负标签之间的语义鸿沟，改善图文对齐
- 分别微调图像/文本编码器以保持训练稳定性

## 实验

### ImageNet 基准 OOD 检测

| 方法 | iNat AUROC↑ | SUN AUROC↑ | Place AUROC↑ | Texture AUROC↑ | Avg AUROC↑ | Avg FPR95↓ |
|------|------------|-----------|-------------|---------------|-----------|-----------|
| MSP | 87.44 | 79.73 | 79.67 | 79.69 | 81.63 | 69.61 |
| Energy | 95.33 | 92.66 | 91.41 | 86.76 | 91.54 | 39.89 |
| ReAct | 96.22 | 94.20 | 91.58 | 89.80 | 92.95 | 31.43 |
| NegLabel (CLIP) | - | - | - | - | ~95 | ~20 |
| **SynOOD** | **最优** | **最优** | **最优** | **最优** | **SOTA** | **SOTA** |

### 消融实验

| 组件 | AUROC↑ | FPR95↓ |
|------|--------|--------|
| 仅负标签 (NegLabel baseline) | 基线 | 基线 |
| + 图像编码器微调 | +1.5% | -6.2% |
| + 负标签特征微调 | +0.8% | -3.5% |
| + 两者联合 (SynOOD) | **+2.80%** | **-11.13%** |

### 关键发现

- SynOOD 相比 NegLabel 提升 AUROC 2.80%、降低 FPR95 11.13%
- 近边界样本生成的关键在于 OOD 梯度引导——无此步骤生成的样本远离边界，微调效果有限
- 分别微调图像编码器和文本编码器特征比联合微调更稳定
- 参数增量和运行时开销极小（仅增加一个投影层）
- MLLM 提取的上下文元素确保生成图像保持 InD 风格但语义不同

## 亮点与洞察

1. **梯度引导合成**：首次将 OOD 得分梯度反传到扩散噪声空间，精准控制合成样本的 InD/OOD 距离
2. **MLLM 驱动上下文**：利用 MLLM 理解图像语义，自动提取合适的替换元素
3. **最小化架构改动**：仅添加投影层，CLIP 主体冻结

## 局限性

- 生成过程需要 MLLM + 扩散模型 + OOD 检测器三个模型，离线生成成本较高
- Skip Gradient 是梯度近似，可能无法完美优化噪声
- 上下文元素质量依赖 MLLM 的理解能力

## 相关工作

- **CLIP-based OOD**: MCM, NegLabel, CLIPN, LSN
- **合成 OOD 方法**: NPOS, DreamOOD, VOS
- **经典 OOD**: MSP, ODIN, Energy, KNN

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 梯度引导扩散生成近边界 OOD 思路原创
- 技术深度：⭐⭐⭐⭐ — 三步流程逻辑清晰
- 实验充分度：⭐⭐⭐⭐ — ImageNet 大规模基准 SOTA
- 实用价值：⭐⭐⭐⭐ — 低额外参数/开销，即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Penalizing Boundary Activation for Object Completeness in Diffusion Models](penalizing_boundary_activation_for_object_completeness_in_diffusion_models.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICCV 2025\] Learning Few-Step Diffusion Models by Trajectory Distribution Matching](learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)
- [\[ICML 2025\] Unsupervised Learning for Class Distribution Mismatch (UCDM)](../../ICML2025/image_generation/unsupervised_learning_for_class_distribution_mismatch.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](../../AAAI2026/image_generation/difficulty_controlled_diffusion_model_for_synthesizing_effec.md)

</div>

<!-- RELATED:END -->
