---
title: >-
  [论文解读] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens
description: >-
  [ICML2025][图像生成][ResGen] ResGen 通过直接预测累积RVQ嵌入而非单个令牌，解耦了生成迭代次数与序列长度和量化深度的关系，实现了高保真、快速采样的高效生成模型。 核心矛盾 核心矛盾：残差向量量化(RVQ)通过迭代量化残差提高重建保真度，但在自回归生成中： - 采样步数 = 序列长度L × 量化深度…
tags:
  - "ICML2025"
  - "图像生成"
  - "ResGen"
  - "残差向量量化"
  - "掩码预测"
  - "离散扩散"
  - "高效生成"
---

# Efficient Generative Modeling with Residual Vector Quantization-Based Tokens

**会议**: ICML2025  
**arXiv**: [2412.10208](https://arxiv.org/abs/2412.10208)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: ResGen, 残差向量量化, 掩码预测, 离散扩散, 高效生成

## 一句话总结
ResGen 通过直接预测累积RVQ嵌入而非单个令牌，解耦了生成迭代次数与序列长度和量化深度的关系，实现了高保真、快速采样的高效生成模型。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：残差向量量化(RVQ)通过迭代量化残差提高重建保真度，但在自回归生成中：
- 采样步数 = 序列长度L × 量化深度D
- L和D增大时推理成本线性增长
- 现有非自回归方法只能在L或D一个维度上改进，无法同时解耦

### 现有痛点

**现有痛点**：需要一种方法：
- 保留RVQ带来的高保真重建优势
- 推理步数不随D增大而增长
- 统一的概率框架处理RVQ层次结构

## 方法详解

### 整体框架：粗到细的迭代填充
ResGen采用"从最高量化层开始逐步填充"的策略：

### 关键设计1：分层掩码策略
- 从最高深度D（细节层）开始逐步掩码
- 掩码调度函数：$n = \lceil \gamma(r) \cdot L \cdot D \rceil$
- 多项分布采样将掩码令牌均匀分布到L个位置
- 从深度D向低深度依次掩码

### 关键设计2：多令牌预测（核心创新）
训练时不预测单个令牌 $x$，而是预测掩码嵌入的累积和：

$$z_i = \sum_j e(x_{i,j}; j) \odot (1 - m_{i,j})$$

通过预测聚合向量，模型捕捉跨深度的令牌关联，**采样步数与RVQ深度D解耦**。

### 关键设计3：概率框架
- 使用离散扩散过程和变分推断形式化
- 基于模型置信度的增强采样策略
- 高斯混合模型估计潜在嵌入

### 采样步数对比

| 方法 | 采样步数 |
|------|---------|
| 自回归 | $O(L \times D)$ |
| 非自回归(现有) | $O(L)$ 或 $O(D)$ |
| ResGen | $O(\text{fixed})$（与L和D均无关） |

## 实验关键数据

### ImageNet 256×256 条件图像生成


### 主实验

| 方法 | 参数量 | 采样步数 | FID | 相对性能 |
|------|--------|---------|-----|---------|
| AR Baseline | 1.5B | 256 | 4.2 | 基准 |
| ResGen (D=4) | 1.5B | 8 | 3.8 | FID↓ + 3.2×加速 |
| ResGen (D=8) | 1.5B | 8 | 3.2 | FID↓↓ + 5.1×加速 |

### 零样本文本转语音合成


### 消融实验

| 方法 | MOS评分 | 采样步数 | 实时因子 |
|------|--------|---------|---------|
| AR基线 | 3.8 | 512 | 1.2× |
| ResGen | 4.1 | 16 | 4.5× |
| ResGen+深度扩展 | 4.2 | 16 | 4.3× |

### 关键发现
1. 少8倍采样步数实现更优FID。
2. 随RVQ深度增加(D:4→8)保真度显著提升。
3. 跨模态(图像+音频)通用有效。
4. MOS提升7.9%且推理速度提升3.75倍。

## 亮点与洞察

1. "预测累积嵌入而非单个令牌" — 简单但突破性的设计选择，一举解耦深度与推理成本。
2. 分层掩码（高层优先）契合信息论直觉：粗糙信息优先。
3. 完整的离散扩散+变分推断概率框架，理论严谨。
4. 两个模态(图像+语音)验证了方法普适性。

## 局限与展望

1. 代码开源状态未确认，复现可能有门槛。
2. 不同RVQ深度(D=4 vs D=8)的最优选择标准不清晰。
3. 聚合嵌入预测的计算成本对比单令牌预测的详细分析不足。
4. 仅在图像和音频验证，文本和视频场景未涉及。
5. 掩码调度函数 $\gamma(\cdot)$ 对任务敏感。

## 相关工作与启发

- **VQ-VAE/VQ-GAN**：ResGen基于其量化框架但用RVQ改进层次结构。
- **Masked Token Modeling**：沿用掩码预测框架但创新了预测目标。
- 启发：
  1. 分层掩码可推广到其他层次化结构（树形、图形令牌）
  2. 多目标预测（聚合而非单体）的设计模式值得跨任务探索
  3. 可联合优化RVQ编码器和生成器

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（4.5/5）— 累积嵌入预测解耦深度与推理是核心创新
- 实验充分度: ⭐⭐⭐⭐☆（4.0/5）— 两个模态验证但消融可更深入
- 写作质量: ⭐⭐⭐⭐☆（4.0/5）
- 价值: ⭐⭐⭐⭐⭐（4.5/5）— 对高效生成模型有实质贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization](modulated_diffusion_accelerating_generative_modeling_with_modulated_quantization.md)
- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[ICML 2025\] Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional](action-minimization_meets_generative_modeling_efficient_transition_path_sampling.md)
- [\[ICML 2025\] Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)
- [\[ICML 2025\] DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)

</div>

<!-- RELATED:END -->
