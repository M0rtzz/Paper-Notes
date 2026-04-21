---
title: >-
  [论文解读] SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation
description: >-
  [ICLR 2026][图像生成][VAR] 提出 Scaled Spatial Guidance (SSG)，一种无需训练的推理时引导方法，通过频域先验构建和语义残差放大，增强视觉自回归模型的粗到细层级生成质量。
tags:
  - ICLR 2026
  - 图像生成
  - VAR
  - 下一尺度预测
  - 信息瓶颈
  - 频域引导
  - 训练免费
---

# SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation

**会议**: ICLR 2026  
**arXiv**: [2602.05534](https://arxiv.org/abs/2602.05534)  
**代码**: [GitHub](https://github.com/Youngwoo-git/SSG)  
**领域**: 视觉自回归模型 / 图像生成 / 推理时引导  
**关键词**: VAR, 下一尺度预测, 信息瓶颈, 频域引导, 训练免费

## 一句话总结

提出 Scaled Spatial Guidance (SSG)，一种无需训练的推理时引导方法，通过频域先验构建和语义残差放大，增强视觉自回归模型的粗到细层级生成质量。

## 研究背景与动机

视觉自回归（VAR）模型通过下一尺度预测（next-scale prediction）生成图像，天然实现粗到细的层级合成。然而：

**训练-推理偏差**：有限的模型容量和累积误差导致模型在推理时偏离粗到细的本质，低频信息被冗余预测

**现有改进方法的限制**：
   - 辅助精炼模块（CoDe、HMAR）需要重新训练
   - 流匹配集成增加开销
   - 自校正机制需要修改架构

**核心问题**：如何在不修改模型参数的情况下，引导每一步生成该尺度特有的新颖高频信息？

## 方法详解

### 1. 信息论视角的推导

从信息瓶颈（IB）原理出发，将 VAR 的逐步生成转化为变分优化问题：

$$\mathcal{L}_{\text{VAR-IB}} = \max_{z_k} \beta I(z_k; H(\hat{f}_K)) - I(z_k; L(\hat{f}_K))$$

- **目标信息量项**：最大化与高频细节的互信息
- **状态冗余项**：最小化与已建立粗结构的冗余

### 2. SSG 公式

将优化目标转化为 MAP 风格的代理函数，得到闭式解：

$$\ell_k^{\text{SSG}} = \ell_k + \beta_k \Delta_k = \ell_k + \beta_k (\ell_k - \ell_{\text{prior}})$$

其中：
- $\ell_k$：第 $k$ 步的残差 logits
- $\ell_{\text{prior}}$：从前一步构建的粗粒度先验
- $\Delta_k = \ell_k - \ell_{\text{prior}}$：语义残差（高频细节）
- $\beta_k$：逐步缩放因子

### 3. 离散空间增强（DSE）

频域先验构建方法：
1. 对前一步 logits $\ell_{k-1}$ 进行空间插值得到 $\ell'_{\text{interp}}$
2. 对两者分别进行 DCT 变换
3. 将 $\ell_{k-1}$ 的低频系数与 $\ell'_{\text{interp}}$ 的高频系数融合
4. IDCT 逆变换得到先验 $\ell_{\text{prior}}$

**相比简单插值的优势**：
- 线性插值过度平滑、衰减先验
- 最近邻插值引入块状不连续和伪高频
- DCT 频域融合保持能量守恒，精确分离频段

### 4. 高效实现

- 无需额外前向传递（使用缓存的 logits）
- 仅几行代码实现
- 计算和显存开销可忽略

## 实验

### ImageNet 256×256 类条件生成

| 模型 | FID↓ | sFID↓ | IS↑ | Pre↑ | Rec↑ |
|------|------|-------|-----|------|------|
| VAR-d16 | 3.42 | 8.70 | 275.6 | 0.84 | 0.51 |
| +SSG | **3.27** | **8.39** | **285.3** | **0.85** | 0.50 |
| VAR-d20 | 2.67 | 7.97 | 299.8 | 0.83 | 0.55 |
| +SSG | **2.49** | **7.60** | **305.2** | 0.83 | **0.56** |
| VAR-d24 | 2.39 | 8.18 | 314.7 | 0.82 | 0.58 |
| +SSG | **2.20** | **6.95** | **324.0** | **0.83** | **0.59** |
| VAR-d30 | 2.02 | 8.52 | 302.9 | 0.82 | 0.60 |
| +SSG | **1.68** | **8.50** | **313.2** | 0.81 | **0.62** |

### 跨模型泛化

SSG 在不同 tokenization 方案上均有效：
- 标准 VAR (Tian et al.)
- HART（混合 token）
- Infinity（bitwise token）

### 与其他生成模型对比

VAR-d30 + SSG (FID 1.68) 与扩散模型和掩码模型具有竞争力，同时保持 VAR 的低延迟优势（10步推理）。

### 消融实验

| 组件 | FID | IS |
|------|-----|-----|
| 无 SSG（基线） | 2.02 | 302.9 |
| SSG + 线性插值先验 | 改善有限 | — |
| SSG + 最近邻先验 | 可能恶化 | — |
| SSG + DSE（频域融合） | **1.68** | **313.2** |

## 亮点

1. **信息论驱动的优雅设计**：从 IB 原理严格推导出 SSG 的闭式解
2. **完全训练免费**：无需修改模型权重、无需额外数据、无需微调
3. **频域先验构建（DSE）理论合理**：利用 DCT 正交性实现能量无损的频段融合
4. **一致性强**：在不同 VAR 模型尺度和 tokenization 设计上均有效
5. **实现极简**：几行代码即可集成

## 局限性

1. SSG 的效果依赖于合理的 $\beta_k$ 调度，需要根据模型调参
2. 在第一步（最粗尺度）无先验可用，SSG 不生效
3. 本质上是后验修正，无法弥补 tokenizer 本身的信息损失
4. 仅适用于离散视觉 token 的 VAR 模型

## 相关工作

- **VAR模型**：VAR (Tian 2024)、HART (Tang 2025)、Infinity (Han 2025)
- **视觉引导**：CFG、SAG、PAG、STG，但均非针对 VAR 设计
- **训练-推理偏差缓解**：CoDe、HMAR，但需要重训

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 信息论到实践的优雅桥接
- **实用性**: ⭐⭐⭐⭐⭐ — 零成本集成，即插即用
- **实验**: ⭐⭐⭐⭐ — 多模型多设置验证
- **写作**: ⭐⭐⭐⭐⭐ — 理论推导清晰，直觉解释充分

<!-- RELATED:START -->

## 相关论文

- [Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [Visual Autoregressive Modeling for Instruction-Guided Image Editing](visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [Randomized Autoregressive Visual Generation](../../ICCV2025/image_generation/randomized_autoregressive_visual_generation.md)
- [Autoregressive Image Generation with Randomized Parallel Decoding](autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [Visual Generation Without Guidance](../../ICML2025/image_generation/visual_generation_without_guidance.md)

<!-- RELATED:END -->
