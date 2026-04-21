---
title: >-
  [论文解读] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings
description: >-
  [ICLR 2026][图像生成][成员推断攻击] 提出 MoFit，首个面向无标题场景的扩散模型成员推断攻击框架，通过构建过拟合于目标模型的代理图像和条件嵌入，利用成员样本对条件错配的不对称敏感性实现有效推断。
tags:
  - ICLR 2026
  - 图像生成
  - 成员推断攻击
  - 扩散模型
  - 无标题设定
  - 模型拟合嵌入
  - 隐私审计
---

# No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings

**会议**: ICLR 2026  
**arXiv**: [2602.22689](https://arxiv.org/abs/2602.22689)  
**代码**: [GitHub](https://github.com/JoonsungJeon/MoFit)  
**领域**: AI 安全 / 隐私攻击  
**关键词**: 成员推断攻击, 扩散模型, 无标题设定, 模型拟合嵌入, 隐私审计

## 一句话总结

提出 MoFit，首个面向无标题场景的扩散模型成员推断攻击框架，通过构建过拟合于目标模型的代理图像和条件嵌入，利用成员样本对条件错配的不对称敏感性实现有效推断。

## 研究背景与动机

- 扩散模型在高保真生成中的记忆化倾向引发隐私和知识产权担忧
- 成员推断攻击（MIA）是审计记忆化的标准方法
- **现有 MIA 的关键假设缺陷**：假设攻击者拥有 ground-truth 标题，但实际中：
    - 艺术家怀疑作品被复制时通常无法获得训练标题
    - 公开生成 AI 平台不披露训练集来源
- 用 VLM 生成的替代标题替换 ground-truth 标题后，SOTA 方法性能显著下降

## 方法详解

### 核心观察

**成员样本和非成员样本对条件错配的敏感性存在系统差异**：
- 成员样本在替代条件下 $\mathcal{L}_{\text{cond}}$ 显著增加
- 非成员样本变化较小
- $\mathcal{L}_{\text{uncond}}$ 对两组均保持稳定

### MoFit 两阶段框架

**阶段 1：模型拟合代理优化**

构建过拟合于目标模型无条件先验的代理图像 $x_0^* = x_0 + \delta^*$：

$$\delta^* = \arg\min_\delta \mathbb{E}_{z_0', t, \hat{\epsilon}} [\|\hat{\epsilon} - \epsilon_\theta(z_t', t, \phi_{\text{null}})\|^2]$$

固定 $\hat{\epsilon}$ 和 $t$ 以稳定扰动方向，沿梯度符号方向迭代更新 $\delta$。

**阶段 2：代理驱动嵌入提取**

从代理图像 $x_0^*$ 优化提取嵌入 $\phi^*$：

$$\phi^* = \arg\min_\phi \mathbb{E}_{z_0^*, t, \hat{\epsilon}} [\|\hat{\epsilon} - \epsilon_\theta(z_t^*, t, \phi)\|^2]$$

以 VLM 生成的标题嵌入为初始化。

**阶段 3：成员推断**

利用模型拟合嵌入 $\phi^*$ 条件化原始查询 $x_0$：

$$\mathcal{L}_{\text{MoFit}} = \mathbb{E}[\|\hat{\epsilon} - \epsilon_\theta(z_t, t, \phi^*)\|^2] - \mathbb{E}[\|\hat{\epsilon} - \epsilon_\theta(z_t, t, \phi_{\text{null}})\|^2]$$

最终决策融合 MoFit 分数和辅助损失（$\mathcal{L}_{\text{uncond}}$ 或 $\mathcal{L}_{\text{VLM}}$）。

## 实验关键数据

### 无标题设定下的 MIA 性能对比

| 方法 | 条件 | Pokemon ASR | Pokemon TPR@1%FPR | MS-COCO ASR | MS-COCO TPR@1%FPR |
|------|------|------------|-------------------|-------------|-------------------|
| CLiD | GT | 96.52 | 90.14 | 86.50 | 68.80 |
| CLiD | VLM | 77.55 | 19.23 | 80.90 | 50.80 |
| PFAMI | VLM | 74.43 | 6.01 | 80.40 | 29.40 |
| SecMI | VLM | 78.51 | 6.97 | 57.30 | 4.20 |
| **MoFit** | **$\phi^*$** | **94.48** | **50.48** | **88.00** | **47.00** |

### 消融实验：代理图像变体

| 输入 | 条件 | Pokemon ASR | MS-COCO ASR | MS-COCO TPR@1%FPR |
|------|------|------------|-------------|-------------------|
| $x_0$（原始） | $\phi$ | 75.63 | 78.00 | 31.00 |
| $x_0 + \delta$（随机噪声） | $\phi$ | 93.99 | 81.70 | 29.20 |
| $x_0 + \delta_{\text{MAX}}$（反向优化） | $\phi$ | 75.87 | 78.00 | 34.00 |
| **MoFit** ($x_0 + \delta^*$) | **$\phi^*$** | **94.48** | **88.00** | **47.00** |

### 关键发现

1. MoFit 在无标题设定下大幅超越 VLM 条件化基线（ASR 提升最高 +25%，TPR@1%FPR 提升 +30-47%）
2. 在 MS-COCO 上甚至超越使用 ground-truth 标题的 CLiD（ASR: 88.00 vs 86.50）
3. 代理优化是关键：仅使用原始图像或随机噪声优化嵌入效果显著较差
4. 在 SD v1.5 预训练模型上同样有效（ASR: 77.61），说明方法具有通用性

## 亮点与洞察

1. **问题定义的实际意义**：无标题 MIA 场景更贴近现实审计需求
2. **理论洞察深刻**：成员样本对条件错配的不对称敏感性提供了可利用的新信号
3. **巧妙的两阶段设计**：先构建过拟合代理再提取嵌入，形成紧密耦合的模型拟合对
4. **无需额外数据或模型**：仅需访问目标模型的推理接口

## 局限性

- 需要访问目标模型的去噪网络参数（灰盒假设）
- 代理优化和嵌入提取增加了计算开销
- 固定时间步 $t=140$ 为超参数，可能需要针对不同模型调整
- 对 LAION 规模的预训练模型效果相对减弱（该场景所有方法都表现不佳）

## 相关工作

- **扩散模型 MIA**：SecMI, PIA, PFAMI, CLiD
- **LLM MIA**：Shokri et al. (2017)
- **无标题生成**：Classifier-free guidance (Ho & Salimans, 2022)

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首个针对无标题场景的扩散模型 MIA 框架
- 技术深度：⭐⭐⭐⭐ — 核心观察深刻，两阶段优化设计合理
- 实验完整性：⭐⭐⭐⭐ — 多数据集、多模型、充分消融
- 实用价值：⭐⭐⭐⭐ — 为数据隐私审计提供了实用工具

<!-- RELATED:START -->

## 相关论文

- [DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)
- [HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation](hierloc_hyperbolic_entity_embeddings_for_hierarchical_visual_geolocation.md)
- [RNE: plug-and-play diffusion inference-time control and energy-based training](rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)
- [VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)

<!-- RELATED:END -->
