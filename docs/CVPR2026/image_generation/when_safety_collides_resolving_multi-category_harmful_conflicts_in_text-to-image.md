---
title: >-
  [论文解读] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance
description: >-
  [图像生成] 提出 Conflict-aware Adaptive Safety Guidance (CASG)，一种无训练的即插即用框架，通过动态识别与当前生成状态最对齐的有害类别并仅沿该方向施加安全引导，解决了现有安全引导方法在多类别聚合时因方向冲突导致的安全性退化问题。
tags:
  - 图像生成
---

# When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance

## 基本信息

- **会议**: CVPR 2026
- **arXiv**: [2602.20880](https://arxiv.org/abs/2602.20880)
- **代码**: [GitHub](https://github.com/tmllab/2026_CVPR_CASG)
- **领域**: 图像生成 / T2I安全
- **关键词**: 扩散模型安全, 有害内容缓解, 安全引导, 多类别冲突, 无训练框架

## 一句话总结

提出 Conflict-aware Adaptive Safety Guidance (CASG)，一种无训练的即插即用框架，通过动态识别与当前生成状态最对齐的有害类别并仅沿该方向施加安全引导，解决了现有安全引导方法在多类别聚合时因方向冲突导致的安全性退化问题。

## 研究背景与动机

文本到图像（T2I）扩散模型（如 Stable Diffusion、Hunyuan-DiT）在生成高质量图像方面取得了巨大进展，但也带来了生成有害内容（仇恨、色情、暴力、违法等）的安全风险。现有的安全引导方法（如 SLD、SAFREE）通过在潜在空间或文本空间中施加安全方向来引导生成远离有害区域，是一类无需修改模型的轻量化安全方案。

**核心问题**：现有安全引导方法在处理多类别有害内容时，简单地将所有有害关键词拼接为一个聚合集，从中导出一个统一的安全方向。这种类别无关的设计隐含假设不同类型的害处是兼容的。然而作者通过大量实验发现，这一假设不成立——不同有害类别定义了各自独特且往往不兼容的安全方向，强行聚合会产生 **有害冲突（Harmful Conflict）**，反而降低安全性能。

有害冲突体现为两种形式：

**方向不一致（Directional Inconsistency）** → 安全错位退化（Safety Misalignment Degradation）：不同类别的安全方向指向不兼容甚至相反的方向。例如对色情提示词施加"仇恨"方向的安全引导，有害率反而从 67.2% 升至 72.4%，远高于正确使用"色情"方向的 3.2%。

**方向衰减（Directional Attenuation）** → 安全平均退化（Safety Averaging Degradation）：多类别聚合导致异质方向部分抵消，削弱净安全信号。"色情"方向单独应用有害率为 3.2%，加入"仇恨"后升至 5.8%，聚合全部类别后升至 48.8%。

## 方法详解

### 整体框架

CASG 是一个即插即用的安全校正框架，由两个核心组件构成：

- **CaCI（Conflict-aware Category Identification）**：动态识别与当前生成状态最对齐的有害类别
- **CrGA（Conflict-resolving Guidance Application）**：仅沿识别出的主导类别方向施加安全校正

CASG 可同时应用于潜在空间安全机制（CASG+SLD）和文本空间安全机制（CASG+SAFREE），形成两种实例化方案。

### 潜在空间方法：Conflict-aware Safety Steering（CASG+SLD）

**STEP 1: 有害引导生成**。对预定义有害关键词集 $\mathcal{H}$ 中的每个类别 $h_i$，计算其在潜在空间中的有害引导。首先基于 classifier-free guidance 原理，生成有害条件下的噪声预测：

$$\hat{\epsilon}_i = \epsilon_\theta(z_t, c_{h_i})$$

然后减去无条件噪声预测得到有害引导方向：

$$g_i = \hat{\epsilon}_i - \epsilon_\theta(z_t)$$

最终获得有害引导集合 $G = \{g_1, \ldots, g_k\}$。

**STEP 2: 对齐类别识别（CaCI）**。类似地计算提示词引导：

$$g_p = \epsilon_\theta(z_t, c_p) - \epsilon_\theta(z_t)$$

通过余弦相似度衡量每个有害引导 $g_i$ 与提示词引导 $g_p$ 的对齐程度：

$$\cos\theta_i = \frac{g_i \cdot g_p}{\|g_i\| \|g_p\|}$$

余弦相似度越大表示该有害类别与当前生成轨迹的对齐度越高。选择最大余弦相似度对应的类别作为主导有害方向：

$$h^* = h_{\arg\max_i \cos\theta_i}$$

**STEP 3: 对齐类别应用（CrGA）**。仅沿主导有害类别 $h^*$ 的方向应用 SLD 安全校正，替代原始的多类别聚合方向。其余所有 SLD 机制和超参数保持不变。

### 文本空间方法：Conflict-aware Orthogonal Projection（CASG+SAFREE）

**STEP 1: 投影残差计算**。对每个有害类别 $h_i$，SAFREE 使用子空间投影矩阵 $P_{h_i}$ 表示其有害概念。计算提示词嵌入在有害子空间正交补上的投影：

$$p_{h_i}^\perp = (I - P_{h_i}) p$$

其中 $p_{h_i}^\perp$ 表示移除与有害类别 $h_i$ 对齐成分后的残差提示词嵌入。

**STEP 2: 对齐类别识别（CaCI）**。残差的范数 $\|p_{h_i}^\perp\|$ 越小，表示提示词与该有害类别的对齐度越高。选择最小残差范数对应的类别：

$$h^* = h_{\arg\min_i \|p_{h_i}^\perp\|}$$

**STEP 3: 对齐类别应用（CrGA）**。仅使用主导有害类别 $h^*$ 对应的子空间 $P_{h^*}$ 进行正交投影，避免跨类别干扰。

### 关键设计亮点

1. **无训练、即插即用**：不需要额外训练或外部 API 调用，直接嵌入现有安全机制
2. **动态逐步适应**：在去噪过程的每个时间步动态更新主导类别识别，捕捉生成过程中有害语义的动态演变
3. **基于生成状态而非文本分类**：区别于基于 LLM 的预分类方案（如 GPT+SLD），CASG 基于实际生成轨迹进行判断，更准确地追踪有害冲突的动态变化

## 实验

### 主实验结果

| 方法 | Conflict-Aware | I2P ↓ | T2VSafetyBench ↓ | Unsafe Diffusion ↓ | CoProv2 ↓ | FID ↓ | CLIP ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SD-v1.5 | - | 42.2 | 58.3 | 52.3 | 28.2 | - | 31.43 |
| ESD | ✗ | 42.0 | 57.4 | 50.6 | 28.1 | 38.15 | 31.35 |
| UCE | ✗ | 26.7 | 28.2 | 33.0 | 19.4 | 77.41 | 29.12 |
| RECE | ✗ | 21.5 | 18.9 | 22.3 | 8.6 | 67.35 | 27.67 |
| SafetyDPO | ✗ | 13.7 | 24.0 | 16.7 | 4.2 | 49.64 | 30.61 |
| SAFREE | ✗ | 20.0 | 41.5 | 24.2 | 14.2 | 43.78 | 30.53 |
| **CASG+SAFREE** | ✓ | **18.9** | **37.5** | **17.5** | **11.8** | 46.25 | 30.35 |
| SLD | ✗ | 12.7 | 25.2 | 15.7 | 7.1 | 52.11 | 29.22 |
| **CASG+SLD** | ✓ | **10.2** | **9.8** | **9.8** | **3.9** | 52.00 | 29.36 |

CASG+SLD 在所有四个基准上达到 SOTA，有害率最高降低 15.4%（T2VSafetyBench 上从 25.2% 降至 9.8%）。同时 FID 和 CLIP 分数几乎不变，说明不损害生成质量。

### LLM辅助 vs CASG 消融对比

| 方法 | I2P ↓ | T2VSafetyBench ↓ | UD ↓ |
|:---|:---:|:---:|:---:|
| SLD | 12.7 | 25.2 | 15.7 |
| GPT-4o+SLD | 11.6 | 12.3 | 20.1 |
| QwenGuard+SLD | 14.0 | 21.1 | 23.3 |
| **CASG+SLD** | **10.2** | **9.8** | **9.8** |

LLM 辅助方案（GPT-4o、QwenGuard 预分类后接 SLD）效果有限甚至更差，原因有二：(1) LLM 对混合/模糊提示词分类出错；(2) 类别在生成开始时固定，无法适应去噪过程中动态演变的冲突。CASG 在每个时间步动态更新，大幅优于 LLM 辅助方案。

### 关键发现

1. **多类别聚合不等于更安全**：这是本文最核心的发现。聚合更多有害类别可能反而削弱安全性（如全类别聚合在色情提示上有害率 48.8%，远高于单类别的 3.2%）
2. **冲突是系统性的**：跨不同基础模型、安全机制、有害关键词定义均存在一致的退化模式
3. **动态识别优于静态分类**：基于生成轨迹的逐步识别比基于 LLM 的一次性文本分类更有效
4. **即插即用提升显著**：CASG 对 SLD 和 SAFREE 均带来一致提升，且不引入额外训练成本

## 亮点

- 提出了一个被广泛忽视但影响深远的问题——多类别有害冲突，首次系统地揭示安全引导方向间的不一致性
- 方法设计极其优雅简洁：仅通过余弦相似度/残差范数选择主导类别，无需训练、无需外部模型
- 通用性强：同时适用于潜在空间（SLD）和文本空间（SAFREE）两类安全机制
- 实验严谨全面：四个安全基准 + 良性生成质量评估 + LLM 辅助对比 + 多模型验证

## 局限性

- 每个时间步需对所有有害类别分别计算噪声预测/投影残差，计算开销随类别数线性增长
- 仅在 SD v1.5 上验证，对更新架构（如 DiT、FLUX 系列）的适用性未充分验证
- 依赖预定义有害关键词集，若关键词覆盖不全或定义不当，可能遗漏某些有害类型
- 每步仅选择一个主导类别，对同时包含多种有害语义的提示词可能存在遗漏

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐⭐ — 「有害冲突」问题的发现极有价值，挑战了"聚合更多类别=更安全"的领域直觉
- **技术深度**: ⭐⭐⭐⭐ — 对冲突机制的分析（方向不一致 + 方向衰减）系统深入，CDRR 分析有说服力
- **实验**: ⭐⭐⭐⭐⭐ — 四个基准、多基线、LLM对比、消融分析，非常全面
- **写作**: ⭐⭐⭐⭐⭐ — 问题引入清晰，PCA方向可视化和CDRR热力图极具说服力
- **影响力**: ⭐⭐⭐⭐ — T2I 安全领域的即插即用增强方案，实用价值高但应用场景相对垂直

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](../../ICCV2025/image_generation/dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](../../ICCV2025/image_generation/mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)

<!-- RELATED:END -->
