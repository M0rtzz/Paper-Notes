---
title: >-
  [论文解读] Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation
description: >-
  [ICCV 2025][LLM效率][素描生成] 提出 Stroke2Sketch，一个无训练的参考式素描生成框架，通过跨图像笔触注意力（CSA）、指导性注意力模块（DAM）和语义保持模块（SPM）三个模块协同工作，在预训练扩散模型中实现精细的笔触属性迁移与内容结构保持。
tags:
  - ICCV 2025
  - LLM效率
  - 素描生成
  - 笔触属性迁移
  - 无训练
  - 扩散模型
  - 跨图像注意力
---

# Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation

**会议**: ICCV 2025  
**arXiv**: [2510.16319](https://arxiv.org/abs/2510.16319)  
**代码**: [https://github.com/rane7/Stroke2Sketch](https://github.com/rane7/Stroke2Sketch)  
**领域**: 素描生成 / 风格迁移  
**关键词**: 素描生成, 笔触属性迁移, 无训练, 扩散模型, 跨图像注意力

## 一句话总结

提出 Stroke2Sketch，一个无训练的参考式素描生成框架，通过跨图像笔触注意力（CSA）、指导性注意力模块（DAM）和语义保持模块（SPM）三个模块协同工作，在预训练扩散模型中实现精细的笔触属性迁移与内容结构保持。

## 研究背景与动机

参考式素描生成的目标是：给定一张内容图像和一张参考素描，生成一幅保持内容结构但采用参考笔触风格（线条粗细、曲率、纹理疏密等）的素描。这一任务面临三个基本挑战：

**语义感知的笔触迁移**：需要将参考笔触属性精确映射到语义对应的内容区域，而非简单的全局风格混合

**前景优先**：人类画家自然地用丰富的笔触强调前景、简化背景，但现有方法对所有区域施加均匀风格化

**内容-风格平衡**：素描通过线条编码内容，内容泄露（content leakage）会破坏关键边缘结构

**现有方法的不足**：
- **训练式方法**（Ref2Sketch、Semi-Ref2Sketch）：对未见风格泛化失败（灾难性遗忘）
- **基于 IP-Adapter/InstantStyle**：擅长纹理迁移但结构完整性差（跨注意力中的内容泄露）
- **ControlNet 增强**：结构保持过于刚性，牺牲风格灵活性
- **渐进式笔触方法**（RB-Modulation）：均匀笔触导致语义不一致

**核心 idea**：笔触属性（线条粗细、曲率、纹理疏密）本质上编码在预训练扩散模型的自注意力和交叉注意力关系中。通过动态对齐内容与参考特征之间的注意力模式，可以在不破坏结构的情况下实现风格迁移。

## 方法详解

### 整体框架

输入：内容图像 $I^{cnt}$ 和参考素描 $I^{ref}$。通过 DDPM 反演获取三者的潜在表示：内容、参考和轮廓（由 TEED 边缘检测器提取）。在去噪过程中通过三个模块协同生成风格化素描。

### 关键设计

1. **跨图像笔触注意力（CSA）**：在扩散模型的自注意力层中进行 Key-Value 交换。将参考素描的 K/V 特征与内容图像的 K/V 特征混合后注入生成过程：

    $K^{ske}_t = K^{ref}_t + \alpha K^{cnt}_t, \quad V^{ske}_t = V^{ref}_t + \alpha V^{cnt}_t$

   其中 $\alpha$ 控制参考与内容的混合比例。这种方式不同于直接的特征混合（如 InstantStyle），而是通过注意力机制让笔触特征自然地映射到语义对应区域。但直接 K-V 交换可能扭曲某些结构元素（如曲线），因此需要后续模块配合。

2. **指导性注意力模块（DAM）**：解决前景/背景不均匀风格化问题。具体流程：

    - 提取 32×32 分辨率的自注意力特征图 $F_{SA}$，通过通道平均聚合
    - 使用 KMeans 聚类获得分割掩码 $M_j$
    - 利用 BLIP 提取的名词的交叉注意力图 $A_n$ 计算每个聚类与前景的相关性分数：$r(j,n) = \frac{\sum M_j \cdot A_n}{\sum M_j + \delta}$
    - 相关性 > 0.35 的聚类标记为前景，抑制背景区域的风格迁移

3. **语义保持模块（SPM）**：解决参考素描与内容图像语义不匹配时的噪声和错位问题。双重指导：

    - **文本引导**：通过 CLIP 损失 $L_{sem} = \lambda \cdot \text{CLIP}(I^{ske}, T^{cnt})$ 保持高层语义
    - **轮廓引导**：将 DDPM 反演过程中缓存的轮廓查询特征注入生成查询：$Q^{ske}_{i+1} = \gamma Q^{cont}_i + (1-\gamma) Q^{ske}_i$（默认 $\gamma=0.25$），轮廓作为"软约束"而非 ControlNet 式的刚性约束

4. **笔触细节传播增强（SDPE）**：通过自适应对比度增强 $\text{Enhance}(A) = (A - \mu(A))\zeta(\sigma(A)) + \mu(A)$ 抑制低对比度噪声。并行双通道 CFG：一路使用跨图像注意力捕捉笔触特征，一路使用文本引导保持语义，最终 noise 预测为：

    $\epsilon^t = \epsilon^{self} + \beta_{sg}(\epsilon^{\times}_{stroke} - \epsilon^{self}) + \beta_{text}(\epsilon^{\times}_{text} - \epsilon^{self})$

### 训练策略

完全无训练——基于预训练 Stable Diffusion v2.1-base，使用 DDPM 反演进行图像反转，DDIM 50 步去噪。所有模块通过操作注意力层实现，不修改网络参数。

## 实验关键数据

### 主实验（Stroke2Sketch-dataset）

| 方法 | ArtFID ↓ | LPIPS ↓ | FID ↓ |
|------|---------|---------|-------|
| Ref2sketch | 45.292 | 0.6982 | 34.650 |
| Semi-ref | 33.242 | 0.5306 | 24.359 |
| IP-Adapter | 33.457 | 0.6634 | 24.068 |
| InstantStyle | 32.532 | 0.5432 | 23.940 |
| InstantStyle+ | 37.656 | 0.6532 | 26.632 |
| StyleID | 35.727 | 0.5426 | 25.658 |
| **Ours** | **32.455** | **0.5315** | **22.435** |

### 消融实验

| 配置 | ArtFID ↓ | FID ↓ | LPIPS ↓ |
|------|---------|-------|---------|
| A: Full（Ours）| **32.45** | **22.43** | **0.530** |
| B: - DAM | 38.67 | 26.53 | 0.672 |
| C: - SPM | 36.89 | 30.47 | 0.637 |
| D: - SDPE | 40.53 | 32.44 | 0.598 |

去除 SDPE 的退化最严重（ArtFID 从 32.45 涨到 40.53），说明细节传播增强对最终质量至关重要。

### 关键发现

- **用户研究**（2000 票 / 100 用户）：Stroke2Sketch 在内容提取、笔触风格化和整体偏好三个维度均获得最高偏好
- 在 FS2K 人脸素描数据集上，方法也取得最低 FID（128.84 vs 次优 185.26）和 LPIPS（0.4057 vs 0.4540）
- 支持彩色素描生成（Fig. 9），保持参考笔触特征和艺术风格
- 超参数 $\gamma$（轮廓权重）、$\beta_{sg}$（笔触引导尺度）、$\zeta$（对比度强度）提供了灵活的用户控制接口

## 亮点与洞察

- **问题定义精准**：明确区分"笔触属性迁移"与一般的"风格迁移"，前者需要更细粒度的语义对应
- **三模块设计互补**：CSA 负责笔触注入、DAM 负责区域选择、SPM 负责结构约束，解耦清晰
- 利用扩散模型自注意力的聚类来实现无监督前景分割（无需额外分割模型），设计巧妙
- 完全无训练：不需要素描数据集，不需要微调，任何参考风格即可适用

## 局限与展望

- 对过于简约（如单线条连续画）或过于复杂（密集细笔触）的参考素描处理效果不佳
- 无法完全解耦语义信息和笔触属性——某些情况下语义泄露仍然存在
- 依赖 BLIP 的文本提取质量和 TEED 的边缘检测质量
- 多超参数需要根据风格类型手动调整（$\gamma$, $\beta_{sg}$, $\zeta$）

## 相关工作与启发

- **Cross-Image Attention（CIA）**和 **StyleAligned** 证明了自注意力中编码了关键的风格信息
- **InstantStyle** 的 CLIP 空间风格减法思路虽然简单但在素描场景中内容泄露严重
- 方法思路可能扩展到其他艺术风格（如水墨画、油画）的参考式生成

## 评分

- 新颖性：⭐⭐⭐⭐ — 跨图像笔触注意力机制 + 无监督前景聚焦是新设计
- 理论深度：⭐⭐⭐ — 以工程设计为主，缺乏形式化理论分析
- 实验充分度：⭐⭐⭐⭐ — 多基线对比、消融完整、用户研究有说服力
- 实用性：⭐⭐⭐⭐ — 无训练即用，但超参数调整有一定门槛

<!-- RELATED:START -->

## 相关论文

- [Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)
- [GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](../../ACL2025/llm_efficiency/gradot_offsite_tuning.md)
- [SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](../../CVPR2026/llm_efficiency/sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)
- [Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_efficiency/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [Retraining-Free Merging of Sparse MoE via Hierarchical Clustering](../../ICML2025/llm_efficiency/retraining-free_merging_of_sparse_moe_via_hierarchical_clustering.md)

<!-- RELATED:END -->
