---
title: >-
  [论文解读] Pyramidal Patchification Flow for Visual Generation
description: >-
  [ICLR 2026][图像生成][金字塔 patch 化] 提出 Pyramidal Patchification Flow (PPFlow)，通过在高噪声时间步使用大 patch、低噪声时使用小 patch，在保持生成质量的同时实现 1.6-2.0× 去噪加速，且无需重噪声技巧。
tags:
  - ICLR 2026
  - 图像生成
  - 金字塔 patch 化
  - 流匹配
  - DiT
  - 推理加速
  - 可变 token 数量
---

# Pyramidal Patchification Flow for Visual Generation

**会议**: ICLR 2026  
**arXiv**: [2506.23543](https://arxiv.org/abs/2506.23543)  
**代码**: [GitHub](https://github.com/fudan-generative-vision/PPFlow)  
**领域**: 扩散模型加速 / 图像生成  
**关键词**: 金字塔 patch 化, 流匹配, DiT, 推理加速, 可变 token 数量

## 一句话总结

提出 Pyramidal Patchification Flow (PPFlow)，通过在高噪声时间步使用大 patch、低噪声时使用小 patch，在保持生成质量的同时实现 1.6-2.0× 去噪加速，且无需重噪声技巧。

## 研究背景与动机

- DiT 在全部时间步使用相同 patch 大小（通常 2×2），导致高噪声时间步浪费计算
- 现有加速方法的局限：
    - **减少步数**（DDIM、蒸馏）：牺牲质量
    - **降低单步成本**（量化、剪枝）：可能性有限
    - **金字塔/级联生成**（Pyramidal Flow）：引入"跳跃点"，需要复杂的重噪声技巧
- **核心观察**：高噪声时空间细节不重要，可用更少 token 表示

## 方法详解

### 金字塔 Patch 化方案

将时间步分为多个阶段，每阶段使用不同 patch 大小：

**三级示例**：
- $[0, t_{s_1})$：patch 大小 $4 \times 4$（高噪声，$L = (I/4)^2$ 个 token）
- $[t_{s_1}, t_{s_2})$：patch 大小 $4 \times 2$
- $[t_{s_2}, 1]$：patch 大小 $2 \times 2$（低噪声，正常 DiT）

### 参数共享策略

- **Patchify/Unpatchify**：每阶段有独立线性投影矩阵 $\mathbf{W}_{s_i} \in \mathbb{R}^{d \times d_{s_i}}$
- **DiT blocks**：所有阶段共享相同参数
- **关键点**：Patchify 成本不依赖 patch 大小（$L_s \times d_s \times d = I^2 C d$），DiT blocks 成本依赖 token 数

### 计算复杂度

每个 DiT block 的复杂度：$\mathcal{O}(L_s^2 d + L_s d)$

DiT-XL/2 中约 99.8% 的 FLOPs 在 DiT blocks 中 → 减少 token 数直接减少计算。
- 二级 PPFlow：FLOPs 减少 37.8%
- 三级 PPFlow：FLOPs 减少 50.6%

### 从预训练 DiT 初始化

**Patchify 初始化**（平均化）：将 $2 \times 2$ patch 投影扩展到 $4 \times 4$：

$$\mathbf{W}_2 = \frac{1}{4}[\mathbf{W}, \mathbf{W}, \mathbf{W}, \mathbf{W}]$$

**Unpatchify 初始化**（复制）：

$$\mathbf{W}_2^u = [(\mathbf{W}^u)^\top, (\mathbf{W}^u)^\top, (\mathbf{W}^u)^\top, (\mathbf{W}^u)^\top]^\top$$

### 与 Pyramidal Flow 的关键区别

| 特性 | PPFlow | Pyramidal Flow |
|------|--------|---------------|
| 操作分辨率 | 全分辨率潜空间 | 金字塔（多分辨率） |
| 连续性方程 | 满足 | 不满足 |
| 跳跃点 | 无 | 有（需重噪声技巧） |
| 推理流程 | 与正常 DiT 相同 | 需特殊处理 |

## 实验关键数据

### 从头训练（ImageNet 256×256, SiT-B）

| 方法 | 训练步数 | FID-50K(↓) | IS(↑) | 测试 FLOPs(%) | 加速 |
|------|---------|-----------|-------|-------------|------|
| SiT-B/2 | 7M | 4.46 | - | 100% | 1.00× |
| PPF-B-2 | 7M | **3.83** | - | ~62% | **1.61×** |
| PPF-B-3 | 7M | 4.43 | - | ~49% | **2.04×** |

### 从预训练 SiT-XL/2 微调

| 方法 | 额外训练 FLOPs | FID-50K(↓) | 测试加速 |
|------|---------------|-----------|---------|
| SiT-XL/2 | 基线 | ~2.06 | 1.00× |
| PPF-XL-2 | +8.9% | 同等质量 | **1.60×** |
| PPF-XL-3 | +7.1% | 同等质量 | **2.02×** |

### 文本到图像（基于 FLUX.1-dev）

| 分辨率 | 加速倍数 | GenEval | DPG-bench |
|-------|---------|---------|-----------|
| 512×512 | 1.61× | 可比 | 可比 |
| 1024×1024 | 1.76× | 可比 | 可比 |
| 2048×2048 | 1.86× | 可比 | 可比 |

### 关键发现

1. 二级和三级 PPFlow 分别实现 1.6× 和 2.0× 推理加速且质量保持
2. 从预训练模型微调仅需 ~8% 的额外训练成本
3. PPF-B-2 从头训练甚至优于基线 SiT-B/2（FID: 3.83 vs 4.46）
4. 高分辨率加速更显著（2048 分辨率 1.86×），因大 patch 阶段 token 减少更多
5. 阶段感知 CFG 调度（如 [1.5, 3.5, 4.0]）进一步提升质量

## 亮点与洞察

1. **极简设计**：仅修改 Patchify/Unpatchify 的线性投影，DiT blocks 完全共享
2. **无重噪声技巧**：始终操作全分辨率潜空间，消除了 Pyramidal Flow 的复杂性
3. **训练-推理一致性**：每个 patch 大小仅在对应时间步训练（不同于 FlexiDiT/Lumina-Video 的全时间步训练）
4. **Patch n' Pack**：利用变长 token 打包减少训练 FLOPs

## 局限性

- 三级以上的更激进金字塔方案未充分探索
- patch 大小和时间步分割点的选择较为启发式
- 对于小分辨率（如 256×256），大 patch 可能丢失过多空间信息
- 阶段感知 CFG 调度增加了超参数搜索空间

## 相关工作

- **推理加速**：DDIM, Progressive Distillation, Consistency Models
- **金字塔/级联**：Pyramidal Flow, PixelFlow, Cascaded Diffusion
- **变 patch 大小**：FlexiViT, FlexiDiT, Lumina-Video
- **DiT 架构**：DiT, SiT, FLUX

## 评分

- 新颖性：⭐⭐⭐⭐ — 思路简单清晰但区别于 Pyramidal Flow 的全分辨率操作是关键创新
- 技术深度：⭐⭐⭐ — 方法直观，理论分析相对简单
- 实验完整性：⭐⭐⭐⭐⭐ — 从头训练和预训练微调双验证，覆盖类条件和文本到图像
- 实用价值：⭐⭐⭐⭐⭐ — 即插即用，低成本微调即可获得显著加速

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Next Visual Granularity Generation](next_visual_granularity_generation.md)
- [\[ICLR 2026\] Purrception: Variational Flow Matching for Vector-Quantized Image Generation](purrception_variational_flow_matching_for_vector-quantized_image_generation.md)
- [\[ICLR 2026\] SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](ssg_scaled_spatial_guidance_for_multi-scale_visual_autoregressive_generation.md)
- [\[ICLR 2026\] K-Sort Eval: Efficient Preference Evaluation for Visual Generation via Corrected VLM-as-a-Judge](k-sort_eval_efficient_preference_evaluation_for_visual_generation_via_corrected_.md)
- [\[ICLR 2026\] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)

<!-- RELATED:END -->
