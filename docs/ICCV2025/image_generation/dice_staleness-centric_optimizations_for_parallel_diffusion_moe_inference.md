---
title: >-
  [论文解读] DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference
description: >-
  [ICCV 2025][图像生成][MoE] 针对 MoE 扩散模型并行推理中的"陈旧性"问题 (staleness)，提出 DICE 框架，通过步级交织并行、层级选择性同步和 token 级条件通信三层优化策略，在 DiT-MoE 上实现 1.26× 加速且质量损失极小。
tags:
  - ICCV 2025
  - 图像生成
  - MoE
  - 扩散模型
  - 专家并行
  - 陈旧性优化
  - 通信优化
---

# DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference

**会议**: ICCV 2025  
**arXiv**: [2411.16786](https://arxiv.org/abs/2411.16786)  
**代码**: [https://github.com/Cobalt-27/DICE](https://github.com/Cobalt-27/DICE)  
**领域**: 图像生成 / 扩散模型并行推理  
**关键词**: MoE, 扩散模型, 专家并行, 陈旧性优化, 通信优化

## 一句话总结

针对 MoE 扩散模型并行推理中的"陈旧性"问题 (staleness)，提出 DICE 框架，通过步级交织并行、层级选择性同步和 token 级条件通信三层优化策略，在 DiT-MoE 上实现 1.26× 加速且质量损失极小。

## 研究背景与动机

基于 Mixture-of-Experts 的扩散模型（如 DiT-MoE）可扩展到 160 亿参数并展现卓越的生成质量，但其对专家并行 (expert parallelism) 的依赖引入了严重的通信瓶颈。在 8 GPU 上测试 DiT-MoE-XL，all-to-all 通信占总推理时间的 61.7%-73.3%。

现有的 Displaced Parallelism（由 DistriFusion 提出）通过异步通信实现计算-通信重叠来缓解阻塞，但引入了一个关键问题——**陈旧性 (staleness)**：即使用来自更早时间步的过时激活值，导致 FID 从 5.31 恶化到 8.27。

作者的核心发现：Displaced 并行在专家并行场景中导致 **2-step 陈旧性**（dispatch 延迟 1 步 + combine 延迟 1 步），这对生成质量的损害在 MoE 架构中尤为严重。同时，作者观察到 DiT-MoE 在相邻扩散步之间存在高度的**路由冗余**（token-专家分配高度相似），这为异步通信提供了可行性基础。

## 方法详解

### 整体框架

DICE 从三个粒度优化陈旧性：步级 (step-level)、层级 (layer-level) 和 token 级 (token-level)，形成一个协同优化框架。

### 关键设计

1. **交织并行 (Interweaved Parallelism)** — 步级优化：

   传统 displaced 并行的陈旧性为 2 步（dispatch 1 步 + combine 1 步）。交织并行通过重新编排通信和计算的时序，实现：
   - 异步 dispatch 在交错执行中于当前步内完成（0 步延迟）
   - 处理专家输出的同时发起下一个 combine 操作
   - combine 结果在下一步才就绪（1 步延迟）

   $$\text{Staleness}_{\text{interweaved}} = \underbrace{0}_{\text{dispatch}} + \underbrace{1}_{\text{combine}} = 1\text{-step}$$

   相比 displaced 并行，陈旧性减半，缓存大小减半（only store combine results），且没有额外开销——是一个 **free-lunch** 优化。单独使用即可将 FID 从 8.27 改善至 6.97。

2. **选择性同步 (Selective Synchronization)** — 层级优化：

   分析发现陈旧性影响具有**层级不对称性**：浅层专家提取低级特征，对异步通信天然鲁棒；深层专家处理高级语义，对激活陈旧性极为敏感。这与 DeepSpeed-MoE 在语言模型中发现深层更受益于 MoE 的观察一致。

   因此，DICE 仅对更深层执行同步通信（保持数据新鲜），浅层继续异步执行。消融实验证实同步深层效果最佳（FID 5.74），远优于同步浅层（6.55）。

3. **条件通信 (Conditional Communication)** — token 级优化：

   利用 MoE 路由的固有属性：高路由分数的 token 通过加权求和主导输出，更易受陈旧性扰动。具体来说，陈旧性激活扰动 $\Delta \mathbf{h}_i^e$ 对输出的传播与路由分数 $s_i^e$ 成正比：

   $$\frac{\partial \|\mathbf{y}_i\|}{\partial \mathbf{h}_i^e} = \frac{s_i^e \cdot \mathbf{y}_i}{\|\mathbf{y}_i\|}$$

   因此，高分 token 每步传输以保持新鲜，低分 token 复用缓存值、降低通信频率。该策略无需训练。

### 损失函数 / 训练策略

DICE 是一个推理优化框架，不涉及训练。核心是在推理时合理编排通信调度以减少陈旧性。

## 实验关键数据

### 主实验 (ImageNet 256×256, DiT-MoE-XL, 50 步)

| 方法 | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ |
|------|------|-------|-----|-----------|---------|
| Expert Parallelism (同步) | 5.31 | 10.10 | 235.89 | 0.75 | 0.60 |
| DistriFusion | 7.79 | 12.13 | 206.24 | 0.72 | 0.59 |
| Displaced Expert Para. | 8.27 | 11.58 | 204.07 | 0.71 | 0.59 |
| Interweaved Para. (单独) | 6.97 | 11.01 | 216.62 | 0.72 | 0.59 |
| **DICE** | **6.11** | **10.93** | **225.65** | **0.73** | 0.59 |

少步数实验 (10 步/20 步)：

| 步数 | 方法 | FID↓ | Speedup↑ |
|------|------|------|----------|
| 10 | Expert Para. | 10.24 | - |
| 10 | Displaced Expert Para. | 27.61 | 1.28× |
| 10 | **DICE** | **15.13** | **1.20×** |
| 20 | Expert Para. | 6.41 | - |
| 20 | Displaced Expert Para. | 15.27 | 1.33× |
| 20 | **DICE** | **8.60** | **1.24×** |

加速效果：DICE 在 batch size 32 时最高达 1.26× 加速，DistriFusion 在多数配置下 OOM。

### 消融实验

| Interweaved | Selective Sync | Conditional Comm | FID↓ | IS↑ |
|-------------|---------------|-----------------|------|-----|
| ✓ | × | × | 6.97 | 216.62 |
| ✓ | Deep | × | **5.74** | **230.23** |
| ✓ | Shallow | × | 6.55 | 221.61 |
| ✓ | Staggered | × | 5.95 | 227.78 |
| ✓ | × | Low Score | 7.24 | 214.10 |
| ✓ | × | High Score | 7.51 | 211.40 |
| ✓ | × | Random | 7.37 | 212.84 |

### 关键发现

- **交织并行是免费午餐**：陈旧性减半、缓存减半、重叠度相同、无额外开销
- **深层比浅层更敏感**：同步深层 FID 5.74 vs 同步浅层 6.55，差距 0.81
- **路由分数是有效的 token 重要性信号**：降低低分 token 通信频率比降低高分 token 或随机降低效果更好
- DICE 在**少步数 (10/20 步) 场景下优势更大**：步数越少，每步陈旧性的相对影响越大
- DistriFusion 在 DiT-MoE-G（33GB 参数）上直接 OOM，说明专家并行是大规模 MoE 扩散模型的必要选择
- All-to-all 通信在 batch size 16 时占比达 73.3%，优化通信是关键

## 亮点与洞察

1. **问题定义精准**："陈旧性"概念的引入为异步并行推理提供了统一的分析框架
2. **多粒度系统性优化**：步级/层级/token 级三个维度各有针对性，协同效果好
3. **交织并行的优雅**：仅通过通信调度变化就减半陈旧性，不增加任何计算或通信开销
4. **实用性强**：所有优化对模型透明、无需重训练、代码开源

## 局限与展望

- 仅在 DiT-MoE 上验证，未覆盖其他 MoE 扩散架构（如 Switch-DiT）
- 条件通信的 stride 参数（通信频率）需要手动设置
- 选择性同步的深浅层分界目前是对半划分，可探索更细粒度的自适应策略
- 硬件环境仅为 PCIe 连接的 4090，NVLink/InfiniBand 环境下的表现可能不同
- 未涉及 video diffusion 等更复杂的生成场景

## 相关工作与启发

- Displaced Parallelism (DistriFusion)：DICE 的直接改进目标
- FasterMoE、DeepSpeed-MoE：MoE 通信优化的先驱，但未针对扩散模型
- 缓存类方法（DeepCache、Learn2Cache）：与通信优化互补
- PipeFusion、AsyncDiff：同样利用激活相似性，DICE 的系统性更强

## 评分

- **新颖性**: ⭐⭐⭐⭐ "陈旧性"的系统性分析和交织并行的设计很巧妙
- **实验充分度**: ⭐⭐⭐⭐ 多配置（batch size/步数/模型规模）、消融充分
- **写作质量**: ⭐⭐⭐⭐⭐ 图示极其清晰，特别是 Figure 2 的执行流对比
- **价值**: ⭐⭐⭐⭐ 对大规模 MoE 扩散模型的部署有直接工程价值

<!-- RELATED:START -->

## 相关论文

- [Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [Inference-Time Diffusion Model Distillation](inference-time_diffusion_model_distillation.md)
- [Accelerating Parallel Diffusion Model Serving with Residual Compression](../../NeurIPS2025/image_generation/accelerating_parallel_diffusion_model_serving_with_residual_compression.md)
- [GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](../../CVPR2025/image_generation/glass_guided_latent_slot_diffusion_for_object-centric_learning.md)
- [IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling](../../ICML2025/image_generation/impact_iterative_mask-based_parallel_decoding_for_text-to-audio_generation_with_.md)

<!-- RELATED:END -->
