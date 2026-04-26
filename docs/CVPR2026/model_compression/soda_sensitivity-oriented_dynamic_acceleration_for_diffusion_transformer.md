---
title: >-
  [论文解读] SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer
description: >-
  [CVPR2026][模型压缩][Transformer] 提出 SODA，通过离线细粒度敏感度建模 + 动态规划优化缓存间隔 + 统一自适应剪枝策略，在无需训练的条件下对 Diffusion Transformer 实现可控加速比下的高保真生成。
tags:
  - CVPR2026
  - 模型压缩
  - Transformer
  - 训练无关加速
  - 缓存
  - 剪枝
  - 敏感度建模
  - 动态规划
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer

**会议**: CVPR2026  
**arXiv**: [2603.07057](https://arxiv.org/abs/2603.07057)  
**代码**: [leaves162/SODA](https://github.com/leaves162/SODA)  
**领域**: 模型压缩  
**关键词**: 扩散 Transformer, 训练无关加速, 缓存, 剪枝, 敏感度建模, 动态规划

## 一句话总结

提出 SODA，通过离线细粒度敏感度建模 + 动态规划优化缓存间隔 + 统一自适应剪枝策略，在无需训练的条件下对 Diffusion Transformer 实现可控加速比下的高保真生成。

## 研究背景与动机

**DiT 推理效率瓶颈**：Diffusion Transformer 在图像和视频生成中表现优异，但重复采样步骤和 Transformer 块带来的低推理效率严重阻碍部署。

**训练方法代价高**：蒸馏/微调等训练加速方法计算开销大、泛化性有限，研究者更倾向于免训练加速。

**缓存与剪枝各有取舍**：缓存加速效率高但损失保真度，剪枝灵活但效率较低；两者结合可兼顾效率与质量。

**现有策略依赖经验设计**：ToCa、DuCa 等方法对缓存间隔和剪枝率使用固定/启发式配置，仅捕捉粗粒度敏感趋势，无法感知时间步、层、模块级别的细粒度变化。

**跳过高敏感计算导致质量退化**：启发式方案不可避免地跳过对加速高度敏感的计算，造成生成保真度下降。

**泛化性差**：手工设计的策略依赖经验，难以跨模型迁移。

## 方法详解

### 整体框架

SODA 由三个模块组成：**(1) OFS** — 离线细粒度敏感度建模；**(2) DCS** — 动态缓存调度优化；**(3) UAS** — 统一自适应策略制定。推理流程为：先离线建模敏感度先验 → 用动态规划求解最优缓存间隔组合 → 在剪枝/缓存复用阶段自适应决定剪枝时机和比率。

### 关键设计

#### 1. 离线细粒度敏感度建模 (OFS)

- **缓存敏感度误差** $\mathcal{E}_c(t,l,m,n) = 1 - \text{Cos}(\mathcal{D}_{t+n,l,m}(x), \mathcal{D}_{t,l,m}(x))$：衡量在时间步 $t$ 复用 $t+n$ 步缓存的 Cosine 距离误差，覆盖时间步/层/模块/缓存间隔四个维度。
- **剪枝敏感度误差** $\mathcal{E}_p(t,l,m,\alpha)$：类似地，衡量在不同剪枝率 $\alpha$ 下的特征误差。
- **离线策略**：使用 100 次随机生成取平均（视频模型用 10 次），将敏感度误差作为模型特定先验存储，每个模型只需建模一次即可永久复用。离线开销极低（DiT-XL/2 约 160s，内存仅增加 0.56GB）。

#### 2. 动态缓存调度优化 (DCS)

- 将缓存间隔组合问题建模为具有最优子结构的动态规划问题：$dp[t][i+1] = \min_{n \in \mathcal{N}} \{\mathcal{E}_{dp}(t,n) + dp[t+n][i]\}$。
- 给定加速预算（缓存次数 $N_s$），求解从 $T$ 到 1 的最小累计敏感度误差，并通过回溯恢复最优缓存时间步和间隔。
- 算法在离线敏感度误差上运行，无额外推理开销。

#### 3. 统一自适应策略制定 (UAS)

- **自适应剪枝时机**：仅当剪枝误差 $\delta_{t+1,l,m} = \mathcal{E}_p(t+1,l,m,\alpha)$ 小于缓存误差 $\mathcal{E}_c(t,l,m,n)$ 时才执行剪枝，否则直接复用缓存，保证剪枝总能降低整体误差。
- **自适应剪枝率**：$\alpha_{t+1,l,m} = \lambda \cdot \mathcal{E}_c(t,l,m,n) + \beta$，其中 $\lambda$ 为缩放系数，$\beta$ 为基础剪枝率（根据加速预算自适应调整），使剪枝同时感知全局预算和局部敏感度。
- **Token 重要性度量**：使用特征均值做 TopK 选择，避免 Flash Attention 下注意力权重不可用的兼容性问题。

### 损失/优化目标

无额外训练损失。核心优化目标为动态规划中最小化累计敏感度误差 $dp[1][N_s]$。

## 实验

### 主要结果

| 模型 / 设置 | 加速比 | FID↓ | sFID↓ | IS↑ | 说明 |
|---|---|---|---|---|---|
| DiT-XL/2 DDPM 原始 | 1.00× | 2.23 | 4.57 | 275.65 | — |
| ToCa (DDPM) | 2.75× | 2.58 | 5.74 | 256.26 | — |
| DuCa (DDPM) | 2.73× | 2.59 | 5.68 | 256.36 | — |
| **SODA (DDPM, $N_s$=72)** | **2.73×** | **2.47** | **5.09** | **262.30** | sFID 降 0.65, IS 升 6+ |
| DiT-XL/2 DDIM 原始 | 1.00× | 2.25 | 4.33 | 239.97 | — |
| DuCa (DDIM) | 2.48× | 3.05 | 4.66 | 233.21 | — |
| **SODA (DDIM, $N_s$=18)** | **2.49×** | **2.75** | **4.56** | **235.65** | FID 降 0.30 |

| 模型 | 加速比 | FID-30K↓ | CLIP↑ | 说明 |
|---|---|---|---|---|
| PixArt-α 原始 | 1.00× | 28.10 | 16.29 | — |
| DuCa | 1.87× | 28.05 | 16.42 | — |
| **SODA ($N_s$=8)** | **1.88×** | **27.33** | **16.42** | FID 降 0.72 vs DuCa |
| **SODA ($N_s$=7)** | **2.21×** | **27.72** | **16.44** | 更高加速仍优于原始 |

### 消融实验

| 配置 | FID↓ | sFID↓ | IS↑ |
|---|---|---|---|
| Vanilla (固定缓存) | 3.83 | 5.24 | 213.12 |
| OFS + DCS | 2.78 | 4.63 | 234.78 |
| OFS + UAS | 2.89 | 4.75 | 235.01 |
| **SODA (完整)** | **2.75** | **4.56** | **235.65** |

- DCS 单独贡献 FID 降 1.05、IS 升 21.66；UAS 单独贡献 FID 降 0.94、IS 升 21.89；两者结合效果最优。
- Cosine 距离作为敏感度度量优于 L1/L2。

### 关键发现

1. **低加速可提升原始性能**：1.55× 时 FID 从 2.23 降至 2.21（DDPM），可能因跳过冗余计算反而去噪更稳定。
2. **离线建模具有高一致性**：离线敏感度分布与实际推理时高度吻合（Fig. 6），方差稳定，证明敏感度是模型固有属性。
3. **超参数不敏感**：$\lambda$ 和 $\beta$ 变化范围内 ΔFID ≤ 0.02、ΔIS ≤ 2.2。
4. **视频任务同样有效**：OpenSora 2.50× 加速下 VBench 仅降 0.64%，优于 ToCa/DuCa/PAB/FORA。

## 亮点

- 首次将敏感度建模推至时间步×层×模块的细粒度，并用动态规划统一求解最优缓存策略，理论上保证全局最优。
- 剪枝决策与缓存决策通过敏感度误差统一，自适应保留高敏感 token，既无需手工设计又具备泛化性。
- 离线建模仅需一次且开销极低（< 1 小时），运行时只加载 0.16MB 先验，零额外推理开销。
- 跨任务泛化：同一框架无修改地适用于类别条件图像生成、文生图、文生视频三类任务。

## 局限性

- 作为免训练方法，加速效果仍落后于蒸馏等训练方法。
- 与蒸馏等训练技术的集成尚未探索。
- 离线建模需要对每个新模型执行一次（虽然开销小），不完全是即插即用。
- 剪枝位置选择依赖特征均值启发式，可能不是最优的 token 重要性度量。
- 动态规划搜索空间随缓存间隔候选集增大而增长，极端大步数场景下需关注求解效率。

## 相关工作

- **缓存加速**：FORA、FasterDiffusion、ToCa (ICLR 2025)、DuCa — 利用相邻时间步相似性复用中间特征，但采用固定/启发式缓存策略。
- **Token 剪枝**：ToMe、AT-EDM — 基于 token 冗余进行裁剪，灵活但效率低于缓存。
- **缓存+剪枝结合**：ToCa、DuCa — 在锚步全计算后缓存，中间步剪枝+缓存复用，但策略手工设计。
- **敏感度分析启发**：本文受 LLM 量化中的敏感度分析（如 SqueezeLLM）启发，将其扩展至扩散模型的多维度加速决策。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将敏感度建模与动态规划结合做缓存优化是新颖的，统一剪枝/缓存决策思路清晰。
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个模型、三种任务、完整消融、离线分析、超参敏感度、定性对比，非常充分。
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题驱动，Fig.1 动机图效果好。
- 价值: ⭐⭐⭐⭐ — 方法实用且泛化性强，开源代码可复现，对 DiT 加速社区有直接贡献。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning](../../CVPR2025/model_compression/tadformer_task-adaptive_dynamic_transformer_for_efficient_multi-task_learning.md)
- [\[CVPR 2026\] PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](ppcl_pluggable_pruning_dit_distillation.md)
- [\[CVPR 2026\] Batch Loss Score for Dynamic Data Pruning](batch_loss_score_for_dynamic_data_pruning.md)
- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](../../ICLR2026/model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)

<!-- RELATED:END -->
