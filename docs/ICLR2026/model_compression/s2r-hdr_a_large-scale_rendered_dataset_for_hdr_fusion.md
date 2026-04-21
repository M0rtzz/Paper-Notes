---
title: >-
  [论文解读] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion
description: >-
  [模型压缩] 提出 S2R-HDR，首个大规模高质量合成 HDR 融合数据集（24,000 样本），并设计 S2R-Adapter 域适应方法弥合合成-真实域差距，在真实数据集上达到 SOTA HDR 融合性能。
tags:
  - 模型压缩
---

# S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion

## 基本信息

- **会议**: ICLR 2026
- **arXiv**: [2504.07667](https://arxiv.org/abs/2504.07667)
- **代码**: [项目主页](https://openimaginglab.github.io/S2R-HDR)
- **领域**: 计算机视觉 / 图像处理
- **关键词**: HDR Fusion, Synthetic Dataset, Domain Adaptation, Unreal Engine, Sim-to-Real

## 一句话总结

提出 S2R-HDR，首个大规模高质量合成 HDR 融合数据集（24,000 样本），并设计 S2R-Adapter 域适应方法弥合合成-真实域差距，在真实数据集上达到 SOTA HDR 融合性能。

## 研究背景与动机

### 问题背景
HDR 融合在计算摄影、自动驾驶等领域至关重要，但现有 HDR 数据集规模极小（最大仅 144 张），且主要局限于人工控制的简单动态场景，难以覆盖直射阳光、大运动等极端情况。

### 现有数据集的局限

**规模极小**：Kalantari (89 对)、SCT (144 张)、Challenge123 (123 张)；

**动态单一**：多数数据集仅包含基本人体运动，缺少动物、车辆等多样动态元素；

**采集困难**：真实 HDR ground truth 需逐帧拍摄不同曝光并手动控制运动，耗时且难以扩展；

**动态范围有限**：分束器仅支持两种曝光，无法覆盖极高动态范围场景。

### 核心思路
利用 Unreal Engine 5 渲染高质量合成 HDR 数据，结合域适应技术弥合合成-真实差距。

## 方法详解

### 1. S2R-HDR 数据集构建

#### 渲染流水线设计

- 修改 UE5 默认色调映射和 gamma 校正，确保输出保持线性 HDR 空间
- 使用 EXR 浮点格式存储，避免数据量化
- 模拟手持拍摄的相机抖动，增加真实感

#### 场景多样性

- **动态元素**：行人、动物、车辆等多种运动物体
- **环境类型**：室内外、白天/黄昏/夜间
- **极端光照**：直射阳光等超高动态范围场景

#### 数据规模

- **1,000 个序列 × 24 帧 = 24,000 张 HDR 图像**
- 分辨率 1920 × 1080，EXR 格式
- 比此前最大数据集大 **166 倍**

### 2. S2R-Adapter 域适应

为弥合合成-真实域差距，提出即插即用的双分支适配器：

#### Share Branch（共享分支）
使用低秩适配器保留合成数据的共享知识，防止知识遗忘：

$$
f_s = U_s V_s x, \quad r_s \ll \min(h_{in}, h_{out})
$$

#### Transfer Branch（迁移分支）
使用高秩适配器学习域特定知识，适应真实数据分布：

$$
f_t = U_t V_t x, \quad r_t \geq \max(h_{in}, h_{out})
$$

#### 最终输出
两分支通过缩放因子加权融合：

$$
f = W_0 x + \alpha_s \times f_s + \alpha_t \times f_t
$$

#### 测试时自适应（TTA）
无标注数据时，通过模型不确定性动态调节缩放因子：

$$
\alpha_s = 1 - \mathcal{U}(x); \quad \alpha_t = 1 + \mathcal{U}(x)
$$

不确定性越大，越依赖迁移分支；不确定性越小，越保留共享知识。

### 3. 推理时零额外开销

通过重参数化（re-parameterization），推理时无额外计算开销。

## 实验

### 主实验：在真实数据集上的 HDR 融合结果

| 方法 | SCT PSNR-μ | SCT SSIM-μ | Challenge123 PSNR-μ | Challenge123 SSIM-μ |
|------|-----------|-----------|---------------------|---------------------|
| DHDRNet | 40.05 | 0.9794 | 37.83 | 0.9707 |
| AHDRNet | 42.08 | 0.9837 | 40.44 | 0.9877 |
| HDR-Transformer | 42.39 | 0.9844 | 40.70 | 0.9881 |
| SCTNet | 42.55 | 0.9850 | 40.65 | — |
| EHDRNet (S2R-HDR) | 42.93 | 0.9858 | **42.15** | **0.9895** |
| **EHDRNet + S2R-Adapter** | **43.47** | **0.9871** | 41.89 | 0.9891 |

### 消融实验：域适应组件分析

| 配置 | SCT PSNR-μ | Challenge123 PSNR-μ |
|------|-----------|---------------------|
| 仅 S2R-HDR 训练 | 41.32 | 39.85 |
| + Share Branch | 42.15 | 40.71 |
| + Transfer Branch | 42.78 | 41.43 |
| + Share + Transfer (S2R-Adapter) | **43.47** | **42.15** |
| 直接真实数据微调 | 42.55 | 40.65 |

### 数据集质量对比

| 指标 | Kalantari | SCT | Challenge123 | **S2R-HDR** |
|------|----------|-----|-------------|------------|
| FHLP ↑ | 15.07 | 12.43 | 26.91 | **28.02** |
| EHL ↑ | 3.07 | 2.43 | 5.19 | **5.47** |
| SI ↑ | 18.4 | 18.25 | 20.47 | **38.02** |
| DR ↑ | 2.71 | 2.55 | 2.36 | **3.86** |
| 样本数 | 89 | 144 | 123 | **24,000** |

### 关键发现

1. **在 S2R-HDR 上训练的模型显著优于在小规模真实数据集上训练的模型**，即使存在域差距；
2. **S2R-Adapter 有效弥合域差距**，在有标注和无标注两种场景下均带来显著提升；
3. **双分支设计优于单分支**：共享分支和迁移分支各自贡献约 1 dB PSNR 提升；
4. **直接微调不如 S2R-Adapter**：直接在真实数据上微调会导致过拟合和知识遗忘；
5. **TTA 模式下仍有效**：即使无 ground truth 标注，测试时自适应也能提升约 0.5 dB。

## 亮点

- 首个大规模合成 HDR 融合数据集，24,000 样本覆盖多样场景和极端光照
- 定制化 UE5 渲染流水线保持线性 HDR 空间，模拟手持拍摄抖动
- S2R-Adapter 即插即用，兼容 CNN 和 Transformer 架构
- 支持有标注域适应和无标注测试时自适应两种模式
- 推理时通过重参数化零额外开销

## 局限性

- 合成数据在纹理分布上仍与真实数据存在差距（t-SNE 可视化可见）
- 渲染场景虽多样但仍有限，可能无法覆盖所有真实世界边缘情况
- UE5 渲染需要较高的计算资源和美术设计投入
- 域适应方法依赖校准数据集的代表性

## 相关工作

- **HDR 数据集**: Kalantari et al. (2017), SCT (Tel et al., 2023), Challenge123 (Kong et al., 2024)
- **HDR 融合方法**: AHDRNet (Yan et al., 2019), HDR-Transformer (Liu et al., 2022), DiffHDR
- **Sim-to-Real 域适应**: LoRA (Hu et al., 2021), TTA (Wang et al., 2022)
- **合成数据**: Li et al. (2023) 用于深度估计, Yang et al. (2023) 用于语义分割

## 评分

- 新颖性：⭐⭐⭐⭐ — 首个大规模 HDR 合成数据集，填补领域空白
- 技术深度：⭐⭐⭐⭐ — 渲染流水线 + 双分支适配器 + TTA 设计完善
- 实验充分度：⭐⭐⭐⭐ — 多基准、多架构对比，消融全面
- 实用价值：⭐⭐⭐⭐⭐ — 数据集和方法均可直接用于 HDR 研究和产品开发

<!-- RELATED:START -->

## 相关论文

- [StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)
- [Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)
- [Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [Variance-Based Pruning for Accelerating and Compressing Trained Networks](../../ICCV2025/model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)

<!-- RELATED:END -->
