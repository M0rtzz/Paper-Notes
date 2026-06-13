---
title: >-
  [论文解读] MSQ: Memory-Efficient Bit Sparsification Quantization
description: >-
  [ICCV 2025][模型压缩][量化] 提出MSQ，通过RoundClamp量化器从权重直接计算最低有效位(LSB)并施加L1正则化诱导稀疏性，无需显式创建bit-level可训练参数即可实现混合精度量化发现，训练参数减少8倍、训练时间减少86%，同时保持竞争性的精度-压缩权衡。 DNN在移动/边缘设备部署需要量化…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "量化"
  - "bit-level sparsity"
  - "memory-efficient"
  - "Hessian"
---

# MSQ: Memory-Efficient Bit Sparsification Quantization

**会议**: ICCV 2025  
**arXiv**: [2507.22349](https://arxiv.org/abs/2507.22349)  
**机构**: Sungkyunkwan University, University of Arizona
**领域**: 模型压缩 / 量化 / 混合精度量化  
**关键词**: mixed-precision quantization, bit-level sparsity, quantization-aware training, memory-efficient, Hessian, model compression

## 一句话总结
提出MSQ，通过RoundClamp量化器从权重直接计算最低有效位(LSB)并施加L1正则化诱导稀疏性，无需显式创建bit-level可训练参数即可实现混合精度量化发现，训练参数减少8倍、训练时间减少86%，同时保持竞争性的精度-压缩权衡。

## 背景与动机
DNN在移动/边缘设备部署需要量化。均匀量化因敏感层累积噪声导致性能下降，混合精度量化效果更好但搜索空间巨大。

现有方法的问题：
1. **搜索类方法（HAQ）**：计算代价高，不考虑训练中敏感度变化
2. **敏感度分析（HAWQ）**：仅分析预训练模型，忽略训练中敏感度动态变化
3. **Bit-level方法（BSQ/CSQ）**：将每个bit作为独立可训练变量——训练参数量翻n倍，GPU内存和时间暴增

## 核心问题
如何在不引入bit-level可训练参数开销的前提下，实现有效的bit-level稀疏性？

## 方法详解

### 核心思想
观察：不需要独立训练每个bit！从浮点权重W直接计算LSB，然后正则化LSB使其趋零，零了就可以安全剪掉→精度降低。

### 关键设计

1. **RoundClamp量化器**：

    - 标准STE量化：W_n = Round[(2^n-1)·W] / (2^n-1)
    - MSQ的bipartite bit-slicing：将n-bit值分为MSB部分和LSB部分
    - RoundClamp：先Round到n-bit，然后Clamp到(n-1)-bit范围
    - Round和Clamp的差值 = LSB的贡献
    - LSB的梯度通过STE回传到浮点权重W

2. **LSB稀疏化正则化**：

    - L_reg = lambda · sum|LSB|
    - L1正则化诱导LSB趋零
    - 当某层所有权重LSB趋零时→安全从n-bit降到(n-1)-bit
    - 无需创建独立bit变量

3. **Hessian感知激进剪枝**：

    - 用Hessian trace估计每层敏感度
    - 不敏感层→更快bit剪枝速率
    - 可同时剪多个LSB（如4-bit直接到2-bit）
    - 大幅加速训练

4. **完整训练流程**：

    - 初始化：所有层从高精度开始
    - 前向：RoundClamp量化
    - 损失：任务损失 + lambda·L1(LSB)
    - 周期性检测LSB稀疏度→超阈值则剪掉最低位
    - Hessian引导不敏感层激进剪枝

### 与BSQ/CSQ的核心差异

| | BSQ/CSQ | MSQ |
|---|---------|-----|
| 可训练参数 | 每bit独立变量 | 仅原始浮点权重 |
| 参数量 | n×原始 | 1×原始 |
| 多bit剪枝 | 每次剪1bit | Hessian引导可剪多bit |

## 实验关键数据

### 训练效率

| 方法 | 可训练参数量 | 训练时间 |
|------|------------|---------|
| BSQ | 8×原始 | 基准 |
| **MSQ** | **1×原始** | **-86%** |

### ResNet精度-压缩权衡

| 方法 | 模型 | Top-1(%) | 压缩比 |
|------|------|----------|--------|
| BSQ | ResNet-18 | 69.2 | 8.0× |
| CSQ | ResNet-18 | 69.4 | 8.0× |
| **MSQ** | ResNet-18 | **69.1** | **8.0×** |

- 精度与BSQ/CSQ几乎持平但训练成本大幅降低

### 扩展性
- 首次将bit-level量化扩展到ViT架构（ViT-S/B）
- 扩展到异构CNN如MobileNetV3

### 消融
- RoundClamp vs 标准STE：RoundClamp提供更准确的LSB梯度方向
- Hessian激进剪枝：同等压缩减少30-40%训练epoch
- L1正则化强度影响：过大→精度下降，过小→收敛慢

## 亮点
- **消除bit-level参数开销是核心贡献**：将BSQ的致命限制（n倍参数膨胀）从根本上解决
- **RoundClamp量化器设计巧妙**：利用Round和Clamp差值构造LSB，数学自然优雅
- **Hessian引导多bit剪枝**：加速训练的同时利用层敏感度信息
- **扩展性好**：首次在ViT和MobileNetV3上验证bit-level量化
- **实际工程价值**：训练效率大幅提升使混合精度量化在资源受限场景下真正可用

## 局限与展望
- 精度比BSQ/CSQ略低（0.1-0.3%），说明bit-level参数的精细控制有其价值
- L1可能不是最优稀疏化选择，可探索Group Lasso
- Hessian计算有开销（虽远小于节省时间）
- 仅验证ImageNet分类，检测/分割未知
- 仅处理权重量化，未涉及激活量化

## 与相关工作的对比
- **vs. HAQ**：搜索类，计算代价高且静态分析；MSQ动态发现精度方案
- **vs. BSQ**：开创性bit-level但参数膨胀严重；MSQ保持优势同时消除开销
- **vs. CSQ**：平滑了BSQ但未解决参数膨胀

## 启发与关联
- "不需要独立训练每个bit"——很多看似需要细粒度参数的问题有更高效的间接方式
- RoundClamp构造思路（两种量化方式的差→有用信号）对其他量化方法有启发
- 对边缘端量化训练有直接工程价值

## 评分
- 新颖性: ⭐⭐⭐⭐ RoundClamp和LSB正则化新颖，但整体是BSQ的自然改进
- 实验充分度: ⭐⭐⭐⭐ 效率对比充分+多架构验证，但缺下游任务评估
- 写作质量: ⭐⭐⭐⭐ 与BSQ/CSQ对比清晰，Figure 1/2对比图好
- 价值: ⭐⭐⭐⭐⭐ 8倍参数减少+86%训练时间减少，使bit-level量化真正可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)
- [\[ACL 2025\] Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](../../ACL2025/model_compression/outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](../../ICML2026/model_compression/nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)

</div>

<!-- RELATED:END -->
