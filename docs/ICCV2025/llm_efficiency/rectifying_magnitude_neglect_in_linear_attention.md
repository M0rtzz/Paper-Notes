---
title: >-
  [论文解读] Rectifying Magnitude Neglect in Linear Attention
description: >-
  [ICCV 2025][LLM效率][注意力机制] 揭示 Linear Attention 完全忽略 Query 幅值信息导致注意力分数分布与 Softmax Attention 显著偏离，提出 Magnitude-Aware Linear Attention (MALA)，通过引入缩放因子 β 和偏移项 γ 使线性注意力恢复幅值感知能力，在分类、检测、分割、NLP、语音、图像生成等任务上全面超越现有方法。
tags:
  - ICCV 2025
  - LLM效率
  - 注意力机制
  - Magnitude-Aware
  - Transformer
  - 注意力分数分布
  - 线性复杂度
---

# Rectifying Magnitude Neglect in Linear Attention

**会议**: ICCV 2025  
**arXiv**: [2507.00698](https://arxiv.org/abs/2507.00698)  
**代码**: [https://github.com/qhfan/MALA](https://github.com/qhfan/MALA)  
**领域**: LLM Efficiency / Vision Transformer  
**关键词**: Linear Attention, Magnitude-Aware, Vision Transformer, 注意力分数分布, 线性复杂度

## 一句话总结

揭示 Linear Attention 完全忽略 Query 幅值信息导致注意力分数分布与 Softmax Attention 显著偏离，提出 Magnitude-Aware Linear Attention (MALA)，通过引入缩放因子 β 和偏移项 γ 使线性注意力恢复幅值感知能力，在分类、检测、分割、NLP、语音、图像生成等任务上全面超越现有方法。

## 研究背景与动机

Softmax Attention 的二次复杂度 $O(N^2)$ 限制了 Vision Transformer 在高分辨率视觉任务中的应用。Linear Attention 通过核函数近似将复杂度降至 $O(N)$，但性能显著下降。

现有改进（如 EfficientViT 加卷积补偿局部感知、Flatten Transformer 的 focused linear attention）多是启发式的"打补丁"策略。本文从**数学公式层面**分析了性能差距的根本原因：

**核心发现**：Linear Attention 的计算形式中，Query 的幅值信息 $\|\phi(Q_i)\|$ 在分子分母中完全约分消失（Eq.4），只保留方向信息 $\vec{\alpha_i}$。这意味着：
- Softmax Attention：Query 幅值增大 → 注意力分布变得更尖锐（高分 Key 获得更多注意力）
- Linear Attention：无论 Query 幅值如何变化，注意力分布保持不变

这解释了 Linear Attention 注意力分数过于平滑、局部感知弱的长期困扰。

## 方法详解

### 整体框架

MALA 修改 Linear Attention 的归一化方式：将除法归一化改为加法归一化，引入与 $\phi(Q_i)$ 幅值相关的缩放因子 β 和偏移项 γ，使注意力分数能随 Query 幅值动态调整。

### 关键设计

1. **幅值忽略问题的形式化证明**：

    - 将 $\phi(Q_i) = \|\phi(Q_i)\| \vec{\alpha_i}$ 代入 Linear Attention 公式，幅值项在分子分母中消掉
    - 实验验证：在 DeiT-T 的 Softmax Attention 中用 $Q/\|Q\|$ 替代 $Q$（丢弃幅值），精度从 72.2% 降至 70.0%，接近 Linear Attention 的 69.8%
    - 注意力分数可视化也收敛到 Linear Attention 的平滑分布

2. **MALA 公式设计**：

    - 注意力分数：$\text{Attn}(Q_i, K_j) = \beta \cdot \phi(Q_i)\phi(K_j)^T - \gamma$
    - 缩放因子：$\beta = 1 + \frac{1}{\phi(Q_i)\sum_m \phi(K_m)^T}$（与 Query 幅值负相关）
    - 偏移项：$\gamma = \frac{\phi(Q_i)\sum_m \phi(K_m)^T}{N}$（与 Query 幅值正相关）
    - 保持归一化：$\sum_j \text{Attn}(Q_i, K_j) = 1$
    - **核心性质**：当 $\|\phi(Q_i)\|$ 增大 $a$ 倍时，高分 Key 与低分 Key 的注意力比值增大（$p_m > p$），与 Softmax Attention 趋势一致

3. **幅值变化速率的差异（关键洞察）**：

    - Softmax Attention 中比值 $p$ 随缩放因子 $a$ **指数增长**（$p^a$）→ 注意力过于尖锐
    - MALA 中比值 $p$ 随 $a$ **分数增长**（更温和）→ 分布更平衡
    - 可视化证实：Softmax 过于聚焦局部、Linear 过于平滑、MALA 取得良好平衡
    - **线性复杂度保持**：$Y_i = \beta \phi(Q_i)\sum_j \phi(K_j)^T V_j - \gamma \sum_j V_j$，仍可先算 $K^TV$ 再与 $Q$ 交互

### 损失函数 / 训练策略

- 构建 MAViT（Magnitude-Aware Vision Transformer）系列模型 T/S/B/L
- 图像分类：ImageNet-1K 从头训练 300 epochs，随机深度最大率 0.1/0.15/0.4/0.55
- 检测/分割：标准 COCO/ADE20K 配置，使用 RetinaNet/Mask R-CNN/Cascade Mask R-CNN/SemanticFPN/UperNet

## 实验关键数据

### 主实验

ImageNet-1K 分类精度对比（关键规模）：

| 模型 | 类型 | Params(M) | FLOPs(G) | Top-1(%) |
|------|------|-----------|----------|----------|
| RMT-S | Trans | 27 | 4.5 | 84.1 |
| SECViT-S | Trans | 27 | 4.6 | 84.3 |
| RAVLT-S | Linear | 26 | 4.6 | 84.4 |
| **MAViT-S** | **Linear** | **27** | **4.6** | **84.7** |
| RMT-B | Trans | 54 | 9.7 | 85.0 |
| RAVLT-B | Linear | 48 | 9.9 | 85.5 |
| **MAViT-B** | **Linear** | **50** | **9.9** | **85.7** |
| RMT-L | Trans | 95 | 18.2 | 85.5 |
| **MAViT-L** | **Linear** | **98** | **16.1** | **86.0** |

COCO 检测（Cascade Mask R-CNN 3×+MS）：MAViT-B 达到 55.5 $AP^b$ / 48.0 $AP^m$，超越更大的 CSwin-B。

### 消融实验

Linear Attention 替换对比（DeiT-T/Swin-T/Swin-S 设置下，仅替换注意力机制）：

| Linear Attention 类型 | DeiT-T | Swin-T | Swin-S |
|----------------------|--------|--------|--------|
| Hydra Attn | 68.3 | 80.7 | — |
| Enhanced Linear Attn | 72.9 | 81.8 | — |
| Focused Linear Attn | 74.1 | 82.1 | 83.5 |
| InLine Attn | 74.5 | 82.4 | 83.6 |
| **MALA** | **75.1** | **83.7** | **85.3** |

β 和 γ 消融（MAViT-T）：去除 β 精度降至 52.3%，去除 γ 导致 NaN，用可学习参数替代降至 71.7%。

核函数不敏感：ELU+1、ReLU、exp 几乎等效（82.9 vs 82.8 vs 82.9）。

### 关键发现

- MALA 在**所有测试任务**上均优于 Softmax Attention，同时保持线性复杂度
- NLP（0.3B 模型/15B tokens）：MALA 在 LMB/PIQA/Hella/Wino 上与 Transformer 和 Mamba 竞争力相当
- 语音识别（Conformer 替换）：WER 从 Softmax 的 2.7/6.3 降至 MALA 的 2.4/5.3
- 图像生成（DiT 框架）：FID 从 68.40 降至 49.62，吞吐量 5.6 imgs/s（最快）
- 高分辨率推理优势明显：512×2048 语义分割中 MAViT 效率显著优于 Softmax 模型
- 实验中**未观察到负/零注意力分数**（理论上可能但实际不发生）

## 亮点与洞察

- **分析切入点精准**：不是启发式地"加卷积补偿"，而是从数学上揭示 Linear Attention 的本质缺陷（幅值消除），解法直接对应问题
- **β 和 γ 的设计极简而有效**：仅两个分析推导出的超参数（非可学习），消融证明不可替代
- 增长速率的差异是一个深刻洞察：Softmax 的指数增长导致过于尖锐，MALA 的分数增长提供更合理的分布
- 跨领域验证（视觉/NLP/语音/生成）说明这是一个基础性改进而非特定任务的技巧

## 局限与展望

- β 和 γ 引入额外的逐 token 标量计算，虽然不影响渐近复杂度但增加常数开销
- 对负注意力分数的理论分析不够深入（仅声明实验中未观察到）
- 未与 Mamba/SSM 类方法在大规模 NLP 上进行充分比较
- Linear Attention 在超长序列（如视频、基因组）上的表现未探索

## 相关工作与启发

- Flatten Transformer (ICCV 2023) 和 InLine Attention (NeurIPS 2024) 是最直接的对比基线
- MILA (NeurIPS 2024) 受 Mamba 启发改造 Linear Attention 的宏架构，但未触及幅值问题
- RMT (CVPR 2024) 是同组前作，使用 Retentive 机制，MALA 提供了更根本的改进
- 对 Softmax 过于尖锐 vs Linear 过于平滑的分析，对设计新注意力机制有普遍指导意义

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 问题发现和解法都极其优雅，从数学本质出发
- **实验充分度**: ⭐⭐⭐⭐⭐ 7 个任务全面验证，消融实验非常细致
- **写作质量**: ⭐⭐⭐⭐ 数学推导清晰，但部分公式排版可改进
- **价值**: ⭐⭐⭐⭐⭐ 对 Linear Attention 的基础性改进，影响面广

<!-- RELATED:START -->

## 相关论文

- [Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](../../NeurIPS2025/llm_efficiency/tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)
- [ZeroS: Zero-Sum Linear Attention for Efficient Transformers](../../NeurIPS2025/llm_efficiency/zeros_zero-sum_linear_attention_for_efficient_transformers.md)
- [RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](../../ICLR2026/llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)
- [xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](../../ICLR2026/llm_efficiency/xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)
- [Tensor Product Attention Is All You Need](../../NeurIPS2025/llm_efficiency/tensor_product_attention_is_all_you_need.md)

<!-- RELATED:END -->
