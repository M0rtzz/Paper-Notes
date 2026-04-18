---
title: >-
  [论文解读] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment
description: >-
  [ICML2025][图像分割][State Space Model] 首次将 Vision Mamba（状态空间模型）引入图像质量评估（IQA），提出 QMamba 框架和 StylePrompt 轻量微调策略，在合成/真实/AIGC 多种 IQA 任务上以更低计算成本超越 CNN 和 Transformer 基线。
tags:
  - ICML2025
  - 图像分割
  - State Space Model
  - Mamba
  - 图像质量评估
  - 迁移学习
  - 提示学习
---

# QMamba: On First Exploration of Vision Mamba for Image Quality Assessment

**会议**: ICML2025  
**arXiv**: [2406.09546](https://arxiv.org/abs/2406.09546)  
**代码**: [GitHub](https://github.com/bingo-G/QMamba)  
**领域**: 图像质量评估 (IQA)  
**关键词**: State Space Model, Mamba, 图像质量评估, transfer learning, prompt tuning

## 一句话总结

首次将 Vision Mamba（状态空间模型）引入图像质量评估（IQA），提出 QMamba 框架和 StylePrompt 轻量微调策略，在合成/真实/AIGC 多种 IQA 任务上以更低计算成本超越 CNN 和 Transformer 基线。

## 研究背景与动机

图像质量评估（IQA）旨在衡量图像的主观感知质量，广泛应用于图像压缩、增强、AIGC 等场景。现有 IQA 主干网络存在固有局限：

- **CNN**：擅长学习局部平移不变特征，但缺乏长程依赖建模能力，难以进行全局质量感知
- **Vision Transformer**：通过自注意力机制有效建模长程依赖，但二次复杂度带来高昂计算开销，尤其对大尺度图像不友好
- **核心问题**：能否找到一种既具备全局建模能力、又保持线性复杂度的骨干网络用于 IQA？

Mamba（State Space Model）在分割、分类等高层任务中已展现出平衡性能与效率的潜力，但其在**低层视觉感知**（如质量评估）上的能力尚未被探索。本文首次回答了"Mamba 能否在低层感知任务上超越现有骨干"这一问题。

## 方法详解

### 整体架构

QMamba 采用层级残差结构，包含多个网络阶段，每个阶段由下采样层 + 增强的 Mamba 处理模块组成。通过逐层抽象构建多尺度表示，提取丰富的感知特征。提供三个变体：

- **QMamba-Tiny**：4 blocks，嵌入维度 96，27.99M 参数，4.47G FLOPs
- **QMamba-Small**：15 blocks，嵌入维度 96，49.37M 参数，8.71G FLOPs
- **QMamba-Base**：15 blocks，嵌入维度 128，87.53M 参数，15.35G FLOPs

### 局部窗口扫描（Local Scanning）

原始 VMamba 的跨扫描策略将 2D 图像展平为 1D 序列，破坏了相邻 token 的空间连续性，不利于捕获 IQA 中关键的**局部失真**信息。QMamba 改用窗口式扫描：

1. 在局部窗口内执行水平/垂直扫描
2. 再跨窗口进行扫描
3. 随网络深度变化窗口大小，实现多尺度感知

这种层级式固定窗口设计（LQMamba）避免了基于注意力的动态路由带来的推理不稳定和高计算开销，同时兼顾局部细节与全局上下文。

### StylePrompt 微调策略

为提升 QMamba 在不同 IQA 领域间的迁移能力，基于"IQA 中的域偏移往往与特征统计/风格相关"这一发现，提出 StylePrompt：

**StylePrompt 生成（SPG）**：在每个网络阶段学习一组提示 $P_s \in \mathbb{R}^{N \times 1 \times 1 \times C}$，通过输入特征的全局池化 + Softmax 预测各提示分量的权重，融合为风格提示：

$$P_f = \sum_{c=1}^{N} w_s P_s, \quad w_s = \text{Softmax}(\text{Conv}_{1 \times 1}(\text{GAP}(F_i)))$$

**StylePrompt 注入（SPI）**：将融合提示通过线性层生成仿射参数 $\gamma_v, \beta_v \in \mathbb{R}^{1 \times 1 \times \hat{C}}$，沿通道维度调节原始特征的均值和方差：

$$\gamma_v = \text{Linear}_\gamma(\text{Conv}(P_f)), \quad \beta_v = \text{Linear}_\beta(\text{Conv}(P_f))$$

$$F_i' = F_i \cdot (1 + \gamma_v) + \beta_v$$

仅需微调约 **3.83M 参数（全部参数的 4%）** 即可达到接近全参数微调的效果。

## 实验关键数据

### Task-Specific IQA（8个数据集平均 PLCC/SRCC）

| 方法 | 参数量 | GFLOPs | 平均性能 |
|------|--------|--------|----------|
| DEIQT | 24.04M | 5.41G | 0.884 |
| Swin-B | 86.74M | 15.47G | 0.872 |
| ViT-B | 85.80M | 17.58G | 0.854 |
| **QMamba-T** | **27.99M** | **4.47G** | **0.893** |
| **LQMamba-S** | **52.91M** | **8.66G** | **0.895** |
| **LQMamba-B** | **93.79M** | **15.30G** | **0.896** |

### Universal IQA（6个数据集混合训练）

| 方法 | GFLOPs | PLCC_Avg | SRCC_Avg |
|------|--------|----------|----------|
| Swin-T | 4.51G | 0.900 | 0.883 |
| DEIQT | 5.41G | 0.895 | 0.873 |
| **LQMamba-T** | **4.44G** | **0.909** | **0.888** |

### Transferable IQA（StylePrompt 迁移效果）

| 微调策略 | 参数量 | PLCC_Avg | SRCC_Avg |
|----------|--------|----------|----------|
| 不微调 | 0 | — | 0.642 |
| Full tuning | 93.79M | — | 0.908 |
| **StylePrompt** | **3.83M** | — | **0.901** |

StylePrompt 仅用 4% 参数即达到 Full tuning 99% 的性能。

### Prompt 策略消融

| 策略 | 参数量 | PLCC_Avg | SRCC_Avg |
|------|--------|----------|----------|
| SSF | 6.1M | 0.750 | 0.735 |
| Crossattn_Prompt | 12.17M | 0.806 | 0.772 |
| Conv_Prompt | 28.33M | 0.883 | 0.856 |
| **StylePrompt** | **3.83M** | **0.911** | **0.890** |

## 亮点与洞察

1. **首次探索 Mamba 的低层感知能力**：系统验证了 SSM 在 IQA 任务上的优势，t-SNE 可视化显示 QMamba 对不同失真类型的特征分离度远优于 CNN/ViT/Swin
2. **局部扫描的关键作用**：对于 TID2013（24类失真）、KADID（25类失真）等复杂数据集，LQMamba 的改进尤为显著（如 TID2013 SRCC: 0.964 vs QMamba 的 0.949）
3. **StylePrompt 设计精巧**：基于"域偏移≈特征风格偏移"的洞察，仅通过调节均值/方差即可高效迁移，3.83M 参数击败需 28.33M 的 Conv_Prompt
4. **效率优势明显**：QMamba-T 仅 4.47G FLOPs 即超越 17.58G 的 ViT-B 和 15.47G 的 Swin-B

## 局限与展望

1. **仅验证图像质量评估**：未扩展到视频质量评估（VQA）或音频质量评估，SSM 的序列建模特性理应适合此类任务
2. **数据集规模有限**：IQA 领域标注数据稀缺，未探索大规模预训练的效果
3. **分辨率受限**：实验中统一裁剪为 224×224，未探索原始分辨率或多分辨率输入的效果
4. **StylePrompt 仅调节一阶/二阶统计量**：对于更复杂的域偏移（如内容分布差异），可能需要更丰富的适配策略
5. **缺少与近期 Mamba 变体的对比**：如 Mamba-2 等更新架构

## 相关工作与启发

- **VMamba / LocalMamba**：QMamba 的骨干设计基础，局部窗口扫描灵感来源
- **DEIQT**：训练策略参考，Transformer-based IQA 强基线
- **SSF (Scale & Shift Feature)**：StylePrompt 的前身，但 SSF 学习固定仿射参数缺乏输入自适应性
- **研究启发**：Mamba 在低层感知任务上的优势暗示其在图像超分辨率、去噪、增强等任务中也有潜力

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将 Mamba 引入 IQA，局部扫描 + StylePrompt 设计有创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 10个数据集、三大任务、多种消融，实验非常全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，分析透彻，t-SNE 可视化有说服力
- 价值: ⭐⭐⭐⭐ — 为 SSM 在低层视觉感知的应用开辟了新方向
