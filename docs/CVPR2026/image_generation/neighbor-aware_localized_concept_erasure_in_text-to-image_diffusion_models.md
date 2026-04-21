---
title: >-
  [论文解读] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models
description: >-
  [CVPR 2026][图像生成][概念擦除] 提出 NLCE，一个 training-free 的三阶段概念擦除框架，通过谱加权表征调制、注意力引导空间门控和门控特征清理三步实现目标概念的精确局部擦除，同时显式保留语义邻近概念，在 Oxford Flowers、Stanford Dogs、名人身份和敏感内容擦除任务上均优于现有方法。
tags:
  - CVPR 2026
  - 图像生成
  - 概念擦除
  - 扩散模型
  - 邻居保留
  - 训练无关
  - 局部擦除
---

# Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2603.25994](https://arxiv.org/abs/2603.25994)  
**代码**: [https://github.com/alirezafarashah/NLCE](https://github.com/alirezafarashah/NLCE)  
**领域**: 图像生成 / AI安全  
**关键词**: 概念擦除, 扩散模型, 邻居保留, 训练无关, 局部擦除

## 一句话总结

提出 NLCE，一个 training-free 的三阶段概念擦除框架，通过谱加权表征调制、注意力引导空间门控和门控特征清理三步实现目标概念的精确局部擦除，同时显式保留语义邻近概念，在 Oxford Flowers、Stanford Dogs、名人身份和敏感内容擦除任务上均优于现有方法。

## 研究背景与动机

1. **领域现状**：T2I 扩散模型的概念擦除方法分为 training-based（ESD、MACE、SPM 等需要微调）和 training-free（UCE、RECE、GLoCE 等仅推理时修改）两大类。局部擦除方法（GLoCE）试图仅在目标区域进行编辑。
2. **现有痛点**：**邻居间隙（Neighbor Gap）**——擦除一个细粒度概念时，语义相近的概念也被意外削弱。例如擦除某个狗品种时，其他品种的生成质量也会下降。
3. **核心矛盾**：概念表征在嵌入空间中高度纠缠，简单的投影/抑制操作无法精确区分目标和邻居。
4. **本文目标** 在不训练的情况下，精确擦除目标概念的同时保留邻近概念的语义完整性。
5. **切入角度**：三阶段渐进式擦除——先在表征空间做谱加权调制削弱目标+增强邻居，再用注意力定位残留，最后硬擦除清理。
6. **核心 idea**：显式建模和保护"概念邻域"结构，实现精准而非粗暴的概念移除。

## 方法详解

### 整体框架

NLCE 在推理时对 UNet 的 cross-attention 层做三阶段干预，不修改模型权重（training-free）。Stage 1 在嵌入空间操作（改 Key/Value 投影矩阵），Stage 2 和 Stage 3 在每个去噪步的特征空间操作。

### 关键设计

1. **Stage 1: 表征空间调制（谱加权抑制 + 邻居增强）**:

    - 功能：在嵌入层面削弱目标概念的语义同时恢复邻近概念的表征
    - 核心思路：对目标概念嵌入做 SVD 分解得到正交基 $U_{F_c}$，构造谱加权投影 $P_{F_c} = U_{F_c}\Lambda_{F_c}U_{F_c}^T$，其中权重 $\lambda_i$ 按奇异值重要性调节（重要方向更强抑制）。类似地为邻居概念构造 $P_{\mathcal{N}_c}$。最终算子 $P_c = (I - \beta P_{F_c}) + \gamma P_{\mathcal{N}_c} P_{F_c}$，全局应用到 $W_K, W_V$。邻居通过 Wikipedia 检索 + RoBERTa 具体度过滤 + CLIP 视觉相似度排序获得
    - 设计动机：不同于 GLoCE 的 gated low-rank adapter（可能漏掉间接注意力路径中的概念重激活），全局应用更可靠。谱加权让抑制强度与语义重要性成正比

2. **Stage 2: 注意力引导空间门控**:

    - 功能：定位仍有目标概念残留激活的空间区域
    - 核心思路：每个去噪步做两次前向（dry pass + real pass）。第一次提取 DownBlock-2 的注意力图。检测哪些 token 仍与目标子空间有较高重叠（$s_j = \|P_{F_c}x_j\|_2 > \delta_{\text{token}}$），标记为 "live token"。对 live token 的注意力求和得到空间门控图 $G_t(x,y)$。第二次前向中，对 live token 在门控区域抑制注意力：$A^\ell(x,y,j) \leftarrow (1-G_t)\cdot A^\ell(x,y,j)$
    - 设计动机：Stage 1 是全局操作可能有残留，Stage 2 用空间注意力精确定位"哪里还有目标概念的影子"

3. **Stage 3: 门控特征硬擦除**:

    - 功能：在门控区域内彻底清除残余目标信号
    - 核心思路：将 Stage 2 的门控图上采样到各 UNet 层分辨率，二值化（阈值 $\delta_{\text{scrub}}$），在 mask=1 的位置直接将隐特征置零：$h_t^\ell(x,y) \leftarrow \mathbf{0}$。这是不可逆的硬擦除
    - 设计动机：投影式抑制理论上仍可能被恢复，硬置零保证严格安全性

### 损失函数 / 训练策略

无训练过程。所有操作在推理时进行。关键超参数：
- $\beta, \gamma \in [0,1]$：控制目标抑制和邻居增强强度
- $\delta_{\text{token}}$：live token 检测阈值
- $\delta_{\text{scrub}}$：硬擦除门控阈值
- 多概念场景下按 prompt 检测激活对应概念的算子，组合为 $P_{\text{multi}} = \prod_{c\in\mathcal{A}} P_c$

## 实验关键数据

### 主实验

**Oxford Flowers / Stanford Dogs 细粒度擦除**：

| 方法 | Alpine Sea Holly Acc_t↓/Acc_r↑/Ho↑ | Bluetick Acc_t↓/Acc_r↑/Ho↑ |
|------|-----------------------------------|---------------------------|
| GLoCE | 32.0/78.91/73.05 | 28.0/73.59/72.79 |
| RECE | 0.0/64.85/78.68 | 0.0/73.33/84.62 |
| **NLCE** | **0.0/82.06/90.15** | **0.0/75.91/86.31** |

**名人身份擦除**：

| 方法 | Anna Kendrick Acc_t↓/Ho↑ | Elon Musk Acc_t↓/Ho↑ |
|------|--------------------------|----------------------|
| SLD | 0.0/96.55 | 3.33/94.28 |
| GLoCE | 1.33/96.63 | 0.67/97.29 |
| **NLCE** | **0.0/96.91** | **0.0/96.55** |

### 消融实验

Stage 逐步添加的效果（从论文 Figure 9 提取）：

| 配置 | 效果趋势 |
|------|---------|
| 仅 Stage 1 | 基本擦除，但可能有残留 |
| Stage 1+2 | 擦除更彻底，空间精准 |
| Stage 1+2+3 | 完全擦除，无残留 |

不同数据集对三阶段的依赖程度不同：简单场景 Stage 1 已足够，复杂场景需完整 pipeline。

### 关键发现

- NLCE 在所有细粒度数据集上获得最高 Acc_r 和 Ho，说明邻居保留效果最好
- GLoCE 的 Acc_t 依然较高（如 32%），说明轻量编辑不够彻底；NLCE 几乎全部降到 0%
- 在 I2P 敏感内容擦除中 NLCE 检测到的裸露内容最少，同时 CLIP Score 29.70 保持较高
- 多概念同时擦除（10 个品种）时 NLCE 仍保持高 Acc_r，而 MACE/UCE/RECE 等的保留准确率崩溃
- KID 值普遍最低，说明视觉质量保持最好

## 亮点与洞察

- **"邻居间隙"问题的发现和形式化**很好地解释了为什么现有方法在细粒度场景下失败。这一洞察对概念擦除领域有普遍意义
- **三阶段渐进式擦除设计**把概念擦除从"一刀切"变成了"精确手术"：先在表征空间削弱，再在注意力空间定位，最后在特征空间清除。每个阶段的设计目标清晰
- **邻居挖掘管线**（Wikipedia 检索 → RoBERTa 具体度过滤 → CLIP 视觉排序）是一套实用的语义邻域构建方法，可复用于其他需要概念边界划定的任务

## 局限与展望

- 每步去噪做两次前向（dry pass + real pass），推理时间翻倍
- 邻居挖掘依赖外部资源（Wikipedia、RoBERTa），对罕见概念可能检索不到合适邻居
- 硬擦除（置零）可能在某些情况下导致局部视觉瑕疵
- $\beta, \gamma$ 需要根据擦除强度需求手动调节，不同场景最优值不同

## 相关工作与启发

- **vs GLoCE**: 同为局部擦除方法，GLoCE 用 gated low-rank adapter 但不显式保护邻居。NLCE 在 neighbor 保留上显著更好（Acc_r 差距 3-7%），且 Acc_t 更低
- **vs RECE**: 擦除彻底但邻居遗忘严重（Acc_r 经常比 NLCE 低 10-15%），因为没有邻居保护机制
- **vs AdaVD**: 谱抑制方法但无空间局部化和邻居增强，在多概念场景下不够稳健

## 评分

- 新颖性: ⭐⭐⭐⭐ 邻居感知的概念擦除是很好的问题抽象，三阶段设计合理；但各阶段技术本身（SVD投影、注意力门控）并不新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖细粒度/名人/敏感内容/艺术风格四大场景，多概念扩展，消融完整
- 写作质量: ⭐⭐⭐⭐ 问题引入清晰，但三阶段描述偏重符号，实际算法流程可以更直观
- 价值: ⭐⭐⭐⭐ 对 T2I 模型的安全部署有直接意义，特别是细粒度概念管控场景

<!-- RELATED:START -->

## 相关论文

- [Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](../../ICLR2026/image_generation/localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)
- [Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)

<!-- RELATED:END -->
