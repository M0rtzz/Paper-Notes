---
title: >-
  [论文解读] SegPoint: Segment Any Point Cloud via Large Language Model
description: >-
  [ECCV 2024][3D视觉][图像分割] 提出 SegPoint，首个利用多模态 LLM 推理能力在统一框架中完成 3D 指令分割、引用分割、语义分割和开放词汇分割四种任务的模型，并构建 Instruct3D 基准测试（2,565 对），mIoU 达 27.5%。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "图像分割"
  - "LLM"
  - "unified framework"
  - "geometric feature"
---

# SegPoint: Segment Any Point Cloud via Large Language Model

**会议**: ECCV 2024  
**arXiv**: [2407.13761](https://arxiv.org/abs/2407.13761)  
**代码**: [https://heshuting555.github.io/SegPoint](https://heshuting555.github.io/SegPoint)  
**领域**: 3D视觉 / 点云分割  
**关键词**: 3D point cloud segmentation, LLM, unified framework, instruction segmentation, geometric feature

## 一句话总结
提出 SegPoint，首个利用多模态 LLM 推理能力在统一框架中完成 3D 指令分割、引用分割、语义分割和开放词汇分割四种任务的模型，并构建 Instruct3D 基准测试（2,565 对），mIoU 达 27.5%。

## 研究背景与动机
**领域现状**：3D 点云分割已有大量进展（Mask3D、SPFormer 等），但每个模型通常只解决一种特定分割任务，缺乏统一性。

**现有痛点**：(a) 现有方法依赖预定义类别或显式文本描述，无法理解隐含的人类意图（如"哪里可以坐？"）；(b) 不同任务需要不同模型，效率低且不实用。

**核心矛盾**：真实场景需要模型能理解隐含指令并推理，但当前方法缺乏推理能力；同时需要统一框架处理多种分割任务。

**切入角度**：利用 LLM 的推理和世界知识来理解复杂/隐含指令，结合几何增强模块弥补点云编码器在密集预测上的不足。

**核心 idea**：将 LLM 的推理能力注入 3D 点云分割，通过几何增强和特征传播实现高质量逐点分割。

## 方法详解

### 整体框架
SegPoint 由四个核心部分组成：(1) 预训练点云编码器 $\mathcal{E}$（Uni3D）用于提取点云特征；(2) 大语言模型 $\mathcal{F}$（LLaMA2-7B）提供推理能力；(3) 几何增强模块（GEM）$\mathcal{G}$ 提取局部几何信息并注入编码器；(4) 几何引导特征传播（GFP）$\mathcal{P}$ 生成高质量逐点嵌入用于精确分割。

输入为点云 $\vec{i}_{point} \in \mathbb{R}^{N \times (3+F)}$ 和文本指令 $\vec{i}_{txt}$，LLM 输出中的 `<SEG>` token 对应的嵌入与逐点特征做点积得到分割掩码。

### 关键设计

#### 1. **Vanilla Baseline 及其问题**
    - 功能：直接将点云编码器特征送入 LLM，检测 `<SEG>` token 后生成掩码嵌入 $\vec{h}_{seg} = \gamma(\vec{y}_{[seg]})$，与上采样后的逐点嵌入做点积得到掩码 $\vec{m} = \vec{h}_{seg} \otimes \text{UpS.}(\vec{f}_{point})$
    - 存在问题：(a) 点云编码器为场景级分类训练，不适合密集预测；(b) FPS 采样从 $N$ 降到 $N_1$ 丢失细节；(c) 从 $N_1$ 直接上采样到 $N$ 引入大量噪声
    - 设计动机：明确了两个核心瓶颈——局部几何信息缺失和上采样质量差

#### 2. **几何增强模块 (Geometric Enhancer Module, GEM)**
    - 功能：提取全场景局部几何上下文，通过交叉注意力注入点云编码器的中间特征
    - 核心思路：
      - GEM 由 3 个 KPConv + BN + ReLU 块组成，输出几何特征 $\vec{g}_f \in \mathbb{R}^{N \times D}$，保留所有 $N$ 个点的信息
      - 通过交叉注意力将几何特征注入编码器的每个 block：$\hat{\vec{f}_i} = \vec{f}_i + g_i \cdot \text{softmax}\left(\frac{\vec{f}_i \vec{g}_f^T}{\sqrt{D}}\right) \vec{g}_f$
      - 可学习门控因子 $g_i$ 初始为零，确保不会突然改变预训练权重的特征分布
    - 设计动机：KPConv 天然适合提取局部 3D 几何信息（vs 普通线性层）；门控因子保护预训练权重；类似 2D 中 ConvStem 增强 ViT 捕获局部信息的思路

#### 3. **几何引导特征传播 (Geometric-guided Feature Propagation, GFP)**
    - 功能：从稀疏点特征高质量上采样到密集逐点嵌入
    - 核心思路：
      - 高层特征 $\vec{f}_3, \vec{f}_4$ 通过 PointNet++ 传播上采样到 $N_3, N_2$ 个点
      - 几何特征 $\vec{g}_f$ 通过 FPS 下采样到相同数量的点
      - 上/下采样特征拼接后通过 FC + ReLU 融合
      - 最后一层特征 $\vec{f}_5$ 与 LLM 输出的隐层嵌入拼接，感知多模态信息
      - **Attentive Propagation**：使用交叉注意力实现不同点密度间的信息交换：$\hat{\tilde{\vec{f}}}_4 = \tilde{\vec{f}}_4 + \text{softmax}\left(\frac{\tilde{\vec{f}}_4 \vec{f}_{54}^T}{\sqrt{D}}\right)\vec{f}_{54}$
    - 设计动机：避免直接上采样导致的信息丢失；几何特征作为"黄金信息"引导上采样过程

#### 4. **任务统一与 Instruct3D 数据集**
    - 功能：通过任务特定提示（task-specific prompts）在统一模型中处理4种分割任务
    - 语义分割模板："Can you segment the {category} in this point cloud?" → "{category} \<SEG\>"
    - 引用分割模板："Can you segment the object {description}?" → "{category} \<SEG\>"
    - Instruct3D 包含 2,565 对指令-点云对，来自 ScanNet++ 的 280 个场景，支持多目标和零目标场景
    - 设计动机：隐含指令需要推理能力（如"哪里可以坐?"→分割椅子），现有数据集不支持

### 损失函数 / 训练策略
总损失：$\mathcal{L} = \lambda_{txt}\mathcal{L}_{txt} + \lambda_{bce}\mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice}$

- $\mathcal{L}_{txt}$：自回归交叉熵损失（文本生成）
- $\mathcal{L}_{bce}$：二元交叉熵损失（分割掩码）
- $\mathcal{L}_{dice}$：DICE 损失（分割掩码）
- 权重：$\lambda_{txt}=1.0, \lambda_{bce}=2.0, \lambda_{dice}=2.0$
- 所有数据集联合训练，评估时在特定数据集上微调

## 实验关键数据

### 指令分割（Instruct3D）

| 阶段 | 方法 | Acc | mIoU |
|------|------|-----|------|
| Two-stage | ScanRefer | 12.0 | 6.9 |
| Two-stage | M3DRef-CLIP | 18.1 | 12.8 |
| Single | BUTD-DETR* | 16.3 | 10.9 |
| Single | EDA* | 16.6 | 12.1 |
| Single | SegPoint† (vanilla) | 21.8 | 16.1 |
| Single | **SegPoint** | **31.6** | **27.5** |

SegPoint 相比最优 baseline 提升 **+14.7 mIoU** (27.5 vs 12.8)。

### 语义分割

| 方法 | ScanNet | ScanNet200 | S3DIS |
|------|---------|-----------|-------|
| PTv2 | 75.4 | 30.2 | 71.6 |
| OctFormer | 75.7 | 32.6 | - |
| Swin3D | 75.5 | - | 72.5 |
| **SegPoint** | 74.1 | **35.3** | 72.4 |

在类别丰富的 ScanNet200 上超越 SOTA +2.7% mIoU。

### 引用分割

| 方法 | ScanRefer | Nr3D | Multi3DRefer |
|------|-----------|------|-------------|
| M3DRef-CLIP | 35.7 | 27.0 | 32.6 |
| 3D-STMN | 39.5 | - | - |
| RefMask3D | 44.8 | - | - |
| **SegPoint** | 41.7 | **32.2** | **36.1** |

Multi3DRefer 上超越对手 +3.5 mIoU。

### 消融实验

| GEM | GFP | Instruct3D mIoU | ScanRefer mIoU |
|-----|-----|-----------------|----------------|
| ✗ | ✗ | 16.1 | 30.3 |
| ✓ | ✗ | 21.4 | 35.8 |
| ✗ | ✓ | 23.2 | 38.1 |
| ✓ | ✓ | **27.5** | **41.7** |

### 关键发现
- GEM 和 GFP 各自独立贡献显著（+5.3/+7.1 mIoU），组合后互补效果更强
- GEM 优于全量微调、LoRA 和 MLP adapter，说明提升不仅来自参数量增加
- 即使 vanilla baseline（SegPoint†）也优于所有现有方法 → 验证了 LLM 作为分割引擎的有效性
- 开放词汇分割中超越多个监督方法（ScanNet++ 19.3 vs KPConv 30.0 监督），展示强泛化能力

## 亮点与洞察
- **统一框架的优雅性**：首次在单一模型中统一四种 3D 分割任务，通过任务特定提示切换，无需为每种任务设计独立模型。这一思路与 2D 领域的统一分割模型（如 SAM）方向一致。
- **几何增强的必要性**：预训练点云编码器（如 Uni3D）是为场景级分类设计的，直接用于密集预测效果差。GEM 通过 KPConv 提取局部几何 + 门控注入的方式，以最小代价适配密集预测。
- **Instruct3D 开创新任务**：将 3D 分割从显式描述扩展到隐含指令理解，需要世界知识和推理能力，推动了更智能的 3D 感知系统发展。

## 局限与展望
- 当前框架仅支持文本提示，不支持点/框等非文本提示（类似 SAM 的 prompt 模式）
- 训练需要 4×A100 约 3 天，计算成本较高
- Instruct3D 数据集规模较小（仅 2,565 对），未来需要扩展
- 开放词汇分割需借助 GPT-4 做类别名匹配，引入额外依赖

## 相关工作与启发
- **vs LISA (2D)**：SegPoint 借鉴了 LISA 将 `<SEG>` token 注入 LLM 的范式，但不依赖 SAM 等额外预训练分割模型，直接端到端生成掩码
- **vs Mask3D/SPFormer**：这些 Transformer 方法在基础分割上很强，但无法处理语言交互和隐含指令
- **vs 3D-LLM/PointLLM**：这些方法主要关注场景级理解，缺乏逐点级别的精细分割能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个统一四种3D分割任务 + Instruct3D新任务
- 实验充分度: ⭐⭐⭐⭐ 四个任务多个benchmark + 详细消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，模块设计动机充分
- 价值: ⭐⭐⭐⭐ 推动3D分割向统一智能框架转变

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] PointLLM: Empowering Large Language Models to Understand Point Clouds](pointllm_empowering_large_language_models_to_understand_point_clouds.md)
- [\[ECCV 2024\] Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration](equi-gspr_equivariant_se3_graph_network_model_for_sparse_point_cloud_registratio.md)
- [\[CVPR 2026\] AnyPcc: Compressing Any Point Cloud with a Single Universal Model](../../CVPR2026/3d_vision/anypcc_compressing_any_point_cloud_with_a_single_universal_model.md)
- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)

</div>

<!-- RELATED:END -->
