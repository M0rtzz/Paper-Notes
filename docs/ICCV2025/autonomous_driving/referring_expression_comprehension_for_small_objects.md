---
title: >-
  [论文解读] Referring Expression Comprehension for Small Objects
description: >-
  [ICCV 2025][自动驾驶][指称表达理解] 提出 SOREC 数据集（10万对小目标指称表达和 bounding box）和 PIZA 适配器模块（渐进式迭代缩放），使 GroundingDINO 等预训练模型能以自回归方式逐步放大定位极小目标，在自动驾驶场景中大幅提升小目标 REC 精度。
tags:
  - ICCV 2025
  - 自动驾驶
  - 指称表达理解
  - 小目标检测
  - 参数高效微调
  - 渐进缩放
  - GroundingDINO
---

# Referring Expression Comprehension for Small Objects

**会议**: ICCV 2025  
**arXiv**: [2510.03701](https://arxiv.org/abs/2510.03701)  
**代码**: [GitHub](https://github.com/mmaiLab/sorec)  
**领域**: 自动驾驶  
**关键词**: 指称表达理解, 小目标检测, 参数高效微调, 渐进缩放, GroundingDINO

## 一句话总结

提出 SOREC 数据集（10万对小目标指称表达和 bounding box）和 PIZA 适配器模块（渐进式迭代缩放），使 GroundingDINO 等预训练模型能以自回归方式逐步放大定位极小目标，在自动驾驶场景中大幅提升小目标 REC 精度。

## 研究背景与动机

指称表达理解（REC）旨在根据自然语言描述定位图像中的特定目标。当前 SOTA 模型在 RefCOCO 等标准数据集上已达 90% 以上准确率，但在**极小目标定位**上仍面临重大挑战：

**缺乏数据集**：现有 REC 数据集（RefCOCO 等）主要包含中等及以上尺寸的目标，缺少针对小目标的标注。SOREC 中目标 bounding box 平均仅占图像面积的 0.05%——远小于常规目标。

**预训练-微调鸿沟**：大规模预训练模型（如 GroundingDINO）在正常尺寸目标上表现优异，但面对极小目标时，低分辨率特征无法提供足够的细粒度信息。直接缩放图像到模型输入大小会丢失小目标的关键细节。

**应用需求紧迫**：自动驾驶中远处的行人、交通标志、路灯等小目标的检测对安全至关重要，但现有方法在这些场景下表现不佳。

核心洞察：人类面对小目标时会**先粗定位再逐步放大**。PIZA 模仿这一行为，通过自回归式缩放逐步聚焦目标。

## 方法详解

### 整体框架

PIZA（Progressive-Iterative Zooming Adapter）将小目标定位建模为搜索过程。给定预训练模型 $F$，PIZA 扩展其为 $F_{\bigcirc *}$，以自回归方式预测一序列 bounding box：

$$P = (\mathbf{b}_0, \mathbf{b}_1, \cdots, \mathbf{b}_T)$$

其中 $\mathbf{b}_0$ 覆盖整张图像，$\mathbf{b}_T$ 为最终定位的小目标。每步预测：

$$\hat{\mathbf{b}}_{i+1} = F_{\bigcirc *}(\mathbf{x}_i, \mathbf{t}, \mathbf{b}_{0:i})$$

$\mathbf{x}_i$ 是第 $i$ 步裁剪后的图像区域。

### 关键设计

1. **PIZA 模块（缩放步嵌入）**：受扩散模型时间步嵌入的启发，PIZA 学习"缩放步嵌入"来表示搜索过程的进展。具体流程：

    - 从 bounding box 序列 $\mathbf{b}_{0:i}$ 提取 6 维低级特征（归一化尺寸、相对尺寸、归一化宽高、中心坐标）
    - 通过可学习 Fourier 嵌入 + Transformer 编码器 + 平均池化提取嵌入 $\mathbf{h} \in \mathbb{R}^d$
    - 两个预测头：EOS 头预测是否停止搜索（二分类），Progress 头预测搜索进度 $\hat{z} \in [0,1]$
    - 总参数量仅 0.27M，特征维度为 16

2. **三种参数高效微调集成方式**：PIZA 可灵活集成到不同 PEFT 方法中：

    - **PIZA-CoOp**：将缩放步嵌入 $\mathbf{h}$ 插入到文本 prompt 的可学习嵌入序列中
    - **PIZA-LoRA**：将 $\mathbf{h}$ 注入 LoRA 的瓶颈层：$W\mathbf{x} + BA\mathbf{x} + BC\mathbf{h}$
    - **PIZA-Adapter+**：将 $\mathbf{h}$ 加到 Adapter+ 的通道缩放层输出上

3. **扩展训练数据集构建**：为训练自回归缩放过程，需要构建搜索过程的 ground truth。核心步骤：

    - 从预训练数据集估计 bounding box 面积比分布 $p(r)$
    - 使用指数加权方案确定每步缩放因子 $z_j^*$，兼顾初始步的精确度
    - 宽高比从全图逐步过渡到目标的宽高比
    - 生成 [CONT] 和 [EOS] 标签用于训练停止决策

### 损失函数 / 训练策略

- 基于 GroundingDINO 的对比损失和定位损失
- 每个 mini-batch 随机采样搜索过程中的某一步进行前向计算
- AdamW 优化器，学习率 $2 \times 10^{-4}$，epoch 3 时衰减 0.5，共训练 5 epochs
- LoRA 应用于所有 self-attention 和 cross-attention 模块，rank=16
- Adapter+ 插入每个 self-attention 和 FFN 模块后

## 实验关键数据

### 主实验

**SOREC 数据集上的参数高效微调结果（Train-L）**

| 方法 | 参数量 | Val mAcc | Test-A mAcc | Test-B mAcc | Test-A Acc50 | Test-B Acc50 |
|------|--------|---------|------------|------------|-------------|-------------|
| Zero-shot | 0 | 0.2 | 0.3 | 0.0 | 1.0 | 0.2 |
| Full fine-tuning | 173.0M | 37.4 | 43.8 | 30.5 | 69.6 | 55.6 |
| LoRA | 1.3M | 25.2 | 30.7 | 19.7 | 50.2 | 37.3 |
| PIZA-LoRA | 1.5M | 34.5 | 39.3 | 29.0 | 54.0 | 43.4 |
| Adapter+ | 3.3M | 34.6 | 40.7 | 27.6 | 65.9 | 51.3 |
| **PIZA-Adapter+** | 3.5M | **39.0** | **45.1** | **31.7** | **66.2** | **52.2** |

### 消融实验

**PIZA 各组件贡献（PIZA-Adapter+, Train-S, mAcc/Acc50/Acc75）**

| 配置 | Val | Test-A | Test-B | 说明 |
|------|-----|--------|--------|------|
| w/o PIZA module | 26.0/48.1/24.8 | 32.0/55.0/33.3 | 20.3/40.4/17.9 | 无缩放：基线 |
| w/o emb. insertion | 36.7/53.2/41.7 | 42.8/59.2/49.9 | 30.3/45.8/34.0 | 无嵌入注入 |
| **Full PIZA-Adapter+** | **36.8/53.5/41.8** | **43.1/59.6/50.1** | **30.4/45.9/34.1** | 完整模型 |

**Adapter 瓶颈维度影响**

| 维度 d | 参数量 | Val mAcc | Test-A mAcc |
|--------|--------|---------|------------|
| 32 | 1.6M | 35.1 | 40.8 |
| 64 | 1.9M | 36.6 | 42.2 |
| 128 | 2.4M | 36.4 | 41.8 |
| 256 | 3.5M | **36.8** | **43.1** |

### 关键发现

- Zero-shot 基线近乎为零（mAcc 0.2%），说明预训练模型完全无法处理极小目标
- PIZA-Adapter+ 以仅 3.5M 参数（vs 173M）超越全微调基线，体现渐进缩放的强大能力
- 去除 PIZA 模块后性能从 36.8 暴跌至 26.0（mAcc），确认自回归缩放是核心
- Test-A（交通标志等）比 Test-B（其他小目标）准确率高约 10+%
- 更大训练集持续提升性能，表明数据集可进一步扩展
- 平均 2.11 步缩放即可完成定位

## 亮点与洞察

- **数据集贡献突出**：SOREC 是首个面向自动驾驶小目标的 REC 数据集，目标面积仅 0.05%，表达平均 25.5 词（RefCOCO 仅 3.5 词），填补了重要空白
- **模仿人类搜索策略**：渐进缩放的思路自然且高效，类似人类"先环顾再聚焦"的视觉搜索行为
- **参数高效**：PIZA 模块仅 0.27M 参数，通过灵活嵌入缩放步信息到不同 PEFT 框架，实现了少量参数的大幅性能提升
- **数据集构建流程可复制**：半自动化构建流程（SAM+GPT-4o+众包），为其他场景创建类似数据集提供了范例

## 局限与展望

- 自回归缩放增加推理次数（平均 2-3 步），对实时性要求高的场景可能不够快
- 目前仅在 GroundingDINO 上验证，其他基础模型（如 GLIPv2、Florence）的迁移性待验证
- 数据集由 GPT-4o 生成表达，可能存在语言多样性不足的问题（18.45% 有小错误）
- 缩放步数的自动决策（EOS 预测）的可靠性有待进一步研究
- 可探索与多尺度特征提取等传统小目标检测方法的结合

## 相关工作与启发

- 将 REC 从"一步定位"重新定义为"多步搜索"问题，提供了处理极端尺度目标的新范式
- PIZA 的缩放步嵌入借鉴扩散模型时间步嵌入的思路，跨领域迁移巧妙
- SOREC 的构建流程（基础模型分割→GPT 生成描述→众包验证）为自训练数据集生产提供参考
- 对自动驾驶场景中的远距离目标理解和安全规划具有直接意义

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 渐进缩放定位的思路新颖直觉，数据集填补重要空白
- **实验充分度**: ⭐⭐⭐⭐ — 多种 PEFT 方法对比，消融充分，但缺乏与其他 REC 方法的对比
- **写作质量**: ⭐⭐⭐⭐⭐ — 论文结构清晰，动机阐述到位，数据集构建描述详尽
- **价值**: ⭐⭐⭐⭐ — 数据集和方法对小目标理解领域均有重要贡献

<!-- RELATED:START -->

## 相关论文

- [Counting Stacked Objects](counting_stacked_objects.md)
- [Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels](../../CVPR2025/autonomous_driving/learning_to_detect_objects_from_multi-agent_lidar_scans_without_manual_labels.md)
- [SABER: Spatially Consistent 3D Universal Adversarial Objects for BEV Detectors](../../CVPR2026/autonomous_driving/saber_spatially_consistent_3d_universal_adversarial_objects_for_bev_detectors.md)
- [TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking](trackany3d_transferring_pretrained_3d_models_for_category-unified_3d_point_cloud.md)
- [LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)

<!-- RELATED:END -->
