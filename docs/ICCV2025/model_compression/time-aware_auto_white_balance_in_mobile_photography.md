---
title: >-
  [论文解读] Time-Aware Auto White Balance in Mobile Photography
description: >-
  [ICCV 2025][模型压缩][自动白平衡] 本文提出一种利用手机上下文元数据（时间戳和地理位置）辅助图像颜色信息的轻量化光照估计方法（约 5K 参数），在自建 3224 张智能手机数据集上达到或超过大模型性能，且可在旗舰手机 DSP 上 0.25ms 内完成推理。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "自动白平衡"
  - "光照估计"
  - "轻量化模型"
  - "移动摄影"
  - "上下文元数据"
---

# Time-Aware Auto White Balance in Mobile Photography

**会议**: ICCV 2025  
**arXiv**: [2504.05623](https://arxiv.org/abs/2504.05623)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 自动白平衡, 光照估计, 轻量化模型, 移动摄影, 上下文元数据

## 一句话总结
本文提出一种利用手机上下文元数据（时间戳和地理位置）辅助图像颜色信息的轻量化光照估计方法（约 5K 参数），在自建 3224 张智能手机数据集上达到或超过大模型性能，且可在旗舰手机 DSP 上 0.25ms 内完成推理。

## 研究背景与动机
自动白平衡（AWB）是相机管线中纠正场景光照导致的色偏的关键步骤。传统方法仅依赖图像颜色信息（raw 图像或颜色直方图）来估计光照色。

然而，手机设备拥有大量额外的上下文元数据——捕获时间戳和地理位置——这些为缩小光照估计的搜索空间提供了强有力的线索。例如：
- **正午**拍摄的照片具有不同于**日落**拍摄照片的色温（CCT）范围
- **室内**与**室外**场景的光照条件截然不同
- ISO、快门速度等拍摄参数可辅助区分环境明暗

核心矛盾在于：现有白平衡数据集（NUS、Cube++ 等）大多使用 DSLR 相机采集，缺少时间戳和地理位置等上下文信息；且现有方法未探索将这些上下文信息整合到光照估计中的可能性。

本文的切入角度是：将拍摄时间转换为太阳事件概率向量（黎明、日出、正午、日落、黄昏、午夜），结合 ISO/快门/闪光灯等拍摄信息和图像颜色直方图，训练一个极轻量化的模型。

## 方法详解

### 整体框架
模型接收两个输入分支：(1) 时间-拍摄特征（time-capture feature），处理上下文和拍摄元数据生成 $\mathbf{v}_t \in \mathbb{R}^{16}$；(2) 直方图特征（histogram feature），处理 raw 图像的 R/G 和 B/G 色度直方图生成 $\mathbf{v}_h \in \mathbb{R}^{16}$。两个特征拼接后通过 MLP 输出光照色度 $\ell_c \in \mathbb{R}^2$，再归一化为 RGB 光照向量。

### 关键设计
1. **时间-拍摄特征（Time-Capture Feature）**

    - 功能：将手机的时间戳和地理位置转换为对模型有用的时间概率向量
    - 核心思路：根据地理位置计算当地的太阳事件时间（dawn, sunrise, noon, sunset, dusk, midnight），然后计算拍摄时间与各事件的概率：$p_g = 1 - \frac{|t_c - t_g|}{t_s}$，其中 $t_s = 86400$ 秒。对概率取平方根以平衡分布，并增加 one-hot 向量标识拍摄时间在事件之前/之后
    - 额外拍摄特征：ISO（$i$）、快门速度（$s$）、闪光灯状态（$f$）、可选噪声信息（noise stats 或 SNR stats）
    - 最终特征：$\mathbf{c} = [\mathbf{p}^T, \log(i), \log(s), f, \mathbf{w}^T]^T$，通过线性层映射到 $\mathbf{v}_t \in \mathbb{R}^{16}$
    - 设计动机：将时钟时间转换为太阳事件概率而非直接使用小时数，使模型能跨时区泛化

2. **直方图特征（Histogram Feature）**

    - 功能：紧凑地表示 raw 图像的颜色分布
    - 核心思路：计算 R/G 和 B/G 的 2D 色度直方图 $\mathbf{H}_c \in \mathbb{R}^{48 \times 48}$，以像素亮度为权重。同时计算边缘图像的色度直方图 $\mathbf{H}_e$，取平方根增强特征
    - 最终特征 $\mathbf{H} \in \mathbb{R}^{48 \times 48 \times 4}$（颜色直方图 + 边缘直方图 + u/v 坐标通道），通过卷积网络 + ELU + 自适应平均池化 + 线性层映射到 $\mathbf{v}_h \in \mathbb{R}^{16}$
    - 设计动机：使用 R/G, B/G（而非传统的 log-uv）色度空间，因为与模型输出空间一致

3. **光照估计头**

    - 功能：融合两个分支特征并输出光照颜色
    - 核心思路：$\mathbf{v} = [\mathbf{v}_t; \mathbf{v}_h] \in \mathbb{R}^{32}$，经过带 BatchNorm 的多层 MLP 输出 R/G 和 B/G 色度，转换为归一化 RGB 光照向量
    - 整个模型仅约 5K 参数

### 损失函数 / 训练策略
- 使用角度误差（angular error）作为损失函数
- Adam 优化器，400 epochs，学习率从 $10^{-6}$ 预热到 $10^{-3}$，后续余弦退火
- 批量大小递增策略：从 8 开始，每 100 epoch 翻倍
- 数据集包含两种 ground truth：中性白平衡（色卡）和用户偏好白平衡（专家标注后用户研究验证）

## 实验关键数据

### 主实验
在自建数据集测试集上的结果（neutral / user-preference 格式）：

| 方法 | Mean↓ | Median↓ | Best25%↓ | Worst25%↓ | 参数量 |
|------|-------|---------|----------|-----------|--------|
| GW | 6.23/5.54 | 6.01/4.52 | 1.02/1.00 | 12.43/11.91 | - |
| FFCC | 2.62/1.50 | 1.46/0.81 | 0.37/0.24 | 6.89/3.99 | 12K |
| C4 | 1.92/1.49 | 1.30/0.90 | 0.36/0.24 | 4.64/3.82 | 5,116K |
| **Ours (w/o n,r)** | **1.93/1.26** | **1.35/0.77** | **0.38/0.23** | **4.56/3.13** | **4.83K** |
| **Ours (w/ n, w/ r)** | **1.84/1.20** | **1.24/0.77** | **0.35/0.19** | **4.41/2.95** | **5.03K** |

在 Simple Cube++ DSLR 数据集上也取得最优或次优结果，验证了方法的跨设备泛化能力。

### 消融实验

| 配置 | Mean↓ | Median↓ | Best25%↓ | 说明 |
|------|-------|---------|----------|------|
| Ours (w/o n, w/o r) | 1.93/1.26 | 1.35/0.77 | 0.38/0.23 | 仅时间+拍摄基础特征 |
| Ours (w/o n, w/ r) | 1.89/1.23 | 1.18/0.79 | 0.32/0.24 | +SNR stats |
| Ours (w/ n, w/o r) | 1.87/1.20 | 1.24/0.72 | 0.37/0.24 | +noise stats |
| Ours (w/ n, w/ r) | 1.84/1.20 | 1.24/0.77 | 0.35/0.19 | 两者兼具 |

### 关键发现
- 仅用 4.83K 参数（无噪声特征）即可匹配或超越 5116K 参数的 C4 方法
- 时间-拍摄特征的加入显著改善了室外场景的估计精度
- 噪声信息（stats 和 SNR）的加入都能进一步提升性能，两者互补
- 模型在旗舰手机 DSP 上运行仅需 0.25ms，CPU 上 0.80ms
- 用户偏好 GT 在 71.95% 的试验中被选择，验证了用户偏好标注的价值

## 亮点与洞察
- **极致的轻量化设计**：5K 参数实现 SOTA 性能，这在移动端部署上极具吸引力
- **上下文元数据的创新利用**：首次系统性地将手机时间戳和地理位置用于光照估计
- **数据集贡献**：3224 张智能手机 raw 图像，包含标准和用户偏好两种 GT，填补了移动端 AWB 数据集的空白
- **太阳事件概率编码**：将绝对时间转换为与太阳事件的相对概率，巧妙实现了时区无关的泛化

## 局限与展望
- 噪声信息的获取依赖于去噪参考图像或特定的 ISP 管线，通用性受限
- 数据集仅用单一品牌手机（Samsung S24 Ultra）采集，传感器特异性可能影响泛化
- 对于极端人工光照（彩色 LED）等场景，时间信息帮助有限
- 仅支持单一全局光源假设，多光源场景需通过掩码处理

## 相关工作与启发
- FFCC 之前尝试过使用元数据控制模型，但未深入利用时间和地理位置信息
- 本文的 time probability vector 编码方式可启发其他利用时间线索的视觉任务（如夜景增强、场景理解）
- 用户偏好白平衡 GT 的研究方法值得其他图像质量评估任务借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐ 上下文元数据利用的思路新颖，但整体方法较为直接
- 实验充分度: ⭐⭐⭐⭐⭐ 大量基线对比、消融实验、跨数据集验证、用户研究俱全
- 写作质量: ⭐⭐⭐⭐ 条理清晰，数据集描述详尽
- 价值: ⭐⭐⭐⭐ 对移动端 AWB 系统有直接工程价值，数据集贡献重要

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search](../../ECCV2024/model_compression/auto-das_automated_proxy_discovery_for_training-free_distillation-aware_architec.md)
- [\[ICCV 2025\] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](../../NeurIPS2025/model_compression/correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[ICCV 2025\] Scheduling Weight Transitions for Quantization-Aware Training](scheduling_weight_transitions_for_quantization-aware_training.md)
- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)

</div>

<!-- RELATED:END -->
