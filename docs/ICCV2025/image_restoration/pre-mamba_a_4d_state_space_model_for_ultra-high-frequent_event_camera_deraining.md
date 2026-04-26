---
title: >-
  [论文解读] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining
description: >-
  [ICCV 2025][图像恢复][事件相机去雨] 首个基于点的事件相机去雨框架，利用4D事件云表示和多尺度状态空间模型（MS3M），在保持微秒级时间精度的同时实现高效去雨，仅0.26M参数即达到SOTA性能。
tags:
  - ICCV 2025
  - 图像恢复
  - 事件相机去雨
  - 状态空间模型
  - Mamba
  - 点云处理
  - 时空建模
---

# PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining

**会议**: ICCV 2025  
**arXiv**: [2505.05307](https://arxiv.org/abs/2505.05307)  
**代码**: [https://github.com/softword-tt/PRE-Mamba](https://github.com/softword-tt/PRE-Mamba)  
**领域**: Image Restoration / Event Camera  
**关键词**: 事件相机去雨, 状态空间模型, Mamba, 点云处理, 时空建模

## 一句话总结

首个基于点的事件相机去雨框架，利用4D事件云表示和多尺度状态空间模型（MS3M），在保持微秒级时间精度的同时实现高效去雨，仅0.26M参数即达到SOTA性能。

## 研究背景与动机

事件相机以微秒级时间分辨率和高动态范围著称，但在雨天场景中会产生大量密集噪声——雨滴的快速运动触发大量强度变化事件，淹没有效场景信息。现有事件相机去雨方法面临三个核心矛盾：

**时间精度 vs 去雨效果**：基于帧的方法（如DistillNet）将事件流转换为灰度帧再去雨，牺牲了事件数据的原生时间分辨率和稀疏性

**去雨效果 vs 计算效率**：Point Transformer虽适合点数据但在高事件率（>10M events/s）下因二次注意力复杂度而不可行

**点方法 vs 帧方法的性能差距**：现有点方法（PointNet、GNN、SNN）性能仍低于帧方法

Mamba以线性复杂度和长程建模能力提供了有前景的解决方案，但将其引入事件去雨并非直接可行——原始Mamba是为因果语言任务设计的，无法直接处理异步、稀疏、高时间分辨率的事件数据。

## 方法详解

### 整体框架

PRE-Mamba将事件去雨建模为逐事件分类任务（类似去噪），而非重建任务，从而降低计算复杂度并避免伪影。框架包含四个核心组件：4D事件云表示、时空解耦与融合模块（STDF）、多尺度状态空间模型（MS3M）和频域正则化优化，整体采用U-Net风格的编码器-解码器结构。

### 关键设计

1. **4D事件云表示（4D Event Cloud）**: 将原始事件流 $\{e_i=(x_i,y_i,t_i,p_i)\}$ 划分为固定时间窗口，窗口内通过时间戳归一化构建3D伪点云 $p_i=(x_i,y_i,\frac{t_i-t_0}{t_e-t_0})$，跨窗口引入第四维 $T_n$（窗口全局索引），形成4D事件云。这种层级化双尺度框架通过归一化时间戳 $z$ 捕获窗口内局部动态，通过窗口索引 $T_n$ 建模窗口间全局演化。使用z-order和Hilbert曲线将4D事件云序列化为SSM兼容的序列。

2. **时空解耦与融合模块（STDF）**: 分别提取空间特征 $f_s$（通过深度1D卷积处理坐标）和时间特征 $f_t$（通过窗口索引嵌入）。关键创新是**以时间信息优先调制空间信息**：从1D卷积沿 $z$ 轴提取窗口内微秒级动态 $f_t^{\text{intra}}$，从1D卷积处理 $T_n$ 提取窗口间长期趋势 $f_t^{\text{inter}}$，通过Hadamard乘积调制空间特征：$f_s^* = f_s \otimes (f_t^{\text{intra}} + f_t^{\text{inter}})$。最终通过残差连接整合三路特征。

3. **多尺度状态空间模型（MS3M）**: 包含双时间尺度全局建模和多空间尺度局部路径：

    - **双时间尺度**：窗口内分支通过反向聚合和1D卷积提取空间外观特征 $f_s^{\text{intra}}$；窗口间分支提取运动感知特征 $f_m^{\text{inter}}$。自适应门控机制生成权重 $f_G$，通过交叉乘积注意力融合后送入SSM进行全局特征学习：$f_{\text{dual}} = \text{SSM}(\sigma(f_{\text{fuse}})) \otimes f_G$
    - **多空间尺度**：使用不同卷积核大小（1,3,5）捕获不同尺度的雨滴特征——小核捕获细雨丝，大核建模弥散雨模式。多尺度特征通过残差加法与全局输出融合

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{fft}$：
- **交叉熵损失** $\mathcal{L}_{ce}$：逐事件的二分类损失
- **频域正则化** $\mathcal{L}_{fft}$：不同于直接嵌入FFT层（对百万级事件开销过大），作为正则项在频域对齐预测与标签的分布。利用事件相机微秒级分辨率捕获雨滴动态——高密度降雨呈低频连续模式，低密度降雨呈高频稀疏模式

训练使用AdamW优化器，初始学习率4.8×10⁻⁴，6卡RTX A6000训练50个epoch，每次处理5个时间窗口（各0.1秒）。

## 实验关键数据

### 主实验

在自建EventRain-27K数据集（18K合成+9K真实）上评估，包含6种降雨强度（5~150mm/h）：

| 方法 | 类型 | 5mm SR/NR/DA | 20mm SR/NR/DA | 50mm SR/NR/DA | 150mm SR/NR/DA | 参数量 | 推理速度 |
|------|------|-------------|---------------|---------------|----------------|--------|----------|
| TS | 滤波 | 0.888/0.265/0.576 | 0.887/0.305/0.596 | 0.883/0.231/0.557 | 0.872/0.243/0.557 | N/A | 1.0× |
| EDnCNN | 学习 | 0.968/0.905/0.937 | 0.954/0.904/0.929 | 0.948/0.888/0.918 | 0.929/0.843/0.886 | 614.55K | 1.0× |
| EDformer | 学习 | 0.981/0.818/0.899 | 0.962/0.832/0.897 | 0.924/0.844/0.884 | 0.839/0.834/0.836 | 49.80K | 8.09× |
| **PRE-Mamba** | **学习** | **0.994/0.914/0.954** | **0.978/0.915/0.947** | **0.955/0.911/0.933** | **0.908/0.895/0.902** | **264.63K** | **204.54×** |

PRE-Mamba以仅0.26M参数达到平均SR/NR/DA=0.95/0.91/0.93，相比EDnCNN FLOPs仅为其2.66%且速度快204.5倍。

### 消融实验

在EventRain-27K验证集上的模块消融：

| 模型 | STDF | MS3M | $\mathcal{L}_{fft}$ | SR↑ | NR↑ | DA↑ |
|------|------|------|------|-----|-----|-----|
| M1 (baseline) | ✗ | ✗ | ✗ | 0.9123 | 0.8394 | 0.8759 |
| M2 | ✓ | ✗ | ✗ | 0.8983 | 0.8747 | 0.8865 |
| M3 | ✗ | ✓ | ✗ | 0.8941 | 0.8906 | 0.8923 |
| M4 | ✓ | ✓ | ✗ | 0.9046 | 0.8954 | 0.9000 |
| **Full** | **✓** | **✓** | **✓** | **0.9080** | **0.8949** | **0.9015** |

时间窗口数量消融：3窗口DA=0.8268，5窗口DA=0.9015（+9.03%），8窗口DA=0.9330（+12.84%），但推理速度分别为1.0×/0.70×/0.52×，选择5窗口平衡精度与效率。

### 关键发现

- STDF和MS3M分别提升基线1.21%和1.87% DA，MS3M贡献更大，体现深层时空特征建模的重要性
- 滤波方法对雨去噪无效，因为雨滴的时空模式与典型噪声有本质区别
- 方法可泛化到雪场景，无需架构改动
- 频域正则化通过对齐频域分布实现额外提升，利用了不同降雨强度在频域中的明显差异

## 亮点与洞察

- **首创性**：首个基于点的事件相机去雨框架，弥补了该方向的空白
- **效率极高**：0.26M参数+6.23G FLOPs，处理100K事件仅需0.1秒，真正适合边缘部署
- **4D表示设计精巧**：双时间尺度（窗口内微秒+窗口间宏观）既保留了事件相机的原生时间精度，又实现了长程依赖建模
- **频域正则化巧妙**：不嵌入FFT层而是作为loss项，既利用了频域信息又避免了计算开销
- **数据集贡献**：EventRain-27K是首个基于点的事件去雨数据集，包含合成、人工和真实场景

## 局限与展望

- 数据集虽包含真实数据但标签仅限合成和人工数据,真实场景无ground truth
- 仅在EventRain-27K上验证，未在其他事件相机数据集上测试跨数据集泛化
- 时间窗口数量固定为5，未探索自适应窗口策略
- 下游任务（如目标追踪、SLAM）的集成效果未验证
- 可扩展到其他恶劣天气（雾、沙尘）的统一框架

## 相关工作与启发

- Mamba/SSM在非语言序列任务中的应用越来越广泛，本文证明了其在事件数据处理中的有效性
- MSSM（运动感知SSM，来自LiDAR移动目标分割）被巧妙复用于雨事件的外观/运动分离
- 事件相机去雨是一个新兴但重要的方向，尤其对无人机和自动驾驶的恶劣天气鲁棒性至关重要

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个基于点的事件去雨框架，4D事件云和频域正则化设计新颖
- **实验充分度**: ⭐⭐⭐⭐ 合成+人工+真实+雪场景验证，消融全面，但缺少下游任务验证
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式排版规范，图表丰富
- **价值**: ⭐⭐⭐⭐ 开辟新方向+极高效率，对事件相机恶劣天气应用有实际价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](../../CVPR2025/image_restoration/efficient_visual_state_space_model_for_image_deblurring.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[ECCV 2024\] MambaIR: A Simple Baseline for Image Restoration with State-Space Model](../../ECCV2024/image_restoration/mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [\[CVPR 2025\] QMambaBSR: Burst Image Super-Resolution with Query State Space Model](../../CVPR2025/image_restoration/qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)
- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](../../AAAI2026/image_restoration/mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)

<!-- RELATED:END -->
