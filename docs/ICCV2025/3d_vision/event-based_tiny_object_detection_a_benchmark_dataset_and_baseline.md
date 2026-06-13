---
title: >-
  [论文解读] Event-based Tiny Object Detection: A Benchmark Dataset and Baseline
description: >-
  [ICCV 2025][3D视觉][Event Camera] 提出首个大规模事件相机反无人机小目标检测基准EV-UAV数据集（147序列/230万事件级标注/平均目标仅6.8×5.4像素），并设计EV-SpSegNet——基于稀疏3D点云分割的检测框架，利用小目标在时空事件点云中形成连续曲线的特征，配合时空相关性损失(STC loss)引导网络保留目标事件，在IoU/ACC/检测概率上全面超越13种SOTA方法，推理速度快10-100倍。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "Event Camera"
  - "目标检测"
  - "Anti-UAV"
  - "点云"
  - "Spatiotemporal Correlation"
  - "benchmark"
---

# Event-based Tiny Object Detection: A Benchmark Dataset and Baseline

**会议**: ICCV 2025  
**arXiv**: [2506.23575](https://arxiv.org/abs/2506.23575)  
**代码**: [https://github.com/ChenYichen9527/Ev-UAV](https://github.com/ChenYichen9527/Ev-UAV)  
**领域**: 3D视觉 / 事件相机 / 小目标检测  
**关键词**: Event Camera, Small Object Detection, Anti-UAV, Sparse Point Cloud, Spatiotemporal Correlation, benchmark

## 一句话总结
提出首个大规模事件相机反无人机小目标检测基准EV-UAV数据集（147序列/230万事件级标注/平均目标仅6.8×5.4像素），并设计EV-SpSegNet——基于稀疏3D点云分割的检测框架，利用小目标在时空事件点云中形成连续曲线的特征，配合时空相关性损失(STC loss)引导网络保留目标事件，在IoU/ACC/检测概率上全面超越13种SOTA方法，推理速度快10-100倍。

## 研究背景与动机

### 问题定义
反无人机(anti-UAV)任务中的小目标检测(SOD)：在复杂背景下实时、准确定位极小的无人机目标。目标极小（平均仅6.8×5.4像素），缺乏纹理和轮廓特征。

### 现有方法的局限

**传统帧相机的局限**：
   - 帧率低(30-60Hz)，无法捕获高速运动的UAV
   - 动态范围有限(~60dB)，极端光照下失效
   - 数据冗余，背景占据绝大多数像素

**事件相机的潜力**：
   - 微秒级时间分辨率(≥10⁶Hz)
   - 高动态范围(120dB)——强光/弱光均可工作
   - 异步稀疏输出，天然避免数据冗余

**现有事件数据集不足**：
   - VisEvent/EventVOT：UAV仅为子类别，目标大(84×66/129×100像素)
   - F-UAV-D：仅室内环境
   - NeRDD：仅大目标+简单背景
   - 缺乏事件级标注——现有数据集仅提供帧级bbox

**现有事件检测方法局限**：
   - 帧转换方法（RVT/RED）：将事件转为图像表示，压缩时间信息，处理冗余背景
   - 图方法/SNN方法：依赖特殊硬件或性能不足
   - 均未针对小目标的特殊几何结构设计

### 核心观察
小运动目标在时空事件点云中形成**连续细长曲线**，与背景的面状结构和噪声的离散点状结构有本质区别。这一几何特征可作为区分目标和非目标的关键线索。

## 方法详解

### 整体框架
EV-SpSegNet是基于U型稀疏卷积架构的事件点云分割网络：
- 输入：8秒的原始事件流，体素化为1像素×1像素×1ms的3D稀疏点云
- 编码器-解码器：对称结构，每层包含GDSCA模块
- 输出：逐事件的二分类（目标/非目标）
- 损失：BCE + STC时空相关性损失

### 关键设计1：分组膨胀稀疏卷积注意力（GDSCA）

GDSCA模块包含三个组件：

1. **Grouped Dilated Sparse Convolution (GDSC) Block**：
    - 将输入特征沿通道维度分组（默认4组）
    - 每组使用不同膨胀率(1,2,3,4)的稀疏卷积
    - 提取多尺度局部时间特征——适应不同运动速度的目标曲线

2. **Sp-SE Block**（Sparse Squeeze-and-Excitation）：
    - 融合来自不同膨胀率分组的特征
    - 通道注意力重新加权

3. **Patch Attention Block**：
    - 将点云划分为较大子区域
    - 子区域间自注意力实现全局上下文交互
    - 先下采样再注意力，降低二次复杂度

设计逻辑：GDSC先捕获局部多尺度特征 → Patch Attention再全局交互。实验证明单用Patch Attention反而降低性能——因为没有充足局部特征时全局注意力会导致特征混淆。

### 关键设计2：时空相关性损失（STC Loss）

标准BCE损失对每个事件独立处理，忽略邻域结构。STC loss引入"周围高置信度支持事件越多，该事件越可能是目标"的先验：

$$L_{stc}(p, y) = \begin{cases} -w_{stc}^\gamma \log(p), & y=1 \\ -(1-w_{stc})^\gamma \log(1-p), & y=0 \end{cases}$$

时空相关性权重：

$$w_{stc}(p) = \text{sigmoid}\left(\sum_{e^j \in V^{k\tau}} p^j\right)$$

其中 $V^{k\tau}$ 是 $k \times k \times \tau$ 邻域。默认 $k=3, \tau=5, \gamma=2$。

效果：
- 对正样本（目标事件）：更多邻居支持 → $w_{stc}$ 更大 → 更大权重 → 鼓励保留高时空相关的目标
- 对负样本（噪声）：孤立事件 → $w_{stc}$ 小 → $(1-w_{stc})$ 大 → 更高惩罚 → 鼓励去除孤立噪声

### 关键设计3：事件级标注方法

解决逐事件标注的效率问题：
1. 先将事件累积为帧，在帧上标注2D bbox
2. 根据帧对应的时间间隔Δt，将2D bbox扩展为XYT 3D空间的3D立方体
3. 取3D立方体内的所有事件作为目标标注

既实现了微秒级精度，又保持标注效率。

## 实验关键数据

### EV-UAV数据集统计
- 147个事件序列（99训练/24验证/24测试）
- 2300万+目标事件标注
- 平均目标大小：6.8×5.4像素（**极小**，是现有数据集的1/50）
- 45%极小目标（<8×8）、覆盖强光/正常/弱光/多场景/多目标
- DAVIS346相机，346×260分辨率

### 主实验结果（与13种SOTA方法对比）

| 方法 | 类型 | IoU(%)↑ | ACC(%)↑ | Pd(%)↑ | Fa(10⁻⁴)↓ | 参数量 | 推理(ms) |
|------|------|---------|---------|--------|-----------|--------|----------|
| YOLOV10-S | 帧 | 32.55 | 33.39 | 32.18 | 589.67 | 7.3M | 1627 |
| RVT | 事件+体素 | 43.21 | 51.38 | 60.35 | 55.68 | 9.9M | 1737 |
| Spike-YOLO | SNN | 43.94 | 48.26 | 59.62 | 55.38 | 69.0M | 1883 |
| COSeg | 点云 | 51.89 | 60.93 | 71.32 | 9.21 | 23.4M | 364 |
| **EV-SpSegNet** | **点云** | **55.18** | **65.02** | **77.53** | **1.63** | **4.0M** | **35.9** |

关键发现：
- IoU领先次优方法3.29个百分点（55.18 vs 51.89 COSeg）
- 虚警率仅1.63×10⁻⁴，比RVT低34倍
- 推理速度35.9ms处理8秒事件数据，比最快的点云方法(RandLA-Net 353ms)快10倍
- 参数量仅4.0M，小于大多数方法

### 消融实验：各组件贡献

| GDSC | PA | STC Loss | IoU↑ | ACC↑ | Pd↑ | Fa↓ | 参数量 |
|:---:|:---:|:---:|------|------|------|------|--------|
| - | - | - | 51.36 | 60.21 | 71.94 | 4.81 | 5.6M |
| ✓ | - | - | 52.73 | 62.64 | 73.12 | 4.28 | 3.8M |
| - | ✓ | - | 51.26 | 59.53 | 71.67 | 4.31 | 5.8M |
| ✓ | ✓ | - | 53.62 | 64.02 | 76.76 | 1.93 | 4.0M |
| ✓ | ✓ | ✓ | **55.18** | **65.02** | **77.53** | **1.63** | 4.0M |

关键发现：
1. 单独PA反而轻微降低性能——需要GDSC先提供局部特征
2. GDSC+PA组合时性能显著跳升（IoU +2.36，Fa从4.28降到1.93）
3. STC loss在GDSC+PA基础上继续提升（IoU +1.56, Pd +0.77）

### STC Loss的泛化性验证

| 方法 | 原始IoU | +STC IoU | 提升 |
|------|---------|----------|------|
| KPConv | 48.19 | 50.32 | +2.13 |
| RandLA-Net | 50.32 | 51.15 | +0.83 |
| COSeg | 51.89 | 53.12 | +1.23 |

STC loss在所有点云分割方法上都有一致提升，证明通用性。

### 膨胀率组合消融

| 膨胀率 | IoU↑ | Fa↓ |
|--------|------|------|
| (1,2,3,4) | **55.18** | **1.63** |
| (1,2,3,5) | 53.21 | 2.41 |
| (1,3,5,7) | 52.75 | 4.61 |
| (1,3,5,9) | 51.35 | 4.76 |

小间隔递增膨胀率(1,2,3,4)效果最佳；过大膨胀率导致对应事件距离过远，无法有效判断时间相关性。

## 亮点与洞察

1. **问题洞察的物理直觉**：小目标在时空点云中形成"连续曲线" vs 背景形成"面" vs 噪声形成"离散点"——这一几何区别是核心设计灵感
2. **事件级标注的巧妙方案**：通过2D bbox→3D XYT立方体的升维，将帧级标注自动转为事件级标注，同时支持退化为任意时刻的bbox标注
3. **STC loss的通用性**：不仅用于EV-SpSegNet，对现有点云分割方法均有持续提升——是事件数据上的通用损失函数
4. **极致效率**：4M参数+35.9ms推理8秒数据（对比帧方法1-4秒处理单帧），利用稀疏卷积天然避免冗余计算
5. **定性结果的说服力**：方法甚至检测到了人工标注遗漏的目标（因强背景杂波被人工忽略）

## 局限性

1. **事件相机固有限制**：静止或慢速目标不产生事件，导致检测失败——需要与帧相机互补
2. **标注近似性**：3D立方体内的事件不一定都属于目标（可能包含背景事件），标注有一定噪声
3. **仅单类别检测**：当前仅检测UAV目标，未扩展到多类别场景
4. **分辨率受限**：DAVIS346仅346×260，更高分辨率事件相机上的可扩展性未验证
5. **动态范围评估不充分**：虽然数据集覆盖多种光照条件，但各条件下的分项性能缺乏报告

## 相关工作与启发

- **从2D检测到3D点云分割的范式转变**：传统事件检测先转帧再用YOLO/DETR——本文直接在3D事件点云上分割，避免帧化带来的时间信息损失
- **STC loss的设计思想**类似于Focal Loss（对困难样本加权），但加权依据从"预测难度"变为"空间邻域一致性"
- GDSC的分组膨胀卷积思想源自图像领域（Dilated Conv/Atrous Conv），但适配到3D稀疏卷积并组合Patch Attention是新颖的
- EV-UAV数据集的标注方法可推广到其他事件相机任务（如事件语义分割、目标跟踪）

## 评分 ⭐⭐⭐⭐
- 创新性：⭐⭐⭐⭐（"曲线vs面vs点"的几何洞察+STC loss+完整基准，贡献扎实）
- 实验：⭐⭐⭐⭐⭐（13种方法对比+多维度消融+STC泛化验证+定性分析，非常完整）
- 写作：⭐⭐⭐⭐（问题动机清晰，数据集描述详尽，方法直觉性好）
- 实用性：⭐⭐⭐⭐⭐（35.9ms的极速推理+开源数据集/代码，直接可用于反无人机系统）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark](../../AAAI2026/3d_vision/griffin_aerial-ground_cooperative_detection_and_tracking_dataset_and_benchmark.md)
- [\[ICCV 2025\] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark](sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)
- [\[ICCV 2025\] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models](mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)

</div>

<!-- RELATED:END -->
