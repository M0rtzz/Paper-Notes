---
title: >-
  [论文解读] GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule
description: >-
  [ICCV 2025][医学图像][图像分割] 提出 GDKVM，一种基于线性键值关联和门控 Delta 规则的心脏超声视频分割架构，通过高效的内存管理和多尺度特征融合，在 CAMUS 和 EchoNet-Dynamic 上实现 SOTA 性能，同时保持实时推理速度。
tags:
  - ICCV 2025
  - 医学图像
  - 图像分割
  - 注意力机制
  - key-value memory
  - gated delta rule
---

# GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule

**会议**: ICCV 2025  
**arXiv**: [2512.10252](https://arxiv.org/abs/2512.10252)  
**代码**: [https://github.com/wangrui2025/GDKVM](https://github.com/wangrui2025/GDKVM)  
**领域**: medical_imaging  
**关键词**: echocardiography segmentation, linear attention, key-value memory, gated delta rule, video segmentation

## 一句话总结

提出 GDKVM，一种基于线性键值关联和门控 Delta 规则的心脏超声视频分割架构，通过高效的内存管理和多尺度特征融合，在 CAMUS 和 EchoNet-Dynamic 上实现 SOTA 性能，同时保持实时推理速度。

## 研究背景与动机

心脏超声视频的精确分割对心功能定量分析至关重要。然而面临多重挑战：

**图像质量差**：超声图像存在严重的斑点噪声和低对比度，组织结构模糊，边界不完整

**剧烈非刚性形变**：心脏在一个周期内形状和尺度变化巨大（收缩期 vs 舒张期）

**现有方法的局限**：
   - CNN（如 U-Net）局部感受野受限，无法直接建模帧间时间依赖
   - Vision Transformer 全局视野好但计算复杂度二次方增长
   - RNN/ConvLSTM 有误差传播风险且计算开销大
   - 空间-时间记忆网络（XMem、Cutie）依赖全标注参考帧且缺乏心脏周期识别

需要一种能高效整合全局时间上下文同时保持局部空间精度的模型。

## 方法详解

### 整体框架

GDKVM 由三个核心模块组成：线性键值关联（LKVA）实现帧间线性复杂度的时间建模；门控 Delta 规则（GDR）控制记忆的更新与遗忘；键像素特征融合（KPFF）整合多尺度特征。Key 编码器使用 ResNet-50 提取特征，Value 编码器编码原始帧和前一预测 mask，通过循环状态 S_t 进行在线分割。

### 关键设计

1. **线性键值关联 (LKVA)**：将传统 softmax 注意力的 O(t²C_d) 复杂度降为 O(tC_vC_k)。核心是用核方法替换指数核：exp(K_i^T Q_t) → ϕ(K_i)^T ϕ(Q_t)，将匹配重写为线性 RNN 形式。隐状态 S_t = Σ V_i ϕ(K_i)^T ∈ R^{C_v×C_k} 是固定大小的二维矩阵，逐帧累积上下文信息而非显式构建注意力矩阵。这种设计特别适合心脏结构的连续/周期性运动建模。

2. **门控 Delta 规则 (GDR)**：解决纯线性注意力中所有历史信息等权叠加导致的信息"模糊"问题。设计两个数据依赖矩阵从前一状态 S_{t-1} 投影而来：

    - **β_t（写入强度）**：控制新信息的学习强度。先移除当前帧键对应的旧关联 V_t^old = S_{t-1}K_t，再用 β_t 插值新旧值。心脏结构快速运动或清晰边界时 β_t 增大以强化当前帧学习
    - **α_t（衰减因子）**：控制过去记忆的衰减，避免记忆饱和。更新公式：S_t = S_{t-1}(α_t(I - β_t K_t K_t^T)) + β_t V_t K_t^T。允许模型在异常心律帧释放内存空间，同时保留长期重要信息

3. **键像素特征融合 (KPFF)**：融合三种尺度的特征以增强对噪声和伪影的鲁棒性：

    - **局部键特征 F_K**：卷积提取的局部语义
    - **全局键特征 F_Global**：通过全局平均池化 + 扩展得到的全局上下文
    - **像素级特征 F_Pix**：保留高频细节信息
    - 通过门控机制融合：G = σ(Conv_gate(F_K + F_Global))，F_fused = G·(F_K + F_Global) + (1-G)·F_Pix。当局部特征因噪声不可靠时，全局特征提供稳定补充；像素特征防止精细信息（如瓣膜运动）丢失

### 损失函数 / 训练策略

- 损失函数：交叉熵 + Soft Dice Loss，等权组合
- 模拟临床场景：预测时不使用 GT，先预测首尾帧形态再计算损失
- 训练：单张 RTX 3090，1500 迭代，AdamW lr=1e-4，batch size=10
- 数据增强：gamma 增强、随机缩放/旋转/对比度调整（各 p=0.5）
- 梯度裁剪 λ=3，稳定数据增强
- CAMUS 分辨率 256×256，EchoNet-Dynamic 128×128，每视频均匀采样 10 帧

## 实验关键数据

### 主实验

CAMUS 和 EchoNet-Dynamic 对比：

| 方法 | CAMUS mDice ↑ | CAMUS mIoU ↑ | CAMUS HD ↓ | CAMUS ASD ↓ | Echo mDice ↑ | Echo mIoU ↑ |
|------|-------------|-------------|-----------|-----------|------------|-----------|
| XMem++ | 89.38 | 85.81 | 4.03 | 4.87 | 87.51 | 83.57 |
| Cutie | 91.09 | 87.97 | 3.89 | 3.74 | 88.96 | 85.63 |
| VideoMamba | 91.96 | 89.04 | 3.48 | 3.31 | 90.22 | 87.03 |
| PKEchoNet | 93.49 | 90.95 | 3.42 | 2.93 | 92.60 | 89.89 |
| DSA | 94.25 | 91.80 | 3.27 | 2.37 | 92.91 | 90.26 |
| MemSAM | 93.63 | 90.97 | 3.47 | 2.60 | 92.71 | 89.90 |
| **GDKVM** | **95.11** | **92.97** | **3.05** | **1.98** | **93.46** | **90.86** |

临床指标 LVEF（CAMUS）：

| 方法 | corr ↑ | bias ± std (%) |
|------|--------|---------------|
| DSA | 0.891 | 0.86 ± 13.4 |
| MemSAM | 0.878 | -0.89 ± 12.3 |
| SimLVSeg | 0.895 | 1.83 ± 13.8 |
| **GDKVM** | **0.904** | **-0.19 ± 11.3** |

### 消融实验

各模块贡献（CAMUS）：

| LKVA | GDR | KPFF | mDice | mIoU | HD | ASD |
|------|-----|------|-------|------|----|-----|
| ✓ | - | - | 93.10 | 90.46 | 3.65 | 2.85 |
| ✓ | ✓ | - | 94.49 | 92.11 | 3.21 | 2.19 |
| ✓ | - | ✓ | 93.30 | 90.78 | 3.55 | 2.74 |
| ✓ | ✓ | ✓ | **95.11** | **92.97** | **3.05** | **1.98** |

GDR 内部消融（状态更新策略）：

| 策略 | mDice | 推理时间(ms) |
|------|-------|-------------|
| Baseline (简单叠加) | 93.30 | 151.61 |
| 完全替换（无 β_t） | 74.68 | 155.09 |
| 仅 β_t（无 α_t） | 94.57 | 158.77 |
| 仅 α_t（无 β_t） | 94.26 | 156.90 |
| **完整 GDR (α_t + β_t)** | **95.11** | 160.62 |

### 关键发现

- GDKVM 在 35.2M 参数下达到 37 FPS，兼顾精度和速度
- GDR 的双门控机制（α_t 衰减 + β_t 写入）缺一不可：完全替换旧信息（无门控）导致 Dice 暴跌至 74.68
- KPFF 对处理超声特有的尺度变化和空间变换至关重要
- 临床指标 LVEF 相关性达 0.904，bias 仅 -0.19%，展示了显著的临床应用价值
- α_t 和 β_t 的梯度分析显示二者呈协调调节模式（块状和对角结构），说明门控机制学到了有意义的记忆管理策略

## 亮点与洞察

- **线性复杂度的时间建模**：LKVA 将帧间匹配从 O(t²) 降至 O(t)，使得实时处理长心脏视频成为可能
- **生物启发的记忆管理**：GDR 的双门控设计类似人类的选择性注意——在稳定帧保持温和更新，在关键帧（结构突变、高噪声）进行激进调整
- **临床导向设计**：模拟临床场景（预测时不使用 GT），LVEF 估计与临床金标准高度一致
- 参数量仅 35.2M，单张 RTX 3090 可训练和推理，部署门槛极低

## 局限与展望

- 中间状态矩阵非标准方阵，限制了分块并行化和硬件加速
- 模型有时过度依赖自身预测导致困难样本的轮廓偏差（图 10 失败案例）
- 仅在超声心动图上验证，其他医学视频分割场景（如内镜、CT 序列）待探索
- 未在全监督和半监督基准上评估长序列一致性
- 可探索自适应边界定义或校正机制处理困难样本的伪影

## 相关工作与启发

- 与 XMem/XMem++ 等 STM 方法相比，GDKVM 不需要全标注参考帧，且计算更高效
- GDR 的设计灵感来自 DeltaNet 等线性 RNN 的 delta rule，但增加了数据依赖的双门控控制
- KPFF 补充了线性模型在空间推理上的不足，融合了 xLSTM 的局部特征和像素级信息
- 方法论可推广：线性键值关联 + 门控记忆管理的组合适用于任何需要长序列实时处理的医学视频分析

## 评分

- **新颖性**: ⭐⭐⭐⭐ 门控 Delta 规则和线性键值关联的组合在医学视频分割中是新颖的
- **实验充分度**: ⭐⭐⭐⭐ 两个数据集 + 临床指标 + 详细消融 + 可视化分析
- **写作质量**: ⭐⭐⭐⭐ 方法推导清晰，从 softmax attention 到线性化的推导自然流畅
- **价值**: ⭐⭐⭐⭐ 实时高精度的超声分割对临床应用有直接价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](../../CVPR2026/medical_imaging/t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[ICCV 2025\] ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users](progait_a_multi-purpose_video_dataset_and_benchmark_for_transfemoral_prosthesis_.md)
- [\[ICCV 2025\] Progressive Test Time Energy Adaptation for Medical Image Segmentation](progressive_test_time_energy_adaptation_for_medical_image_segmentation.md)

<!-- RELATED:END -->
