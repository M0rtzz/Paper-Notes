---
title: >-
  [论文解读] High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation
description: >-
  [ICCV 2025][人体理解][位姿估计] 提出 GLSMamba，首个纯 Mamba 的视频人体姿态估计框架，通过 Global Spatiotemporal Mamba（6D 选择性时空扫描 + 时空调制融合）和 Local Refinement Mamba（窗口化时空扫描）分别建模全局动态上下文和局部关键点细节，在四个基准上以线性复杂度达到 SOTA。
tags:
  - ICCV 2025
  - 人体理解
  - 位姿估计
  - Mamba
  - State Space Model
  - Spatiotemporal Modeling
  - High-Resolution
---

# High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation

**会议**: ICCV 2025  
**arXiv**: [2510.11017](https://arxiv.org/abs/2510.11017)  
**代码**: 无  
**领域**: human_understanding  
**关键词**: Video Human Pose Estimation, Mamba, State Space Model, Spatiotemporal Modeling, High-Resolution

## 一句话总结

提出 GLSMamba，首个纯 Mamba 的视频人体姿态估计框架，通过 Global Spatiotemporal Mamba（6D 选择性时空扫描 + 时空调制融合）和 Local Refinement Mamba（窗口化时空扫描）分别建模全局动态上下文和局部关键点细节，在四个基准上以线性复杂度达到 SOTA。

## 研究背景与动机

视频人体姿态估计（VHPE）需要密集的时空分析，关键在于同时捕获：
- **全局动态上下文**：整体人体运动模式和趋势
- **局部运动细节**：关键点的高频变化

现有方法的固有问题：

**CNN 方法**（如 TDMI）：固定感受野限制全局推理能力，在遮挡和模糊场景下产生大偏差

**Transformer 方法**（如 DiffPose）：能捕获全局依赖但忽略局部高频细节，且在高分辨率序列上复杂度为二次方——直接在 1/4 分辨率 × T 帧（15,360 tokens）上 OOM

**现有视频 Mamba**（如 VideoMamba）：仅做逐帧双向扫描展平空间 token，拉长时间相邻 token 的距离，缺乏对局部细节的专门设计

核心观察：需要一种能 (1) 在高分辨率时空序列上进行全局建模且保持线性复杂度，(2) 同时增强局部关键点运动细节的架构。

## 方法详解

### 整体框架

输入视频序列 → 视觉编码器（ViTPose，冻结）提取高分辨率特征 → Global Spatiotemporal Mamba (GSM, 4 blocks) → Local Refinement Mamba (LRM, 2 blocks) → 检测头 → 姿态热力图。

特征分辨率为 1/4 × H × W × T，对 5 帧序列（δ=2，前后各 2 帧 + 当前帧），token 数量达 15,360。

### 关键设计

1. **Global Spatiotemporal Mamba (GSM)**：

    - **Sequential Channel Attention**：将特征序列拼接后通过 GAP → MLPs → sigmoid 得到逐帧通道注意力权重，自适应激活重要时空信息
    - **6D selective Space-Time Scan (STS6D)**：沿 6 条时空扫描路径展平特征序列为 1D 后分别送入 S6 块。具体是将多帧特征堆叠成全景时空表示，水平/垂直遍历得到 $\tilde{\mathbf{y}}_1, \tilde{\mathbf{y}}_4$（统一扫描，捕获高层时空表示），空间逐帧遍历得到 $\tilde{\mathbf{y}}_2, \tilde{\mathbf{y}}_5$（空间扫描，完整人体空间上下文），时间轴像素遍历得到 $\tilde{\mathbf{y}}_3, \tilde{\mathbf{y}}_6$（时间扫描，密集运动趋势）
    - **Spatial- and Temporal-Modulated scan Merging (STMM)**：先将双向扫描结果按类型合并（$\tilde{\mathbf{y}}_u, \tilde{\mathbf{y}}_s, \tilde{\mathbf{y}}_t$），然后通过 Deformable Convolution 进行空间调制和时间调制补偿，自适应聚合不同语义的扫描知识

   设计动机：将 1D Mamba 适配到视频时空建模，通过 6 方向扫描充分挖掘各维度信息，用 DCN 自适应融合避免简单加法的信息损失。

2. **Local Refinement Mamba (LRM)**：

    - **Windowed Space-Time Scan (WSTS)**：将特征序列分为不重叠的 3D 时间管道窗口（如 8×6×T），在每个窗口内逐帧进行正反向扫描并送入 S6 块
    - 保持序列大小感受野的同时增强局部细节
    - 去掉 Sequential Channel Attention，将 STS6D/STMM 替换为 WSTS

   设计动机：GSM 关注全局理解但缺乏关键点的局部高频细节，LRM 通过局部窗口内的密集扫描补充细粒度运动信息。

3. **双流门控设计**：GSM 块中主流经 STS6D+STMM 得到全局特征 $\tilde{\mathcal{F}}$，另一流经深度卷积 + LayerNorm + SiLU 得到门控注意力 $\bar{\mathcal{A}}$，两者相乘后再通过 FFN。

### 损失函数 / 训练策略

- 标准热力图估计损失：$\mathcal{L}_H = \|\hat{\mathbf{H}}^i_t - \mathbf{H}^i_t\|_2^2$
- 使用 ViTPose 预训练权重（在 COCO 上），推理时冻结 backbone
- AdamW 优化器，初始 lr 1e-4，第 6 epoch 降为 1e-5，第 12 epoch 降为 1e-6
- 数据增强：随机旋转/缩放、截断、翻转
- 时间跨度 δ=2（共 5 帧），单卡 TITAN RTX 训练 20 epochs

## 实验关键数据

### 主实验 (表格)

**PoseTrack2017 验证集（mAP）：**

| 方法 | Backbone | Mean mAP |
|------|----------|----------|
| PoseWarper | HRNet-W48 | 81.2 |
| DCPose | HRNet-W48 | 82.8 |
| FAMI-Pose | HRNet-W48 | 84.8 |
| TDMI | HRNet-W48 | 85.7 |
| DiffPose | ViT-B | 86.4 |
| DSTA | ViT-H | 85.6 |
| **GLSMamba-B** | **ViT-B** | **86.9** |
| **GLSMamba-H** | **ViT-H** | **88.0** |

**PoseTrack2018 / PoseTrack21 / Sub-JHMDB：**

| 数据集 | GLSMamba-B | GLSMamba-H | 前 SOTA |
|--------|------------|------------|---------|
| PoseTrack2018 | 84.2 | **84.9** | 83.5 (TDMI/DSTA) |
| PoseTrack21 | 84.1 | **84.7** | 83.5 (TDMI/DSTA) |
| Sub-JHMDB | **97.9** | - | 96.0 (FAMI-Pose) |

### 消融实验 (表格)

**组件消融 (PoseTrack2017)：**

| 设置 | GSM | LRM | mAP |
|------|-----|-----|-----|
| Backbone only | - | - | 74.2 |
| + GSM | ✓ | - | 86.0 (+11.8) |
| + GSM + LRM (完整) | ✓ | ✓ | **86.9** (+0.9) |

**STS6D 扫描方向消融：**

| 扫描方向 | #Params | GFLOPs | mAP |
|----------|---------|--------|-----|
| 统一扫描 | 9.1M | 137.4 | 85.8 |
| + 空间扫描 | 9.4M | 138.1 | 86.5 |
| + 空间 + 时间扫描 (完整 STS6D) | 9.8M | 138.9 | **86.9** |
| 完整 STS6D 无 STMM | 9.1M | 137.4 | 86.2 |

**分辨率影响与计算效率：**

| 方法 | 分辨率 | Token数 | #Params | GFLOPs | mAP |
|------|--------|---------|---------|--------|-----|
| GLSMamba-B | 1/4×T | 15,360 | 9.8M | 138.9 | **86.9** |
| GLSMamba-BLR | 1/16×T | 960 | 9.8M | 85.1 | 85.7 |
| TransLR | 1/16×T | 960 | 46.3M | 125.7 | 84.2 |
| TransNR | 1/8×T | 3,840 | 47M | 315.2 | 84.8 |
| TransHR | 1/4×T | 15,360 | - | - | OOM |

### 关键发现

1. **GSM 贡献最大**：引入 GSM 直接将 mAP 从 74.2 提升到 86.0（+11.8），说明全局时空建模对 VHPE 至关重要
2. **6 方向扫描逐步提升**：从统一 → +空间 → +时间，mAP 从 85.8 → 86.5 → 86.9，且额外计算量可忽略
3. **STMM 比简单加法好 0.7 mAP**：自适应融合不同语义的扫描结果很重要
4. **高分辨率显著有利**：1/4 分辨率比 1/16 好 1.2 mAP，但 Transformer 架构在同分辨率 OOM
5. **参数效率极高**：仅 9.8M 可训练参数（比需微调 backbone 的方法减少 86.2%），GFLOPs 也仅 138.9（PoseWarper 的 66%）

## 亮点与洞察

- **首个纯 Mamba 的 VHPE 框架**：证明 SSM 在计算机视觉密集预测任务上的巨大潜力
- **解耦全局-局部建模的设计哲学**：GSM 和 LRM 各司其职，比统一架构更有效
- **线性复杂度处理高分辨率序列**：在 15,360 tokens 上 Transformer OOM 而 Mamba 仅 138.9G FLOPs
- **STS6D 的多方向扫描设计精巧**：统一/空间/时间三种扫描各捕获不同语义，互补性强
- **极低训练成本**：冻结 backbone + 仅训练 9.8M 参数，单卡 TITAN RTX 即可训练

## 局限与展望

1. backbone 权重完全冻结，可能限制在特定领域的适应性
2. 时间跨度固定为 δ=2（5帧），更长时间范围可能进一步提升
3. Sub-JHMDB 上与后处理方法 DeciWatch (98.8) 差距较大（97.9），后处理方法在姿态坐标空间操作性质不同
4. 局部窗口大小（8×6×T）为固定设置，自适应窗口可能更优
5. 未探索 3D 人体姿态估计、视频分割等其他密集时空任务

## 相关工作与启发

- **SSM/Mamba 系列**：从 S4 → Mamba 的演进，本文将 1D Mamba 拓展到高分辨率视频时空建模
- **CNN vs Transformer 的局限**：CNN 感受野有限，Transformer 二次复杂度，Mamba 在高分辨率上取得最佳平衡
- **VideoMamba 对比**：VideoMamba 仅做简单逐帧展平，本文 6D 扫描 + 窗口化局部扫描更全面
- 对跟踪、动作识别等其他时空任务的 Mamba 改造有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个纯 Mamba VHPE 框架，STS6D 多方向扫描和 STMM 融合设计新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 四个基准全面验证，消融非常细致（组件/扫描方向/分辨率/计算效率）
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式推导完整，可视化（激活图、对比）丰富
- **价值**: ⭐⭐⭐⭐ 开辟 Mamba 在密集时空预测任务的新方向，计算效率优势突出
