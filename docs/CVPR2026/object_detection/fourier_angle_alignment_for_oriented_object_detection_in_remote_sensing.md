---
title: >-
  [论文解读] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing
description: >-
  [CVPR 2026][目标检测][旋转目标检测] 利用傅里叶旋转等变性在频域估计并对齐目标方向，提出 FAAFusion（解决 Neck 层方向不一致）和 FAA Head（解决检测头分类-回归任务冲突）两个即插即用模块，在 DOTA 和 HRSC2016 上达到新 SOTA。
tags:
  - CVPR 2026
  - 目标检测
  - 旋转目标检测
  - 傅里叶变换
  - 方向对齐
  - 特征融合
  - 遥感
---

# Fourier Angle Alignment for Oriented Object Detection in Remote Sensing

**会议**: CVPR 2026  
**arXiv**: [2602.23790](https://arxiv.org/abs/2602.23790)  
**代码**: [https://github.com/gcy0423/Fourier-Angle-Alignment](https://github.com/gcy0423/Fourier-Angle-Alignment)  
**领域**: 目标检测  
**关键词**: 旋转目标检测, 傅里叶旋转等变性, 频域方向估计, 特征融合, 遥感

## 一句话总结

利用傅里叶旋转等变性在频域估计目标主方向并对齐特征，提出 FAAFusion 和 FAA Head 两个即插即用模块分别解决 FPN 跨尺度方向不一致和检测头分类-回归任务冲突，在 DOTA-v1.0/v1.5 和 HRSC2016 上取得新 SOTA。

## 研究背景与动机

**旋转目标检测的核心挑战**：遥感图像中船舶、飞机、车辆等目标方向任意，检测器需在标准 HBB $(x,y,w,h,c)$ 基础上额外预测旋转角 $\theta$。现有方法聚焦于旋转敏感卷积（ARC、GRA）、新 backbone（ReDet、LSKNet、PKINet、Strip R-CNN）或角度回归损失优化（GWD、KLD），但忽视了两个结构性瓶颈。

**瓶颈一：Neck 层方向不一致 (Directional Incoherence)**。FPN 中高层特征语义强但经过多次下采样后方向信号模糊（低频），只能粗略感知目标大致水平/垂直方向；低层特征保留了丰富的边缘和纹理，方向线索精确（高频）。传统 FPN 直接逐元素相加融合这两种方向不一致的特征，引入方向噪声，损害角度预测精度。

**瓶颈二：检测头任务冲突 (Task Conflict)**。同一 RoI 特征需同时服务分类和角度回归两个任务——分类需要旋转不变特征（飞机无论朝向都是飞机），回归需要旋转敏感特征（朝向不同角度预测应不同）。单一特征被迫折中，既非完全不变也非完全敏感，限制了两个任务的精度。

**核心洞察：傅里叶旋转等变性**。空间域信号旋转角度 $\phi$，其频谱也精确旋转 $\phi$（即 $\mathbf{F}_\phi(\boldsymbol{\omega}) = \mathcal{F}\{\mathbf{I}(\mathbf{R}_{-\phi}\mathbf{x})\}$）。同时，矩形目标的功率谱主方向垂直于其长轴（因 $a > b$ 时 $\operatorname{sinc}(2au)$ 主瓣更窄，高频能量沿 $v$ 轴集中）。这意味着可以从频域可靠估计目标主方向并执行显式对齐，这是对纯空间域方案的本质补充。

## 方法详解

### 整体框架

Fourier Angle Alignment (FAA) 包含两个即插即用模块：**FAAFusion** 嵌入 FPN neck 替换逐元素加法，解决跨尺度方向不一致；**FAA Head** 替换原始检测头，解决分类-回归任务冲突。两者共享核心的傅里叶角度估计（FAE）流程。

### 关键设计

1. **傅里叶角度估计 (Fourier Angle Estimation, FAE)**：

    - 功能：从频域估计特征图的主方向角
    - 核心思路：给定方阵特征图 $\mathbf{X} \in \mathbb{R}^{H \times H}$，执行 2D DFT 获得频谱 $\mathbf{F} = \mathcal{F}(\mathbf{X})$；将零频分量移至中心（乘以 $(-1)^{u+v}$）；从笛卡尔坐标 $(u,v)$ 转换到极坐标 $(\rho, \theta)$，计算能量谱 $E(\rho, \theta) = |\mathbf{F}_c(u(\rho,\theta), v(\rho,\theta))|^2$；沿径向加权求和得一维角度能量分布 $E_\theta(\theta) = \sum_\rho \rho \cdot E(\rho, \theta)$；取峰值方向 $\hat{\theta} = \arg\max_\theta E_\theta(\theta)$，范围约束到 $[0, \pi)$
    - 设计动机：基于矩形目标频谱主方向垂直于长轴的数学性质，用径向加权求和增强对高频方向分量的敏感性

2. **FAAFusion（方向一致特征融合）**：

    - 功能：嵌入 FPN 替换逐元素加法，对齐高层与低层特征方向后融合
    - 核心思路：高层特征 $\mathbf{Y}^{l+1}$ 上采样至低层分辨率；对上采样后的高层和低层特征分别用 $1 \times 1$ 卷积降维至 $C_{mid}$，再 unfold 提取局部 patch $\{\mathbf{p}_i^h\}, \{\mathbf{p}_i^l\}$；对每个位置 $i$，用 FAE 估计低层 patch 主方向 $\theta_i^l$；以 $\theta_i^l$ 为目标角旋转对应高层 patch 得 $\mathbf{p}_i^{rh} = \text{FAA}(\mathbf{p}_i^h; \theta_i^l)$；fold 重建对齐后的高层特征 $\mathbf{Y}_{recon}^{l+1}$，再用 $1 \times 1$ 卷积恢复通道维度；最终融合为三路之和：$\mathbf{Y}^l = \mathbf{X}^l + \mathbf{Y}_u^{l+1} + \mathbf{Y}_{recon}^{l+1}$
    - 设计动机：低层特征方向精确（高频边缘），作为基准对齐高层的模糊方向，消除直接相加的方向信号冲突；保留原始上采样特征加法确保语义信息不丢失

3. **FAA Head（方向感知检测头）**：

    - 功能：替换标准检测头，预对齐 RoI 特征到规范方向以解耦分类和回归
    - 核心思路：取 RoI 对齐特征 $\mathbf{F}_{roi}$，用 FAA 将主方向对齐到 $0°$ 得旋转不变特征 $\mathbf{F}_{inv} = \text{FAA}(\mathbf{F}_{roi}; 0°)$；残差相加 $\mathbf{F}_{final} = \mathbf{F}_{inv} + \mathbf{F}_{roi}$；展平后通过两层共享 FC（第一层输出维度 $1024 + 256 = 1280$），最后分别送入分类和回归分支
    - 设计动机：$\mathbf{F}_{inv}$ 消除了方向变化，对同类目标近似一致，有利于分类；$\mathbf{F}_{roi}$ 保留方向敏感信息，有利于角度回归；残差连接确保两个任务都能获得各自所需的信号

### 损失函数 / 训练策略

采用 Oriented R-CNN 标准损失（RPN 分类+回归 + Head 分类+回归），无新增损失项。优化器 AdamW（weight decay 0.05），DOTA 初始学习率 0.0001 训练 16 epochs，HRSC2016 初始学习率 0.0004 训练 36 epochs，batch size 2，单卡 RTX 3090。FAAFusion 在 FPN 第三层与第二层之间融合处部署。

## 实验关键数据

### 主实验 — DOTA-v1.0（单尺度训练测试）

| 方法 | Backbone | mAP |
|------|----------|-----|
| O-RCNN | ResNet50 | 75.87% |
| **O-RCNN + ours** | ResNet50 | **76.55%** (+0.68) |
| LSKNet | LSKNet-S | 77.49% |
| **LSKNet + ours** | LSKNet-S | **78.49%** (+1.00) |
| PKINet | PKINet-S | 78.39% |
| S-RCNN | StripNet-S | 78.09% |
| **S-RCNN + ours** | StripNet-S | **78.72%** (+0.63, 新 SOTA) |

### 主实验 — DOTA-v1.5（单尺度训练测试）

| 方法 | Backbone | mAP |
|------|----------|-----|
| O-RCNN | ResNet50 | 66.77% |
| **O-RCNN + ours** | ResNet50 | **67.14%** (+0.37) |
| S-RCNN | StripNet-S | 69.84% |
| **S-RCNN + ours** | StripNet-S | **71.57%** (+1.73) |
| PKINet | PKINet-S | 71.47% |
| LSKNet | LSKNet-S | 70.26% |
| **LSKNet + ours** | LSKNet-S | **72.28%** (+2.02, 新 SOTA) |

### 主实验 — HRSC2016

| 方法 | Params | FLOPs | AP50 (VOC07) | AP75 | mAP |
|------|--------|-------|--------------|------|-----|
| O-RCNN | 41.13M | 134.46G | 89.7 | 79.5 | 64.77 |
| **O-RCNN + ours** | 63.27M | 140.70G | **89.8** | **80.0** | **66.94** (+2.17) |
| LSKNet | 30.96M | 111.42G | 90.2 | 87.9 | 68.78 |
| **LSKNet + ours** | 48.34M | 114.89G | **90.6** | **89.8** | **70.74** (+1.96) |
| S-RCNN | 45.12M | 157.19G | 89.5 | 78.8 | 69.18 |
| **S-RCNN + ours** | 49.05M | 115.91G | **90.0** | 78.6 | **70.41** (+1.23) |

### 消融实验（DOTA-v1.0, LSKNet-S backbone）

| FAAFusion | FAA Head | Params | GFLOPs | mAP |
|-----------|----------|--------|--------|-----|
| ✘ | ✘ | 30.98M | 173.68G | 77.49% |
| ✘ | ✔ | 48.35M | 177.15G | 78.27% (+0.78) |
| ✔ | ✘ | 32.18M | 175.59G | 77.91% (+0.42) |
| ✔ | ✔ | 49.56M | 179.06G | **78.49%** (+1.00) |

### 检测头对比（DOTA-v1.0）

| Backbone | 检测头 | Params | GFLOPs | mAP |
|----------|--------|--------|--------|-----|
| ResNet50 | Original Head | 41.14M | 211.43G | 75.81% |
| ResNet50 | Strip Head | 55.82M | 258.35G | 76.11% |
| ResNet50 | **FAA Head** | 58.51M | 214.90G | **76.18%** |
| LSKNet-S | Original Head | 30.98M | 173.68G | 77.49% |
| LSKNet-S | Strip Head | 45.65M | 220.60G | 78.04% |
| LSKNet-S | **FAA Head** | 48.35M | 177.15G | **78.27%** |
| StripNet-S | Original Head | 30.46M | 171.79G | 77.03% |
| StripNet-S | Strip Head | 45.14M | 218.71G | 78.09% |
| StripNet-S | **FAA Head** | 47.83M | 175.26G | **78.52%** |

### 关键发现

- FAAFusion 和 FAA Head 互补：单独使用分别提升 +0.42% 和 +0.78%，组合提升 +1.00%
- 三个不同 backbone 上均一致有效，证明模块的即插即用通用性
- DOTA-v1.5 提升最显著（LSKNet +2.02%），该数据集包含大量 <10 像素的极小目标，方向对齐对小目标尤为重要
- HRSC2016 船舶检测提升大（O-RCNN +2.17%），高长宽比目标的频域方向估计优势明显
- FAA Head 与 Strip Head 参数量相近但 FLOPs 低 40G+，精度更高——频域对齐比空间域条状卷积更高效
- 高 IoU 阈值分析：在 IoU 0.70-0.90 范围内优势随阈值升高而扩大，表明方向对齐提升了精确定位能力

## 亮点与洞察

- **频域切入角度新颖**：首次将傅里叶旋转等变性系统应用于旋转目标检测，从频域估计方向角具有严格的数学推导支撑（矩形 sinc 主瓣分析），物理可解释性极强
- **问题诊断精准**：明确分离了两个独立问题（Neck 方向不一致 + Head 任务冲突），并分别用 FAAFusion 和 FAA Head 对症下药
- **FAAFusion 以低层为基准的对齐策略**直觉正确：低层特征边缘清晰、方向可靠，用它来校准高层的模糊方向
- **FAA Head 残差设计**简洁有效：$\mathbf{F}_{inv} + \mathbf{F}_{roi}$ 一步完成分类-回归的隐式解耦，无需复杂的双分支架构
- 高 IoU 下的持续优势证明方向对齐确实提升了精细定位

## 局限与展望

- **参数增量明显**：O-RCNN 从 41M 增至 63M（+54%），FAAFusion 的 unfold/fold 操作引入额外开销，可考虑更轻量的频域处理
- **矩形假设**：频域方向估计基于目标近似矩形的先验，对不规则形状目标（如圆形油罐）准确性可能下降
- **框架绑定**：仅在 Oriented R-CNN 两阶段框架验证，未测试单阶段检测器（S2A-Net 等）或 anchor-free 方法
- **FAAFusion 仅部署于一个层级**：仅替换了 P3-P2 融合处的加法，可探索全层级部署的收益与开销平衡
- 未在更大规模数据集（如 DIOR-R）或多尺度训练测试条件下验证

## 相关工作与启发

- **FreqFusion** 将特征分解为高低频组件分别处理，FAA 则直接利用旋转等变性做方向估计——两者都在频域操作但目标不同
- **ReDet** 用旋转等变 backbone（ReResNet）建模方向信息，FAA 以更轻量的即插即用方式在 neck 和 head 层级引入方向感知
- **Strip R-CNN** 用条状卷积建模高长宽比目标几何特征，FAA Head 以更低 FLOPs 达到更高精度，说明频域对齐可能比显式几何卷积更高效
- 频域方向估计方法有望扩展到实例分割、遥感变化检测、姿态估计等需要方向建模的任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 频域旋转等变性切入旋转目标检测，理论清晰、视角全新，FAAFusion 和 FAA Head 设计动机透彻
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、三个 backbone、消融完整、检测头对比有力，缺多尺度和更多框架验证
- 写作质量: ⭐⭐⭐⭐ 理论推导详尽，Formulation 和 Motivation 节清晰，图文配合好
- 价值: ⭐⭐⭐⭐⭐ 即插即用 + 一致稳定提升 + 物理可解释 + 开源代码，实用性强
---
title: >-
  [论文解读] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing
description: >-
  [CVPR 2026][目标检测][旋转目标检测] 利用傅里叶旋转等变性在频域估计并对齐目标方向，提出 FAAFusion（解决 Neck 层方向不一致）和 FAA Head（解决检测头分类-回归任务冲突）两个即插即用模块，在 DOTA 和 HRSC2016 上达到新 SOTA。
tags:
  - CVPR 2026
  - 目标检测
  - 旋转目标检测
  - 傅里叶变换
  - 方向对齐
  - 特征融合
  - 遥感
---

# Fourier Angle Alignment for Oriented Object Detection in Remote Sensing

**会议**: CVPR 2026  
**arXiv**: [2602.23790](https://arxiv.org/abs/2602.23790)  
**代码**: https://github.com/gcy0423/Fourier-Angle-Alignment (有)  
**领域**: 遥感 / 目标检测  
**关键词**: 旋转目标检测, 傅里叶变换, 方向对齐, 特征融合, 遥感

## 一句话总结

利用傅里叶旋转等变性在频域估计并对齐目标方向，提出 FAAFusion（解决 Neck 层方向不一致）和 FAA Head（解决检测头分类-回归任务冲突）两个即插即用模块，在 DOTA 和 HRSC2016 上达到新 SOTA。

## 研究背景与动机

遥感图像中的目标方向任意，旋转目标检测 (ROD) 需同时预测类别和方向角。现有方法从旋转敏感卷积、backbone 修改、角度回归损失等入手，但存在两个被忽视的核心瓶颈：

**Neck 层方向不一致 (Directional Incoherence)**：FPN 高层特征语义强但方向模糊（低频），低层特征边缘清晰方向精确（高频），简单相加导致方向信号冲突

**检测头任务冲突 (Task Conflict)**：分类需旋转不变特征，回归需旋转敏感特征，单一 RoI 特征无法同时满足

核心洞察：**傅里叶旋转等变性**——空间域信号旋转 phi，频谱也精确旋转 phi。可在频域可靠估计目标主方向并显式对齐。

## 方法详解

### 整体框架

FAA 含两个即插即用模块：FAAFusion 嵌入 FPN 解决方向不一致；FAA Head 替换检测头解决任务冲突。两者共享傅里叶角度估计核心。

### 关键设计

1. **傅里叶角度估计 (Fourier Angle Estimation)**

    - 功能：从频域估计特征图主方向
    - 核心思路：2D DFT -> 中心化 -> 极坐标能量谱 -> 径向求和得角度能量分布 -> 取最大值
    - 数学基础：矩形目标功率谱中主谱方向垂直于长轴（sinc 函数衰减差异）
    - 设计动机：利用傅里叶旋转等变性，物理可解释性强

2. **FAAFusion（方向一致特征融合）**

    - 功能：替换 FPN 逐元素加法，对齐高低层特征方向后融合
    - 核心思路：高层上采样 -> 1x1 卷积+unfold 提取局部特征 -> 估计低层主方向 -> 旋转高层对齐 -> fold+三路相加
    - 设计动机：低层方向精确（高频边缘），以其为基准对齐高层模糊方向

3. **FAA Head（方向感知检测头）**

    - 功能：预对齐 RoI 特征到规范方向，解耦分类和回归
    - 核心思路：估计主方向旋转到 0 度得 F_inv -> 残差 F_final = F_inv + F_roi -> 两层 FC -> 分类回归
    - 设计动机：F_inv 旋转不变利于分类，F_roi 保留方向敏感信息利于回归

### 损失函数 / 训练策略

Oriented R-CNN 标准损失，AdamW (wd 0.05)，DOTA lr 0.0001，HRSC lr 0.0004，batch 2，单卡 RTX 3090。DOTA 16 epochs，HRSC 36 epochs。

## 实验关键数据

### 主实验

| 数据集 | 方法 | Backbone | mAP | 提升 |
|--------|------|----------|-----|------|
| DOTA-v1.0 | LSKNet | LSKNet-S | 77.49% | - |
| DOTA-v1.0 | **LSKNet+ours** | LSKNet-S | **78.49%** | +1.00% |
| DOTA-v1.0 | **S-RCNN+ours** | StripNet-S | **78.72%** | +0.63% (新SOTA) |
| DOTA-v1.5 | PKINet | PKINet-S | 71.47% | - |
| DOTA-v1.5 | **LSKNet+ours** | LSKNet-S | **72.28%** | +2.02% (新SOTA) |
| HRSC2016 | O-RCNN | ResNet50 | 64.77 | - |
| HRSC2016 | **O-RCNN+ours** | ResNet50 | **66.94** | +2.17% |

### 消融实验

| FAAFusion | FAA Head | Params | GFLOPs | mAP |
|-----------|----------|--------|--------|-----|
| No | No | 30.98M | 173.68G | 77.49% |
| No | Yes | 48.35M | 177.15G | 78.27% (+0.78%) |
| Yes | No | - | - | 77.91% (+0.42%) |
| Yes | Yes | 48.34M | 176.57G | **78.49%** (+1.00%) |

### 关键发现

- 两模块互补：FAAFusion +0.42%，FAA Head +0.78%，组合 +1.00%
- 三个 backbone 上一致有效，证明即插即用
- DOTA-v1.5 提升更显著（+2.02%），小目标方向对齐尤为重要
- HRSC2016 船舶检测提升最大（+2.17%），高长宽比目标频域方向估计优势明显

## 亮点与洞察

- 频域角度切入旋转目标检测非常新颖，物理可解释性强
- FAAFusion 以低层精确方向为基准对齐高层模糊方向，直觉精准
- FAA Head 残差设计简洁有效
- 即插即用可集成到任何 FPN+检测头框架

## 局限与展望

- 参数增加较多（O-RCNN 41M->63M），可考虑更轻量设计
- 假设目标为矩形，不规则形状可能不适用
- 仅在 Oriented R-CNN 框架验证
- unfold/fold 在高分辨率特征图上开销可能较大

## 相关工作与启发

- 与 FreqFusion 不同：后者分解高低频组件，FAA 直接利用旋转等变性
- ReDet 更重的旋转等变 backbone 方案，FAA 更轻量灵活
- 频域方向估计可扩展到实例分割、姿态估计

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 频域旋转等变性切入旋转目标检测，理论清晰视角全新
- 实验充分度: ⭐⭐⭐⭐ 三个数据集三个 backbone，消融完整
- 写作质量: ⭐⭐⭐⭐ 理论推导详尽，问题定义清晰
- 价值: ⭐⭐⭐⭐⭐ 即插即用+一致提升+物理可解释

<!-- RELATED:START -->

## 相关论文

- [Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)
- [MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](../../ECCV2024/object_detection/mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)
- [SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](../../AAAI2026/object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)
- [SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](../../ICLR2026/object_detection/spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](../../ICCV2025/object_detection/openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)

<!-- RELATED:END -->
