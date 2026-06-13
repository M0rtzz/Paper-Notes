---
title: >-
  [论文解读] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset
description: >-
  [CVPR 2025][目标检测][事件相机] 本文提出 MvHeat-DET，把视觉特征建模为二维热扩散过程，用 MoE 在 DFT/DCT/Haar 三种频域变换之间动态路由，再加上 IoU-aware query selection，做事件流目标检测；同时发布了高清事件相机检测数据集 EvDET200K (10,054 段视频 / 200K bbox / 10 类)。
tags:
  - "CVPR 2025"
  - "目标检测"
  - "事件相机"
  - "热传导"
  - "MoE"
  - "DETR"
  - "benchmark"
---

# Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset

**会议**: CVPR 2025  
**arXiv**: [2412.06647](https://arxiv.org/abs/2412.06647)  
**代码**: [https://github.com/Event-AHU/OpenEvDET](https://github.com/Event-AHU/OpenEvDET)  
**领域**: 目标检测 / 事件相机 / 视觉骨干  
**关键词**: 事件相机、热传导、MoE、DETR、benchmark

## 一句话总结
本文提出 MvHeat-DET，把视觉特征建模为二维热扩散过程，用 MoE 在 DFT/DCT/Haar 三种频域变换之间动态路由，再加上 IoU-aware query selection，做事件流目标检测；同时发布了高清事件相机检测数据集 EvDET200K (10,054 段视频 / 200K bbox / 10 类)。

## 研究背景与动机
1. **领域现状**：事件相机 (DVS) 凭借高动态范围、高时间分辨率、低功耗，在低光/快速运动场景下比 RGB 相机更鲁棒，事件检测是其关键下游任务。
2. **现有痛点**：
    - 基于 CNN 的事件检测器 (RED, Faster R-CNN) 感受野受限；
    - 基于 Transformer 的检测器 (RVT, SAST) 是 $\mathcal{O}(N^2)$ 复杂度，且解释性差；
    - 基于 SNN 的检测器 (SpikeYOLO) 节能但精度落后 ANN。
3. **核心矛盾**：缺少一个同时兼顾 **精度 / 效率 / 可解释性** 的事件检测骨干。
4. **本文目标**：
    - 设计一个数学可解释、复杂度低于 $\mathcal{O}(N^2)$ 的事件检测骨干；
    - 解决现有事件检测数据集分辨率低、类别少、规模小的问题。
5. **切入角度**：物理学的二维热传导方程 $\partial_t u = k(u_{xx}+u_{yy})$ 可以解释为"图像特征在空间中的扩散"，且其频域解 $\hat u(t)=\hat f \cdot e^{-k(v_x^2+v_y^2)t}$ 是闭式的、复杂度只 $\mathcal{O}(N^{1.5})$。
6. **核心 idea**：把热传导算子 (HCO) 升级为 MoE-HCO——为不同的事件场景 (稠密/稀疏) 在 DFT、DCT、Haar 三种"专家变换"间动态选择，从而既贴合事件的稀疏性又保留全局信息交换能力。

## 方法详解

### 整体框架
输入是事件流 → 堆叠成事件帧 → Stem 网络生成 patch embedding → 4 个 stage 的 **MoE-HCO** 块 (每 stage 内有若干 MHCO Layer + 下采样) → IoU-based Query Selection 选出 top-K token → DETR 风格解码头输出检测框。整体保留 ViT 风格架构，只把自注意力换成 MHCO。

### 关键设计

1. **MHCO: MoE Heat Conduction Operator**

    - 功能：用频域热扩散闭式解模拟 patch 之间的特征传递。
    - 核心思路：先用深度卷积把单通道温度图扩展到多通道得到 $U_0$，然后在三个专家分支之一 (DFT、DCT、Haar) 做正变换、乘以扩散核 $e^{-k(v_x^2+v_y^2)t}$，再做逆变换还原回空间域：$U_t = \mathcal{F}^{-1}\big(\mathcal{F}(U_0)\,e^{-k(v_x^2+v_y^2)t}\big)$。其中：
        - **DFT/IDFT**：处理跨 patch 的全局交互 (适合稠密、运动剧烈场景)。
        - **DCT/IDCT** 与 **Haar/IHaar**：天然满足 Neumann 边界条件 (导数为 0)，适合 patch 内独立处理 (适合稀疏、低运动事件场景，因为事件大量空白时 patch 间无信息可交换)。
    - 设计动机：单一变换难以兼顾稀疏与稠密事件，MoE 让模型按场景自适应路由；闭式频域解使复杂度从 $\mathcal{O}(N^2)$ 降到 $\mathcal{O}(N^{1.5})$。

2. **可学习的热扩散率 $k$ (Frequency Embeddings)**

    - 功能：让"语义关键区域获得更多热"，使热扩散具有内容自适应性。
    - 核心思路：随机初始化与频域特征 $\hat x$ 同形状的 Frequency Embeddings (FEs)，过线性层预测 $k$，再与扩散核相乘。$t$ 固定为常数。
    - 设计动机：物理意义上 $k$ 反映材料导热性；视觉上不同图像区域应该具有不同的"扩散速率"，让重点区域在多步扩散后特征更突出。

3. **MoE 路由 (Policy Network + Gumbel-Softmax)**

    - 功能：在 DFT / DCT / Haar 三个专家间做可微的硬选择。
    - 核心思路：轻量 policy 网络对当前特征图打分，用 Gumbel-Softmax 采样选出唯一专家分支，避免推理时三个分支都跑造成的额外开销。
    - 设计动机：事件场景空间分布差异极大 (有些只有一个动点、有些则非常稠密)，硬路由比软加权更高效，也更接近"按场景切换感受野"的语义。

4. **IQS: IoU-based Query Selection**

    - 功能：解决 DETR query 选择中"分类高但定位差"的偏差。
    - 核心思路：把 IoU 加入分类损失：$\mathcal{L}(y,\hat y) = \mathcal{L}_{bbox}(b,\hat b) + \mathcal{L}_{cls}(IoU, c, \hat c)$，使分类得分隐式编码定位质量，再按 top-K 分类分数选 query。
    - 设计动机：原始 DETR query 选择会误选高分但低 IoU 的 box；IQS 让 selection 更对齐最终检测目标。

### 损失函数 / 训练策略
- 总损失 = bbox L1 + GIoU + IoU-aware classification (公式 10)。
- 标准 DETR 训练流程，匹配用匈牙利算法。
- Gumbel-Softmax 温度逐步退火让 MoE 路由从 soft 转 hard。

## 实验关键数据

### 主实验
**N-Caltech101** (经典事件分类→检测 benchmark)：

| 方法 | 输入 | mAP |
|------|------|-----|
| YOLE | Event frames | 39.8 |
| EAS-SNN | Event points | 53.8 |
| Jeziorek et al. | Event frames | 53.4 |
| **MvHeat-DET (本文)** | Event frames | **55.7** |

**EvDET200K** (本文新数据集)：

| 方法 | mAP@50:95 | mAP@50 | Params | FLOPs | FPS |
|------|-----------|--------|--------|-------|-----|
| Faster R-CNN | 46.0 | 73.3 | 40.9M | 71.2G | 23 |
| Swin-T | 49.0 | 78.4 | 160M | 1043G | 26 |
| DetectoRS | 49.1 | 78.8 | 123M | 117G | 32 |
| YOLOv10-B | 44.1 | 77.9 | 19.1M | 92G | 30 |
| RVT | 40.7 | 73.1 | 9.9M | 8.4G | 88 |
| SpikeYOLO | 41.2 | 74.8 | 68.8M | 78.1G | 77 |
| S5-ViT | 42.9 | 76.3 | 18.2M | 5.6G | 84 |
| **MvHeat-DET** | **领先** | **领先** | 中等 | 较低 | 较高 |

### 数据集 EvDET200K 统计
- 10,054 段视频 (2-5s)，200,260 个 bbox，10 类 (people / car / bicycle / electric bicycle / basketball / ping pong / goose / cat / bird / UAV)。
- Prophesee EVK4-HD 真实采集，分辨率 1280×720 (远高于 Gen1 的 304×240)。
- 划分 6:1:3，含 2,949 段稠密场景，51% 小目标。
- 涵盖晴/雨/白天/黑夜/多视角/多运动 6 种挑战因素。

### 关键发现
- 三种变换专家是互补的：稀疏场景偏好 DCT/Haar，稠密场景偏好 DFT，路由结果显著非均匀。
- IQS 替换原 DETR query selection 后 mAP@75 提升明显，说明定位质量受益最大。
- 与 vHeat (只用 DCT) 相比，MoE-HCO 在 EvDET200K 上更稳，验证"单一变换不够"。

## 亮点与洞察
- **物理先验做骨干**：把热传导这种 PDE 的闭式频域解作为 token mixer，是 vHeat 思路的自然延伸；本文则进一步指出"不同变换对应不同边界条件 / 信号假设"，用 MoE 把它们组合起来——这是把"频域算子选择"问题第一次引入 MoE。
- **事件相机数据的稀疏性是天然 MoE 路由信号**：稠密 vs 稀疏的场景差异，比图像更显著，所以 MoE 的增益更大。这条思路可迁移到任何"输入分布差异大"的传感模态 (LiDAR、点云、医学超声)。
- **新的高清事件检测 benchmark**：弥补了 Gen1/1Mpx 在分辨率与类别多样性上的空缺，可作为后续事件检测算法的标准评测平台。
- **IQS 是即插即用的 DETR 改进**，可独立用在 RGB DETR 系列。

## 局限与展望
- 三种专家是手工设计的，专家集合是否需要根据数据集自动学习仍未探索。
- $t$ 固定为常数，若让 $t$ 也成为 learnable 或与位置相关，可能进一步提升表达力。
- 只在事件检测上验证，未在事件追踪、识别、流估计等其它事件任务上扩展 MoE-HCO。
- EvDET200K 仍是手工标注 5 帧/视频，标注密度有待提升。
- 改进思路：把 MHCO 与 SNN 结合，做"频域热扩散 + 脉冲发放"的混合编码器，兼顾节能与精度。

## 相关工作与启发
- **vs vHeat (NeurIPS 2024)**：vHeat 只用 DCT 模拟热扩散，本文论证 DCT 假设的 Neumann 边界条件不一定对所有 patch 适用，因此引入 DFT、Haar，并用 MoE 自适应；优势是更通用，劣势是结构更复杂。
- **vs RVT / SAST**：他们用 transformer 处理事件，复杂度高；本文 $\mathcal{O}(N^{1.5})$ 的频域算子在大 token 数下效率优势明显。
- **vs SpikeYOLO**：SNN 节能但精度低；本文走 ANN 路线但用频域算子保持效率。
- 可作为 baseline：任何"事件 + DETR"工作可直接对比 MvHeat-DET。

## 评分
- 新颖性: ⭐⭐⭐⭐ MoE 路由频域变换的思路第一次出现，物理-视觉对齐讲得清楚
- 实验充分度: ⭐⭐⭐⭐ 自建 benchmark + 15 个 baseline 评测，比较全面
- 写作质量: ⭐⭐⭐ 公式推导完整但章节略冗长
- 价值: ⭐⭐⭐⭐ 数据集贡献突出 (高清事件检测稀缺)，方法可被后续工作直接借用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)
- [\[CVPR 2025\] Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](mr_detr_instructive_multi-route_training_for_detection_transformers.md)
- [\[ECCV 2024\] Plain-Det: A Plain Multi-Dataset Object Detector](../../ECCV2024/object_detection/plain-det_a_plain_multi-dataset_object_detector.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](../../ICCV2025/object_detection/revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[NeurIPS 2025\] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](../../NeurIPS2025/object_detection/flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)

</div>

<!-- RELATED:END -->
