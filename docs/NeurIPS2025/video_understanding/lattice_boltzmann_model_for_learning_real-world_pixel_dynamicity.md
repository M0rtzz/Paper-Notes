---
title: >-
  [论文解读] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity
description: >-
  [NeurIPS 2025][视频理解][point tracking] 受流体力学中格子玻尔兹曼方法启发，提出 LBM（Lattice Boltzmann Model）用于在线实时像素跟踪，将视频像素建模为流体格子并通过碰撞-流式过程求解运动状态，以 18M 参数实现 SOTA 在线跟踪性能且可在边缘设备上实时运行。
tags:
  - NeurIPS 2025
  - 视频理解
  - point tracking
  - online tracking
  - lattice Boltzmann
  - real-time
  - object tracking
---

# Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity

**会议**: NeurIPS 2025  
**arXiv**: [2509.16527](https://arxiv.org/abs/2509.16527)  
**代码**: [项目主页](https://george-zhuang.github.io/lbm)  
**领域**: video_understanding  
**关键词**: point tracking, online tracking, lattice Boltzmann, real-time, object tracking

## 一句话总结

受流体力学中格子玻尔兹曼方法启发，提出 LBM（Lattice Boltzmann Model）用于在线实时像素跟踪，将视频像素建模为流体格子并通过碰撞-流式过程求解运动状态，以 18M 参数实现 SOTA 在线跟踪性能且可在边缘设备上实时运行。

## 研究背景与动机

- **离线/半在线方法的实际局限**：主流点跟踪方法（TAPIR、CoTracker、LocoTrack）要求完整视频或时间窗口输入，导致高内存占用、不可避免的延迟、无法即时响应新出现像素、以及隐私/存储风险。
- **时空完整性依赖是根本瓶颈**：离线方法依赖双向时序优化，半在线方法依赖多次迭代——LocoTrack 从 1 次迭代到 4 次迭代吞吐量下降超 60%。
- **边缘设备部署需求**：机器人操控、医学视觉等应用需要在资源受限的嵌入式设备上实时推理。
- **开放世界物体跟踪的脆弱性**：传统 MOT 将目标作为整体实体，在目标变形、自遮挡和快速运动时性能急剧下降。

## 方法详解

### 理论基础：格子玻尔兹曼方法

经典 LBM 将流体离散为格子，分布函数 $\mathbf{f}$ 经历碰撞和流式过程：

$$\mathbf{f}(\mathbf{x}, t) = \sum_i [f_i(\mathbf{x} - \mathbf{c}_i \Delta t, t - \Delta t) + \Omega_i(\mathbf{x} - \mathbf{c}_i \Delta t, t - \Delta t)]$$

其中 $\mathbf{c}_i$ 为第 $i$ 方向离散速度，$\Omega$ 为碰撞算子描述向平衡分布的松弛。求解后得到密度和速度：

$$\rho(\mathbf{x},t) = \sum_i f_i(\mathbf{x},t), \quad \rho \mathbf{u}(\mathbf{x},t) = \sum_i \mathbf{c}_i f_i(\mathbf{x},t)$$

### 点跟踪的 LBM

给定图像 $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$ 和 $N$ 个查询点 $\mathbf{q} \in \mathbb{R}^{N \times 2}$，估计后续帧中的位置 $\mathbf{p} \in \mathbb{R}^{N \times 2}$ 和可见性 $\mathbf{v} \in \mathbb{R}^N$。

**视觉编码**：使用 ImageNet 预训练的 ResNet18 前三层，所有特征图上采样到 stride-4 后拼接，得到 $\mathbf{o} \in \mathbb{R}^{d \times H/4 \times W/4}$。设计以效率为优先。

**分布初始化**：对查询点进行双线性采样：$\mathbf{f}_{init} = \text{BilinearSample}(\mathbf{o}, \mathbf{q}) \in \mathbb{R}^{N \times d}$

**分布预测（Predict）**：不同于经典 LBM 使用固定邻域，LBM 使用可学习邻域 $\delta$ 计算碰撞交互：

$$\mathbf{f}(x, t | \delta) = \mathbf{f}(x, t - \Delta t | \delta) + \Omega(x, t - \Delta t | \delta)$$

碰撞算子 $\Omega$ 通过可变形注意力实现。时序上下文扩展至 $N_s$ 步历史，维护流式分布 $\mathbf{f}_s$ 和碰撞分布 $\mathbf{f}_c$：

$$\mathbf{f} = \phi_c(\phi_s(\mathbf{f}_{init}, \mathbf{f}_s), \mathbf{f}_c)$$

其中 $\phi_s, \phi_c$ 为交叉注意力模块。

**分布更新（Update）**：计算像素分布与视觉特征的相关图，选取 top-$k$ 响应作为参考点 $\mathbf{r}$，通过可变形注意力 $\psi$ 更新分布：$\psi(\mathbf{f}, \mathbf{o}, \mathbf{r})$。

**多层 Predict-Update Transformer**：堆叠多层，每层包含预测+更新步骤。随层数增加参考点逐渐减少，最终层仅保留一个确定性参考点 $\mathbf{r}_{last}$。

**输出头**：跟踪头预测偏移 $\Delta \mathbf{p} = \mathcal{H}_{track}(\mathbf{f}, \mathbf{o}, \mathbf{r}_{last})$，可见性头预测置信度和可见性 $\{\rho, \mathbf{v}\} = \mathcal{H}_{vis}(\mathbf{f}, \mathbf{o}, \mathbf{r}_{last})$。

### 损失函数

$$\mathcal{L} = \lambda_{cls} \mathcal{L}_{cls} + \mathcal{L}_{reg} + \mathcal{L}_{vis} + \mathcal{L}_{conf}$$

- $\mathcal{L}_{cls}$：相关图的交叉熵损失（每层）
- $\mathcal{L}_{reg}$：偏移的 L1 损失（仅可见点）
- $\mathcal{L}_{vis}$：可见性交叉熵
- $\mathcal{L}_{conf}$：置信度交叉熵（以 $\|\mathbf{p} - \mathbf{p}_{gt}\| < 8$ 为正例）

### 物体跟踪的 LBM

将目标分解为细粒度像素集合，通过像素跟踪建立物体关联：
- **初始化**：在检测框内随机采样 $N$ 个像素
- **匹配**：预测像素位置/可见性后，评估预测像素与新帧检测框内像素的空间对应
- **动态更新**：持续在框外的像素被剔除（outlier pruning），在当前框内补充新像素（inlier replenishment），维持鲁棒表示

## 实验

### 表1：真实世界点跟踪性能（TAP-Vid DAVIS / Kinetics / RoboTAP）

| 模型 | 参数 | 类型 | DAVIS AJ↑ | DAVIS $\delta_{avg}^x$↑ | DAVIS OA↑ | Kinetics AJ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| TAPIR | 31M | 离线 | 56.2 | 70.0 | 86.5 | 49.6 |
| LocoTrack | 12M | 离线 | 62.9 | 75.3 | 87.2 | 52.9 |
| CoTracker3 | 25M | 窗口在线 | 64.5 | 76.7 | 89.7 | 54.1 |
| Track-On | 49M | 在线 | 65.0 | 78.0 | 90.8 | 53.9 |
| **LBM** | **18M** | **在线** | **65.1** | **77.5** | **89.5** | **53.4** |

LBM 以仅 18M 参数达到 SOTA 在线性能（DAVIS AJ 65.1），仅为 Track-On 参数量的 37%，同时超越大多数离线和窗口在线方法。

### 边缘设备效率

在 NVIDIA Jetson Orin NX Super 上：LBM 达到 **14.3 FPS** 实时推理，速度是 Track-On 的 **3.9 倍**。

### 表2：开放世界物体跟踪（TAO 验证集）

| 模型 | 额外训练 | TETA↑ | LocA↑ | AssocA↑ |
|:---|:---:|:---:|:---:|:---:|
| MASA | 是 | 37.1 | 51.8 | 35.8 |
| NetTrack | 否 | 36.1 | 50.2 | 31.0 |
| **LBM** | **否** | **37.4** | **51.7** | **35.1** |

LBM 无需物体跟踪领域数据训练，即可达到 SOTA 水平（TETA 37.4），超越需要额外训练的 MASA。

### BFT 和 OVT-B 数据集

LBM 在 BFT 上 OWTA 达 50.3、在 OVT-B 上 TETA 达 41.2，均为无额外训练方法中的最优。

## 亮点

- **物理类比巧妙**：将像素运动建模为流体格子的碰撞-流式过程，理论类比清晰且工程实现高效
- **极致效率**：18M 参数 + 边缘设备实时推理（14.3 FPS），在所有在线方法中参数效率最高
- **多层级使用统一框架**：同一 LBM 既做点跟踪又做物体跟踪，通过像素分解和动态剪枝无缝衔接
- **无需训练即可迁移**：在 TAO/BFT/OVT-B 等物体跟踪数据集上零样本超越专门训练的方法

## 局限性

- **在线方法的固有限制**：无法利用未来帧信息进行双向优化，在严重遮挡恢复上可能弱于离线方法
- **ResNet18 编码器容量有限**：为效率选择轻量编码器，在复杂纹理场景下特征表达力可能不足
- **TAP-Vid Kinetics 上略低于 Track-On**：AJ 53.4 vs 53.9，说明在更多样的视频场景下仍有改进空间
- **固定历史长度 $N_s$**：时序上下文窗口固定，对极长遮挡的适应性可能受限

## 相关工作

- **离线点跟踪**（TAPIR [Doersch+ 2023]、LocoTrack [Cho+ 2024]、CoTracker3 [Karaev+ 2024]）：性能强但延迟高、无法实时
- **在线点跟踪**（MFT [Neoral+ 2024]、Track-On [Aydemir+ 2025]、DOT [Le Moing+ 2024]）：LBM 在效率上大幅领先
- **开放世界物体跟踪**（MASA [Li+ 2024]、NetTrack [Zheng+ 2024]）：LBM 将点跟踪优势泛化至物体跟踪
- **格子玻尔兹曼方法**（[Mohamad 2011]）：经典流体仿真方法，LBM 是其在视觉跟踪中的首次应用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 从流体力学引入格子玻尔兹曼类比，在线跟踪范式新颖
- 实验充分度: ⭐⭐⭐⭐ — 覆盖点跟踪和物体跟踪两大任务 + 边缘设备效率测试
- 写作质量: ⭐⭐⭐⭐ — 物理类比阐述清晰，框架图直观
- 价值: ⭐⭐⭐⭐⭐ — 为实时在线跟踪提供了高效且统一的解决方案

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks](egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)
- [LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](../../CVPR2026/video_understanding/realworld_point_tracking_with_verifierguided_pseud.md)
- [PiTe: Pixel-Temporal Alignment for Large Video-Language Model](../../ECCV2024/video_understanding/pite_pixel-temporal_alignment_for_large_video-language_model.md)
- [OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](../../CVPR2025/video_understanding/ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)

<!-- RELATED:END -->
