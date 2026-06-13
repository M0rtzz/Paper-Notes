---
title: >-
  [论文解读] ETAP: Event-based Tracking of Any Point
description: >-
  [CVPR 2025][视频理解][事件相机] 本文提出首个纯事件相机的任意点跟踪（TAP）方法 ETAP，通过新颖的特征对齐对比损失解决事件数据运动依赖性问题，并基于新构建的合成数据集 EventKubric 训练，在多个数据集上大幅超越基线方法（AJ 指标提升 136%）。 领域现状：任意点跟踪（TAP）是近年来运动估计…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "事件相机"
  - "任意点跟踪"
  - "对比学习"
  - "特征对齐"
  - "运动鲁棒性"
---

# ETAP: Event-based Tracking of Any Point

**会议**: CVPR 2025  
**arXiv**: [2412.00133](https://arxiv.org/abs/2412.00133)  
**代码**: [https://github.com/tub-rip/ETAP](https://github.com/tub-rip/ETAP)  
**领域**: 视频理解  
**关键词**: 事件相机, 任意点跟踪, 对比学习, 特征对齐, 运动鲁棒性

## 一句话总结
本文提出首个纯事件相机的任意点跟踪（TAP）方法 ETAP，通过新颖的特征对齐对比损失解决事件数据运动依赖性问题，并基于新构建的合成数据集 EventKubric 训练，在多个数据集上大幅超越基线方法（AJ 指标提升 136%）。

## 研究背景与动机

**领域现状**：任意点跟踪（TAP）是近年来运动估计的重要范式转变，从关注单个显著性特征点转向追踪任意点，代表性方法如 CoTracker、TAPIR 等已在常规场景中取得出色表现。

**现有痛点**：现有 TAP 方法全部基于传统帧式相机，在极端光照条件和高速运动场景下严重受限。传统相机的固有缺陷——有限帧率、运动模糊和饱和伪影——导致视觉混叠和算法退化，这在机器人感知等实际应用中是关键瓶颈。

**核心矛盾**：事件相机凭借高时间分辨率（μs 级）和高动态范围（HDR）天然适合高速跟踪，但事件数据有一个根本挑战：特征外观依赖于场景运动方向。同一场景在不同运动方向下产生的事件数据截然不同，这使得基于特征相关性的跟踪方法难以直接适用。此外，训练数据方面也面临挑战：现有事件相机合成数据集过于简单，仅使用 2D 平面运动，真实世界泛化性差。

**本文目标**：（1）构建首个基于事件相机的 TAP 方法；（2）解决事件特征的运动依赖性问题；（3）构建高质量合成训练数据集。

**切入角度**：作者观察到事件数据在时间反转下的数学性质——时间反转改变运动方向但保留场景结构——并利用这一特性设计对比损失来强制学习运动不变的特征。

**核心 idea**：通过时间反转产生同一场景不同运动方向的数据对，用对比损失约束对应点的特征描述子在不同运动下保持一致，从而学习运动鲁棒的相关特征。

## 方法详解

### 整体框架
ETAP 的整体流程为：输入事件数据流 → 转换为事件栈表示（image-like tensor）→ 多尺度特征编码器提取空间特征 → 基于 Transformer 的迭代优化模块更新点位置和描述子 → 输出各点轨迹、可见性标志和描述子。在训练时，额外生成时间反转+旋转的变体数据，用于计算特征对齐损失。

### 关键设计

1. **事件栈表示与多尺度特征编码**:

    - 功能：将异步稀疏的事件数据转换为与 CNN 兼容的规则网格表示
    - 核心思路：采用 Mixed-Density 事件栈，将每个时间步前的 $N_e$ 个事件分层分 bin 到 $C=10$ 个通道中，每个通道 $h_c$ 聚合 $N_e/2^{c-1}$ 个事件，形成从细粒度到粗粒度的多尺度时间信息。特征编码器 $\phi_\lambda$ 在 4 个尺度上提取 $d$ 维特征图，用于初始化点描述子和计算相关特征
    - 设计动机：层级化的时间 bin 设计既保留近期事件的精细时间信息，又覆盖更长时间范围的上下文，比简单的体素网格更有效（消融实验证实略优于 voxel grid）

2. **特征对齐对比损失（FA-loss）**:

    - 功能：强制特征编码器学习运动不变的描述子
    - 核心思路：对每个训练样本生成时间反转 + 随机旋转（$\theta \in \{0, 90°, 180°, 270°\}$）的变体，保持场景结构不变但改变运动方向。从原始和变体中提取对应点的描述子 $d_{t}^{s,i}$ 和 $\tilde{d}_{t}^{s,i}$，通过最小化它们归一化后的余弦相似度损失 $\mathcal{L}_{fa} = \sum_t \frac{1}{|\mathcal{P}_t|} \sum_{i,s} (1 - \langle u(d), u(\tilde{d}) \rangle)^2$ 来对齐特征。数学上，时间反转下事件虽不同但触发条件等价（由线性事件生成模型推导），因此对应点描述子理应一致
    - 设计动机：这是纯事件跟踪的核心挑战——事件依赖运动方向会导致相关特征随时间退化。对比损失提供显式的运动不变性约束，实验证明可将特征的 inter-cluster 和 intra-cluster 相似度差距从 0.38 降到 0.067

3. **Transformer 迭代优化跟踪器**:

    - 功能：并行跟踪多个点，迭代更新位置和描述子
    - 核心思路：遵循 CoTracker 架构，构建每个点在每个时间步的 token $\mathcal{O}_t^{s,i,m}$，包含位移、可见性、描述子、相关特征和位置编码。通过交替的 intra-point attention（跨点）和 temporal attention（跨时间）进行 $M=4$ 次迭代优化。相关特征在 $49 \times 4 = 196$ 维空间中通过描述子与周围特征图的内积计算
    - 设计动机：并行多点跟踪可利用点间空间关系（如刚体约束），交替注意力机制在保持效率的同时捕获时空依赖

### 损失函数 / 训练策略
总损失为 $\mathcal{L} = 0.1 \mathcal{L}_{tp} + \mathcal{L}_{vis} + 0.1 \mathcal{L}_{fa}$，其中 $\mathcal{L}_{tp}$ 为轨迹预测误差（绝对差），$\mathcal{L}_{vis}$ 为可见性交叉熵。训练分两阶段：前 $10^5$ 步仅优化轨迹和可见性损失，随后加入 FA-loss 再训练 $1.2 \times 10^5$ 步。训练数据来自 EventKubric 数据集（10173 样本），通过 Kubric 渲染 + FILM 上采样 + ESIM 事件模拟三步流程生成。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | ETAP | E2Vid+CoTracker | 提升 |
|------------|------|------|-----------------|------|
| TAP / EventKubric | AJ | 0.539 | 0.229 | +136% |
| TAP / EventKubric | $\delta_{avg}^x$ | 0.668 | 0.328 | +104% |
| TAP / E2D2 (fidget spinner) | AJ | 0.389 | 0.179 | +117% |
| Feature Tracking / EDS | Feature Age | 0.701 | - | - |
| Feature Tracking / EDS | Expected FA | 0.610 | - | - |

| 方法 | 输入 | EDS FA↑ | EDS EFA↑ | EC FA↑ | EC EFA↑ |
|------|------|---------|----------|--------|---------|
| ETAP (Ours) | E | **0.701** | **0.610** | **0.891** | **0.886** |
| FE-TAP (E+F) | E+F | 0.676 | 0.589 | 0.844 | 0.838 |
| DDFT (E+F) | E+F | 0.576 | 0.472 | 0.825 | 0.818 |
| HASTE (E) | E | 0.096 | 0.063 | 0.442 | 0.427 |

### 消融实验

| 配置 | EDS FA↑ | EDS EFA↑ | EC FA↑ | 说明 |
|------|---------|----------|--------|------|
| ETAP 完整模型 | 0.701 | 0.610 | 0.891 | 所有设计决策最优组合 |
| w/o FA-loss | 0.686 | 0.593 | 0.887 | 去掉对比损失掉 2.1% |
| 低分辨率 256×256 | 0.598 | 0.500 | 0.780 | 分辨率影响最大 |
| 高分辨率 512×512 | 0.659 | 0.561 | 0.808 | 提升明显 |
| MOVi-F 基线数据 | 0.598 | 0.500 | 0.780 | EventKubric 比 MOVi-F 环境提升 8% |

### 关键发现
- 分辨率是影响最大的因素，从 256 到 512 带来约 10% FA 提升
- 随机对比度阈值 $\sim \mathcal{U}(0.16, 0.34)$ 比固定值提升约 5%
- EventKubric 比 MOVi-F 预渲染数据集提升 8%，验证了高质量合成数据的重要性
- FA-loss 在 EDS 上提升 2.1%，且特征独立性实验表明其有效缩小运动方向间的特征差距
- ETAP 是首个在 Feature Tracking benchmark 上超越 events+frames 联合方法的纯事件方法

## 亮点与洞察
- **时间反转产生训练对**的思路非常优雅：利用事件生成模型的数学性质，不需要额外数据或标注就能获得运动变体对，且物理上有坚实的理论基础。这一思路可泛化到任何运动依赖的传感器数据
- **系统性的数据工程**：不仅构建了新的合成数据pipeline，还对每个设计决策（分辨率、帧率、阈值、场景动态）进行了细致消融，展示了如何通过数据工程来提升模型性能
- **跨模态超越**：纯事件方法在 feature tracking benchmark 上超越了使用帧+事件的方法（FE-TAP），证明了事件相机在高速追踪场景中的独特优势

## 局限与展望
- 事件相机目前只能提供单色信息，无法利用颜色线索建立外观对应
- 在无运动（无事件）时初始化的跟踪特征质量差，这是事件数据的固有问题
- EventKubric 虽然比之前的合成数据更真实，但仍与真实事件数据有 sim-to-real gap
- 可能的改进方向：结合少量帧信息进行特征初始化，或在检测到运动后重新初始化特征

## 相关工作与启发
- **vs CoTracker**: CoTracker 是帧式 TAP 的 SOTA，ETAP 采用类似的 Transformer 架构但适配事件输入。两者在无挑战间段表现接近，但 ETAP 在高速/HDR 场景中占优
- **vs DDFT**: DDFT 是之前最强的事件特征跟踪方法，但训练数据只使用简单 2D 平面运动，且需要自监督微调。ETAP 用更真实的 3D 数据和 FA-loss 实现显著超越（19%）
- **vs FE-TAP**: FE-TAP 结合帧和事件进行相关性跟踪，但继承了帧在高速场景下的缺陷。ETAP 纯事件方案反而更鲁棒

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个事件TAP方法，FA-loss设计新颖且有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集、2个任务、8个表格、详尽消融
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、数学推导完整
- 价值: ⭐⭐⭐⭐ 填补事件相机TAP空白，对机器人高速感知有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video](../../NeurIPS2025/video_understanding/tapvid-360_tracking_any_point_in_360_from_narrow_field_of_view_video.md)
- [\[ECCV 2024\] Self-Supervised Any-Point Tracking by Contrastive Random Walks](../../ECCV2024/video_understanding/self-supervised_any-point_tracking_by_contrastive_random_walks.md)
- [\[CVPR 2025\] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)
- [\[CVPR 2025\] Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](holmes-vau_towards_long-term_video_anomaly_understanding_at_any_granularity.md)
- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](../../ICCV2025/video_understanding/online_dense_point_tracking_with_streaming_memory.md)

</div>

<!-- RELATED:END -->
