---
title: >-
  [论文解读] Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition
description: >-
  [ECCV 2024][视频理解][第一人称动作识别] 提出 EVI-MAE，首个联合第一人称视频与身体穿戴 IMU 的多模态表示学习方法，通过 MAE 自监督预训练学习视频-IMU 跨模态对齐，并用图神经网络建模多 IMU 设备间的协同运动关系，在动作识别中取得 SOTA 且具备优秀的鲁棒性。
tags:
  - ECCV 2024
  - 视频理解
  - 第一人称动作识别
  - 惯性测量单元 (IMU)
  - 多模态掩码自编码器
  - 图神经网络
  - 自监督学习
---

# Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition

**会议**: ECCV 2024  
**arXiv**: [2407.06628](https://arxiv.org/abs/2407.06628)  
**代码**: 无  
**领域**: 视频理解 / 动作识别  
**关键词**: 第一人称动作识别, 惯性测量单元 (IMU), 多模态掩码自编码器, 图神经网络, 自监督学习

## 一句话总结

提出 EVI-MAE，首个联合第一人称视频与身体穿戴 IMU 的多模态表示学习方法，通过 MAE 自监督预训练学习视频-IMU 跨模态对齐，并用图神经网络建模多 IMU 设备间的协同运动关系，在动作识别中取得 SOTA 且具备优秀的鲁棒性。

## 研究背景与动机

第一人称 (egocentric) 动作识别是理解人类行为的关键研究方向。相比视觉信号，佩戴在人体肢体上的 IMU 传感器能精确捕获运动信号，且不受光照和遮挡影响。然而两种模态各有局限。

**现有方法的局限性**：

**IMU 利用不足**：多数第一人称视频数据集仅有相机 IMU (头部安装)，提供的运动线索可从视频推断，无法捕捉四肢运动。含身体穿戴 IMU 的数据集非常稀少 (最大仅 9 小时标注视频)

**多设备关系建模困难**：单个 IMU 仅能提供一个身体部位的运动，而动作识别需要对多肢体运动的整体理解。现有 IMU 方法 (DeepConvLSTM, LIMU-BERT) 忽视了多设备间的协作关系

**标注数据稀缺**：同步的视频-身体 IMU 标注数据极少，直接套用全监督多模态框架效果受限

**核心动机**：
- 第一人称视频与身体 IMU 存在天然关联：上肢运动反映在画面中的手部运动，全局运动对应下肢 IMU 信号
- 利用大量无标注数据进行自监督预训练可弥补标注数据不足
- 多个 IMU 设备反映人体关节的相对运动，天然适合图结构建模

## 方法详解

### 整体框架

EVI-MAE 包含两个阶段：(1) **MAE 自监督预训练**：双分支结构——多模态像素重建分支学习视频-IMU 跨模态对齐，IMU 特征重建分支学习多设备协作关系；(2) **微调**：丢弃解码器，仅用编码器 + 线性分类器进行动作分类。

输入：$T=2$ 秒的同步视频 $D_v \in \mathbb{R}^{T \times S_v \times H \times W \times 3}$ 和 $N_{imu}=4$ 个 IMU 信号 $D_{raw} \in \mathbb{R}^{N_{imu} \times T \times S_{imu} \times 3}$。

### 关键设计

1. **IMU 数据预处理与图结构建模**：解决多 IMU 协作关系

    - IMU 原始加速度信号经重采样、归一化后，通过 STFT 转为频谱图 $D_{spec} \in \mathbb{R}^{T_{imu} \times M_{imu}}$，再分为 $16 \times 16$ patches
    - 各 IMU 的频谱图 patches 经 IMU 编码器提取特征后，构建 IMU 特征图 $\mathcal{G} = (\mathcal{V}, A, f_d)$
    - 节点 $\mathcal{V}$ 对应 $N_{imu}$ 个 IMU 设备，邻接矩阵 $A$ 设为全连接（动作通常需要多肢体协调）
    - **设计动机**：灵感来自骨架动作识别 (ST-GCN 等)，IMU 设备类似于骨架关节点，图结构可捕捉不同身体部位运动间的关系

2. **多模态像素重建分支**：学习视频-IMU 跨模态对齐

    - 视频采用 tube masking (掩码率 $R_v = 90\%$)，IMU 频谱图采用非结构化随机掩码 (掩码率 $R_{imu} = 75\%$)
    - 可见 patches 分别通过视频编码器 (ViT + 时空注意力) 和 IMU 编码器，再经统一编码器融合
    - 解码器在掩码位置填入可学习 mask token，重建原始视频帧和 IMU 频谱图：$O_p = \mathcal{D}_{uni}([e_{imu}, e_v] + [m'_{imu}, m'_v] + [p'_{imu}, p'_v])$
    - **设计动机**：通过从一种模态的可见部分重建另一种模态的被掩码部分，迫使模型学习视频与 IMU 的天然关联

3. **IMU 特征重建分支**：节点级自监督学习多设备关系

    - 将 IMU 特征图中比例 $R_g$ 的节点特征替换为 mask token，形成损坏图 $f_{dc}$
    - 用 GIN (Graph Isomorphism Network) 编码器-解码器重建原始图：$f_g = \text{GraphEnc}(A, f_{dc}),\ \hat{\mathcal{G}} = \text{GraphDec}(A, f_g)$
    - **设计动机**：通过从其余设备特征推断被掩码设备，学习多 IMU 间的协同运动模式

### 损失函数 / 训练策略

预训练总损失由三项组成：$L = \alpha L_{mse} + \beta L_{cos} + \gamma L_{con}$

- **$L_{mse}$**：像素重建的 MSE 损失 (视频帧 + IMU 频谱图)
- **$L_{cos}$**：IMU 特征图重建的余弦相似度损失：$L_{cos} = \frac{1}{|\mathcal{V}_c|} \sum_{\nu_n \in \mathcal{V}_c} (1 - \frac{(\hat{f}_d^{\nu_n})^T f_d^{\nu_n}}{\|\hat{f}_d^{\nu_n}\| \cdot \|f_d^{\nu_n}\|})$
- **$L_{con}$**：视频-IMU 跨模态对比损失 (InfoNCE 风格)
- 超参数：$\alpha=1, \beta=10, \gamma=0.01, \tau=0.05$

## 实验关键数据

### 主实验

与 SOTA 方法在 CMU-MMAC 和 WEAR 数据集上的比较：

| 方法 | 模态 | 预训练 | CMU-MMAC (mAP) | WEAR (Acc) |
|------|------|--------|----------------|------------|
| DeepConvLSTM | IMU | ✗ | 7.46 | 74.37 |
| LIMU-BERT | IMU | ✓ | 15.30 | 79.60 |
| **Ours (IMU only)** | IMU | ✓ | **31.68** | **86.53** |
| VideoMAE | Video | ✓ | 84.63 | 88.47 |
| **Ours (Video only)** | Video | ✓ | **85.07** | **89.78** |
| AV-MAE | IMU-Video | ✓ | 84.75 | 91.60 |
| **Ours (IMU-Video)** | IMU-Video | ✓ | **87.96** | **92.78** |

关键提升：IMU only 模态下，对比 LIMU-BERT 提升了 **16.38%** mAP (CMU-MMAC)。

### 消融实验

组件消融 (CMU-MMAC / WEAR)：

| 模态 | 预训练 | IMU 图 | CMU-MMAC | WEAR |
|------|--------|--------|----------|------|
| IMU | ✗ | ✗ | 24.87 | 79.66 |
| IMU | ✗ | ✓ | 27.36 | 83.55 |
| IMU | ✓ | ✗ | 28.96 | 85.29 |
| IMU | ✓ | ✓ | **31.68** | **86.53** |
| IMU-Video | ✗ | ✗ | 71.53 | 90.93 |
| IMU-Video | ✓ | ✓ | **87.96** | **92.78** |

IMU 设备缺失鲁棒性 (遮掩 2/4 个 IMU)：

| 方法 | 正常 (CMU) | 缺失 (CMU) | 下降幅度 |
|------|-----------|-----------|----------|
| ADCL | 8.59 | 5.25 | ↓38% |
| LIMU-BERT | 15.30 | 11.40 | ↓25% |
| **Ours (IMU)** | 31.68 | 26.47 | **↓16%** |
| **Ours (IMU-Video)** | 87.96 | 85.86 | **↓3%** |

### 关键发现

- **图结构建模有效**：无论是否有预训练，引入 IMU 特征图均一致提升性能 (CMU-MMAC: +2.49% 无预训练, +2.72% 有预训练)
- **高掩码率适用于 IMU**：75%-90% 的掩码率效果最佳，因为人体运动信号存在周期性重复
- **非结构化随机掩码略优于结构化掩码** (时间/频率分离掩码)
- **跨数据集泛化**：从 WEAR 预训练迁移到 CMU-MMAC 微调，在所有模态下均优于从零训练
- **低光照鲁棒性**：多模态模型在视频质量退化时可自动将依赖转移到 IMU 模态

## 亮点与洞察

- 首个联合第一人称视频与身体穿戴 IMU 的多模态 MAE 预训练方法
- 巧妙利用 IMU 设备的空间分布构建图结构，借鉴骨架动作识别的思路但应用于原始传感器信号
- 实验设计考虑周全：不仅评估标准准确率，还测试了 IMU 缺失、视频退化、跨数据集等实际挑战场景
- IMU only 模态下的巨大提升 (15.30 → 31.68) 展示了自监督预训练对小规模数据的显著增益

## 局限与展望

- 当前仅使用 4 个 IMU，图结构全连接，更多 IMU 时需设计更精细的图拓扑
- CMU-MMAC 数据集中 IMU only 准确率仍较低 (31.68%)，因涉及物体/环境交互，IMU 无法感知
- 当前 GIN 编码器-解码器相对简单，可探索更强的图学习方法 (如 GAT、图 Transformer)
- 可扩展到更多模态 (如音频、眼动追踪) 的联合学习

## 相关工作与启发

- 与 VideoMAE 相比，EVI-MAE 引入了互补的 IMU 模态和图结构，在标准设置和挑战场景下均更优
- MAE 框架在视频 (VideoMAE)、音频视频 (AV-MAE)、点云 (Point-MAE) 等多种模态已验证有效，本文扩展到 IMU-视频组合
- IMU 特征图的设计可推广到其他多传感器融合场景 (如自动驾驶中的多 LiDAR)

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将身体穿戴 IMU 与视频进行 MAE 联合预训练，图建模 IMU 关系有创意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多数据集 + 详细消融 + 三种挑战场景 + 跨数据集泛化，极其全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，但部分符号可以更简化
- **价值**: ⭐⭐⭐⭐ — 为穿戴式设备的动作识别提供了有效的多模态学习范式

<!-- RELATED:START -->

## 相关论文

- [Text-Guided Video Masked Autoencoder](text-guided_video_masked_autoencoder.md)
- [Data Collection-Free Masked Video Modeling](data_collection-free_masked_video_modeling.md)
- [Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)
- [Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective](occluded_gait_recognition_with_mixture_of_experts_an_action_detection_perspectiv.md)

<!-- RELATED:END -->
