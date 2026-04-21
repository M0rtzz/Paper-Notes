---
title: >-
  [论文解读] CryptoFace: End-to-End Encrypted Face Recognition
description: >-
  [人体理解] 提出 CryptoFace，首个端到端全同态加密（FHE）人脸识别系统，通过混合浅层 patch CNN 架构（CryptoFaceNet）大幅降低乘法深度，实现比 SOTA FHE 网络快 7 倍的加密推理，同时提升验证精度。
tags:
  - 人体理解
---

# CryptoFace: End-to-End Encrypted Face Recognition

| 属性 | 值 |
|------|------|
| 会议 | CVPR 2025 |
| arXiv | [2509.00332](https://arxiv.org/abs/2509.00332) |
| 代码 | [GitHub](https://github.com/human-analysis/CryptoFace) |
| 领域 | 人体理解 / 人脸识别 / 隐私保护 |
| 关键词 | fully homomorphic encryption, face recognition, patch CNN, privacy-preserving, CryptoFaceNet |

## 一句话总结

提出 CryptoFace，首个端到端全同态加密（FHE）人脸识别系统，通过混合浅层 patch CNN 架构（CryptoFaceNet）大幅降低乘法深度，实现比 SOTA FHE 网络快 7 倍的加密推理，同时提升验证精度。

## 研究背景与动机

### 领域现状

人脸识别广泛应用于设备解锁、执法和金融服务，但面临严重隐私风险。生物特征数据不可更改，一旦泄露后果不可逆。同态加密（HE）允许在加密数据上计算，提供可证明的后量子安全性。现有安全人脸识别系统仅加密特征而非原始图像。

### 现有痛点

1. **不完整的安全保护**：现有系统只加密特征阶段，客户端需在本地执行特征提取，无法将计算委托给服务器。恶意客户端可通过模板恢复攻击推断服务器数据库中的特征
2. **FHE CNN 推理极慢**：SOTA FHE 网络（MPCNN、AutoFHE）的高乘法深度导致需要大量 bootstrapping 操作，推理时间高达数小时
3. **无法处理高分辨率人脸图像**：现有 FHE CNN 仅支持小分辨率（如 CIFAR 的 32×32），无法直接处理人脸图像
4. **余弦相似度无法在 FHE 下直接计算**：$\ell_2$ 归一化涉及非同态的非线性运算

### 本文目标

构建首个**端到端**加密人脸识别系统：从加密图像输入到加密匹配结果输出，全程不解密，同时实现可接受的推理延迟和高识别精度。

### 切入角度与核心 idea

将人脸图像切分为 patch，用多个浅层 CNN 独立处理每个 patch，降低单个网络的乘法深度（仅需 1 次 bootstrapping）。多个 patch CNN 在 FHE 下并行评估，实现近似分辨率无关的推理延迟。

## 方法详解

### 整体框架

CryptoFace 系统分为离线阶段（注册参考人脸）和在线阶段（验证探测人脸）。CryptoFaceNet 将人脸图像切分为 $L$ 个 patch，每个 patch 由独立的浅层 PCNN 处理，局部特征经线性融合得到全局特征，随后用多项式近似的余弦相似度在加密域完成匹配。

### 关键设计 1：混合浅层 Patch CNN (CryptoFaceNet)

- **功能**：在 FHE 兼容约束下高效提取人脸特征
- **核心思路**：将图像 $x \in \mathbb{R}^{C \times H \times W}$ 切分为 $L = HW/P^2$ 个 patch，每个 patch $x_i \in \mathbb{R}^{C \times P \times P}$ 由独立的 PCNN $f_{\omega_i}$ 处理。各 PCNN 不共享权重，独立学习不同面部区域的特征。特征融合为简单的矩阵乘法 $y = y'A^T + b$
- **设计动机**：(1) Patch 分辨率远小于原图（$P \ll H$），允许使用更浅的网络（更低乘法深度）；(2) 各 PCNN 在 FHE 下并行评估，不增加推理时间；(3) 融合矩阵 $A$ 被分解为 $L$ 个方阵的块操作，避免大矩阵 FHE 乘法的高开销
- **辅助任务**：Jigsaw puzzle 任务，用局部特征预测 patch 原始位置，为特征注入位置信息
- **损失函数**：$\mathcal{L} = \mathcal{L}_{\text{ArcFace}}(\omega, W, A, b) + \alpha \mathcal{L}_{\text{Jigsaw}}(\omega)$，$\alpha = 0.005$

### 关键设计 2：深度优化的卷积块

- **功能**：最小化 FHE 乘法深度
- **核心思路**：将 AESPA 的 Hermite 多项式激活 $ax^2 + bx + c$（深度 2）的系数 $a$ 融合到卷积权重中，变为 $x^2 + \frac{b}{a}x + \frac{c}{a}$（深度 1），每个 block 节省 2 个乘法层级
- **设计动机**：bootstrapping 是 FHE 中最耗时的操作（比普通乘法慢 100 倍），每消耗完可用层级就需要一次 bootstrapping。深度优化使得整个 CryptoFaceNet 仅需 1 次 bootstrapping（对比 MPCNN 需要 31-43 次）

### 关键设计 3：分布感知的多项式 $\ell_2$ 归一化

- **功能**：在 FHE 下近似计算余弦相似度所需的 $\ell_2$ 归一化
- **核心思路**：用二次多项式 $p(t) = \beta_2 t^2 + \beta_1 t + \beta_0$ 近似 $q(t) = 1/\sqrt{t}$，其中 $t = \|y\|_2^2$。三个控制点选取为 $t_1 = \text{Mean}(t) - \text{Std}(t)$, $t_2 = \text{Mean}(t)$, $t_3 = \text{Mean}(t) + \text{Std}(t)$
- **设计动机**：传统 minimax 近似需要高次多项式（高乘法深度），Taylor 展开在宽域上不准确。分布感知的控制点选择使得二次多项式即可实现 $|p(t) - q(t)| \leq 2^{-10}$ 的近似精度，仅消耗 2 层乘法深度

## 实验关键数据

### 主实验表（端到端加密人脸识别，64×64）

| 方法 | Backbone | Avg Acc(%) | 延迟(s) | RAM | #Boot |
|------|----------|-----------|---------|-----|-------|
| MPCNN | ResNet44 | 89.64 | 9,845 | 286G | 43 |
| MPCNN | ResNet32 | 85.60 | 7,367 | 286G | 31 |
| AutoFHE | ResNet32 | 82.69 | 4,001 | 286G | 8 |
| **CryptoFace** | CFNet4 | 89.42 | **1,364** | **269G** | **1** |

**关键数字**：
- 比 MPCNN-ResNet44 快 **7.2×**（节省 8,481 秒），精度仅降 0.22%
- 比 AutoFHE 快 **2.9×**，精度提升 **+6.73%**

### 分辨率扩展性

| 分辨率 | 模型 | Avg Acc(%) | 延迟(s) |
|--------|------|-----------|---------|
| 64×64 | CFNet4 | 89.42 | 1,364 |
| 96×96 | CFNet9 | 90.99 | 1,395 |
| 128×128 | CFNet16 | 91.46 | 1,446 |

分辨率从 64 提升到 128 时，精度提升 +2.04%，延迟仅增加 82 秒（**近分辨率无关**）。

### 操作延迟分析

Bootstrapping 在 MPCNN 中占 ~70% 时间，在 CryptoFace 中仅占 ~10%。卷积操作占主导（~63%），并行化开销 <3.26%。

### 关键发现

- 1:128 闭集检索任务中 Rank-1 准确率 92.19%，比 MPCNN 高 +3.91%
- IJB-B/IJB-C 困难基准上 AUC 一致性优于基线
- 多项式 $\ell_2$ 近似延迟仅 0.3 秒，占推理时间 0.02%

## 亮点与洞察

1. **首个端到端加密 FR 系统**：从加密图像到加密匹配结果，全程无解密，安全性无死角
2. **Patch 并行化策略极其巧妙**：将高分辨率问题转化为多个低分辨率并行问题，一石多鸟——降低深度、支持并行、实现分辨率无关
3. **深度优化的卷积块**：将激活函数系数融合到卷积权重的技巧，简单但效果显著
4. **分布感知的多项式近似**：优美地回避了 FHE 下非线性函数近似的传统难题

## 局限性

1. 半诚实安全模型假设较弱，不能防御恶意行为方
2. Patch-based 设计丢失了全局上下文信息，在极端遮挡/姿态变化下性能可能下降
3. 即使优化后仍需 ~23 分钟在线推理，距实时应用尚有差距
4. 训练时使用 cleartext 数据，安全性保证仅限推理阶段

## 相关工作与启发

- **MPCNN**（2023）：multiplexed convolution 的 FHE 卷积方案，CryptoFace 复用其卷积实现
- **AESPA**（2023）：低阶 Hermite 多项式激活函数，CryptoFace 进一步优化深度
- **AutoFHE**（USENIX Security 2024）：搜索 FHE 架构的方法，CryptoFace 证明手工设计也能更好
- **启发**：Patch 分而治之的思路可推广到其他隐私保护 CV 任务（目标检测、分割等）

## 评分

⭐⭐⭐⭐ — 系统工程和密码学架构设计出色，首次实现端到端加密 FR 有开创意义。7 倍加速和分辨率无关性是强有力的贡献，但实际部署仍受限于 FHE 固有的高延迟

<!-- RELATED:START -->

## 相关论文

- [ControlFace: Harnessing Facial Parametric Control for Face Rigging](controlface_harnessing_facial_parametric_control_for_face_rigging.md)
- [Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)
- [DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)
- [CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](../../ICML2025/human_understanding/sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)

<!-- RELATED:END -->
