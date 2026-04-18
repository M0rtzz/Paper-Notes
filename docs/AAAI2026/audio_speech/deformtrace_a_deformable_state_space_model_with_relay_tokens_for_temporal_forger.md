---
title: >-
  [论文解读] DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization
description: >-
  [AAAI2026][语音][Temporal Forgery Localization] 提出 DeformTrace，将可变形动态感受野和中继令牌机制引入状态空间模型，结合 Transformer 的全局建模与 SSM 的高效推理，实现时序伪造定位的 SOTA 精度与显著效率提升。
tags:
  - AAAI2026
  - 语音
  - Temporal Forgery Localization
  - State Space Model
  - Deformable Mechanism
  - Relay Token
  - Deepfake Detection
---

# DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization

**会议**: AAAI2026  
**arXiv**: [2603.04882](https://arxiv.org/abs/2603.04882)  
**代码**: 待确认  
**领域**: video_understanding  
**关键词**: Temporal Forgery Localization, State Space Model, Deformable Mechanism, Relay Token, Deepfake Detection

## 一句话总结
提出 DeformTrace，将可变形动态感受野和中继令牌机制引入状态空间模型，结合 Transformer 的全局建模与 SSM 的高效推理，实现时序伪造定位的 SOTA 精度与显著效率提升。

## 背景与动机
时序伪造定位（Temporal Forgery Localization, TFL）旨在精确识别视频和音频中被篡改的时间段，比二分类的伪造检测提供更细粒度的可解释性，对安全取证至关重要。现有 TFL 方法（如 BA-TFD+、UMMAFormer、DiMoDif）主要依赖 CNN 或多尺度 Transformer，虽然精度逐步提升，但计算开销大、推理速度慢。

状态空间模型（SSM）尤其是 Mamba 系列在长序列建模上展现了线性复杂度和更快推理的优势，但直接应用于 TFL 面临三个核心挑战：

1. **边界模糊性**：伪造边界不像动作检测那样明确，标准 SSM 的固定状态更新会产生时序平滑效应，降低定位精度
2. **伪造稀疏性**：大部分帧是真实的，SSM 的递归更新被非伪造模式主导，削弱了对稀疏伪造的敏感度
3. **长程衰减**：SSM 虽然对长序列高效，但信息随距离指数衰减，限制了捕获远距离上下文的能力

## 核心问题
如何设计一种既能利用 SSM 线性复杂度优势，又能克服其在边界模糊、伪造稀疏和长程衰减方面固有缺陷的 TFL 架构？

## 方法详解

### 整体架构
DeformTrace 基于 TadTR 的 query-based 架构，包含三个主要模块：

- **多尺度音视频特征提取**：使用冻结的 Raven 预训练编码器提取视觉和音频特征，通过拼接和线性投影得到融合特征，再经 $L-1$ 个下采样层生成 $L=6$ 级多尺度特征
- **可变形编码器**：包含 Deformable Self-SSM (DS-SSM)、Deformable Self-Attention 和 FFN
- **可变形解码器**：包含 Deformable Cross-SSM (DC-SSM)、Multi-Head Self-Attention、Deformable Cross-Attention 和 FFN，$M=3$ 层

### Deformable Self-SSM (DS-SSM) — 解决边界模糊
首次将可变形动态感受野机制引入时序状态空间模型。核心思想：

1. 为每个特征 $f_n^l$ 计算归一化时序参考点 $p_n^l$，将特征索引映射到视频时间轴上
2. 通过 MLP 预测偏移矩阵 $o \in \mathbb{R}^{L \times N_s}$（跨尺度偏移），加到参考点上形成可变形采样点
3. 用双线性插值从多尺度特征中采样，再由 MLP 聚合

区别于图像域的可变形 Mamba 变体（需 patch 分割和 token 排序），DS-SSM 利用视频/音频固有的时序连续性，省略了冗余操作，显著降低计算开销。

### Relay Token 机制 — 解决长程衰减
SSM 中 token 间交互依赖控制矩阵的幂次 $\bar{A}^k$，$\bar{A}$ 的元素小于 1 导致信息随距离指数衰减。借鉴无线通信中的中继节点思想：

- 在输入序列中均匀插入 $N_r=8$ 个可学习的全局中继令牌
- 将序列划分为 $N_r+1$ 个子空间，局部状态高效传递信息给中继令牌，中继令牌再将聚合信息广播到其他子空间
- 形成稀疏的 sequence-to-token 跨序列信息流

配套两个辅助损失：

- **增强损失** $\mathcal{L}_{enh}$：鼓励每个中继令牌更好地聚合邻近子序列的信息（通过余弦相似度）
- **协作损失** $\mathcal{L}_{coop}$：鼓励不同中继令牌之间保持多样性，减少冗余（通过最小化互信息）

### Deformable Cross-SSM (DC-SSM) — 解决伪造稀疏
引入跨序列交互到可变形状态空间建模：

1. 每个 query $q_j^m$ 以其锚点提议 $(t_j^m, d_j^m)$ 为参考
2. MLP 预测多尺度偏移，计算可变形采样点
3. 从编码器输出中采样特征，拼接可学习空令牌后进行前向 SSM 更新
4. 利用 SSM 的聚合特性，仅保留最后追加的令牌作为输出

这一机制将全局状态空间划分为 query 特定的子空间，减少非伪造信息的累积，提升对稀疏伪造的敏感度。

### 总体训练损失
$$\mathcal{L} = \mathcal{L}_{match} + \mathcal{L}_{cls} + \lambda_1 \cdot \mathcal{L}_{enh} + \lambda_2 \cdot \mathcal{L}_{coop}$$
其中 $\lambda_1=0.5$，$\lambda_2=0.2$。$\mathcal{L}_{match}$ 采用 DETR 风格的匈牙利匹配损失。

## 实验关键数据

### 数据集
- **LAV-DF**：78K/31K/26K 训练/验证/测试，平均视频 8.6 秒
- **AV-Deepfake1M**：746K/57K/343K，更大规模、更细粒度伪造

### 主要结果（LAV-DF）
| 方法 | mAP@0.5 | mAP@0.75 | mAP@0.95 | mAP Avg | mAR Avg |
|------|---------|----------|----------|---------|---------|
| UMMAFormer | 98.8 | 95.5 | 37.6 | 77.3 | 92.3 |
| DiMoDif | 95.5 | 87.9 | 20.6 | 67.8 | 91.9 |
| FullFormer (纯 Transformer 基线) | 94.6 | 85.7 | 29.4 | 69.9 | 87.3 |
| **DeformTrace** | **97.1** | **90.7** | **38.1** | **75.3** | **92.9** |

### 主要结果（AV-Deepfake1M）
| 方法 | mAP Avg | mAR Avg | AUC |
|------|---------|---------|-----|
| UMMAFormer | 22.2 | 42.8 | - |
| DiMoDif | 49.3 | 79.6 | - |
| FullFormer | 40.4 | 66.8 | - |
| **DeformTrace** | **52.9** | **81.8** | **99.2** |

在 AV-Deepfake1M 上超越次优 DiMoDif 平均 3.6% mAP 和 2.2% mAR。

### 效率对比
| 方法 | 可训练参数 | FLOPs(G) | 推理时间(ms) |
|------|-----------|----------|-------------|
| UMMAFormer | 49.72M | 1563.9 | 857 |
| BA-TFD+ | 152.9M | 218.2 | 681 |
| **DeformTrace** | **20.8M** | **212.4** | **104** |

推理速度比 UMMAFormer 快 **8.2 倍**，可训练参数减少 **58%**。

### 消融实验（AV-Deepfake1M）
- Vanilla SSM 基线：mAP 41.2，mAR 68.7
- +DS-SSM +DC-SSM：mAP 49.8（+8.6）
- +Relay Token 全部损失：mAP 52.9（+11.7），mAR 81.8（+13.1）
- 中继令牌数 $N_r=8$ 为最优，过多导致子空间过度分割

## 亮点
1. **三个"首次"**：首次将可变形感受野引入时序 SSM；首次提出中继令牌机制显式缓解 SSM 长程衰减；首次将跨序列交互引入状态空间建模
2. **效率与精度兼得**：用 Transformer 1/7 的推理时间和 1/2.4 的可训练参数，取得更高精度
3. **鲁棒性强**：在 10 种音视频干扰（压缩、噪声、模糊等）下均优于现有方法
4. **中继令牌设计巧妙**：借鉴通信领域中继思想，辅助损失确保信息聚合与多样性

## 局限与展望
1. DC-SSM 目前仅用于 query 与特征序列之间的交互，作者提到可推广到音频-视频对应关系学习等更广泛场景
2. 特征提取器（Raven）被冻结，未探索端到端微调对性能的影响
3. 中继令牌数量与视频时长相关，实际部署可能需要自适应调整策略
4. 实验仅在 talking face 场景的伪造检测上验证，未涉及更通用的视频篡改场景

## 与相关工作的对比
- **vs UMMAFormer/DiMoDif**：这些方法依赖注意力和特征金字塔，计算量大。DeformTrace 用 SSM 替代部分注意力模块，以线性复杂度实现更高精度
- **vs 可变形 Mamba 变体（图像域）**：图像域方法需 patch 分割和 token 排序，DeformTrace 利用时序连续性简化设计
- **vs LongMamba**：LongMamba 通过固定阈值的通道分类和 token 过滤解决长程衰减，适应性有限；中继令牌机制更灵活且可学习
- **vs TadTR/TE-TAD**：这些 query-based TAD 方法在 TFL 上精度不足，DeformTrace 在此基础上引入 SSM 模块显著提升性能

## 启发与关联
1. 中继令牌机制是一个通用思路，可应用于任何需要长程依赖建模的 SSM 架构
2. DC-SSM 的跨序列交互设计为多模态融合提供了新的 SSM 方案，不局限于伪造检测
3. 将可变形机制从注意力/卷积领域迁移到 SSM 的思路值得在其他时序任务中探索

## 评分
- 新颖性: ⭐⭐⭐⭐ （三个首次创新，但整体是已有技术的组合创新）
- 实验充分度: ⭐⭐⭐⭐⭐ （两个大规模数据集 + 完整消融 + 鲁棒性测试 + 效率分析）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图示直观，公式完整）
- 价值: ⭐⭐⭐⭐ （TFL 领域实用性强，效率提升显著，通用组件有借鉴价值）
