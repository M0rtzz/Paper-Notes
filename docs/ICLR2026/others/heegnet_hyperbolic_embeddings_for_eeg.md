---
title: >-
  [论文解读] HEEGNet: Hyperbolic Embeddings for EEG
description: >-
  [ICLR 2026][EEG] 首次系统验证EEG数据具有双曲性（层次结构），提出HEEGNet混合双曲网络架构，结合欧几里得编码器提取时空频谱特征和双曲编码器捕捉层次关系，配合创新的粗到细域适应策略(DSMDBN)，在视觉诱发电位、情感识别和颅内EEG多个跨域任务上达到SOTA。
tags:
  - ICLR 2026
  - EEG
  - 双曲空间
  - 域适应
  - 层次结构
  - 脑机接口
---

# HEEGNet: Hyperbolic Embeddings for EEG

**会议**: ICLR 2026  
**arXiv**: [2601.03322](https://arxiv.org/abs/2601.03322)  
**代码**: [GitHub](https://github.com/fightlesliefigt/HEEGNet)  
**领域**: 脑机接口/几何深度学习  
**关键词**: EEG, 双曲空间, 域适应, 层次结构, 脑机接口

## 一句话总结
首次系统验证EEG数据具有双曲性（层次结构），提出HEEGNet混合双曲网络架构，结合欧几里得编码器提取时空频谱特征和双曲编码器捕捉层次关系，配合创新的粗到细域适应策略(DSMDBN)，在视觉诱发电位、情感识别和颅内EEG多个跨域任务上达到SOTA。

## 研究背景与动机

**领域现状**：EEG脑机接口因被试间/会话间分布偏移导致泛化差。域适应方法（矩对齐）是当前SOTA，但在大偏移下失效。EEG解码几乎完全基于欧几里得嵌入。

**现有痛点**：(1) 大脑的视觉处理、情感调节等认知过程具有层次结构，但欧几里得空间难以高效表示层次数据——圆的周长线性增长而树节点数指数增长；(2) 仅做矩对齐无法保证正迁移，特别在大域偏移下。

**核心矛盾**：EEG特征的层次结构→需要指数级表达力→欧几里得空间不够→但双曲神经网络还未被系统探索用于EEG。

**切入角度**：先验研究发现EEG具有双曲性（$\delta_{rel}$低），双曲MLR替换欧几里得MLR就能提升跨域性能→证明双曲嵌入确实有助于EEG泛化。

**核心 idea**：用双曲空间捕捉EEG的层次结构+两阶段域适应(矩对齐→分布对齐)实现跨域泛化。

## 方法详解

### 整体框架
HEEGNet = 欧几里得编码器(时间→空间→时间卷积) → 投影到双曲空间 → 双曲卷积层 → DSMDBN域适应 → 双曲MLR分类器。

### 关键设计

1. **混合欧几里得-双曲架构**:

    - 功能：欧几里得卷积提取频谱-空间-时间特征，投影后双曲卷积精炼层次关系
    - 核心思路：3层EEGNet风格卷积 → ProjX投影到Lorentz模型 $\mathbb{L}_K^n$ → 双曲点卷积
    - 设计动机：欧几里得卷积擅长信号处理（有神经生理学解释），双曲空间擅长表达层次关系。混合取各自所长。

2. **DSMDBN (两阶段域适应)**:

    - 第一阶段 DSMDBN(1)：Riemannian批归一化做域特异矩对齐（双曲空间中的centering=gyro减法，scaling=gyro乘法）
    - 第二阶段 DSMDBN(2)：最小化HHSW散度将每个源域分布对齐到标准双曲高斯 $\mathcal{N}(\bar{0}, 1)$
    - 设计动机：仅矩对齐不够（大偏移下失效），加上分布对齐提供理论保证。先矩后分布=粗到细。

3. **Lorentz模型操作**:

    - 双曲加法(gyroaddition)、标量乘法(gyromultiplication)、逆元(gyroinverse)
    - Fréchet均值和方差在Lorentz模型上的定义
    - 双曲MLR利用点到超平面的双曲距离做分类

### 损失函数 / 训练策略
- 分类loss + HHSW分布对齐loss
- 域特异动量批归一化：训练时衰减动量更新，测试时固定动量
- Riemannian Adam优化器

## 实验关键数据

### 先验研究
| 数据集 | $\delta_{rel}$原始EEG | $\delta_{rel}$嵌入层 | 说明 |
|--------|---------------------|---------------------|------|
| Nakanishi | 低 | 低 | 视觉 |
| Wang | 低 | 低 | 视觉 |
| Seed | 低 | 低 | 情感 |
| Faced | 低 | 低 | 情感 |
| Boran | 低 | 低 | 颅内 |

→ 所有数据集都展示低 $\delta_{rel}$，确认EEG的双曲性。

### 主实验
跨被试/跨会话适应：

| 方法 | 视觉EEG | 情感EEG | 颅内EEG | 平均 |
|------|---------|---------|---------|------|
| EEGNet | 基线 | 基线 | 基线 | 基线 |
| EEGNet+HMLR | ↑ | ↑ | ↑ | 稳定提升 |
| HEEGNet | **SOTA** | **SOTA** | **SOTA** | **全面最优** |

### 关键发现
- 仅把MLR换成双曲MLR就能在所有数据集上提升→双曲几何确实更适合EEG
- t-SNE可视化显示双曲嵌入的类别分离性明显优于欧几里得
- DSMDBN的两阶段策略比仅矩对齐有显著提升
- 在运动想象数据集（不报告层次性）上也有提升→可能存在未被识别的层次结构

## 亮点与洞察
- **首次系统验证EEG的双曲性**：$\delta_{rel}$ 量化分析+多数据集验证，为这个方向提供了坚实的实证基础。
- **混合架构的合理性**：不是纯双曲（那样会丢失信号处理先验），而是先用欧几里得提取有意义的特征再映射到双曲空间——这是值得其他领域借鉴的设计哲学。
- **DSMDBN的粗到细**：矩对齐→分布对齐是一个自然的两步策略，前者拉近均值和尺度，后者对齐整体形状。

## 局限与展望
- 双曲操作计算开销比欧几里得大（指数/对数映射），对实时BCI有影响
- 曲率K作为超参数需要调优，自适应曲率学习可能更好
- HHSW在高维可能需要大量投影方向才准确
- 颅内EEG各被试电极数不同，限制了跨被试的实验设置

## 相关工作与启发
- **vs EEGNet**: HEEGNet在EEGNet基础上增加双曲层和DSMDBN，全面提升
- **vs Chang等人的双曲EEG**: 他们只做对比学习预训练，HEEGNet设计了完整的架构+域适应方案
- **vs SPDNet等Riemannian方法**: SPDNet在协方差矩阵流形上操作，HEEGNet在双曲空间操作，关注不同几何结构

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ EEG双曲性的系统论证+混合双曲架构都是首次
- 实验充分度: ⭐⭐⭐⭐⭐ 先验研究→多数据集→多任务→消融全面
- 写作质量: ⭐⭐⭐⭐ 背景知识介绍充分，方法描述清晰
- 价值: ⭐⭐⭐⭐ 为EEG解码引入新的几何视角，开辟双曲EEG方向

<!-- RELATED:START -->

## 相关论文

- [HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](../../CVPR2026/others/hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)
- [Better Embeddings with Coupled Adam](../../ACL2025/others/better_embeddings_with_coupled_adam.md)
- [CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding](../../AAAI2026/others/cat-net_a_cross-attention_tone_network_for_cross-subject_eeg-emg_fusion_tone_dec.md)
- [Partial Colexifications Improve Concept Embeddings](../../ACL2025/others/partial_colexifications_improve_concept_embeddings.md)
- [Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](../../ICCV2025/others/learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)

<!-- RELATED:END -->
