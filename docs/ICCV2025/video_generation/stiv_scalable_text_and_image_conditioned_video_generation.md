---
title: >-
  [论文解读] STiV: Scalable Text and Image Conditioned Video Generation
description: >-
  [ICCV 2025][视频理解][视频生成] 本文提出 STIV，一个基于 Diffusion Transformer 的统一文本-图像条件视频生成框架，通过帧替换策略整合图像条件并引入联合图像-文本 classifier-free guidance，在单一模型中同时实现 T2V 和 TI2V 生成，8.7B 参数模型在 VBench T2V 和 I2V 上分别达到 83.1 和 90.1 的 SOTA 成绩。
tags:
  - ICCV 2025
  - 视频理解
  - 视频生成
  - Transformer
  - 图像条件
  - 文本到视频
  - 可扩展训练
---

# STiV: Scalable Text and Image Conditioned Video Generation

**会议**: ICCV 2025  
**arXiv**: [2412.07730](https://arxiv.org/abs/2412.07730)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频生成, 扩散 Transformer, 图像条件, 文本到视频, 可扩展训练

## 一句话总结
本文提出 STIV，一个基于 Diffusion Transformer 的统一文本-图像条件视频生成框架，通过帧替换策略整合图像条件并引入联合图像-文本 classifier-free guidance，在单一模型中同时实现 T2V 和 TI2V 生成，8.7B 参数模型在 VBench T2V 和 I2V 上分别达到 83.1 和 90.1 的 SOTA 成绩。

## 研究背景与动机
视频生成领域在 Sora 之后迅速发展，Diffusion Transformer (DiT) 架构成为主流。然而，实现 Sora 级别的视频生成仍面临多个挑战：

**痛点 1：图像条件整合方式不清晰**。如何将图像条件有效整合到 DiT 架构中仍未有定论。基于 U-Net 的方法（如 ConsistI2V）需要额外的空间自注意力和窗口时间注意力，不够优雅。

**痛点 2：大规模训练不稳定**。随着模型规模增大，训练不稳定性和内存占用成为主要瓶颈。QK-norm 等技巧在更大模型上表现不够。

**痛点 3：缺乏系统化的 recipe**。现有工作通常独立研究各个方面（架构设计、训练策略、数据处理），缺乏对它们之间相互作用的系统性研究。

**核心 idea：提供一个透明、可扩展的视频生成 recipe**，从 T2I 到 T2V 再到 TI2V 逐步构建，以帧替换作为图像条件的核心设计，并通过全面的消融实验验证每个设计选择。

## 方法详解

### 整体框架
STIV 基于 PixArt-α 架构，使用冻结 VAE 将视频帧编码为时空隐向量，通过堆叠的 DiT-like blocks 处理。T5 和 CLIP 编码器处理文本提示。渐进式训练流程为：T2I → T2V → STIV (TI2V)，通过分解的空间-时间注意力处理视频帧。

### 关键设计
1. **帧替换策略（Frame Replacement）**:

    - 功能：在训练时将噪声视频 latent 的第一帧替换为未加噪的图像条件 latent，并 mask 掉该帧的损失
    - 核心思路：利用 DiT 架构中堆叠的空间-时间注意力层，图像条件信息自然地通过注意力传播到后续帧。无需额外的 cross attention 或投影层
    - 设计动机：DiT 架构本身就能通过 self-attention 传播首帧信息，这使得帧替换成为一个**极简但高效**的设计。对比实验表明，添加额外的 cross attention 或大投影层虽然提升了主体/背景一致性，但会显著降低动态度（22.4 vs 36.6），过度约束生成输出

2. **联合图像-文本 Classifier-Free Guidance (JIT-CFG)**:

    - 功能：在训练时随机 dropout 图像条件和文本条件，推理时使用联合 CFG
    - 核心思路：速度场修正为 $\hat{F}_\theta(x_t, c_T, c_I, t) = F_\theta(x_t, \emptyset, \emptyset, t) + s \cdot (F_\theta(x_t, c_T, c_I, t) - F_\theta(x_t, \emptyset, \emptyset, t))$，仅需两次前向传播
    - 设计动机：解决高分辨率 STIV 模型的**运动静止问题**（staleness issue）。图像条件 dropout 防止模型被动过拟合到图像条件，鼓励从底层视频数据学习运动信息。同时自然地实现了 T2V 和 TI2V 的多任务训练

3. **稳定高效的大规模训练**:

    - 功能：组合多种技术确保训练稳定性和效率
    - 核心技术：
        - **QK-Norm + Sandwich-Norm**：对 MHA 和 FFN 都使用前后双层归一化，结合无状态层归一化
        - **MaskDiT**：随机 mask 50% 的空间 token 进行训练（后续无 mask 微调），显著减少内存
        - **AdaFactor 优化器**：替代 AdamW，降低内存占用
        - **RoPE**：2D RoPE 用于空间注意力，1D RoPE 用于时间注意力，支持分辨率外推
    - 设计动机：单一稳定性技巧不足以支撑大模型训练。三项效率技术（MaskDiT+AdaFactor+梯度检查点）协同工作才能在合理资源下训练 8.7B 模型

4. **渐进式训练（Progressive Training）**:

    - 功能：从 T2I → T2V → STIV，从低分辨率 → 高分辨率，从短时长 → 长时长逐步训练
    - 核心思路：高分辨率 T2V 模型同时从高分辨率 T2I 和低分辨率 T2V 初始化（空间权重来自 T2I，时间权重来自 T2V），使用 RoPE 插值适配新的分辨率/时长
    - 设计动机：直接训练高分辨率长时长模型代价过高。渐进式训练在相同计算预算下效果更好

### 损失函数 / 训练策略
- 使用 Flow Matching 目标（而非传统 diffusion loss）：$\min_\theta \mathbb{E}[\|F_\theta(x_t, c, t) - v_t\|_2^2]$，其中 $v_t = x_1 - \epsilon$
- T2I 训练 400k steps，batch size 4096；T2V/TI2V 训练 400k steps，batch size 1024
- EMA decay rate 0.9999
- MaskDiT 的 50% mask 训练后，进行 50k/100k steps 的无 mask 微调

## 实验关键数据

### 主实验
| 模型 | 参数量 | 分辨率 | VBench T2V Total | VBench I2V | 备注 |
|------|--------|--------|-----------------|------------|------|
| STIV-M | 8.7B | 512² | **83.1** | **90.1** | SOTA |
| CogVideoX-5B | 5B | - | 81.6 | - | 开源SOTA |
| Pika | - | - | 80.6 | - | 商用产品 |
| Kling | - | - | 81.8 | - | 商用产品 |
| Gen-3 | - | - | 82.2 | - | 商用产品 |

### 消融实验
| 配置 | VBench Quality | VBench Semantic | VBench Total | 说明 |
|------|---------------|----------------|-------------|------|
| Base T2V-XL | 80.19 | 70.51 | 78.25 | 基线 |
| + temporal patch=1 | 80.92 | 71.69 | 79.07 | 最佳但2x计算 |
| + causal temp atten | 74.59 | 73.13 | 74.30 | 大幅下降 |
| + temp mask | 77.58 | 65.95 | 75.25 | 时间mask损害严重 |
| - spatial mask | 80.57 | 70.31 | 78.52 | 轻微提升但需更多计算 |

TI2V 图像条件整合方式消融：

| 方法 | I2V Avg Score | Total Avg Score | Dynamic Degree | 说明 |
|------|-------------|----------------|---------------|------|
| Cross Attention (CA) | 68.2 | 73.0 | 42.4 | 基线 |
| CA + Large Proj | 72.3 | 75.3 | 22.2 | 过度约束 |
| Frame Replace (FR) | **75.8** | **77.3** | 36.6 | 最佳平衡 |
| FR + CA | 74.4 | 77.1 | 35.4 | 无额外收益 |

### 关键发现
- 帧替换是整合图像条件的最优方式：简单、高效、且不牺牲动态度
- **非因果时间注意力远优于因果注意力**（Total: 78.25 vs 74.30），这与 Sora 猜测的因果设计相矛盾
- 时间维度 masking 会严重损害性能（-3.0 Total），而空间 masking 几乎无损
- 图像条件 dropout 不仅是多任务训练的手段，更是解决高分辨率静止运动问题的关键
- Flow Matching + CFG-Renormalization 是最大的单项性能提升因素
- 渐进式初始化（同时利用 T2I 和低分辨率 T2V 权重）优于单源初始化

## 亮点与洞察
- **极简设计哲学**：帧替换+JIT-CFG 两个核心设计都极其简单，但效果显著，体现了"简单但正确"的工程美学
- **系统化的 recipe**：从 T2I 到 T2V 到 TI2V 的渐进路径，每步都有详细消融，对社区极有参考价值
- **统一框架的灵活性**：单一模型通过改变条件即可实现 T2V、TI2V、视频预测、帧插值、多视角生成、长视频生成等多种任务
- **数据引擎**：构建了完整的视频数据处理流水线（PySceneDetect + captioning + DSG-Video 过滤），处理了 90M+ 视频-文本对

## 局限与展望
- 依赖大规模内部数据（42M 内部视频），可复现性受限
- 帧替换策略在多图像条件（如视频编辑）场景的表现未深入探讨
- 8.7B 模型的推理成本仍然很高，未讨论推理效率优化
- 评估主要依赖 VBench，缺乏人工评估对比
- 长视频生成（>100帧）的质量和一致性需要更多验证

## 相关工作与启发
- 帧替换在 U-Net 时代（ConsistI2V）已被探索但效果一般，在 DiT 架构中因纯 self-attention 设计而变得自然有效——**架构决定了最优策略**
- MaskDiT + AdaFactor + 梯度检查点的组合是大模型高效训练的实用方案
- 渐进式训练（分辨率/时长/架构三个维度逐步提升）是视频生成模型 scaling 的关键路径

## 评分
- 新颖性: ⭐⭐⭐ 各组件均为已有技术的组合，帧替换也非首创，核心贡献在于系统性整合
- 实验充分度: ⭐⭐⭐⭐⭐ 消融极其全面，从T2I到T2V到TI2V每步都有详细ablation
- 写作质量: ⭐⭐⭐⭐ 结构清晰，recipe式呈现对实践者友好
- 价值: ⭐⭐⭐⭐⭐ 作为视频生成的系统性 recipe，对社区有极高参考价值
