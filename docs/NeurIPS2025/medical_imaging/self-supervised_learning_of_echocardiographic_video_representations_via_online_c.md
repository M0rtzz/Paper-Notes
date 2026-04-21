---
title: >-
  [论文解读] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation
description: >-
  [NeurIPS 2025][医学图像][自监督学习] 提出 DISCOVR，一种自监督双分支框架，通过在线语义聚类蒸馏将图像编码器的细粒度空间语义传递到视频编码器的时序表示中，在六个跨胎儿/儿科/成人心脏超声数据集上实现了异常检测、分类和分割的全面领先。
tags:
  - NeurIPS 2025
  - 医学图像
  - 自监督学习
  - 超声心动图
  - 视频表示学习
  - 跨模态蒸馏
  - 语义聚类
---

# Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2506.11777](https://arxiv.org/abs/2506.11777)  
**代码**: [GitHub](https://github.com/mdivyanshu97/DISCOVR)  
**领域**: 医学图像  
**关键词**: 自监督学习, 超声心动图, 视频表示学习, 跨模态蒸馏, 语义聚类

## 一句话总结

提出 DISCOVR，一种自监督双分支框架，通过在线语义聚类蒸馏将图像编码器的细粒度空间语义传递到视频编码器的时序表示中，在六个跨胎儿/儿科/成人心脏超声数据集上实现了异常检测、分类和分割的全面领先。

## 研究背景与动机

超声心动图视频理解面临独特的挑战，现有自监督学习方法在该领域表现不佳：

**高帧间相似度**：心脏超声帧间余弦相似度高达 0.99（使用预训练 VideoMAE），使得区分帧变得极为困难。正常与异常心脏在视觉上几乎相同，但细微差异（如双心室收缩功能障碍、扩张的球形左心室）在临床上至关重要

**现有 SSL 方法的局限**：
   - **掩码视频建模**（VideoMAE, MGMAE）聚焦于重建低信号噪比的超声像素，难以捕捉高层语义
   - **对比学习**（MoCo, SimCLR）在高帧间相似度下难以构建有效正负样本对
   - **聚类方法**（SIGMA）依赖激进的数据增强，可能破坏临床相关的解剖细节

**缺乏领域专用预训练模型**：与自然图像/视频不同，超声心动图领域缺少大规模预训练骨干

DISCOVR 的核心洞察是：超声心动图分析需要同时建模**时序动态**（如心壁运动、瓣膜功能的周期性变化）和**细粒度空间语义**（如房间隔厚度、心内膜边界），而现有方法通常只关注其中之一。

## 方法详解

### 整体框架

DISCOVR 是一个双分支自监督框架：视频分支通过掩码自蒸馏捕获全局心脏运动动态，图像分支通过掩码图像自蒸馏学习细粒度空间语义。两个分支通过语义聚类蒸馏（SCD）损失连接，将图像编码器不断演化的解剖知识传递给视频编码器。

### 关键设计

1. **视频自蒸馏（Video Self-Distillation）**：采用 student-teacher 架构，输入视频被分割为3D时空管状token。Teacher 编码器 $E_{\theta_t}$ 处理完整视频产生全局 CLS 表示 $z_t$，Student 编码器 $E_{\theta_s}$ 处理多个掩码变体 $v_{\mathcal{M}_m}$ 产生 $z_s^{(m)}$。Teacher 参数通过 EMA 更新：$\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$。通过温度缩放 softmax 和交叉熵损失对齐：$\mathcal{L}_{\text{ssl}}^{vid} = \frac{1}{M}\sum_{m=1}^M H(P_t, P_s^{(m)})$。设计动机：让 student 在仅观察不完整视频的情况下学会恢复完整的心脏运动表示。

2. **掩码图像自蒸馏（Masked Image Self-Distillation）**：并行训练的图像编码器 $\mathcal{I}_\theta$，对视频中的每一帧独立处理。Teacher 接收完整帧，Student 接收 $N$ 个掩码变体，通过类似的自蒸馏损失训练：$\mathcal{L}_{\text{ssl}}^{img} = \frac{1}{N}\sum_{i=1}^N H(P_t, P_s^{(i)})$。设计动机：学习空间锚定的表示，编码如胎儿心脏瓣膜、心室解剖结构、房间隔轮廓等细粒度临床概念。

3. **语义聚类蒸馏（Semantic Cluster Distillation, SCD）**：这是 DISCOVR 的核心创新。视频解码器重建的token级特征 $\hat{\mathbf{z}}_v$ 和图像编码器产生的空间特征 $\hat{\mathbf{z}}_i$（梯度截断）投影到共享的可学习原型集 $P \in \mathbb{R}^{K \times D}$，通过 Sinkhorn-Knopp 算法生成软聚类分配，并用对称交叉熵对齐：$\mathcal{L}_{\text{SCD}} = \text{CE}(\mathbf{s}_v, \text{stopgrad}(\mathbf{q}_i)) + \text{CE}(\mathbf{s}_i, \text{stopgrad}(\mathbf{q}_v))$。SCD 仅通过视频模型和原型矩阵传播梯度，图像编码器仅通过自身的自蒸馏损失更新。设计动机：将图像编码器发现的空间语义聚类锚定到视频编码器的token表示中，使时序特征同时包含细粒度解剖细节。

### 损失函数 / 训练策略

- **总损失**：$\mathcal{L} = \mathcal{L}_{\text{ssl}}^{vid} + \mathcal{L}_{\text{ssl}}^{img} + \mathcal{L}_{\text{SCD}}$
- 输入配置：64帧视频片段，步长3采样；时空管状嵌入 $2 \times 16 \times 16$，90%掩码率
- ViT-Base 骨干，teacher 和 student 使用各自独立的投影头
- 仅在正常视频上训练，将病理视为正常心脏动态的偏差
- **无需预训练模型、标签或激进数据增强**

## 实验关键数据

### 主实验

零样本异常检测（kNN 分类器，在六个数据集上评估）：

| 数据集 | 人群 | DISCOVR F1 | 最佳基线 F1 | 基线方法 | 提升 |
|---|---|---|---|---|---|
| EchoNet-Dynamic | 成人 | **61.45** | 57.56 | MVD | +6.8% |
| RVENET | 儿科/成人 | **53.88** | 52.18 | MNAD | +3.3% |
| EchoPediatric-LVH | 儿科 | **54.63** | 51.31 | C2FPL | +6.5% |
| FetalEcho 1 | 胎儿 | **61.79** | 60.64 | MGMAE | +1.9% |
| FetalEcho 2 | 胎儿 | **56.69** | 56.09 | MGMAE | +1.1% |

线性探测分类：

| 数据集 | DISCOVR F1 | SIGMA F1 | VideoMAE F1 |
|---|---|---|---|
| EchoNet-Dynamic | **77.63** | 75.50 | 70.85 |
| FetalEcho 2 | **63.59** | 55.81 | 51.60 |
| RVENET | **62.65** | 58.98 | 59.70 |

### 消融实验

| 配置 | Balanced Acc. | F1 | 说明 |
|---|---|---|---|
| 仅 $\mathcal{L}_{\text{ssl}}^{vid}$ | 52.27 | 48.23 | 仅视频自蒸馏，缺少空间语义 |
| $\mathcal{L}_{\text{ssl}}^{vid} + \mathcal{L}_{\text{SCD}}$ (完整) | **63.20** | **61.45** | SCD 带来+13.22%的F1提升 |
| ViT-Small | 59.44 | 57.52 | 更小骨干仍有不错表现 |
| ViT-Base | **63.20** | **61.45** | 更大模型有明显收益 |
| 50% 掩码率 | 55.60 | 52.98 | 低掩码率不足以学习鲁棒表示 |
| 90% 掩码率 | **63.20** | **61.45** | 高掩码率促进语义学习 |
| 16帧 | 57.89 | 55.68 | 短片段不足以覆盖心动周期 |
| 64帧 | **63.20** | **61.45** | 长片段对超声视频至关重要 |

### 关键发现

- 分割评估：在 CAMUS 数据集上，冻结骨干+简单线性头，DISCOVR 的 Dice 达到 0.844，超越 UNet (0.816) 和 DeepLabV3 (0.819) 等专用分割架构
- LVEF 预测：线性探测 MAE 7.79，微调后3块达到 6.32，超过多个端到端监督基线（MC3: 6.59, EchoNet-Dynamic: 7.35）
- 计算开销：训练仅需微增 GPU 内存（10.5GB vs 9.0-9.5GB），推理与所有方法完全相同

## 亮点与洞察

- **任务无关的通用表示**：一个自监督预训练模型同时胜任异常检测、分类、分割和功能评估，展示了极强的迁移能力
- **跨人群泛化**：首个在胎儿、儿科和成人三个人群上系统评估的心脏超声 SSL 方法
- **SCD 损失的优雅设计**：通过原型聚类实现跨模态知识传递，避免了直接特征对齐的困难
- **长序列建模的重要性**：64帧（覆盖约2个心动周期）比短片段有显著优势，这对超声视频领域有重要指导意义

## 局限与展望

- 对比的视频异常检测基线（MNAD、MemAE、C2FPL）主要设计用于自然场景，医学专用异常检测方法的对比不足
- 当前采用固定的90%掩码率，自适应掩码策略可能更适合超声视频的高冗余特性
- 图像编码器和视频编码器使用相同架构，探索异构架构设计可能有额外收益
- 缺乏对其他超声模态（如多普勒、造影增强超声）的验证

## 相关工作与启发

- 与 DINO/iBOT 的关系：DISCOVR 扩展了图像自蒸馏到双模态（图像+视频），并通过 SCD 实现跨模态连接
- 与 SIGMA 的区别：SIGMA 使用固定聚类目标，DISCOVR 的图像编码器在线演化提供动态语义引导
- 对医学视频 SSL 的启发：空间-时序解耦并通过聚类蒸馏重新融合的思路可推广到其他医学视频模态

## 评分

- **新颖性**: ⭐⭐⭐⭐ 双分支+SCD蒸馏的设计优雅，但各组件均有前人工作基础
- **实验充分度**: ⭐⭐⭐⭐⭐ 六个数据集、四种任务、跨人群评估，消融完整，开源代码
- **写作质量**: ⭐⭐⭐⭐ 问题动机阐述清晰，可视化丰富（特别是Fig.1和Fig.3的对比）
- **价值**: ⭐⭐⭐⭐⭐ 为超声心动图分析提供了强大的通用预训练方案，临床应用前景广阔

<!-- RELATED:START -->

## 相关论文

- [Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [An OpenMind for 3D Medical Vision Self-supervised Learning](../../ICCV2025/medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data](../../ICML2025/medical_imaging/scssl-bench_benchmarking_self-supervised_learning_for_single-cell_data.md)

<!-- RELATED:END -->
