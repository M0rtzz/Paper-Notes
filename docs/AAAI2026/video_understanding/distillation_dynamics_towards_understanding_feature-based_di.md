---
title: >-
  [论文解读] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers
description: >-
  [AAAI 2026][视频理解][知识蒸馏] 提出"蒸馏动力学"分析框架（通道维FFT频谱分析+Shannon熵+激活幅值追踪），揭示ViT具有独特的U型信息处理模式（先压缩后扩展）…
tags:
  - "AAAI 2026"
  - "视频理解"
  - "知识蒸馏"
  - "Transformer"
  - "频谱分析"
  - "信息瓶颈"
  - "负迁移"
---

# Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers

**会议**: AAAI 2026  
**arXiv**: [2511.06848](https://arxiv.org/abs/2511.06848)  
**代码**: [GitHub](https://github.com/thy960112/Distillation-Dynamics)  
**领域**: 视频理解  
**关键词**: 知识蒸馏, Vision Transformer, 频谱分析, 信息瓶颈, 负迁移

## 一句话总结

提出"蒸馏动力学"分析框架（通道维FFT频谱分析+Shannon熵+激活幅值追踪），揭示ViT具有独特的U型信息处理模式（先压缩后扩展），证明feature-based蒸馏在ViT中失败的根本原因是teacher后层的分布式高维编码范式与student有限通道容量之间的表征范式不匹配，而非简单的容量差距。

## 研究背景与动机

Feature-based知识蒸馏（如FitNet、AT等）在CNN压缩中非常成功——通过让小模型模仿大模型中间层特征，可以显著提升小模型性能。然而一个令人困惑的现象是：这些方法在Vision Transformer上不仅不奏效，反而比简单的logit蒸馏更差。

ViTKD等工作虽然观察到了这个现象并提出了ViT-specific蒸馏方法，但始终没有解释**为什么**CNN的成功经验不能迁移到ViT。这个理论空白严重制约了ViT压缩策略的设计——研究者只能靠经验试错，缺乏理论指导。

本文的切入角度是：不急于提出新的蒸馏方法，而是先彻底理解ViT内部的信息处理机制，找到feature蒸馏失败的根本原因。通过设计三个互补的分析工具，从不同角度"三角定位"ViT表征的本质特性。

## 方法详解

### 整体框架

提出三维度分析框架"蒸馏动力学"：(1) 沿通道维度的FFT频谱分析揭示特征编码策略；(2) Shannon熵分析量化各层信息复杂度；(3) 激活幅值追踪信号传播强度。三个视角交叉验证，确保观察到的模式不是单一测量的artifact。在此分析基础上，设计SpectralKD和ProjectorKD两种方法验证分析结论。

### 关键设计

1. **通道维度FFT频谱分析**

    - 功能：揭示ViT各层特征的编码策略
    - 核心思路：对每层激活张量 $\mathbf{A} \in \mathbb{R}^{L \times B \times C \times H \times W}$，**沿通道轴**（而非传统的空间轴）做一维FFT：$\mathbf{F}_{l,b,h,w}[k] = \frac{1}{C}\sum_{c=0}^{C-1}\mathbf{A}_{l,b,c,h,w} e^{-j2\pi kc/C}$。低频主导表示通道高度相关（压缩编码），高频均匀表示通道去相关（分布式编码）。对batch和空间维度取平均得到每层频谱签名 $\mathbf{S} \in \mathbb{R}^{L \times C}$
    - 设计动机：空间FFT只能看到特征图的空间频率特性（已被研究过），而通道FFT揭示的是**特征空间本身的结构**——通道间相关性高意味着特征冗余，去相关意味着充分利用表达容量

2. **Shannon熵 + 激活幅值联合分析**

    - 功能：量化各层信息复杂度和信号传播强度
    - 核心思路：Shannon熵——将每个空间位置的通道激活向量离散化为100个bin，计算概率分布的熵 $E_{l,b,h,w} = -\sum_{n:p_n>0} p_n \log_2 p_n$，对batch和空间取平均。激活幅值——计算每层激活的均值绝对值 $M_l = \frac{1}{BCHW}\sum|\mathbf{A}_{l,b,c,h,w}|$
    - 设计动机：熵低→表征集中结构化（类似Information Bottleneck的瓶颈），熵高→表征均匀扩展。与频谱分析交叉验证：U型熵对应频谱从均匀→低通→均匀的三阶段演化

3. **蒸馏演化（Distillation Evolution）分析**

    - 功能：追踪不同蒸馏策略对student训练动态的影响
    - 核心思路：在训练过程中每30个epoch记录student的逐层熵profile，观察U型模式如何形成或被破坏。比较SoftKD、SpectralKD-First、SpectralKD-Last、SpectralKD-Both四种配置下student的发展轨迹
    - 设计动机：知识蒸馏不是静态的知识复制，而是对student"发展轨迹"的引导。错误的引导（后层对齐）会扰乱student自然形成U型模式的过程

### 损失函数 / 训练策略

SpectralKD：$\mathcal{L}_{\text{Freq}} = \text{MSE}(\mathcal{F}_{\text{stack}}(\mathbf{A}_s), \mathcal{F}_{\text{stack}}(\mathbf{A}_t))$，对空间维做2D RFFT后栈接实虚部。

ProjectorKD：$\mathcal{L}_{\text{Proj}} = \text{MSE}(\text{Projector}(\mathbf{A}_s), \mathbf{A}_t)$，可学习投影层匹配维度。

总损失：$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{KD}} + \beta \mathcal{L}_{\text{Feature}}$，$\beta$ 控制feature蒸馏权重。Teacher: CaiT-S24, Student: DeiT-Tiny, 300 epochs (部分500 epochs)。

## 实验关键数据

### 主实验

| 方法 | 对齐层 | β | Top-1 Acc (%) |
|------|--------|---|--------------|
| SoftKD (logit only) | - | - | 76.99 |
| SpectralKD | First1+Last1 | 0.2 | **77.08** |
| SpectralKD | First1 | 0.2 | 77.00 |
| SpectralKD | Last1 | 0.2 | 76.83 (-0.16) |
| SpectralKD | Last1 | 0.1 | 76.48 (-0.51) |
| SpectralKD | Last8 | 0.2 | 76.69 (-0.30) |
| ProjectorKD | First1 | 0.2 | 76.86 |
| ProjectorKD | Last1 | 0.2 | 76.72 (-0.27) |
| ProjectorKD | First1+Last1 | 0.2 | 76.80 |
| SoftKD (500ep) | - | - | 78.07 |
| SpectralKD (500ep) | Last1 | 0.2 | 77.59 (-0.48) |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 减小β (0.2→0.1) | 76.83→76.48 | 更弱的蒸馏信号反而更差——问题是方向而非强度 |
| 延长训练 (300→500ep) | 差距从0.16扩大到0.48 | 更长训练无法弥补后层蒸馏的负面影响 |
| CNN (ResNet) 后层频谱 | 保持低通特性 | CNN不充分利用通道容量→student可模仿→CNN蒸馏成功 |
| ViT/CaiT/MAE U型一致性 | 三种架构/训练范式 | U型模式是ViT的universal特征，非特定模型artifact |

### 关键发现

- **U型信息处理是ViT的fundamental特征**：CaiT-S24、标准ViT、MAE预训练ViT均展现一致的U型熵和激活幅值曲线。层1-9压缩→层9-24扩展，对应Information Bottleneck的两阶段
- **后层频谱从低通变为均匀分布**：Phase 1（早层）均匀嘈杂 → Phase 2（中间层）低通滤波 → Phase 3（后层）均匀高能量。Phase 3代表分布式高维编码——信息分散纠缠在整个通道空间
- **CNN与ViT的关键差异**：CNN（ResNet）后层仍保持Phase 2的低通特性，**不利用全部通道容量**→小student可以模仿→蒸馏成功。ViT后层充分利用通道容量→student无法复制→蒸馏失败
- **减小β反而更差**的反直觉现象：弱蒸馏信号破坏了student的学习均衡，导致在teacher编码范式和自身最优方案间摇摆不定
- **蒸馏是"发展轨迹引导"而非"静态知识复制"**：SoftKD下student自然发展出U型模式；后层对齐压制了自然扩展阶段的形成

## 亮点与洞察

- **U型信息处理模式的发现**有很高理论价值——它是ViT的学习行为而非架构属性，在supervised/self-supervised训练中一致出现，为理解ViT内部机制提供了新视角
- **通道维度FFT分析**是genuinely original的工具——区别于常见的空间FFT，揭示了特征空间本身的编码结构
- **CNN vs ViT的后层频谱差异**精确解释了蒸馏效果差异——这是最核心的insight
- **"蒸馏引导发展轨迹"的视角**深刻——后层蒸馏没有给错信号的"量"，而是给了错误的"方向"
- 减小β反而更差的发现和解释非常有启发性：揭示了蒸馏损失与分类损失之间存在微妙的动态均衡

## 局限与展望

- **主要是分析论文**：SpectralKD/ProjectorKD只是验证分析的工具，实际性能提升极其有限（最佳仅77.08 vs baseline 76.99）
- **未设计有效的ViT蒸馏方法**："phase-specific distillation"停留在建议层面，未实现
- **实验仅在ImageNet分类**：未验证在检测/分割等下游任务中是否有同样的U型模式和蒸馏失败
- **U型模式的形成机制未深入分析**：为什么ViT学到这个而CNN不会？是self-attention的固有属性还是训练数据/目标的影响？
- **仅分析一对teacher-student**：CaiT-S24→DeiT-Tiny，未验证其他teacher-student组合（如Swin→Swin-Tiny）

## 相关工作与启发

与ViTKD相比，ViTKD提出了ViT-specific蒸馏方法但未解释失败原因，本文首次给出机制性解释。与FitNet相比，本文揭示了FitNet在CNN上成功的深层原因——CNN后层仍是紧凑编码，student可以模仿。与Information Bottleneck理论相比，本文提供了ViT中IB理论的第一个直接经验证据。

核心启发：ViT蒸馏应该只对齐早期-中间层（压缩阶段），避免后层对齐。更进一步的想法是设计"编码翻译器"——将teacher的分布式表示转换为student可消化的紧凑编码，而非让student直接模仿。U型模式对VLM token pruning也有启示：应在entropy最低点（瓶颈处）进行token压缩效率最高。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从信息论+频域角度解释ViT蒸馏失败，U型模式和通道FFT都是genuinely original的发现
- 实验充分度: ⭐⭐⭐⭐ 分析非常充分（频谱+熵+幅值+蒸馏演化），但蒸馏方法验证偏少，且仅ImageNet分类
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑严密，从现象→分析→解释→验证层层递进，配图精美直观
- 价值: ⭐⭐⭐⭐⭐ 为ViT压缩提供fundamental theoretical guidance，对后续蒸馏方法设计有长期影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](../../CVPR2026/video_understanding/beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](../../CVPR2025/video_understanding/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[CVPR 2025\] ViTED: Video Temporal Evidence Distillation](../../CVPR2025/video_understanding/vited_video_temporal_evidence_distillation.md)
- [\[ACL 2025\] From Teacher to Student: Tracking Memorization Through Model Distillation](../../ACL2025/video_understanding/from_teacher_to_student_tracking_memorization_through_model_distillation.md)
- [\[CVPR 2025\] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](../../CVPR2025/video_understanding/enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)

</div>

<!-- RELATED:END -->
