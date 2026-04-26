---
title: >-
  [论文解读] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis
description: >-
  [CVPR 2026][医学图像][自监督学习] 提出 FPRL，一个受临床认知启发的层次化自监督框架，通过先"聚焦"帧内病灶关键静态语义、再"感知"帧间上下文演化来缓解运动偏差，在 11 个内窥镜数据集上取得 SOTA。
tags:
  - CVPR 2026
  - 医学图像
  - 自监督学习
  - 内窥镜视频
  - 层次化语义建模
  - 掩码重建
  - Mamba
---

# Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis

**会议**: CVPR 2026  
**arXiv**: [2603.25778](https://arxiv.org/abs/2603.25778)  
**代码**: [有](https://github.com/MLMIP/FPRL)  
**领域**: Medical Imaging / 内窥镜视频分析  
**关键词**: 自监督学习, 内窥镜视频, 层次化语义建模, 掩码重建, Mamba

## 一句话总结

提出 FPRL，一个受临床认知启发的层次化自监督框架，通过先"聚焦"帧内病灶关键静态语义、再"感知"帧间上下文演化来缓解运动偏差，在 11 个内窥镜数据集上取得 SOTA。

## 研究背景与动机

内窥镜视频分析对胃肠道疾病的早期筛查至关重要，但高质量标注稀缺严重限制了算法性能。自监督视频预训练是应对标注不足的有力方向，然而现有方法（如 VideoMAE、VideoMAE V2 等）主要面向自然视频设计，强调密集时空建模和运动语义——这对动作识别等任务有效，但与内窥镜视频的核心特征相矛盾。

内窥镜视频的关键语义依赖于**静态、局部的视觉线索**（如病灶的形态、颜色和纹理），而非显著的时间动态。当密集时空建模直接迁移到内窥镜视频时，模型倾向于过度关注相机抖动、组织位移等无关运动（作者称之为"运动偏差"），忽略了对诊断至关重要的静态语义。

作者观察到，有经验的内镜专家在阅片时遵循"先聚焦、后感知"的认知模式：先仔细检查单帧中的语义显著区域（颜色、纹理异常），再追踪这些候选区域的时间演化。这一临床认知过程启发了 FPRL 框架的设计。

## 方法详解

### 整体框架

FPRL 由两个层次化的语义建模组件组成：

1. **静态语义聚焦（Static Semantic Focus）**：捕获帧内以病灶为中心的局部语义
2. **上下文语义感知（Contextual Semantic Perception）**：建模跨视图的时间演化，保持上下文一致性

整体采用教师-学生（Teacher-Student）范式：教师编码器（预训练 VideoMamba-S，冻结权重）处理当前视图，学生编码器（EndoMamba-S，从头训练）处理过去和未来视图。

### 关键设计

#### 1. 多视图稀疏采样

从视频序列的一个时间窗口中独立采样 3 个稀疏视图（过去、当前、未来），每个视图包含 2 帧。这种采样策略直觉上抑制了视频输入的动态冗余，同时保持了视图间的语义多样性。

#### 2. 教师先验自适应掩码（TPAM）

TPAM 是本文的核心创新之一。其设计动机在于：传统随机掩码无法区分病灶区域与背景区域。

- **教师显著性先验**：对教师特征进行 $\ell_2$ 归一化，得到显著性图 $H$，增强病灶相关 token 的选择
- **轻量注意力头**：对当前视图的嵌入施加多头自注意力 + 线性投影，生成逻辑值 $R$，捕获互补的图像特定显著性
- **融合采样**：$S = \alpha H + (1-\alpha)R$，通过 Top-K 选择可见 patch，得到可学习的二值掩码 $M$
- 学生编码器仅对可见 patch 编码，使模型集中表征能力于病灶相关语义

#### 3. 跨视图掩码特征补全（CVMFC）

在隐空间中建立精细的跨视图对应关系，通过从相邻视图检索语义来补全当前视图的被掩码特征：

- 使用 Transformer 风格的跨注意力块（cross-attention → self-attention → FFN）
- 当前视图特征作为 query，过去/未来视图特征分别作为 key/value
- 输出 $z_c^p$ 和 $z_c^f$ 分别与冻结的教师特征 $z_t$ 对齐

#### 4. 注意力引导的时间预测（AGTP）

在视图级别强制时间对应一致性，补充 CVMFC 的 token 级检索：

- 利用 CVMFC 计算的跨注意力图对相邻视图 token 进行加权池化
- 通过 EMA 更新的目标头提供稳定的非退化目标
- 与当前视图的全局平均池化特征进行对比学习

### 损失函数 / 训练策略

总损失由三部分加权组合：

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Rec} + \lambda_2 \mathcal{L}_{Align} + \lambda_3 \mathcal{L}_{CL}$$

| 损失项 | 作用 | 权重 |
|--------|------|------|
| 像素重建损失 $\mathcal{L}_{Rec}$ | 恢复被掩码 token 的病灶纹理/边界细节 | $\lambda_1 = 1.0$ |
| 跨视图特征对齐损失 $\mathcal{L}_{Align}$ | 建立 token 级的时间对应（余弦相似度 + $\ell_2$ 一致性） | $\lambda_2 = 0.8$ |
| 时间预测对比损失 $\mathcal{L}_{CL}$ | InfoNCE 对比学习，保持视图级时间一致性 | $\lambda_3 = 1.0$ |

训练使用 AdamW 优化器，学习率 1.5e-4，余弦调度，400 个 epoch，batch size 64，前 40 个 epoch 线性 warmup。在 4 块 NVIDIA A800 上训练。

## 实验关键数据

### 主实验

| 方法 | 会议/年份 | 预训练时间(h) | PolypDiag F1(%) | CVC-12k Dice(%) | KUMC F1(%) |
|------|-----------|---------------|-----------------|-----------------|------------|
| Scratch | - | N/A | 83.5 | 53.2 | 73.5 |
| VideoMAE | NeurIPS'22 | 25.3 | 91.4 | 80.9 | 82.8 |
| Endo-FM | MICCAI'23 | 20.4 | 90.7 | 73.9 | 84.1 |
| M2CRL | NeurIPS'24 | 24.3 | 94.2 | 81.4 | 86.3 |
| EndoMamba | MICCAI'25 | 38.2 | 94.5 | 84.5 | 88.8 |
| **FPRL (Ours)** | - | **18.2** | **95.2** | **86.1** | **89.8** |

在相同模型架构下，FPRL 比 EndoMamba 分别提升了 0.7%/1.6%/1.0%，同时预训练时间减少了 52%。

### 消融实验

| $\mathcal{L}_{Rec}$ | $\mathcal{L}_{CL}$ | $\mathcal{L}_{pt}$ | $\mathcal{L}_{ft}$ | $\mathcal{L}_{pf}$ | 分类 | 分割 | 检测 |
|:---:|:---:|:---:|:---:|:---:|------|------|------|
| ✓ | | | | | 92.3 | 83.8 | 84.0 |
| ✓ | ✓ | ✓ | ✓ | | 94.2 | 84.0 | 86.1 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **95.2** | **86.1** | **89.8** |

掩码策略消融：

| 掩码策略 | 分类(%) | 分割(%) | 检测(%) |
|---------|---------|---------|---------|
| Random | 93.8 | 85.6 | 87.8 |
| Adaptive | 94.5 | 85.6 | 83.9 |
| Teacher-Prior + Adaptive (Ours) | **95.2** | **86.1** | **89.8** |

最优掩码比例为 90%，过高（95%）或过低（70%）都会降低性能。

### 关键发现

- 层次化语义建模（分离静态 + 上下文语义）是性能提升的核心
- 双路径掩码补全（past + future）比单路径分别提升约 1.9%/2.0%/3.6%
- 教师先验与自适应掩码的组合效果最佳，仅用其一不能充分捕获病灶特征
- 4 层 decoder + 1 个 CVMFC block 就够了，更深的设计导致特征过度平滑

## 亮点与洞察

1. **认知启发的设计范式**：将临床医生"先聚焦、后感知"的诊断流程系统地转化为技术方案，具有很强的可解释性
2. **运动偏差的显式建模**：明确提出内窥镜视频的"运动偏差"概念，并通过层次化框架系统地应对
3. **效率优势**：预训练时间仅 18.2h，比 EndoMamba（38.2h）少 52%，比 VideoMamba（55.4h）少 67%
4. **TPAM 设计精巧**：将教师网络的全局先验与轻量注意力头的局部信息融合，使掩码学习具有自适应性

## 局限与展望

- 单帧预训练变体因内窥镜常见伪影（运动模糊、光照闪烁、镜面反射）效果不佳
- 未来可探索质量感知采样策略，避免低质量帧对训练的干扰
- 框架目前仅在内窥镜领域验证，向其他医学影像领域的泛化尚待探索
- Mamba 架构对超长序列的扩展性值得进一步研究

## 相关工作与启发

- EndoMamba 的双向/单向 Mamba 混合拓扑为 FPRL 提供了空间-时间解耦的基础
- 教师-学生范式借鉴了 BYOL 等自监督方法的 EMA 更新思想
- M2CRL 在多视图掩码对比学习方面的探索启发了 CVMFC 的设计
- 对于内窥镜视频分析这一特定领域，领域知识（如临床诊断流程）可以显著指导框架设计

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "聚焦-感知"层次化范式原创性强，TPAM 融合教师先验与自适应注意力的设计有新意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 11 个数据集、4 个下游任务、丰富的消融实验覆盖了掩码策略/比例/架构/损失等多个维度
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法描述系统，但数学符号较多可能增加阅读负担
- **价值**: ⭐⭐⭐⭐ — 对内窥镜视频领域的自监督学习有实质性推进，认知启发的设计思路可推广到其他医学影像分析任务

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)
- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_large_endoscopic_monocular_dataset_foundation_model_surgical.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)
- [\[AAAI 2026\] MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis](../../AAAI2026/medical_imaging/medeyes_learning_dynamic_visual_focus_for_medical_progressive_diagnosis.md)

<!-- RELATED:END -->
