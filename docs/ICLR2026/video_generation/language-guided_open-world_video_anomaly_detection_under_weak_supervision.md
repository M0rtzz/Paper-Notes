---
title: >-
  [论文解读] Language-guided Open-world Video Anomaly Detection under Weak Supervision
description: >-
  [ICLR 2026][视频异常检测] 提出语言引导的开放世界视频异常检测范式LaGoVAD，通过将异常定义建模为随机变量并以自然语言形式输入，结合动态视频合成和对比学习正则化策略，在七个数据集上实现零样本SOTA性能。
tags:
  - ICLR 2026
  - 视频异常检测
  - 开放世界
  - 语言引导
  - 概念漂移
  - 弱监督
---

# Language-guided Open-world Video Anomaly Detection under Weak Supervision

**会议**: ICLR 2026  
**arXiv**: [2503.13160](https://arxiv.org/abs/2503.13160)  
**代码**: [GitHub](https://github.com/Kamino666/LaGoVAD-PreVAD)  
**领域**: video_understanding  
**关键词**: 视频异常检测, 开放世界, 语言引导, 概念漂移, 弱监督

## 一句话总结

提出语言引导的开放世界视频异常检测范式LaGoVAD，通过将异常定义建模为随机变量并以自然语言形式输入，结合动态视频合成和对比学习正则化策略，在七个数据集上实现零样本SOTA性能。

## 研究背景与动机

1. **领域现状**: 视频异常检测（VAD）旨在识别偏离预期模式的视频帧，广泛应用于智能监控等领域。近年来弱监督方法在封闭集设定下取得了不错的性能。

2. **现有痛点**: 现有方法假设异常定义是固定不变的，无法应对开放世界中异常定义可能随需求变化的情况。例如，不戴口罩在流感期间是异常行为，但平时是正常的——这构成了概念漂移（concept drift）问题。

3. **核心矛盾**: 开放集和领域泛化方法虽然能检测训练集之外的新类别异常，但仍假设异常定义不变，无法处理同一行为在不同场景下标签改变的情况（如行人在公路上行走在犯罪数据集中是正常的，但在高速公路监控中是异常的）。

4. **本文目标**: 提出一种允许用户在推理时通过自然语言动态定义异常的开放世界VAD范式，从根本上避免概念漂移。

5. **切入角度**: 将异常定义 $Z$ 显式建模为随机变量，将预测条件化为视频 $V$ 和定义 $Z$ 的联合函数 $\Phi:(V,Z)\rightarrow Y$，使 $P(Y|V,Z)$ 恒定不变，从而理论上消除概念漂移。

6. **核心 idea**: 通过将异常定义作为输入条件，学习视频-文本-标签三元组的联合映射，并以大规模多样化数据集支撑泛化能力。

## 方法详解

### 整体框架

LaGoVAD 采用双分支架构：视频分支通过预训练CLIP图像编码器和Transformer时序编码器提取视频特征 $v^t = \mathcal{F}(v)$；文本分支通过CLIP文本编码器提取异常定义特征 $z^t = \mathcal{G}(z)$。两组特征经Transformer融合模块 $\mathcal{U}$ 融合后，分别送入二分类检测头 $\mathcal{H}^{\text{bin}}$ 和多分类头 $\mathcal{H}^{\text{mul}}$。训练过程中使用四个损失函数：MIL损失、MIL-align损失、动态视频合成损失 $\mathcal{L}_{\text{dvs}}$ 和对比学习损失 $\mathcal{L}_{\text{neg}}$。

### 关键设计

**1. 动态视频合成（Dynamic Video Synthesis）**

- **功能**: 增加训练数据中异常持续时间比例的多样性
- **核心思路**: 实时动态合成不同长度的视频。模块先决定生成正常/异常视频，然后确定段数，从K近邻中选择语义相似的视频片段拼接。锚点位置转化为二值伪标签 $y^p \in \{0,1\}^L$，通过 $\mathcal{L}_{\text{dvs}}$ 监督
- **设计动机**: 真实场景中异常通常只占视频的少部分，而网络来源的数据集异常比例偏高。通过合成不同时长比例的视频来缓解这种分布偏差

**2. 对比学习与硬负样本挖掘（Contrastive Learning with Hard Negative Mining）**

- **功能**: 增强正常/异常帧的特征可分性，实现细粒度的跨模态对齐
- **核心思路**: 使用异常分数作为权重将帧级视频特征聚合为视频级特征。异常视频中的正常部分作为硬负样本与对应的异常描述进行对比学习。损失函数包括文本→视频和视频→文本两个方向的对比损失
- **设计动机**: 异常视频中正常和异常帧之间的边界模糊，需要通过对比学习增强判别性；多模态联合空间的样本密度指数衰减，需要更好的对齐策略

**3. PreVAD 大规模预训练数据集**

- **功能**: 提供多样化的 $(v, z, y)$ 三元组支撑语言引导范式的训练
- **核心思路**: 通过可扩展的数据管线，从视频文本数据集、网络资源和监控流中聚合视频，利用MLLM自动化清洗和标注。包含35,279个视频、7大类35小类异常、每个异常视频附有文本描述
- **设计动机**: 现有数据集规模小（最大5K视频）、领域覆盖有限、缺乏语义描述标注，无法支撑开放世界范式的训练

### 损失函数 / 训练策略

总损失为四项之和：$\mathcal{L} = \mathcal{L}_{\text{MIL}} + \mathcal{L}_{\text{MIL-align}} + \mathcal{L}_{\text{dvs}} + \mathcal{L}_{\text{neg}}$

- $\mathcal{L}_{\text{MIL}}$: 多实例学习损失，用于时序二分类检测
- $\mathcal{L}_{\text{MIL-align}}$: MIL对齐损失，用于视频级多类分类
- $\mathcal{L}_{\text{dvs}}$: 动态视频合成损失（公式7-8），基于合成视频的伪标签监督
- $\mathcal{L}_{\text{neg}}$: 对比损失（公式10-11），包含硬负样本挖掘的双向对比学习

## 实验关键数据

### 主实验

Protocol 1：零样本跨数据集二分类异常检测（AUC/AP）

| 数据集 | 指标 | 本文(LaGoVAD) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| UCF-Crime | AUC | 82.81 | 82.42 (OVVAD) | +0.39 |
| XD-Violence | AP | 76.28 | 63.74 (OVVAD) | +12.54 |
| MSAD | AUC | 88.09+ | — | — |
| DoTA | AUC | 优于所有基线 | — | — |
| TAD | AUC | 优于所有基线 | — | — |

Protocol 2：概念漂移评估（drift@5），LaGoVAD在不同异常定义下均表现稳定，优于VadCLIP和LLM-based方法。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整模型 | 最佳 | 所有组件协同工作 |
| 去除动态视频合成 | 下降 | 异常持续时间多样性不足导致overfitting |
| 去除对比学习 | 下降 | 特征对齐质量降低 |
| 去除文本分支 | 大幅下降 | 退化为固定定义模式，无法处理概念漂移 |

### 关键发现

- 在XD-Violence上检测和分类分别提升20%和32%，体现了语言引导范式在跨域泛化上的巨大优势
- PreVAD的规模和多样性对模型性能至关重要，35K视频的多样化训练集是泛化的关键
- 概念漂移评估协议（drift@5）证明模型能有效应对异常定义的动态变化

## 亮点与洞察

- **理论贡献扎实**: 通过概率论形式化了概念漂移问题，证明了将异常定义作为条件输入可以消除概念漂移，理论与实践紧密结合
- **范式创新**: 从固定异常定义到动态语言引导定义的转变，开创了VAD领域的新范式
- **大规模数据集**: PreVAD是目前最大最多样的视频异常检测数据集，具有独立的评估价值
- **实用性强**: 用户可通过自然语言灵活定义异常，适应不同场景需求

## 局限与展望

- 依赖CLIP作为骨干网络，可能继承其在细粒度视觉理解上的局限
- 数据集虽大但仍以网络视频为主，与真实监控场景存在域差距
- 推理时需要用户提供合适的异常定义文本，定义质量直接影响检测效果
- 可考虑引入更强的视频理解模型（如VideoLLM）替代CLIP特征提取

## 相关工作与启发

- OVVAD等开放词汇方法虽能检测新类别但假设定义固定，本文的"定义作为输入"思路更灵活
- 与LAVAD等LLM-based方法相比，LaGoVAD在保持轻量级的同时实现了更好的性能
- 动态视频合成的思路可推广到其他视频理解任务的数据增强中

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 开创性地将概念漂移问题形式化并提出语言引导VAD范式
- 实验充分度: ⭐⭐⭐⭐ 七个数据集的零样本评估+两个评估协议，非常全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，理论推导严谨
- 价值: ⭐⭐⭐⭐⭐ 范式级贡献，PreVAD数据集也具有独立价值

<!-- RELATED:START -->

## 相关论文

- [DisMo: Disentangled Motion Representations for Open-World Motion Transfer](../../NeurIPS2025/video_generation/dismo_disentangled_motion_representations_for_openworld_moti.md)
- [GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection](../../AAAI2026/video_generation/genvidbench_a_6-million_benchmark_for_ai-generated_video_detection.md)
- [LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)
- [LAMP: Language-Assisted Motion Planning for Controllable Video Generation](../../CVPR2026/video_generation/lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision](../../CVPR2026/video_generation/vanast_virtual_try-on_with_human_image_animation_via_synthetic_triplet_supervisi.md)

<!-- RELATED:END -->
