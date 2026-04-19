---
title: >-
  [论文解读] Unleashing Video Language Models for Fine-grained HRCT Report Generation
description: >-
  [CVPR 2026][医学图像][CT报告生成] 提出 AbSteering 两阶段框架，利用异常中心的 CoT 推理和 DPO 硬负样本对比学习，将通用 VideoLM 高效适配到 HRCT 报告生成，在临床效能指标上大幅超越专用 CT 基础模型。
tags:
  - CVPR 2026
  - 医学图像
  - CT报告生成
  - 视频语言模型
  - Chain-of-Thought
  - DPO
  - 异常检测
---

# Unleashing Video Language Models for Fine-grained HRCT Report Generation

**会议**: CVPR 2026  
**arXiv**: [2603.12469](https://arxiv.org/abs/2603.12469)  
**代码**: [GitHub](https://anonymous.4open.science/r/hrct-report-generation-video-vlm-728C/)  
**领域**: 医学图像  
**关键词**: CT报告生成, 视频语言模型, Chain-of-Thought, DPO, 异常检测

## 一句话总结

提出 AbSteering 两阶段框架，利用异常中心的 CoT 推理和 DPO 硬负样本对比学习，将通用 VideoLM 高效适配到 HRCT 报告生成，在临床效能指标上大幅超越专用 CT 基础模型。

## 研究背景与动机

**临床需求**：高分辨率计算机断层扫描（HRCT）是胸部和心肺疾病诊断与纵向监测的关键模态，AI 驱动的报告生成能够减轻临床工作量、标准化诊断叙述并缓解观察者间差异。然而相比 2D 胸片，3D HRCT 报告生成面临更大挑战：每个研究包含数百层切片，计算和内存开销巨大；同时临床关键异常通常细微、空间局部化且多样，稀疏分布在体积中，常被占主导的正常解剖结构所掩盖。

**现有方法的不足**：早期方法将 CT 体积压缩为低维表示后复用 X-ray 报告生成器，信息损失严重。后续工作如 Dia-LLaMA 设计了 CT 专用视觉编码器并接入 LLM 解码器。近期的模态特定基础模型（RadFM、CT-CHAT、M3D）虽然进一步提升了性能，但仍依赖从头训练或大量微调模态特定编码器，数据和计算成本高昂，且在长尾异常的细粒度识别上仍有瓶颈。

**核心洞察**：HRCT 体积可以自然视为"视频式切片序列"，而 VideoLM 的架构（时空 tokenization + 3D 注意力 + token 合并 + LLM 解码）与 CT 基础模型本质相似，二者的差异不在于架构本身而在于训练域和监督信号。这引出三个关键问题：(1) VideoLM 的编码器能否捕获临床相关的 3D 特征？(2) 如何高效地将通用 VideoLM 适配到领域特定的医学报告生成？(3) 这种迁移与模态特定 CT 基础模型相比表现如何？

## 方法详解

### 整体框架

AbSteering 以预训练 VideoLM 为骨干，分两个阶段进行域适配：Stage 1 通过异常中心的 Chain-of-Thought 训练建立结构化推理路径；Stage 2 通过 Direct Preference Optimization 进行细粒度异常辨别。整个框架不修改视觉编码器架构，仅通过语言层面的引导实现域适配。

### 关键设计

1. **异常中心的 Chain-of-Thought 训练（Stage 1）**：

    - 功能：将 vision-to-text 任务解耦为"先检测异常，再生成报告"的两步推理过程
    - 核心思路：首先将原始 CT-RATE 报告标准化为统一的 (region: abnormality) 模板，覆盖 10 个解剖区域（Lung、Trachea and Bronchi、Mediastinum、Heart、Esophagus、Pleura、Bone、Thyroid、Breast、Abdomen），使用 GPT-4o 将报告句子分配到对应区域并人工校验，生成 CT-RATE-AB 数据集。然后采用序列生成目标，目标序列 $Y = [R_{AB}; R_{Full}]$，即先生成异常检测列表 $R_{AB}$，再生成完整报告 $R_{Full}$，通过自回归损失 $\mathcal{L}_{gen} = -\sum_{t=1}^{T} \log P(y_t | x, y_{<t})$ 训练
    - 设计动机：强制模型在生成最终报告前显式进行临床推理，学习疾病类别多样性，抑制被正常组织主导的描述和幻觉；在推理层面，从离散发现到叙述的过渡让模型捕获解剖约束和病理间依赖关系（如相关疾病共现或矛盾发现的互斥性）

2. **基于 DPO 的细粒度异常辨别（Stage 2）**：

    - 功能：增强模型对细微病理差异的区分能力，抑制幻觉
    - 核心思路：用 GPT-4o 从真实异常报告 $R_{AB}$ 自动构造硬负样本 $R_{AB\_Fake}$——将目标异常替换为同一解剖区域内临床易混淆的异常，保持区域标签、句子模板和位置信息不变。然后以 Stage 1 模型为参考模型 $\pi_{ref}$，通过 DPO 目标优化目标模型 $\pi_\theta$：$\mathcal{L}_{DPO} = \log \sigma(\beta \log \frac{\pi_\theta(y_w|x,v)}{\pi_{ref}(y_w|x,v)} - \beta \log \frac{\pi_\theta(y_l|x,v)}{\pi_{ref}(y_l|x,v)})$，其中 $y_w = R_{AB}$（正确报告），$y_l = R_{AB\_Fake}$（篡改报告）
    - 设计动机：CT 异常常呈现细微且视觉易混淆的模式，细粒度辨别高度依赖领域特定临床知识。通过对比学习正确报告与临床混淆报告，迫使模型关注决定报告质量的细微视觉线索

3. **VideoLM 骨干架构**：

    - 功能：提供时空推理基础
    - 核心思路：输入视频 $X \in \mathbb{R}^{T \times H \times W \times C}$ 经时空 cube tokenization 得到视觉 token，通过带有分解式 3D 位置嵌入的 Transformer 处理，再由 merger 压缩为语言对齐 token 送入 LLM 解码。本文评估了 Qwen2.5-VL-7B 和 InternVL3-8B 两个骨干
    - 设计动机：VideoLM 与 CT 基础模型架构高度相似，差异仅在训练域，因此可直接复用其时空推理能力

### 损失函数 / 训练策略

- **Stage 1**：标准自回归交叉熵损失，目标序列为异常列表与完整报告的级联 $[R_{AB}; R_{Full}]$
- **Stage 2**：DPO 损失，超参数 $\beta$ 控制偏离参考模型的幅度
- **数据预处理**：每个 HRCT 转换为 240 帧 480×480 像素切片，HU 窗口 [-1000, 200]，保存为 MP4 格式，帧率 18fps
- **训练设置**：2 块 80GB A100 GPU，总 batch size 4；视觉编码器冻结，不做 LoRA 微调
- **数据集**：CT-RATE 训练集 46,717 个 CT 扫描（20,000 患者），验证集 3,039 个扫描（1,314 患者）

## 实验关键数据

### 主实验

CT-RATE 基准上的全面对比，评估自然语言生成（NLG）和临床效能（CE）指标：

| 方法 | BL-1 | BL-4 | RG-L | BERT | CE Micro P | CE Micro R | CE Micro F1 | CE Macro F1 | CE Wtd F1 | CE Samp F1 |
|------|------|------|------|------|-----------|-----------|------------|------------|----------|----------|
| CT2Rep | 47.91 | 28.04 | 45.43 | 88.10 | 26.39 | 10.50 | 14.10 | 10.65 | 11.35 | 10.86 |
| RadFM | 50.20 | 17.02 | 30.46 | 86.17 | 36.10 | 13.48 | 19.63 | 13.05 | 17.74 | 12.14 |
| Reg2RG | 44.89 | 21.08 | 24.41 | 86.18 | 28.47 | 11.06 | 15.93 | 10.48 | 14.51 | 12.19 |
| CT-CHAT | 42.81 | 17.63 | 32.50 | 86.35 | 25.13 | 37.48 | 30.08 | 21.66 | 28.35 | 25.31 |
| M3D-8B | 44.95 | 22.98 | 37.76 | 87.52 | 47.60 | 28.54 | 35.69 | 26.74 | 33.13 | 25.21 |
| Qwen2.5-VL-7B | 43.67 | 21.25 | 36.71 | 87.30 | 48.06 | 25.88 | 33.64 | 25.57 | 32.19 | 24.95 |
| InternVL3-8B | 45.57 | 22.05 | 38.49 | 87.40 | 53.57 | 37.99 | 44.45 | 38.91 | 43.28 | 32.14 |
| M3D-AbSteer | 45.22 | 23.09 | 38.58 | 87.83 | 44.95 | 41.66 | 43.24 | 36.18 | 41.89 | 36.54 |
| Qwen2.5-VL-AbSteer | 45.64 | 21.40 | 37.99 | 87.13 | 49.15 | 43.22 | 45.99 | 37.90 | 44.05 | 37.39 |
| **InternVL3-AbSteer** | **48.32** | **23.58** | **40.49** | 87.59 | **57.88** | **51.58** | **54.55** | **47.66** | **52.80** | **44.80** |

### 消融实验

**AbSteering 策略消融**（基于 InternVL3-8B）：

| 配置 | CE Micro P | CE Micro R | CE Micro F1 |
|------|-----------|-----------|------------|
| Baseline（无 steering） | 53.57 | 37.99 | 44.45 |
| + CoT（Stage 1） | — | ↑↑ | ↑ |
| + CoT + DPO（完整 AbSteering） | 57.88 | 51.58 | 54.55 |

CoT 显著提升 recall，DPO 进一步提升 precision 并抑制幻觉，二者协同实现 F1 从 44.45 → 54.55（+22.7%）。

**视觉编码器消融**（基于 Qwen2.5-VL + Stage 1 CoT）：

| 编码器策略 | 效果 |
|-----------|------|
| 从头训练（无预训练） | 性能急剧下降 |
| 冻结预训练编码器 | 最优 |
| LoRA 微调（rank=8） | 无额外增益 |

**LLM 规模消融**：

| LLM 规模 | 趋势 |
|----------|------|
| 3B | 基线 |
| 7B | 性能提升 |
| 32B | 性能反而下降 |

### 关键发现

- **通用 VideoLM 具备强迁移性**：未经 steering 的 InternVL3-8B 在 CE Micro F1 上达到 44.45，已超越最强专用基础模型 M3D-8B 的 35.69
- **AbSteering 大幅提升**：InternVL3 经 AbSteering 后 CE Micro F1 从 44.45 → 54.55（+22.7%），CE Macro F1 从 38.91 → 47.66（+22.5%）
- **跨模型通用性**：AbSteering 对 M3D（专用模型）和两种 VideoLM 均有效，但 VideoLM 的增益幅度更大
- **视频预训练至关重要**：从头训练导致性能急剧下降，冻结编码器即已足够，LoRA 微调无额外增益——说明通用视频预训练的时空特征已足够鲁棒
- **LLM 并非越大越好**：7B → 32B 性能反降，当前瓶颈在视觉-文本对齐而非 LLM 容量
- **VideoLM 在不增加幻觉的前提下实现了最高召回率**

## 亮点与洞察

1. **跨模态迁移范式的成功验证**：系统证明了通用视频预训练的时空推理能力可高效迁移到 3D 医学影像，为 CT 报告生成提供了一条数据高效、计算友好的新路径，避免了从头训练模态特定基础模型的高昂成本
2. **两阶段设计精准对症**：CoT 解决的是"找不全异常"的 recall 问题（通过强制先推理再生成），DPO 解决的是"分不清异常"的 precision 问题（通过临床混淆硬负样本对比），二者协同效应显著
3. **冻结编码器的启示**：LoRA 微调不带来额外增益这一发现令人惊讶，暗示 VideoLM 的视觉特征已具备足够的泛化性，域适配的关键在语言引导层面而非视觉表示层面
4. **结构化 CoT 数据集贡献**：CT-RATE-AB 将原始报告重组为 region-abnormality 格式并经人工校验，有助于社区后续研究

## 局限与展望

- **单一数据集验证**：仅在 CT-RATE（胸部 CT）上评估，未验证对腹部、头颅等其他部位 CT 的泛化能力
- **依赖 GPT-4o**：报告结构化和硬负样本构造均依赖 GPT-4o，引入额外成本和潜在偏差，且可能限制方法的可复现性
- **CT 转 MP4 的信息损失**：将 HU 值映射到视频格式不可避免地损失了 CT 特有的密度信息精度
- **大规模 LLM 的瓶颈**：32B 模型性能下降暗示当前视觉-文本对齐策略在更大规模上可能需要额外设计
- **临床部署距离**：评估仍基于自动指标（RadBERT 分类器），缺乏放射科医师的人工评估

## 相关工作与启发

- **CT 报告生成**：CT2Rep 首先提出从 3D CT 直接生成报告的基准；M3D 和 CT-CHAT 分别探索了专用 3D 医学基础模型路线；Reg2RG 引入区域引导的 referring and grounding 机制。本文的贡献在于证明通用 VideoLM 经适当引导即可超越这些专用模型
- **VideoLM 的医学应用**：本文是首批系统研究 VideoLM 向 3D 医学影像迁移的工作之一，启发了将视频理解领域的大量预训练知识复用于医学的新思路
- **DPO 在医学中的应用**：将 DPO 的硬负样本构造策略引入医学报告生成是新颖的尝试，临床混淆异常作为负样本比随机负样本更有效

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实验 | ⭐⭐⭐⭐ |
| 写作 | ⭐⭐⭐⭐ |
| 价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](orapo_oracle_rl_radiology_report_generation.md)
- [Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)
- [InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)

<!-- RELATED:END -->
