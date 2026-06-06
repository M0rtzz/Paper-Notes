---
title: >-
  [论文解读] FaceCoT: Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing
description: >-
  [CVPR 2026][人体理解][人脸反欺骗] 构建了首个面向人脸反欺骗（FAS）的大规模 VQA 数据集 FaceCoT（108 万样本，覆盖 14 种攻击类型），包含六层级 CoT 推理标注（从全局描述到局部推理到最终结论）…
tags:
  - "CVPR 2026"
  - "人体理解"
  - "人脸反欺骗"
  - "CoT推理"
  - "VQA数据集"
  - "渐进式学习"
  - "强化学习标注"
---

# FaceCoT: Harnessing Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing

**会议**: CVPR 2026  
**arXiv**: [2506.01783](https://arxiv.org/abs/2506.01783)  
**代码**: 即将开源（数据集 FaceCoT 将公开）  
**作者**: Honglu Zhang, Zhiqin Fang, Ningning Zhao, Saihui Hou, Long Ma, Renwang Pei, Zhaofeng He
**领域**: 人体理解  
**关键词**: Face Anti-Spoofing, CoT Reasoning, VQA Dataset, Progressive Learning, Reinforcement Learning Annotation

## 一句话总结

构建了首个面向人脸反欺骗（FAS）的大规模 VQA 数据集 FaceCoT（108 万样本，覆盖 14 种攻击类型），包含六层级 CoT 推理标注；提出 CoT-Enhanced Progressive Learning (CEPL) 两阶段训练策略（先视觉增强再联合训练），在 11 个跨域基准上平均 AUC 提升 4.06%、HTER 降低 5.00%。

## 研究背景与动机

- **领域现状**：人脸反欺骗（FAS）是人脸识别系统的关键安全屏障，需要应对打印照片、屏幕回放、3D 面具等多种呈现攻击。现有方法主要依赖单一视觉模态（RGB/IR/Depth），虽然在域内测试中表现良好，但跨设备、跨环境、跨攻击类型的泛化能力严重不足。
- **现有痛点**：① 现有 FAS 数据集只有图像 + 二分类标签（real/fake），缺乏结构化的语言推理信息，无法为 MLLM 训练提供足够的监督信号；② 现有方法（DA/DG 范式）难以找到通用的假脸特征空间，面对训练中未见的攻击类型（如新型 3D 面具）泛化能力差；③ 模型决策过程不可解释，在安防等高风险场景中缺乏可信度。
- **核心矛盾**：MLLM 在图文理解和语义推理上的能力突破理论上非常适合 FAS 任务（融合视觉和语言进行因果推理），但缺乏高质量的 FAS 视觉-语言多模态数据集是关键瓶颈。直接用二分类标签训练 MLLM 会导致过拟合，也无法提供可解释的推理链。
- **本文目标** 如何构建大规模、高质量的 FAS CoT VQA 数据集？如何设计训练策略让 MLLM 充分利用 CoT 数据，同时解决推理与分类任务之间的干扰问题？
- **切入角度**：从数据和训练策略两个维度同时切入——数据端构建 108 万样本的六层级 CoT 标注数据集（Gold + Silver），训练端设计两阶段渐进学习策略避免任务干扰。
- **核心 idea**：用 GPT-4o + 人工精修构建 Gold100K 高质量 CoT 数据，再用 RL 增强的 caption 模型扩展到 Silver982K，通过两阶段渐进学习（先 CoT 预训练视觉编码器，再联合训练推理+分类）最大化利用 CoT 数据。

## 方法详解

### 整体框架

FaceCoT 分为 **数据构建** 和 **训练策略** 两大部分：

1. **数据构建流水线**：FaceCoT-Gold100K（GPT-4o 自动标注 + 人工精修）→ 训练 RL 增强的 FAS caption 模型 → FaceCoT-Silver982K（caption 模型自动标注）= 总计 108 万样本
2. **训练策略 CEPL**：Stage 1 Visual Enhancement Pre-training（全参数 SFT on CoT）→ Stage 2 Multi-task Joint Training（继承视觉编码器 + LoRA 联合训练推理与分类）

### 关键设计一：六层级 CoT 标注结构

模拟人类判断图像真伪的 **"全局到局部"** 认知路径，将 CoT 标注分为六个层级模块：

1. **Caption**：对整张图像的全局场景描述，帮助模型理解环境上下文和宏观欺骗特征
2. **Facial Description**：聚焦面部区域，描述面部特征——面部是欺骗攻击最容易伪造的区域
3. **Facial Attributes**：进一步描述面部属性（面部纹理、表情、肤色等），增强对细粒度面部细节的感知
4. **Reasoning**：基于前三层的多尺度信息进行综合分析，判断图像中是否存在欺骗行为
5. **Spoofing Description**：描述具体的欺骗特征和欺骗方法（如反光/切割痕迹），提升可解释性
6. **Conclusion**：最终判定（Yes/No）

每个层级用 XML 标签格式化（如 `<Caption></Caption>`），为模型提供清晰的结构化输入。

- **设计动机**：人类判断真伪的认知过程是层级式的——先看整体场景/环境，再聚焦面部细节，最后综合推理得出结论。这种结构化标注不仅保证逻辑一致性，还为下游推理模型提供了自然的学习路径。

### 关键设计二：数据构建流水线（Gold + Silver）

**FaceCoT-Gold100K 构建**：

- **数据来源**：从 CelebA-Spoof（625K 图像，10 种攻击）和 WFAS（1.38M 图像，14 种攻击）中精选样本
- **类别平衡**：分为 live/replay/print/mask 四大类，每类约 25K 样本，各子类型尽量均匀采样
- **GPT-4o 标注**：为不同攻击类型提供针对性 hint（如"拍摄海报构成欺骗"），防止模型识别出特征却无法正确划定决策边界
- **质量控制**：正则匹配提取 conclusion 标签与 ground-truth 对比 → 98,976 样本正确标注 → 错误样本进行二轮 GPT-4o 标注 → 仍然错误的 581 个 hard case 由专业标注员人工修正

**FaceCoT-Silver982K 构建**：

- **FAS Caption 模型训练**：在 Gold100K 上 SFT 训练 caption 模型
- **RL 增强**：针对 SFT 模型生成的 CoT 存在语义错误和格式错误的问题，设计双奖励 RL 策略：
    - **准确性奖励**：`<Conclusion>` 标签匹配 ground-truth 标签 → 奖励 1，否则 0
    - **格式奖励**：输出符合 FaceCoT XML 模板 → 奖励 1，否则 0
- **效果**：标注准确率从纯 SFT 的 88% 提升至 **99.6%**
- 最终用 RL 增强后的 caption 模型标注 CelebA-Spoof 和 WFAS 的训练集，生成 982K 条高质量 CoT 标注

### 关键设计三：CoT-Enhanced Progressive Learning (CEPL)

**动机**：如果用端到端单阶段方式同时训练 CoT 推理和二分类，分类任务收敛更快，会导致推理任务欠优化——模型无法充分利用 CoT 标注中的细粒度视觉线索，视觉编码器无法学到关键的欺骗伪影特征。

**Stage 1 — Visual Enhancement Pre-training（视觉增强预训练）**：

- 对 MLLM 进行 **全参数 SFT**，仅使用 CoT 数据
- 输入图像，监督信号为对应的推理文本
- 目标：通过语言引导的监督信号驱动视觉编码器精确对齐视觉特征和语言描述，增强对微妙欺骗伪影的敏感性
- 此阶段不做分类，视觉编码器可以集中精力学习细粒度面部特征

**Stage 2 — Multi-task Joint Training（多任务联合训练）**：

- **继承** Stage 1 的视觉编码器（保留其细粒度面部表征）
- **重置** 连接层（connector）和语言解码器为原始预训练权重
- 对 LLM 施加 **LoRA** 模块进行目标微调
- 在 CoT 标注数据 + 二分类标签数据上 **联合训练**，同时优化推理和分类
- 最终模型既能准确区分真假脸，又能产生连贯的 CoT 推理解释

### 损失函数 / 训练策略

- **Backbone**: MiniCPMV-2.6-8B（轻量级多模态架构，强跨模态融合能力）
- **输入分辨率**: 448×448×3（RGB）
- **优化器**: AdamW，初始 lr = 1e-6，weight decay = 0.1
- **训练**: 10 epochs，batch size 256，8× A100 GPU
- **推理评估**: 从生成的第一个 token 提取 Yes/No 的 logits，通过 softmax 计算连续置信度分数 $p_{\text{real}}$，用于计算 AUC 和 HTER
- **鲁棒性**: 所有实验使用三个不同随机种子运行并取平均

## 实验关键数据

### 主实验：1-to-11 跨域泛化（训练源→11个目标域）

| 方法 | CASIA-MFSD HTER/AUC | 3DMask HTER/AUC | HKBU HTER/AUC | HiFiMask HTER/AUC | MSU HTER/AUC | OULU HTER/AUC | Replay HTER/AUC | Rose HTER/AUC | SIW HTER/AUC | SIW-M HTER/AUC | WMCA HTER/AUC | **Avg HTER↓** | **Avg AUC↑** |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| ViTAF | 3.11/99.48 | 6.18/98.40 | 49.29/57.28 | 37.30/67.10 | 12.86/93.14 | 26.73/81.28 | 12.38/95.73 | 69.34/74.22 | 14.74/92.51 | 26.72/80.70 | 29.88/77.14 | 23.85 | 82.82 |
| FLIP | 4.88/98.48 | 8.83/96.93 | 17.25/88.31 | 28.32/76.50 | 19.37/89.98 | 20.57/87.30 | 25.67/81.37 | 80.73/73.60 | 11.01/95.40 | 25.95/80.78 | 19.36/88.73 | 18.73 | 87.90 |
| I-FAS | 1.11/99.88 | 6.18/98.40 | 18.64/88.77 | 28.23/77.17 | 5.63/98.73 | 14.86/91.68 | 9.15/95.12 | 5.52/98.48 | 4.02/98.34 | 10.89/95.02 | 20.07/89.17 | 11.30 | 93.71 |
| **Ours-100K** | 0.00/100 | 1.33/99.79 | 11.74/96.37 | 18.63/88.52 | 4.58/99.56 | 12.70/92.99 | 7.37/97.07 | 5.51/98.57 | 0.03/99.97 | 9.50/95.93 | 12.77/93.66 | **7.65** | **96.59** |
| **Ours-All** | 0.00/100 | 0.40/99.98 | 7.34/98.39 | 15.93/91.30 | 5.00/99.35 | 5.86/97.72 | 12.75/95.53 | 4.56/99.12 | 0.48/99.96 | 6.81/97.61 | 10.16/96.52 | **6.30** | **97.77** |

Ours-All 相比前 SOTA I-FAS：平均 HTER **降低 5.00%**（11.30→6.30），平均 AUC **提升 4.06%**（93.71→97.77），**在全部 11 个评测集上均取得最高性能**。

### Leave-One-Out 跨域协议（O/C/I/M 四数据集）

| 方法 | OCI→M HTER/AUC | OMI→C HTER/AUC | OCM→I HTER/AUC | ICM→O HTER/AUC | **Avg HTER↓** | **Avg AUC↑** |
|------|------|------|------|------|------|------|
| FLIP | 4.95/98.11 | 0.54/99.98 | 4.25/99.07 | 2.31/99.63 | 3.01 | 99.20 |
| I-FAS | 0.32/99.88 | 0.04/99.99 | 3.22/98.48 | 1.74/99.66 | 1.33 | 99.50 |
| **Ours** | 0.42/99.92 | 0.00/100.00 | 1.00/99.83 | 2.81/99.63 | **1.06** | **99.85** |

在已饱和的经典协议上仍取得最优，平均 HTER 1.06%，平均 AUC 99.85%。

### 消融实验

**CEPL 训练策略消融（Tab. 2）**：

| 方法 | Stage 1 | Stage 2 | Stage 3 | HTER(%)↓ | AUC(%)↑ |
|------|---------|---------|---------|----------|---------|
| Single-stage | - | MJT | - | 8.84 | 95.91 |
| CEPL + RL | VEP | MJT | RL | 7.80 | 96.82 |
| **CEPL (Ours)** | VEP | MJT | - | **7.65** | **96.59** |

CEPL 比单阶段降低 HTER 1.19%、提升 AUC 0.68%，证明两阶段渐进学习有效解决任务干扰。在已有大量高质量 SFT 数据后，额外 RL 的边际收益有限。

**CoT 数据 vs 纯标签（Tab. 3, 448×448）**：

| 数据类型 | HTER(%)↓ | AUC(%)↑ |
|----------|----------|---------|
| Label only | 9.07 | 95.05 |
| **Label + CoT** | **7.65** | **96.59** |

**低分辨率下 CoT 数据消融（Tab. 11, 224×224）**：

| 数据类型 | HTER(%)↓ | AUC(%)↑ |
|----------|----------|---------|
| Label only | 17.07 | 90.42 |
| **Label + CoT** | **11.28** | **94.05** |

低分辨率下 CoT 收益更大（HTER 降 5.79% vs 高分辨率降 1.42%），说明 CoT 帮助模型补偿了低分辨率下丢失的细粒度信息。

**RL 增强标注质量（Tab. 10）**：

| 训练方式 | HTER(%)↓ | AUC(%)↑ |
|----------|----------|---------|
| SFT | 8.00 | 96.97 |
| **SFT + RL** | **6.87** | **97.27** |

RL 不仅提升了 conclusion 准确率（88%→99.6%），还提升了 CoT 文本的语义连贯性和逻辑一致性。

**Zero-shot vs CoT 微调（Tab. 9, 跨 backbone）**：

| 方法 | Avg HTER(%)↓ | Avg AUC(%)↑ |
|------|-------------|-------------|
| I-FAS (AAAI 2025) | 11.31 | 93.71 |
| Zero-shot (Qwen2.5-VL-7B) | 19.60 | 83.75 |
| Zero-shot (MiniCPMV-2.6-8B) | 17.91 | 87.32 |
| **Ours (Qwen2.5-VL-7B)** | **9.32** (↓10.28) | **95.72** (↑11.97) |
| **Ours (MiniCPMV-2.6-8B)** | **6.30** (↓11.61) | **97.77** (↑10.45) |

FaceCoT + CEPL 在两种不同 backbone VLM 上均带来稳定显著的提升，证明方法的架构无关性。

### 关键发现

- **跨域泛化突出**：对训练中完全未见的攻击类型（如 HKBU 的透明/石膏/树脂面具、HiFiMask 的高保真面具），AUC 分别提升约 10% 和 14%，说明 CoT 推理提供了比传统特征学习更好的泛化能力。
- **分辨率效应**：448 vs 224 分辨率的 HTER 差异为 2.44%（单阶段），但 CoT 数据在低分辨率下的收益更大，说明 CoT 补偿了高频细节的损失。
- **细粒度攻击类型分析**（Rose-Youtu 7 种攻击）：Zero-shot 模型对 print/video 攻击的检测准确率为 **0%**，SFT 后分别提升到 68%-96%（Pq/Ps/Vl/Vm），而面具攻击的提升为 6-26%——说明 CoT 训练对视觉相似度高的 2D 攻击改善最大。

## 亮点与洞察

1. **开创性数据集**：FaceCoT 是 FAS 领域首个 VQA 数据集，108 万样本覆盖 14 种攻击类型。六层级 CoT 标注格式为下游模型提供了从粗到细的结构化推理路径，这种"认知路径模拟"的思路值得在其他安全检测任务中借鉴。

2. **RL 增强标注管线的可扩展性**：Gold100K（GPT-4o + 人工）→ RL 增强 caption 模型 → Silver982K 的流水线，将标注准确率从 88% 提至 99.6%，提供了低成本、高质量的数据扩展范式。这套流水线可复用到医学影像 VQA、遥感检测 VQA 等领域。

3. **两阶段训练设计的精妙**：先让视觉编码器通过 CoT 文本监督学习细粒度欺骗特征（Stage 1），再重置其他模块 + LoRA 联合训练（Stage 2），巧妙解决了推理与分类任务的收敛速度不一致问题。Stage 2 中保留 Stage 1 视觉编码器但重置其他模块的设计，保证了视觉表征的质量不被后续训练破坏。

4. **可解释性的实际意义**：在安防场景中，模型不仅给出 real/fake 判断，还输出完整的推理链（哪些视觉线索→如何推理→最终判断），这在合规性和可信 AI 方面有重要价值。

5. **超越原生 MLLM 的 zero-shot 能力**：MiniCPMV 和 Qwen2.5-VL 的 zero-shot FAS 性能均低于传统 SOTA（I-FAS），但通过 FaceCoT + CEPL 微调后大幅超越，说明通用 MLLM 的视觉推理能力需要任务特定的 CoT 数据激活。

## 局限与展望

1. **数据源局限**：FaceCoT 源自 CelebA-Spoof 和 WFAS，人口统计学多样性取决于原始数据集。部分罕见攻击类型（如 adultdull）仅 165 样本，数据极度不平衡。
2. **设备和环境覆盖不足**：虽然包含 14 种攻击类型，但某些真实场景中的设备变异（不同品牌手机、不同分辨率显示器）和环境变化（极端光照）未被充分覆盖。
3. **仅在 FAS 领域验证**：六层级 CoT 结构和 CEPL 训练策略是否可推广到其他安全检测任务（如 deepfake 检测、文档伪造检测）有待验证。
4. **评估维度单一**：FaceCoT 提供了推理链，具有作为标准化 benchmark 评估模型可解释性和推理连贯性的潜力，但本文未在此方向进行系统性探索。
5. **Leave-one-out 协议上优势微弱**：在已饱和的经典四数据集协议上（AUC 已 >99%），FaceCoT 的提升幅度有限（1.33→1.06 HTER），方法的优势主要体现在更困难的跨域场景。

## 相关工作与启发

| 方法 | 类型 | 数据模态 | 是否可解释 | 跨域泛化 | 核心局限 |
|------|------|---------|----------|---------|---------|
| ViTAF / ViT-B/L | 视觉分类器 | RGB | 否 | 弱 | 无推理链，泛化差 |
| FLIP (CVPR 2023) | CLIP+FAS | RGB+Text | 部分 | 中 | 文本仅做对齐，非推理 |
| I-FAS (AAAI 2025) | MLLM+FAS | RGB+Text | 简单描述 | 较强 | 仅提供简单描述，非结构化推理 |
| **FaceCoT** | MLLM+CoT+FAS | RGB+六层CoT | **完整推理链** | **强** | 数据源覆盖待扩展 |

**启发方向**：
- FaceCoT 的数据构建流水线（GPT-4o + 人工精修 → RL 增强 caption 模型扩展）可复用到医学影像、遥感、工业检测等领域的 VQA 数据集构建
- 两阶段训练策略（先视觉增强再联合训练）对其他需要细粒度视觉理解的 MLLM 任务（如细粒度分类、医学图像诊断）有参考价值
- RL 增强标注质量的双奖励设计（准确性 + 格式）可推广到其他自动数据标注场景
- 六层级 CoT 的"认知路径模拟"思路——为特定任务设计领域定制化的思维链结构——是 MLLM 在垂直领域落地的有效范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个 FAS VQA 数据集 + CoT 渐进式学习，将 MLLM 推理引入传统 CV 安全任务，方向新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 11 个跨域基准 + 两种协议 + 多种消融 + 跨 backbone 验证 + 细粒度攻击类型分析 + 分辨率消融 + RL 质量消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据构建→方法设计→实验验证的逻辑线流畅，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 数据集将开源，对 FAS 和更广泛的安全 AI 领域有重要推动作用；数据构建流水线和训练策略具有跨领域复用价值
---
title: >-
  [论文解读] FaceCoT: Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing
description: >-
  [CVPR 2026][LLM推理][人脸反欺骗] 构建了首个面向人脸反欺骗（FAS）的大规模 VQA 数据集 FaceCoT（108 万样本，覆盖 14 种攻击类型），包含六层级 CoT 推理标注（从全局描述到局部推理到最终结论）；同时提出 CoT-Enhanced Progressive Learning (CEPL) 两阶段训练策略，在 11 个基准数据集上平均 AUC 提升 4.06%、HTER 降低 5.00%，超越所有 SOTA 方法。
tags:
  - CVPR 2026
  - LLM推理
  - 人脸反欺骗
  - CoT推理
  - VQA数据集
  - 渐进式学习
  - 强化学习标注
---

# FaceCoT: Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing

**会议**: CVPR 2026  
**arXiv**: [2506.01783](https://arxiv.org/abs/2506.01783)  
**代码**: 即将开源 (数据集 FaceCoT 将公开)  
**领域**: 人体理解  
**关键词**: 人脸反欺骗, CoT推理, VQA数据集, 渐进式学习, 强化学习标注  

## 一句话总结
构建了首个面向人脸反欺骗（FAS）的大规模 VQA 数据集 FaceCoT（108 万样本，覆盖 14 种攻击类型），包含六层级 CoT 推理标注（从全局描述到局部推理到最终结论）；同时提出 CoT-Enhanced Progressive Learning (CEPL) 两阶段训练策略，在 11 个基准数据集上平均 AUC 提升 4.06%、HTER 降低 5.00%，超越所有 SOTA 方法。

## 背景与动机
现有 FAS 方法主要依赖单一视觉模态，泛化能力差且缺乏可解释性。MLLM 在图文理解和语义推理上的突破，为 FAS 提供了融合视觉和语言共同推理的新思路。然而关键瓶颈是**缺乏高质量的视觉-语言多模态 FAS 数据集**——现有 FAS 数据集仅提供图像 + 二分类标签，没有结构化的推理链信息。

## 核心问题
如何构建大规模、高质量的 FAS CoT VQA 数据集，并设计有效的训练策略让 MLLM 充分利用 CoT 数据提升检测性能和可解释性？

## 方法详解

### 整体框架
1. **数据构建**：FaceCoT-Gold100K（GPT-4o + 人工精修）+ FaceCoT-Silver982K（RL 增强的 caption 模型自动标注）= 108 万样本
2. **训练策略**：两阶段 CoT-Enhanced Progressive Learning (CEPL)

### 关键设计

1. **六层级 CoT 标注结构**: 模拟人类"全局到局部"推理路径：Caption（全局场景描述）→ Facial Description（面部特征描述）→ Facial Attributes（面部属性列举）→ Reasoning（基于多尺度信息的逻辑推理）→ Spoofing Description（欺骗特征和方法描述）→ Conclusion（最终判断 Yes/No）。用 XML 标签格式化，为模型提供清晰的推理结构。

2. **数据构建流水线**: 

    - Gold100K：GPT-4o 自动标注 + 为不同攻击类型提供针对性 hint（如"拍摄海报构成欺骗"）+ 正则匹配检查 → 二轮标注失败的 581 个 hard case 由专家人工修正
    - Silver982K：在 Gold100K 上 SFT 训练 caption 模型，再用双奖励 RL（准确性奖励：结论匹配标签=1；格式奖励：输出符合模板=1）增强，标注准确率从 88% 提升至 **99.6%**

3. **CEPL 两阶段训练**:

    - Stage 1（Visual Enhancement Pre-training）：全参数 SFT on CoT 数据，让视觉编码器学习提取细粒度欺骗特征。直觉：语言引导的监督信号可以驱动视觉编码器关注微妙的伪造痕迹
    - Stage 2（Multi-task Joint Training）：继承 Stage 1 的视觉编码器，重置连接层和语言解码器为预训练权重 + LoRA 微调，联合训练 CoT 推理和二分类损失。解决了端到端训练中分类目标快速收敛导致推理任务欠优化的问题

### 损失函数 / 训练策略
- 输入分辨率 448×448，backbone 为 MiniCPMV-2.6-8B
- AdamW 优化器，初始 lr=1e-6，weight decay=0.1
- 10 epochs，batch size 256，8× A100
- 评估：从第一个生成 token 提取 Yes/No logits 做 softmax 计算连续置信度分数

## 实验关键数据

### 1-to-11 跨域泛化（最挑战设置）

| 方法 | 平均 HTER ↓ | 平均 AUC ↑ |
|------|------------|-----------|
| I-FAS (AAAI 2025) | 11.30% | 93.71% |
| **Ours-100K** | 7.65% | 96.59% |
| **Ours-All** | **6.30%** | **97.77%** |

在全部 11 个评测集上均取得最高性能。特别是 HKBU-MARs-V1+ 和 HiFiMask（含训练中未见的攻击类型），AUC 分别提升约 10% 和 14%。

### Leave-one-out 协议

| 方法 | 平均 HTER ↓ | 平均 AUC ↑ |
|------|------------|-----------|
| I-FAS | 1.33% | 99.50% |
| **Ours** | **1.06%** | **99.85%** |

### 消融实验要点
- **CEPL vs 单阶段**：CEPL 降低 HTER 1.19%，提升 AUC 0.68%——渐进式学习有效解决任务干扰
- **CoT 数据 vs 纯标签**：CoT 数据训练在 224 分辨率下降低 HTER 5.79%——低分辨率下收益更大
- **RL vs 纯 SFT caption 模型**：RL 将 HTER 从 8.00% 降至 6.87%，证明 RL 不仅提升准确率还提升语义质量
- **零样本 vs CoT 微调**：MiniCPMV 零样本 17.91% HTER → 微调后 6.30%，降低 11.61 个点

## 亮点
- **开创性数据集**：108 万样本的 FAS VQA 数据集，是该领域首个，覆盖 14 种攻击类型
- **RL 增强标注**：双奖励 RL 将 caption 模型标注准确率从 88% 提升到 99.6%，提供了低成本高质量数据扩展路径
- **可解释性**：模型不仅给出判断还输出完整推理链，在安全敏感场景中至关重要
- **跨域泛化强**：对训练中未见的 3D 面具攻击仍有强泛化能力，AUC 提升 10%+
- **两阶段训练设计合理**：先让视觉编码器通过 CoT 学习细粒度特征，再联合训练分类，避免任务干扰

## 局限与展望
- 数据集源自 CelebA-Spoof 和 WFAS，人口统计学多样性取决于原始数据集
- 部分罕见攻击类型（如 adultdull 仅 165 样本）数据量极少
- 仅在 FAS 领域验证，CoT 构建方法是否可推广到其他安全检测任务有待验证

## 与相关工作的对比
- **vs I-FAS (AAAI 2025)**: I-FAS 也用 MLLM 做可解释 FAS 但仅提供简单描述；FaceCoT 提供六层级结构化推理链，信息密度更高
- **vs FLIP (CVPR 2023)**: FLIP 用 CLIP 做跨域 FAS；FaceCoT 用 MLLM + CoT 推理，泛化能力更强
- **vs LLaVA-CoT**: LLaVA-CoT 是通用 CoT 推理框架，FaceCoT 是专门为 FAS 设计的 CoT 结构

## 启发与关联
- FaceCoT 的数据构建流水线（GPT-4o + 人工精修 → RL 增强 caption 模型扩展）可以复用到其他安全检测任务的 VQA 数据集构建
- 两阶段训练策略（先视觉增强再联合训练）对其他需要细粒度视觉理解的 MLLM 任务有参考价值
- RL 提升标注质量的方法值得在更多自动数据标注场景中尝试

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 FAS VQA 数据集 + CoT 渐进式学习，将 MLLM 推理引入传统 CV 安全任务
- 实验充分度: ⭐⭐⭐⭐⭐ 11 个跨域基准 + 两种协议 + 多种消融 + 跨 backbone 验证 + 细粒度攻击类型分析
- 写作质量: ⭐⭐⭐⭐ 整体清晰但信息量极大，补充材料内容丰富
- 价值: ⭐⭐⭐⭐⭐ 数据集和方法论对 FAS 和更广泛的安全 AI 领域都有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[ECCV 2024\] Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing](../../ECCV2024/human_understanding/towards_unified_representation_of_invariant-specific_features_in_missing_modalit.md)
- [\[ECCV 2024\] TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](../../ECCV2024/human_understanding/tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)
- [\[CVPR 2025\] ControlFace: Harnessing Facial Parametric Control for Face Rigging](../../CVPR2025/human_understanding/controlface_harnessing_facial_parametric_control_for_face_rigging.md)
- [\[CVPR 2026\] IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations](idperturb_enhancing_variation_in_synthetic_face_generation_via_angular_perturbat.md)

</div>

<!-- RELATED:END -->
