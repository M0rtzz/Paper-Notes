---
title: >-
  [论文解读] Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment
description: >-
  [AAAI 2026][多模态][弱监督视频异常检测] 本文提出DSANet，通过自引导正常模式建模（SG-NM，粗粒度）和解耦对比语义对齐（DCSA，细粒度）从两个层面增强弱监督视频异常检测中正常与异常特征的可区分性，在XD-Violence上AP达86.95%（+1.14%），在UCF-Crime细粒度mAP达13.01%（+3.39%），均为SOTA。
tags:
  - AAAI 2026
  - 多模态
  - 弱监督视频异常检测
  - 语义解耦
  - 正常模式建模
  - 对比对齐
  - CLIP
---

# Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment

**会议**: AAAI 2026  
**arXiv**: [2511.10334](https://arxiv.org/abs/2511.10334)  
**代码**: [https://github.com/lessiYin/DSANet](https://github.com/lessiYin/DSANet)  
**领域**: 多模态VLM  
**关键词**: 弱监督视频异常检测, 语义解耦, 正常模式建模, 对比对齐, CLIP

## 一句话总结

本文提出DSANet，通过自引导正常模式建模（SG-NM，粗粒度）和解耦对比语义对齐（DCSA，细粒度）从两个层面增强弱监督视频异常检测中正常与异常特征的可区分性，在XD-Violence上AP达86.95%（+1.14%），在UCF-Crime细粒度mAP达13.01%（+3.39%），均为SOTA。

## 研究背景与动机

### 领域现状
弱监督视频异常检测（WS-VAD）旨在仅使用视频级标签（正常/异常，无帧级标注）来定位视频中的异常片段。主流方法基于多示例学习（MIL）框架：先用预训练backbone（I3D或CLIP）提取特征，再通过二分类器产生帧级异常分数。近期方法（VadCLIP, PEMIL, ITC等）结合CLIP的视觉-语言预训练能力，通过文本提示识别异常类别。

### 现有痛点
现有WS-VAD方法虽然取得了不错的检测性能，但仍存在两个根本性缺陷：

**粗粒度层面：正常模式理解不完整**
   - MIL的判别性本质导致模型专注于发现最显著的异常片段
   - 忽略了视频中丰富多样的正常模式的显式建模
   - 无法构建鲁棒的正常表示→正常与异常边界模糊→误报率高
   - 例如，一个复杂但正常的场景（如拥挤的超市）可能被误判为异常

**细粒度层面：类别混淆严重**
   - 不同异常类别可能外观相似（如"抢劫"和"偷窃"都涉及物品被拿走）
   - 异常与正常上下文之间的背景模式可能相似
   - 在缺乏帧级监督的情况下，模型常将共出现的背景模式与真实异常混淆
   - 不同异常类别的特征在嵌入空间中纠缠，影响类别可分性

### 核心idea
从两个层面"学会区分"：
- 粗粒度：通过生成式的正常模式重建来补充MIL判别学习的盲区
- 细粒度：通过解耦事件/背景特征并分别与对应语义对齐来消除类别混淆

## 方法详解

### 整体框架
DSANet包含三个协同分支：
1. **异常检测分支**：基于MIL框架产生帧级二分类异常分数（基础）
2. **自引导正常模式建模分支（SG-NM）**：挖掘视频特有的正常模式并引导特征重建（粗粒度增强）
3. **异常分类分支**：将视频特征与文本类别嵌入对齐进行细粒度分类，融入DCSA机制（细粒度增强）

### 关键设计

#### 1. **自引导正常模式建模（SG-NM）**

核心思路：即使在异常视频中，局部区域仍具有固有的正常性（如异常事件发生时的正常背景），SG-NM直接从输入视频中动态挖掘正常原型，无需外部记忆库。

具体步骤：
- **正常帧选择**：使用检测分支的异常分数 $S_{det}$ 选择分数最低的M帧（M=80%视频长度），构成候选正常特征集 $F_n \in \mathbb{R}^{M \times D}$
- **动态正常模式（DNP）提取**：K=16个可学习query通过单层cross-attention从 $F_n$ 中提取K个蒸馏的正常原型 $P \in \mathbb{R}^{K \times D}$
- **正常集中损失**：确保DNP纯粹代表正常特征

$$\mathcal{L}_{compact} = \frac{1}{M} \sum_{i=1}^{M} \min_{j \in \{1,...,K\}} d(F_n(i), P(j))$$

- **特征重建**：8层cross-attention解码器用DNP作为key/value重建视频特征。**关键设计**：第一层解码器去除残差连接，确保重建完全依赖正常模式（防止异常泄漏）
- **一致性损失**：将重建异常分数 $S_{rec}$ 与检测分数 $S_{det}$ 对齐

$$\mathcal{L}_{consist} = \frac{1}{N} \sum_{i=1}^{N} (S_{det}(i) - S_{rec}(i))^2$$

**设计动机**：MIL只学"什么是异常"（判别式），SG-NM补充学"什么是正常"（生成式）。两者互为补充：MIL提供异常分数帮助SG-NM选择正常帧，SG-NM的重建错误又校准MIL的检测边界。

#### 2. **解耦对比语义对齐（DCSA）**

核心思路：将视频特征显式解耦为事件成分和背景成分，分别与对应的文本语义对齐。

**视觉特征解耦**：利用检测分支的异常分数做软解耦（不是硬切割）

$$F_{event} = w_{event}^\top F_{video}, \quad F_{bkg} = w_{bkg}^\top F_{video}$$

其中 $w_{event} = \text{Softmax}(S_{det})$，$w_{bkg} = 1 - w_{event}$

**文本特征准备**：CLIP文本编码器生成C个类别的文本嵌入 $T_{text} = \{t_0, t_1, ..., t_{C-1}\}$，其中 $t_0$ 代表"normal"类

**分离损失**：推动normal嵌入远离所有异常类嵌入

$$\mathcal{L}_{sep} = \sum_{a=1}^{C-1} \left| \frac{t_0^\top t_a}{\|t_0\| \|t_a\|} \right|$$

**双重对比对齐损失** $\mathcal{L}_{dcsa} = \mathcal{L}_{event} + \mathcal{L}_{bkg}$：
- 事件对齐：$F_{event}$ 对齐真实类别 $t_c$（异常视频→对应异常类，正常视频→normal类）
- 背景对齐：$F_{bkg}$ **始终**对齐normal类 $t_0$（无论视频是否异常，背景都应偏向正常）

**设计动机**：这种解耦避免了异常相关特征与背景模式的纠缠。背景始终对齐normal起到正则化作用，防止模型将共出现的背景模式误认为异常。

#### 3. **轻量文本适配器**

在CLIP文本编码器的前L层Transformer blocks中插入轻量适配器：

$$x_{out} = (1 - \omega_t) \cdot x + \omega_t \cdot \text{Norm}(x_{adapt})$$

保持CLIP通用知识的同时实现领域适配（比prompt学习更有效）。

### 损失函数 / 训练策略

$$\mathcal{L}_{total} = \mathcal{L}_{det} + \lambda \mathcal{L}_{align} + \mathcal{L}_{consist} + \mathcal{L}_{compact} + \mathcal{L}_{dcsa} + \mathcal{L}_{sep}$$

训练细节：
- 视觉特征：冻结CLIP (ViT-B/16) 提取
- 优化器：AdamW，batch 64/96
- 训练10 epochs，单卡4090
- SG-NM分支**仅训练时使用**，推理时不增加计算开销

推理策略：层次信念调制（Hierarchical Belief Modulation）——用 $S_{det}$ 作为时间先验，$S_{align}$ 分配类别概率，通过温度比 $\beta$ 校准。

## 实验关键数据

### 主实验

**粗粒度检测**：

| 方法 | XD-Violence AP(%) | UCF-Crime AUC(%) |
|------|-------------------|------------------|
| VadCLIP (CLIP) | 84.51 | 88.02 |
| ITC (CLIP) | 85.45 | 89.04 |
| ReFLIP (CLIP) | 85.81 | 88.57 |
| **DSANet** | **86.95** | **89.44** |

**细粒度检测（mAP@IoU，XD-Violence）**：

| 方法 | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | AVG |
|------|------|------|------|------|------|-----|
| VadCLIP | 37.03 | 30.84 | 23.38 | 17.90 | 14.31 | 24.70 |
| ReFLIP | 39.24 | 33.45 | 27.71 | 20.86 | 17.22 | 27.36 |
| **DSANet** | **40.93** | **34.63** | **28.21** | **22.70** | **17.89** | **28.87** |

**细粒度检测（UCF-Crime）**：

| 方法 | @0.1 | AVG |
|------|------|-----|
| ReFLIP | 14.23 | 9.62 |
| **DSANet** | **21.39** | **13.01** |

UCF-Crime上细粒度提升尤为显著（AVG +3.39%），证明DCSA在困难场景下的类别区分能力。

### 消融实验

| 配置 | AP(%) | AVG(%) | 说明 |
|------|-------|--------|------|
| Baseline (VadCLIP) | 84.51 | 24.70 | — |
| + Adapter | 85.00 | 28.15 | 文本适配有效 |
| + Adapter + SG-NM | 85.94 | 28.39 | 正常建模改善检测 |
| + Adapter + DCSA | 85.67 | 28.25 | 语义解耦改善分类 |
| **+ All (DSANet)** | **86.95** | **28.87** | 组件协同 |

**文本编码器调整方式对比**：

| 方式 | AP(%) | AVG(%) |
|------|-------|--------|
| 冻结 | 81.57 | 27.60 |
| 手动prompt | 81.05 | 28.05 |
| 可学习prompt (CoOp) | 82.88 | 28.26 |
| **Adapter (本文)** | **86.95** | **28.87** |

### 关键发现

1. **DNP的质量验证**：正常帧到DNP的最小余弦距离分布（均值0.35）与异常帧距离分布（均值0.69）有明显差异，证明DNP确实形成了紧凑的正常表示
2. **DCSA的效果验证**：背景原型对齐normal类的准确率达99.63%（VadCLIP仅87.63%），混淆矩阵的对角线主导性明显更强
3. **t-SNE可视化**：DSANet的特征空间中不同异常类别的类间边界更清晰、类内聚集更紧密
4. 时序可视化中DSANet的预测与ground truth更吻合，VadCLIP倾向于只捕获最显著片段导致时间边界不准

## 亮点与洞察

1. **互补的双重学习范式**：MIL判别式学"什么是异常" + SG-NM生成式学"什么是正常"——这个互补思路很有启发性，可以推广到其他检测任务
2. **解耦对齐的设计**：事件→对应类别、背景→永远normal，这种asymmetric alignment约束非常巧妙，比简单的对齐损失多了结构化先验
3. **SG-NM的自包含设计**：无需外部记忆库，直接从当前视频中动态挖掘正常模式，scalable且data-efficient
4. **重建解码器第一层去残差连接**的设计细节虽小但关键——防止异常信息通过残差连接泄漏到重建结果中
5. 在单卡4090上即可训练，计算预算友好

## 局限与展望

- SG-NM仅在训练时使用，推理时不使用重建分支——是否可以在推理时也利用重建误差作为辅助判断？
- 候选正常帧选择依赖检测分支的初始异常分数——early training阶段分数不准可能影响DNP质量
- 文本类别标签的质量影响DCSA效果——如何自动生成更好的类别描述？
- 仅验证了CLIP ViT-B/16，是否在更大模型（ViT-L）上也有效？
- 异常类别数量固定为训练集中的类别，开放世界场景下的泛化有待验证

## 相关工作与启发

- VadCLIP作为baseline的地位——DSANet在其基础上增加了正常建模和语义解耦两个维度
- 从CLIP的视觉-语言对齐到任务特定的语义解耦，展示了预训练模型适配下游任务的一种路径
- 事件/背景解耦与视频理解中的前景/背景分离有概念上的联系，可能在其他视频理解任务中也有用

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)
- [No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](../../CVPR2026/multimodal_vlm/no_need_for_real_anomaly_mllm_empowered_zero-shot_video_anomaly_detection.md)
- [Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](../../ICLR2026/multimodal_vlm/bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)
- [Harnessing Vision-Language Models for Time Series Anomaly Detection](harnessing_vision-language_models_for_time_series_anomaly_detection.md)

<!-- RELATED:END -->
