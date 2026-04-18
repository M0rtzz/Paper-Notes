---
title: >-
  [论文解读] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization
description: >-
  [CVPR 2026][图像生成][视频配音] 提出DiFlowDubber，基于**离散流匹配(DFM)**的自动视频配音框架，通过两阶段训练（零样本TTS预训练→视频配音适配）将大规模TTS知识迁移到视频驱动配音，设计FaPro模块捕获面部表情-韵律映射、Synchronizer模块实现精准唇音同步。
tags:
  - CVPR 2026
  - 图像生成
  - 视频配音
  - 离散流匹配
  - 跨模态对齐
  - 语音合成
  - 唇音同步
---

# DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization

**会议**: CVPR 2026  
**arXiv**: [2603.14267](https://arxiv.org/abs/2603.14267)  
**代码**: [Demo](https://nngocson2002.github.io/projects/diflowdubber) (有)  
**领域**: Image Generation / 多模态语音生成  
**关键词**: 视频配音, 离散流匹配, 跨模态对齐, 语音合成, 唇音同步  

## 一句话总结

提出DiFlowDubber，基于**离散流匹配(DFM)**的自动视频配音框架，通过两阶段训练（零样本TTS预训练→视频配音适配）将大规模TTS知识迁移到视频驱动配音，设计FaPro模块捕获面部表情-韵律映射、Synchronizer模块实现精准唇音同步。

## 研究背景与动机

**领域现状**：视频配音(V2C)在影视制作、多媒体创作和辅助语音技术中有广泛应用。需要在保持说话人音色的同时，生成与唇型同步、承载视觉情感的自然语音。

**现有痛点**：
   - 直接在有限配音数据上训练的方法难以产生富有表现力的韵律和音质
   - 两阶段方法（先TTS预训练再适配）虽能利用大规模语料，但难以同时保证韵律表达和唇音同步

**核心矛盾**：Speaker2Dubber仅适配了音素编码器，未充分利用TTS的韵律和声学建模能力；ProDubber虽引入韵律增强，但依赖时长预测器估计唇型运动，**不受视频实际长度约束**导致同步能力差（LSE分数低）。

**本文目标**：设计一个框架充分利用大规模TTS预训练的韵律、内容和声学建模能力，并适配到视频配音中同时保证发音准确、自然度和精准唇音同步。

**切入角度**：使用FACodec将语音分解为韵律、内容、声学三类离散token，分别建模不同属性——韵律和声学用生成式DFM建模，内容用确定性架构建模。

**核心idea**：两阶段训练管线 + 离散流匹配骨干 + 面部表情→韵律映射 + 双重对齐同步器。

## 方法详解

### 整体框架

两阶段管线（图2）：
- **阶段1（零样本TTS预训练）**：在LibriTTS上训练零样本TTS系统，学习语音的韵律/内容/声学表示
    - **内容建模**：确定性架构直接预测内容token
    - **韵律-声学建模**：离散流匹配(DFPA)模块生成表现力丰富的韵律和声学token
- **阶段2（视频配音适配）**：将预训练TTS适配到配音任务
    - FaPro模块从面部表情提取全局韵律先验
    - Synchronizer实现文本-视频-语音三模态对齐
    - CCTA确保内容一致性
    - DFPA适配为视觉条件生成

### 关键设计

#### 1. 离散流匹配韵律-声学(DFPA)模块

- **功能**：联合建模韵律和声学token的分布
- **核心思路**：基于DiT架构的去噪器，输入当前去噪目标 $\mathbf{x}_t$（韵律+声学token拼接），条件包括内容潜在表示 $\tilde{\mathbf{h}}_c$、说话人嵌入等。优化目标为：
  $$\mathcal{L}_{\text{DFM}} = -\sum_{i \in \mathcal{T}} \mathbb{E}_{t \sim \mathcal{U}[0,1]} [\log p_{1|t}(\mathbf{x}_1^i | \mathbf{x}_t, \mathbf{c}; \theta)]$$
- **设计动机**：FACodec将语音属性解耦为可独立控制的token，DFM比自回归模型能更好捕获表现力丰富的韵律变化

#### 2. FaPro（面部表情→韵律映射）

- **功能**：从面部视频帧提取全局韵律先验 $\tilde{\mathbf{z}}_p \in \mathbb{R}^{m \times L \times D}$
- **核心思路**：面部特征 $\mathbf{v}_{\text{face}}$ → 上采样对齐到语音长度 → ConvNeXt V2增强时序建模 → $m$ 层FFT迭代提取各RVQ码本的韵律表示
- **设计动机**：面部表情与语音韵律有强相关性，全局韵律（情感、语速、语调）主要由面部表情决定

#### 3. Synchronizer（双重对齐同步器）

- **功能**：桥接文本、视频、语音三个模态间的gap
- **核心思路**：
    - **视频-文本对齐**：唇部特征作为query、音素特征作为key/value的cross-attention，利用MFA对齐矩阵 $\mathcal{M}_{VT}$ 施加对比损失 $\mathcal{L}_{VT}$
    - **时长正则化**：通过MAS将注意力权重转为音素-帧时长，按时长复制音素嵌入后上采样到语音长度
    - **语音-文本对齐**：对上采样后的表示再施加一层对比对齐 $\mathcal{L}_{ST}$，修正上采样引入的微小错位
    - ConvNeXt V2细化得到最终对齐表示 $\mathbf{h}_{\text{sync}} \in \mathbb{R}^{L \times D}$
- **设计动机**：ProDubber的时长预测器不受视频长度约束，双重对齐机制显式强制唇型与语音的时序一致性

#### 4. CCTA（内容一致性时序适配）

- **功能**：从TTS域迁移语义内容知识，保持语言一致性
- **核心思路**：用预训练TTS权重初始化，冻结投影层和内容头，用Synchronizer替换TTS中的时长预测。添加蒸馏损失保持师生特征一致性：
  $$\mathcal{L}_{\text{distill}} = \frac{1}{B} \sum_{i=1}^{B} [1 - \cos(\phi(\mathbf{z}_t^{(i)}), \phi(\mathbf{z}_c^{(i)}))]$$

### 损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{align}} + \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_{\text{CTC}} + \lambda_3 \mathcal{L}_{\text{distill}} + \lambda_4 \mathcal{L}_{\text{DFM}}$$

其中 $\mathcal{L}_{\text{align}} = \lambda_5 \mathcal{L}_{VT} + \lambda_6 \mathcal{L}_{ST}$。

## 实验关键数据

### 主实验：Chem数据集 (Setting 1.0)

| 方法 | LSE-C↑ | LSE-D↓ | WER↓ | SECS↑ | UTMOS↑ |
|------|--------|--------|------|-------|--------|
| HPMDubbing (CVPR'23) | 7.85 | 7.19 | 16.05 | 85.09 | 2.16 |
| EmoDubber (CVPR'25) | 8.11 | 6.92 | 11.72 | 90.62 | 3.82 |
| ProDubber (CVPR'25) | 2.58 | 12.54 | **9.45** | 72.13 | 3.85 |
| **DiFlowDubber** | **8.31** | **6.73** | 9.65 | 84.59 | **4.02** |

### GRID数据集 (Setting 1.0)

| 方法 | LSE-C↑ | LSE-D↓ | WER↓ | UTMOS↑ |
|------|--------|--------|------|--------|
| EmoDubber (CVPR'25) | 7.12 | 6.82 | 18.53 | 3.83 |
| **DiFlowDubber** | **7.32** | **6.73** | **16.79** | **3.95** |

### 消融实验

| 配置 | LSE-C↑ | WER↓ | UTMOS↑ |
|------|--------|------|--------|
| 完整模型 | **8.31** | **9.65** | **4.02** |
| 无TTS预训练 | 8.17 | 12.04 | 3.53 |
| 无 $\mathcal{L}_{VT}+\mathcal{L}_{ST}$ | 8.26 | 17.15 | 3.90 |
| 无 $\mathcal{L}_{VT}$ | 8.36 | 16.57 | 3.87 |
| 无 $\mathcal{L}_{ST}$ | 8.31 | 12.60 | 3.93 |
| 无 $\mathcal{L}_{\text{distill}}$ | 8.33 | 12.62 | 3.97 |

### 关键发现

1. **同步性**：DiFlowDubber在LSE-C/LSE-D上显著领先，证明Synchronizer有效。ProDubber虽WER低，但同步性极差（LSE-C仅2.58），更像TTS而非配音
2. **TTS预训练关键**：去掉预训练后WER从9.65→12.04，UTMOS从4.02→3.53，证明大规模TTS知识迁移的必要性
3. **双重对齐互补**：$\mathcal{L}_{VT}$ 主要帮助同步，$\mathcal{L}_{ST}$ 主要帮助发音准确性，二者缺一不可
4. MOS-S主观评测中超越所有最新基线，证明真实感知质量最优

## 亮点与洞察

1. **FACodec分解策略精妙**：将语音解耦为韵律/内容/声学三类token，使得不同属性可以用不同建模策略（确定性 vs 生成式），且便于视觉条件注入
2. **Synchronizer的双重保险**：先做视频-文本对齐，上采样后再做语音-文本对齐修正，比单层对齐更鲁棒
3. **首次将离散流匹配(DFM)应用于视频配音**，证明了DFM在跨模态语音生成中的有效性

## 局限与展望

1. **SECS分数不突出**：说话人相似度指标不如一些基线，虽然作者归因于评估偏差（基线模型与SECS计算共享说话人编码器），但仍是改进空间
2. **依赖MFA外部工具**：需要Montreal Forced Aligner提供音素-帧对齐信息作为训练监督
3. **仅在Chem和GRID两个数据集评测**：均为受控环境数据，在野外(in-the-wild)场景的泛化能力未知
4. 生成速度未详细对比——128 NFE的离散流匹配推理可能较慢

## 相关工作与启发

- **FACodec语音分解**：来自NaturalSpeech 3，将语音属性因子化的思路值得在其他跨模态任务中借鉴
- **对比对齐损失**：视频-文本和语音-文本的双重对比对齐，可推广到任何需要多模态时序对齐的场景
- **两阶段预训练-适配范式**：先在大规模数据上学好单模态能力，再通过适配引入新模态条件——这是跨模态生成的有效通用策略

## 评分

⭐⭐⭐⭐ — 系统性强、模块设计合理、同步性显著提升，但评测规模有限且文字较冗长
