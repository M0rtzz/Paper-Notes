---
title: >-
  [论文解读] EvSign: Sign Language Recognition and Translation with Streaming Events
description: >-
  [ECCV 2024][LLM评测][手语识别] 首次构建面向连续手语识别（CSLR）和手语翻译（SLT）任务的事件相机基准数据集 EvSign，并提出基于稀疏Transformer的高效框架，在仅0.34% FLOPs和44.2%参数量下达到与SOTA RGB方法可比或更优的性能。
tags:
  - "ECCV 2024"
  - "LLM评测"
  - "手语识别"
  - "手语翻译"
  - "事件相机"
  - "稀疏卷积"
  - "时序建模"
---

# EvSign: Sign Language Recognition and Translation with Streaming Events

**会议**: ECCV 2024  
**arXiv**: [2407.12593](https://arxiv.org/abs/2407.12593)  
**代码**: [项目页](https://zhang-pengyu.github.io/EVSign)  
**领域**: LLM评测  
**关键词**: 手语识别, 手语翻译, 事件相机, 稀疏卷积, 时序建模

## 一句话总结

首次构建面向连续手语识别（CSLR）和手语翻译（SLT）任务的事件相机基准数据集 EvSign，并提出基于稀疏Transformer的高效框架，在仅0.34% FLOPs和44.2%参数量下达到与SOTA RGB方法可比或更优的性能。

## 研究背景与动机

手语是听障群体的重要沟通工具，基于视频的手语识别和翻译是重要的研究方向。现有方法面临以下挑战：

**RGB传感器的固有缺陷**：快速手部运动导致运动模糊、衣物纹理造成背景干扰、极端光照条件下信息退化

**事件相机的天然优势**：异步捕获亮度变化，时间分辨率极高（1MHz vs RGB 120Hz），高动态范围，低延迟——天然适合捕获动态手部运动

**现有事件手语数据集的不足**：
   - 仅支持孤立手语识别（ISLR），不支持连续识别和翻译
   - 词汇量极小（SL-Animals-DVS仅19词，EvASL仅56词）
   - 传感器分辨率低（128×128）

**现有方法未充分利用事件特性**：直接用为RGB设计的网络（AlexNet、ResNet）处理事件数据，忽略了事件数据的稀疏性

## 方法详解

### 整体框架

方法包含两部分贡献：

**A. EvSign 数据集**：大规模中文手语事件基准
**B. 高效Transformer框架**：专门针对事件数据特性设计的SLR+SLT模型

整体流程：事件流 → 体素网格表示 → 稀疏卷积骨干 → 局部Token融合 → Gloss感知时序聚合 → 识别头/翻译头

### 关键设计

#### 1. EvSign 数据集

**采集设备**：iniVation DVXplorer-S-Duo 双目相机（事件流640×480，RGB 480×320@25FPS）

**语料来源**：日常生活场景（购物、教育、医疗、旅游、社交），手语词汇源自中国国家手语词典和CSL-Daily

**数据规模**：
- 6773个事件流视频（训练5570/开发553/测试650）
- 1387个手语词汇（gloss），1947个中文词汇
- 9位专业聋人志愿者
- 总时长约8.5小时

**标注流程**：两步式标注——先由标注者在RGB视频中识别手语词汇，再由作者核对确保每个手语对应唯一gloss标注

相比现有数据集的优势：首个支持CSLR和SLT的事件数据集，词汇量远超同类（vs SL-Animals 19词, EvASL 56词），分辨率更高（640×480 vs 128×128）。

#### 2. 稀疏骨干网络 (Sparse Backbone, SConv)

事件数据天然稀疏（仅编码运动区域），因此采用 ResNet18 架构的稀疏卷积网络处理体素网格表示：
- 充分利用数据稀疏性大幅降低计算量
- 比普通卷积更好地保持特征级稀疏性，生成更锐利的边界
- 输出视觉Token集合 $\mathbf{O}^v \in \mathbb{R}^{P \times C}$

#### 3. 局部Token融合 (Local Token Fusion, LTF)

对长视频时序建模前，先聚合局部运动信息并降低Token数量：
- 两层结构，每层包含窗口多头自注意力（W-MSA）+ 最大池化
- 窗口大小 $I$，下采样比 $\gamma=4$
- 公式：$\mathbf{O}^f = \text{MaxPool}(\text{W-MSA}(\tilde{\mathbf{O}}^v) + \tilde{\mathbf{O}}^v)$
- 产生融合Token $\mathbf{O}^f \in \mathbb{R}^{L \times C}$，$L = P/\gamma$

#### 4. Gloss感知时序聚合 (GATA)

核心时序建模模块，将时序信息解耦为**gloss内**和**gloss间**两级：

**Gloss感知掩码注意力（GAMA）— gloss内聚合**：
- 使用交叉注意力从视觉Token $\mathbf{O}^v$ 聚合信息到融合Token $\mathbf{O}^f$
- 关键创新：引入gloss感知掩码 $\mathbf{M} = \mathcal{N}(\rho) \odot \mathcal{N}(\delta)$
    - $\rho$：特征空间相似度——同类Token具有高相关性表示
    - $\delta$：时间空间约束——RBF核衡量伪时间戳距离，避免不同位置同类gloss间的错误聚合
- 公式：$\text{GAMA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \text{softmax}(\frac{\mathbf{QK}^T}{\sqrt{d}} \odot \mathbf{M})\mathbf{V}$

**gloss间时序聚合（IGTA）**：
- 标准多头自注意力建模全局运动连贯性
- 学习不同gloss之间的时序依赖

#### 5. 任务头

- **识别头（RH）**：全连接层+softmax，输出gloss序列概率，CTC loss监督
- **翻译头（TH）**：自回归Transformer解码器，将gloss-aware tokens翻译为口语句子，交叉熵损失监督

### 损失函数 / 训练策略

**SLR损失**：$\mathcal{L}_{SLR} = \lambda_{inter}\mathcal{L}_{inter} + \lambda_{final}\mathcal{L}_{final}$（中间+最终两个CTC损失，权重均为1）

**SLT损失**：$\mathcal{L}_{SLT} = \mathcal{L}_{SLR} + \lambda_{ce}\mathcal{L}_{ce}$（额外加翻译的交叉熵损失）

训练细节：Adam优化器，余弦退火，初始学习率3e-5，batch size 2，200 epochs，单张RTX 3090。

## 实验关键数据

### 主实验 — 手语识别（WER↓）

| 方法 | 模态 | PHOENIX14T Dev | PHOENIX14T Test | EvSign Dev | EvSign Test | FLOPs | Params |
|------|------|---------------|----------------|------------|------------|-------|--------|
| VAC | RGB | 20.17 | 21.60 | 32.08 | 30.43 | 228.87G | 31.64M |
| CorrNet | RGB | 18.90 | 20.50 | 32.37 | 32.04 | 234.59G | 32.04M |
| CorrNet | EV | 24.57 | 24.55 | 29.98 | 29.95 | 244.63G | 32.05M |
| **Ours** | **EV** | 23.89 | 24.03 | **29.19** | **28.69** | **0.84G** | **14.19M** |

在EvSign上，事件方法全面优于RGB方法；本方法以0.34% FLOPs实现最低WER。

### SLT结果（EvSign数据集，部分指标）

| 方法 | 模态 | Dev ROUGE-L↑ | Test ROUGE-L↑ | Dev BLEU-4↑ | Test BLEU-4↑ |
|------|------|-------------|-------------|------------|------------|
| SLT | RGB | - | - | - | - |
| CorrNet+TH | EV | - | - | - | - |
| **Ours** | **EV** | 最优 | 最优 | 竞争性 | 最优 |

在合成PHOENIX14T上，事件方法的SLT结果ROUGE比SLT(RGB)提升1.06%/0.89%（dev/test）。

### 消融实验

通过去除各模块验证设计有效性：
- 稀疏骨干是效率提升的关键（计算量降低两个数量级）
- LTF降低Token数量的同时保留局部运动信息
- GAMA中的特征相似度掩码和时间距离约束都对性能有贡献
- gloss间自注意力补充全局时序建模

### 关键发现

1. **事件相机在真实数据上全面优于RGB**：所有方法在EvSign上用事件流的WER都低于RGB（如CorrNet: 29.95 vs 32.04）
2. **合成数据上事件优势不明显**：因为PHOENIX14T视频质量差（模糊、低帧率），合成事件质量受限
3. **极致效率**：本方法仅用0.84G FLOPs/视频（vs CorrNet 244.63G），效率提升290倍
4. **参数量节省**：14.19M参数（44.2% of CorrNet），适合边缘部署
5. **事件相机的隐私优势**：仅捕获运动信息，不记录静态面部特征，天然保护隐私

## 亮点与洞察

1. **首个CSLR+SLT事件基准**：填补了事件手语领域からCISLR到连续识别和翻译的空白
2. **稀疏性的双重利用**：事件数据天然稀疏 → 稀疏卷积处理 → 局部融合降维 → 极低计算量
3. **Gloss感知掩码的优雅设计**：同时考虑特征相似度（哪些Token属于同一gloss）和时间距离（避免远距离同类误聚合），比简单全局注意力更合理
4. **实用价值**：事件相机的高时间分辨率+低延迟+隐私保护，非常适合可穿戴手语翻译设备

## 局限与展望

- EvSign仅含中文手语，泛化到其他手语系统需要验证
- 9位手语者规模相对较小，个体差异覆盖有限
- 仍依赖gloss作为中间表示（Sign2Gloss2Text），未探索端到端gloss-free方案在事件数据上的潜力
- 体素网格表示丢失了事件的异步性，可考虑直接处理原始事件流的方法
- 合成事件质量受限于源RGB视频质量，需要更多真实高质量事件数据

## 相关工作与启发

- **CorrNet**：当前RGB手语识别SOTA，通过相关性网络构建判别性空间表示
- **VAC**：视觉对齐约束方法，本文将其翻译头扩展到事件模态
- **SL-Animals-DVS / EvASL**：先前事件手语数据集，但规模小、仅ISLR、低分辨率
- **PixelCNN/VQ-VAE 事件采样**：Shi et al. 设计事件采样策略，但仍用CNN处理，未利用稀疏性
- **SEN/TLP**：近期RGB手语识别方法，本文在事件域上全面超越

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个CSLR+SLT事件基准+针对性框架设计
- 技术深度: ⭐⭐⭐⭐ — 稀疏骨干+GATA模块的设计环环相扣
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实数据、SLR+SLT双任务、效率对比详尽
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，图表直观，但SLT实验细节略显不足

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition](../../ICCV2025/llm_evaluation/svtrv2_ctc_beats_encoder-decoder_models_in_scene_text_recognition.md)
- [\[ACL 2025\] SwiLTra-Bench: The Swiss Legal Translation Benchmark](../../ACL2025/llm_evaluation/swiltra-bench_the_swiss_legal_translation_benchmark.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](../../NeurIPS2025/llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[ICCV 2025\] StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition](../../ICCV2025/llm_evaluation/streammind_unlocking_full_frame_rate_streaming_video_dialogue_through_event-gate.md)
- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](../../ACL2026/llm_evaluation/beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)

</div>

<!-- RELATED:END -->
