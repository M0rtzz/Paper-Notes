---
title: >-
  [论文解读] LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics
description: >-
  [CVPR 2026][人体理解][情感建模] 提出 LaScA 框架，利用大语言模型生成确定性语义词典为手工制作的面部和声学特征提供语义先验，通过冻结的句子编码器生成语义嵌入并与原始特征融合，在 Aff-Wild2 和 SEWA 数据集上的情感变化预测中一致性地超越纯特征基线…
tags:
  - "CVPR 2026"
  - "人体理解"
  - "情感建模"
  - "语言模型"
  - "语义先验"
  - "Valence-Arousal"
  - "偏好学习"
---

# LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics

**会议**: CVPR 2026  
**arXiv**: [2604.07193](https://arxiv.org/abs/2604.07193)  
**代码**: 无  
**领域**: Affective Computing / 情感计算  
**关键词**: 情感建模, 语言模型, 语义先验, Valence-Arousal, 偏好学习

## 一句话总结

提出 LaScA 框架，利用大语言模型生成确定性语义词典为手工制作的面部和声学特征提供语义先验，通过冻结的句子编码器生成语义嵌入并与原始特征融合，在 Aff-Wild2 和 SEWA 数据集上的情感变化预测中一致性地超越纯特征基线，并在一致性、效率和可解释性上与端到端深度模型持平或更优。

## 研究背景与动机

在非受控环境（in-the-wild）中建模情感行为是情感计算的核心挑战。当前方法面临以下问题：

**端到端深度模型不透明**：CNN/RNN/Transformer 直接从视觉和音频流学习高维潜在表示，但信号提取和情感推理在不透明嵌入中纠缠，难以分析具体行为线索如何影响预测

**手工特征缺乏上下文抽象**：面部几何特征和声学描述符虽然紧凑高效且有领域知识基础，但无法捕获影响情感感知的高级语义关系——例如同一面部动作在不同上下文中的情感含义不同

**标注噪声大**：自然场景下的情感标注具有高度主观性和文化差异，直接预测绝对值不如预测变化方向可靠

LaScA 的核心洞察：手工特征是好的表示基础，但需要语言模型提供的上下文语义增强，而非被深度嵌入替代。

## 方法详解

### 整体框架

LaScA 的完整流水线：
1. 提取手工面部特征（58 个 blendshape 系数）和声学特征（15 个 MFCC）
2. 基于 Otsu 阈值的逐样本显著性估计，选择活跃特征
3. 将活跃特征对应的语义描述组装成模板文本
4. 冻结的句子编码器生成语义嵌入
5. 特征融合 + 偏好学习器预测情感变化方向

### 关键设计

1. **确定性语义词典（Affect-Aware Semantic Lexicon）**：

    - 功能：为每个手工特征（如 blendshape 系数、MFCC）生成固定的文本描述
    - 核心思路：使用 ChatGPT 5.2 以"情感计算研究者"身份一次性生成所有特征的情感含义描述，存储为固定映射 $\mathcal{L} = \{(f_i, \ell_i)\}_{i=1}^d$
    - 设计动机：词典生成一次即可，消除了推理时 LLM 的随机性和计算开销，确保可复现性
    - 消融验证：LLM 生成的词典比仅使用特征名称的词典在 arousal 上提升约 2%

2. **逐样本显著性估计（Otsu 阈值）**：

    - 功能：对每个时间片的归一化特征值排序，用 Otsu 方法二分为显著/非显著
    - 核心思路：最大化类间方差的无监督分割，得到二值掩码 $\mathbf{m}_t \in \{0,1\}^d$
    - 设计动机：在强烈的个体差异下自适应选择主要行为线索

3. **语义编码与融合**：

    - 功能：将活跃特征的描述插入结构化模板，用冻结的句子 Transformer 编码为语义嵌入 $\mathbf{s}_t$
    - 核心思路：语义嵌入捕获活跃行为线索之间的上下文关系
    - 融合方式：简单拼接 $\mathbf{z}_t = [\mathbf{x}_t \| \mathbf{s}_t]$
    - 评估了 5 种句子编码器：MPNet、QAMPNet、DistilRoBERTa、MiniLM、DistilBERT

4. **偏好学习器（Preference Learner）**：

    - 功能：预测连续时间窗口之间情感是增加还是减少
    - 核心思路：构造偏好对 $(x_t, x_{t+1})$，当相对变化超过阈值 τ 时才保留，使用嵌入差 $\Delta\mathbf{z}$ 通过两层 MLP + sigmoid 预测方向
    - 设计动机：相对预测比绝对预测更鲁棒，可以缓解标注噪声

### 损失函数 / 训练策略

- 二元交叉熵损失
- Adam 优化器，最多 25 次迭代
- L2 正则化 α=1，早停 3 次无改善
- 训练窗口：3s 和 5s 两种
- 相对阈值：10% 和 20% 两种
- 15 折交叉验证（SEWA）/ 15 个随机种子（Aff-Wild2）
- 可训练参数仅 129-230K（MLP head），极其轻量

## 实验关键数据

### 主实验（Aff-Wild2 上与 SOTA 对比，5s/20% 配置）

| 模态 | 方法 | Arousal | Valence |
|------|------|---------|---------|
| Visual | VGGFace2 | 0.71 | 0.72 |
| Visual | SwinFace | 0.74 | 0.73 |
| Visual | MAE-Face | 0.72 | 0.71 |
| Visual | **LaScA** | **0.74** | **0.74** |
| Audio | Wav2Vec2 | 0.71 | 0.60 |
| Audio | MAE-Audio | 0.69 | 0.60 |
| Audio | **LaScA** | **0.72** | 0.58 |
| Multimodal | HiCMAE | 0.75 | 0.63 |
| Multimodal | MMA-DFER | 0.75 | 0.63 |
| Multimodal | **LaScA** | 0.74 | 0.61 |

### SEWA DB 上视觉模态最佳结果（5s/20%）

| 方法 | Arousal | Valence |
|------|---------|---------|
| SwinFace | 0.71 | 0.82 |
| MAE-Face | 0.70 | 0.81 |
| **LaScA** | 0.70 | **0.83** |

### 消融实验

| 配置 | Arousal (5s/20%) | Valence (5s/20%) | 说明 |
|------|-----------------|-----------------|------|
| 纯特征 (Features) | 0.55 | 0.52 | 基线最弱 |
| 纯文本 (Sentence Transformer) | 0.60 | 0.67 | 语义本身有价值 |
| 融合 (F) | 0.74 | 0.74 | 融合最优 |
| Feature-based 词典 | 0.74 | 0.61 | 特征名做描述 |
| LLM-based 词典 | 0.74 | 0.63 | LLM 描述更优 |

### 关键发现

1. **融合一致有效**：无论视觉/音频/多模态，融合版本始终优于纯特征和纯文本
2. **SEWA 上增益更大**：会话式交互场景下，语义上下文的补偿作用更明显（纯特征几乎是随机水平 50%）
3. **5s 窗口 > 3s 窗口**：更长的时间上下文对情感建模有益
4. **Arousal 增益 > Valence 增益**：语义先验对情感强度建模帮助更大
5. **编码器选择影响不大**：融合后不同句子编码器的性能差异很小，说明融合策略比编码器选择更重要

## 亮点与洞察

1. **"不要替代手工特征，增强它们"的范式**：与端到端黑盒模型形成鲜明对比，保持了可解释性
2. **确定性词典的优雅设计**：LLM 仅在离线阶段使用一次，推理时完全确定、高效、可复现
3. **极致轻量**：可训练参数仅 129-230K，推理 80-140ms/样本（在笔记本 GPU 上），适合实时部署
4. **跨数据集一致性**：在实验室级（SEWA）和野外级（Aff-Wild2）数据集上均有效
5. **Otsu 阈值选特征**：简单但有效的无监督显著性估计，避免了学习门控的额外复杂度

## 局限与展望

1. 所有编码器完全冻结，选择性微调可能进一步提升性能
2. SEWA 实验受限于预提取声学特征（无原始音频访问），未能评估端到端音频模型
3. 词典固定，跨文化/多语言场景需要适应性词典
4. 仅建模相邻时间窗口的局部变化，缺乏长程时序建模（如序列编码器、时间注意力）
5. 未扩展到离散情感类别或更高维情感表示

## 相关工作与启发

- **情感动态预测范式**：用相对方向（增/减）替代绝对值预测，缓解标注噪声
- **LLM 作为语义先验**：与直接将 LLM 集成到端到端架构不同，LaScA 将 LLM 的知识蒸馏为固定词典
- **Sentence Transformers**：通用句子编码为下游任务提供即插即用的语义表示
- 启发："小模型 + 大模型词典"是一种值得推广的混合范式——利用 LLM 的知识但不引入其计算开销

## 评分

- 新颖性: ⭐⭐⭐⭐ — LLM 语义词典 + 手工特征融合用于情感建模是新颖的
- 实验充分度: ⭐⭐⭐⭐ — 跨数据集、跨模态、跨编码器的评估很全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，但表格过多影响阅读流畅性
- 价值: ⭐⭐⭐⭐ — 为可解释情感计算提供了高效实用的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation](sketch2colab_sketch-conditioned_multi-human_animation_via_controllable_flow_dist.md)
- [\[CVPR 2026\] Sign Language Recognition in the Age of LLMs](sign_language_recognition_llms.md)
- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)
- [\[ECCV 2024\] Facial Affective Behavior Analysis with Instruction Tuning](../../ECCV2024/human_understanding/facial_affective_behavior_analysis_with_instruction_tuning.md)
- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](molingo_motion-language_alignment_for_text-to-motion_generation.md)

</div>

<!-- RELATED:END -->
