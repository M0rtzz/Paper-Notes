---
title: >-
  [论文解读] Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach
description: >-
  [CVPR 2026 (ABAW Workshop)][人体理解][效价-唤醒估计] 首次将 VLM（Qwen3-VL-4B-Instruct）提取的情感行为描述嵌入作为独立第三模态，与 GRADA 人脸编码器和 WavLM 音频特征通过 DCMMOE 和 RAAV 两种融合策略组合，在 Aff-Wild2 上达到连续 VA 估计 CCC 0.658（dev）/ 0.62（test），验证了 VLM 行为语义对连续情感识别的价值。
tags:
  - CVPR 2026 (ABAW Workshop)
  - 人体理解
  - 效价-唤醒估计
  - 多模态融合
  - VLM行为描述
  - Mamba
  - ABAW竞赛
---

# Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach

**会议**: CVPR 2026 (ABAW Workshop)  
**arXiv**: [2603.13056](https://arxiv.org/abs/2603.13056)  
**代码**: [GitHub](https://github.com/SMIL-SPCRAS/CVPRW-26)  
**领域**: 人体理解 / 情感计算  
**关键词**: 效价-唤醒估计, 多模态融合, VLM行为描述, Mamba, ABAW竞赛

## 一句话总结

首次将 VLM（Qwen3-VL-4B-Instruct）提取的情感行为描述嵌入作为独立第三模态，与 GRADA 人脸编码器和 WavLM 音频特征通过 DCMMOE 和 RAAV 两种融合策略组合，在 Aff-Wild2 上达到连续 VA 估计 CCC 0.658（dev）/ 0.62（test），验证了 VLM 行为语义对连续情感识别的价值。

## 研究背景与动机

**领域现状**：连续效价-唤醒（VA）估计在野外条件下仍然困难——外观变化大、头部姿态多样、遮挡频繁、音频噪声大。ABAW 挑战赛是该领域最权威基准，此前 SOTA 方法主要使用人脸+音频+跨注意力融合管线。

**现有痛点**：现有多模态方法仅用传统特征提取器（EfficientNet 视觉 + VGGish/WavLM 音频），无法捕捉丰富的行为级语义——面部表情变化趋势、手势含义、身体姿态与情境的关系。VLM 在视频理解中展现强大上下文捕捉能力，但尚未用于连续 VA 估计。

**核心矛盾**：传统帧级视觉特征只编码外观，缺乏对行为语义和情境上下文的理解。VLM 能提供这种高层语义，但其输出是段级而非帧级，且时间分辨率、信息密度与传统模态差异巨大，如何有效整合是关键挑战。

**本文目标** (1) VLM 段级输出如何与帧级视觉/音频对齐？(2) 噪声严重的野外音频如何可靠利用？(3) 三种时间分辨率和信息密度差异巨大的模态如何自适应融合？

**切入角度**：用 Qwen3-VL 处理视频段并通过情感导向 prompt 提取行为级嵌入→Mamba 建模段级时序→帧级展开；用嘴部开合做音频可靠性过滤；设计 DCMMOE 和 RAAV 两种融合策略。

**核心 idea**：VLM 行为描述嵌入作为第三模态 + 两种非对称融合策略（DCMMOE/RAAV），将 VLM 的行为理解能力注入连续情感估计。

## 方法详解

### 整体框架

三路独立单模态编码（GRADA 人脸 + Qwen3-VL 行为描述 + WavLM 音频）→两种可选融合策略（DCMMOE 或 RAAV）→帧级 VA 预测。各模态有独立的时序建模器（Transformer / Mamba / 块级池化），融合前投影到共享隐空间。

### 关键设计

1. **GRADA 人脸模型 + Transformer 时序回归**

    - 功能：提取帧级情感表示并建模短/中期时序动态
    - 核心思路：EfficientNet-B1 在 10 个情感数据集上多任务微调（7.9M 参数），输出 256 维帧级情感嵌入。YOLO 人脸检测 + 手动身份标注确保单一目标。Transformer 回归模型在长度 $L=400$、步长 $S=150$ 的滑动窗口上处理，含投影块（FC+LN+Dropout）、多层 Transformer（$N=5, H=16$）和回归头（FC+LN+GELU+Dropout+FC）
    - 设计动机：EfficientNet-B1 在多架构评比中展现最佳泛化/效率平衡。Transformer 滑动窗口保持时序连续性同时增加训练样本

2. **Qwen3-VL 行为描述模型 + Mamba 时序编码**

    - 功能：利用 VLM 捕捉传统特征提取器无法编码的行为级语义
    - 核心思路：Qwen3-VL-4B-Instruct 处理 16 帧视频段 + 情感导向 prompt（引导关注面部表情、头部动作、手势、姿态和场景），提取最后隐藏层 token 作为段级嵌入 $e \in \mathbb{R}^d$。两种设置：纯视觉嵌入（仅视觉 token）和多模态嵌入（视频+文本联合）。段级嵌入经 Mamba 编码器建模时序（视觉版 4 层 hidden=128 state=8 kernel=3；多模态版 12 层 hidden=256 state=8 kernel=5），帧级预测通过段到帧展开+重叠平均
    - 设计动机：多模态嵌入（CCC 0.539）大幅优于纯视觉（0.401），证明 prompt 引导文本上下文对行为理解至关重要

3. **WavLM 音频模型 + 跨模态可靠性过滤**

    - 功能：从音频提取情感线索，同时过滤噪声严重的非语音段
    - 核心思路：4 秒段 2 秒重叠，用 MediaPipe 检测嘴部开合做跨模态过滤——仅保留张嘴时长和标注覆盖率超阈值的段。微调 WavLM-Large 顶部 4 层（基于 MSP-Podcast 预训练），4 秒段分 4 个时间块，每块用注意力统计池化（加权均值+加权标准差）聚合，回归头输出 VA
    - 设计动机：Aff-Wild2 野外音频噪声极大，嘴部开合检测是简单有效的语音存在近似指标

4. **DCMMOE 融合策略**

    - 功能：建模所有模态有向交互对，自适应加权融合
    - 核心思路：各模态投影到共享 $d_h$ 维空间，所有有序对 $(q,k)$ 构成交叉注意力专家（$|\mathcal{E}|=M(M-1)$ 个），每专家 $N$ 层 $H$ 头交叉注意力（query=模态 $q$，key/value=模态 $k$）。门控网络从平均多模态状态计算专家权重 $\mathbf{g}_l = \mathbf{W}_g \bar{\mathbf{h}}_l + \mathbf{b}_g$，融合 $\mathbf{z}_l = \sum_{(q,k)} \text{softmax}(\mathbf{g}_{(q,k),l}) \mathbf{Z}_{(q,k),l}$
    - 设计动机：显式建模有向跨模态交互（query 和 context 不对称）+ 数据依赖的专家加权

5. **RAAV 融合策略**

    - 功能：帧中心的非对称融合——视觉模态确定时间分辨率，音频提供补充上下文
    - 核心思路：面部和行为特征在每帧通过 masked 可靠性感知门控融合 $\mathbf{z}_{\text{vis},l} = \sum_m \alpha_l^{(m)} \mathbf{h}_l^{(m)}$，其中 $\alpha$ 由学习的评分函数 + 模态先验决定。融合视觉序列再通过 bottleneck 交叉注意力从音频 $\mathbf{B}_a$ 提取上下文 $\mathbf{Z}_0 = \text{LN}(\mathbf{Z}_\text{vis} + \text{CrossAttn}(\mathbf{Z}_\text{vis}, \mathbf{B}_a, \mathbf{B}_a))$
    - 设计动机：反映 VA 估计中视觉主导、音频辅助的任务特性

### 损失函数 / 训练策略

- 混合 CCC 损失 + 可选 MAE 项，valence/arousal 可独立加权
- AdamW，lr=1e-4，batch=8，ReduceLROnPlateau
- 人脸骨干 lr=5e-6，头 lr=2e-4；WavLM 微调顶部 4 层；50 epoch

## 实验关键数据

### 主实验

| ID | 配置 | Valence CCC | Arousal CCC | Avg CCC | Test Avg |
|----|------|-------------|-------------|---------|----------|
| 1 | 人脸 GRADA+Transformer | 0.587 | 0.651 | 0.619 | 0.54 |
| 2 | 行为 Qwen3 视觉+Mamba | 0.250 | 0.552 | 0.401 | - |
| 3 | 行为 Qwen3 多模态+Mamba | 0.429 | 0.648 | **0.539** | - |
| 4 | 音频 WavLM+块池化 | 0.342 | 0.464 | 0.403 | - |
| 5 | 人脸+音频 DCMMOE | 0.625 | 0.667 | 0.646 | 0.58 |
| 7 | 人脸+行为(多)+音频 DCMMOE | 0.610 | 0.688 | 0.649 | 0.61 |
| 8 | 人脸+行为(多)+音频 **RAAV** | 0.608 | **0.707** | **0.658** | **0.62** |

### 消融实验

| 对比 | Avg CCC | 差异 | 说明 |
|------|---------|------|------|
| Qwen3 多模态 vs 纯视觉 | 0.539 vs 0.401 | +0.138 | prompt 引导文本上下文至关重要 |
| 三模态 vs 双模态(人脸+音频) | 0.649 vs 0.646 | +0.003 | VLM 模态带来一致但小幅提升 |
| RAAV vs DCMMOE (三模态) | 0.658 vs 0.649 | +0.009 | RAAV 在 arousal 上优势明显 |
| 融合 vs 最佳单模态 | 0.658 vs 0.619 | +0.039 | 融合一致优于单模态 |

### 关键发现

- Qwen3 多模态嵌入（0.539）大幅优于纯视觉（0.401），差距 0.138 CCC——VLM 纯视觉特征直接做回归效果很差，必须有文本上下文引导
- 三模态融合一致优于双模态和单模态，但 VLM 增量收益仅 +0.003，可能受限于段级→帧级展开的时间分辨率损失
- RAAV 在 arousal 上特别强（0.707 vs 0.688），DCMMOE 在 valence 上稍优（0.625 vs 0.608），反映不同融合策略对不同维度的偏好
- Dev（0.658）到 test（0.62）下降 0.038 提示泛化性仍有改进空间

## 亮点与洞察

- **首次将 VLM 行为描述作为独立模态**用于连续 VA——多模态 vs 纯视觉嵌入的巨大差距（0.539 vs 0.401）清晰展示了 prompt 引导行为语义的价值。思路可推广到动作识别、社交信号处理等
- RAAV 的非对称设计（视觉决定时间分辨率，音频提供补充上下文）合理反映 VA 估计的任务特性
- 嘴部开合做音频可靠性过滤是简单但有效的跨模态策略——利用视觉信号预筛音频质量，成本近零
- DCMMOE 的 $M(M-1)$ 有向对专家 + 自适应门控比简单拼接更精细地建模模态间非对称交互

## 局限与展望

- Qwen3 段级嵌入→帧级展开存在时间分辨率硬损失，token-level 嵌入可能改善
- VLM 推理成本高（Qwen3-VL-4B），实时部署困难
- Aff-Wild2 约 3M 帧但仅 584 受试者，个体差异可能主导结果
- Dev(0.658)到 test(0.62)下降表明跨受试者泛化不足
- 未探索将 VLM 输出直接作为帧级描述而非段级嵌入

## 相关工作与启发

- **vs Yu et al. (9th ABAW 冠军)**：使用 ResNet + VGGish/LogMel + TCN + 跨模态注意力。本文增加 VLM 行为模态，用 Mamba 替代 TCN，CCC 有竞争力（0.62 test）
- **vs Praveen et al. (8th ABAW)**：GR-JCA 做双模态融合。本文 DCMMOE 建模所有有向对 + 自适应门控，融合粒度更细
- **vs Lee et al. (9th ABAW)**：Time-aware Gated Fusion。本文创新在于引入 VLM 行为模态

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐：首次将 VLM 行为描述用于连续 VA，多模态 vs 视觉嵌入对比有洞察
- **实验充分度** ⭐⭐⭐⭐：8 种配置系统对比，两种融合策略，单/双/三模态全覆盖
- **写作质量** ⭐⭐⭐⭐：结构清晰，融合策略公式化完整
- **价值** ⭐⭐⭐⭐：对情感计算 + VLM 交叉领域有参考价值

<!-- RELATED:START -->

## 相关论文

- [PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning](pulse_privileged_knowledge_transfer_from_rich_to_deployable_sensors_for_embodied.md)
- [SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size](teamhoi_learning_a_unified_policy_for_cooperative_human-object_interactions_with.md)
- [Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_multimodal_task_generation_agents.md)
- [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)

<!-- RELATED:END -->
