---
title: >-
  [论文解读] TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization
description: >-
  [ICLR 2026][语音][视频摘要] 提出 TripleSumm，通过多尺度时序块（层级滑动窗口注意力）和跨模态融合块（融合 token 自适应加权视觉/文本/音频），实现帧级模态重要性动态调整，并发布首个大规模三模态视频摘要数据集 MoSu（52678 视频），在 4 个 benchmark 上达到 SOTA。
tags:
  - ICLR 2026
  - 语音
  - 视频摘要
  - 三模态融合
  - 自适应权重
  - 多尺度时序
  - 大规模数据集
---

# TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization

**会议**: ICLR 2026  
**arXiv**: [2603.01169](https://arxiv.org/abs/2603.01169)  
**代码**: https://github.com/smkim37/TripleSumm  
**领域**: 视频摘要 / 多模态  
**关键词**: 视频摘要, 三模态融合, 自适应权重, 多尺度时序, 大规模数据集

## 一句话总结
提出 TripleSumm，通过多尺度时序块（层级滑动窗口注意力）和跨模态融合块（融合 token 自适应加权视觉/文本/音频），实现帧级模态重要性动态调整，并发布首个大规模三模态视频摘要数据集 MoSu（52678 视频），在 4 个 benchmark 上达到 SOTA。

## 研究背景与动机

**领域现状**：视频摘要提取关键片段代表原视频内容。现有方法主要使用视觉特征+注意力机制。

**现有痛点**：模态重要性帧到帧动态变化（评委说话时文本重要，机器人表演时视觉+音频重要），但现有方法用静态/模态无关的融合策略。且无大规模三模态数据集。

**核心 idea**：自适应帧级模态融合 + 大规模三模态 benchmark。

## 方法详解

### 整体框架
原始视频的视觉/文本/音频流 → 模态特定预训练编码器（GoogLeNet/CLIP + RoBERTa + AST）→ 线性投影+LayerNorm 到共同维度 D → 逐帧聚合为融合 token $\mathbf{E}^f$ → 交替堆叠 L 层 Multi-scale Temporal block (MST) + Cross-modal Fusion block (CMF) → 预测头输出帧级重要性分数 $\hat{S} \in [0,1]$ → 选择高分帧组成摘要。

### 关键设计

1. **多尺度时序块（MST）**：

    - 使用窗口自注意力（WSA），窗口大小 $w$ 逐层增大（初始层小窗口捕局部依赖，后续层大窗口捕长程依赖）
    - 复杂度从全注意力的 $O(N^2)$ 降至 $O(w \cdot N)$
    - 跨模态共享参数——同一 WSA 对四种 token（融合/视觉/文本/音频）使用相同参数
    - 设计动机：视频帧的时序变化（如场景切换、节奏变化）是模态无关的，共享参数可高效捕获通用时序模式

2. **跨模态融合块（CMF）**：

    - 融合 token $\mathbf{h}^f_i$ 作为 query，三个模态 token $\mathbf{h}^{\{v,t,a\}}_i$ 作为 key/value
    - 交叉注意力实现帧级自适应加权——每个时间步独立决定关注哪个模态
    - 设计动机：避免传统方法中的模态偏向（如总是以视觉为 query），融合 token 作为中立锚点公平对待三模态

3. **融合 token 设计**：

    - $\mathbf{e}^f_i = \text{Agg}(\mathbf{e}^v_i, \mathbf{e}^t_i, \mathbf{e}^a_i)$，聚合函数可为平均或 MLP
    - 加入时序位置编码（TPE）+ 可学习模态嵌入（LME）区分时间步和模态来源
    - 关键：融合 token 在 CMF 中被更新后携带了最相关模态的信息，但不直接修改各模态 token

4. **MoSu 数据集构建**：

    - 从 YouTube-8M 筛选：(1) 英语字幕和音轨可用；(2) >50000 观看量以获取 "Most Replayed" 统计；(3) ≥120 秒确保足够长
    - 最终 52678 个视频，涵盖视觉+文本+音频三模态特征
    - 标注来自 YouTube "Most Replayed" 热力图——每个视频至少 50000 名观众的集体行为反馈

### 损失函数 / 训练策略
L2 回归损失：$\mathcal{L} = \|S - \hat{S}\|_2^2$，预测帧级重要性分数。最终摘要通过选择最大化预测分数的时序连贯片段生成。

## 实验关键数据

### 主实验

| 基准 | 指标 | TripleSumm | 之前SOTA | 提升 |
|------|------|-----------|---------|------|
| MoSu | Kendall τ | 0.145 | 0.107 (CFSum) | +35.5% |
| Mr.HiSum | Kendall τ | 0.105 | 0.089 | +18.0% |
| SumMe | F1 | 52.3 | 50.1 | +2.2 |
| TVSum | F1 | 63.7 | 61.5 | +2.2 |

### 消融实验

| 配置 | MoSu τ | 说明 |
|------|--------|------|
| Full TripleSumm | 0.145 | 完整 |
| 仅视觉 | 0.091 | 大幅退化 |
| 视觉+文本 | 0.128 | 音频贡献明显 |
| w/o MST | 0.121 | 多尺度重要 |
| w/o CMF | 0.118 | 自适应融合关键 |

### 关键发现
- 缺失模态时 TripleSumm 鲁棒退化——动态依赖可用模态，不会因单模态缺失而崩溃
- 定性分析显示融合 token 在不同帧自适应地关注不同模态（如评委发言帧→文本权重高，音乐表演帧→音频权重高）
- 仅视觉特征的模型在 MoSu 上 τ=0.091，加入文本后提升至 0.128，再加音频到 0.145——每个模态都有独立贡献
- MST 的多尺度设计对长视频特别重要——单一窗口大小的模型在 MoSu 上 τ 下降约 16%
- 参数效率高：TripleSumm 的参数量与仅视觉的 PGL-SUM 相当，但利用了三倍的信息源

## 亮点与洞察
- 融合 token 作为跨模态交互的"中立锚点"是关键创新——避免了传统方法中以视觉为 query 的模态偏向问题
- MST 的层级窗口设计让模型从局部到全局逐步建立时序理解——这对长视频（>2分钟）尤其重要
- 三模态的加入不仅提升了性能，更提升了鲁棒性——缺失任何单一模态时性能退化可控
- MoSu 的 "Most Replayed" 标注方案是务实的选择——利用集体观看行为作为帧重要性的免费代理标注

## 局限与展望
- MoSu 基于 YouTube "Most Replayed"，可能偏向娱乐性内容，教育/专业视频的覆盖不足
- 融合 token 的初始化使用简单平均聚合，更复杂的初始化（如门控）可能进一步提升
- 仅使用预训练编码器的冻结特征，端到端微调编码器可能释放更多潜力
- WSA 窗口大小按固定 schedule 增大，自适应调整窗口大小可能更优
- 未探索 LLM-based 摘要方法（如 VideoLLM）的对比

## 相关工作与启发
- **vs CFSum**：CFSum 也用三模态但采用静态融合，TripleSumm 的帧级自适应加权是关键差异
- **vs A2Summ**：A2Summ 侧重音频-视觉双模态，TripleSumm 完整覆盖三模态
- **vs PGL-SUM/CSTA**：这些仅用视觉特征的 Transformer 方法在 MoSu 上大幅落后，验证了多模态的必要性
- **MoSu 数据集**：首个大规模三模态视频摘要 benchmark，基于 52678 个YouTube视频的 "Most Replayed" 统计，每个视频至少 50000 观看者投票
- **启发**：融合 token 作为跨模态交互的"锚点"的设计思想可迁移到其他多模态任务（如多模态检索、视频问答）

## 评分
- 新颖性: ⭐⭐⭐⭐ 自适应三模态融合 + 大规模数据集，方法和资源双贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 4 benchmark + 消融 + 定性分析 + 缺失模态鲁棒性测试
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ MoSu 数据集和三模态融合方法都具有持久价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion](../../CVPR2026/audio_speech/solution_for_10th_competition_on_ambivalencehesitancy_ah_video_recognition_chall.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](../../AAAI2026/audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[CVPR 2026\] OmniRet: Efficient and High-Fidelity Omni Modality Retrieval](../../CVPR2026/audio_speech/omniret_efficient_and_high-fidelity_omni_modality_retrieval.md)

<!-- RELATED:END -->
