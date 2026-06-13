---
title: >-
  [论文解读] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction
description: >-
  [ACL 2025][Turn-Taking] 提出 MM-VAP 多模态预测性话轮转换模型，将面部表情、头部姿态和注视方向等视觉线索引入语音预测模型，在视频会议语料上将 hold/shift 预测准确率从 79% 提升至 84%。 - 核心事实：两人对话中话轮之间平均仅 200ms 静默，而语言生成至少需要 600ms…
tags:
  - "ACL 2025"
  - "Turn-Taking"
  - "多模态"
  - "Facial Action Units"
  - "Predictive Model"
  - "Video Conferencing"
---

# Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction

**会议**: ACL 2025  
**arXiv**: [2505.21043](https://arxiv.org/abs/2505.21043)  
**代码**: [github.com/russelsa/mm-vap](https://github.com/russelsa/mm-vap)  
**领域**: 其他  
**关键词**: Turn-Taking, Multimodal, Facial Action Units, Predictive Model, Video Conferencing  

## 一句话总结

提出 MM-VAP 多模态预测性话轮转换模型，将面部表情、头部姿态和注视方向等视觉线索引入语音预测模型，在视频会议语料上将 hold/shift 预测准确率从 79% 提升至 84%。

## 研究背景与动机

- **核心事实**：两人对话中话轮之间平均仅 200ms 静默，而语言生成至少需要 600ms，说明话轮转换是**预测性**的——听者在说者尚未说完时就开始规划下一轮。
- **当前问题**：几乎所有预测性话轮转换模型（PTTM）仅依赖语音特征，忽略了视觉线索。这在电话场景下可接受，但在互相可见的场景（视频会议、面对面）中是一个明显的缺陷。
- **心理语言学证据**：研究表明，当受试者同时看到音频和视频时，判断话轮结束点的准确率更高（Barkhuysen et al., 2008）；眉毛皱起可加速问句识别（Nota et al., 2023）。
- **研究空白**：视觉线索是否能提升 PTTM 性能尚不明确，相关工作极少且数据规模小。

## 方法详解

### 整体框架

MM-VAP 基于 SOTA 纯语音模型 VAP (Ekstedt & Skantze, 2022) 扩展而来。VAP 使用 Transformer 持续预测未来 2 秒内的说话活动（Voice Activity Projection），以 8 个二值 bin 编码两位说话者的活动状态，共 256 种 VAP 状态。MM-VAP 在此基础上融合了视觉特征。

### 关键设计

1. **视觉特征提取**：使用 OpenFace 从视频的每一帧提取 60 维视觉特征向量，包括：
    - 17 个面部动作单元（FAU）：描述面部肌肉运动（如下颌张开、嘴唇运动），强度 0-5
    - 双眼注视向量：每眼一个 3D 单位向量
    - 头部位置（X,Y,Z）和旋转（roll, pitch, yaw）
    - 15 个面部关键点（眉毛、下颌、鼻子、嘴唇区域）

2. **模型架构（Late Fusion）**：
    - 音频通过预训练特征提取器得到 256 维特征向量（50Hz）
    - 视觉特征通过 MLP 投影到 256 维，从 30Hz 线性插值上采样至 50Hz
    - 先用 Self-Attention 块分别建模每个说话者的音频和视频时序模式
    - 再用 Cross-Attention 块学习同一说话者的音视频交互
    - 最后用 Cross-Attention 块学习两个说话者之间的跨模态时序模式
    - 因果掩码确保模型只能从过去帧预测
    - 总参数量 8.7M（VAP 为 5.8M）

3. **ASR 对齐验证**：首次在 PTTM 中使用自动语音识别替代手动对齐来提取 Voice Activity 标签，更贴近真实部署场景。在 Switchboard 上验证 ASR 导致的性能下降可控。

### 损失函数

交叉熵损失，训练模型输出的 256 维 softmax 分布与 256 种 VAP 状态标签之间的交叉熵。

## 实验

### 主实验结果（Candor 视频会议语料，710 小时）

| 模型 | F₁ (加权) | F₁ (Hold) | F₁ (Shift) | 平衡准确率 |
|------|----------|----------|-----------|-----------|
| Dummy (全 Hold) | 0.70 | 0.82 | 0.00 | 50% |
| VAP (纯语音) | 0.83 | 0.89 | 0.71 | 79% |
| **MM-VAP (Late)** | **0.86** | **0.90** | **0.77** | **83%** |
| **MM-VAP (Early)** | **0.87** | **0.91** | **0.79** | **84%** |

视觉线索的加入使 Shift 的 F₁ 提升了 6-8 个百分点（0.71→0.77/0.79），平衡准确率提升 4-5 个百分点。

### 按沉默时长分层分析（首次在 PTTM 中进行）

| 沉默时长 (FTO) | VAP 平衡准确率 | MM-VAP 平衡准确率 |
|---------------|--------------|-----------------|
| > 0 ms | 79% | 83% |
| > 250 ms | 79% | 83% |
| > 500 ms | 77% | 81% |
| > 750 ms | 75% | 78% |
| > 1000 ms | 73% | 76% |

MM-VAP 在所有沉默时长上都优于 VAP，且两者性能均随沉默时长增加而下降（长间隔更难预测）。

### 消融实验

| 视觉特征子集 | 相对于 VAP 的 F₁(Shift) 提升 |
|-------------|---------------------------|
| 完整视觉特征 | +6-8% |
| 仅面部动作单元（FAU） | **贡献最大** |
| 仅头部姿态 | 有提升但较小 |
| 仅注视方向 | 有提升但较小 |
| 仅面部关键点 | 有提升但最小 |

面部表情（通过面部动作单元编码）是最重要的视觉线索，与面部动作单元分析的发现一致——话轮转换前下一说者的嘴部、嘴唇、下颌和下巴运动显著增强。

### 关键发现

- **视觉线索确实有效**：在互相可见的场景中，视觉特征对话轮转换预测有显著贡献
- **面部表情是最关键的视觉线索**：FAU 的贡献远超注视和头部姿态
- **ASR 对齐可行**：自动语音识别导致的对齐误差约 480ms，但对 PTTM 性能影响有限
- **长沉默更难预测**：所有模型在长间隔场景下性能下降，但视觉线索在各时长上都有帮助

## 亮点

- 首次在大规模语料（710 小时）上系统验证了视觉线索对预测性话轮转换的价值
- 首创按沉默时长分层分析 PTTM 性能的评估方法，比传统的单一总体指标更有信息量
- 验证了 ASR 替代手动对齐的可行性，大幅降低了数据标注门槛
- 代码开源，便于复现和后续研究

## 局限性

- 仅在视频会议场景下验证，未测试面对面交互或更自然的场景
- OpenFace 的面部特征提取在部分视频中失败（238/1656 sessions），被排除在外
- 视觉特征提取为帧级别处理，计算成本较高，可能不适合实时部署
- 未探索更先进的视觉特征提取器（如基于 Transformer 的面部分析模型）
- Candor 语料仅包含美式英语的休闲对话，文化和语言的泛化性有待验证

## 相关工作

- **VAP 模型** (Ekstedt & Skantze, 2022)：当前 SOTA 的纯语音 PTTM，本文的基线和扩展基础
- **Roddy et al. (2018)**：首次在 PTTM 中引入注视向量，在 Mahnob 语料（11h）上小幅提升
- **Onishi et al. (2024)**：扩展 VAP 支持视觉，但仅在 1.5-2h 数据上测试且推理需要 VA 标签
- **Kurata et al. (2023)**：用视觉特征对已知话轮末端的 5s 片段分类，不是预测性模型
- **Stivers et al. (2009)**：跨语言研究确认话轮间平均 200ms 静默的普遍性

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 5 |
| **总分** | **4.5** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)
- [\[ACL 2025\] VAQUUM: Are Vague Quantifiers Grounded in Visual Data?](vaquum_are_vague_quantifiers_grounded_in_visual_data.md)
- [\[ACL 2025\] Unique Hard Attention: A Tale of Two Sides](unique_hard_attention_a_tale_of_two_sides.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)
- [\[CVPR 2025\] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](../../CVPR2025/others/care_transformer_linear_attention.md)

</div>

<!-- RELATED:END -->
