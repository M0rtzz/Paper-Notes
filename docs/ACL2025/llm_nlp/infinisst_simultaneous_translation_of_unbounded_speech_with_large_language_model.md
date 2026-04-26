---
title: >-
  [论文解读] InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model
description: >-
  [ACL 2025][LLM/NLP][simultaneous translation] 提出 InfiniSST，将无界流式语音同声传译建模为 LLM 多轮对话任务，结合鲁棒片段训练数据构造、多延迟增强策略和 Λ-shaped KV cache 管理，在 MuST-C En-Es/De/Zh 三个方向上将计算感知延迟降低 0.5-1 秒而不损失翻译质量。
tags:
  - ACL 2025
  - LLM/NLP
  - simultaneous translation
  - streaming speech
  - KV cache
  - multi-turn dialogue
  - unbounded input
---

# InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model

**会议**: ACL 2025  
**作者**: Siqi Ouyang, Xi Xu, Lei Li (CMU)  
**arXiv**: [2503.02969](https://arxiv.org/abs/2503.02969)  
**代码**: [GitHub](https://github.com/LeiLiLab/InfiniSST)  
**领域**: LLM/NLP, 语音翻译  
**关键词**: simultaneous translation, streaming speech, KV cache, multi-turn dialogue, unbounded input  

## 一句话总结

提出 InfiniSST，将无界流式语音同声传译建模为 LLM 多轮对话任务，结合鲁棒片段训练数据构造、多延迟增强策略和 Λ-shaped KV cache 管理，在 MuST-C En-Es/De/Zh 三个方向上将计算感知延迟降低 0.5-1 秒而不损失翻译质量。

## 研究背景与动机

- **核心问题:** 同声传译（SST）需要在翻译质量和延迟之间取得平衡，但现有方法大多假设语音已预分割（SST-S），无法处理真实场景中连续无界的流式语音输入（SST-U）。
- **现有不足:** 传统 LLM-based SST-S 方法每当新语音块到达时需重新计算历史语音和已生成翻译的特征，计算开销巨大。虽有工作将 SST 建模为多轮对话以利用 KV 缓存（如 Yu et al., 2025; Wang et al., 2024），但这些方法仅针对已分段语音，无法无缝扩展到无界输入。
- **研究动机:** LLM 具备强大的长上下文建模能力（RoPE 位置编码 + 注意力窗口），是处理 SST-U 的理想基础。关键挑战在于：(1) 如何构造适合无界语音的训练数据；(2) 如何在推理时管理 KV cache 使内存使用恒定。

## 方法详解

### 整体框架

InfiniSST 由三个核心组件构成：
1. **流式语音编码器**（改造的 wav2vec2）：增量计算语音表示，避免重复计算
2. **语音-token 嵌入适配器**：两层 1-D 卷积（kernel=2, stride=2）+ 线性投影，将 48 帧压缩为 12 个 LLM 嵌入向量
3. **多轮 LLM 解码器**（Llama-3.1-8B-Instruct）：交替读取语音输入和生成翻译，通过 EOT token 控制读写切换

推理流程：系统指令 → 循环{USER token + 12 个语音嵌入 + EOT → ASSISTANT 生成翻译 → 遇到 EOT 切回读取}。

### 关键设计

1. **流式语音编码器改造**：将 wav2vec2 的双向注意力替换为 chunk-wise 因果注意力（块内双向、块间因果），卷积位置编码替换为 RoPE，加入滑动窗口 $w^s=10$ 个 chunk（约 9.6 秒上下文）。每个 chunk 48 帧、时长 960ms。
2. **训练数据构造**：(a) 用 MFA 强制对齐 + SimAlign 建立语音→转录→翻译的单调映射；(b) 将演讲切分为 30 chunk 的"鲁棒片段"以包含非语言声音增强鲁棒性；(c) 多延迟增强——随机选取延迟乘数 $m \in [1, M]$ 合并连续 chunk 及其翻译，$M=12$。
3. **KV Cache 管理策略**：推理时 LLM 保持系统指令的 KV cache + 最近 $w^t=1000$ 个 token 的 KV cache，所有 KV 值不嵌入位置信息（存储前移除 RoPE），拼接后重新应用 RoPE。实现了 Λ-shaped 注意力窗口的无限长度外推。

### 损失函数

标准交叉熵损失（仅作用于翻译 token 和 EOT）。两阶段训练：(1) 冻结 LLM，训练编码器+适配器 6 epoch（lr=2e-4）；(2) 冻结编码器+适配器，训练 LLM 1 epoch（lr=7e-6）。8×L40S GPU 单节点训练。

## 实验

### 主实验——MuST-C 完整 TED 演讲（27 场，3-23 分钟）

| 对比 | 计算感知延迟 (StreamLAAL_CA) | 非计算感知延迟 (StreamLAAL) | 翻译质量 |
|------|---------------------------|--------------------------|---------|
| InfiniSST vs StreamAtt+ | **降低 0.5-1 秒** | 相当或略优 | BLEU 持平或高 0.5-1.0 |
| InfiniSST RTF vs StreamAtt+ RTF | **不到一半** | — | — |

在 StreamLAAL≤1.5s 时 InfiniSST 在所有三个语言方向上 BLEU 略高（0.5-1.0），COMET 相当。

### 消融实验

| 消融组件 | 结果 |
|---------|------|
| 鲁棒片段 vs 原始分段 | 无鲁棒片段时模型遇到笑声等非语言声音陷入重复或停止翻译，COMET 从 69.2 退化到 50.5 |
| 多延迟增强 M=1 vs 12 | M 越大质量-延迟权衡越优，但推理 m 不应超过 M |
| 语音窗口 $w^s$=5/10/20/40 | 推理窗口与训练不匹配时质量下降（66.1 vs 69.2），建议用最大可允许窗口 |
| LLM 窗口 $w^t$=500→4000 | COMET 仅从 69.0 升至 69.4，LLM 对窗口大小鲁棒 |
| 移除指令 KV cache | LLM 在窗口滑动后停止翻译——指令缓存不可或缺 |
| Llama-3 (8K ctx) vs Llama-3.1 (128K) | 8K 模型仍可泛化到 >10min 演讲（COMET 67.1 vs 68.0） |

### 关键发现

- 鲁棒片段训练是使模型适应无界语音的关键——没有它模型完全无法处理非语言声音
- KV cache 管理使 RTF 降至 StreamAtt+ 的不到一半，是计算效率提升的核心
- 即使基础 LLM 上下文仅 8K，通过 KV cache 管理仍可处理远超 8K token 的无界语音

## 亮点

- 将无界流式 SST 建模为多轮对话是一个优雅的抽象，完美利用 LLM 的 KV cache 机制
- Λ-shaped 注意力窗口在语音翻译中的首次成功应用
- 训练数据构造（鲁棒片段 + 多延迟增强）设计精巧，对模型泛化至关重要

## 局限性

- 高理论延迟水平下仍落后于 AlignAtt 和 StreamAtt（chunk-wise 因果注意力限制了双向信息）
- 仅评估 En-X 方向，未测试 X-En 和 X-X
- 受计算预算限制未探索其他语音编码器和非 Llama LLM
- StreamLAAL 指标因 mWERSegmenter 对齐误差不完全可靠
- 未进行人工翻译质量评估

## 相关工作

- **级联 SST:** ASR 分割 + MT 翻译的流水线方法（Fugen et al., 2006），分段错误累积
- **端到端 SST-U:** AlignAtt/StreamAtt 扩展到无界语音（Papi et al., 2024a），但需全量存储历史
- **LLM 长度外推:** RoPE（Su et al., 2021）、Λ-shaped 注意力窗口（Han et al., 2024; Xiao et al., 2024）

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验完整度 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models](locateandfocus_enhancing_terminology_translation_in_speech.md)
- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [\[ACL 2025\] Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [\[ACL 2025\] Representation Bending for Large Language Model Safety](repbend_representation_bending_safety.md)
- [\[ACL 2025\] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)

<!-- RELATED:END -->
