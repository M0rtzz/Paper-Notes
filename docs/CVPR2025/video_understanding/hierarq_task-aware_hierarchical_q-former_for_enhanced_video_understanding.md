---
title: >-
  [论文解读] HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding
description: >-
  [CVPR 2025][视频理解][视频理解] 提出 HierarQ，一种任务感知的层次化 Q-Former 框架，通过双流语言引导特征调制器（实体流 + 场景流）和短/长期记忆库实现自回归式逐帧视频处理，无需帧采样即可绕过 LLM 上下文长度限制，在 10 个视频理解基准上取得 SOTA 或接近 SOTA 的性能。
tags:
  - CVPR 2025
  - 视频理解
  - Q-Former
  - 层次化
  - 任务感知
  - 记忆库
---

# HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding

**会议**: CVPR 2025  
**arXiv**: [2503.08585](https://arxiv.org/abs/2503.08585)  
**代码**: 无（有 Project page）  
**领域**: 视频理解  
**关键词**: 视频理解, Q-Former, 层次化, 任务感知, 记忆库

## 一句话总结

提出 HierarQ，一种任务感知的层次化 Q-Former 框架，通过双流语言引导特征调制器（实体流 + 场景流）和短/长期记忆库实现自回归式逐帧视频处理，无需帧采样即可绕过 LLM 上下文长度限制，在 10 个视频理解基准上取得 SOTA 或接近 SOTA 的性能。

## 研究背景与动机

当前多模态大语言模型（MLLM）在中长视频理解中面临三大瓶颈：

1. **上下文长度限制**：LLM 的上下文窗口限制了可处理的帧数，延长上下文虽可行但计算成本极高且难以达到理论承诺
2. **帧采样的信息丢失**：常用的帧采样（uniform/key-frame）在长视频中可能遗漏关键信息，且缺乏任务相关性——模型盲目处理所有帧，无法优先关注与任务相关的内容
3. **时空压缩的过度简化**：token 压缩、时空池化等方法虽减少 token 数量，但可能丢失关键细节

HierarQ 的核心思路是：像人类认知一样，**在观看视频时同时关注帧级实体细节（谁在做什么）和跨帧场景上下文（事件如何演进）**，并根据任务（prompt）动态调整关注重点。通过自回归逐帧处理，完全避免帧采样。

## 方法详解

### 整体框架

给定视频 $V$ 和文本 prompt $T_P$，HierarQ 的处理流程为：
1. 逐帧通过冻结的 ViT 提取视觉特征 $f_i = \mathcal{V}(v_i)$
2. 双流特征调制器对特征进行任务相关的调制
3. 调制后的特征存入短期/长期记忆库
4. 层次化 Q-Former（HierarQ）从记忆库中查询并融合信息
5. 最终时间步的输出经 FC 层投影后送入 LLM 生成回答

### 关键设计

1. **双流语言引导特征调制器（Two-stream Feature Modulator）**:
    - 功能：根据 prompt 的语义动态调制每帧的视觉特征，使模型"重点关注"与任务相关的帧
    - 核心思路：
        - **实体引导调制器 $L_f^e$**：从 prompt 中提取名词（人/物），用 BERT 编码为 $T_P^e$，通过交叉注意力与帧特征交互：$f_i^e = C.Attn(T_P^e, f_i, f_i)$。使帧聚焦于 prompt 中提及的实体
        - **场景引导调制器 $L_f^s$**：用完整 prompt 的 BERT 编码 $T_P^s$，通过交叉注意力：$f_i^s = C.Attn(T_P^s, f_i, f_i)$。捕获更宏观的场景级关系
    - 设计动机：实体流和场景流关注不同粒度：实体流在帧内定位"谁/什么"，场景流理解"事件/关系"。两者互补——实体细节支撑场景理解。轻量级 Transformer 设计保持高效

2. **短期/长期记忆库（Short/Long-term Memory Banks）**:
    - 功能：为 Q-Former 提供丰富的时间上下文，平衡即时细节和长期演化
    - 核心思路：
        - **短期记忆 $M_e$**：存储实体调制后的视觉特征和 query 历史，使用 FIFO 更新（达到容量 $M$ 时丢弃最旧条目），代价低廉
        - **长期记忆 $M_s$**：存储场景调制后的特征，使用 Memory Bank Compression (MBC) 更新——找到相似度最高的相邻 token 对 $k = \arg\max_t \cos(f_t, f_{t+1})$ 并取均值合并，保留时序顺序同时压缩冗余
    - 设计动机：实体信息是帧级的短期细节，FIFO 足够（旧帧的实体信息不重要）。场景信息需要跨越整个视频的长期上下文，简单 FIFO 会丢失关键场景连续性，需要更智能的压缩策略

3. **层次化 Q-Former（HierarQ）**:
    - 功能：将实体级和场景级信息层次化整合，输出固定数量（32）的 token 给 LLM
    - 核心思路：包含两个 Q-Former：
        - **实体级 $QF_e$**：标准 Q-Former，含 self-attention（query 间交互 + 短期 query 记忆）+ cross-attention（query 与短期视觉记忆交互），总结帧级实体信息
        - **场景级 $QF_s$**：扩展版 Q-Former，含 4 个子模块：①cross-attention（与长期视觉记忆交互）→ ②self-attention（与长期 query 记忆交互）→ ③self-attention（query 自身交互）→ ④cross-attention（与 $QF_e$ 的输出交互），实现从实体到场景的信息整合
    - 设计动机：层次化设计模拟人类认知——先关注具体实体，再在场景层面理解实体间关系。最后一步的跨 Q-Former 交叉注意力 $Q=\hat{z}_t^s, K=z_t^e, V=z_t^e$ 是将短期实体细节注入长期场景理解的关键。最终只输出 $N$ 个 token（而非 $N \times T$），从根本上解决了 LLM 上下文限制

### 损失函数 / 训练策略

使用标准交叉熵损失训练视频-文本对。冻结 ViT G/14 (EVA-CLIP) 和 Vicuna 7B，微调特征调制器 + HierarQ + FC 层，LLM 用 LoRA (rank=32) 微调。HierarQ 权重从 InstructBLIP 初始化。4 块 A100 GPU 训练。

## 实验关键数据

### 主实验

**中长视频理解 (LVU/Breakfast/COIN)**:

| 模型 | LVU Avg | Breakfast | COIN |
|------|---------|-----------|------|
| S5 | 60.9 | 90.7 | 90.8 |
| MA-LMM | 61.1 | 93.0 | 93.2 |
| VideoMamba | 57.8 | 94.3 | 86.2 |
| **HierarQ** | **67.9** (+6.8) | **97.4** (+3.1) | **96.0** (+2.8) |

**短视频问答 (MSRVTT-QA / MSVD-QA / ActivityNet-QA)**:

| 模型 | MSR-QA | MSVD-QA | ANet-QA |
|------|--------|---------|---------|
| Mirasol3B | 50.4 | - | 51.1 |
| MA-LMM | 48.5 | 60.6 | 49.8 |
| **HierarQ** | **54.1** (+3.7) | **66.2** (+5.6) | **57.1** (+6.0) |

### 消融实验

**各组件贡献 (LVU / Breakfast)**:

| 配置 | LVU | Breakfast | 说明 |
|------|-----|-----------|------|
| Baseline (MA-LMM) | 60.7 | 93.0 | 仅标准 Q-Former + 记忆 |
| + Entity Mod. | 58.7 | 88.5 | 仅实体调制反而下降（缺场景上下文）|
| + Prompt Mod. | 62.0 | 94.1 | 场景调制单独有效 |
| + HierarQ + 双流 | 66.8 | 96.1 | 层次化整合大幅提升 |
| + LLM LoRA | **67.9** | **97.4** | 完整模型 |

**记忆更新策略**:

| 短期更新 | 长期更新 | LVU | Breakfast |
|---------|---------|-----|-----------|
| FIFO | FIFO | 65.2 | 93.6 |
| MBC | MBC | 67.4 | 97.3 |
| **FIFO** | **MBC** | **67.9** | **97.4** |

### 关键发现

- 单独使用实体调制器反而降低性能（缺少场景上下文），但与场景调制器和 HierarQ 组合时效果最佳——验证了"实体细节补充场景理解"的层次化设计
- 长期记忆比短期记忆更重要（62.5 vs 61.8），但两者组合达最优（66.8）
- 短期记忆最佳长度 ~10，超过后过多的实体信息反而干扰场景理解
- 隔离 $QF_e$ 和 $QF_s$（取消层次交互改为拼接）导致 LVU 下降 3.6%，证明层次化建模的必要性
- 增加参数到与 HierarQ 相同量级的单 Q-Former 仍落后 4.7%——提升来自架构而非参数量
- 随视频长度增加，MA-LMM 性能持续下降，而 HierarQ 保持稳定

## 亮点与洞察

- **认知科学启发的设计**：实体流/场景流的双流设计直接对应人类的"局部注意 + 全局理解"认知模式
- **工程上的优雅**：自回归处理 + 固定 N token 输出，从根本上解决了 LLM 上下文长度问题（$N$ 而非 $N \times T$）
- **任务感知**：通过 prompt 引导特征调制，不同任务会自动"关注"不同帧——这比盲目处理所有帧更符合人类观看视频的方式
- **记忆策略的精细化**：短期用 FIFO、长期用 MBC 的差异化更新策略设计简洁有效

## 局限与展望

- 依赖 BERT 提取实体名词，NPL 解析错误会影响实体流的准确性
- 虽然框架通用，但消融实验主要在中长视频上进行，对超长视频（>10min）的验证较少
- 未探索记忆库容量的自适应调整（当前固定 M=10）
- HierarQ 从 InstructBLIP 初始化，对预训练数据有一定依赖

## 相关工作与启发

- **与 MA-LMM 的关系**：HierarQ 在 MA-LMM 的 Q-Former + 记忆基础上引入双流调制和层次化设计，是直接的改进和扩展
- **与 MovieChat 的关系**：两者都用记忆机制处理长视频，但 MovieChat 的记忆合并较粗糙，HierarQ 的 MBC 更精细
- **启发**：（1）双流/层次化的思路可推广到其他多粒度理解任务（如文档理解：词级 + 段落级）；（2）任务感知的特征调制是一种通用的效率提升策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 层次化 Q-Former + 双流任务感知调制的组合设计新颖，但各组件思路有迹可循
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个基准覆盖视频理解/QA/描述三大任务，消融极其详尽
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，图示专业，消融分析深入
- 价值: ⭐⭐⭐⭐⭐ 为中长视频 MLLM 提供了实用且高效的方案，在多数基准 SOTA

<!-- RELATED:START -->

## 相关论文

- [MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)
- [Object-Shot Enhanced Grounding Network for Egocentric Video](object-shot_enhanced_grounding_network_for_egocentric_video.md)
- [Progress-Aware Video Frame Captioning](progress-aware_video_frame_captioning.md)
- [Context-Enhanced Memory-Refined Transformer for Online Action Detection](context-enhanced_memory-refined_transformer_for_online_action_detection.md)
- [FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)

<!-- RELATED:END -->
