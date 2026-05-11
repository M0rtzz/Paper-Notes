---
title: >-
  [论文解读] StreamReady: Learning What to Answer and When in Long Streaming Videos
description: >-
  [CVPR 2026][视频理解][流式视频理解] 提出就绪性感知的流式视频理解范式，通过可学习的 `<RDY>` token 和 Answer Readiness Score (ARS) 指标，让模型不仅回答正确，还能在证据出现的恰当时刻作答，在 9 个流式/离线视频基准上取得 SOTA。
tags:
  - CVPR 2026
  - 视频理解
  - 流式视频理解
  - 回答就绪性
  - 时序推理
  - 多模态大语言模型
  - 主动式问答
---

# StreamReady: Learning What to Answer and When in Long Streaming Videos

**会议**: CVPR 2026  
**arXiv**: [2603.08620](https://arxiv.org/abs/2603.08620)  
**代码**: [项目页面](https://sacrcv.github.io/StreamReady-website/)  
**领域**: 视频理解  
**关键词**: 流式视频理解, 回答就绪性, 时序推理, 多模态大语言模型, 主动式问答

## 一句话总结

提出就绪性感知的流式视频理解范式，通过可学习的 `<RDY>` token 和 Answer Readiness Score (ARS) 指标，让模型不仅回答正确，还能在证据出现的恰当时刻作答，在 9 个流式/离线视频基准上取得 SOTA。

## 研究背景与动机

**长视频流式理解的迫切需求**：现实场景（监控、体育分析、机器人、辅助系统）要求模型在视频帧顺序到达时实时推理，而非离线处理完整视频。

**现有方法只关注"答什么"忽视"何时答"**：大多数流式视频模型仅评估回答的正确性，完全忽略回答时机——过早回答意味着臆测，过晚回答降低实时性。

**主动式推理场景的挑战**：在 proactive 设定中，问题先于证据出现，模型需要持续观察视频流并判断何时积累了足够的视觉证据才能作答。

**缺乏时序标注的评估基准**：现有流式基准缺少答案证据时间窗口标注，无法系统评估模型的回答时机是否得当。

**现有延迟回答方案的局限**：辅助 MLLM 延迟（StreamBridge）或基于提示词的延迟策略存在非确定性或额外计算开销，且缺乏与推理模块的深度耦合。

**精细时序评估指标的空白**：需要一种能同时衡量正确性和时机的指标，对过早回答（臆测）施加严厉惩罚、对轻微延迟给予宽容。

## 方法详解

### 整体框架

StreamReady 采用 Qwen-2-VL (7B) 作为骨干 LLM，使用 HierarQ 的预训练权重初始化双分支 Q-Former，构建"分层记忆 → 查询感知推理 → 就绪性门控"的三阶段流水线：

1. **Memory Storage（记忆存储）**：将流式帧存入 Visual Memory Tree（三层结构：原始帧缓冲 ℳ_V1、EMA 聚类质心 ℳ_V2、抽象原型 ℳ_V3）+ Contextual Memory Bank ℳ_C（历史 QA 对的语义记忆，存储问题嵌入和答案表示）。三层记忆从细到粗，实现紧凑但信息丰富的视频表示。
2. **Query-Aware Reasoning（查询感知推理）**：问题到达后，短期分支 Q_s 直接在 ℳ_V1 上提取局部证据得到 z_s；长期分支 Q_ℓ 通过粗到细检索在 ℳ_V2/ℳ_V3 上定位并提取远距离证据，结合 z_s 进行跨尺度融合得到 z_ℓ；上下文推理步骤进一步将历史 QA 语义融入 z_ℓ。
3. **Readiness Mechanism（就绪性机制）**：`<RDY>` token 附加到长期推理表示 z_ℓ 中，由轻量 Readiness Head（2 层 MLP）输出就绪分数 R_pred ∈ [0,1]。推理时当 R_pred 超过阈值 0.35 才触发 LLM 生成回答；未就绪则继续观察后续帧。回答后，融合表示作为 a_i 存入 ℳ_C 中。

### 关键设计

- **Visual Memory Tree 三级压缩**：ℳ_V1 为 FIFO 原始帧缓冲，保存最近的原始帧嵌入用于短期细节感知；ℳ_V2 通过 EMA 聚类动态维护质心集合，当 ℳ_V1 满时淘汰帧通过公式 (1) 更新质心，阈值 τ_t 根据场景稳定性自适应调节（稳定场景收紧促进合并、新颖场景放松允许新质心）；ℳ_V3 在 ℳ_V2 容量饱和或分布漂移时进一步抽象为粗粒度原型集合，周期性触发 mini-K-means 重对齐以保持一致性。
- **粗到细查询检索**：先在 ℳ_V3 原型上 Top-K 定位相关时段（softmax 归一化稳定路由），再从对应 ℳ_V2 质心中 Top-m 提取细粒度证据（不归一化以保持锐利排序），类比"情景回忆"——原型提供粗时序锚点，质心提供细节。
- **Contextual Memory Bank**：存储历史 QA 对的问题嵌入 q_i 和答案表示 a_i，通过软门控匹配当前问题与历史问题，选取最相关条目经轻量交叉注意力融合到长期视觉特征 z_ℓ 中，支持多轮推理的语义连续性。
- **`<RDY>` token 与推理模块共演化**：`<RDY>` 直接嵌入长期推理分支 Q_ℓ 的学习表示中，与查询对齐的证据共同演化，能自然感知从"弱对齐、低置信"到"强对齐、可回答"的状态迁移。消融证明此放置位置（ARS 0.68）远优于短期分支（0.31）或长期分支输入端（0.54）。
- **弱监督就绪性学习**：训练时利用 z_ℓ 与 ℳ_V2 的时序相似度构建伪正/负区域，通过对比损失 ℒ_ctr 学习就绪性，无需真实证据时间戳标注。选择 ℳ_V2 层级因其在细节和紧凑性之间达到最佳平衡。
- **梯度隔离设计**：ℒ_rdy 仅更新 Readiness Head 和 `<RDY>` token，不回传梯度到推理模块，让"答什么"和"何时答"各自独立优化，避免两个目标相互干扰。

### 损失函数

- **对比就绪损失**：$\mathcal{L}_{ctr} = -\log \sigma(R_{pred}(t^+) - R_{pred}(t^-))$，其中 $t^+ \in P$（伪正区域），$t^- \in N$（伪负区域）。
- **时序平滑正则**：$\mathcal{L}_{rdy} = \mathcal{L}_{ctr} + \lambda_{reg} \|\nabla_t R_{pred}(t)\|_1$，L1 正则化抑制就绪信号的噪声抖动。
- **ARS 评估指标**：$\text{ARS} = \frac{1}{N}\sum_{i}(\text{EP}_i \cdot \text{LP}_i)$，Early Penalty 用 sigmoid 锐利惩罚过早回答（$\gamma_e=6$），Late Penalty 温和衰减延迟回答（$\gamma_\ell=1$）。有效精度 $\text{Acc}_e = \text{Acc} \times \text{ARS}$。

## 实验

### 主要结果

**ProReady-QA 就绪性感知评估**（Table 2）：

| 方法 | Size | Avg Acc. | Avg ARS | Acc_e |
|------|------|----------|---------|-------|
| Qwen-2-VL (baseline) | 7B | 41.4 | 0.34 | 0.20 |
| HierarQ | 7B | 46.0 | 0.40 | 0.27 |
| StreamBridge | 7B | 53.1 | 0.60 | 0.42 |
| InfiniPot-V | 7B | 52.0 | 0.47 | 0.36 |
| **StreamReady** | **7B** | **56.4** | **0.69** | **0.53** |

StreamReady 比最佳竞争者 StreamBridge 高出 ~3% 准确率和 ~9% ARS，有效精度提升 11 个百分点。
最大 ARS 增益出现在 REC (+0.25)、GSD (+0.11)、CTD (+0.10) 任务上，说明就绪机制在需要等待证据的任务上尤其有效。

**流式基准泛化**（Table 3）：

- StreamingBench 平均 63.4（vs ViSpeak 58.6），proactive 子集 48.2（vs ViSpeak 43.9）
- OVOBench 平均 68.2（vs StreamBridge 62.6），proactive 子集 63.7（vs ViSpeak 61.6）
- VStream-QA RE/RM 分别为 64.8/57.2，均为最优

**离线长视频基准**（Table 4）：

- VideoMME 65.8、MLVU 71.3、MVBench 71.8、EgoSchema 70.4
- 全面超越 StreamBridge（64.4/69.6/64.4/66.9）和 Flash-VStream（61.2/66.3/65.4/68.2）
- 离线评估时禁用就绪机制和上下文推理，仅靠记忆层级和查询推理即可取得优势

### 消融实验

| 配置 | REC Acc/ARS | GSD Acc/ARS | CTD Acc/ARS |
|------|-------------|-------------|-------------|
| Baseline (Qwen-2-VL) | 20.7/0.31 | 35.1/0.52 | 30.3/0.28 |
| + Memory + QA Reasoning | 39.4/0.48 | 60.9/0.53 | 43.6/0.39 |
| + Readiness Mechanism | **39.6/0.68** | **61.2/0.68** | **43.5/0.59** |

- 记忆+推理模块主要提升准确率（+19 on REC），就绪机制在此基础上大幅提升 ARS（+0.20 on REC/CTD）。
- `<RDY>` + MLP Head 与 Transformer Head 性能相当，但计算更轻。
- `<RDY>` 放置在长期推理的学习表示上效果最佳（ARS 0.68），放在短期分支仅 0.31。

### 关键发现

- 仅靠基础推理模块（如 HierarQ 的 Q-Former）提升准确率但对时机帮助甚微；就绪机制必须与强推理模块配合才能同时提升 Acc 和 ARS。
- 辅助 MLLM 判断就绪性（如 StreamBridge 策略）ARS 仅 0.60，不如与推理模块深度耦合的 `<RDY>` token（0.68）。
- StreamReady 在视频长度增长时保持恒定延迟和内存，得益于固定大小的质心/原型记忆，而 Qwen-2-VL 在长视频上 OOM。

## 亮点

- **首次形式化"就绪性感知"流式视频理解**：将回答时机纳入评估，提出非对称惩罚的 ARS 指标，填补了该领域的评估空白。
- **轻量优雅的就绪机制**：单个 `<RDY>` token + MLP Head，无需辅助模型或启发式规则，零额外推理开销。
- **弱监督时序学习**：无需真实证据时间戳标注，利用推理表示与记忆的相似度自动构建伪监督信号。
- **完整的 benchmark 贡献**：ProReady-QA 提供 5 类主动式任务、5K QA 对、30-60 分钟长视频、标注的证据时间窗口，支持局部和全局多轮依赖。
- **广泛的泛化能力**：在 9 个基准（流式 + 离线）上全面 SOTA，证明设计的通用性。

## 局限性

- ProReady-QA 仅包含 32 个视频（10 个 Ego-4D + 22 个 MovieNet），规模有限，视频类型多样性不足，可能无法覆盖所有现实流式场景。
- 就绪性学习依赖伪监督（z_ℓ 与 ℳ_V2 的相似度），在证据分布极度分散或模糊的场景下伪正/负区域可能不准确，导致就绪性判断失误。
- 就绪阈值 0.35 为固定超参数，不同任务和视频类型可能需要调优；论文未探索自适应阈值或基于置信度的动态策略。
- 三级记忆树的 K-means/EMA 更新引入多个超参（α、τ_t、J、U、K、m），调参负担较重，且 EMA 衰减因子的选择对记忆质量有较大影响。
- 仅在 7B 量级模型上验证，未探索更大（如 70B）或更小（如 2B）模型的表现，可扩展性有待验证。
- ARS 指标的非对称惩罚参数 γ_e=6、γ_ℓ=1 虽经实验验证稳健，但在不同应用场景（如安全监控 vs 运动分析）中最优惩罚权重可能不同。

## 相关工作

- **离线长视频理解**：HierarQ、LLaVA-Video、LongVU 等使用记忆或查询条件存储，但需完整视频和记忆重建，不适用流式场景。StreamReady 借鉴查询感知条件化思路但适配流式处理，无需记忆重置。
- **流式视频理解**：StreamBridge 通过辅助 MLLM 延迟回答实现主动行为，Flash-VStream/StreamForest/InfiniPot-V 关注在线记忆管理和检索效率，ViSpeak 探索语音交互场景，但均缺乏显式时机控制。StreamReady 的就绪机制可作为这些方法的互补扩展。
- **流式基准**：ODVBench、StreamBench、OVBench 仅支持过去依赖型问答；StreamingBench、OVOBench、Omni-MMI 引入主动式场景但限于短视频和局部上下文；ProReady-QA 首次在长视频上提供证据时间窗口标注、全局多轮依赖和五类主动推理任务。
- **回答时机控制**：现有方案包括基于提示词的延迟（"Answer whenever you are ready"）、辅助 MLLM 判断就绪性；StreamReady 提出将就绪性判断内嵌到推理模块中的新范式，避免外部依赖和非确定性。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次形式化回答时机问题并提出完整的指标+方法+基准
- 实验充分度: ⭐⭐⭐⭐⭐ — 9 个基准全面评估，消融详尽覆盖每个组件和设计选择
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式规范，图表丰富
- 价值: ⭐⭐⭐⭐⭐ — 定义新问题+新指标+新方法+新基准，对流式视频理解领域有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing](color_when_it_counts_grayscale-guided_online_triggering_for_always-on_streaming_.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)

</div>

<!-- RELATED:END -->
