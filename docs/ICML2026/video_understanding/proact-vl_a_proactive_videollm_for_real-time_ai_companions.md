---
title: >-
  [论文解读] ProAct-VL: A Proactive VideoLLM for Real-Time AI Companions
description: >-
  [ICML 2026][视频理解][视频大语言模型] ProAct-VL 通过分块输入-输出范式 + 轻量级 FLAG 决策头 + 过渡感知损失函数，使视频大语言模型在流式输入下能自主决定**何时响应**并生成短片段评论…
tags:
  - "ICML 2026"
  - "视频理解"
  - "视频大语言模型"
  - "流式推理"
  - "主动响应"
  - "实时交互"
  - "游戏解说"
---

# ProAct-VL: A Proactive VideoLLM for Real-Time AI Companions

**会议**: ICML 2026  
**arXiv**: [2603.03447](https://arxiv.org/abs/2603.03447)  
**代码**: 待确认  
**领域**: 视频理解 / 实时多模态交互  
**关键词**: 视频大语言模型, 流式推理, 主动响应, 实时交互, 游戏解说

## 一句话总结
ProAct-VL 通过分块输入-输出范式 + 轻量级 FLAG 决策头 + 过渡感知损失函数，使视频大语言模型在流式输入下能自主决定**何时响应**并生成短片段评论，同时实现 ~1 秒低延迟与强主动性——在游戏解说任务上响应时机 TimeDiff 仅 1.20 秒、触发 F1 = 63.25%，全面超越 GPT-4o 等离线模型。

## 研究背景与动机

**领域现状**：近年视频大语言模型（VideoLLM）快速发展，支持视频感知和用户实时交互。但大多数工作要么采用"分块-顺序处理"的被动响应方式，要么采用低延迟但缺响应控制的被动流式方式。

**现有痛点**：
- 主动响应模型（proactive）决定何时说话，但一旦触发就生成完整长答案——延迟高、时间粒度粗；
- 低延迟模型（real-time）强调快速生成，但缺对"说话行为"的显式控制，常过度说话；
- 现有方法难以平衡主动性时机与内容质量。

**核心矛盾**：真实的 AI 伙伴（如游戏解说员）需要三层协调——（1）低延迟推理、（2）自主决定何时响应、（3）控制生成内容的质量与数量——这三者构成三角形难以同时优化。

**本文目标**：构建统一框架同时解决"何时说"、"说什么"、"说多快"三个问题。

**切入角度**：游戏评论（commentary）和游戏指导（guidance）这两个场景具有丰富、可自动化评估的交互模式，因此选作具体评估场景；构建大规模标注数据集驱动模型训练。

**核心 idea**：分块级 I/O 范式 + FLAG 令牌决策头 + 过渡感知损失函数统一建模流式视频理解与主动响应。

## 方法详解

### 整体框架
每个时间步 $t$（1 秒分块）：
1. **输入**：三元组 $(V_t, Q_t, B_t)$——当前时间窗口视觉内容、可选用户查询、环境上下文（含前文评论摘要）。
2. **处理**：持久化 KV 缓存 $\mathcal{K}_{t-1}$ 维持完整上下文，因果 Transformer 处理。
3. **决策**：从特殊令牌 `<|FLAG|>` 隐状态 $h_t$ 提取说话概率 $p_t$，与阈值 $\tau$ 比较得二元决策 $a_t$。
4. **输出**：若 $a_t = 1$ 则生成短片段评论 $U_t$（约 1 秒），否则输出沉默令牌。生成的 $U_t$ 自动追加上下文，成为 $t+1$ 的输入。

### 关键设计

1. **分块级 I/O 范式**:

    - 功能：把连续视频流离散化为固定时长分块（本文 1 秒），支持在线因果处理。
    - 核心思路：每步 $t$ 模型从 $(V_t, Q_t, B_t)$ 和持久 KV 缓存 $\mathcal{K}_{t-1}$ 生成 $(U_t, \mathcal{K}_t)$；生成的 $U_t$ 立即追加为 $t+1$ 输入的一部分，形成连续对话历史。
    - 设计动机：通过分块 + 持久缓存避免每步重新处理全文的计算浪费，同时保持完整时间上下文；多分块长回答能自然跨越后续时步连接。

2. **轻量级主动响应机制**:

    - 功能：在每个用户消息末尾插入特殊 FLAG 令牌，从其隐状态通过轻量 MLP + sigmoid 计算说话概率，阈值 $\tau$ 得二元决策。
    - 核心思路：$p_t = \sigma(\text{MLP}(h_t))$、$a_t = \mathbb{I}[p_t \geq \tau]$。决策头极轻量，不增加推理瓶颈。
    - 设计动机：相比把主动性和生成耦合的设计，解耦的决策机制允许模型独立学"何时说"的策略，提高训练效率和推理效率。

3. **多层次稳定性损失（过渡感知 + 稳定性正则）**:

    - 功能：通过"过渡感知分类损失"和"稳定性正则化"两部分组成 $\mathcal{L}_{\text{resp}}$，与主要语言建模损失 $\mathcal{L}_{\text{main}}$ 加权组合。
    - 核心思路：过渡感知分类损失 $\mathcal{L}_{\text{cls}}$ 用权重 $w_t = \gamma$（当 $y_t \neq y_{t-1}$ 时）和 $w_t = 1$（状态持续），重点强调"说话-沉默"状态转换这种稀有但关键事件。稳定性正则 $\mathcal{L}_{\text{reg}}$ 含两项——局部时间一致性 $\mathbb{E}[(p_t - p_{t-1})^2 \mid y_t = y_{t-1}]$（状态持续期间平滑概率变化）+ 全局说话率约束 $(\mathbb{E}[p_t] - \mathbb{E}[y_t])^2$（使模型平均说话时长与人类解说员匹配）。最终 $\mathcal{L} = \mathcal{L}_{\text{main}} + \alpha \mathcal{L}_{\text{resp}}$。
    - 设计动机：把响应状态视为序列学习问题而非独立二分类，显著提升"何时转换状态"的学习效果；全局说话率约束防止模型过度 / 不足说话。

## 实验关键数据

### 主实验（Live Gaming Benchmark）

| 模型类别 | 模型 | CC ↑ | LiveU ↑ | FinalQ ↑ | TimeDiff ↓ | F1 ↑ |
|---------|------|------|---------|---------|----------|------|
| 离线 | GPT-4o | 39.42 | 4.62 | 4.80 | 3.07 | 54.88 |
| 离线 | Gemini 2.5 Pro | — | 4.70 | 4.82 | 2.59 | 49.23 |
| 主动 | VideoLLM-online | 13.78 | 3.56 | 1.74 | 12.59 | 6.54 |
| 主动 | MMDuet | 20.08 | 2.67 | 2.68 | 26.72 | 0.18 |
| 主动 | Livestar | 8.59 | 3.14 | 2.41 | 27.33 | 0.20 |
| 低延迟 | LiveCC-7B-Base | 38.88 | 3.85 | 3.83 | 11.35 | 36.10 |
| 低延迟 | StreamingVLM | 14.89 | 3.49 | 2.65 | 2.21 | 50.67 |
| **本文** | **ProAct-VL** | **49.23** | **6.52** | **5.03** | **1.20** | **63.25** |

CC = 与 Gemini 2.5 Pro 的赢率；LiveU = 流式片段评论质量；FinalQ = 整体脚本质量；TimeDiff = 响应时间偏差（秒）；F1 = 触发精准度。ProAct-VL 在所有指标上最优，尤其响应时机（1.20s）和触发精准度（63.25%）远超基线。

### 消融实验

| 配置 | CC | TimeDiff | P | R | F1 | 说明 |
|------|-----|---------|------|-----|------|------|
| 仅 $\mathcal{L}_{\text{cls}}$ | 45.54 | 18.50 | 12.13 | 14.00 | 11.03 | 分类损失单独 |
| 仅 $\mathcal{L}_{\text{reg}}$ | 47.53 | 8.28 | 45.20 | 67.02 | 47.39 | 稳定性正则单独 |
| **完整** | **50.91** | **3.41** | **65.72** | **62.41** | **60.08** | 两个损失项结合 |

### 关键发现
- 移除 $\mathcal{L}_{\text{reg}}$ 影响最大——F1 下降 49.05，TimeDiff 增加 15.09，稳定性正则至关重要。
- 移除 $\mathcal{L}_{\text{cls}}$ 也导致性能下降但影响不如正则化大；两个损失项互补。
- 长序列稳定性：Streaming Commentary 从 73.75% 增至 82.03%（10-50 分钟视频），响应质量虽轻微衰减但趋于稳定（F1 从 74.42% 降至 69.23%）；相比 StreamingVLM 长期稳定性显著更优。

## 亮点与洞察
- **主动性与流式实时性的统一**：传统权衡"要么被动快速、要么主动但慢"；本文通过解耦决策与生成，~1 秒延迟下实现强主动性。设计思路可迁移到实时决策的交互任务（客服系统、实时字幕）。
- **过渡感知的加权机制**：把状态转换视为稀有事件并赋予高权重（$\gamma = 5$），核心洞察是"序列决策中转换往往比持续更重要"，对任何时序分类任务都有启发。
- **Live Gaming Dataset 的高质量标注流水线**：WhisperX ASR + Qwen3 情感标注 + DeepSeek 领域纠错三阶段，确保高精度转录；流水线（特别是 LLM 纠错 + 清洗）可复用到其他多模态数据集。

## 局限与展望
- 数据集限于游戏域（虽 12 款热门游戏，但核心是娱乐）；体育解说 / 新闻播报等领域的泛化能力有限。
- CC / LiveU / FinalQ 等指标均由闭源 LLM 计算（GPT-5.1），可重现性受限；跨语言 / 模态的人工验证仍需补充。
- 响应决策机制相对简朴——仅 FLAG 令牌隐状态 + MLP，可能忽略细粒度视觉信号（运动幅度、画面变化）。
- 改进方向：扩展到更多实时交互领域；引入多模态特征（音频情感、手势）增强决策；探索无阈值决策策略（直接回归延迟而非二分类）。

## 相关工作与启发
- **vs VideoLLM-online / MMDuet 等主动模型**：他们在"说话"时生成完整答案，延迟高（> 10s）且触发精准度低（F1 < 10%）；本文强制生成短片段（1s）+ 解耦决策保证主动性同时避免长尾延迟。
- **vs LiveCC / StreamingVLM 等低延迟模型**：优化推理速度但缺"何时说"的控制，常过度生成；ProAct-VL 通过显式响应头添加"沉默"能力，使其像人类一样有节制地交互。
- **vs GPT-4o / Gemini 离线模型**：理解能力强但无法实时；ProAct-VL 在性能接近（CC 49.23 vs GPT-4o 39.42）同时支持真正实时部署。

## 评分
- 新颖性: ⭐⭐⭐⭐  主动性与实时性的统一框架；过渡感知加权损失 + FLAG 决策机制虽单个不复杂但组合工程巧妙。
- 实验充分度: ⭐⭐⭐⭐⭐  覆盖 3 个交互场景 + 2 个测试集（in-domain + out-of-domain） + 长序列稳定性 + 消融 + 推理效率 + 人工验证。
- 写作质量: ⭐⭐⭐⭐  逻辑清晰、图表直观；某些细节（ChatML 格式、RoPE 修正）推至附录。
- 价值: ⭐⭐⭐⭐⭐  解决 AI 伙伴这一明确应用需求的真实问题；提供可部署系统 + 561 小时标注数据集；对直播 / 游戏 / 虚拟助手有直接推动力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos](../../AAAI2026/video_understanding/uncovering_zero-shot_generalization_gaps_in_time-series_foundation_models_using_.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](../../CVPR2025/video_understanding/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[ACL 2026\] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](../../ACL2026/video_understanding/response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)
- [\[ECCV 2024\] EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere](../../ECCV2024/video_understanding/egoposer_robust_real-time_egocentric_pose_estimation_from_sparse_and_intermitten.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](../../CVPR2026/video_understanding/streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)

</div>

<!-- RELATED:END -->
