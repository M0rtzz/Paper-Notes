---
title: >-
  [论文解读] GazeInterpreter: Parsing Eye Gaze to Generate Eye-Body-Coordinated Narrations
description: >-
  [AAAI 2026][人体理解][眼动分析] 提出 GazeInterpreter，一种基于 LLM 的层次化框架，通过符号化眼动解析器将原始注视信号转化为文本叙述，再与身体运动叙述整合生成眼-体协调描述，并通过自我纠正循环迭代优化，显著提升文本驱动的运动生成、动作预测和行为摘要等下游任务的性能。
tags:
  - "AAAI 2026"
  - "人体理解"
  - "眼动分析"
  - "人体行为理解"
  - "大语言模型"
  - "多模态融合"
  - "运动生成"
---

# GazeInterpreter: Parsing Eye Gaze to Generate Eye-Body-Coordinated Narrations

**会议**: AAAI 2026  
**arXiv**: [2511.16245](https://arxiv.org/abs/2511.16245)  
**代码**: [https://github.com/EvergreenChang/GazeInterpreter](https://github.com/EvergreenChang/GazeInterpreter)  
**领域**: LLM评测  
**关键词**: 眼动分析, 人体行为理解, 大语言模型, 多模态融合, 运动生成

## 一句话总结

提出 GazeInterpreter，一种基于 LLM 的层次化框架，通过符号化眼动解析器将原始注视信号转化为文本叙述，再与身体运动叙述整合生成眼-体协调描述，并通过自我纠正循环迭代优化，显著提升文本驱动的运动生成、动作预测和行为摘要等下游任务的性能。

## 研究背景与动机

全面解释人类行为是人感知 AI 的核心挑战。然而，现有工作主要聚焦于**身体行为解释**，几乎完全忽略了**眼动注视（eye gaze）**及其与身体运动的协同关系：

**眼动是意图的直接窗口**：当一个人要抓杯子时，眼睛通常在手臂运动之前或期间就注视目标，揭示了潜在意图

**眼-体运动高度相关**：已有大量研究证明眼动与头部、躯干、全身运动之间存在强相关性

**现有方法的缺陷**：MotionGPT、MotionLLM 等将人体运动或视频投射到语言空间生成描述，但完全缺失眼动信息

**核心挑战**：如何将低层次的连续数值眼动传感器数据可靠地转化为高层次的结构化语义表示？直接使用 LLM 处理原始数值存在事实幻觉和信号脱节的风险。

**核心解决思路**：先将原始眼动数据抽象为中间符号化事件词汇表（symbolic gaze events），提供可靠的语义基础，再通过 LLM 进行多层次融合。

## 方法详解

### 整体框架

GazeInterpreter 采用三阶段层次化粗到细架构：

1. **Phase 1**：解析原始注视信号 → 文本叙述（符号化解析 + LLM 生成）
2. **Phase 2**：整合眼动叙述与身体运动原子叙述 → 眼-体协调叙述
3. **Phase 3**：自我纠正循环 → 多维度迭代优化

### 关键设计

#### 1. **符号化注视解析器（Symbolic Gaze Parser）**

这是一个确定性模块，将连续眼动信号转化为离散符号事件序列。

**具体步骤**：
- 对原始注视信号 $S_i^g \in \mathbb{R}^{N_g \times 2}$（yaw, pitch 序列），计算瞬时角速度：
$$\omega_j = \frac{\sqrt{(y_j - y_{j-1})^2 + (p_j - p_{j-1})^2}}{t_j - t_{j-1}}$$
- 使用 I-VT（Identification-by-Velocity-Threshold）算法，通过双阈值（$v_{\text{low}}=30°/s$, $v_{\text{high}}=100°/s$）将信号分为三类事件原语：**Fixation（注视）**、**Saccade（扫视）**、**SmoothPursuit（平滑追踪）**
- 每个事件不仅包含类别，还封装了持续时间、幅度、峰值速度等定量属性及对应的定性描述符

**设计动机**：将噪声的高维信号抽象为紧凑的、机器可读的符号表示，避免 LLM 直接处理数值时的幻觉问题。

#### 2. **符号-文本合成器（Symbolic-to-Text Synthesizer）**

利用 LLM（Gemini-2.5-Flash）将符号事件序列 $E_i$ 翻译为连贯的文本叙述 $T_i^g$。核心思想是将 LLM 的任务从"高风险的数值推断"转变为"低风险的事实翻译"——将符号化的、可验证的行为描述转为流畅自然语言。使用精心设计的 few-shot 提示。

#### 3. **眼-体运动整合（Phase 2）**

**历史上下文**：使用滑动观察窗口（$W=2$）聚合历史上下文 $\mathcal{H}_i$，包含 (i) 之前推断的整合叙述和 (ii) 上一轮自我纠正的反馈内容。

**整合叙述生成**：构建结构化提示模板：
$$\Pi_{\text{integ}}(i) = [\texttt{CTX}:\mathcal{H}_i;\ \texttt{GAZE}:T_i^g;\ \texttt{MOTION}:S_i^m]$$

LLM 对结构化输入进行推理（不仅是总结），例如将"注视转移"与"用户正在走路"关联推断出"用户在走路时仔细扫描地面"。

#### 4. **自我纠正循环（Phase 3）**

多维度评估+迭代优化，使用 $\text{LLM}_{\text{eval}}$ 和 $\text{LLM}_{\text{refine}}$ 协作：

**评估维度**（每个 1-5 分）：

| 类型 | 维度 | 高分定义 | 低分定义 |
|------|------|---------|---------|
| 注视叙述 | 连续性 | 自然流畅的注视过渡 | 含突兀、不合逻辑的事件描述 |
| 整合叙述 | 模态匹配 | 模态间相互支持的整合 | 模态脱节、冗余或矛盾 |
| 整合叙述 | 时间一致性 | 清晰的时间逻辑进程 | 缺乏可辨识的时间结构 |
| 整合叙述 | 完整性 | 完整包含所有关键元素 | 遗漏关键信息或行为事件 |

循环最多迭代 $K_{\text{max}}=3$ 次，直到所有分数 $\geq \tau=4.5$ 或达到最大次数。

### 损失函数 / 训练策略

GazeInterpreter 不需要传统意义上的模型训练，而是利用预训练 LLM（Gemini-2.5-Flash）的 few-shot in-context learning 能力。关键超参数：
- I-VT 阈值：$v_{\text{low}}=30°/s$, $v_{\text{high}}=100°/s$
- 滑动窗口大小：$W=2$
- 自我纠正最大迭代：$K_{\text{max}}=3$，分数阈值 $\tau=4.5$

## 实验关键数据

### 主实验

在 Nymeria 大规模基准上评估文本驱动运动生成（固定 MotionGPT 权重，比较不同文本输入）：

| 场景类型 | 方法 | MM Dist↓ | FID↓ | Top-1↑ | Top-3↑ | MM↑ |
|---------|------|----------|------|--------|--------|-----|
| Low-level | MotionGPT | 6.748 | 7.458 | 0.052 | 0.187 | 3.469 |
| Low-level | **+GazeInterpreter** | **6.406** | **6.801** | **0.102** | **0.214** | **3.727** |
| High-level | MotionGPT | 7.133 | 8.804 | 0.054 | 0.162 | 3.223 |
| High-level | **+GazeInterpreter** | **6.862** | **8.134** | **0.062** | **0.193** | **3.864** |
| All | MotionGPT | 6.941 | 8.131 | 0.053 | 0.175 | 3.346 |
| All | **+GazeInterpreter** | **6.634** | **7.468** | **0.082** | **0.204** | **3.796** |

**下游任务**：

| 任务 | 方法 | Cosine Sim↑ | BERT F1↑ | ROUGE-L↑ | Action F1↑ |
|------|------|------------|----------|----------|------------|
| 动作预测 | Nymeria | 0.459 | 0.868 | 0.202 | 0.226 |
| 动作预测 | **GazeInterpreter** | **0.506** | **0.879** | **0.231** | **0.248** |
| 行为摘要 | Nymeria | 0.480 | 0.836 | 0.197 | 0.150 |
| 行为摘要 | **GazeInterpreter** | **0.537** | **0.860** | **0.575** | **0.229** |

### 消融实验

| 配置 | MM Dist↓ | FID↓ | Top-1↑ | 说明 |
|------|----------|------|--------|------|
| w/o 层次结构 | 8.135 | 9.124 | 0.059 | 最大性能下降，验证层次化整合的核心地位 |
| w/o 符号解析器 | 7.642 | 7.893 | 0.061 | 直接用原始信号导致退化 |
| w/o 自我纠正 | 7.425 | 7.831 | 0.063 | 缺乏迭代优化降低质量 |
| **完整 GazeInterpreter** | **6.634** | **7.468** | **0.082** | 全部模块 |

**自我纠正质量维度的逐步分析**：

| 连续性 | 匹配 | 时间 | 完整性 | Top-1↑ | FID↓ |
|--------|------|------|--------|--------|------|
| | | | | 0.063 | 7.831 |
| ✓ | | | | 0.069 | 7.722 |
| ✓ | ✓ | | | 0.072 | 7.644 |
| ✓ | ✓ | ✓ | | 0.074 | 7.573 |
| ✓ | ✓ | ✓ | ✓ | **0.082** | **7.468** |

每个评估维度的引入都带来了累积性的性能提升。

### 关键发现

1. **眼动信息对运动生成至关重要**：仅通过提升文本描述质量（加入眼动信息），即可在不改变生成模型的情况下显著提升 FID（8.131→7.468）
2. **Low-level 场景提升更显著**：眼动提供的细粒度意图信息对精确原子运动生成更有帮助
3. **眼-体协调叙述比人工标注更具预测力**：动作预测中，GazeInterpreter 叙述的 Action F1 超过人工标注的 Nymeria 数据
4. **符号化中间层是关键**：直接让 LLM 处理原始数值信号会导致显著退化
5. **滑动窗口 W=2 最优**：增大窗口带来的边际收益递减，反而引入冗余噪声

## 亮点与洞察

- **开辟了全新研究方向**：首次系统性地将眼动注视解析与身体运动叙述相整合，揭示了眼动对行为理解的巨大潜力
- **数值→符号→文本的分解策略**非常聪明：避免了 LLM 直接处理传感器数值的幻觉风险
- **自我纠正循环的多维度评估框架**可迁移到其他生成式任务
- **无需训练的纯推理框架**：基于 LLM 的 few-shot + 多阶段推理，无需昂贵的专用模型训练
- 在运动生成、动作预测、行为摘要三个任务上一致性地展示了优势

## 局限与展望

- **仅在 Nymeria 单一数据集上验证**：目前是唯一同时包含眼动和运动标注的公开数据集，泛化性受限
- **推理成本较高**：三阶段 LLM 推理 + 自我纠正循环需要多次 LLM 调用
- **依赖预定义的阈值**：I-VT 分类器的速度阈值需要手工设定，不同场景可能需要调整
- **缺乏端到端联合优化**：符号解析、叙述生成、整合三个阶段完全独立
- 未探索第一人称视角图像/视频信号与注视的联合利用

## 相关工作与启发

- 与 MotionGPT/MotionLLM 的本质区别：不仅关注身体运动，更关注眼动-身体的协同
- I-VT 算法在经典眼动分析中已被广泛使用，本文首次将其与 LLM 结合
- 自我纠正循环的思想与 Constitutional AI 中的自我修正类似
- 对具身智能的启发：机器人理解人类意图时，眼动信号可能比肢体运动更早、更直接地揭示目标

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 开创性地将眼动解析与 LLM 行为理解结合，开辟新方向
- 实验充分度: ⭐⭐⭐⭐ — 运动生成+下游任务+完整消融，但仅单一数据集
- 写作质量: ⭐⭐⭐⭐ — 框架描述清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 揭示眼动在行为理解中的巨大潜力，具有长期影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](../../CVPR2025/human_understanding/ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)
- [\[CVPR 2025\] Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](../../CVPR2025/human_understanding/enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)
- [\[CVPR 2026\] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body](../../CVPR2026/human_understanding/vibes_a_conversational_agent_with_behaviorally_intelligent_3d_virtual_body.md)
- [\[ECCV 2024\] De-confounded Gaze Estimation](../../ECCV2024/human_understanding/de-confounded_gaze_estimation.md)
- [\[ICCV 2025\] Multi-view Gaze Target Estimation](../../ICCV2025/human_understanding/multi-view_gaze_target_estimation.md)

</div>

<!-- RELATED:END -->
