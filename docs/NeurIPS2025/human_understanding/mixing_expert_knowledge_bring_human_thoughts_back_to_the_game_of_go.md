---
title: >-
  [论文解读] Mixing Expert Knowledge: Bring Human Thoughts Back to the Game of Go
description: >-
  [NeurIPS 2025][人体理解][LLM] 提出 LoGos，通过混合领域专家数据（围棋）与通用长 CoT 推理数据进行冷启动微调 + GRPO 强化学习，使通用 LLM 在围棋中达到职业棋手水平的同时保持优秀的通用推理能力。
tags:
  - NeurIPS 2025
  - 人体理解
  - LLM
  - 围棋
  - 领域专家知识
  - 强化学习
  - GRPO
---

# Mixing Expert Knowledge: Bring Human Thoughts Back to the Game of Go

**会议**: NeurIPS 2025  
**arXiv**: [2601.16447](https://arxiv.org/abs/2601.16447)  
**代码**: [GitHub](https://github.com/Entarochuan/LoGos) (有)  
**领域**: Human Understanding / LLM Reasoning  
**关键词**: LLM, 围棋, 领域专家知识, 强化学习, GRPO

## 一句话总结

提出 LoGos，通过混合领域专家数据（围棋）与通用长 CoT 推理数据进行冷启动微调 + GRPO 强化学习，使通用 LLM 在围棋中达到职业棋手水平的同时保持优秀的通用推理能力。

## 研究背景与动机

**领域现状**: AlphaGo/AlphaZero 等专用 AI 系统早已在围棋领域超越人类，但通用大语言模型（LLM）在围棋任务上表现极差，甚至不如业余初学者。

**现有痛点**: 围棋等专业领域缺乏大规模自然语言推理语料（不像数学/编程有丰富的人类推理过程），直接蒸馏或预训练方法效果有限。

**核心矛盾**: 通用 LLM 有强大的推理泛化能力，但缺乏领域专业知识；而专用模型（如 KataGo）虽能力强大，但只能输出结构化预测，不具备自然语言推理能力。

**本文目标**: 如何在通用 LLM 中注入围棋等专业领域的专家级能力，同时保持通用推理能力。

**切入角度**: 利用围棋领域现有的大量结构化专业数据（棋谱 + KataGo 标注），通过启发式规则构建合成训练数据，再用 RL 自探索对齐推理与专业知识。

**核心 idea**: 混合领域专家合成数据 + 通用 CoT 数据做冷启动，然后用 GRPO 让模型自我探索出自然语言推理策略。

## 方法详解

### 整体框架

系统分为三个阶段：(1) 数据构建——收集围棋棋谱并用 KataGo 标注，通过启发式模板构建合成数据集；(2) 混合冷启动——将围棋专业数据与通用长 CoT 推理数据混合微调；(3) GRPO 强化学习——用分段奖励函数引导模型自我探索围棋推理策略。

### 关键设计

1. **围棋建模方式**: 将围棋对局序列化为 move list（如 X-D4, O-Q16...），每步用字母+数字坐标表示，X/O 区分黑白，模型预测下一步落子位置 $x_{k+1} = \pi_\theta(x_1, x_2, \ldots, x_k)$。

2. **专家级围棋数据集构建**:

    - **下一步预测数据集**（1000万级）: 从500万+职业/顶级业余棋谱中采样棋局状态，用 KataGo 标注每个状态的 top-10 候选落子及其胜率/后续变化，再通过启发式模板构建为四部分结构：确认执棋方 → 分析候选走法 → 总结选择最优 → 结构化输出。
    - **解说数据集**（10万级）: 收集开源围棋解说数据，处理为"棋局状态→解说评论"的训练对。

3. **混合冷启动微调**: 将围棋专业数据集与通用长 CoT 推理数据（Openthoughts-114K、NuminaMath-QwQ-CoT-5M、OpenCodeReasoning 等数学/代码推理数据）混合，对 Qwen2.5 基座模型进行微调。模型由此同时获得围棋领域知识注入和长 CoT 推理格式冷启动。

4. **GRPO 自探索强化学习**: 基于冷启动模型，用设计好的查询和奖励函数鼓励模型在围棋下一步预测任务上自我探索长 CoT 推理策略。关键发现是模型会自发地将从 CoT 数据中学到的推理能力迁移到围棋任务。

### 损失函数 / 训练策略

**分段奖励函数**: 根据模型预测落子在 KataGo top-10 中的排名分层给予奖励，并加入胜率预测准确度的额外奖励：

$$r_i = \begin{cases} 1 - \alpha_1 \cdot \frac{\beta_1|(\hat{w}_i - w_i)|}{1 + \beta_1|(\hat{w}_i - w_i)|} & \text{rank}(i) = 1 \\ c_1 - \alpha_1 \cdot (\text{胜率项}) - \alpha_2 \cdot (\text{排名惩罚项}) & \text{rank}(i) \in [2,3] \\ c_2 - \ldots & \text{rank}(i) \in [4,10] \\ c_3 - \alpha_1 - \alpha_2 & \text{rank}(i) \notin [1,10] \land \text{格式正确} \\ 0 & \text{其他} \end{cases}$$

参数设定: $c_1=0.8, c_2=0.6, c_3=0.4$（分层奖励）, $\alpha_1=0.1, \alpha_2=0.2, \beta_1=\beta_2=10$（胜率预测和排名内精细奖励）。使用 GRPO 目标函数进行优化，KL 系数设为 $5 \times 10^{-4}$。

## 实验关键数据

### 主实验

| 模型 | KataGo-Bench | GPQA Diamond | AIME | MATH | LiveCodeBench |
|------|-------------|-------------|------|------|--------------|
| DeepSeek-R1 | 17.6 | 69.7 | 86.7 | 97.6 | 83.8 |
| Claude3.7-Sonnet | 34.3 | 67.7 | 30.0 | 79.8 | 63.2 |
| DS-R1-Distill-7B | 0.6 | 41.4 | 33.3 | 88.2 | 20.4 |
| **LoGos(7B)** | **88.1** | 37.9 | 40.0 | 93.2 | 23.4 |
| DS-R1-Distill-32B | 4.7 | 56.1 | 46.7 | 94.5 | 36.5 |
| **LoGos(32B)** | **88.6** | 63.6 | 56.7 | 96.5 | 50.9 |
| KataGo-HumanSL-9d | 87.8 | - | - | - | - |

LoGos 在围棋基准上达到 88.6%，超越模拟顶级人类棋手的 KataGo-HumanSL-9d（87.8%），是 Claude3.7-Sonnet（34.3%）的约 2.6 倍。

### 消融实验

| 实验设置 | 结果 |
|---------|------|
| 无冷启动直接 RL | 性能上限显著低于初学者水平（<67.4%） |
| 冷启动用直接预测替代启发式模板 | 性能上限 <50%（vs 88%） |
| 奖励函数: 仅 top-1 | 性能稍低（稀疏奖励） |
| 奖励函数: 仅 top-3 不分层 | 性能稍低 |
| 围棋数据 2 epoch vs 1 epoch | 1 epoch 最终 RL 上限更高 |
| 混合不同量围棋数据(500K-10M) | 大量围棋数据不会显著损害通用能力 |

### 关键发现

- **冷启动必不可少**: 不进行冷启动直接 RL，模型无法达到初学者水平。即使 DeepSeek-R1 蒸馏模型也无法仅通过 RL 自探索达标。
- **启发式模板关键**: 用直接预测替代启发式规则构建的模板数据，RL 后性能上限骤降至 <50%。
- **Context Curse**: 随着棋局序列变长，模型对局面的理解能力下降。引入 2D 棋盘状态渲染（19×19 矩阵）可有效缓解此问题，200 步后仍保持高准确率。
- 人工评估显示 96.5% 的落子预测正确，55.6% 的解释也是正确的。

## 亮点与洞察

- **开创性工作**: 首个在围棋上达到职业棋手水平的通用 LLM，证明了通用推理能力可以与领域专家知识有效整合。
- **方法论通用性**: 提出的"混合专家知识"范式（结构化专业数据 → 启发式模板合成 → 混合冷启动 → RL自探索）可推广到其他领域专业数据稀缺但有结构化数据的场景。
- **惊人的自发推理迁移**: 冷启动后模型自发地将数学/代码推理中学到的 CoT 模式泛化到围棋，这个发现非常有意义。
- **围棋数据集贡献**: 首个面向 LLM 训练的大规模围棋数据集和评估基准。

## 局限与展望

- 方法依赖可扩展的专业知识来源（如 KataGo），需要设计领域特定的启发式规则和合成数据模板。
- RL 自探索训练速度较慢，相比传统围棋 AI（AlphaGo），LLM 在 RL 阶段训练的棋局状态数量少得多。
- autoregressive 架构可能限制了 LLM 在实时推理任务中的训练效率。
- 围棋术语使用偶有错误（>600 个专业术语），尤其在开局阶段。
- 部分解释模糊性较大，提供的有效信息有限。

## 相关工作与启发

- **LLM 游戏玩家**: ChessGPT、PokerGPT 等探索 LLM 玩棋类游戏，但围棋复杂度远超关于国际象棋($b \approx 250$ vs $b \approx 35$)。
- **领域专家 AI**: AlphaGeometry、DeepSeek-Prover 等利用工具和启发式规则让 LLM 获得领域专家能力，本文思路类似但面向围棋这一更开放的领域。
- **启发**: 该方法为"如何将 RL 与特定领域结构化知识整合到通用 LLM"提供了一个可复制的范例。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次让通用 LLM 达到围棋职业级水平，混合专家知识的训练范式新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 消融实验全面，包含冷启动必要性、模板设计、奖励函数设计、数据量影响等
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，叙述有说服力
- 价值: ⭐⭐⭐⭐⭐ 方法论贡献大，对领域专业知识注入通用 LLM 的研究具有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning](k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso.md)
- [If Open Source Is to Win, It Must Go Public](../../ICML2025/human_understanding/if_open_source_is_to_win_it_must_go_public.md)
- [MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](../../ICCV2025/human_understanding/mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)
- [Distributionally Robust Online Markov Game with Linear Function Approximation](../../AAAI2026/human_understanding/distributionally_robust_online_markov_game_with_linear_function_approximation.md)
- [Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](../../ICCV2025/human_understanding/weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)

<!-- RELATED:END -->
