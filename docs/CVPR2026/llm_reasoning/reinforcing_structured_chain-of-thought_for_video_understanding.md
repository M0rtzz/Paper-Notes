---
title: >-
  [论文解读] Reinforcing Structured Chain-of-Thought for Video Understanding
description: >-
  [CVPR 2026][LLM推理][视频QA] 提出 SDRL（Summary-Driven Reinforcement Learning），一种无需 SFT 的单阶段 RL 框架，通过结构化 CoT（Summarize→Think→Answer）和两个自监督机制（CVK 和 DVR）增强视频时序推理，在 7 个 VideoQA 基准上达到 SOTA。
tags:
  - CVPR 2026
  - LLM推理
  - 视频QA
  - 强化学习
  - 结构化CoT
  - GRPO
  - 时序推理
---

# Reinforcing Structured Chain-of-Thought for Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2603.25942](https://arxiv.org/abs/2603.25942)  
**代码**: 无  
**领域**: 视频理解 / 视频推理  
**关键词**: 视频QA, 强化学习, 结构化CoT, GRPO, 时序推理

## 一句话总结

提出 SDRL（Summary-Driven Reinforcement Learning），一种无需 SFT 的单阶段 RL 框架，通过结构化 CoT（Summarize→Think→Answer）和两个自监督机制（CVK 和 DVR）增强视频时序推理，在 7 个 VideoQA 基准上达到 SOTA。

## 研究背景与动机

多模态大语言模型（MLLMs）在视频理解中展现了潜力，但仍面临两个核心挑战：

**思维漂移（Thinking Drift）**：现有 RL 方法（如 GRPO）仅依赖最终答案的奖励信号来优化，中间推理步骤不受约束。这导致模型生成冗长或与视觉证据无关的推理内容，严重影响结果稳定性。

**时序理解薄弱**：MLLMs 通常将视频表示为堆叠或平均的帧嵌入，忽略了细粒度时序依赖关系，导致在时序敏感的 VideoQA 任务上表现较差。

现有解决方案的局限：
- **纯 RL 方法**：推理不受约束，不稳定
- **SFT+RL 方法**：需要昂贵的 CoT 标注，多阶段训练复杂，且 SFT 的逐 token 模仿会限制泛化能力，可能导致过拟合

SDRL 的核心创新在于**将结构化 CoT 直接集成到 RL 目标中**，通过自监督方式约束推理过程，无需额外的 SFT 阶段或 CoT 标注数据。

## 方法详解

### 整体框架

SDRL 采用 Qwen2.5-VL-7B 作为骨干，输入（视频, 问题）后要求模型生成结构化输出：
- **Summary 段**（`<summary>`）：提取关键动作及其时序顺序
- **Think 段**（`<think>`）：基于摘要进行逻辑推理
- **Answer 段**：给出最终答案

对每个输入采样 G 组输出，通过 token 级权重（CVK+DVR）和标准奖励（准确率+格式）计算组优势值来优化策略。

### 关键设计

1. **结构化 CoT（Summarize→Think→Answer）**：

    - 通过实证分析验证：正确预测的 CoT 与 ground-truth CoT 之间具有更高的 BLEU 和 sBERT 相似度
    - 有效的 CoT 需要捕捉两个核心要素：(1) 关键动作/事件 (2) 事件的时序顺序
    - Summary 段是自上而下推理的锚点，为后续 Think 段提供事实基础
    - 设计动机：将摘要作为结构化锚点，可以从根本上解决思维漂移问题

2. **视觉知识一致性（CVK，Consistency of Vision Knowledge）**：

    - 核心假设：视频的视觉内容是固定和事实性的，因此同一输入的多次采样摘要应该具有高度语义一致性
    - **GT 监督模式**：当有 ground-truth 摘要时，用 sBERT+BLEU 的组合相似度度量与 GT 的对齐程度，作为额外奖励
    - **自监督模式**：从正确预测中动态导出一致性锚点 $S^C$（位置级中心），通过 KL 散度度量不一致性，转化为 Summary Token Weight $\omega_t^S$
    - KL 散度越大→一致性越低→权重越小，鼓励模型学习稳定一致的摘要部分
    - 设计动机：不直接监督摘要内容，而是通过组内一致性间接约束事实忠实度

3. **推理动态多样性（DVR，Dynamic Variety of Reasoning）**：

    - 在 Think 段鼓励推理路径的多样性，通过 token 分布的熵来度量
    - 高熵 token → 更高的多样性权重 $\omega_{g,t}^d$
    - **动态调制**：根据组准确率 $\mathcal{A}$ 调整多样性激励
        - 低准确率组：$(1-\mathcal{A})$ 大，强化探索
        - 高准确率组：$(1-\mathcal{A})$ 小，保持稳定推理路径
    - 设计动机：避免在模型已经有信心的情况下强制多样性引入噪声，同时在不确定时鼓励探索

4. **EventFlowQA 数据集构建**：

    - 专注于复杂动作序列和时序因果的 VideoQA 数据集
    - 53K 高质量 QA 对（50K 训练 + 3K 验证），覆盖 15 个时序维度
    - 作为所有消融实验的核心基准

### 损失函数 / 训练策略

结构化策略目标：$\mathcal{J}_{total}(\theta) = \mathcal{J}_{grpo}^{SCoT}(\theta) - \mathcal{J}_{reg}(\theta)$

Token 级权重：
$$W_{g,t} = \begin{cases} \omega_t^S & \text{(Summary 段, 一致性权重)} \\ \omega_{g,t}^{d'} & \text{(Think 段, 动态多样性权重)} \end{cases}$$

训练配置：
- 单阶段 RL（无 SFT），32 张 A100 GPU
- GRPO 组大小 G=8，共 1000 次 RL 迭代
- 16 帧均匀采样，分辨率 128×28×28
- 超参数：$\alpha=0.7$, $\beta=0.3$, $\gamma_1=1$, $\gamma_2=1$, $\lambda=0.5$, $\lambda'=0.7$

## 实验关键数据

### 主实验

在 7 个公开 VideoQA 基准上的表现（Accuracy %）：

| 数据集 | SDRL (Ours) | Video-R1 (SFT+RL) | VideoRFT (SFT+RL) | TW-GRPO (RL) | 提升 (vs best RL) |
|--------|-------------|--------------------|--------------------|--------------|-------------------|
| NExT-GQA | 79.3 | 74.3 | 75.1 | 76.1 | +3.2 |
| MMVU | 68.6 | 64.2 | 67.3 | 65.8 | +1.3 |
| VideoMMMU | 51.3 | 52.4 | 50.6 | - | +0.7 |
| VSIBench | 32.9/36.1† | 34.6 | 35.7 | - | +0.4† |
| MVBench | 64.2 | 62.7 | 61.4 | 63.3 | +0.9 |
| TempCompass | 74.4† | 72.6 | 73.1 | 73.3 | +1.1† |
| VideoMME | 54.7 | 57.4 | 58.1 | 55.1 | - |

注：† 表示在 EventFlowQA 上训练的变体（仅 Video-R1 数据量的 20%）。

### 消融实验

CVK 和 DVR 模块在 EventFlowQA 上的消融：

| 配置 | Accuracy | 说明 |
|------|----------|------|
| 原始 GRPO | 42.37 | 基线 |
| +sBERT (GT) | 43.85 | 语义一致性有帮助 |
| +BLEU (GT) | 46.32 | 词法一致性帮助更大 |
| +sBERT+BLEU (GT) | 48.56 | 组合最优 |
| +GT CVK + 静态 Entropy DVR | 50.09 | 多样性进一步提升 |
| +GT CVK + 动态 DVR (完整) | 52.22 | 动态调整最优 |
| 自监督 CVK | 54.28 | **自监督优于 GT 监督** |
| 自监督 CVK + 动态 DVR | **56.10** | 最佳配置 |

模型规模对监督方式的影响：

| 配置 | 3B 模型 | 7B 模型 |
|------|---------|---------|
| GT 监督提升 | +3.01 | +6.19 |
| 自监督提升 | +2.40 | **+11.91** |

### 关键发现

1. **自监督优于 GT 监督（7B）**：大模型从自监督一致性中获益更多（+11.91 vs +6.19），可能因为严格的 GT 对齐会抑制预训练语义先验，导致灾难性遗忘
2. **小模型更依赖 GT 指导**：3B 模型在 GT 监督下略优（+3.01 vs +2.40）
3. **Entropy 优于 KL 散度作为多样性度量**：Entropy 作为全局不确定性控制更能保持语义多样性，而 KL 散度的位置依赖对齐会抑制全局可变性
4. **动态多样性调制显著优于静态**：避免在高准确率组过度探索引入噪声
5. **仅用 20% 数据量即可达到竞争性能**：EventFlowQA 训练的 SDRL 在 TempCompass 上超越所有基线，展示了高数据效率

## 亮点与洞察

- **单阶段 RL 替代 SFT+RL 流水线**：通过结构化 CoT 和自监督约束，消除了对昂贵 CoT 标注和多阶段训练的需求，是一个优雅的简化
- **Summary 作为事实锚点**：将摘要定位在推理链的最前端，让事实提取先于逻辑推理，从根本上解决思维漂移
- **对齐与探索的平衡**：CVK 负责一致性/对齐，DVR 负责多样性/探索，两者通过 token 级权重在同一目标函数中统一
- **自监督一致性的意外发现**：大模型自监督效果反超 GT 监督，暗示过强的监督信号可能约束表达能力

## 局限与展望

1. 当前仅在 16 帧设置下实验，对于更长视频（如 64 帧或分钟级）的扩展性未知
2. Summary 段生成本身可能引入额外开销，对实时应用的影响需评估
3. EventFlowQA 数据集的构建细节在正文中较少，质量控制机制不够透明
4. 在 VideoMME 上未达到 SFT+RL 方法的最佳水平（54.7 vs 58.1），说明泛化性还有改进空间
5. 自监督一致性锚点依赖于正确预测的存在，在极低准确率场景下可能失效

## 相关工作与启发

- **GRPO/DAPO**：提供了 RL 优化的基础框架，SDRL 在此基础上引入结构化约束
- **Video-R1**：首次将 GRPO 引入视频理解，但依赖 SFT+RL 的两阶段流水线
- **GRPO-CARE**：组级一致性的思想与 CVK 相关，但未区分推理的不同段
- **Process Reward Models**：过程级监督的思路与 CVK/DVR 的 token 级权重设计有相通之处

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （结构化CoT+自监督RL的创新组合，单阶段流水线简洁有效）
- 实验充分度: ⭐⭐⭐⭐⭐ （7个基准、详尽消融、多尺度分析、可视化对比）
- 写作质量: ⭐⭐⭐⭐ （方法描述清晰但公式较多，数据集细节不足）
- 价值: ⭐⭐⭐⭐⭐ （为视频推理提供了更简洁高效的训练范式）

<!-- RELATED:START -->

## 相关论文

- [Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)
- [PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation](../../ICLR2026/llm_reasoning/prismaudio_decomposed_chain-of-thoughts_and_multi-dimensional_rewards_for_video-.md)
- [Video-T1: Test-Time Scaling for Video Generation](../../ICCV2025/llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)
- [VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](../../CVPR2025/llm_reasoning/videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)

<!-- RELATED:END -->
