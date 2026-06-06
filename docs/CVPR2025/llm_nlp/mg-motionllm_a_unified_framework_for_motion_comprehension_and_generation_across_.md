---
title: >-
  [论文解读] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities
description: >-
  [CVPR 2025][LLM/NLP][动作理解] MG-MotionLLM 提出了一个统一的多粒度动作-语言模型，通过 Motion VQ-VAE + T5 语言模型的架构和精心设计的多粒度协同预训练方案（含 28 种任务），同时支持粗粒度和细粒度的动作理解与生成…
tags:
  - "CVPR 2025"
  - "LLM/NLP"
  - "动作理解"
  - "多粒度"
  - "大语言模型"
  - "动作生成"
  - "细粒度编辑"
---

# MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities

**会议**: CVPR 2025  
**arXiv**: [2504.02478](https://arxiv.org/abs/2504.02478)  
**代码**: [https://github.com/CVI-SZU/MG-MotionLLM](https://github.com/CVI-SZU/MG-MotionLLM)  
**领域**: LLM/NLP  
**关键词**: 动作理解, 多粒度, 大语言模型, 动作生成, 细粒度编辑

## 一句话总结
MG-MotionLLM 提出了一个统一的多粒度动作-语言模型，通过 Motion VQ-VAE + T5 语言模型的架构和精心设计的多粒度协同预训练方案（含 28 种任务），同时支持粗粒度和细粒度的动作理解与生成，在经典任务上达到 SOTA 的同时开启了细粒度动作编辑等新应用。

## 研究背景与动机

1. **领域现状**：现有的动作感知大语言模型（如 MotionGPT、MotionLLM）已展示了统一动作理解与生成的潜力，但它们主要聚焦于粗粒度的动作-文本建模——文本通常只用几个词描述整段动作的整体语义。

2. **现有痛点**：粗粒度的描述无法处理细粒度的动作相关任务，比如理解和控制特定身体部位的运动。现有尝试引入详细描述的方法（如 SemanticBoost、MotionScript）仅关注了细粒度生成，未能整合细粒度理解。

3. **核心矛盾**：详细描述信息量巨大（部分超过 1000 个 token），直接与简洁的运动 token（不超过 50 个）建立对应关系非常困难。作者实验发现，直接用粗+细描述指导生成，Top-3 检索准确率反而从 77.3% 降到 75.0%。

4. **本文目标** 如何在一个统一模型内，同时支持多粒度的动作理解和生成，并让粗粒度和细粒度任务互相增强。

5. **切入角度**：从短动作片段（snippet）的细粒度描述入手，先建立局部的动作-详细文本关系，再扩展到全局。设计辅助任务（如时序定位、片段描述）让不同粒度任务相互促进。

6. **核心 idea**：通过 28 种不同粒度的运动相关任务的协同预训练，让粗粒度任务帮助细粒度任务捕获语义，细粒度任务反过来增强粗粒度任务的细节理解。

## 方法详解

### 整体框架
MG-MotionLLM 由两个核心组件组成：(1) Motion VQ-VAE 负责将原始动作数据编码为离散 motion token，以及将 token 解码回动作序列；(2) 基于 T5 的动作感知语言模型，通过扩展词表将 motion token 和文本 token 统一建模。输入是包含不同粒度文本和/或动作 token 的指令序列，输出是对应的文本或动作 token 序列。

### 关键设计

1. **Motion VQ-VAE（动作离散化）**:

    - 功能：将连续的 T 帧动作序列 $\bm{M}$ 编码为 $T/l$ 个离散 token
    - 核心思路：使用编码器将动作映射到潜在空间，通过 codebook 量化为离散 token，再用解码器重建动作。训练时使用重建损失、embedding 损失和 commitment 损失（$\mathcal{L}_{\text{VQVAE}} = \|M - \hat{M}\|_2 + \|SG(Z) - \hat{Z}\|_2 + \beta\|Z - SG(\hat{Z})\|_2$）。训练完成后冻结参数。
    - 设计动机：离散化使得动作可以像词汇一样被 LLM 处理，实现动作和语言的统一建模。沿用 T2M-GPT 的设计保证了公平比较。

2. **动作感知语言模型（统一词表设计）**:

    - 功能：将动作理解和生成统一为一个 seq2seq 问题
    - 核心思路：扩展 T5 的文本词表 $V_t$ 以包含动作词表 $V_m$（codebook 索引）和特殊 token $V_s$（如 `<Motion Tokens>`、`<SEP>`、`<Motionless>`）。所有任务都被格式化为"输入 token 序列 → 输出 token 序列"的形式，用交叉熵损失训练 $\mathcal{L}_{CE} = -\sum \log P(v^i_{out} | X_{in}, v^j_{out}, \theta)$。
    - 设计动机：使用 T5 的 encoder-decoder 架构作为条件生成模型，天然适合将多模态任务统一为文本生成任务。

3. **多粒度协同预训练（Granularity-Synergy Pre-training）**:

    - 功能：有效建立动作与不同粒度文本的对应关系
    - 核心思路：设计了 28 种任务（12 种经典粗粒度 + 16 种新提出的细粒度），涵盖三类信息（文本描述、时间信息、动作数据）的各种组合。关键创新包括：(a) 从短动作片段和对应的详细描述入手，先建立局部关系再推广到全局；(b) 引入时序定位任务（根据详细文本定位动作片段的时间边界）；(c) 排除仅含细粒度描述的动作生成任务（因为不同动作可能共享相同的细粒度描述，如"走路"和"跑步"都涉及腿部前后交替运动）。
    - 设计动机：直接训练细粒度任务效果不佳（信息过多导致全局语义丢失），需要粗粒度任务辅助语义捕获，同时细粒度任务增强粗粒度任务的细节理解，形成互利关系。

### 损失函数 / 训练策略
采用两阶段训练：
- **阶段一（Granularity-Synergy Pre-training）**：用 28 种任务联合训练，学习率 $2 \times 10^{-4}$，batch size 16，训练 300K 迭代
- **阶段二（Task-Specific Instruction Tuning）**：针对特定任务继续微调，学习率 $10^{-4}$，训练 300K 迭代
- 两阶段均使用 AdamW 优化器，在单块 A100 80G GPU 上完成

## 实验关键数据

### 主实验

**Text-to-Motion（HumanML3D）**：

| 方法 | 类型 | R-Top3↑ | FID↓ | MM-Dist↓ | Diversity↑ |
|------|------|---------|------|----------|------------|
| MoMask (CVPR'24) | 仅生成 | 0.807 | **0.045** | 2.958 | - |
| MotionGPT (NeurIPS'23) | 统一 | 0.778 | 0.232 | 3.096 | 9.528 |
| **MG-MotionLLM** | 统一 | **0.802** | 0.303 | **2.952** | **9.960** |

**Motion-to-Text（HumanML3D）**：

| 方法 | R-Top1↑ | R-Top3↑ | MM-Dist↓ | BertScore↑ |
|------|---------|---------|----------|------------|
| MotionGPT | 0.543 | 0.827 | 2.821 | 32.4 |
| **MG-MotionLLM** | **0.592** | **0.866** | **2.581** | **36.7** |

### 消融实验

| 预训练粒度 | 指令微调 | T2M Top3↑ | M2T Top1↑ | M2DT BertScore↑ |
|-----------|---------|-----------|-----------|-----------------|
| 仅粗粒度 | ✗ | 0.725 | 0.431 | - |
| 仅细粒度 | ✗ | - | - | 43.1 |
| 粗+细 | ✗ | 0.767 | 0.514 | 47.7 |
| **粗+细** | **✓** | **0.802** | **0.592** | **52.3** |
| 无预训练 | ✓ | 0.773 | 0.516 | 50.5 |

### 关键发现
- **多粒度协同预训练的增益显著**：粗+细粒度联合预训练相比单独使用任一粒度，在所有任务上都有提升，验证了"不同粒度任务互相促进"的核心假设
- **指令微调是必要的**：预训练分配给每个任务的样本量有限（约 1/30），专项微调可进一步激发性能
- **模型规模影响细粒度任务更大**：T5-Large 在 Motion-to-Detailed Text 的 snippet 级别上比 T5-Small 提升约 4 个 Bleu@4 点，但在粗粒度任务上差异较小

## 亮点与洞察
- **多粒度互利训练范式**：粗粒度任务帮助细粒度任务捕获全局语义（因为细粒度描述太长导致信息过载），细粒度任务帮助粗粒度任务理解局部细节。这种"从简单到复杂"的设计思路非常巧妙，可以迁移到其他多尺度理解任务。
- **统一框架支持全新应用**：通过细粒度描述实现的动作编辑（时序编辑、空间编辑、时空编辑）是前所未有的——用户可以用单一模型完成：生成初步动作→获取细粒度描述→编辑描述→重新生成。这个交互范式很有启发性。
- **辅助任务设计**：时序定位任务（根据文本描述定位动作片段时间边界）既是一个有意义的独立任务，又能作为桥梁帮助建立动作和长文本之间的对应关系。

## 局限与展望
- **仅支持单人动作**：当前系统只处理单人运动序列，无法理解多人交互场景
- **依赖 T2M-GPT 的 VQ-VAE**：动作 tokenizer 的质量直接决定了上限，但没有探索更好的 tokenizer（如 RVQ）
- **细粒度描述生成的评估不够**：Motion-to-Detailed Text 主要用自动指标评估，缺乏人工评估和真实应用场景验证
- **FineMotion 数据集的模板化描述**可能限制了模型对自然语言细粒度描述的泛化能力
- 改进思路：引入 RVQ 多层 codebook 提升动作重建质量；扩展到多人交互场景；探索与视觉模态（视频）的联合训练

## 相关工作与启发
- **vs MotionGPT [Jiang et al., NeurIPS'23]**: MotionGPT 同样使用 T5 统一动作和语言，但仅处理粗粒度任务。MG-MotionLLM 在此基础上拓展到细粒度领域，且在粗粒度任务上也全面超越（R-Top3 提升 2.4%）。
- **vs FineMoGen [Zhang et al., NeurIPS'24]**: FineMoGen 尝试了细粒度动作生成但设计了专用网络，且只做生成不做理解。MG-MotionLLM 用统一模型同时处理理解和生成。
- **vs SemanticBoost/MotionScript**: 这些方法将身体部位运动翻译为预定义状态来增强生成，但仅关注生成不关注理解，且缺少时序信息。

## 评分
- 新颖性: ⭐⭐⭐⭐ 多粒度协同训练的 idea 新颖且有效，但整体框架（VQ-VAE + T5）是已有方法的组合
- 实验充分度: ⭐⭐⭐⭐ 消融实验全面验证了各组件贡献，但缺少人工评估和下游应用评测
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机推导自然，图表设计合理
- 价值: ⭐⭐⭐⭐ 开创了多粒度动作理解与生成的研究方向，细粒度动作编辑应用有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](../../ACL2026/llm_nlp/how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)
- [\[ACL 2025\] BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework](../../ACL2025/llm_nlp/bipro_zero-shot_chinese_poem_generation_via_block_inverse_prompting_constrained_.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](../../ICCV2025/llm_nlp/vim_versatile_interactive_motion_language_model.md)
- [\[ICLR 2026\] BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](../../ICLR2026/llm_nlp/bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)
- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](../../ACL2025/llm_nlp/segment_level_diffusion.md)

</div>

<!-- RELATED:END -->
