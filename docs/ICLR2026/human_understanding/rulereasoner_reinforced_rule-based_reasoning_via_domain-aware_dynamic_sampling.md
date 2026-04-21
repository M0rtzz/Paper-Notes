---
title: >-
  [论文解读] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling
description: >-
  [人体理解] RuleReasoner 通过构建多样化的规则推理数据集 RuleCollection-32K 和提出域感知动态采样（Dads）策略，在 RLVR 框架下训练 8B 模型，在域内推理任务上比 OpenAI-o1 高 4.1%，在域外任务上高 10.4%，同时训练效率提升 ~1.4×。
tags:
  - 人体理解
---

# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling

- **会议**: ICLR2026
- **arXiv**: [2506.08672](https://arxiv.org/abs/2506.08672)
- **代码**: 待公开
- **领域**: LLM 推理 / 强化学习
- **关键词**: rule-based reasoning, RLVR, dynamic sampling, GRPO, domain reweighting, OOD generalization

## 一句话总结

RuleReasoner 通过构建多样化的规则推理数据集 RuleCollection-32K 和提出域感知动态采样（Dads）策略，在 RLVR 框架下训练 8B 模型，在域内推理任务上比 OpenAI-o1 高 4.1%，在域外任务上高 10.4%，同时训练效率提升 ~1.4×。

## 研究背景与动机

**领域现状**: 基于规则的推理（rule-based reasoning）是 AI 推理的基础能力，涵盖法律、数学、医疗诊断等领域。大型推理模型（LRM）通过强化学习（RL）获得了长链思考能力，但在真实场景中仍面临规则格式多样、类型复杂、组合爆炸等挑战。

**痛点**: (1) 传统方法依赖模型规模扩大或从更强模型蒸馏，成本高且不可持续；(2) 随着上下文窗口扩大，模型出现"lost in the middle"现象，难以关注相关的规则和事实；(3) 现有 RLVR 方法对多域训练数据的采样策略粗糙——静态混合导致域间不平衡，过度优化简单域而欠优化困难域。

**核心矛盾**: RLVR 在数学/代码推理上的成功尚未迁移到规则推理领域——缺乏高质量多样化的训练数据，且多域训练的数据调度问题未被充分研究。

**目标**: 训练一个小型（4B/8B）但在规则推理上超越前沿 LRM（o1, R1）的专用推理模型，同时提高训练效率。

**切入角度**: 从**数据**和**采样策略**两个维度同时改进 RLVR：(1) 构建覆盖 8 类规则推理任务的 RuleCollection-32K；(2) 设计基于历史奖励的动态域采样算法 Dads，自动调度训练批次中各域的比例。

**核心 idea**: 用域感知动态采样（Dads）替代静态数据混合——每个训练步骤根据各域的历史奖励计算"欠优化程度"，动态提高欠优化域的采样权重，实现自适应在线数据调度。

## 方法详解

### 整体框架

RuleReasoner 训练流程：
1. 在 RuleCollection-32K 上初始化标准 RLVR 训练
2. 每个训练步骤：Rollout → 计算域平均奖励 → 指数加权移动平均更新域奖励估计 → softmax 计算域权重 → 按权重重采样下一批次
3. 使用 GRPO 变体进行策略优化（去除 KL 项和熵奖励）
4. 规则顺序随机打乱防止记忆化

### 关键设计

#### 1. 域感知动态采样（Dads）

**功能**: 在 RLVR 训练的每个步骤自动调整各推理域的采样概率，优先采样欠优化域。

**核心思路**: 对域 $d_i$，维护指数加权移动平均奖励 $\widetilde{r}_{s,d_i}$，计算其欠优化程度 $v_{s,d_i} = 1 - \widetilde{r}_{s,d_i}$（相对于目标奖励 1 的差距），然后通过 softmax 转化为采样权重：

$$\widetilde{r}_{s,d_i} = \alpha \widetilde{r}_{s-1,d_i} + (1-\alpha) \bar{r}_{s,d_i}$$

$$w_{s,d_i} = \frac{\exp(v_{s,d_i}/\tau) + \epsilon}{\sum_{j=1}^n [\exp(v_{s,d_j}/\tau) + \epsilon]}$$

其中 $\alpha=0.5$ 为平滑因子避免奖励估计波动，$\tau=0.5$ 控制权重锐度，$\epsilon=0.1$ 保证每个域的最小采样概率。

**设计动机**: 静态数据混合无法适应训练动态——简单域（如 ProntoQA）很快收敛，继续采样浪费计算资源；困难域（如 AR-LSAT）需要更多训练但采样不足。Dads 像一个**在线调度器**，自动将计算资源从已收敛域转移到欠优化域。

#### 2. RuleCollection-32K 数据集

**功能**: 构建覆盖多种规则格式、推理类型和复杂度的训练数据集。

**核心思路**: 遵循五项原则构建：
- **变化深度**: 0-7 跳推理，支持从简单到复杂的课程学习
- **不同格式**: 显式规则（前提/约束）和隐式规则（原则/上下文）
- **多种推理规则**: 演绎、归纳、分析推理
- **上下文依赖**: 规则需结合具体问题自适应应用，不能仅靠背诵
- **鲁棒评测**: 优先使用布尔/选择题，便于精确评测

涵盖 8 个任务：Clutrr（归纳）、ProntoQA/ProofWriter（演绎）、FOLIO/LogicNLI（一阶逻辑）、AR-LSAT/Logical Deduction/LogiQA（其他）。

#### 3. 训练正则化

**功能**: 防止模型识别特定数据集或记忆规则模式。

**核心思路**: 三项正则化措施：
- **去除熵奖励**: 避免无冷启动 bootstrap 时的熵爆炸
- **去除 KL 散度项**: 规则奖励函数消除了分布偏移顾虑，去除 KL 可节省计算并鼓励探索
- **规则顺序打乱**: 每个训练样本的上下文规则顺序随机打乱，防止位置记忆

### 损失函数

使用基于精确匹配的**规则奖励函数**：

$$\mathcal{R}_{\text{EM}}(\hat{y}, y) = \begin{cases} 1 & \text{is\_equivalent}(\hat{y}, y) \\ -1 & \text{otherwise} \end{cases}$$

策略优化采用 GRPO 目标，去除 KL 项和熵奖励，仅保留裁剪后的策略梯度。严格在线策略——每次 rollout 后仅更新一次梯度。

## 实验关键数据

### 主实验：域内 8 任务 Pass@1 对比

| 方法 | Clutrr | ProntoQA | ProofWriter | FOLIO | LogicNLI | AR-LSAT | Log.Ded. | LogiQA | **Avg** |
|------|--------|----------|-------------|-------|----------|---------|----------|--------|---------|
| OpenAI o1 | 52.2 | 91.0 | 91.0 | 77.0 | 60.0 | 98.0 | 88.0 | 82.1 | 79.9 |
| Claude-3.7 | 65.7 | 92.8 | 90.0 | 74.7 | 58.0 | 76.2 | 97.0 | 81.5 | 79.5 |
| DeepSeek-R1 | 71.6 | 40.0 | 27.0 | 72.7 | 49.0 | 89.7 | 98.3 | 85.0 | 66.7 |
| DAPO | 86.5 | 96.0 | 94.8 | 80.9 | 65.8 | 40.0 | 95.3 | 74.6 | 79.2 |
| AdaRFT | 92.5 | 96.0 | 97.4 | 81.8 | 64.4 | 44.6 | 96.6 | 80.5 | 81.7 |
| **RuleReasoner-8B** | **95.5** | **96.4** | **97.0** | **84.7** | 70.4 | 46.8 | 98.3 | 83.5 | **84.0** |

> RuleReasoner-8B 以 84.0% 平均准确率超越 OpenAI-o1（79.9%）4.1 个百分点，且各域性能方差最低。

### 消融实验：域外 OOD 泛化

| 方法 | BBH | BBEH | ProverQA | **OOD Avg** |
|------|-----|------|----------|------------|
| Qwen3-8B (base) | 22.9 | 13.0 | 15.3 | — |
| SFT w/ Long CoT | 31.3 | 28.0 | 43.8 | 34.4 |
| GRPO | 35.5 | 24.3 | 34.1 | 31.3 |
| DAPO | 39.8 | 27.3 | 42.8 | 36.6 |
| OpenAI o1 | 46.4 | 33.5 | 52.5 | 44.1 |
| **RuleReasoner-8B** | **52.3** | **45.8** | **65.4** | **54.5** |
| **RuleReasoner-4B** | — | — | — | **54.5** (Δ+7.3 vs o1) |

> RuleReasoner-8B 在三个 OOD 基准上平均超越 o1 达 10.4 个百分点。即使 4B 版本也达到 78.3% 三基准平均。

### 关键发现

1. **Dads 优于静态课程学习**: 相比 data-balance RL（79.1%）和 easy-to-hard RL（80.4%），Dads（84.0%）在 ID 任务上高出 3-5 个百分点——在线调度远优于静态排列
2. **训练效率**: RuleReasoner 达到 DAPO 同等 OOD 性能需要少 ~72 步（约 1.4× 加速），且无需额外 rollout 计算
3. **SFT vs RLVR**: SFT 在 ID 上接近 RLVR（81.9 vs 84.0），但 OOD 上差距巨大（34.4 vs 54.5），确认 **RL 泛化、SFT 记忆**
4. **元自省能力涌现**: 训练后模型展现出自我验证和逻辑一致性检查的能力——在未见过的规则上也能自纠错

## 亮点与洞察

- Dads 的设计精妙简约：仅用历史奖励估计域权重，无需代理模型或人类先验，可作为 RLVR 的通用数据调度插件
- 8B 模型超越 o1/R1 等前沿推理模型，说明规则推理场景下**专用数据+智能调度**比模型规模更重要
- RuleCollection-32K 的构建原则（变化深度、多格式、上下文依赖）值得其他推理数据集借鉴
- 实验中 DeepSeek-R1 在部分任务上崩溃（ProntoQA 40.0%、ProofWriter 27.0%），暴露了通用推理模型在结构化规则推理上的脆弱性

## 局限性

- 训练数据受限于当前可获得的规则推理任务，可能未覆盖自然语言中所有规则格式和复杂度
- 规则过滤质量有限——含噪或冗余规则可能影响推理质量
- 未在 >8B 模型上验证，大规模模型可能获益更多但受计算限制
- 评测以精确匹配为主，对需要自由文本输出的规则应用场景评估不足

## 相关工作与启发

- **Logic-RL** (Xie et al., 2025): 逻辑推理上的 RLVR，但属于"无规则"推理，与本文"规则给定"设定不同
- **DAPO** (Yu et al., 2025): 通过过采样+过滤提升 RLVR 效率，但未做细粒度域调度
- **AdaRFT** (Shi et al., 2025): 课程学习式采样但依赖人类先验或模型成功率估计，Dads 完全自适应
- **GRPO** (Shao et al., 2024): 基础策略优化算法，RuleReasoner 在此基础上加入域感知采样
- **启发**: "域感知动态采样" 的思路可推广到数学推理、代码生成等任何多域 RLVR 训练场景

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — Dads 作为 RLVR 数据调度的通用方法有较强新意，但核心是 softmax 重加权并非全新范式
- **实验**: ⭐⭐⭐⭐⭐ — 基线覆盖前沿 LRM + SFT + 多种 RLVR + 课程学习，消融详尽，5 次运行报告均值和标准差
- **实用性**: ⭐⭐⭐⭐ — 方法简洁高效，可直接集成到现有 RLVR 流程
- **写作**: ⭐⭐⭐⭐ — 算法伪代码清晰，但表格较多导致部分内容密集

<!-- RELATED:START -->

## 相关论文

- [ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)
- [SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/human_understanding/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](../../ICCV2025/human_understanding/wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)
- [SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](../../ICCV2025/human_understanding/semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)

<!-- RELATED:END -->
