---
title: >-
  [论文解读] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment
description: >-
  [CVPR 2026][终身模仿学习] 提出终身模仿学习框架，通过 Multimodal Latent Replay（在冻结编码器的潜空间中存储和回放紧凑多模态特征）和 Incremental Feature Adjustment（基于角距离的自适应间隔约束防止任务间表示漂移），在 LIBERO 基准上实现 AUC 提升 10-17 点、遗忘减少 65%。
tags:
  - CVPR 2026
  - 终身模仿学习
  - 多模态潜在回放
  - 增量特征调整
  - 灾难性遗忘
  - 机器人操作
---

# Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment

**会议**: CVPR 2026  
**arXiv**: [2603.10929](https://arxiv.org/abs/2603.10929)  
**代码**: https://github.com/yfqi/lifelong_mlr_ifa (有)  
**领域**: 强化学习  
**关键词**: 终身模仿学习, 多模态潜在回放, 增量特征调整, 灾难性遗忘, 机器人操作

## 一句话总结
提出终身模仿学习框架，通过 Multimodal Latent Replay（在冻结编码器的潜空间中存储和回放紧凑多模态特征）和 Incremental Feature Adjustment（基于角距离的自适应间隔约束防止任务间表示漂移），在 LIBERO 基准上实现 AUC 提升 10-17 点、遗忘减少 65%。

## 研究背景与动机
模仿学习（IL）让机器人通过观察人类示范学习行为，但现实环境是动态变化的，新的物体、目标和场景不断出现。标准 IL 假设固定任务集，无法处理持续到来的新任务。终身模仿学习（LIL）需要在不断获取新技能的同时保持已学行为，核心挑战是灾难性遗忘。

现有 LIL 方法的局限：(1) 经验回放方法（如 ER）存储原始轨迹数据，内存占用大，且对新旧任务相似度敏感；(2) 渐进式模型扩展方法（如 TAIL）需要测试时的任务标识符，限制实际应用；(3) 蒸馏方法（如 M2Distill）结构复杂。更关键的是，当新任务与旧任务在潜空间中表示重叠时，共享 embedding 空间内的干扰会恶化遗忘。

核心 idea：(1) 用冻结的预训练编码器提取多模态特征，在潜空间而非原始数据空间进行回放，大幅降低存储开销；(2) 通过角距离间隔约束，主动将新任务的表示推离旧任务的参考 embedding，保持任务间可区分性。

## 方法详解

### 整体框架
两阶段训练：(1) 多任务预训练阶段——联合训练所有模块（视觉/语言/状态编码器、FiLM 调制层、时序解码器、策略头）；(2) 终身学习阶段——冻结编码器，仅更新时序解码器和策略头，结合 MLR 和 IFA 在新任务上持续学习。

### 关键设计

1. **Multimodal Latent Replay (MLR)**:

    - 功能：在终身学习阶段，存储冻结编码器输出的多模态潜特征（而非原始图像和轨迹），学习新任务时与新数据混合训练
    - 核心思路：缓冲区 $\mathcal{B} = \{(\mathbf{H}_n, a_n)\}_{n=1}^{N_B}$，其中 $\mathbf{H}_n \in \mathbb{R}^{M \times L \times E}$ 是拼接了视觉、语言、状态的多模态潜特征序列。学习新任务 $T_k$ 时最小化 $\mathcal{L}_{BC}$ on $\mathcal{D}_k \cup \mathcal{B}$
    - 设计动机：原始轨迹（高维图像序列）存储开销巨大。通过冻结编码器，潜特征是稳定且紧凑的，每个任务仅需 ~37-188MB（取决于采样概率），远小于存储原始图像。冻结编码器还消除了对 PEFT（如 LoRA）的需求

2. **Incremental Feature Adjustment (IFA)**:

    - 功能：对新旧任务在潜空间中的表示施加角距离间隔约束，防止表示漂移导致的干扰
    - 核心思路：$\mathcal{L}_{IFA} = \frac{1}{|\mathcal{P}|}\sum_{(j,k) \in \mathcal{P}} \max(0, d(g_t(T_k), h^{(r)}(T_k)) - d(g_t(T_k), h^{(r)}(T_j)) + \delta)$，其中 $d(a,b) = \arccos(\frac{a^\top b}{\|a\|_2 \|b\|_2})$ 为角距离，间隔 $\delta = \alpha \cdot d(h^{(r)}(T_k), h^{(r)}(T_j))$ 自适应地随任务参考间距缩放
    - 设计动机：当新任务的全局潜表示 $g_t(T_k)$ 与旧任务的参考 $h^{(r)}(T_j)$ 过于接近时，共享时序解码器会混淆两个任务。IFA 产生排斥力将新任务推离旧任务，同时保持与自身参考的吸引力。自适应间隔避免了固定间隔对不同任务相似度的不适应

3. **Task Pair Selection 与 Reference Choice**:

    - 功能：选择哪些任务对需要施加 IFA 约束，以及用什么作为每个任务的参考 embedding
    - 核心思路：计算所有任务对在 agent-view 和 language 模态上的余弦相似度，仅对同时在两个模态 top 50% 相似且包含新旧任务的对施加 IFA。参考选用语言 embedding $h^{(l)}$——因为语言描述对任务具有稳定、固定的表示
    - 设计动机：不是所有任务对都需要分离——仅相似的任务对容易发生表示干扰，针对性约束更高效

### 损失函数 / 训练策略
终身学习阶段总损失：$\mathcal{L} = \mathcal{L}_{BC} + \lambda_{IFA} \mathcal{L}_{IFA}$，$\lambda_{IFA} = 0.1$。使用 AdamW 优化器，学习率 $10^{-4}$，batch size 10，训练 100 epoch。缓冲区大小等价于每任务 5 个演示。基座编码器为冻结的 CLIP。

## 实验关键数据

### 主实验

| 方法 | LIBERO-OBJECT AUC↑ | LIBERO-GOAL AUC↑ | LIBERO-50 AUC↑ | LIBERO-GOAL NBT↓ |
|------|------|------|------|------|
| Sequential | 30.0 | 23.0 | 14.0 | 70.0 |
| ER | 49.0 | 47.0 | 36.0 | 36.0 |
| LOTUS | 65.0 | 56.0 | 45.0 | 30.0 |
| M2Distill | 69.0 | 57.0 | NA | 20.0 |
| ISCIL | 66.3 | 60.5 | 37.7 | 19.4 |
| **MLR + IFA** | **79.4** | **77.2** | **56.1** | **6.9** |

### 消融实验

| 配置 | LIBERO-OBJ AUC | LIBERO-GOAL AUC | 说明 |
|------|---------------|----------------|------|
| MLR only | 77.6 | 74.6 | 仅潜在回放已大幅超越 SOTA |
| MLR + IFA | 79.4 | 77.2 | IFA 进一步改善 |
| 不同模态 pair: Lan+AV | 79.4 | 77.2 | 最佳组合 |
| 仅 Language | 77.3 | 71.8 | 不如 Lan+AV |
| Mean global 作为参考 | 75.7 | 70.5 | 不如 Language 参考 |

### 关键发现
- MLR 单独就已大幅超越所有 SOTA：LIBERO-OBJECT AUC 77.6 vs ISCIL 66.3，LIBERO-GOAL AUC 74.6 vs 60.5
- IFA 在所有数据集上进一步改善，在 LIBERO-GOAL 上 NBT 从 10.0 降至 6.9（遗忘减少 31%）
- 在最挑战的 LIBERO-50 上，MLR+IFA 的 NBT 为 8.6，远低于 LOTUS 的 43.0，表明方法在长任务序列上特别有效
- 语言 + agent-view 是 IFA 任务对选择的最佳模态组合

## 亮点与洞察
- "冻结编码器 + 潜在回放"的设计极其简洁高效——不需要 PEFT、不需要蒸馏、不需要开放词汇编码器，却超越了所有使用这些复杂技术的方法
- IFA 的自适应间隔 $\delta = \alpha \cdot d(h^{(r)}(T_k), h^{(r)}(T_j))$ 意味着任务越相似间隔越小，任务越不同间隔越大，自然适应任务结构
- 使用语言 embedding 作为参考锚点是一个优雅的选择——语言描述稳定不变，是任务的天然标识符

## 局限与展望
- 依赖预训练阶段的 CLIP 编码器质量，如果编码器表示不适合目标域可能需要微调
- 缓冲区的统一随机采样策略可能不是最优的，基于信息量的优先采样可能更好
- IFA 的 $\alpha$ 对不同数据集需要手动调整（0.1-0.7），自动搜索机制值得探索
- 仅在 LIBERO 基准上验证，更多实际机器人平台的验证是必要的

## 相关工作与启发
- LOTUS 提出了开放词汇的终身学习方案，本文证明冻结 CLIP + 潜在回放是更简洁有效的替代
- M2Distill 用多模态蒸馏保持表示稳定，IFA 用角距离约束达到类似目标但更简单
- 潜在回放的思想可推广到其他需要持续学习的多模态系统
- TAIL 的每任务 LoRA 方案需要测试时任务 ID，本文实现了任务无关的 LIL
- ER 直接存储原始轨迹，MLR 存储潜特征减少 >90% 的存储开销

## 补充细节
- 缓冲区存储采用随机采样策略，保持所有已遇任务的平衡分配
- FiLM 层用语言特征调制视觉和状态 embedding，实现任务条件化的表示
- GPT-2 作为时序解码器，将多模态潜特征序列映射为全局表示
- 实验使用单张 A100 GPU，计算资源需求低
- LIBERO-50 中每阶段引入 5 个新任务，共 5 阶段 25 个新任务，充分测试长序列可扩展性

## 评分
- 新颖性: ⭐⭐⭐⭐ MLR+IFA 的设计简洁有效，自适应间隔约束有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 LIBERO 数据集、多基线对比、详细消融（模态、参考、比例、缓冲区大小）
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，消融分析系统化，IFA 的图示直观
- 价值: ⭐⭐⭐⭐⭐ AUC +10-17 点、NBT 下降 65% 的提升巨大，方法简洁可复现

<!-- RELATED:START -->

## 相关论文

- [Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation](rethinking_camera_choice_an_empirical_study_on_fisheye_camera_properties_in_robo.md)
- [GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion](graspldp_towards_generalizable_grasping_policy_via_latent_diffusion.md)
- [MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](msrl_scaling_generative_multimodal_reward_modeling.md)
- [Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)
- [BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)

<!-- RELATED:END -->
