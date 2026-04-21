---
title: >-
  [论文解读] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training
description: >-
  [CVPR 2026][多模态] 提出GTR-Turbo框架，通过合并RL训练过程中产生的历史checkpoint作为免费教师模型，在无需依赖昂贵外部API模型的条件下，实现了与GTR相当甚至更优的多轮视觉代理训练效果，同时将训练时间减少50%、计算成本降低60%。
tags:
  - CVPR 2026
  - 多模态
---

# GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training

**会议**: CVPR 2026  
**arXiv**: [2512.13043](https://arxiv.org/abs/2512.13043)  
**代码**: [GitHub](https://github.com/tongweiICML/GTR-Turbo)  
**领域**: 多模态与VLM

## 一句话总结

提出GTR-Turbo框架，通过合并RL训练过程中产生的历史checkpoint作为免费教师模型，在无需依赖昂贵外部API模型的条件下，实现了与GTR相当甚至更优的多轮视觉代理训练效果，同时将训练时间减少50%、计算成本降低60%。

## 背景与动机

1. **多轮RL训练的核心挑战**：基于VLM的多模态代理在多轮强化学习中面临稀疏奖励和长期信用分配问题，导致"思维崩塌"（thought collapse），输出变得重复、不连贯且模板化。
2. **现有方法依赖昂贵教师模型**：GTR等方法通过外部教师模型（如GPT-4o）提供步骤级指导来稠密化奖励，但训练LLaVA-1.6-7B模型15,000步需约4天时间和150美元，严重限制了可扩展性。
3. **弱教师模型无法提供有效指导**：使用较小的教师模型虽然能降低开销，但会显著降低最终性能。例如Qwen2.5-VL-7B作为教师时完全无法提供有效的思维指导（成功率为0%）。
4. **数据隐私与可访问性限制**：使用闭源API模型不仅带来成本问题，还引入网络延迟和数据隐私隐患，且前沿模型在某些场景下不可获取。

## 方法详解

### 3.1 核心思想：合并检查点即免费教师

GTR-Turbo的关键洞察是：RL训练过程中产生的历史checkpoint，经过模型合并后可以构成一个稳定且能力更强的教师模型，完全不需要额外训练或外部模型依赖。

在第 $k$ 次更新时，合并教师模型定义为：

$$\pi^{(k)}_{\text{merged}} = \sum_{i=1}^{k-1} w_i \pi_{\theta}^{(i)}$$

### 3.2 TIES合并方法

为避免直接合并所有参数导致的有害干扰，采用TIES（Trim, Elect Sign, and Merge）方法：

1. **修剪（Trimming）**：仅保留幅值在top-$k$%的参数变化，移除冗余参数
2. **符号选举（Sign Election）**：对每个参数计算正负值的总幅度，通过多数投票确定最终符号
3. **选择性平均（Selective Averaging）**：仅将符号匹配的参数纳入合并计算

### 3.3 权重分配策略

支持两种策略：

- **简单移动平均（SMA）**：等权对待所有检查点，$\pi_{\text{merged}}^{(k)} = \frac{1}{k-1}\sum_{i=1}^{k-1}\pi_{\theta}^{(i)}$
- **指数移动平均（EMA）**：通过平滑因子 $\alpha$ 给予近期检查点更大权重，$\pi_{\text{merged}}^{(k)} = \alpha \cdot \pi_{\theta}^{(k-1)} + (1-\alpha) \cdot \pi_{\text{merged}}^{(k-1)}$

### 3.4 变体一：GTR-Turbo (SFT)

将合并教师替换GTR中的外部纠正器，在每个RL步骤中：
- Agent生成思维和动作后，教师在相同上下文下生成参考思维
- 通过最小化SFT损失实现在线模仿学习

$$\min_{\theta} \mathbb{E}_{(o,a)\sim\mathcal{B}} \mathcal{L}_{\text{PPO}}(o,a) + \mathbb{E}_{(o,\hat{th})\sim\mathcal{D}} \mathcal{L}_{\text{SFT}}(o,\hat{th})$$

### 3.5 变体二：GTR-Turbo (KL) — 软logit蒸馏

用KL散度替代SFT目标，计算Agent与教师之间的反向KL散度作为辅助奖励：

$$A' = A^{\pi_\theta}(o,a) - \text{RevKL}(\pi_\theta, \pi_{\text{merged}}; th)$$

$$\text{RevKL}(\pi_\theta, \pi_{\text{merged}}; th) = \mathbb{E}_l \left[\log\pi_\theta(th_{[l]}|th_{[<l]}) - \log\pi_{\text{merged}}(th_{[l]}|th_{[<l]})\right]$$

KL变体的优势：(1) 仅需单次前向传播，无需教师自回归生成；(2) 捕获所有候选token的概率信息而非one-hot监督；(3) 减少内存消耗，无需额外思维数据集。采用clip方法将负KL值截断为0以确保奖励有效性。

## 实验结果

### Points24卡牌游戏

| 模型 | 成功率(%) | 回合回报(ER) |
|------|-----------|-------------|
| GPT-4o | 2.5 | -6.35 |
| Qwen2.5-VL-72B | 5.6 | -5.69 |
| Qwen2.5-VL-7B-sft | 22.0 | -3.2 |
| RL4VLM | 3.5 | -13.3 |
| GTR (GPT-4o教师) | 44.5 | 0.53 |
| **GTR-Turbo (SFT)** | **48.0** | **1.32** |
| **GTR-Turbo (KL)** | **53.5** | **2.39** |

### 计算成本对比

| 环境 | 方法 | 成功率 | 训练时间 | 成本估算 |
|------|------|--------|----------|----------|
| Points24 | RL4VLM | 4% | 86h | $0 |
| Points24 | GTR | 41% | 191h | $307.78 |
| Points24 | GTR-Turbo (SFT) | 48% | 168h | $216.72 |
| Points24 | **GTR-Turbo (KL)** | **54%** | **89h** | **$114.81** |
| ALFWorld | RL4VLM | 8% | 70h | $0 |
| ALFWorld | GTR | 16% | 164h | $145.76 |
| ALFWorld | GTR-Turbo (KL) | 15% | 78h | $100.62 |

### Android-in-the-Wild (AitW) 实验

| 方法 | 成功率 | 推理评分 |
|------|--------|----------|
| DigiRL | 71.9% | - |
| PPO | 75.0% | 3.26 |
| **GTR-Turbo** | **80.2%** | **3.93** |

### 关键消融实验

- **TIES合并 vs 线性平均**：TIES通过缓解冗余参数干扰显著优于简单线性合并
- **KL估计方法**：clip方法最佳，通过控制KL值量级提供更细粒度更新
- **反向KL vs 前向KL**：反向KL因"mode-seeking"特性更有效
- **SMA vs EMA**：SMA已有很强表现，EMA中 $\alpha=0.5$ 效果最佳
- **指导范围**：仅指导思维（thought）优于同时指导思维和动作，因为后者限制了探索

## 亮点

- **零外部依赖**：完全不需要外部API模型，通过合并自身训练过程的checkpoints构建教师模型，解决了GTR的核心瓶颈
- **显著效率提升**：KL变体在Points24上将训练时间从191h降至89h（-53%），成本从$308降至$115（-63%），且性能更优
- **自我进化范式**：教师模型随训练进程持续积累知识并变强，而非GTR中固定外部模型无法进一步学习的限制
- **灵活的两种指导方式**：SFT变体保留GTR的在线模仿学习，KL变体进一步提升效率并鼓励探索
- **强泛化性**：在Points24、ALFWorld和Android-in-the-Wild三种不同类型的视觉代理任务上均展现一致优势

## 局限性

- **弱基座模型失效**：当基座模型初始成功率极低（<5%）时，自我改进方法可能失败，此时仍需更强的外部教师
- **模型规模受限**：实验主要在7B-8B模型上验证，更大规模模型上的效果尚未充分探索
- **Checkpoint存储开销**：需要保存训练过程中的多个完整checkpoints用于合并，对存储和内存有一定要求

## 评分

- ⭐⭐⭐⭐ 新颖性：将模型合并技术巧妙地引入RL代理训练中，"免费教师"的洞察新颖且直觉清晰
- ⭐⭐⭐⭐ 实用性：完全消除对外部API的依赖，大幅降低训练成本，实际部署价值很高
- ⭐⭐⭐⭐ 实验充分度：覆盖3个环境、多种消融实验、成本分析全面，还在最新Qwen3-VL上验证了兼容性
- ⭐⭐⭐⭐ 写作质量：动机清晰、方法阐述完整、实验设计合理，图表丰富且直观

<!-- RELATED:START -->

## 相关论文

- [HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [Training-Free Personalization via Retrieval and Reasoning on Fingerprints](../../ICCV2025/multimodal_vlm/training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)
- [Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](../../ICLR2026/multimodal_vlm/visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)

<!-- RELATED:END -->
