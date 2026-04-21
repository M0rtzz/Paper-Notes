---
title: >-
  [论文解读] MARS-Sep: Multimodal-Aligned Reinforced Sound Separation
description: >-
  [ICLR 2026][Sound Separation] MARS-Sep 将查询条件声音分离重新建模为强化学习问题，通过分解 Beta 掩码策略在时频域上进行随机决策，并利用渐进式对齐的多模态编码器提供语义奖励信号，在信号保真度和语义一致性上同时取得提升。
tags:
  - ICLR 2026
  - Sound Separation
  - 强化学习
  - 多模态
  - Beta Policy
  - Preference Reward
---

# MARS-Sep: Multimodal-Aligned Reinforced Sound Separation

**会议**: ICLR 2026  
**arXiv**: [2510.10509](https://arxiv.org/abs/2510.10509)  
**代码**: https://github.com/mars-sep/MARS-Sep (有)  
**领域**: 音频处理 / 强化学习  
**关键词**: Sound Separation, Reinforcement Learning, Multimodal Alignment, Beta Policy, Preference Reward

## 一句话总结

MARS-Sep 将查询条件声音分离重新建模为强化学习问题，通过分解 Beta 掩码策略在时频域上进行随机决策，并利用渐进式对齐的多模态编码器提供语义奖励信号，在信号保真度和语义一致性上同时取得提升。

## 研究背景与动机

**领域现状**：通用声音分离（Universal Sound Separation）旨在从任意音频混合中分离出单独的声源。查询条件声音分离进一步允许用户通过音频、文本或图像查询来指定目标声源。当前主流方法（如 AudioSep、OmniSep）主要优化信号级别的损失函数（如 SDR、SI-SDR），通过预测时频掩码来重建目标波形。

**现有痛点**：现有方法面临一个根本性的"指标困境"——针对波形重建优化的模型可能在信号指标上得分很高，但输出中仍然残留感知上显著的干扰成分，违反了与查询的语义对应关系。例如，优化 SDR 的模型可能无法区分声学特征相似但语义完全不同的声源（如小提琴和中提琴），因为信号级损失不encoding 语义信息。

**核心矛盾**：信号级别的优化目标（低级别特征匹配）与语义级别的分离需求（高级别语义对齐）之间存在根本性的不对齐。传统的回归式掩码预测直接对齐 ground-truth 掩码，无法将查询的语义意图融入优化过程。

**本文目标** (1) 如何让分离模型的优化目标同时考虑信号保真度和语义一致性？(2) 如何将掩码预测从确定性回归转变为可探索的随机决策？(3) 如何获得稳定且语义丰富的奖励信号？

**切入角度**：受 RLHF 启发，作者将查询条件声音分离类比为偏好对齐问题——用户查询就是偏好，目标是产生最大化与查询语义对齐的输出。分离模型视为 base policy，通过强化学习优化。

**核心 idea**：用分解 Beta 分布策略（factorized Beta policy）在时频 bin 上进行随机掩码采样，通过渐进式对齐的多模态编码器提供语义奖励，以信赖域代理目标稳定训练。

## 方法详解

### 整体框架

MARS-Sep 建立在 OmniSep 架构之上。输入为混合音频频谱图 $X$ 和多模态查询 $Q$（音频/文本/图像）。分离器预测确定性掩码提案 $P_\theta(X,Q) \in [0,1]^{H \times W \times K}$，然后参数化为分解 Beta 分布策略 $\pi_\theta(M|X,Q)$，从中采样随机掩码 $M$。采样掩码应用于频谱图后通过 iSTFT 重建波形 $\hat{y}$，由渐进对齐的多模态编码器（基于 ImageBind）计算分离音频与查询之间的语义一致性奖励 $R$。策略通过截断信赖域代理目标更新。

### 关键设计

1. **分解 Beta 掩码策略（Factorized Beta Mask Policy）**:

    - 功能：将确定性掩码预测转化为可探索的随机策略
    - 核心思路：将分离器的输出 $P_\theta$ 转化为 Beta 分布的参数：$\pi_\theta(M|X,Q) = \prod_{h,w,k} \text{Beta}(M_{h,w,k}; \alpha_{h,w,k}, \beta_{h,w,k})$，其中 $\alpha = 1 + \kappa P_\theta$，$\beta = 1 + \kappa(1-P_\theta)$。浓度尺度 $\kappa > 0$ 控制探索-利用平衡。分解结构使得每个时频 bin 独立采样，log-概率在 bin 上可因式分解
    - 设计动机：Beta 分布的 $[0,1]$ 支撑与掩码值域天然匹配。$\kappa$ 退火可以避免训练早期出现退化的近二值掩码。相比高斯策略或离散化，Beta 分布更自然且无截断问题

2. **截断信赖域代理目标（Clipped Trust-Region Surrogate）**:

    - 功能：稳定策略更新，避免 plain policy gradient 的高方差和崩溃
    - 核心思路：定义重要性比 $r_\theta(M) = \pi_\theta(M|X,Q) / \pi_{\theta_{\text{old}}}(M|X,Q)$，使用组相对优势 $\tilde{A} = (A - \mu(A))/(\sigma(A) + \varepsilon)$。优化截断代理目标：$\mathcal{J}_{\text{clip}}(\theta) = \mathbb{E}[\min(r_\theta \tilde{A}, \text{clip}(r_\theta, 1-\epsilon, 1+\epsilon)\tilde{A}) + \lambda_H \mathcal{H}(\pi_\theta) - \lambda_{\text{KL}} \text{KL}(\pi_\theta \| \pi_{\theta_{\text{old}}})]$，结合熵正则化和 KL 惩罚
    - 设计动机：单步 PPO 更新保持了训练循环的简洁性，无需额外的 value network 或复杂的优势估计器。GRPO 式的归一化优势消除了奖励尺度的影响

3. **渐进式多模态编码器对齐（Progressive Alignment）**:

    - 功能：训练可靠的多模态奖励模型，避免 reward hacking
    - 核心思路：分三阶段微调 ImageBind 编码器。Stage 1：音频-文本对齐，仅解冻投影头和温度参数，用对称 InfoNCE 损失 $\mathcal{L}_{S1}$ 建立语义锚点。Stage 2：音频-音频判别，加入 triplet loss 和一致性损失 $\mathcal{L}_{S2}$，增强类内区分能力，混合部分 Stage 1 数据防遗忘。Stage 3：音频-视频接地，联合 InfoNCE 和 triplet loss $\mathcal{L}_{S3}$，同时保留前两阶段能力
    - 设计动机：直接用预训练 ImageBind 作为奖励模型会导致 reward hacking——策略学会"欺骗"奖励而非真正改善分离质量。渐进式训练使编码器逐步获得声源判别能力，提供更稳定、更有信息量的奖励信号

4. **多模态奖励聚合（Query-Pooling Reward）**:

    - 功能：将音频、文本、视觉三种查询模态融合为统一的奖励信号
    - 核心思路：使用多模态低秩双线性池化（MLBP）融合目标侧嵌入：$z^* = \text{MLBP}(\phi_a(y^*), \phi_t(t^*), \phi_v(v^*))$，标量奖励为 $R = \text{sim}(\phi_a(\hat{y}), z^*)$。分离音频保留其原生表示，目标模态融合为语义锚点
    - 设计动机：如果分别比较各模态，奖励可能过度偏向某个模态。双线性池化显式建模跨模态交互（如文本指定的乐器在视觉上也要出现），鼓励分离音频同时与所有模态对齐

### 损失函数 / 训练策略

总训练损失为 $\mathcal{L}_{\text{RL}}(\theta) = -\mathcal{J}_{\text{clip}}(\theta)$，包含截断代理目标、熵正则化和 KL 惩罚三部分。每步训练从冻结的旧策略 $\pi_{\theta_{\text{old}}}$ 采样掩码，计算奖励后更新当前策略，然后将当前策略快照为下一步的旧策略。

## 实验关键数据

### VGGSOUND-clean+ 主实验

| 方法 | 查询 | SDR↑ | SIR↑ | SAR↑ | SI-SDRi↑ | CLAP↑ |
|------|------|------|------|------|----------|-------|
| AudioSep | Text | 6.26 | 8.69 | 12.85 | 4.01 | 8.21 |
| OmniSep | Text | 6.70 | 9.04 | 13.61 | 4.38 | 8.98 |
| **MARS-Sep** | **Text** | **6.91** | **9.14** | **13.73** | **4.55** | **9.03** |
| OmniSep | Image | 6.66 | 10.00 | 13.73 | 4.43 | 8.79 |
| **MARS-Sep** | **Image** | **6.93** | **10.18** | **13.41** | **4.57** | **9.19** |
| OmniSep | Omni | 7.79 | 10.76 | 14.53 | 5.16 | 8.85 |
| **MARS-Sep** | **Omni** | **7.93** | **10.65** | **14.49** | **5.20** | **9.22** |

### MUSIC-clean+ 跨域验证

| 方法 | 查询 | SDR↑ | SIR↑ | SAR↑ | SI-SDRi↑ | CLAP↑ |
|------|------|------|------|------|----------|-------|
| CLIPSEP-NIT | Text | 11.03 | 16.40 | 17.37 | 7.53 | 5.29 |
| OmniSep | Text | 12.37 | 17.51 | 17.96 | 9.18 | 5.41 |
| **MARS-Sep** | **Text** | **12.91** | **17.61** | **18.28** | **9.85** | **6.18** |
| OmniSep | Image | 13.03 | 18.97 | 17.88 | 10.21 | 6.53 |
| **MARS-Sep** | **Image** | **13.64** | **19.24** | **18.05** | **10.70** | **6.94** |

### 关键发现

- **信号指标与语义指标同步提升**：MARS-Sep 不仅在 CLAP score 上稳定领先（证明语义对齐改善），SDR/SIR/SI-SDRi 也全面提升，说明 RL 奖励没有牺牲信号质量
- **跨域泛化能力强**：从包含 300+ 声音类别的 VGGSound 到专注乐器演奏的 MUSIC，MARS-Sep 的增益保持甚至扩大，MUSIC 上 CLAP 提升 +0.77（14.2%相对提升）
- **生成式方法对比**：FlowSep、ZeroSep 等生成式方法的 CLAP score 方差极大（如 ZeroSep 在 MUSIC 上 $20.02 \pm 15.14$），而 MARS-Sep 仅 $6.18 \pm 0.93$，稳定性远优
- **渐进式对齐的必要性**：不经过三阶段微调直接使用预训练 ImageBind 作为奖励模型会导致 reward hacking，分离质量反而下降

## 亮点与洞察

- **声音分离 × RLHF 的类比精炼**：将"用户查询 = 偏好"的类比落地为完整的 RL 框架，Beta 分布策略与掩码值域的天然匹配是一个优雅的工程选择
- **渐进式对齐策略的鲁棒性**：三阶段递进（语义锚定 → 类内判别 → 跨模态接地）的课程设计避免了一步对齐的不稳定性，每阶段混合前阶段数据防遗忘
- **Actor-only 设计的简洁性**：放弃了 value network 和复杂的优势估计，用单步 PPO + 滑动平均基线就实现了稳定训练，说明在掩码预测这个"单步 MDP"中不需要复杂的 RL 基础设施

## 局限与展望

- **单步 MDP 的局限**：当前将掩码预测视为单步决策，忽略了时序结构——分离长音频时，序列化决策可能更有效
- **奖励模型的通用性**：渐进对齐依赖 ImageBind 作为骨干网络，对 ImageBind 不覆盖的声音类别可能效果有限
- **计算开销**：RL 训练需要多次采样掩码和计算奖励，训练速度相比直接监督学习慢多少未报告
- **缺乏人类评估**：仅使用客观指标，未进行主观听感评估来验证语义对齐的感知效果

## 相关工作与启发

- **vs OmniSep (Cheng et al., 2025)**：OmniSep 提供了统一多模态查询的基座分离器，但训练目标仍是加权 BCE 损失。MARS-Sep 保留 OmniSep 的架构，在其上叠加 RL 训练来注入语义监督
- **vs AudioSep (Liu et al., 2024)**：AudioSep 使用 CLAP 编码器和 14k 小时语料实现零样本分离，但训练仍是回归式。MARS-Sep 表明即使训练数据更少，RL + 语义奖励可以超越纯监督方法
- **vs RLHF in LLMs**：本文是 RLHF 范式在音频生成/处理领域的创新应用，reward model 的渐进训练策略可迁移到其他跨模态生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 RL 偏好对齐引入声音分离是新颖的跨领域迁移，Beta 策略设计精巧
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、四种查询模态、多个 baseline 对比全面，但缺乏人类评估和计算开销分析
- 写作质量: ⭐⭐⭐⭐ 框架清晰，RLHF 类比有效，但符号较多
- 价值: ⭐⭐⭐⭐ 为音频处理引入了 RL 对齐范式，有望推动该领域方法论的发展

<!-- RELATED:START -->

## 相关论文

- [UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [Spotlight on Token Perception for Multimodal Reinforcement Learning](spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)
- [STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision](../../AAAI2026/reinforcement_learning/stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)
- [TextShield-R1: Reinforced Reasoning for Tampered Text Detection](../../AAAI2026/reinforcement_learning/textshield-r1_reinforced_reasoning_for_tampered_text_detection.md)
- [MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy](../../AAAI2026/reinforcement_learning/mathsmith_towards_extremely_hard_mathematical_reasoning_by_forging_synthetic_pro.md)

<!-- RELATED:END -->
