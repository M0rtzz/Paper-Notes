---
title: >-
  [论文解读] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos
description: >-
  [ICCV 2025][机器人][视频预训练] 提出 Moto 框架，通过无监督学习的潜在运动 Token（Latent Motion Token）将视频帧间的视觉运动编码为离散序列，利用 GPT 式自回归预训练学习运动先验，再通过 co-fine-tuning 策略将学到的运动知识迁移到真实机器人操作，在 SIMPLER 和 CALVIN 基准上取得与 55B 参数大模型匹敌的性能（仅 98M 参数）。
tags:
  - ICCV 2025
  - 机器人
  - 视频预训练
  - 运动Token
  - 自回归
  - 机器人操作
  - 跨具身迁移
---

# Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos

**会议**: ICCV 2025  
**arXiv**: [2412.04445](https://arxiv.org/abs/2412.04445)  
**代码**: [https://chenyi99.github.io/moto/](https://chenyi99.github.io/moto/)  
**领域**: 机器人  
**关键词**: 视频预训练, 运动Token, 自回归, 机器人操作, 跨具身迁移

## 一句话总结

提出 Moto 框架，通过无监督学习的潜在运动 Token（Latent Motion Token）将视频帧间的视觉运动编码为离散序列，利用 GPT 式自回归预训练学习运动先验，再通过 co-fine-tuning 策略将学到的运动知识迁移到真实机器人操作，在 SIMPLER 和 CALVIN 基准上取得与 55B 参数大模型匹敌的性能（仅 98M 参数）。

## 研究背景与动机

**领域现状**：大语言模型（LLM）在 NLP 中通过大规模语料的自回归预训练取得了巨大成功。机器人领域一直受限于动作标注数据的高昂成本。视频数据包含丰富的交互知识且易于获取，但如何有效利用视频数据预训练机器人策略仍是开放问题。

**现有痛点**：先前的视频预训练方法（如 GR-1、MT-R3M）主要关注静态帧的视觉特征，强调帧级别的细节，忽略了帧间的动态变化。这些方法通常需要额外的输入模态（如 gripper RGB、本体感知）来弥补运动信息的缺失。现有 VLA（如 RT-2-X、OpenVLA）要么参数量极大（55B），要么需要预训练数据中就包含动作标签。

**核心矛盾**：视频数据丰富但缺乏动作标签，机器人动作数据稀少但有动作标签——如何用视频的海量运动知识来增强动作标签稀缺条件下的机器人策略学习？

**本文目标** 找到一种有效的表示，将视频中的运动知识编码为可供自回归预训练的序列，并能无缝迁移到下游机器人控制。

**切入角度**：人类通过观察动态环境变化来学习新技能，关注的是"运动"而非静态视觉细节。运动信息与低级动作紧密相关且硬件无关（hardware-agnostic），便于跨具身迁移。

**核心 idea**：用 VQ-VAE 将视频帧间的运动信息编码为离散的潜在运动 Token，以此作为"运动语言"进行 GPT 式自回归预训练，再通过 co-fine-tuning 将运动先验无缝转化为真实机器人动作。

## 方法详解

### 整体框架

Moto 分三个训练阶段：
1. **潜在运动 Tokenizer 训练**：无监督学习，将连续两帧间的运动压缩为 8 个离散 token
2. **Moto-GPT 自回归预训练**：以初始帧和语言指令为条件，预测运动 token 序列
3. **Co-fine-tuning**：在带动作标签的机器人数据上联合微调，将运动先验迁移为精确的机器人控制

### 关键设计

1. **潜在运动 Tokenizer**:

    - 功能：将连续两帧视频之间的关键视觉运动编码为紧凑的离散 token
    - 核心思路：编码器（M-Former）是一个多层 Transformer，以冻结 ViT 编码器提取的当前帧 $o_t$ 和前一帧 $o_{t-1}$ 的 patch 特征为输入，与 8 个可学习 query 嵌入拼接后通过自注意力交互。Query 输出特征经 VQ codebook（词汇量 128）量化为离散运动 token。解码器（ViT Decoder）根据 $o_{t-1}$ 的 patch 嵌入和运动 token 的紧凑嵌入（MLP 压缩为 1 个 token，加到每个 patch 嵌入上）重建 $o_t$ 的像素值。训练使用标准 VQ-VAE 目标（重建 MSE + VQ loss + commitment loss）
    - 设计动机：运动 token 作为信息瓶颈，迫使编码器只保留关键的动态变化信息。每帧只用 8 个 token（vs 原始 196 个 patch token），压缩率 24.5 倍，但保留了 79.7% 的语义分类准确率

2. **Moto-GPT 自回归预训练**:

    - 功能：在运动 token 序列上进行 GPT 式预训练，学习视频中的运动先验
    - 核心思路：对视频片段 $[o_0, o_1, ..., o_T]$，提取每对相邻帧的运动 token chunk 并按时间顺序拼接成序列。GPT Transformer 以冻结 T5 的文本特征 $\boldsymbol{l}$ 和冻结 ViT 的初始帧视觉特征 $\boldsymbol{v}$ 为前缀，通过 next-token prediction 训练：$\mathcal{L}_{motion} = -\sum_{i=1}^{M} \log P(m_i | \boldsymbol{l}, \boldsymbol{v}, \boldsymbol{m}_{<i}; \boldsymbol{\Theta})$，其中 $M = K \times T$（$K=8$ tokens/帧，$T$ 为视频长度）
    - 设计动机：以运动 token 而非像素/patch 作为预训练目标，使模型专注于学习"做什么"（运动意图）而非"看到什么"（视觉细节），与下游控制任务的需求更贴合

3. **Co-fine-tuning 策略**:

    - 功能：将预训练运动先验迁移为精确的机器人动作
    - 核心思路：在每个时间步的运动 token chunk 后追加 $N$ 个可学习的 action query token（$N$ 对应两帧之间的动作数量）。Action query 通过 MLP action head 预测真实动作空间（位移 $\Delta x$、旋转 $\Delta\theta$、夹持 $\Delta grip$）。关键设计：(a) 运动 token 不 attend action query（保持与预训练一致），(b) 随机 mask 50% 的 action query 到运动 token 的注意力（减少对 GT 条件的依赖），(c) 保留运动 token 预测损失。总损失 $\mathcal{L}_{ft} = \mathcal{L}_{motion} + \mathcal{L}_{action}$，其中 $\mathcal{L}_{action} = \mathcal{L}(\Delta x) + \mathcal{L}(\Delta\theta) + \mathcal{L}(\Delta grip)$（连续部分用 Smooth-L1，开关用 BCE）
    - 设计动机：直接丢弃运动 token（如 Moto-DM）会损失预训练知识；不保留运动预测损失（如 Moto-IML）会导致先验退化。Co-fine-tuning 让 action query 通过注意力从运动 token 中直接获取知识迁移

### 损失函数 / 训练策略

- Tokenizer：MSE 重建损失 + VQ loss + commitment loss
- 预训练：交叉熵 next-motion-token prediction
- 微调：$\mathcal{L}_{ft} = \mathcal{L}_{motion} + \mathcal{L}_{action}$
- 推理时可以用 padding token 替代运动 token 并阻断 attention，直接从 action query 输出动作，提高效率

## 实验关键数据

### 主实验

SIMPLER 基准（Google Everyday Robot，3 类任务）：

| 方法 | 参数量 | Pick Coke Can | Move Near | Open/Close Drawer | Overall |
|------|--------|---------------|-----------|-------------------|---------|
| RT-1-X | - | 0.567 | 0.317 | 0.597 | 0.534 |
| RT-2-X | **55B** | 0.787 | 0.779 | 0.250 | 0.607 |
| OpenVLA | 7B | 0.163 | 0.462 | 0.356 | 0.248 |
| OpenVLA (ft) | 7B | 0.363 | 0.542 | 0.231 | 0.349 |
| **Moto** | **98M** | **0.740** | **0.604** | **0.431** | **0.614** |
| Moto w/o MT | 98M | 0.503 | 0.554 | 0.398 | 0.480 |

CALVIN (ABC→D) 零样本长期任务完成：

| 方法 | 输入 | 1 task | 2 tasks | 3 tasks | 4 tasks | 5 tasks | Avg. Len. |
|------|------|--------|---------|---------|---------|---------|-----------|
| GR-1 | RGB+Gripper+Proprio | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06 |
| SuSIE | RGB | 0.870 | 0.690 | 0.490 | 0.380 | 0.260 | 2.69 |
| **Moto** | **RGB** | **0.897** | **0.729** | **0.601** | **0.484** | **0.386** | **3.10** |
| Moto w/o MT | RGB | 0.779 | 0.555 | 0.380 | 0.256 | 0.167 | 2.14 |

### 消融实验

微调策略消融（CALVIN ABC→D）：

| 配置 | Avg. Len.↑ | 说明 |
|------|-----------|------|
| Moto w/o Motion Token | 2.14 | 从头训练，无预训练先验 |
| Moto-DM（去掉输入中运动token） | ~2.6 | 有预训练但微调时无运动token |
| Moto-IML（去掉运动预测损失） | ~2.7 | 有运动token但不保留预测目标 |
| **Moto（完整co-fine-tuning）** | **3.10** | 保留运动token+预测损失，最优 |

运动 Token 语义性验证：

| 视频表示 | 语义分类准确率 |
|----------|-------------|
| 仅初始帧 | 29.2% |
| 初始帧重复 8 次 | 28.3% |
| 初始帧 + 7 后续帧（全 patch） | 82.8% |
| 初始帧 + 7 运动token chunk | **79.7%** |

### 关键发现

- **运动先验贡献巨大**：去掉运动 token（从头训练）在 SIMPLER 上降 13.4%（0.614→0.480），CALVIN 上 Avg. Len. 从 3.10 降到 2.14（降 31%）
- **98M 参数匹敌 55B 模型**：Moto 在 SIMPLER 上 Overall 0.614 略优于 RT-2-X 的 0.607，且仅使用单视角 RGB
- **跨具身迁移有效**：人类视频预训练可进一步提升性能——加入 SSV2 人类视频后 Move Near 任务提升明显
- **数据效率极高**：仅用 1% CALVIN 标注数据微调即达 52.5% 成功率（vs 从头训练的 0%）
- 运动 token 具有跨具身一致性：相同的 token chunk 在不同初始观测/机器人上产生语义一致的运动效果
- 预训练 Moto-GPT 可通过 log-likelihood 区分成功/失败/随机轨迹，可作为奖励信号

## 亮点与洞察

- **运动 Token 作为"语言"的类比非常到位**：就像自然语言的 token 是离散的、可组合的、语义丰富的，运动 token 也具备这些特性——可拼接成轨迹"句子"、跨场景迁移、支持自回归生成
- **VQ 信息瓶颈设计精妙**：8 个 token（词汇量 128）捕获两帧间运动，既够紧凑支持长序列自回归，又够表达保留 79.7% 语义信息。解码器只用 1 个条件 embedding 加到所有 patch 上，迫使运动信息高度压缩
- **Co-fine-tuning 的注意力设计很有讲究**：运动 token 不看 action query（保持一致性）+ 50% mask（减少依赖）+ 保留运动预测损失（防退化），每个设计决策都有消融验证
- 实际价值：为"用互联网视频训练机器人"提供了可行路径，预训练不需要任何动作标签

## 局限与展望

- 预训练仅用 109K OXE 视频，远未达到互联网规模，扩大数据规模的效果待验证
- 运动 token 的 VQ codebook 大小（128）和 token 数（8）的选择可能需要针对不同任务调整
- 仅关注简单的抓取/放置/推拉任务，未验证在精细操作（如装配、绑绳）上的效果
- 解码器的帧重建质量有限，可能限制了运动 token 的表达力上限
- 真实世界实验规模较小（每任务 30 个 demo，10 次测试），成功率方差未报告
- 与当前最先进的 VLA（如 π0）的对比缺失

## 相关工作与启发

- **vs GR-1**：GR-1 预训练预测未来帧像素值，需要 gripper RGB + 本体感知作为额外输入。Moto 仅用静态相机 RGB 就达到可比性能，说明运动级预训练优于帧级预训练
- **vs Genie**：Genie 也学习潜在动作用于 2D 游戏模拟器，但未迁移到真实机器人控制。Moto 完成了从视频到真实机器人的完整链路
- **vs LAPA/IGOR**：LAPA 预测单步未来潜在动作，IGOR 用潜在动作作为中间目标。Moto 自回归预测整个轨迹的运动 token 序列，更自然地建模连续运动
- 开源代码为未来探索提供了基础，运动 token 的表示范式可能催生新的机器人预训练路线

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 潜在运动 Token 的概念和三阶段训练范式非常创新，为视频→机器人的知识迁移提供了全新思路
- 实验充分度: ⭐⭐⭐⭐ SIMPLER + CALVIN + 真实世界 + 数据效率 + 人类视频 + 多角度消融，但真实世界规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 故事线清晰（token可解释性→先验学习→策略性能），每个实验回答一个明确的问题
- 价值: ⭐⭐⭐⭐⭐ 提出了一条可能颠覆机器人学习范式的路线——用无标签视频的运动知识大规模预训练，极大降低了动作标注的需求

<!-- RELATED:START -->

## 相关论文

- [CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](../../CVPR2026/robotics/como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [iManip: Skill-Incremental Learning for Robotic Manipulation](imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning](interaction-merged_motion_planning_effectively_leveraging_diverse_motion_dataset.md)
- [Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)

<!-- RELATED:END -->
