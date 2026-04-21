---
title: >-
  [论文解读] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning
description: >-
  [CVPR 2026][自动驾驶][VLA模型] NoRD 证明自动驾驶 VLA 不需要大规模推理标注和海量数据：通过识别 GRPO 在弱 SFT 策略上失败的根因是 **difficulty bias**（高方差 rollout 组的学习信号被压制），采用 Dr. GRPO 替代标准 GRPO 做 RL 后训练，仅用 <60% 数据、无推理标注、3× 更少 token，在 NAVSIM（85.6 PDMS）和 WaymoE2E（7.709 RFS）上达到与推理型 VLA 竞争的性能。
tags:
  - CVPR 2026
  - 自动驾驶
  - VLA模型
  - 无推理驾驶
  - 数据高效
  - Dr.GRPO
  - 强化学习后训练
---

# NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning

**会议**: CVPR 2026  
**arXiv**: [2602.21172](https://arxiv.org/abs/2602.21172)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: VLA模型, 无推理驾驶, 数据高效, Dr.GRPO, 强化学习后训练

## 一句话总结

NoRD 证明自动驾驶 VLA 不需要大规模推理标注和海量数据：通过识别 GRPO 在弱 SFT 策略上失败的根因是 **difficulty bias**（高方差 rollout 组的学习信号被压制），采用 Dr. GRPO 替代标准 GRPO 做 RL 后训练，仅用 <60% 数据、无推理标注、3× 更少 token，在 NAVSIM（85.6 PDMS）和 WaymoE2E（7.709 RFS）上达到与推理型 VLA 竞争的性能。

## 研究背景与动机

**VLA 主流范式的三重成本**：当前自动驾驶 VLA 的标准训练管线是"大规模 SFT + CoT 推理标注 + GRPO 后训练"。AutoVLA 等模型虽然性能强，但需要 21.2 万+ 样本、密集推理标注（annotation cost）、推理时生成推理 token 增加延迟。这三重成本（数据、标注、计算）不可扩展。

**推理是否必要？** 已有理论和实证工作质疑推理的必要性：(a) "Reasoning-Planning Decoupling Hypothesis" 表明文本先验就能匹配完整多模态推理的性能；(b) RL 不创造新的推理能力，只在 SFT 模型已有分布上优化。

**初始尝试失败**：用 8 万样本（无推理标注）训练 NoRD-base（Qwen-2.5VL-3B），然后 GRPO 后训练仅提升 +0.67%（76.66→77.18），而 AutoVLA 的 GRPO 提升了 +9%。这似乎证明"推理数据不可或缺"。

**核心发现——Difficulty Bias**：GRPO 失败不是因为弱 SFT 策略本身不行，而是 GRPO 的 advantage 归一化机制有系统性缺陷。组内标准差 $\text{std}$ 做分母时，低方差组（简单/极难场景）的 advantage 被放大，高方差组（中等难度、占多数）的 advantage 被压制。弱 SFT 模型恰好产生大量高方差 rollout，导致 GRPO 无法从主体样本中学习。

## 方法详解

### 整体框架

输入：前方、前左、前右 3 个相机 RGB 图像 + 历史自车轨迹 + 当前速度/加速度 → Qwen-2.5VL-3B-Instruct → 直接预测未来轨迹 token（无推理 token）。

训练流程：(1) 小规模数据 SFT（NAVSIM 8 万样本/WaymoE2E 1.2 万样本）→ (2) Dr. GRPO 强化学习后训练。

### 关键设计

1. **K-disc 轨迹 token 化**：

    - 功能：将连续轨迹离散化为 2048 个 codebook token
    - 核心思路：训练集所有轨迹插值到 10Hz → 切分为 0.5s 片段 → K-means 聚类到 2048 个 cluster → cluster 中心组成离散轨迹词汇表
    - 设计动机：将轨迹预测转化为 next-token prediction，与 VLM 的自回归范式天然契合。codebook 大小 2048 平衡了精度和泛化性
    - Token 嵌入初始化：用 Qwen 原有 token 嵌入的均值和协方差参数化多元正态分布采样，保证新 token 与预训练分布兼容

2. **Difficulty Bias 分析与 Dr. GRPO**：

    - **GRPO 的 advantage**：$\hat{A}_{i,t}^{\text{GRPO}} = \frac{r(o_i|x) - \frac{1}{G}\sum_{j=1}^G r(o_j|x)}{\text{std}_{j=1,...,G}(r(o_j|x))}$
    - **问题根源**：分母 $\text{std}$ 在低方差组时很小（$\ll 1$），advantage 被放大；高方差组时很大，advantage 被压制。弱 SFT 模型的 reward 分布呈两极化——简单场景（均值 ≥0.8）和极难场景（≤0.15）方差低，中等难度（0.2-0.65）方差高且占多数
    - **Dr. GRPO 的修正**：移除标准差归一化，$\hat{A}_{i,t}^{\text{DrGRPO}} = r(o_i|x) - \frac{1}{G}\sum_{j=1}^G r(o_i|x)$，确保"难"场景也贡献足够梯度信号
    - **辅助稳定措施**：DAPO 风格的非对称 clipping（$1-\epsilon_l, 1+\epsilon_h$）防止 entropy collapse；不使用 KL 散度正则化

3. **数据高效 SFT**：

    - 功能：有意用少量数据做 SFT，将主要学习负担转移到 RL 后训练阶段
    - NAVSIM：仅 8 万样本（vs AutoVLA 21.2万+）
    - WaymoE2E：仅 1.2 万样本 SFT + 450 样本 RLFT
    - 设计动机：验证 VLA 不需要大规模数据，RL 后训练可以弥补 SFT 阶段的性能差距

### 损失函数 / 训练策略

- **SFT 阶段**：标准 next-token prediction 交叉熵损失
- **RL 阶段**：Dr. GRPO 目标函数，reward 来自 PDM score（NAVSIM）或 RFS（WaymoE2E），加上轨迹长度和输出格式的辅助 reward（权重 0.25），归一化到 $[0,1]$
- **训练细节**：SFT 用 16×A100，lr=5e-5，batch=128；RLFT 用 30×A100（NAVSIM）或 32 GPU（WaymoE2E），lr=5e-6/1e-6，group size=8，采样温度 1.0
- **推理**：温度 0.01 的确定性采样，无推理 token 开销

## 实验关键数据

### 主实验

| 方法 | 推理? | 数据量 | NAVSIM PDMS↑ | WaymoE2E RFS↑ | WaymoE2E ADE↓ |
|------|-------|--------|-------------|---------------|---------------|
| UniAD | N/A | - | 83.4 | - | - |
| DiffusionDrive | N/A | - | 88.1 | - | - |
| AutoVLA | 有 | 212K+ | 89.1 | 7.556 | 1.3507 |
| RecogDrive | 有 | 2.7M+ | 89.6 | - | - |
| Poutine | 有 | 212K+ | - | **7.986** | 1.2055 |
| **NoRD** | **无** | **<90K** | **85.6** | **7.709** | **1.2504** |

- NAVSIM 上 NoRD-BoN（最佳 6 选 1）达 92.4，超越 AutoVLA-BoN（92.1）
- WaymoE2E 上 NoRD 是 RFS 第三名，但是唯一无推理+无 ensemble 的顶级模型
- ADE 指标上 NoRD 超越所有竞争者（包括 Poutine），证明轨迹精度高

### 消融实验

| 配置 | NAVSIM PDMS↑ | 说明 |
|------|-------------|------|
| NoRD-base（仅 SFT） | 76.66 | 弱 SFT 策略基线 |
| NoRD-base + GRPO | 77.18 (+0.67%) | GRPO 几乎无效 |
| NoRD-base + Dr. GRPO | **85.62 (+11.68%)** | Dr. GRPO 成功优化弱策略 |

### 关键发现

- **GRPO 对弱策略失效是 difficulty bias 导致**：训练过程中，GRPO 仅优化了少量低方差样本（已知简单行为），高方差样本的 group-mean 分布几乎不变化
- **Dr. GRPO 解锁中等难度样本的学习**：训练过程中，[0.2, 0.65] 范围的 group-mean 分布显著右移，模型学会了复杂操作（急转弯、变道）
- **NoRD 是最 token 高效和推理最快的 VLA**：无推理 token 意味着 3× token 减少和显著更低的推理延迟
- **数据效率极高**：WaymoE2E 上仅用 1.2 万样本（Poutine 用 20 万+），RFS 仅差 0.277，ADE 反而更优

## 亮点与洞察

- **首次将 difficulty bias 概念引入自动驾驶 VLA 训练**：从 LLM 推理领域借鉴 Dr. GRPO 到自动驾驶，跨领域迁移非常有效
- **挑战"推理必要性"假设**：用实证结果反驳"VLA 必须有 CoT 推理才能达到高性能"的流行观点，为轻量化 VLA 提供依据
- **训练失败的诊断方法论**：通过分析 rollout 的 reward 分布特征（均值-方差关系）来诊断 RL 优化失败的原因，这一方法论具有通用价值
- **Pareto 前沿分析**：在效率-性能二维图上，NoRD 占据了高效率+高性能的独特位置

## 局限与展望

- **绝对性能仍有差距**：NAVSIM 上 NoRD（85.6）vs AutoVLA（89.1），-3.5 分。推理数据在某些场景下可能确实有帮助
- **Dr. GRPO 并非完美**：论文承认 Dr. GRPO 只是缓解而非消除 difficulty bias，仍有改进空间
- **模型规模固定为 3B**：未探索更大/更小模型的表现，缺乏 scaling 分析
- **仅用前方 3 相机**：未利用后方和侧后方视图，可能限制在复杂交通场景中的表现
- **改进方向**：(a) 设计更好的 reward shaping 缓解两极化分布；(b) 结合少量推理标注做混合训练；(c) 扩展到闭环评估

## 相关工作与启发

- **vs AutoVLA**：AutoVLA 是推理型 VLA 的代表（212K 数据 + CoT + GRPO），NoRD 以 <40% 数据、无推理达到竞争性能，证明推理不是必需品
- **vs EMMA/SimLingo**：同为无推理 VLA，但此前仅在简单 benchmark（nuScenes）上验证；NoRD 首次在 NAVSIM + WaymoE2E 等复杂 benchmark 上证明可行
- **vs LLM 推理领域**：Dr. GRPO 最初为解决 LLM 数学推理中的 difficulty bias 设计；本文是首次验证其在自动驾驶密集 reward 场景中的有效性
- **启发**：RL 后训练的潜力被 SFT 规模的假设严重低估；正确的优化算法可以替代大量数据。弱策略+好优化器可能是更经济的范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次识别 GRPO difficulty bias 在自动驾驶中的影响，跨领域创新
- 实验充分度: ⭐⭐⭐⭐ NAVSIM + WaymoE2E 双 benchmark，详细的 reward 分布分析和训练动态
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析深刻，可视化优秀（Figure 2/3 的 reward 分布演化图非常有说服力）
- 价值: ⭐⭐⭐⭐⭐ 挑战了主流范式假设，为数据高效和推理高效的 VLA 开辟新路径

<!-- RELATED:START -->

## 相关论文

- [Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)
- [AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/autonomous_driving/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [ST4VLA: Spatially Guided Training for Vision-Language-Action Models](../../ICLR2026/autonomous_driving/st4vla_spatially_guided_training_for_vision-language-action_models.md)
- [MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving](minddriver_introducing_progressive_multimodal_reasoning_for_autonomous_driving.md)
- [WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation](walkgpt_grounded_vision-language_conversation_with_depth-aware_segmentation_for_.md)

<!-- RELATED:END -->
