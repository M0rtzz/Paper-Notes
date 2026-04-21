---
title: >-
  [论文解读] PACE: Pretrained Audio Continual Learning
description: >-
  [ICLR 2026][语音][音频持续学习] 首次系统性构建音频持续学习基准，揭示预训练音频模型因底层频谱特征主导导致的上游-下游不匹配问题，提出 PACE 方法（改进首会话适应 + 自适应子空间正交 PEFT + 边界感知扰动），在 6 个音频 CL 基准上大幅超越 SOTA。
tags:
  - ICLR 2026
  - 语音
  - 音频持续学习
  - 预训练模型
  - 参数高效微调
  - 分析式分类器
  - 灾难性遗忘
---

# PACE: Pretrained Audio Continual Learning

**会议**: ICLR 2026  
**arXiv**: [2602.03355](https://arxiv.org/abs/2602.03355)  
**代码**: 有（将随论文发布）  
**领域**: 音频语音  
**关键词**: 音频持续学习, 预训练模型, 参数高效微调, 分析式分类器, 灾难性遗忘

## 一句话总结

首次系统性构建音频持续学习基准，揭示预训练音频模型因底层频谱特征主导导致的上游-下游不匹配问题，提出 PACE 方法（改进首会话适应 + 自适应子空间正交 PEFT + 边界感知扰动），在 6 个音频 CL 基准上大幅超越 SOTA。

## 研究背景与动机

预训练音频模型在静态任务上表现优异，但面临数据分布持续演化的场景时容易灾难性遗忘。将视觉域的持续学习（CL）方法直接迁移到音频域面临根本性障碍：

**上游-下游不匹配严重**：音频骨干（如 EAT）通过谱图重建预训练，强调低层时频模式而非结构化语义，但下游 CL 需要高层判别性表示

**表示漂移更剧烈**：音频域相邻会话间的表示变化远超视觉域（t-SNE/CKA 量化证实），导致更严重遗忘

**PEFT 方法失效**：L2P、DualPrompt 等在音频上退化幅度约为视觉上的 3 倍

三个关键发现驱动方法设计：

| 发现 | 内容 | 影响 |
|------|------|------|
| Finding 1 | 统计方法（FSA + 分析式分类器）优于 PEFT 方法 | 确立技术路线 |
| Finding 2 | 粗粒度存在表示饱和：首会话已捕获大部分信息 | 需改进 FSA |
| Finding 3 | 细粒度差距更大：首会话不足以弥合语义鸿沟 | 需多会话适应 |

## 方法详解

### 整体框架

PACE 是分阶段框架，三个 Stage：

- **Stage 1（FSA）**：冻结输出头，用 LoRA 适应深层骨干，然后替换为分析式分类器
- **Stage 2（MSA）**：会话 $t \in (1, T_3]$ 引入子空间正交 PEFT，逐步对齐表示
- **Stage 3（冻结）**：$t > T_3$ 冻结骨干，仅更新分析式分类器

### 关键设计

#### 1. 改进首会话适应（Improved FSA）

**受限头部学习**：
- 传统 FSA 联合训练头部和骨干导致头部过拟合、骨干适应不足
- PACE 采用不对称优化：$\eta_{head} \ll \eta_{bb}$
- 分阶段：先冻结骨干训练头部 $E_{head}$ 轮，再冻结头部微调骨干 $E_0$ 轮
- 与视觉 CL 的 LAE/SLCA **策略相反**——音频骨干需要鼓励适应而非抑制

**后层 LoRA**：
- CKA 分析：浅层编码域通用时频模式，深层编码任务特定语义
- 冻结前 $L_{tune}-1$ 层，仅对 $l \geq L_{tune}$ 层施加 LoRA：
  $$W_1^l = W_0^l + A_1^l B_1^l, \quad L_{tune} \leq l \leq L$$
- 边界层 $L_{tune}$ 通过 CKA 偏差阈值 $\rho_{layer}$ 自动确定

**分析式分类器**（替代可训练头部）：
- 随机投影 $W_{proj}$ 增强特征判别性
- Woodbury 恒等式递归更新自相关矩阵：
  $$R_t = R_{t-1} - R_{t-1}\hat{Z}_t^\top(I + \hat{Z}_t R_{t-1} \hat{Z}_t^\top)^{-1}\hat{Z}_t R_{t-1}$$
- 闭式更新分类器权重——无样本存储、非破坏性更新

#### 2. 自适应多会话子空间正交 PEFT

**多会话适应（MSA）**：每个会话引入独立 LoRA，前会话参数冻结：
$$W_t = W_0 + \sum_{\tau=0}^{t-1} B_\tau A_\tau + B_t A_t$$

**梯度投影约束**——确保更新不破坏旧任务表示：
$$g_{update} = P_{\mathcal{U}_t} \nabla_\theta \mathcal{L}_{ce}(g_t(f_t(\mathcal{X}_t)), \mathcal{Y}_t)$$

**高效零空间计算**（基于 LoRA 减法）：
- 构建"遗忘模型"：$W_t^{unlearn} = W_0 - \sum_{\tau=0}^{t-1} A_\tau B_\tau$
- 计算当前会话特征的非中心协方差矩阵 $X_t^{ucov}$
- SVD 分解确定投影子空间，保留能量比 $> \rho_{svd}$ 的主成分
- 无需存储历史特征，大幅降低存储开销

**自适应冻结**：当 $\sum_{i=0}^{T_3} N_t > N_{stop}$ 时转入 Stage 3 冻结骨干。

#### 3. 边界感知正则化

解决新旧类别表示纠缠导致的决策边界混淆：

**边界样本检测**：
- 对每个输入生成 $N_p$ 个时频掩码扰动 $\tilde{x}_{i,t}^k = \mathcal{Q}(x_{i,t}, r_T, r_F)$
- 临时模型 $\theta_{temp}$ 对扰动的误分类率超阈值 $\rho_p$ → 归入边界集 $\mathcal{B}_t$

**正则化损失**：
$$\mathcal{L}_{reg}(i) = \max(0, \delta + \frac{1}{|\mathcal{S}_i|}\sum_{u \in \mathcal{S}_i}\|f_t(u) - \mu(x_c)\|_2^2 - \min_{b \in \mathcal{B}_t}\|f_t(x_{i,t}) - b\|_2^2)$$

效果：拉向类中心 + 推离边界点，增大类间间距。

### 损失函数 / 训练策略

- FSA 阶段：交叉熵 $\mathcal{L}_{ce}$ + 边界正则化 $\mathcal{L}_{reg}$
- MSA 阶段：交叉熵 + 正则化 + 子空间正交梯度投影
- Stage 3：仅更新分析式分类器（闭式解，无梯度训练）
- 预训练骨干：EAT（12 层 ViT，AudioSet-2M 自监督预训练，~5000 小时）
- 数据增强：SpecAugment 风格的时频遮掩

## 实验关键数据

### 主实验

**表1：6 个音频 CL 基准的平均 Top-1 准确率（%）**

| 方法 | ESC-50 | US8K | SC2 | TIMIT-2 | TIMIT-3 | VocalSet |
|------|--------|------|-----|---------|---------|----------|
| Joint Training (上界) | 96.50 | 98.07 | 95.91 | 95.22 | 95.22 | 76.65 |
| L2P | 39.50 | 38.75 | 14.70 | 1.50 | 2.53 | 20.39 |
| RanPAC (w/ FSA) | 92.25 | 97.08 | 90.53 | 85.63 | 89.92 | 62.82 |
| HiDe-Prompt | 83.75 | 79.89 | 40.10 | 47.78 | 49.60 | 48.36 |
| **PACE** | **95.75** | **97.49** | **91.87** | **90.95** | **94.05** | **69.08** |

**与联合训练上界的差距**：ESC-50 仅 0.75%，US8K 仅 0.58%，TIMIT-3 仅 1.17%。

**表2：消融——改进 FSA 组件（粗粒度）**

| 策略 | ESC-50 | US8K | SC2 |
|------|--------|------|-----|
| w/o FSA | 92.50 | 96.49 | 81.22 |
| Naive FSA | 92.25 | 97.08 | 90.53 |
| + Low LR | 93.75 | 97.35 | 90.95 |
| + Later Layer LoRA | **95.75** | **97.49** | **91.87** |

### 消融实验

PACE 在 SSLAM 骨干上同样保持优势，验证骨干无关性。

细粒度基准上 MSA 的贡献：
- FSA only → +MSA: +3.2% (TIMIT-2)
- +子空间正交: +1.5%
- +边界感知正则化: +0.6%

### 关键发现

1. **音频 vs 视觉 CL 的本质差异**：音频骨干强调低层频谱导致表示漂移 3× 于视觉
2. **FSA 反直觉发现**：音频 CL 需鼓励骨干适应（与视觉 CL 相反），冻结浅层+调深层是关键
3. **分析式分类器稳定性**：避免累积偏差和表示漂移传播
4. **LoRA 减法创新用法**：不需存储历史特征即可近似旧任务表示子空间

## 亮点与洞察

- **首个系统性音频 CL 基准**：6 个基准覆盖粗/细粒度、语音/音乐/环境声
- **"需要适应而非冻结"**：与视觉域"冻结骨干足矣"形成鲜明对比，揭示音频预训练模型独特性质
- **三阶段渐进式框架**：FSA→MSA→冻结自然平衡可塑性与稳定性
- **LoRA 减法构建"遗忘模型"**：利用参数算术近似历史表示子空间，优雅高效

## 局限与展望

1. LoRA 减法假设近似性：减去 LoRA ≠ 精确遗忘，高 rank/强适应场景可能偏差较大
2. 边界检测依赖临时模型 $\theta_{temp}$ 质量
3. 自适应冻结阈值 $N_{stop}$ 需手动设定，不同场景最优值可能不同
4. 仅验证类增量设置，任务感知/域增量等其他 CL 设置未涉及
5. VocalSet 差距仍达 7.57%，细粒度音乐任务的不匹配最严重

## 相关工作与启发

- **RanPAC** 的分析式分类器是技术路线基石
- **LoRA Subtraction** 的参数减法被创新性地用于构建零空间投影
- **EAT** 的谱图重建预训练目标与下游分类的不匹配是核心问题来源
- 启发：预训练目标与下游任务的对齐程度决定 CL 难度

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个音频 CL 基准 + 三阶段框架
- 技术深度: ⭐⭐⭐⭐⭐ — 子空间正交 PEFT + 边界感知正则化理论完整
- 实验充分度: ⭐⭐⭐⭐⭐ — 6 个基准、多骨干验证、全面消融
- 实用价值: ⭐⭐⭐⭐ — 音频 CL 的切实需求，但部署场景待明确

<!-- RELATED:START -->

## 相关论文

- [Learning to Highlight Audio by Watching Movies](../../CVPR2025/audio_speech/learning_to_highlight_audio_by_watching_movies.md)
- [EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)
- [AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](../../ECCV2024/audio_speech/coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning](../../CVPR2026/audio_speech/cleaning_the_pool_progressive_filtering_of_unlabeled_pools_in_deep_active_learni.md)

<!-- RELATED:END -->
